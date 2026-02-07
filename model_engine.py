import os
# Force legacy Keras handling if older models rely on it (safe to leave set)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import models as keras_models
from PIL import Image, ImageOps, ImageFile
import numpy as np

# Handle images that might be slightly corrupted or truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Standard 38-Class PlantVillage Labels (module-level so UI can reuse them)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

def load_plant_model():
    model_path = 'models/plant_disease_model.h5'
    # Prefer to load the full model directly from the .h5 file if possible.
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Full model loaded from .h5 successfully!")
            return model
        except Exception as e:
            # If loading the full model fails, attempt JSON-fix fallback
            print(f"⚠️ Full model load failed, attempting JSON-fix fallback: {e}")
            try:
                import h5py

                with h5py.File(model_path, 'r') as f:
                    raw = None
                    if 'model_config' in f.attrs:
                        raw = f.attrs['model_config']
                    elif 'model_config' in f:
                        raw = f['model_config'][()]

                if raw is not None:
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8')
                    fixed_json = raw.replace('"batch_shape"', '"batch_input_shape"')
                    fixed_json = fixed_json.replace('"module": "keras"', '"module": "tensorflow.keras"')
                    try:
                        model = tf.keras.models.model_from_json(fixed_json)
                        model.load_weights(model_path)
                        print("✅ Loaded model from fixed JSON + weights")
                        return model
                    except Exception as e2:
                        print(f"⚠️ JSON-fallback load failed: {e2}")
                else:
                    print("⚠️ No model_config found in .h5 to attempt JSON fallback.")
            except Exception as e2:
                print(f"⚠️ JSON fallback attempt raised: {e2}")

    # Rebuild the expected architecture and try to inject weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(38, activation='softmax')
    ])
    # Helper: try to map h5 stored weights to rebuilt model by matching layer names.
    def inject_weights_from_h5_by_name(kmodel, h5_path):
        import h5py

        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)

        f = h5py.File(h5_path, 'r')
        # Prefer the 'model_weights' group if present
        weights_root = f['model_weights'] if 'model_weights' in f else f

        assigned = 0
        for grp_name in weights_root:
            try:
                # group may be a container (e.g., mobilenetv2_1.00_224) or a direct layer
                grp = weights_root[grp_name]
                # If this group contains subgroups for layers, iterate them
                if any(isinstance(grp[k], h5py.Group) for k in grp.keys()):
                    for sub_name in grp:
                        # try several variants of the layer name to maximize matches
                        candidates = [sub_name, f"{grp_name}/{sub_name}", f"{grp_name}_{sub_name}"]
                        layer_group = grp[sub_name]
                        weight_values = [np.array(layer_group[w]) for w in layer_group.keys()]
                        matched = False
                        for layer_name in candidates:
                            try:
                                layer = kmodel.get_layer(name=layer_name)
                                try:
                                    layer.set_weights(weight_values)
                                    assigned += 1
                                    matched = True
                                    break
                                except Exception:
                                    # shape mismatch for this candidate; try next
                                    continue
                            except ValueError:
                                continue
                        if not matched:
                            # last attempt: try to find any layer whose name endswith the sub_name
                            for lyr in kmodel.layers:
                                if lyr.name.endswith(sub_name):
                                    try:
                                        lyr.set_weights(weight_values)
                                        assigned += 1
                                        break
                                    except Exception:
                                        continue
                else:
                    # try to match top-level group name to a layer
                    layer_name = grp_name
                    try:
                        layer = kmodel.get_layer(name=layer_name)
                        layer_group = grp
                        weight_values = [np.array(layer_group[w]) for w in layer_group.keys()]
                        try:
                            layer.set_weights(weight_values)
                            assigned += 1
                        except Exception:
                            pass
                    except ValueError:
                        # not a direct layer name - skip
                        pass
            except Exception:
                # be resilient for unexpected group structures
                continue

        f.close()
        return assigned

    # Attempt to inject weights using the custom HDF5 reader. This helps when
    # model.load_model/model.load_weights fail due to cross-version metadata issues.
    try:
        assigned = inject_weights_from_h5_by_name(model, model_path)
        if assigned > 0:
            print(f"✅ Injected weights into {assigned} layers from HDF5 (by-name).")
            return model
        else:
            print("⚠️ No matching layers found to inject via by-name HDF5 mapping.")
    except Exception as e:
        print(f"⚠️ Custom HDF5 injection raised: {e}")

    # If injection only hit a few layers (e.g., just the classifier head), try a
    # pragmatic fallback: use an ImageNet-pretrained MobileNetV2 base and then
    # inject the classifier weights we were able to extract. This often works if
    # the original model was fine-tuned from ImageNet weights.
    try:
        print("🔁 Attempting pragmatic fallback with ImageNet-pretrained base...")
        base_model_im = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        model_im = Sequential([
            base_model_im,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(38, activation='softmax')
        ])
        assigned_im = inject_weights_from_h5_by_name(model_im, model_path)
        if assigned_im > 0:
            print(f"✅ Injected weights into {assigned_im} layers on ImageNet base model.")
            return model_im
        else:
            print("⚠️ Fallback injection with ImageNet base found no matching layers.")
    except Exception as e:
        print(f"⚠️ Pragmatic ImageNet fallback raised: {e}")

    # Final attempt: try standard weight loading API (may raise a clear error)
    try:
        model.load_weights(model_path)
        print("✅ Model weights injected successfully via load_weights()!")
        return model
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Fallback: full model loaded on second attempt")
            return model
        except Exception as e2:
            msg = (
                "Could not load the provided .h5 model file in this runtime.\n"
                "This commonly happens when the model was saved with a different Keras/TensorFlow\n"
                "version or a standalone 'keras' package.\n\n"
                "Recommended fixes:\n"
                "  - If you have the original training environment, re-save the model using the\n"
                "    current TensorFlow's tf.keras: `model.save('plant_disease_model.h5')` or\n"
                "    preferably as a SavedModel: `model.save('exported_model')`.\n"
                "  - If you can't re-save, try converting the file in an environment matching the\n"
                "    Keras version used to create it, then re-save with tf.keras.\n\n"
                f"Original deserialization error: {e}\nFinal attempt error: {e2}"
            )
            print("❌ Critical: could not load model or weights:", e2)
            raise RuntimeError(msg)

def predict_disease(image_data, model):
    # 1. Preprocess the image
    size = (224, 224)
    # Ensure 3-channel RGB input (some uploads may be RGBA or grayscale)
    if image_data.mode != 'RGB':
        image_data = image_data.convert('RGB')

    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalize pixel values to [-1, 1] (Standard for MobileNetV2)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1.0
    
    # 2. Prepare the batch (Data)
    # Reshape from (224, 224, 3) to (1, 224, 224, 3)
    data = np.expand_dims(normalized_img_array, axis=0)
    
    # 3. Perform Prediction
    # Use verbose=0 to keep output quiet under Streamlit
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])
    
    # 4. Map to the standard class list and return probabilities
    if index < len(CLASS_NAMES):
        return CLASS_NAMES[index], confidence, prediction[0]
    else:
        return "Unknown Class", confidence, prediction[0]