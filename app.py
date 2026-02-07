import streamlit as st
from PIL import Image, ImageFile
from model_engine import load_plant_model, predict_disease

# Allow loading of slightly corrupted images often found in datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgroDetect AI",
    page_icon="🌿",
    layout="wide"
)

# --- SIDEBAR ---
st.sidebar.title("🌿 AgroDetect AI")
st.sidebar.info(
    "This AI-powered tool helps farmers identify plant diseases "
    "instantly using Computer Vision."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How to use:")
st.sidebar.write("1. Upload a clear photo of a plant leaf.")
st.sidebar.write("2. Click the 'Analyze' button.")
st.sidebar.write("3. Review the diagnosis and recommendations.")

# --- MAIN INTERFACE ---
st.title("Plant Disease Detection Engine")
st.markdown("### Scalable Agricultural Intelligence")

# Load the model with a spinner for better UX
@st.cache_resource # This prevents the model from reloading every time you click a button
def get_model():
    return load_plant_model()

with st.spinner("Initializing AI Engine..."):
    try:
        model = get_model()
    except Exception as e:
        # Show an actionable error in the Streamlit UI instead of crashing.
        import traceback
        tb = traceback.format_exc()
        st.error("Failed to load the ML model. See details below.")
        st.code(tb)
        model = None

# Image Upload Section
uploaded_file = st.file_uploader("Choose a leaf image (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create two columns: one for the image, one for the results
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.write("### Analysis")
        if model is None:
            st.error("Model is not available. Check the initialization error above.")
        else:
            if st.button("🔍 Run Diagnostic"):
                with st.spinner("Analyzing patterns..."):
                    label, confidence, probs = predict_disease(image, model)
                
                # Display Prediction
                clean_label = label.replace("___", " ").replace("_", " ")
                st.success(f"**Detected:** {clean_label}")
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                st.progress(float(confidence))

                # Debug: show top-3 predicted classes and probabilities
                try:
                    import numpy as _np
                    from model_engine import CLASS_NAMES
                    top_idxs = _np.argsort(probs)[-3:][::-1]
                    st.write("### Top-3 Predictions (debug)")
                    for i in top_idxs:
                        lbl = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
                        st.write(f"- {lbl}: {_np.round(probs[i]*100, 2)}%")
                except Exception:
                    # If debug display fails, skip silently
                    pass

                st.write("---")

                # --- RECOMMENDATION ENGINE ---
                recommendations = {
                    "Late_blight": {
                        "color": "error",
                        "title": "🔴 Urgent: Late Blight Detected",
                        "advice": "Remove infected leaves and apply Copper-based fungicides immediately."
                    },
                    "Early_blight": {
                        "color": "warning",
                        "title": "🟠 Warning: Early Blight Detected",
                        "advice": "Improve air circulation and avoid overhead watering to stop spread."
                    },
                    "scab": {
                        "color": "warning",
                        "title": "🟠 Warning: Scab Detected",
                        "advice": "Prune infected branches and use sulfur-based sprays."
                    },
                    "healthy": {
                        "color": "success",
                        "title": "🟢 Plant is Healthy!",
                        "advice": "Continue current care. Maintain regular nutrient and water cycles."
                    }
                }

                # Default fallback advice
                advice_found = False
                for key, content in recommendations.items():
                    if key.lower() in label.lower():
                        if content["color"] == "success":
                            st.balloons()
                            st.success(content["title"])
                        elif content["color"] == "warning":
                            st.warning(content["title"])
                        else:
                            st.error(content["title"])
                        
                        st.info(f"💡 **Recommendation:** {content['advice']}")
                        advice_found = True
                        break
                
                if not advice_found:
                    st.warning("⚠️ Action Required")
                    st.write("Specific treatment for this variant is not in the quick-database. Consult an agricultural specialist.")

# --- FOOTER ---
st.markdown("---")
st.caption("AgroDetect AI Hackathon Prototype | Optimized for MobileNetV2 Inference")
