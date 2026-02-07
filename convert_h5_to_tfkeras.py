"""
Attempt to convert a model saved with standalone Keras to a tf.keras HDF5/SavedModel.

Usage (PowerShell):
  # (optional) create a virtualenv matching the training environment
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1

  # Install standalone keras if needed (choose a version close to training time)
  pip install "keras>=2.3.0,<3.0.0" h5py

  # Run the converter
  python convert_h5_to_tfkeras.py

This script will:
  - Try to import standalone `keras` and load `models/plant_disease_model.h5`.
  - Re-save the model using `tf.keras` as `models/plant_disease_model_tf.h5` and
    also export a SavedModel directory `models/plant_disease_model_saved/`.

If the loading fails, the script prints diagnostics.
"""

import os
import traceback

MODEL_PATH = 'models/plant_disease_model.h5'
OUT_H5 = 'models/plant_disease_model_tf.h5'
OUT_SAVED = 'models/plant_disease_model_saved'

print('Checking files...')
if not os.path.exists(MODEL_PATH):
    print('Error: model file not found at', MODEL_PATH)
    raise SystemExit(1)

# Step 1: try to load with standalone keras
loaded = False
try:
    print('\nTrying to import standalone keras ("import keras")...')
    import keras
    print('standalone keras version:', getattr(keras, '__version__', 'unknown'))
    print('Loading model with standalone keras...')
    model = keras.models.load_model(MODEL_PATH, compile=False)
    loaded = True
    print('Loaded model using standalone keras')
except Exception as e:
    print('Could not load with standalone keras:', e)
    traceback.print_exc()

# Fallback: try tf.keras load (might already be what we tried earlier)
if not loaded:
    try:
        print('\nTrying to load with tf.keras...')
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        loaded = True
        print('Loaded model using tf.keras')
    except Exception as e:
        print('Could not load with tf.keras either:', e)
        traceback.print_exc()

if not loaded:
    print('\nAll load attempts failed. See errors above. Consider re-saving the model in the training environment using tf.keras.')
    raise SystemExit(2)

# Step 2: Re-save with tf.keras
try:
    import tensorflow as tf
    print('\nRe-saving model with tf.keras to HDF5 and SavedModel formats...')
    # Save HDF5
    model.save(OUT_H5)
    print('Saved HDF5 to', OUT_H5)
    # Save SavedModel dir
    tf.keras.models.save_model(model, OUT_SAVED, include_optimizer=False)
    print('Saved SavedModel to', OUT_SAVED)
    print('\nConversion completed successfully. Replace the old model with the new HDF5 or use the SavedModel dir in your loader.')
except Exception as e:
    print('Failed to re-save model with tf.keras:', e)
    traceback.print_exc()
    raise
