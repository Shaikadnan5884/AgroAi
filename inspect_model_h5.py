import h5py
import json
import sys

path = 'models/plant_disease_model.h5'
try:
    f = h5py.File(path, 'r')
except Exception as e:
    print('Failed to open HDF5 file:', e)
    sys.exit(1)

print('Top-level keys:', list(f.keys()))
print('\nAttributes on file:', dict(f.attrs))

# Check for model_config in attrs
if 'model_config' in f.attrs:
    raw = f.attrs['model_config']
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    print('\nFound model_config in attrs (first 2000 chars):')
    print(raw[:2000])
else:
    print('\nNo model_config in file attrs')

# Check for 'model_config' dataset
if 'model_config' in f:
    try:
        raw = f['model_config'][()]
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        print('\nFound model_config dataset (first 2000 chars):')
        print(raw[:2000])
    except Exception as e:
        print('\nCould not read model_config dataset:', e)

# Try to inspect layer names under 'model_weights' or other groups
for grp in ['model_weights', 'weights', 'layer_names']:
    if grp in f:
        print(f"\nGroup '{grp}' exists. Listing children:")
        try:
            def recurse(name, obj):
                print(name)
            f[grp].visititems(recurse)
        except Exception as e:
            print('Could not list children:', e)

f.close()
print('\nDone')
