from model_engine import load_plant_model

if __name__ == '__main__':
    print('Loading model...')
    try:
        model = load_plant_model()
        print('Model object:', type(model))
        try:
            model.summary()
        except Exception as e:
            print('Could not print summary:', e)
    except Exception as e:
        print('Model loading raised an exception:')
        import traceback
        traceback.print_exc()
    print('Done')
