import joblib
import pandas as pd
import json

def read_attributes():
    try:
        model = joblib.load('model/best_model(1).pkl')
        print(f"Model type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            print("Features:", features)
        else:
            print("Model doesn't have feature_names_in_")
            
        le = joblib.load('model/label_encoders(1).pkl')
        encoders = {}
        for k, v in le.items():
            encoders[k] = list(v.classes_)
            
        with open('features_dump.json', 'w') as f:
            json.dump({'features': features if 'features' in locals() else [], 'encoders': encoders}, f, indent=4)
        print("Successfully read encoders and features.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    read_attributes()
