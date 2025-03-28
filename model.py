import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model


class ProfileDetector:
    def _init_(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def train(self, data, target):
        # Preprocessing: standardize features
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(scaled_data, target, test_size=0.2, random_state=42)
        
        # Training
        self.model.fit(X_train, y_train)

        # Save the model and scaler
        joblib.dump(self.model, 'model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
    def predict(self, features):
        scaled_features = self.scaler.transform([features])
        return self.model.predict(scaled_features)[0]

    def load_model(self):
        self.model = joblib.load('model.pkl')
        self.scaler = joblib.load('scaler.pkl')
