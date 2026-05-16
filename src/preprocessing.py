from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    joblib.dump(scaler, 'models/scaler.pkl')
    return scaled_data