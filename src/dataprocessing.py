from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def define_category_column(data):
    print("Extracting features and target column.", data.head())
    X = data.drop(columns=['target'], axis=1)
    print("Features extracted successfully.")
    y = data['target']
    return X, y


# age,sex,bmi,bp,s1,s2,s3,s4,s5,s6,target
def preprocess_data(data):
    # Fill missing values with the mean of the column
    data.fillna(data.mean(), inplace=True)
    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    # Apply standard scaling to features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    scaled_features_df = pd.DataFrame(
        scaled_features, columns=data.columns)
    return scaled_features_df


def load():
    row_data = load_data('./data/diabetes_dataset.csv')
    print("Data loaded successfully.")
    processed_data = preprocess_data(row_data)
    print("Data preprocessed successfully.")
    X, y = define_category_column(processed_data)
    print("Category column defined successfully.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test
