# app.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the dataset
        data = pd.read_csv('./dataset_trainning.csv')

        # Preprocess the data
        data = preprocess_data(data)

        # Split the data into features and target
        X = data.drop(columns=['price', 'availability', 'society'])
        y = data['price']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get the input data from the API request
        input_data = request.get_json()

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        input_df = preprocess_data(input_df, data)

        # Make predictions using the trained model
        prediction = model.predict(input_df)

        return jsonify({'predicted_price': prediction[0]})

    except Exception as e:
        print('Error:', str(e))  # Print the error for debugging purposes
        return jsonify({'error': 'Something went wrong on the server.'}), 500

def preprocess_data(df, full_data=None):
    # Convert area_type and availability to numerical values using LabelEncoder
    le = LabelEncoder()
    df['area_type'] = le.fit_transform(df['area_type'])
    # df['availability'] = le.fit_transform(df['availability'])

    # Convert size to the number of bedrooms
    df['size'] = df['size'].apply(lambda x: int(x.split()[0]))

    if full_data is not None:
        # One-hot encode the 'location' and 'society' columns
        df = pd.get_dummies(df, columns=['location'])

        # Ensure all columns are present in the input DataFrame
        missing_cols = set(full_data.columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        # Reorder columns to match the original dataset
        df = df[full_data.columns]

    else:
        # One-hot encode the 'location' column for input data
        df = pd.get_dummies(df, columns=['location'])

        # Ensure all columns are present in the input DataFrame
        if full_data is not None:
            missing_cols = set(full_data.columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0

    return df

if __name__ == '__main__':
    app.run(debug=True)
