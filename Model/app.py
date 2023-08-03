from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import joblib

app = Flask(__name__)

# Load the data
df1 = pd.read_csv("./dataset_trainning.csv")
# print(df1.shape)

# Data preprocessing
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df3 = df2.dropna()
# print(df3.head())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

df3[~df3['total_sqft'].apply(is_float)]

df4 = df3.copy()
df4 = df4[df4.total_sqft.notnull()]
# print(df4.head())

df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
# print(df5.head())

df5.to_csv("bhp.csv", index=False)

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)

location_stats_less_than_10 = location_stats[location_stats <= 10]

df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
# print(df6.head())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
# print(df7.head())

df8 = df7.copy()
df8[df8.bath > 10]

df8[df8.bath > df8.bhk + 2]
# print(df8.head())

df9 = df8[df8.bath < df8.bhk + 2]
# print(df9.head())

df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
# print(df10.head())

dummies = pd.get_dummies(df10.location, dtype=int)
# print(dummies.head(3))
if 'other' in dummies.columns:
    df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
else:
    df11 = pd.concat([df10, dummies], axis='columns')

# print(df11.head())

df12 = df11.drop('location', axis='columns')
# print(df12.head())

# Prepare Training Data
X = df12.drop(['price'], axis='columns')
y = df12.price
# print(X.head())
# print(y.head())

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(lr_clf, "model.pkl")


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        location = data['location']
        sqft = float(data['sqft'])
        bath = int(data['bath'])
        bhk = int(data['bhk'])

        # print(location, sqft, bath, bhk)

        return jsonify({"prediction": predict_price(location, sqft, bath, bhk)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
