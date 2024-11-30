import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import streamlit as st
import re

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['City'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())
    df['Area'] = df['Location'].apply(lambda x: x.split(',')[-2].strip())
    df.drop(columns=['Location', 'Name', 'Property Title'], inplace=True)
    
    def extract_bhk(description):
        match = re.search(r'(\d+) BHK', description)
        return match.group(1) if match else np.nan

    df['BHK'] = df['Description'].apply(extract_bhk)

    def fillBHK(row):
        if pd.isna(row['BHK']):
            matching_row = df[(df['Baths'] == row['Baths']) & 
                              (df['Balcony'] == row['Balcony']) & 
                              (df['City'] == row['City']) & 
                              (~df['BHK'].isna())]
            if not matching_row.empty:
                return matching_row['BHK'].iloc[0]
        return row['BHK']

    df['BHK'] = df.apply(fillBHK, axis=1)
    df.drop(columns='Description', inplace=True)

    def convert_price_to_numeric(price):
        if 'acs' in price.lower():
            return np.nan
        price = price.replace('₹', '').replace(',', '').strip()
        if 'Cr' in price:
            return float(price.replace('Cr', '').strip()) * 10**7
        elif 'L' in price:
            return float(price.replace('L', '').strip()) * 10**5
        if re.match(r"^\d+(\.\d+)?$", price):
            return float(price)
        return np.nan

    df['Price'] = df['Price'].apply(convert_price_to_numeric)
    df = df.dropna(subset=['Price'])
    df['Balcony'] = df['Balcony'].map({'Yes': 1, 'No': 0})
    return df

def predict_price(df, city, area, baths, balcony, bhk):
    X = df[['City', 'Area', 'Baths', 'Balcony', 'BHK']]
    Y = df['Price']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    new_data = pd.DataFrame({'City': [city], 'Area': [area], 'Baths': [baths], 'Balcony': [balcony], 'BHK': [bhk]})
    if new_data.isnull().any().any():
        raise ValueError("Input data contains missing values. Please provide complete data.")
    for col in ['City', 'Area', 'BHK']:
        if city not in df['City'].unique() or area not in df['Area'].unique() or bhk not in df['BHK'].unique():
            raise ValueError(f"Invalid input: Ensure City, Area, and BHK values match the dataset categories.")
    new_encoded = encoder.transform(new_data)
    predicted_price = model.predict(new_encoded)
    return predicted_price[0]

def main():
    st.title("House Price Prediction App")
    file_path = st.file_uploader("Upload Housing Data CSV", type="csv")
    if file_path is not None:
        df = preprocess_data(file_path)
        st.success("Data Preprocessed Successfully")
        city = st.selectbox("City", df['City'].unique())
        filtered_areas = df[df['City'] == city]['Area'].unique()
        area = st.selectbox("Area", filtered_areas)
        baths = st.number_input("Number of Baths", min_value=1, step=1)
        balcony = st.selectbox("Balcony", [1, 0])
        bhk = st.selectbox("BHK", df['BHK'].unique())
        if st.button("Predict"):
            try:
                predicted_price = predict_price(df, city, area, baths, balcony, bhk)
                st.success(f"Estimated Price: ₹{predicted_price:,.2f}")
            except ValueError as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
