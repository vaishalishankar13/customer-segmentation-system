import streamlit as st
import pandas as pd
import joblib

# Load models
kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title('Customer Segmentation System')

st.write('Enter customer details')

age = st.number_input('Age', 18, 100)
annual_spend = st.number_input('Annual Spend')
visits_per_month = st.number_input('Visits Per Month')
basket_size = st.number_input('Basket Size')
days_since_last_visit = st.number_input('Days Since Last Visit')
num_categories_purchased = st.number_input('Categories Purchased')

if st.button('Predict Cluster'):

    input_data = pd.DataFrame([[
        age,
        annual_spend,
        visits_per_month,
        basket_size,
        days_since_last_visit,
        num_categories_purchased
    ]], columns=[
        'age',
        'annual_spend',
        'visits_per_month',
        'basket_size',
        'days_since_last_visit',
        'num_categories_purchased'
    ])

    scaled_input = scaler.transform(input_data)

    prediction = kmeans.predict(scaled_input)[0]

    st.success(f'Customer belongs to Cluster {prediction}')

    # Business recommendations
    if prediction == 0:
        st.write('Low-value frequent shoppers')
        st.write('Recommendation: Offer bundles and discounts')

    elif prediction == 1:
        st.write('High-value inactive customers')
        st.write('Recommendation: Retention and loyalty campaigns')

    else:
        st.write('Mid-value regular customers')
        st.write('Recommendation: Personalized promotions')