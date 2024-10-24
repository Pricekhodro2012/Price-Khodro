import streamlit as st
import pandas as pd
import joblib

# عنوان صفحه
st.markdown("""
    <h1 style='text-align: right;'>مدل پیش‌بینی قیمت خودرو</h1>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    .css-1dq8tca {
        text-align: right;
        direction: rtl;
    }
    .stSelectbox, .stNumberInput {
        text-align: right;
        direction: rtl;
    }
    </style>
    """, unsafe_allow_html=True)
train_data=pd.read_csv('train.csv')
# ورودی‌های کاربر برای ستون‌ها
title = st.selectbox("نام خودرو", list(train_data['title'].unique()))

year = st.number_input("سال ساخت", min_value=1350, max_value=1403, value=1390)
mileage = st.number_input("کارکرد (به کیلومتر)", min_value=0.0, value=110000.0, step=1000.0)
transmission = st.selectbox("نوع گیربکس", list(train_data['transmission'].unique()))
fuel = st.selectbox("نوع سوخت", list(train_data['fuel'].unique()))

body_color = st.selectbox("رنگ بدنه", list(train_data['body_color'].unique()))
inside_color = st.selectbox("رنگ داخل", list(train_data['inside_color'].unique()))
body_status = st.selectbox("وضعیت بدنه", list(train_data['body_status'].unique()))
body_type = st.selectbox("نوع بدنه", list(train_data['body_type'].unique()))
volume = st.number_input("حجم موتور (به سی‌سی)", min_value=0.0, value=1.3)
engine = st.selectbox("نوع موتور", list(train_data['engine'].unique()))
acceleration = st.number_input("شتاب (0 تا 100)", min_value=0.0, value=13.0, step=0.1)

# آماده‌سازی ورودی‌ها به صورت یک DataFrame
input_data = pd.DataFrame({
    "title": [title],
    "year": [year],
    "mileage": [mileage],
    "transmission": [transmission],
    "fuel": [fuel],
    "body_color": [body_color],
    "inside_color": [inside_color],
    "body_status": [body_status],
    "body_type": [body_type],
    "volume": [volume],
    "engine": [engine],
    "acceleration": [acceleration]
})

# بارگذاری مدل و پردازشگر از پیش ذخیره شده
model = joblib.load("car_price_model.joblib")
# preprocessor = joblib.load("car_price_model_preprocessor.joblib")  # Ensure the preprocessor is also loaded

# پردازش داده‌های ورودی
# processed_data = preprocessor.transform(input_data)

# دکمه پیش‌بینی قیمت
st.markdown("""
    <style>
    .css-1dq8tca {
        text-align: right;
        direction: rtl;
    }
    .stButton > button {
        float: right;
    }
    </style>
    """, unsafe_allow_html=True)

# سایر ورودی‌های شما

if st.button("پیش‌بینی قیمت"):
    predicted_price = model.predict(input_data)
    st.markdown(f"""
        <div style='text-align: right;'>
            <span style='font-weight: bold; font-size: 18px;'>قیمت پیش‌بینی شده: </span>
            <span style='color: purple; font-size: 24px; font-weight: bold;'>{predicted_price[0]:,.0f}  </span>
            <span style='font-weight: bold; font-size: 18px;'>تومان </span>  
        </div>
    """, unsafe_allow_html=True)



