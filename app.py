import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Title
# Create a Streamlit app
st.set_page_config(page_title='Phone Price Predictor')
st.title("Phone Price Predictor")
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.10wallpaper.com/wallpaper/1920x1200/1908/2019_Purple_Abstract_4K_HD_Design_1920x1200.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


try:
    
    # Load Decision Tree model
    dtcModel = pickle.load(open('DTC.pkl', 'rb'))
    
    # Load dataset
    data = pd.read_csv('dataset.csv')
    
    X = data.drop(['price_range'], axis = 1)
    y = data['price_range']
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1)



    
    
except Exception as e:
    st.error(f"Terjadi kesalahan dalam memuat dataset atau model: {e}")
    
    

# Create a form for the user to input features
# form = st.form(key='input_form')
# ratings = form.number_input("Ratings", min_value=0.0, max_value=5.0, step=0.1)
# ram = form.number_input("RAM")
# rom = form.number_input("ROM")
# mobile_size = form.number_input("Mobile Size")
# primary_cam = form.number_input("Primary Camera")
# selfi_cam = form.number_input("Selfie Camera")
# battery_power = form.number_input("Battery Power")
# submit_button = form.form_submit_button(label='Submit')

# # Make a prediction based on the user's input
# if submit_button:
#     input_data = [[ratings, ram, rom, mobile_size, primary_cam, selfi_cam, battery_power]]
#     prediction = dtcModel.predict(input_data)[0]
#     st.header(f"The predicted price is {prediction:.2f}")

# Streamlit App
st.sidebar.header('Input Fitur')

def user_input_features():
    features = {}
    for column in data.columns:
        features[column] = st.sidebar.slider(f'{column}', int(data[column].min()), int(data[column].max()), int(data[column].mean()))
    return pd.DataFrame(features, index=[0])

user_input = user_input_features()

st.subheader('Data Fitur:')
st.write(user_input)

# # Pilih hanya fitur yang digunakan saat melatih model
selected_features = user_input[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range']]

# # Prediksi harga dengan model Decision Tree
# decision_tree_prediction = dtcModel.predict(selected_features.values)

    # Melakukan prediksi
y_pred = dtcModel.predict(X_test)

    # Menampilkan accuracy score  dan classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
st.write(f"Accuracy: {accuracy}")
st.text("Classification Report:")
st.text(classification_rep)


# Tampilkan hasil prediksi Decision Tree
st.subheader('Hasil Prediksi Harga Telepon (Decision Tree):')

# Tampilkan kelas yang diprediksi dalam format mata uang Rupiah
st.write(f'Harga yang Diprediksi: {float(y_pred[0])}')

# Informasi tentang dataset
if st.checkbox("Detail"):
    st.subheader('Detail Dataset Fitur:')
    st.write(data)
    
    fig = px.pie(data, names='price_range', title='Distribusi Kisaran Harga Telepon')
    st.plotly_chart(fig)

    st.bar_chart(data['price_range'].value_counts())

    # Line chart untuk menunjukkan tren harga terhadap ram
    st.subheader('Tren Harga vs RAM:')
    fig_line = px.line(data, x='ram', y='price_range', title='Tren Harga vs RAM')
    st.plotly_chart(fig_line)

    # Area chart untuk menunjukkan persebaran harga berdasarkan px_height dan px_width
    st.subheader('Persebaran Harga berdasarkan Px_Height dan Px_Width:')
    fig_area = px.area(data, x='px_height', y='px_width', color='price_range', title='Persebaran Harga berdasarkan Px_Height dan Px_Width')
    st.plotly_chart(fig_area)


    