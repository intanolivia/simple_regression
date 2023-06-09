import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Judul apk
st.title('Aplikasi Analisis Regresi Linear Sederhana')

# Deskripsi apk
st.write('Aplikasi ini memungkinkan Anda melakukan analisis regresi linear sederhana dengan memasukkan angka-angka x dan y.')

# Navigasi sidebar
selected = st.sidebar.selectbox('Analisis Regresi Linear Sederhana', ['Pengujian t', 'Pengujian F'], index=0)

# Input data
st.write('Masukkan data x dan y:')

num_points = st.number_input('Jumlah angka', min_value=2, value=10)

x_values = []
y_values = []

for i in range(num_points):
    x = st.number_input(f'Masukkan nilai x{i+1}')
    y = st.number_input(f'Masukkan nilai y{i+1}')
    x_values.append(x)
    y_values.append(y)

# Membuat dataframe dari data x dan y
data = pd.DataFrame({'x': x_values, 'y': y_values})

# Menampilkan data
st.write('Data:')
st.write(data)

# Melakukan regresi linear
x = data['x'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

# Menampilkan hasil regresi linear
st.write('Hasil Regresi Linear:')
st.write('Koefisien Slope:', model.coef_[0][0])
st.write('Koefisien Intercept:', model.intercept_[0])

# Prediksi
prediction_input = st.number_input('Masukkan nilai x untuk diprediksi')
prediction_output = model.predict([[prediction_input]])
st.write('Hasil Prediksi (y):', prediction_output[0][0])
