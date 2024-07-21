import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt

# Fungsi untuk melakukan Fuzzy C-Means Clustering
def fuzzy_cmeans_clustering(data, n_clusters):
    data = np.array(data, dtype=np.float64)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)
    return cluster_membership, cntr, u

# Fungsi untuk membuat koneksi ke database
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='sql12.freesqldatabase.com',  # Ganti dengan alamat IP server database
            database='sql12721187',
            user='sql12721187',
            password='wMAju6LHZZ'
        )
        if connection.is_connected():
            st.write("Koneksi ke database berhasil.")
        return connection
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")
        return None

# Fungsi untuk menyimpan hasil ke database
def save_to_database(data, centroids):
    try:
        connection = create_connection()
        if connection and connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hasil_klaster (
                    BMI FLOAT, Sistolik INT, Diastolik INT, Hba1C FLOAT,
                    Gula_Darah_2_Jam_PP INT, Gula_Darah_Acak INT, Gula_Darah_Puasa INT,
                    Usia INT, Klaster INT, Centroid FLOAT
                )
            """)
            for index, row in data.iterrows():
                # Mendapatkan nilai centroid yang sesuai dengan klaster
                centroid = centroids[int(row['Klaster']), :]  # Mengambil centroid untuk klaster ini
                centroid_value = np.mean(centroid)  # Mengambil rata-rata dari centroid
                cursor.execute("""
                    INSERT INTO hasil_klaster (BMI, Sistolik, Diastolik, Hba1C,
                    Gula_Darah_2_Jam_PP, Gula_Darah_Acak, Gula_Darah_Puasa, Usia, Klaster, Centroid)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(row) + (centroid_value,))
            connection.commit()
            st.success("Data berhasil disimpan ke database.")
    except Error as e:
        st.error(f"Error: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Fungsi untuk mengambil data dari database
def fetch_data():
    try:
        connection = create_connection()
        if connection and connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT BMI, Sistolik, Diastolik, Hba1C, Gula_Darah_2_Jam_PP, Gula_Darah_Acak, Gula_Darah_Puasa, Usia, Klaster FROM hasil_klaster")
            data = cursor.fetchall()
            
            if data:
                # Konversi data menjadi DataFrame
                df = pd.DataFrame(data, columns=[
                    'BMI', 'Sistolik', 'Diastolik', 'Hba1C', 
                    'Gula_Darah_2_Jam_PP', 'Gula_Darah_Acak', 
                    'Gula_Darah_Puasa', 'Usia', 'Klaster'
                ])
                
                # Konversi tipe data ke tipe dasar
                df = df.astype({
                    'BMI': 'float',
                    'Sistolik': 'int',
                    'Diastolik': 'int',
                    'Hba1C': 'float',
                    'Gula_Darah_2_Jam_PP': 'int',
                    'Gula_Darah_Acak': 'int',
                    'Gula_Darah_Puasa': 'int',
                    'Usia': 'int',
                    'Klaster': 'int'
                })

                # Pastikan tidak ada nilai yang tak valid
                df = df.dropna()

                return df
            else:
                st.warning('Tidak ada data yang tersedia.')
                return None
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Inisialisasi halaman Streamlit
st.title('Aplikasi Klasterisasi Data Pasien dengan Fuzzy C-Means Clustering')

menu = st.sidebar.selectbox('Menu', options=['Input Data', 'Lihat Hasil Klaster'])

# Halaman Input Data
if menu == 'Input Data':
    st.header('Input Data Pasien')
    
    # Form untuk input data pasien
    with st.form(key='input_form'):
        num_patients = st.number_input('Jumlah Pasien', min_value=1, value=1)
        patient_data = []
        for i in range(num_patients):
            st.write(f'Pasien {i+1}')
            bmi = st.number_input(f'BMI Pasien {i+1}', min_value=0.0, step=0.01, format="%.2f")
            sistolik = st.number_input(f'Sistolik Pasien {i+1}', min_value=0, step=1)
            diastolik = st.number_input(f'Diastolik Pasien {i+1}', min_value=0, step=1)
            hba1c = st.number_input(f'Hba1C Pasien {i+1}', min_value=0.0, step=0.01, format="%.2f")
            gula_darah_2_jam_pp = st.number_input(f'Gula Darah 2 Jam PP Pasien {i+1}', min_value=0, step=1)
            gula_darah_acak = st.number_input(f'Gula Darah Acak Pasien {i+1}', min_value=0, step=1)
            gula_darah_puasa = st.number_input(f'Gula Darah Puasa Pasien {i+1}', min_value=0, step=1)
            usia = st.number_input(f'Usia Pasien {i+1}', min_value=0, step=1)
            patient_data.append([bmi, sistolik, diastolik, hba1c, gula_darah_2_jam_pp, gula_darah_acak, gula_darah_puasa, usia])
        
        n_clusters = st.number_input('Jumlah Klaster', min_value=2, max_value=10, value=3)
        submit_button = st.form_submit_button(label='Lakukan Klasterisasi')
    
    if submit_button:
        data_values = np.array(patient_data, dtype=float)
        cluster_membership, cntr, u = fuzzy_cmeans_clustering(data_values, n_clusters)

        data = pd.DataFrame(patient_data, columns=[
            'BMI', 'Sistolik', 'Diastolik', 'Hba1C', 'Gula Darah 2 Jam PP', 
            'Gula Darah Acak', 'Gula Darah Puasa', 'Usia'
        ])
        data['Klaster'] = cluster_membership.astype(int)
        st.session_state['cntr'] = cntr
        st.session_state['u'] = u
        save_to_database(data, cntr)  # Mengirim data dan centroids ke fungsi save_to_database

# Halaman Lihat Hasil Klaster
elif menu == 'Lihat Hasil Klaster':
    st.header('Hasil Klasterisasi Data Pasien')
    data = fetch_data()
    if data is not None:
        # Menampilkan data klaster
        data = data.astype({
            'BMI': 'float',
            'Sistolik': 'int',
            'Diastolik': 'int',
            'Hba1C': 'float',
            'Gula_Darah_2_Jam_PP': 'int',
            'Gula_Darah_Acak': 'int',
            'Gula_Darah_Puasa': 'int',
            'Usia': 'int',
            'Klaster': 'int'
        })

        # Format kolom BMI dan Hba1C dengan dua desimal
        data['BMI'] = data['BMI'].map('{:.2f}'.format)
        data['Hba1C'] = data['Hba1C'].map('{:.2f}'.format)

        st.dataframe(data)

        # Menyiapkan plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data berdasarkan klaster
        for cluster in data['Klaster'].unique():
            cluster_data = data[data['Klaster'] == cluster]
            ax.scatter(cluster_data['BMI'].astype(float), cluster_data['Usia'], label=f'Klaster {cluster}')
        
        # Menambahkan centroid ke plot
        if 'cntr' in st.session_state:
            centroids = st.session_state['cntr']
            for i, centroid in enumerate(centroids):
                ax.scatter(centroid[0], centroid[3], c='black', marker='x', s=100, label=f'Centroid {i}')
        
        # Mengatur tick marks dan batas sumbu x
        ax.set_xticks(np.arange(15, 51, 5))  # Menetapkan tick marks untuk sumbu x
        ax.set_xlim(15, 50)  # Menetapkan batas sumbu x dari 0 hingga 50
        
        ax.set_xlabel('BMI')
        ax.set_ylabel('Usia')
        ax.set_title('Scatter Plot Klasterisasi Pasien')
        ax.legend()
        
        st.pyplot(fig)
    else:
        st.warning('Tidak ada data yang tersedia. Silakan lakukan klasterisasi terlebih dahulu.')
