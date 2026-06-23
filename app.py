# Atur backend Matplotlib di paling atas SEBELUM mengimport pyplot
# Ini mencegah error GUI/Tcl pada beberapa sistem operasi atau server headless
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Mengatur konfigurasi halaman agar terlihat profesional
st.set_page_config(
    page_title="Klasifikasi Kualitas Buah - Kelompok 4",
    page_icon="🍎",
    layout="wide"
)

# ==========================================
# 1. SIMULASI DATASET DEFAULT (KUALITAS BUAH)
# ==========================================
@st.cache_data
def load_default_data():
    # Membuat dataset buah tiruan yang logis untuk klasifikasi kualitas
    np.random.seed(42)
    n_samples = 200
    
    # Buah Kualitas Baik (Sedikit bintik, ukuran & berat ideal, warna bervariasi)
    baik_diameter = np.random.normal(8.0, 1.2, int(n_samples/2))
    baik_berat = np.random.normal(180, 25, int(n_samples/2))
    # Apel baik: 60% Merah (matang), 20% Hijau (jenis Granny Smith), 20% Kuning (jenis Golden)
    baik_warna = np.random.choice(['Merah', 'Hijau', 'Kuning'], size=int(n_samples/2), p=[0.6, 0.2, 0.2])
    baik_persentase_bintik = np.random.normal(5, 5, int(n_samples/2))
    
    # Buah Kualitas Buruk (Banyak bintik, ukuran & berat tidak ideal)
    buruk_diameter = np.random.normal(5.5, 1.5, int(n_samples/2))
    buruk_berat = np.random.normal(120, 35, int(n_samples/2))
    # Apel buruk bisa warna apa saja, tapi memiliki banyak bintik/kerusakan
    buruk_warna = np.random.choice(['Hijau', 'Kuning', 'Merah'], size=int(n_samples/2), p=[0.4, 0.4, 0.2])
    buruk_persentase_bintik = np.random.normal(40, 15, int(n_samples/2))
    
    data_baik = pd.DataFrame({
        'Diameter (cm)': baik_diameter,
        'Berat (gram)': baik_berat,
        'Warna': baik_warna,
        'Persentase Bintik (%)': baik_persentase_bintik,
        'Kualitas': 'Baik'
    })
    
    data_buruk = pd.DataFrame({
        'Diameter (cm)': buruk_diameter,
        'Berat (gram)': buruk_berat,
        'Warna': buruk_warna,
        'Persentase Bintik (%)': buruk_persentase_bintik,
        'Kualitas': 'Buruk'
    })
    
    df = pd.concat([data_baik, data_buruk], ignore_index=True)
    # Batasi nilai agar tetap di rentang yang logis
    df['Persentase Bintik (%)'] = df['Persentase Bintik (%)'].clip(0, 100)
    return df

# Memuat data
df_original = load_default_data()

# PREPROCESSING: Mengubah data kategorikal (Warna) menjadi numerik dengan One-Hot Encoding
df_processed = pd.get_dummies(df_original, columns=['Warna'], drop_first=False)
# Pastikan semua kolom warna ada, bahkan jika tidak ada di data sampel
for color in ['Warna_Hijau', 'Warna_Merah', 'Warna_Kuning']:
    if color not in df_processed.columns:
        df_processed[color] = 0

# ==========================================
# SIDEBAR - NAVIGASI DAN INFORMASI KELOMPOK
# ==========================================
with st.sidebar:
    # Menggunakan try-except untuk load gambar agar tidak crash jika offline
    try:
        st.image("apel.jpg", caption="Klasifikasi Apel", use_container_width=True)
    except Exception:
        st.subheader("🍎 Klasifikasi Apel")
        
    st.title("🍎 Kelompok 4 (Apel)")
    st.subheader("Studi Kasus: Kualitas Buah")
    st.markdown("---")
    
    # Menu Navigasi
    menu = st.radio(
        "Pilih Menu Aplikasi:",
        ["Dashboard & Teori", "Eksplorasi Data", "Latih & Evaluasi Model", "Prediksi Interaktif (Uji Coba)"]
    )
    
    st.markdown("---")
    st.markdown("### 👥 Anggota Kelompok 4:")
    st.info("""
    1. **M. Ali Ikhsan** (NIM: 2455202054)
    2. **Riska Destia** (NIM: 2455202063)
    3. **Rya Aman Danu** (NIM: 2455202066)
    4. **Suci Setiani** (NIM: 2455202071)
    5. **Varenza Andre Saputra** (NIM: 2455202073)
    """)
    st.caption("Proyek Algoritma Naive Bayes - Kelas Kecerdasan Buatan / Data Science")

# ==========================================
# HALAMAN 1: DASHBOARD & TEORI NAIVE BAYES
# ==========================================
if menu == "Dashboard & Teori":
    st.title("🎯 Aplikasi Klasifikasi Kualitas Apel Menggunakan Naive Bayes")
    st.subheader("Selamat Datang di Projek Proyek Kelompok 4")
    
    st.markdown("""
    Aplikasi ini dirancang untuk mengklasifikasikan kualitas buah apel (Kategori: **Baik** atau **Buruk**) berdasarkan fitur fisiknya. 
    Metode yang kita gunakan adalah **Gaussian Naive Bayes**, yang sangat cocok untuk mengolah data fitur kontinu (numerik).
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📌 Mengapa Studi Kasus Kualitas Buah?\n"
                "Dalam industri pertanian, menyortir buah berkualitas tinggi dari yang rusak secara manual memerlukan banyak waktu. "
                "Dengan Machine Learning, proses penyortiran dapat diotomatisasi secara cepat berdasarkan fitur visual seperti **ukuran, berat, warna (matang/tidak), dan ada tidaknya kerusakan (bintik)** yang bisa dideteksi oleh sensor atau kamera.")
        
    with col2:
        st.warning("### 🧪 Apa itu Algoritma Naive Bayes?\n"
                   "Naive Bayes adalah algoritma klasifikasi probabilistik berdasarkan **Teorema Bayes**. Disebut 'Naive' karena algoritma ini mengasumsikan "
                   "bahwa semua fitur pendukung keputusan bersifat independen satu sama lain (tidak saling mempengaruhi).")
        
    st.markdown("### 🧮 Persamaan Teorema Bayes:")
    st.latex(r"P(C_k | X) = \frac{P(X | C_k) \cdot P(C_k)}{P(X)}")
    
    st.markdown("""
    **Keterangan:**
    - $P(C_k | X)$: Probabilitas posterior dari kelas $C_k$ (misal: Baik/Buruk) jika diberikan fitur input $X$.
    - $P(X | C_k)$: Likelihood / Probabilitas kemunculan fitur $X$ pada kelas $C_k$. Karena fitur kita bersifat kontinu, kita menggunakan rumus Distribusi Normal (Gaussian):
    """)
    st.latex(r"P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} e^{-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}}")
    st.markdown("""
    - $P(C_k)$: Probabilitas prior dari kelas $C_k$ (seberapa sering buah berstatus Baik atau Buruk muncul secara umum di dataset).
    - $P(X)$: Probabilitas marginal prediktor (pembagi konstan untuk semua kelas).
    """)

# ==========================================
# HALAMAN 2: EKSPLORASI DATASET
# ==========================================
elif menu == "Eksplorasi Data":
    st.title("📊 Eksplorasi Data Kualitas Apel")
    st.write("Di bawah ini adalah data sampel yang kita gunakan untuk melatih model. Perhatikan bagaimana fitur 'Warna' diubah menjadi kolom numerik `Warna_Hijau`, `Warna_Merah`, dan `Warna_Kuning`.")
    
    st.write("### 📂 Data Table Asli (Sebelum Encoding)")
    st.dataframe(df_original, use_container_width=True)
    
    st.write("### 📂 Data Table Siap Proses (Setelah One-Hot Encoding)")
    st.dataframe(df_processed, use_container_width=True)
    
    # Statistik Deskriptif
    st.write("### 📈 Ringkasan Statistik Berdasarkan Kualitas Buah")
    st.write(df_original.groupby('Kualitas').describe(include='all'))
    
    # Visualisasi Distribusi Fitur
    st.write("### 🖼️ Visualisasi Hubungan Antar Fitur")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df_original, x='Warna', hue='Kualitas', palette='Set1', ax=ax, order=['Merah', 'Hijau', 'Kuning'])
        plt.title("Distribusi Warna berdasarkan Kualitas")
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df_original, x='Persentase Bintik (%)', hue="Kualitas", fill=True, common_norm=False, palette="Set1", alpha=0.5, ax=ax)
        plt.title("Distribusi Persentase Bintik berdasarkan Kualitas")
        st.pyplot(fig)
        plt.close(fig)
    
    st.success("💡 **Observasi Singkat:** Apel berkualitas 'Baik' cenderung memiliki **Persentase Bintik** yang rendah, terlepas dari warnanya. Sebaliknya, apel 'Buruk' memiliki banyak bintik.")

# ==========================================
# HALAMAN 3: PELATIHAN DAN EVALUASI MODEL
# ==========================================
elif menu == "Latih & Evaluasi Model":
    st.title("⚙️ Proses Pelatihan & Evaluasi Naive Bayes")
    
    # Memisahkan Fitur dan Target dari data yang sudah diproses
    X = df_processed.drop(columns=['Kualitas'])
    y = df_processed['Kualitas']
    
    test_size = st.slider("Pilih Persentase Data Uji (Test Set %):", min_value=10, max_value=50, value=20, step=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)
    
    # Cek apakah model sudah ada di session state, jika tidak, latih dan simpan
    if 'trained_model' not in st.session_state or st.session_state.test_size != test_size:
        st.session_state.test_size = test_size
        # Melatih model Gaussian Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)
        # Simpan model yang telah dilatih ke dalam session state
        st.session_state.trained_model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
    
    # Ambil model dan data uji dari session state
    model = st.session_state.trained_model
    
    # Prediksi data uji
    y_pred = model.predict(st.session_state.X_test)
    akurasi = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 Akurasi Model Naive Bayes", value=f"{akurasi*100:.2f} %")
        st.write("Akurasi menunjukkan seberapa tepat model dalam mengklasifikasikan kualitas buah pada data uji baru yang belum pernah dilihat sebelumnya.")
        
    with col2:
        st.write("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=['Baik', 'Buruk'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Baik', 'Buruk'], yticklabels=['Baik', 'Buruk'], ax=ax)
        plt.xlabel('Prediksi Model')
        plt.ylabel('Kualitas Sebenarnya')
        st.pyplot(fig)
        plt.close(fig)
        
    st.write("### 📝 Laporan Klasifikasi Detil (Classification Report)")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df, use_container_width=True)

# ==========================================
# HALAMAN 4: PREDIKSI INTERAKTIF & CARA KERJA MATEMATIS
# ==========================================
elif menu == "Prediksi Interaktif (Uji Coba)":
    st.title("🔬 Prediksi Kualitas Buah Interaktif")
    st.write("Sesuaikan spesifikasi apel di bawah ini, dan lihat apakah algoritma mengklasifikasikannya sebagai **Baik** atau **Buruk**!")
    
    # Gunakan model yang dilatih pada seluruh dataset untuk prediksi optimal
    X_full = df_processed.drop(columns=['Kualitas'])
    y_full = df_processed['Kualitas']
    model_full = GaussianNB()
    model_full.fit(X_full, y_full)
    
    # 1. FORM INPUT USER
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📥 Spesifikasi Fisik Apel:")
        input_diameter = st.slider("Diameter Apel (cm)", min_value=3.0, max_value=15.0, value=8.0, step=0.1)
        input_berat = st.slider("Berat Apel (gram)", min_value=50.0, max_value=300.0, value=180.0, step=1.0)
        
    with col2:
        st.markdown("### 🎨 Spesifikasi Warna & Kondisi:")
        input_warna = st.radio("Pilih Warna Apel:", ('Merah', 'Hijau', 'Kuning'), horizontal=True)
        input_persentase_bintik = st.slider("Persentase Bintik (%)", min_value=0, max_value=100, value=5, step=1)
        
    # Menyiapkan array data baru sesuai dengan One-Hot Encoding
    warna_merah = 1 if input_warna == 'Merah' else 0
    warna_hijau = 1 if input_warna == 'Hijau' else 0
    warna_kuning = 1 if input_warna == 'Kuning' else 0
    
    # Urutan kolom harus SAMA PERSIS dengan saat melatih model
    feature_order = ['Diameter (cm)', 'Berat (gram)', 'Persentase Bintik (%)', 'Warna_Hijau', 'Warna_Kuning', 'Warna_Merah']
    data_input_array = np.array([[
        input_diameter, 
        input_berat, 
        input_persentase_bintik,
        warna_hijau,
        warna_kuning,
        warna_merah
    ]])
    
    # Membuat DataFrame dari input dengan urutan kolom yang benar
    input_df = pd.DataFrame(data_input_array, columns=feature_order)
    
    prediksi = model_full.predict(input_df)[0]
    probabilitas = model_full.predict_proba(input_df)[0]
    
    # Menampilkan Hasil Prediksi
    st.markdown("---")
    st.write("## 🏁 Hasil Klasifikasi Model:")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if prediksi == 'Baik':
            st.success(f"### Kualitas Apel: **BAIK** 🟢")
            st.write(f"Probabilitas Apel Berkualitas Baik: **{probabilitas[0]*100:.2f}%**")
        else:
            st.error(f"### Kualitas Apel: **BURUK** 🔴")
            st.write(f"Probabilitas Apel Berkualitas Buruk: **{probabilitas[1]*100:.2f}%**")
            
    with col_res2:
        # Tampilkan grafik probabilitas klasifikasi
        # Cari index kelas 'Baik' dan 'Buruk' dari model
        class_order = list(model_full.classes_)
        prob_baik_idx = class_order.index('Baik')
        prob_buruk_idx = class_order.index('Buruk')
        
        probs_display = [probabilitas[prob_baik_idx], probabilitas[prob_buruk_idx]]
        
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.barh(np.arange(2), np.array(probs_display) * 100, color=['#4CAF50', '#F44336'], height=0.5)
        ax.set_yticks(np.arange(2))
        ax.set_yticklabels(class_order)
        ax.set_xlabel('Keyakinan Probabilitas (%)')
        ax.set_xlim(0, 100)
        st.pyplot(fig)
        plt.close(fig)

    # ==========================================
    # PENJELASAN STEP-BY-STEP (SANGAT BAGUS UNTUK PRESENTASI)
    # ==========================================
    st.markdown("---")
    st.markdown("### 🧮 Cara Kerja Matematika Naive Bayes di Balik Layar:")
    st.write("Mari kita lihat bagaimana algoritma menghitung probabilitas di atas langkah demi langkah menggunakan data yang Anda masukkan!")
    
    # Gunakan dataset yang sudah diproses untuk kalkulasi
    df_calc = df_processed

    # Hitung Prior Probabilities P(C)
    total_samples = len(df_calc)
    total_baik = len(df_calc[df_calc['Kualitas'] == 'Baik'])
    total_buruk = len(df_calc[df_calc['Kualitas'] == 'Buruk'])
    
    prior_baik = total_baik / total_samples
    prior_buruk = total_buruk / total_samples
    
    st.markdown(f"#### **Langkah 1: Menghitung Probabilitas Prior (Awal)**")
    st.write(f"Dari total **{total_samples}** sampel data, kita punya **{total_baik}** apel 'Baik' dan **{total_buruk}** apel 'Buruk'.")
    st.latex(rf"P(\text{{Kualitas}} = \text{{Baik}}) = \frac{{{total_baik}}}{{{total_samples}}} = {prior_baik:.3f}")
    st.latex(rf"P(\text{{Kualitas}} = \text{{Buruk}}) = \frac{{{total_buruk}}}{{{total_samples}}} = {prior_buruk:.3f}")
    
    # Hitung Mean dan Variansi untuk setiap fitur pada setiap kelas
    features = list(X_full.columns)
    input_values = data_input_array[0]
    
    st.markdown("#### **Langkah 2: Menghitung Likelihood Probabilitas Fitur Menggunakan Distribusi Normal (Gaussian)**")
    st.write("Untuk setiap fitur, kita hitung seberapa 'cocok' nilai input Anda dengan distribusi data 'Baik' dan 'Buruk' menggunakan rumus PDF Gaussian.")
    st.latex(r"P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} e^{-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}}")
    
    # Tabel parameter latih
    param_data = []
    likelihoods_baik = []
    likelihoods_buruk = []
    
    for i, feat in enumerate(features):
        val = input_values[i]
        
        # Kelas Baik
        mean_b = df_calc[df_calc['Kualitas'] == 'Baik'][feat].mean()
        var_b = df_calc[df_calc['Kualitas'] == 'Baik'][feat].var()
        var_b = max(var_b, 1e-9) # Hindari variansi nol
        like_b = (1 / np.sqrt(2 * np.pi * var_b)) * np.exp(-((val - mean_b)**2) / (2 * var_b))
        likelihoods_baik.append(like_b)
        
        # Kelas Buruk
        mean_bu = df_calc[df_calc['Kualitas'] == 'Buruk'][feat].mean()
        var_bu = df_calc[df_calc['Kualitas'] == 'Buruk'][feat].var()
        var_bu = max(var_bu, 1e-9) # Hindari variansi nol
        like_bu = (1 / np.sqrt(2 * np.pi * var_bu)) * np.exp(-((val - mean_bu)**2) / (2 * var_bu))
        likelihoods_buruk.append(like_bu)
        
        param_data.append({
            "Fitur": feat, "Nilai Input": f"{val:.1f}",
            "Likelihood P(X|Baik)": f"{like_b:.5f}",
            "Likelihood P(X|Buruk)": f"{like_bu:.5f}"
        })
        
    st.table(pd.DataFrame(param_data))
    
    # Langkah 3: Mengalikan Semua Likelihood dan Prior
    total_like_baik = float(np.prod(likelihoods_baik)) * prior_baik
    total_like_buruk = float(np.prod(likelihoods_buruk)) * prior_buruk
    
    # Normalisasi
    normalizer = total_like_baik + total_like_buruk
    prob_baik_calc = total_like_baik / normalizer if normalizer > 0 else 0.5
    prob_buruk_calc = total_like_buruk / normalizer if normalizer > 0 else 0.5
    
    st.markdown("#### **Langkah 3: Mengalikan Semua Probabilitas & Normalisasi**")
    st.write("Kita kalikan probabilitas Prior dengan semua Likelihood untuk setiap kelas, lalu normalisasi hasilnya agar totalnya menjadi 100%.")
    st.latex(r"P(\text{Kelas} | X) \propto P(\text{Kelas}) \times \prod_{i=1}^{n} P(x_i | \text{Kelas})")
    
    st.markdown("**Hasil Probabilitas Akhir (Setelah Normalisasi):**")
    st.latex(rf"P(\text{{Baik}}) = \frac{{{total_like_baik:.2e}}}{{{total_like_baik:.2e} + {total_like_buruk:.2e}}} = {prob_baik_calc * 100:.2f}\%")
    st.latex(rf"P(\text{{Buruk}}) = \frac{{{total_like_buruk:.2e}}}{{{total_like_baik:.2e} + {total_like_buruk:.2e}}} = {prob_buruk_calc * 100:.2f}\%")
    
    st.success(f"👉 Kelas dengan probabilitas tertinggi dipilih oleh model. Maka hasil akhirnya adalah kelas **{prediksi.upper()}**.")
