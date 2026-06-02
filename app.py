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
    
    # Buah Kualitas Baik (Manis tinggi, asam rendah, ukuran sedang-besar, berat ideal)
    baik_diameter = np.random.normal(8.0, 1.2, int(n_samples/2))
    baik_berat = np.random.normal(180, 25, int(n_samples/2))
    baik_kemanisan = np.random.normal(8.0, 1.0, int(n_samples/2))
    baik_keasaman = np.random.normal(3.0, 0.8, int(n_samples/2))
    
    # Buah Kualitas Buruk (Kemanisan rendah, keasaman tinggi, ukuran terlalu kecil/besar acak)
    buruk_diameter = np.random.normal(5.5, 1.5, int(n_samples/2))
    buruk_berat = np.random.normal(120, 35, int(n_samples/2))
    buruk_kemanisan = np.random.normal(4.0, 1.5, int(n_samples/2))
    buruk_keasaman = np.random.normal(7.0, 1.5, int(n_samples/2))
    
    data_baik = pd.DataFrame({
        'Diameter (cm)': baik_diameter,
        'Berat (gram)': baik_berat,
        'Kemanisan (Skala 1-10)': baik_kemanisan,
        'Keasaman (Skala 1-10)': baik_keasaman,
        'Kualitas': 'Baik'
    })
    
    data_buruk = pd.DataFrame({
        'Diameter (cm)': buruk_diameter,
        'Berat (gram)': buruk_berat,
        'Kemanisan (Skala 1-10)': buruk_kemanisan,
        'Keasaman (Skala 1-10)': buruk_keasaman,
        'Kualitas': 'Buruk'
    })
    
    df = pd.concat([data_baik, data_buruk], ignore_index=True)
    # Batasi nilai skala kemanisan/keasaman agar tetap di rentang 1-10
    df['Kemanisan (Skala 1-10)'] = df['Kemanisan (Skala 1-10)'].clip(1.0, 10.0)
    df['Keasaman (Skala 1-10)'] = df['Keasaman (Skala 1-10)'].clip(1.0, 10.0)
    return df

# Memuat data
df = load_default_data()

# ==========================================
# SIDEBAR - NAVIGASI DAN INFORMASI KELOMPOK
# ==========================================
with st.sidebar:
    # Menggunakan try-except untuk load gambar agar tidak crash jika offline
    try:
        st.image("[https://images.unsplash.com/photo-1619546813926-a78fa6372cd2?auto=format&fit=crop&q=80&w=300](https://images.unsplash.com/photo-1619546813926-a78fa6372cd2?auto=format&fit=crop&q=80&w=300)", caption="Klasifikasi Buah", use_container_width=True)
    except Exception:
        st.subheader("🍎 Klasifikasi Buah")
        
    st.title("🍎 Kelompok 4")
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
    st.title("🎯 Aplikasi Klasifikasi Kualitas Buah Menggunakan Naive Bayes")
    st.subheader("Selamat Datang di Projek Proyek Kelompok 4")
    
    st.markdown("""
    Aplikasi ini dirancang untuk mengklasifikasikan kualitas buah (Kategori: **Baik** atau **Buruk**) berdasarkan fitur fisik luar dan rasa buah. 
    Metode yang kita gunakan adalah **Gaussian Naive Bayes**, yang sangat cocok untuk mengolah data fitur kontinu (numerik).
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📌 Mengapa Studi Kasus Kualitas Buah?\n"
                "Dalam industri pertanian dan pangan, menyortir buah berkualitas tinggi dari yang rusak secara manual memerlukan banyak waktu dan tenaga. "
                "Dengan memanfaatkan algoritma Machine Learning seperti Naive Bayes, proses penyortiran dapat diotomatisasi secara cepat berdasarkan berat, ukuran, "
                "tingkat kemanisan (menggunakan sensor brix), dan tingkat keasaman (sensor pH).")
        
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
    st.title("📊 Eksplorasi Data Kualitas Buah")
    st.write("Di bawah ini adalah data sampel yang kita gunakan untuk melatih kecerdasan buatan dalam membedakan buah yang baik dan buruk.")
    
    st.write("### 📂 Data Table (200 Sampel Buah)")
    st.dataframe(df, use_container_width=True)
    
    # Statistik Deskriptif
    st.write("### 📈 Ringkasan Statistik Berdasarkan Kualitas Buah")
    st.write(df.groupby('Kualitas').mean())
    
    # Visualisasi Distribusi Fitur
    st.write("### 🖼️ Visualisasi Hubungan Antar Fitur")
    fitur_pilihan = st.selectbox("Pilih Fitur yang Ingin Dilihat Distribusinya:", df.columns[:-1])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(data=df, x=fitur_pilihan, hue="Kualitas", fill=True, common_norm=False, palette="Set1", alpha=0.5, ax=ax)
    plt.title(f"Distribusi Fitur: {fitur_pilihan} berdasarkan Kualitas Buah")
    st.pyplot(fig)
    plt.close(fig) # Menutup plot untuk membebaskan memori
    
    st.success("💡 **Observasi Singkat:** Buah berkualitas 'Baik' cenderung memiliki tingkat kemanisan yang lebih tinggi dan tingkat keasaman yang lebih rendah dibandingkan buah berkualitas 'Buruk'.")

# ==========================================
# HALAMAN 3: PELATIHAN DAN EVALUASI MODEL
# ==========================================
elif menu == "Latih & Evaluasi Model":
    st.title("⚙️ Proses Pelatihan & Evaluasi Naive Bayes")
    
    # Memisahkan Fitur dan Target
    X = df.drop(columns=['Kualitas'])
    y = df['Kualitas']
    
    # Slider untuk menentukan ukuran porsi data latih (Train-Test Split)
    test_size = st.slider("Pilih Persentase Data Uji (Test Set %):", min_value=10, max_value=50, value=20, step=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    # Melatih model Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Prediksi data uji
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 Akurasi Model Naive Bayes", value=f"{akurasi*100:.2f} %")
        st.write("Akurasi menunjukkan seberapa tepat model dalam mengklasifikasikan kualitas buah pada data uji baru yang belum pernah dilihat sebelumnya.")
        
    with col2:
        st.write("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Baik', 'Buruk'], yticklabels=['Baik', 'Buruk'], ax=ax)
        plt.xlabel('Prediksi Model')
        plt.ylabel('Kualitas Sebenarnya')
        st.pyplot(fig)
        plt.close(fig)
        
    st.write("### 📝 Laporan Klasifikasi Detil (Classification Report)")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df, use_container_width=True)

# ==========================================
# HALAMAN 4: PREDIKSI INTERAKTIF & CARA KERJA MATEMATIS
# ==========================================
elif menu == "Prediksi Interaktif (Uji Coba)":
    st.title("🔬 Prediksi Kualitas Buah Interaktif")
    st.write("Sesuaikan spesifikasi buah di bawah ini menggunakan slider, dan lihat apakah algoritma Naive Bayes mengklasifikasikannya sebagai buah **Baik** atau **Buruk**!")
    
    # Melatih model pada seluruh dataset untuk prediksi optimal
    X = df.drop(columns=['Kualitas'])
    y = df['Kualitas']
    model = GaussianNB()
    model.fit(X, y)
    
    # 1. FORM INPUT USER
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📥 Spesifikasi Fisik Buah:")
        input_diameter = st.slider("Diameter Buah (cm)", min_value=3.0, max_value=15.0, value=7.5, step=0.1)
        input_berat = st.slider("Berat Buah (gram)", min_value=50.0, max_value=300.0, value=160.0, step=1.0)
        
    with col2:
        st.markdown("### 👅 Spesifikasi Rasa Buah:")
        input_kemanisan = st.slider("Tingkat Kemanisan (Skala 1 - 10)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
        input_keasaman = st.slider("Tingkat Keasaman (Skala 1 - 10)", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
        
    # Menyiapkan array data baru
    data_input = np.array([[input_diameter, input_berat, input_kemanisan, input_keasaman]])
    prediksi = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0] # [Probabilitas Baik, Probabilitas Buruk]
    
    # Menampilkan Hasil Prediksi
    st.markdown("---")
    st.write("## 🏁 Hasil Klasifikasi Model:")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if prediksi == 'Baik':
            st.success(f"### Kualitas Buah: **BAIK** 🟢")
            st.write(f"Probabilitas Buah Berkualitas Baik: **{probabilitas[0]*100:.2f}%**")
        else:
            st.error(f"### Kualitas Buah: **BURUK** 🔴")
            st.write(f"Probabilitas Buah Berkualitas Buruk: **{probabilitas[1]*100:.2f}%**")
            
    with col_res2:
        # Tampilkan grafik probabilitas klasifikasi
        fig, ax = plt.subplots(figsize=(6, 2.5))
        y_pos = np.arange(2)
        ax.barh(y_pos, probabilitas * 100, color=['#4CAF50', '#F44336'], height=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Baik', 'Buruk'])
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
    
    # Hitung Prior Probabilities P(C)
    total_samples = len(df)
    total_baik = len(df[df['Kualitas'] == 'Baik'])
    total_buruk = len(df[df['Kualitas'] == 'Buruk'])
    
    prior_baik = total_baik / total_samples
    prior_buruk = total_buruk / total_samples
    
    st.markdown(f"#### **Langkah 1: Menghitung Probabilitas Prior (Awal)**")
    st.write(f"Total Sampel Buah = **{total_samples}** (Baik = **{total_baik}**, Buruk = **{total_buruk}**)")
    st.latex(rf"P(\text{{Kualitas}} = \text{{Baik}}) = \frac{{{total_baik}}}{{{total_samples}}} = {prior_baik:.3f}")
    st.latex(rf"P(\text{{Kualitas}} = \text{{Buruk}}) = \frac{{{total_buruk}}}{{{total_samples}}} = {prior_buruk:.3f}")
    
    # Hitung Mean dan Variansi untuk setiap fitur pada setiap kelas
    features = ['Diameter (cm)', 'Berat (gram)', 'Kemanisan (Skala 1-10)', 'Keasaman (Skala 1-10)']
    input_values = [input_diameter, input_berat, input_kemanisan, input_keasaman]
    
    st.markdown("#### **Langkah 2: Menghitung Likelihood Probabilitas Fitur Menggunakan Distribusi Normal (Gaussian)**")
    st.write("Karena fitur kita berupa angka kontinu, Naive Bayes menggunakan rumus Distribusi Gaussian untuk mencari kecocokan nilai input Anda terhadap pola rata-rata ($\mu$) dan standar deviasi ($\sigma$) data latih.")
    
    # Tabel parameter latih
    param_data = []
    likelihoods_baik = []
    likelihoods_buruk = []
    
    for i, feat in enumerate(features):
        val = input_values[i]
        
        # Kelas Baik
        mean_b = df[df['Kualitas'] == 'Baik'][feat].mean()
        var_b = df[df['Kualitas'] == 'Baik'][feat].var()
        # Hindari pembagian dengan nol dengan menambahkan epsilon kecil jika variansi nol
        var_b = max(var_b, 1e-9)
        # Rumus Gaussian Probability Density Function
        like_b = (1 / np.sqrt(2 * np.pi * var_b)) * np.exp(-((val - mean_b)**2) / (2 * var_b))
        likelihoods_baik.append(like_b)
        
        # Kelas Buruk
        mean_bu = df[df['Kualitas'] == 'Buruk'][feat].mean()
        var_bu = df[df['Kualitas'] == 'Buruk'][feat].var()
        var_bu = max(var_bu, 1e-9)
        like_bu = (1 / np.sqrt(2 * np.pi * var_bu)) * np.exp(-((val - mean_bu)**2) / (2 * var_bu))
        likelihoods_buruk.append(like_bu)
        
        param_data.append({
            "Fitur": feat,
            "Nilai Input": val,
            "Rata2 (Baik)": f"{mean_b:.2f}",
            "Variansi (Baik)": f"{var_b:.2f}",
            "Likelihood P(X|Baik)": f"{like_b:.5f}",
            "Rata2 (Buruk)": f"{mean_bu:.2f}",
            "Variansi (Buruk)": f"{var_bu:.2f}",
            "Likelihood P(X|Buruk)": f"{like_bu:.5f}"
        })
        
    st.table(pd.DataFrame(param_data))
    
    # Langkah 3: Mengalikan Semua Likelihood dan Prior (Aman dari math.prod Python 3.7 ke bawah)
    total_like_baik = float(np.prod(likelihoods_baik)) * prior_baik
    total_like_buruk = float(np.prod(likelihoods_buruk)) * prior_buruk
    
    # Normalisasi agar total probabilitas = 100%
    normalizer = total_like_baik + total_like_buruk
    if normalizer == 0:
        prob_baik_calc = 0.5
        prob_buruk_calc = 0.5
    else:
        prob_baik_calc = total_like_baik / normalizer
        prob_buruk_calc = total_like_buruk / normalizer
    
    st.markdown("#### **Langkah 3: Mengalikan Semua Probabilitas & Normalisasi**")
    st.write("Persamaan perkalian Naive Bayes:")
    st.latex(r"P(\text{Kelas} | X) \propto P(\text{Kelas}) \times P(\text{Diameter} | \text{Kelas}) \times P(\text{Berat} | \text{Kelas}) \times P(\text{Kemanisan} | \text{Kelas}) \times P(\text{Keasaman} | \text{Kelas})")
    
    st.markdown("**Hasil Perkalian Mentah (Posterior Tanpa Normalisasi):**")
    st.latex(rf"P(\text{{Baik}} | X) \propto {prior_baik:.3f} \times {likelihoods_baik[0]:.5f} \times {likelihoods_baik[1]:.5f} \times {likelihoods_baik[2]:.5f} \times {likelihoods_baik[3]:.5f} = {total_like_baik:.2e}")
    st.latex(rf"P(\text{{Buruk}} | X) \propto {prior_buruk:.3f} \times {likelihoods_buruk[0]:.5f} \times {likelihoods_buruk[1]:.5f} \times {likelihoods_buruk[2]:.5f} \times {likelihoods_buruk[3]:.5f} = {total_like_buruk:.2e}")
    
    st.markdown("**Hasil Probabilitas Akhir (Setelah Normalisasi):**")
    st.latex(rf"P(\text{{Baik}}) = \frac{{{total_like_baik:.2e}}}{{{total_like_baik:.2e} + {total_like_buruk:.2e}}} = {prob_baik_calc * 100:.2f}\%")
    st.latex(rf"P(\text{{Buruk}}) = \frac{{{total_like_buruk:.2e}}}{{{total_like_baik:.2e} + {total_like_buruk:.2e}}} = {prob_buruk_calc * 100:.2f}\%")
    
    st.info(f"👉 Kelas dengan probabilitas tertinggi dipilih oleh model. Maka hasil akhirnya adalah kelas **{prediksi.upper()}**.")