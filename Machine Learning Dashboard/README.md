# Machine Learning Dashboard

Aplikasi Streamlit yang komprehensif untuk machine learning dengan fitur-fitur lengkap untuk analisis data dan pemodelan.

## Fitur Utama

✅ **Upload Data & Tampilan Data**

- Upload file CSV atau Excel
- Preview data dengan informasi lengkap
- Analisis missing values dan tipe data

✅ **Pemilihan Task (Klasifikasi/Regresi)**

- Deteksi otomatis task yang sesuai berdasarkan target
- Validasi kompatibilitas task dengan data
- Panduan pemilihan task yang tepat

✅ **Pemilihan Kolom Target**

- Analisis karakteristik target column
- Deteksi binary vs multiclass classification
- Visualisasi distribusi target

✅ **Preprocessing Data**

- Handle missing values (mean untuk numerik, mode untuk kategorikal)
- Encoding variabel kategorikal
- Feature scaling (StandardScaler/MinMaxScaler)

✅ **Pembagian Data**

- Konfigurasi train-test split yang fleksibel
- Visualisasi proporsi pembagian data
- Stratified split untuk klasifikasi

✅ **Cross Validation**

- Optional cross validation dengan K-fold
- Stratified K-fold untuk klasifikasi
- Visualisasi hasil cross validation

✅ **Pemilihan Model Multiple**

- **Klasifikasi**: Random Forest, Logistic Regression, SVM, Decision Tree, KNN, Naive Bayes, Gradient Boosting
- **Regresi**: Random Forest, Linear Regression, SVM, Decision Tree, KNN, Gradient Boosting
- Dapat memilih lebih dari 1 model sekaligus

✅ **Evaluasi Komprehensif**

- **Klasifikasi**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report
- **Regresi**: MSE, RMSE, MAE, R² Score
- Evaluasi dengan cross validation dan test set

✅ **Perbandingan Model**

- Tabel perbandingan performa semua model
- Visualisasi bar chart dan radar chart
- Rekomendasi model terbaik otomatis
- Confusion matrix untuk setiap model klasifikasi

## Instalasi

1. **Clone atau download repository ini**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**

   ```bash
   streamlit run app.py
   ```

4. **Buka browser dan akses:**
   ```
   http://localhost:8501
   ```

## Struktur File

```
├── app.py              # Frontend Streamlit (tampilan)
├── ml_backend.py       # Backend logic (machine learning)
├── requirements.txt    # Dependencies
└── README.md          # Dokumentasi
```

## Cara Penggunaan

### 1. Upload Data

- Upload file CSV atau Excel Anda
- Review informasi data yang ditampilkan

### 2. Pilih Task

- Pilih antara **Classification** atau **Regression**
- Aplikasi akan memberikan panduan pemilihan

### 3. Pilih Target Column

- Pilih kolom yang ingin diprediksi
- Aplikasi akan menganalisis dan memvalidasi pilihan Anda

### 4. Preprocessing (Opsional)

- **Handle Missing Values**: Isi nilai yang hilang
- **Encode Categorical**: Konversi variabel kategorikal
- **Scale Features**: Normalisasi fitur

### 5. Split Data

- Atur persentase data training dan testing
- Set random state untuk reproducibility

### 6. Cross Validation (Opsional)

- Aktifkan cross validation untuk evaluasi yang lebih robust
- Pilih jumlah fold (3-10)

### 7. Pilih Model

- Pilih satu atau lebih model machine learning
- Model yang tersedia disesuaikan dengan task yang dipilih

### 8. Evaluasi

- Pilih metrik evaluasi yang diinginkan
- Jalankan evaluasi pada semua model

### 9. Lihat Hasil

- Review performa setiap model
- Bandingkan model dengan visualisasi
- Dapatkan rekomendasi model terbaik

## Format Data yang Didukung

- **CSV** (.csv)
- **Excel** (.xlsx, .xls)

## Catatan Penting

- Data dengan missing values pada target column akan dihapus otomatis
- Untuk klasifikasi, aplikasi melakukan stratified split
- Cross validation menggunakan seluruh dataset (train + test)
- Model terbaik dipilih berdasarkan metrik utama (accuracy untuk klasifikasi, R² untuk regresi)

## Troubleshooting

### Error saat upload data

- Pastikan file dalam format CSV atau Excel
- Periksa encoding file (gunakan UTF-8)

### Model tidak bisa dilatih

- Pastikan preprocessing sudah selesai
- Periksa apakah ada missing values di target column

### Evaluasi error

- Pastikan model sudah dilatih
- Periksa kompatibilitas metrik dengan task type

## Kontribusi

Jika Anda ingin berkontribusi atau melaporkan bug, silakan buat issue atau pull request.

## Lisensi

MIT License
