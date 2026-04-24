# Bitcoin Price Forecast

Proyek ini berfokus pada forecasting harga Bitcoin menggunakan pendekatan multivariate time series dengan dua arsitektur deep learning:

- Baseline LSTM + Custom Multi-Head Attention
- Seq2Seq LSTM + Attention (teacher forcing)

Notebook utama ada di file `Notebook.ipynb`.

## Tujuan Proyek

- Memprediksi harga Bitcoin untuk beberapa langkah ke depan.
- Membandingkan pendekatan baseline LSTM vs Seq2Seq.
- Menggunakan sinyal teknikal dan fitur multivariat untuk meningkatkan kualitas prediksi.

## Struktur Folder

```text
Bitcoin Price Forecast/
|- Notebook.ipynb
|- requirements.txt
|- model_baseline_lstm (2).keras
`- seq2seq_model.keras
```

## Dataset

- Dataset dimuat langsung dari URL Google Drive pada notebook.
- Data diindeks berdasarkan kolom tanggal (`Date`).
- Kolom target utama adalah `Price` (hasil rename dari `Close`).

## Alur Eksperimen

1. Import library dan set random seed untuk reproducibility.
2. Load data, data cleaning, dan EDA.
3. Analisis time series:
   - Dekomposisi tren-musiman-residual.
   - ACF/PACF untuk menentukan konfigurasi window.
4. Feature engineering (rolling statistics).
5. Split data train/validation/test dan normalisasi.
6. Membuat pipeline `tf.data.Dataset` untuk windowing multivariat.
7. Training dan evaluasi model Baseline LSTM.
8. Training dan evaluasi model Seq2Seq LSTM.
9. Simpan model ke format `.keras`.

## Konfigurasi Penting

- Window size: `300`
- Forecast horizon: `24`
- Framework utama: TensorFlow/Keras

## Dependencies

Paket utama di `requirements.txt`:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `statsmodels`

## Cara Menjalankan

1. Masuk ke folder proyek:

```bash
cd "Time-Series/Bitcoin Price Forecast"
```

2. (Opsional) Buat virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Jalankan notebook:

```bash
jupyter notebook "Notebook.ipynb"
```

## Output Proyek

- Model baseline: `model_baseline_lstm (2).keras`
- Model seq2seq: `seq2seq_model.keras`

## Catatan

- Fokus proyek adalah eksperimen end-to-end dalam notebook.
- Jika ingin deployment/inference script terpisah, bisa ditambahkan pada iterasi berikutnya.
