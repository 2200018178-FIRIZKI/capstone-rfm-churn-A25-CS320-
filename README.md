
# Customer Segmentation & Churn Prediction with RFM (Capstone A25-CS320)

Proyek ini merupakan bagian dari **Asah Led by Dicoding in association with Accenture**  
Case: **[AC-06] Customer Segmentation for Personalized Retail Marketing**

Kami mengembangkan pipeline end-to-end untuk:
- melakukan **segmentasi pelanggan** berbasis metrik **RFM (Recency, Frequency, Monetary)**, dan  
- membangun **model prediksi churn** dengan beberapa algoritma machine learning,

kemudian menghasilkan **dataset final** yang siap digunakan di **Tableau Dashboard** untuk pengambilan keputusan bisnis.

Notebook utama: **`Segmentation.ipynb`**  
Output utama: **`final_result_scores.csv`**

---

## Daftar Isi

1. [Latar Belakang & Problem Statement](#1-latar-belakang--problem-statement)  
2. [Tujuan Proyek & Research Questions](#2-tujuan-proyek--research-questions)  
3. [Dataset & Fitur Utama](#3-dataset--fitur-utama)  
4. [Arsitektur & Alur Analisis](#4-arsitektur--alur-analisis)  
5. [Struktur Notebook](#5-struktur-notebook)  
6. [Cara Menjalankan & Replikasi](#6-cara-menjalankan--replikasi)  
7. [Output & Integrasi ke Tableau](#7-output--integrasi-ke-tableau)  
8. [Ringkasan Hasil & Insight](#8-ringkasan-hasil--insight)  
9. [Tim Pengembang](#9-tim-pengembang)  
10. [Lisensi](#10-lisensi)

---

## 1. Latar Belakang & Problem Statement

Perusahaan ritel dengan basis pelanggan yang besar sering menghadapi beberapa masalah:

- **Promosi yang kurang tepat sasaran**  
  Kampanye dilakukan secara massal, tidak mempertimbangkan nilai pelanggan (customer value) dan perilaku transaksi. Akibatnya biaya marketing tinggi namun dampak terhadap penjualan dan retensi tidak optimal.

- **Kesulitan mengidentifikasi pelanggan berisiko churn**  
  Perusahaan sering terlambat menyadari bahwa pelanggan sudah tidak aktif lagi. Tanpa pemantauan risiko churn, banyak peluang pendapatan jangka panjang yang hilang.

- **Keterbatasan insight operasional**  
  Data transaksi sudah tersedia, tetapi belum diolah menjadi insight yang mudah digunakan oleh tim bisnis (misalnya: segmen mana yang harus diprioritaskan, berapa kontribusi revenue tiap segmen, dll).

**Problem Statement**  
> Bagaimana cara mengelompokkan pelanggan berdasarkan perilaku transaksi dan memprediksi risiko churn mereka, sehingga tim bisnis dapat menjalankan strategi pemasaran dan retensi yang lebih terarah?

---

## 2. Tujuan Proyek & Research Questions

### Tujuan Proyek

1. Membangun **model segmentasi pelanggan** berbasis RFM untuk membedakan segmen bernilai tinggi, medium, dan rendah.
2. Mengembangkan **model prediksi churn** yang memberikan **probabilitas risiko churn** per pelanggan.
3. Menghasilkan **dataset final** yang dapat dengan mudah divisualisasikan dalam **dashboard interaktif** (Tableau), sehingga tim non-teknis dapat mengeksplorasi data tanpa menulis kode.

### Research Questions

1. **Bagaimana distribusi Recency, Frequency, dan Monetary pelanggan pada data penjualan ritel?**  
2. **Berapa jumlah segmen pelanggan yang optimal bila menggunakan metode clustering pada fitur RFM?**  
3. **Seberapa baik model machine learning (Logistic Regression, Random Forest, DNN) dalam memprediksi churn?**  
4. **Segmen pelanggan mana yang memiliki kontribusi revenue terbesar dan mana yang memiliki risiko churn tertinggi?**

---

## 3. Dataset & Fitur Utama

Nama file yang digunakan di notebook:

- `online_sales_dataset.csv`

> **Catatan:** File ini tidak disertakan di repo jika bersifat privat.  
> Silakan menambahkan file tersebut secara lokal dan menyesuaikan path di notebook.

Beberapa kolom penting:

- **Informasi transaksi**
  - `InvoiceNo`, `InvoiceDate`
  - `Quantity`, `UnitPrice`, `Discount`, `ShippingCost`
  - `ReturnStatus`
- **Informasi pelanggan**
  - `CustomerID`
  - `Country`
- **Informasi penjualan**
  - `Category`
  - `SalesChannel` (online / in-store)
  - `PaymentMethod`
  - `ShipmentProvider`
  - `WarehouseLocation`
  - `OrderPriority`

### Data Filtering yang dilakukan

Di dalam `Segmentation.ipynb`, kami melakukan langkah berikut:

1. Hanya mengambil transaksi **mulai tahun 2022**:
   - `InvoiceDate.dt.year >= 2022`
2. Menghapus baris dengan:
   - `CustomerID` kosong
   - `Quantity <= 0` atau `UnitPrice <= 0`
3. Hanya menyertakan transaksi yang **tidak dikembalikan**:
   - `ReturnStatus == "Not Returned"`
4. Membuat fitur baru:
   - `TotalPrice = Quantity * UnitPrice * (1 - DiscountClipped)`,  
     dengan `Discount` di-*clip* antara 0–1.

---

## 4. Arsitektur & Alur Analisis

Secara garis besar pipeline di notebook adalah sebagai berikut:

1. **Data Understanding & Cleaning**
   - Membaca dataset, mengecek missing value, tipe data, dan statistik dasar.
   - Membersihkan data sesuai kriteria pada bagian sebelumnya.
   - Menghitung `TotalPrice` sebagai dasar perhitungan Monetary.

2. **RFM Aggregation**
   - Menentukan `snapshot_date` = tanggal transaksi terakhir + 1 hari.
   - Menghitung:
     - `Recency` = selisih hari antara `snapshot_date` dan transaksi terakhir.
     - `Frequency` = jumlah invoice unik per pelanggan.
     - `Monetary` = total `TotalPrice` per pelanggan.

3. **Scaling & Transformasi**
   - Melakukan **log transform (`np.log1p`)** pada Recency, Frequency, Monetary untuk mengurangi skewness.
   - Menggunakan **StandardScaler** untuk men-distandarisasi fitur RFM.

4. **Clustering RFM (K-Means)**
   - Mencoba berbagai nilai `k` (2–10).
   - Menghitung:
     - **Silhouette Score** per k.
     - **Inertia (SSE)** untuk Elbow Method.
   - Memilih **k = 4** berdasarkan kombinasi Silhouette & Elbow.
   - Menjalankan **KMeans(k=4)** dan menambahkan kolom `Cluster` pada tabel RFM.

5. **Menentukan Cluster “Churn” & Membentuk Label**
   - Menghitung statistik rata-rata `Recency`, `Monetary` per cluster.
   - Menandai cluster sebagai **churn cluster** jika:
     - Recency di atas median (sudah lama tidak transaksi), dan
     - Monetary di bawah median (nilai belanja kecil).
   - Jika tidak ada cluster yang memenuhi, digunakan skor kombinasi Recency tinggi & Monetary rendah.
   - Membentuk label:
     - `Churn = 1` untuk cluster yang dianggap churn.
     - `Churn = 0` untuk cluster lain.

6. **Agregasi Fitur Tambahan per Customer**
   - Dari data transaksi asli dihitung:
     - `TotalQuantity`, `AvgUnitPrice`, `AvgDiscount`, `AvgShippingCost`.
     - Mode (nilai terbanyak) untuk `Country`, `PaymentMethod`, `Category`, `SalesChannel`, `ReturnStatus`, `ShipmentProvider`, `WarehouseLocation`, `OrderPriority`.
   - Menggabungkan dengan tabel RFM + Churn → `data_cust`.

7. **Persiapan Data untuk Model Churn**
   - Memisahkan **fitur numerik** dan **kategorikal**.
   - Melakukan **one-hot encoding** (`pd.get_dummies`) untuk fitur kategorikal.
   - Train-test split (70:30) dengan `stratify` pada label `Churn`.
   - Scaling numerik untuk model tertentu dengan `StandardScaler`.

8. **Training Model Machine Learning**
   - Model yang diuji:
     - Logistic Regression
     - Random Forest Classifier
     - Deep Neural Network (Keras Sequential)
   - Terdapat fungsi `evaluate_model` untuk:
     - Melatih model.
     - Menampilkan Accuracy, Precision, Recall, F1-score pada data uji.
   - DNN dilatih dengan:
     - Beberapa hidden layer dan dropout.
     - **EarlyStopping** berdasarkan `val_loss`.

   Dari hasil evaluasi, **Random Forest** dipilih sebagai model utama karena memberikan keseimbangan terbaik antara akurasi dan recall.

9. **Menghasilkan Churn Score**
   - Menggunakan `rf.predict_proba()` untuk menghitung probabilitas churn (`churn_score_rf`) antara 0–1.
   - Menyimpan ke dataframe `result_scores`.

10. **Ekspor Dataset Final**
    - Menyimpan `result_scores` dalam file:
      - `final_result_scores.csv`
    - File ini yang kemudian digunakan di Tableau untuk membuat berbagai visualisasi dan dashboard.

11. **Interpretasi Segmen & Rekomendasi Bisnis**
    - Notebook juga berisi:
      - Perhitungan **profil cluster** (rata-rata Recency, Frequency, Monetary).
      - **Ranking RFM score** untuk memberi label segmen (High Value, Low Value, Almost Lost, dsb).
      - Penulisan **rekomendasi bisnis** per segmen.

---

## 5. Struktur Notebook

`Segmentation.ipynb` secara garis besar terbagi menjadi beberapa bagian dengan heading:

1. **DATA UNDERSTANDING (Izza Tsamaro Hammidyah)**
2. **Data Cleaning**
3. **EDA (Fakih Widatmojo)**
4. **Agregasi RFM**
5. **Scaling (Fakih Widatmojo)**
6. **Clustering KMEANS (Fakih Widatmojo)**
7. **Feature Engineering & Agregasi Fitur Tambahan**
8. **Percobaan Logistic Regression, Random Forest, dan Deep Neural Network**
9. **Evaluasi Model & Pemilihan Random Forest**
10. **INTERPRETASI SEGMEN (Shah Firizki Azmi)**
11. **Rekomendasi Bisnis**
12. **Dashboard & Laporan**

Struktur ini memudahkan pembaca mengikuti alur dari **raw data → model → business insight**.

---

## 6. Cara Menjalankan & Replikasi

### 6.1. Menjalankan di Google Colab (disarankan)

1. Buka notebook **`Segmentation.ipynb`** dari repo ini.  
   Bila ada badge “Open in Colab”, klik badge tersebut.
2. Upload file `online_sales_dataset.csv` ke Colab, atau:
   - Mount Google Drive, lalu sesuaikan path di sel pertama notebook.
3. Jalankan seluruh sel **secara berurutan dari atas ke bawah**.
4. Setelah selesai:
   - File `final_result_scores.csv` akan muncul di direktori kerja / drive.
   - Unduh file tersebut untuk digunakan di Tableau.

> Kelebihan Colab: tidak perlu instalasi lokal, semua dependensi (`pandas`, `scikit-learn`, `tensorflow`, dll.) sudah tersedia.

### 6.2. Menjalankan Secara Lokal (Jupyter Notebook)

#### a. Kloning Repo

```bash
git clone https://github.com/<username>/capstone-rfm-churn-A25-CS320-.git
cd capstone-rfm-churn-A25-CS320-
