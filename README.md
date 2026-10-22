# Sistem Optimasi Reorder Point (ROP), Safety Stock Dinamis, dan Pencegahan Kehabisan Stok
### *Inventory Control & Stockout Prevention System (Fashion Vintage 90s Shopee Store)*

---

## 📌 1. Latar Belakang & Masalah Bisnis *(Business Understanding)*

Dalam industri ritel daring (*e-commerce*) pakaian *vintage* 90an, manajemen rantai pasok (*supply chain*) menghadapi tantangan *trade-off* fundamental:
1. **Risiko Kehabisan Stok (*Stockout Cost*):** Kehilangan potensi omzet penjualan harian, penalti performa toko dari Shopee, dan penurunan loyalitas pelanggan saat stok barang populer habis.
2. **Risiko Penumpukan Stok (*Overstock / Holding Cost*):** Terkuncinya modal kerja (*working capital*) dan naiknya biaya simpan barang.
3. **Variabilitas Lead Time Supplier:** Waktu tunggu pengiriman (*lead time*) dari pemasok lokal/vintage sering kali mengalami keterlambatan hingga $+4$ hari meleset dari target, yang memicu kehabisan stok tak terduga jika hanya mengandalkan rumus persediaan statis.

**Tujuan Proyek:**  
Membangun sistem *Supply Chain Data Science* berbasis Python untuk menghitung *Safety Stock* ($SS$) dan *Reorder Point* ($ROP$) secara dinamis menggunakan **King's Combined Formula**, mengelompokkan SKU dengan segmentasi **ABC-XYZ**, memprediksi permintaan harian via **Machine Learning**, memvalidasi ketahanan persediaan lewat **Simulasi Monte Carlo 1.000 Siklus**, serta menghasilkan rekomendasi *Purchase Order* (PO) otomatis dengan kepatuhan batas $MOQ$ supplier.

---

## 🏆 2. Pencapaian Key Performance Indicators (KPIs)

Berdasarkan pengujian simulasi empiris 1.000 siklus pemesanan terhadap 15 SKU aktif:

| Key Performance Indicator (KPI) | Baseline (Aturan ROP Statis) | Model ROP Dinamis King's | Evaluasi Capaian Proyek |
| :--- | :---: | :---: | :--- |
| **Stockout Rate (Simulasi Monte Carlo 1.000 Siklus)** | **18.7%** | **2.1%** | **Target KPI < 5% Tercapai** (Penurunan risiko sebesar 16.6% poin). |
| **Pendeteksian Sinyal Palsu (*False Security*)** | 2 SKU lolos (*Track Jacket* & *Raglan*) | Terdeteksi **PERLU RESTOCK** | Menyelamatkan potensi kehilangan omzet pada produk bernilai tinggi. |
| **Kepatuhan Batas MOQ Supplier** | 0% (tidak ada *guardrail*) | **100% Guardrail MOQ** | Pesanan PO disesuaikan otomatis dengan batas minimum order 6 supplier. |
| **Penanganan Produk Kurasi 1-of-1** | Tercampur dengan barang batch | Terisolasi di `Sourcing per item` | Melindungi barang antik langka dari penerbitan pesanan batch yang keliru. |
| **Akurasi Status Ketersediaan Stok** | 60.0% | **93.3%** | Status dinamis (`Aman`, `Mendekati ROP`, `PERLU RESTOCK`) sangat presisi. |

---

## 🔬 3. Landasan Matematika, Metodologi, & Machine Learning

### A. Formula Safety Stock Dinamis (King's Formula)
Berbeda dengan rumus konvensional yang hanya melihat fluktuasi permintaan, **King's Formula** menggabungkan variansi permintaan ($\sigma_d$) bersamaan dengan variansi waktu tunggu pemasok ($\sigma_{LT}$):
$$SS = Z \times \sqrt{\left(\overline{LT} \times \sigma_d^2\right) + \left(\bar{d}^2 \times \sigma_{LT}^2\right)}$$

- $Z$: Nilai faktor *Service Level* (ditentukan dari segmentasi matriks ABC-XYZ, rentang 90% - 98%).
- $\overline{LT}$: Rata-rata *Lead Time* aktual pemasok (data riil: 8.46 hari).
- $\sigma_{LT}$: Standar deviasi *Lead Time* pemasok (data riil: 2.50 hari).
- $\bar{d}$: Rata-rata penjualan harian per SKU ($\mu_d$).
- $\sigma_d$: Standar deviasi penjualan harian per SKU.

### B. Formula Reorder Point Dinamis
$$ROP_{dynamic} = \lceil (\bar{d} \times LT_{target}) + SS \rceil$$

### C. Segmentasi Portofolio 9-Kuadran ABC - XYZ
- **Analisis ABC (Kontribusi Omzet Kumulatif):**
  - **Kelas A (Kumulatif $\le 80\%$):** Produk unggulan omzet tertinggi (*Kemeja Flanel Kotak, Jeans Straight, Hoodie Zip, Track Jacket, Celana Cargo*). Diberi proteksi *Service Level* **98%** ($Z = 2.05$).
  - **Kelas B (Kumulatif $80\% - 95\%$):** Produk kontributor menengah (*Kaos Grunge, Raglan Baseball Tee, Crewneck*). Diberi *Service Level* **93% - 95%** ($Z = 1.48 - 1.65$).
  - **Kelas C (Kumulatif $> 95\%$):** Produk *long-tail*. Diberi *Service Level* **85% - 90%** ($Z = 1.04 - 1.28$) untuk efisiensi modal simpan.
- **Analisis XYZ (Volatilitas Penjualan):**
  - Mengukur Koefisien Variasi ($CV_d = \sigma_d / \mu_d$). Seluruh SKU fashion vintage berada pada kategori Z ($CV > 1.0$) karena pola pembelian harian yang *lumpy/intermittent*.

### D. Optimasi Ukuran Pesanan: EOQ & Guardrail MOQ Supplier
$$Qty\_PO = \max\left(\lceil EOQ \rceil, MOQ_{Supplier}\right)$$
Di mana Economic Order Quantity ($EOQ$) dihitung dari:
$$EOQ = \sqrt{\frac{2 \times D \times S}{H}}$$
- $D$: Permintaan disetahunkan ($\bar{d} \times 365$).
- $S$: Biaya administrasi & logistik per pemesanan (Rp 50.000 / PO).
- $H$: Biaya simpan tahunan per unit ($20\% \times Modal\_Unit\_Rp$).

### E. Predictive Analytics & Machine Learning
- Membangun model regresi (*Random Forest* dan *Ridge Regressor*) untuk memprediksi volume penjualan harian menggunakan fitur: *Day of Week*, *Is Weekend*, *Diskon Promo Shopee*, *Lag-1 Demand*, dan *Rolling Mean 3-Hari*.
- Klasifikasi bahaya kehabisan stok (*Time-to-Stockout*) berdasarkan metrik *Days of Inventory Remaining* ($DoI = Stok\_Akhir / \hat{d}$).

### F. Simulasi Monte Carlo (1.000 Iterasi Siklus Pemesanan)
- Mensimulasikan 1.000 skenario acak di mana *lead time* berdistribusi normal $\mathcal{N}(\overline{LT}, \sigma_{LT})$ dan permintaan berdistribusi Poisson $\text{Pois}(\bar{d} \times LT_{sim})$.
- Menguji frekuensi kegagalan pemenuhan persediaan pada ROP statis vs ROP dinamis.

---

## 🗂️ 4. Struktur Direktori Proyek

```text
├── dataset_supply_chain_pakaian_90an_shopee.xlsx   # Dataset inti (5 tabel relasional Excel)
├── 01_eda_dan_segmentasi_abc_xyz.ipynb             # Notebook Tahap 1: EDA & Segmentasi ABC/XYZ
├── 02_pemodelan_rop_dan_simulasi_stok.ipynb        # Notebook Tahap 2-4: King's SS, ML, Monte Carlo
├── src/
│   └── inventory_engine.py                         # Production Engine kalkulator ROP & generator PO
├── app.py                                          # Web Dashboard Interaktif Streamlit
├── rekomendasi_po_supplier.xlsx                    # Berkas Excel resmi rekomendasi pemesanan PO
├── requirements.txt                                # Daftar dependensi pustaka Python
├── .gitignore                                      # Konfigurasi pengabaian file Git
└── README.md                                       # Dokumentasi lengkap arsitektur sistem
```

---

## ⚙️ 5. Panduan Instalasi & Persiapan Lingkungan

### Langkah 1: Kloning Repositori & Masuk ke Direktori
```bash
git clone <URL_REPOSITORI_ANDA>
cd "Data Scientist - Analisis Efisiensi Logistik & Durasi Pemenuhan Pesanan"
```

### Langkah 2: Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
```bash
# Membuat virtual environment
python -m venv .venv

# Aktivasi di Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Aktivasi di Windows (Command Prompt)
.venv\Scripts\activate.bat

# Aktivasi di macOS/Linux
source .venv/bin/activate
```

### Langkah 3: Pasang Seluruh Dependensi
```bash
pip install -r requirements.txt
```

---

## 🚀 6. Panduan Menjalankan Sistem

### A. Membuka Notebook Analisis (Jupyter Lab / VS Code)
Buka file `.ipynb` di VS Code atau jalankan Jupyter Notebook:
- [01_eda_dan_segmentasi_abc_xyz.ipynb](file:///c:/Users/hpvic/OneDrive/Documents/Data%20Scientist%20-%20Analisis%20Efisiensi%20Logistik%20&%20Durasi%20Pemenuhan%20Pesanan/01_eda_dan_segmentasi_abc_xyz.ipynb): Eksplorasi data, profil kecepatan penjualan, variabilitas lead time pemasok, dan kurva Pareto ABC.
- [02_pemodelan_rop_dan_simulasi_stok.ipynb](file:///c:/Users/hpvic/OneDrive/Documents/Data%20Scientist%20-%20Analisis%20Efisiensi%20Logistik%20&%20Durasi%20Pemenuhan%20Pesanan/02_pemodelan_rop_dan_simulasi_stok.ipynb): Formulasi matematika King's SS, perbandingan ROP statis vs dinamis, evaluasi ML Regressor, dan simulasi Monte Carlo 1.000 iterasi.

### B. Menjalankan Skrip Generator PO Otomatis
Jalankan modul CLI produksi untuk menghitung status stok dan mengekspor file rekomendasi PO terbaru:
```bash
python src/inventory_engine.py
```
*Output:* File Excel [rekomendasi_po_supplier.xlsx](file:///c:/Users/hpvic/OneDrive/Documents/Data%20Scientist%20-%20Analisis%20Efisiensi%20Logistik%20&%20Durasi%20Pemenuhan%20Pesanan/rekomendasi_po_supplier.xlsx) langsung diperbarui otomatis.

### C. Menjalankan Web Dashboard Interaktif Streamlit
Jalankan aplikasi dashboard visual:
```bash
streamlit run app.py
```
Akses di browser pada: **`http://localhost:8501`**

**Fitur Unggulan Dashboard:**
- **Pemantauan Sisa Stok vs ROP:** Bar chart dinamis membandingkan posisi stok aktual terhadap ambang batas aman.
- **Slider Interaktif Service Level:** Mengubah target *Service Level* (90% - 99%) secara *real-time* untuk melihat dampaknya ke jumlah pengaman stok.
- **Matriks 9-Kuadran ABC-XYZ:** Heatmap interaktif distribusi 15 SKU.
- **One-Click PO Downloader:** Mengunduh berkas pesanan pembelian (.xlsx) siap kirim ke supplier.

---

## 📋 7. Rekomendasi Purchase Order (PO) Darurat

Berdasarkan status persediaan aktual, berikut adalah 6 SKU *Restockable* yang berada pada atau di bawah titik ROP dinamis dan membutuhkan penerbitan PO segera:

| SKU | Nama Produk | Pemasok Rekanan | Stok Akhir | ROP Dinamis | MOQ Supplier | Qty Rekomendasi PO | Estimasi Modal PO |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V90-FLN-001** | Kemeja Flanel Kotak | Flanel Archive | 8 | 9 | 10 | **39 unit** *(EOQ)* | Rp 2.028.000 |
| **V90-TEE-002** | Kaos Washed Boxy | Studio Washed Apparel | 5 | 5 | 24 | **36 unit** *(EOQ)* | Rp 1.368.000 |
| **V90-CGO-001** | Celana Cargo 90s | Garasi Vintage Supply | 5 | 8 | 24 | **34 unit** *(EOQ)* | Rp 2.584.000 |
| **V90-SWT-001** | Crewneck Colorblock | Retro Outerwear Select | 7 | 8 | 8 | **32 unit** *(EOQ)* | Rp 2.176.000 |
| **V90-TEE-003** | Raglan Baseball Tee | Studio Washed Apparel | 7 | 7 | 12 | **37 unit** *(EOQ)* | Rp 1.665.000 |
| **V90-TRK-001** | Track Jacket Retro | Retro Outerwear Select | 7 | 7 | 6 | **22 unit** *(EOQ)* | Rp 1.936.000 |
| **TOTAL** | | | | | | **200 unit** | **Rp 11.757.000** |

---

## 💡 8. Wawasan Bisnis Khusus (Supply Chain Vintage)

1. **Segmentasi Dua Jalur Pasokan (*Dual Supply Mode*):**
   - **Restockable (10 SKU):** Barang pasokan berulang dari vendor konveksi/grosir dengan batasan MOQ. Dikelola menggunakan formula King's SS, ROP dinamis, dan kalkulasi EOQ.
   - **One-off Vintage (5 SKU):** Barang kurasi per unit (*thrift 1-of-1* seperti jaket denim dan windbreaker antik). Dipisahkan ke status `Sourcing per item` dengan ROP = 0 untuk mencegah penerbitan pesanan batch yang keliru.
2. **Mitigasi Deviasi Lead Time:**
   - Karena supplier memiliki deviasi keterlambatan pengiriman $\sigma_{LT} = 2.50$ hari, penggabungan variansi pada rumus King's terbukti krusial untuk mencegah kehabisan stok saat terjadi lonjakan pesanan di Shopee (*Payday Sale* atau *Spill Produk*).
