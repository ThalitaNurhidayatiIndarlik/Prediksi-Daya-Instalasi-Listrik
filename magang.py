import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# ==========================
# BACA FILE EXCEL
# ==========================
file_path = "FEBRUARI.xlsx"

raw = pd.read_excel(file_path, header=None)

# Cari baris header
header_row = None
for i in range(len(raw)):
    if str(raw.iloc[i, 2]).strip().upper() == "TGL PERMOHONAN":
        header_row = i
        break

if header_row is None:
    raise ValueError("Header 'TGL PERMOHONAN' tidak ditemukan")

# Baca ulang dengan header yang benar
df = pd.read_excel(file_path, header=header_row)

# Hapus kolom kosong
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

# Isi NaN dengan 0
df = df.fillna(0)

# ==========================
# AMBIL KOLOM DAYA
# ==========================
daya_cols = [col for col in df.columns if "DAYA" in str(col).upper()]

if len(daya_cols) == 0:
    raise ValueError("Kolom DAYA tidak ditemukan")

# Total permohonan per daya
total_per_daya = df[daya_cols].sum()

# =====================================================
# FIGURE 2
# FORECAST JUMLAH INSTALASI 7 HARI KE DEPAN
# =====================================================

# Cari kolom tanggal
tanggal_col = None
for col in df.columns:
    if "TGL" in str(col).upper():
        tanggal_col = col
        break

if tanggal_col is None:
    raise ValueError("Kolom tanggal tidak ditemukan.")

# Ubah menjadi datetime
bulan = {
    "JANUARI": "JANUARY",
    "FEBRUARI": "FEBRUARY",
    "MARET": "MARCH",
    "APRIL": "APRIL",
    "MEI": "MAY",
    "JUNI": "JUNE",
    "JULI": "JULY",
    "AGUSTUS": "AUGUST",
    "SEPTEMBER": "SEPTEMBER",
    "OKTOBER": "OCTOBER",
    "NOVEMBER": "NOVEMBER",
    "DESEMBER": "DECEMBER"
}

df[tanggal_col] = df[tanggal_col].astype(str)

for indo, eng in bulan.items():
    df[tanggal_col] = df[tanggal_col].str.replace(indo, eng, regex=False)

df[tanggal_col] = pd.to_datetime(df[tanggal_col], dayfirst=True)

# Hitung total instalasi setiap hari
df["TOTAL_INSTALASI"] = df[daya_cols].sum(axis=1)

# Urutkan berdasarkan tanggal
df = df.sort_values(tanggal_col)

# ===========================================
# EXPONENTIAL SMOOTHING
# ===========================================

model = SimpleExpSmoothing(df["TOTAL_INSTALASI"])
    
model_fit = model.fit(
    smoothing_level=0.3,
    optimized=True
)

# Forecast 7 hari
forecast = model_fit.forecast(7)

# Membuat tanggal prediksi (dibuat lebih dulu)
forecast_dates = pd.date_range(
    start=df[tanggal_col].max() + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

hasil_forecast = pd.DataFrame({
    "Tanggal": forecast_dates,
    "Prediksi ES": forecast.round().astype(int)
})

print("\n===== HASIL FORECAST EXPONENTIAL SMOOTHING =====")
print(hasil_forecast)

# ===========================================
# MOVING AVERAGE
# ===========================================

window = 7

history = list(df["TOTAL_INSTALASI"])

forecast_ma = []

for i in range(7):
    pred = np.mean(history[-window:])
    forecast_ma.append(pred)
    history.append(pred)

forecast_ma = np.array(forecast_ma)

hasil_ma = pd.DataFrame({
    "Tanggal": forecast_dates,
    "Prediksi MA": np.round(forecast_ma).astype(int)
})

print("\n===== HASIL FORECAST MOVING AVERAGE =====")
print(hasil_ma)

# ==========================
# FIGURE 1
# VISUALISASI TREN
# ==========================
plt.figure(figsize=(14, 6))

plt.plot(
    total_per_daya.index,
    total_per_daya.values,
    marker='o',
    linewidth=2
)

plt.title("Visualisasi Tren Permohonan Berdasarkan Daya")
plt.xlabel("Kategori Daya")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=90)
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 2
# FORECAST MOVING AVERAGE
# ==========================

plt.figure(figsize=(14,6))

plt.plot(
    df[tanggal_col],
    df["TOTAL_INSTALASI"],
    marker='o',
    linewidth=2,
    label="Data Aktual"
)

plt.plot(
    forecast_dates,
    forecast_ma,
    marker='o',
    linestyle='--',
    linewidth=2,
    label="Moving Average"
)

plt.title("Forecast Jumlah Instalasi 7 Hari ke Depan Menggunakan Moving Average")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Instalasi")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==============================
# FIGURE 3
# FORECAST EXPONENTIAL SMOOTHING
# ==============================
plt.figure(figsize=(14,6))

plt.plot(
    df[tanggal_col],
    df["TOTAL_INSTALASI"],
    marker='o',
    linewidth=2,
    label="Data Aktual"
)

plt.plot(
    forecast_dates,
    forecast,
    marker='o',
    linestyle='--',
    linewidth=2,
    label="Exponential Smoothing"
)

plt.title("Forecast Jumlah Instalasi 7 Hari ke Depan Menggunakan Exponential Smoothing")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Instalasi")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 4
# PERBANDINGAN METODE
# ==========================

plt.figure(figsize=(14,6))

plt.plot(
    df[tanggal_col],
    df["TOTAL_INSTALASI"],
    color="black",
    linewidth=2,
    label="Data Aktual"
)

plt.plot(
    forecast_dates,
    forecast,
    marker='o',
    linestyle='--',
    linewidth=2,
    label="Exponential Smoothing"
)

plt.plot(
    forecast_dates,
    forecast_ma,
    marker='s',
    linestyle=':',
    linewidth=2,
    label="Moving Average"
)

plt.title("Perbandingan Forecasting 7 Hari ke Depan")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Instalasi")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

