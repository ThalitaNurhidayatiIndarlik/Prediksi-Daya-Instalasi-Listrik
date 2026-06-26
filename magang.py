import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# ==========================
# FILTERING DAYA
# ==========================

keyword = input(
    "Masukkan daya yang ingin dicari: "
)

hasil_filter = total_per_daya[
    total_per_daya.index.astype(str)
    .str.contains(keyword, case=False, na=False)
]

if len(hasil_filter) == 0:
    print("Data tidak ditemukan")
else:
    print("\nHasil pencarian:")
    print(hasil_filter)

# =====================================================
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

# Urutkan berdasarkan tanggal
df = df.sort_values(tanggal_col).reset_index(drop=True)

df["TOTAL_INSTALASI"] = df[daya_cols].sum(axis=1)

# ==========================
# DATA TRAINING DAN TESTING
# ==========================

split = int(len(df) * 0.8)

train = df["TOTAL_INSTALASI"][:split]
test = df["TOTAL_INSTALASI"][split:]

# Membuat tanggal forecast 7 hari
forecast_dates = pd.date_range(
    start=df[tanggal_col].max() + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

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

# ===========================================
# EVALUASI MOVING AVERAGE
# ===========================================

window = 7

history = list(train)

pred_ma_test = []

for actual in test:
    pred = np.mean(history[-window:])
    pred_ma_test.append(pred)
    history.append(actual)

mse_ma = mean_squared_error(test, pred_ma_test)

print("\n===== MSE MOVING AVERAGE =====")
print(f"MSE : {mse_ma:.2f}")

# ===========================================
# SIMPLE EXPONENTIAL SMOOTHING
# ===========================================

model = SimpleExpSmoothing(df["TOTAL_INSTALASI"])
    
model_fit = model.fit(
    smoothing_level=0.3,
    optimized=False
)

# Forecast 7 hari
forecast = model_fit.forecast(7)

hasil_forecast = pd.DataFrame({
    "Tanggal": forecast_dates,
    "Prediksi SES": forecast.round().astype(int)
})

print("\n===== HASIL FORECAST SIMPLE EXPONENTIAL SMOOTHING =====")
print(hasil_forecast)

# ===========================================
# EVALUASI SIMPLE EXPONENTIAL SMOOTHING
# ===========================================

model_eval = SimpleExpSmoothing(train)

model_fit_eval = model_eval.fit(
    smoothing_level=0.3,
    optimized=False
)

pred_ses_test = model_fit_eval.forecast(len(test))

mse_ses = mean_squared_error(test, pred_ses_test)

print("\n===== MSE SIMPLE EXPONENTIAL SMOOTHING =====")
print(f"MSE : {mse_ses:.2f}")

# ===========================================
# LINEAR REGRESSION
# ===========================================

# Variabel waktu
X_train = np.arange(len(train)).reshape(-1,1)

# Membuat model
lr_model = LinearRegression()
lr_model.fit(X_train, train)

# Prediksi 7 hari ke depan
future_X = np.arange(len(df), len(df)+7).reshape(-1,1)

forecast_lr = lr_model.predict(future_X)
forecast_lr = np.clip(forecast_lr, 0, None)

hasil_lr = pd.DataFrame({
    "Tanggal": forecast_dates,
    "Prediksi LR": np.round(forecast_lr).astype(int)
})

print("\n===== HASIL FORECAST LINEAR REGRESSION =====")
print(hasil_lr)

# ===========================================
# EVALUASI LINEAR REGRESSION
# ===========================================

X_train = np.arange(len(train)).reshape(-1,1)
X_test = np.arange(len(train), len(train)+len(test)).reshape(-1,1)

lr_eval = LinearRegression()
lr_eval.fit(X_train, train)

pred_lr_test = lr_eval.predict(X_test)

mse_lr = mean_squared_error(test, pred_lr_test)

print("\n===== MSE LINEAR REGRESSION =====")
print(f"MSE : {mse_lr:.2f}")

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
# FORECAST SIMPLE EXPONENTIAL SMOOTHING
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
    label="Simple Exponential Smoothing"
)

plt.title("Forecast Jumlah Instalasi 7 Hari ke Depan Menggunakan Simple Exponential Smoothing")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Instalasi")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==============================
# FIGURE 4
# FORECAST LINEAR REGRESSION
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
    forecast_lr,
    marker='o',
    linestyle='--',
    linewidth=2,
    label="Linear Regression"
)

plt.title("Forecast Jumlah Instalasi 7 Hari ke Depan Menggunakan Linear Regression")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Instalasi")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 5
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
    forecast_ma,
    marker='s',
    linestyle=':',
    linewidth=2,
    label="Moving Average"
)

plt.plot(
    forecast_dates,
    forecast,
    marker='o',
    linestyle='--',
    linewidth=2,
    label="Simple Exponential Smoothing"
)

plt.plot(
    forecast_dates,
    forecast_lr,
    marker='^',
    linestyle='-.',
    linewidth=2,
    label="Linear Regression"
)

plt.title("Perbandingan Forecasting 7 Hari ke Depan")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Instalasi")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==========================
# TABEL EVALUASI
# ==========================

hasil_mse = pd.DataFrame({
    "Metode": [
        "Moving Average",
        "Simple Exponential Smoothing",
        "Linear Regression"
    ],
    "MSE":[
    round(mse_ma,2),
    round(mse_ses,2),
    round(mse_lr,2)
    ]
})

print("\n===== HASIL EVALUASI =====")
print(hasil_mse)

# ==========================
# METODE TERBAIK
# ==========================

metode_terbaik = hasil_mse.loc[hasil_mse["MSE"].idxmin()]

print("\n===== METODE TERBAIK =====")
print(f"Metode : {metode_terbaik['Metode']}")
print(f"MSE    : {metode_terbaik['MSE']:.2f}")
