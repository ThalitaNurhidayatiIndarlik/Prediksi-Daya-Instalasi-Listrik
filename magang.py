import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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
# MOVING AVERAGE (MA-3)
# ==========================
moving_avg = total_per_daya.rolling(
    window=3,
    min_periods=1
).mean()

# ==========================
# EXPONENTIAL SMOOTHING
# α = 0.3
# (1-α) = 0.7
# ==========================
alpha = 0.3
beta = 0.7

exp_smoothing = [total_per_daya.iloc[0]]

for i in range(1, len(total_per_daya)):
    value = (
        alpha * total_per_daya.iloc[i]
        + beta * exp_smoothing[i - 1]
    )
    exp_smoothing.append(value)

exp_smoothing = pd.Series(
    exp_smoothing,
    index=total_per_daya.index
)

# ==========================
# LINEAR REGRESSION
# ==========================
X = np.arange(len(total_per_daya)).reshape(-1, 1)
y = total_per_daya.values

lr_model = LinearRegression()
lr_model.fit(X, y)

y_pred = lr_model.predict(X)

# ==========================
# NEURAL NETWORK
# ==========================
nn_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=5000,
    random_state=42
)

nn_model.fit(X, y)

y_nn = nn_model.predict(X)

# ==========================
# FILTERING 
# ==========================

print("\n=== DAFTAR KATEGORI DAYA ===")
for col in total_per_daya.index:
    print(col)

keyword = input(
    "\nMasukkan daya yang ingin dicari "
    "(contoh: 450, 900, 1300): "
)

hasil_filter = total_per_daya[
    total_per_daya.index.astype(str)
    .str.contains(keyword, case=False, na=False)
]

print("\n=== HASIL PENCARIAN ===")

if len(hasil_filter) > 0:
    print(hasil_filter)
else:
    print("Data daya tidak ditemukan.")
    
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
# MOVING AVERAGE
# ==========================
plt.figure(figsize=(14, 6))

plt.plot(
    total_per_daya.index,
    total_per_daya.values,
    marker='o',
    label='Data Asli'
)

plt.plot(
    moving_avg.index,
    moving_avg.values,
    marker='s',
    linewidth=2,
    label='Moving Average (3)'
)

plt.title("Moving Average (Window = 3)")
plt.xlabel("Kategori Daya")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 3
# EXPONENTIAL SMOOTHING
# ==========================
plt.figure(figsize=(14, 6))

plt.plot(
    total_per_daya.index,
    total_per_daya.values,
    marker='o',
    label='Data Asli'
)

plt.plot(
    exp_smoothing.index,
    exp_smoothing.values,
    marker='s',
    linewidth=2,
    label='Exponential Smoothing (α=0.3)'
)

plt.title("Exponential Smoothing (α = 0.3, 1-α = 0.7)")
plt.xlabel("Kategori Daya")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 4
# LINEAR REGRESSION
# ==========================
plt.figure(figsize=(14, 6))

plt.plot(
    total_per_daya.index,
    total_per_daya.values,
    marker='o',
    label='Data Asli'
)

plt.plot(
    total_per_daya.index,
    y_pred,
    linewidth=2,
    label='Linear Regression'
)

plt.title("Linear Regression Permohonan Berdasarkan Daya")
plt.xlabel("Kategori Daya")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 5
# NEURAL NETWORK
# ==========================
plt.figure(figsize=(14, 6))

plt.plot(
    total_per_daya.index,
    total_per_daya.values,
    marker='o',
    label='Data Asli'
)

plt.plot(
    total_per_daya.index,
    y_nn,
    linewidth=2,
    label='Neural Network'
)

plt.title("Neural Network Permohonan Berdasarkan Daya")
plt.xlabel("Kategori Daya")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================
# FIGURE 6
# HASIL FILTERING DAYA
# ==========================

if len(hasil_filter) > 0:

    plt.figure(figsize=(10,5))

    plt.bar(
        hasil_filter.index,
        hasil_filter.values
    )

    plt.title(
        f"Hasil Pencarian Daya: {keyword}"
    )
    plt.xlabel("Kategori Daya")
    plt.ylabel("Jumlah Permohonan")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()