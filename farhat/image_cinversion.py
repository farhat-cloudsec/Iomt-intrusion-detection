import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import os

print("Libraries loaded!")

# Step 1: Data load karo
print("Loading data...")
df = pd.read_csv('data/preprocessed_data.csv')
print(f"Data shape: {df.shape}")
print(f"Labels: {df['Label'].unique()}")

# Step 2: Features aur Labels alag karo
feature_cols = [col for col in df.columns 
                if col != 'Label']
X = df[feature_cols].values
y = df['Label'].values

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Step 3: Pehle 20 features use karo
X_20 = X[:, :20]
print(f"Using 20 features: {X_20.shape}")

# Step 4: GASF+GADF converters
gasf = GramianAngularField(
    image_size=20,
    method='summation')

gadf = GramianAngularField(
    image_size=20,
    method='difference')

# Step 5: Small test pehle
print("Testing on 100 samples...")
X_test = X_20[:100]

gasf_test = gasf.fit_transform(X_test)
gadf_test = gadf.fit_transform(X_test)

print(f"GASF shape: {gasf_test.shape}")
print(f"GADF shape: {gadf_test.shape}")

# Step 6: Sample image dekho
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4))

axes[0].imshow(
    gasf_test[0], cmap='rainbow')
axes[0].set_title('GASF - Real Data')
axes[0].axis('off')

axes[1].imshow(
    gadf_test[0], cmap='rainbow')
axes[1].set_title('GADF - Real Data')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(
    'farhat/results/real_data_sample.png')
plt.show()

print("Test complete!")
print("Real data images ready!")