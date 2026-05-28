import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print("Libraries loaded!")

# Data load
print("Loading data...")
df = pd.read_csv('data/preprocessed_data.csv')
print(f"Shape: {df.shape}")

# Features aur labels
feature_cols = [col for col in df.columns 
                if col != 'Label']
X = df[feature_cols].values
y = df['Label'].values

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")

# Train test split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded))

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# Pehle 20 features
X_train_20 = X_train[:, :20]
X_test_20 = X_test[:, :20]

# Converters
gasf = GramianAngularField(
    image_size=20,
    method='summation')
gadf = GramianAngularField(
    image_size=20,
    method='difference')

# Convert function
def convert_to_dual(X, batch_size=500):
    all_images = []
    total = len(X)
    
    for i in range(0, total, batch_size):
        batch = X[i:i+batch_size]
        
        gasf_imgs = gasf.fit_transform(batch)
        gadf_imgs = gadf.fit_transform(batch)
        
        dual = np.stack(
            [gasf_imgs, gadf_imgs], axis=-1)
        all_images.append(dual)
        
        print(f"Progress: {i}/{total}")
    
    return np.concatenate(all_images, axis=0)

# Train convert
print("\nConverting train images...")
X_train_img = convert_to_dual(X_train_20)
print(f"Train images: {X_train_img.shape}")

# Test convert
print("\nConverting test images...")
X_test_img = convert_to_dual(X_test_20)
print(f"Test images: {X_test_img.shape}")

# Save karo
print("\nSaving files...")
np.save('data/X_train_images.npy', X_train_img)
np.save('data/X_test_images.npy', X_test_img)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

import pickle
pickle.dump(le, open(
    'data/label_encoder.pkl', 'wb'))

print("\nAll files saved!")
print("Files ready for Nafeesa!")
print(f"X_train_images: {X_train_img.shape}")
print(f"X_test_images: {X_test_img.shape}")