import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import pickle

print("✅ Libraries loaded!")

# Data folder
data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'
# Farhat ki files load karo
X_train = np.load(f'{data_folder}/X_train_images.npy')
X_test = np.load(f'{data_folder}/X_test_images.npy')
y_train = np.load(f'{data_folder}/y_train.npy')
y_test = np.load(f'{data_folder}/y_test.npy')

print(f"✅ Train images: {X_train.shape}")
print(f"✅ Test images: {X_test.shape}")
print(f"✅ Train labels: {y_train.shape}")
from tensorflow.keras.utils import to_categorical

# Labels encode karo
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

print(f"✅ Classes: {num_classes}")

# MobileNetV2 - 20x20 resize to 32x32
X_train_resized = tf.image.resize(X_train, [32, 32]).numpy()
X_test_resized = tf.image.resize(X_test, [32, 32]).numpy()

# 2 channels to 3 channels
X_train_resized = np.concatenate([X_train_resized, X_train_resized[:,:,:,:1]], axis=-1)
X_test_resized = np.concatenate([X_test_resized, X_test_resized[:,:,:,:1]], axis=-1)

print(f"✅ Final shape: {X_train_resized.shape}")

# Model
base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights=None)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("✅ Model ready!")

# Training
print("\n🚀 Training shuru...")
history = model.fit(
    X_train_resized, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Test
test_loss, test_acc = model.evaluate(X_test_resized, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Save
model.save(r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection-1\data\mobilenetv2_gasf_gadf.h5')
print("✅ Model save ho gaya!")