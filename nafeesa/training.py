import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical

print("✅ Libraries loaded!")

# Data load karo
data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data'
df = pd.read_csv(f'{data_folder}/preprocessed_data.csv')
print(f"✅ Data loaded: {len(df)} rows")

# X aur y alag karo
X = df.drop('Label', axis=1).values
y = df['Label'].values

# Label encode karo
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=5)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Image shape mein convert karo - 32x32 resize
X_train_img = X_train.reshape(-1, 5, 9, 1)
X_train_img = np.repeat(X_train_img, 3, axis=-1)
X_train_img = tf.image.resize(X_train_img, [32, 32]).numpy()

X_test_img = X_test.reshape(-1, 5, 9, 1)
X_test_img = np.repeat(X_test_img, 3, axis=-1)
X_test_img = tf.image.resize(X_test_img, [32, 32]).numpy()

print(f"✅ Image shape: {X_train_img.shape}")
print("✅ Data ready!")

# MobileNetV2 Model banana
base_model = MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights=None
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model ready!")
print(f"Total parameters: {model.count_params():,}")

# Training
print("\n🚀 Training shuru...")
history = model.fit(
    X_train_img, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

print("\n✅ Training complete!")
# Model save karo
model.save(f'{data_folder}/mobilenetv2_model.h5')
print("✅ Model save ho gaya!")

# Test pe evaluate karo
test_loss, test_acc = model.evaluate(X_test_img, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")