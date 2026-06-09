import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("✅ Libraries loaded!")

data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'

# Labels load karo
y_all = np.load(f'{data_folder}/wifi_labels.npy', allow_pickle=True)

# Balanced sampling
selected = []
for cls in ['ARP_Spoofing', 'Benign', 'TCP_IP-DDoS-ICMP1', 'TCP_IP-DDoS-SYN1']:
    idx = np.where(y_all == cls)[0]
    np.random.shuffle(idx)
    idx = idx[:8000]
    selected.extend(idx)

np.random.shuffle(selected)
selected = np.array(selected)
print(f"✅ Selected {len(selected)} samples")

# Images load
X_all = np.load(f'{data_folder}/wifi_images.npy', allow_pickle=True)
X = X_all[selected]
y = y_all[selected]
del X_all, y_all

# Labels encode
le = LabelEncoder()
y = le.fit_transform(y)
print(f"✅ Classes: {le.classes_}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Resize to 32x32
X_train = tf.image.resize(X_train, [32, 32]).numpy()
X_test = tf.image.resize(X_test, [32, 32]).numpy()

# 2 to 3 channels
X_train = np.concatenate([X_train, X_train[:,:,:,:1]], axis=-1)
X_test = np.concatenate([X_test, X_test[:,:,:,:1]], axis=-1)

print(f"✅ Shape: {X_train.shape}")

# Model
base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("✅ Model ready!")

# Callbacks
checkpoint = ModelCheckpoint(
    f'{data_folder}/wifi_model_v3_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# Training
print("\n🚀 WiFi V3 Training shuru...")
model.fit(X_train, y_train_cat, epochs=30, batch_size=32,
          validation_split=0.1, verbose=1,
          callbacks=[checkpoint, early_stop, reduce_lr])

# Best model load
model = tf.keras.models.load_model(f'{data_folder}/wifi_model_v3_best.h5')

# Test
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ WiFi V3 Test Accuracy: {test_acc*100:.2f}%")

model.save(f'{data_folder}/wifi_model_v3.h5')
print("✅ WiFi V3 Model save ho gaya!")