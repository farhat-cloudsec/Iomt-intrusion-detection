import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

print("📊 Metrics nikaal rahe hain!")

data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'

# WiFi
print("\n🔄 WiFi metrics...")
y_all = np.load(f'{data_folder}/wifi_labels.npy', allow_pickle=True)
X_all = np.load(f'{data_folder}/wifi_images.npy', allow_pickle=True)

known_classes = ['ARP_Spoofing', 'Benign', 'TCP_IP-DDoS-ICMP1', 'TCP_IP-DDoS-SYN1']
le = LabelEncoder()
le.fit(known_classes)

selected = []
for cls in known_classes:
    idx = np.where(y_all == cls)[0][:1000]
    selected.extend(idx)
selected = np.array(sorted(selected))

X = X_all[selected]
y = y_all[selected]
del X_all, y_all

y_encoded = le.transform(y)
X_resized = tf.image.resize(X, [32, 32]).numpy()
X_3ch = np.concatenate([X_resized, X_resized[:,:,:,:1]], axis=-1)

model = tf.keras.models.load_model(f'{data_folder}/wifi_model_v2_best.h5')
preds = model.predict(X_3ch, verbose=0)
pred_classes = np.argmax(preds, axis=1)

print("\n✅ WiFi Classification Report:")
print(classification_report(y_encoded, pred_classes, target_names=le.classes_))

# MQTT
print("\n🔄 MQTT metrics...")
y_mqtt = np.load(f'{data_folder}/mqtt_labels.npy', allow_pickle=True)
X_mqtt = np.load(f'{data_folder}/mqtt_images.npy', allow_pickle=True)

mqtt_classes = ['Benign', 'MQTT-DDoS-Connect_Flood']
le_mqtt = LabelEncoder()
le_mqtt.fit(mqtt_classes)

selected_mqtt = []
for cls in mqtt_classes:
    idx = np.where(y_mqtt == cls)[0][:1000]
    selected_mqtt.extend(idx)
selected_mqtt = np.array(sorted(selected_mqtt))

X_m = X_mqtt[selected_mqtt]
y_m = y_mqtt[selected_mqtt]
del X_mqtt, y_mqtt

y_m_encoded = le_mqtt.transform(y_m)
X_m_resized = tf.image.resize(X_m, [32, 32]).numpy()
X_m_3ch = np.concatenate([X_m_resized, X_m_resized[:,:,:,:1]], axis=-1)

model_mqtt = tf.keras.models.load_model(f'{data_folder}/mqtt_model_best.h5')
preds_mqtt = model_mqtt.predict(X_m_3ch, verbose=0)
pred_mqtt = np.argmax(preds_mqtt, axis=1)

print("\n✅ MQTT Classification Report:")
print(classification_report(y_m_encoded, pred_mqtt, target_names=le_mqtt.classes_))

# Bluetooth
print("\n🔄 Bluetooth metrics...")
y_bt = np.load(f'{data_folder}/bluetooth_labels.npy', allow_pickle=True)
X_bt = np.load(f'{data_folder}/bluetooth_images.npy', allow_pickle=True)

bt_classes = ['ARP_Spoofing', 'Benign']
le_bt = LabelEncoder()
le_bt.fit(bt_classes)

selected_bt = []
for cls in bt_classes:
    idx = np.where(y_bt == cls)[0][:1000]
    selected_bt.extend(idx)
selected_bt = np.array(sorted(selected_bt))

X_b = X_bt[selected_bt]
y_b = y_bt[selected_bt]
del X_bt, y_bt

y_b_encoded = le_bt.transform(y_b)
X_b_resized = tf.image.resize(X_b, [32, 32]).numpy()
X_b_3ch = np.concatenate([X_b_resized, X_b_resized[:,:,:,:1]], axis=-1)

model_bt = tf.keras.models.load_model(f'{data_folder}/bluetooth_model_best.h5')
preds_bt = model_bt.predict(X_b_3ch, verbose=0)
pred_bt = np.argmax(preds_bt, axis=1)

print("\n✅ Bluetooth Classification Report:")
print(classification_report(y_b_encoded, pred_bt, target_names=le_bt.classes_))

print("\n🎉 Sab metrics complete!")