import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

print("🔍 Zero-Day Test shuru!")

data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'

# WiFi V2 model load karo - ARP_Spoofing WiFi se seekha
model = tf.keras.models.load_model(f'{data_folder}/wifi_model_v2_best.h5')
print("✅ WiFi V2 model loaded!")

# LabelEncoder
known_classes = ['ARP_Spoofing', 'Benign', 'TCP_IP-DDoS-ICMP1', 'TCP_IP-DDoS-SYN1']
le = LabelEncoder()
le.fit(known_classes)
print(f"✅ Known classes: {le.classes_}")

# Bluetooth ARP_Spoofing load karo - model ne BLUETOOTH data kabhi nahi dekha!
y_bt = np.load(f'{data_folder}/bluetooth_labels.npy', allow_pickle=True)
X_bt = np.load(f'{data_folder}/bluetooth_images.npy', allow_pickle=True)
arp_idx = np.where(y_bt == 'ARP_Spoofing')[0]
X_zeroday = X_bt[arp_idx[:500]]
print(f"✅ Bluetooth Zero-Day samples: {X_zeroday.shape}")
del X_bt, y_bt

# Preprocess
X_zeroday = tf.image.resize(X_zeroday, [32, 32]).numpy()
X_zeroday = np.concatenate([X_zeroday, X_zeroday[:,:,:,:1]], axis=-1)

# Predict
predictions = model.predict(X_zeroday, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1)

# Threshold
threshold = 0.95
suspicious = confidence < threshold
detected = np.sum(suspicious)

print("\n📊 Cross-Protocol Zero-Day Test Results:")
print(f"Total Bluetooth ARP_Spoofing tested: {len(X_zeroday)}")
print(f"Average confidence: {confidence.mean()*100:.2f}%")
print(f"Threshold: {threshold*100}%")
print(f"\n⚠️ Suspicious (Zero-Day Detected): {detected} samples")
print(f"✅ Zero-Day Detection Rate: {detected/len(confidence)*100:.2f}%")

print(f"\nModel ne kya predict kiya:")
for i, cls in enumerate(le.classes_):
    count = np.sum(pred_classes == i)
    print(f"  {cls}: {count} ({count/len(pred_classes)*100:.1f}%)")

# Graph 1 - Confidence
plt.figure(figsize=(10, 4))
plt.hist(confidence, bins=20, color='steelblue', edgecolor='black')
plt.title('Bluetooth ARP_Spoofing Confidence (Cross-Protocol Zero-Day Test)')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.axvline(x=threshold, color='red', linestyle='--',
            label=f'Threshold {threshold*100}%')
plt.legend()
plt.tight_layout()
plt.savefig(f'{data_folder}/zeroday_confidence.png')
plt.show()

# Graph 2 - Pie
labels = [f'Zero-Day Detected\n({detected})', 
          f'Classified as Known\n({len(confidence)-detected})']
colors = ['#ff6b6b', '#51cf66']
plt.figure(figsize=(7, 7))
plt.pie([detected, len(confidence)-detected], labels=labels,
        colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Cross-Protocol Zero-Day Detection Results')
plt.tight_layout()
plt.savefig(f'{data_folder}/zeroday_pie.png')
plt.show()
print("✅ Graphs save ho gaye!")