import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("📊 Graphs banana shuru!")

data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'

# =====================
# Graph 1: Protocol-wise Accuracy Bar Chart
# =====================
protocols = ['WiFi\n(4 classes)', 'MQTT\n(2 classes)', 'Bluetooth\n(2 classes)']
accuracies = [96.27, 99.72, 89.52]
colors = ['#4dabf7', '#51cf66', '#ff922b']

plt.figure(figsize=(10, 6))
bars = plt.bar(protocols, accuracies, color=colors, edgecolor='black', width=0.5)
plt.ylim(80, 102)
plt.ylabel('Accuracy (%)', fontsize=13)
plt.title('Protocol-wise Classification Accuracy\nMobileNetV2 on CICIoMT2024', fontsize=14)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{data_folder}/protocol_accuracy.png', dpi=150)
plt.show()
print("✅ Graph 1 done!")

# =====================
# Graph 2: Zero-Day Results
# =====================
labels = ['Correctly Classified\nas ARP_Spoofing (500)', 'Flagged as\nSuspicious (79)']
sizes = [421, 79]
colors2 = ['#51cf66', '#ff6b6b']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors2, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12})
plt.title('Cross-Protocol Zero-Day Test Results\n(Bluetooth ARP_Spoofing — Unseen Attack)',
          fontsize=13)
plt.tight_layout()
plt.savefig(f'{data_folder}/zeroday_final.png', dpi=150)
plt.show()
print("✅ Graph 2 done!")

# =====================
# Graph 3: WiFi Confusion Matrix
# =====================
print("\n🔄 WiFi Confusion Matrix bana raha hai...")

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

cm = confusion_matrix(y_encoded, pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('WiFi Protocol — Confusion Matrix', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{data_folder}/wifi_confusion_matrix.png', dpi=150)
plt.show()
print("✅ Graph 3 done!")

print("\n🎉 Sab graphs save ho gaye!")
# =====================
# Graph 4: GASF vs GADF vs Grad-CAM Comparison
# =====================
print("\n🔄 GASF/GADF/Grad-CAM comparison graph bana raha hai...")

from PIL import Image

y_all = np.load(f'{data_folder}/wifi_labels.npy', allow_pickle=True)
X_all = np.load(f'{data_folder}/wifi_images.npy', allow_pickle=True)

known_classes = ['ARP_Spoofing', 'Benign', 'TCP_IP-DDoS-ICMP1', 'TCP_IP-DDoS-SYN1']
le2 = LabelEncoder()
le2.fit(known_classes)

model = tf.keras.models.load_model(f'{data_folder}/wifi_model_v2_best.h5')

samples = []
sample_labels = []
for cls in known_classes:
    idx = np.where(y_all == cls)[0][0]
    samples.append(X_all[idx])
    sample_labels.append(cls)
samples = np.array(samples)
del X_all, y_all

samples_resized = tf.image.resize(samples, [32, 32]).numpy()
samples_3ch = np.concatenate([samples_resized, samples_resized[:,:,:,:1]], axis=-1)

def get_gradcam2(model, img, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('out_relu').output, model.output]
    )
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(np.expand_dims(img, 0), tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for j in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, j] *= pooled_grads[j]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((32, 32))
    return np.array(heatmap_img) / 255.0

fig, axes = plt.subplots(4, 3, figsize=(12, 14))
fig.suptitle('GASF vs GADF vs Grad-CAM Comparison\nWiFi Protocol — CICIoMT2024', 
             fontsize=14, fontweight='bold')

col_titles = ['GASF Channel', 'GADF Channel', 'Grad-CAM Overlay']
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

for i, (sample, label) in enumerate(zip(samples_3ch, sample_labels)):
    pred = model.predict(np.expand_dims(sample, 0), verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = np.max(pred[0]) * 100
    heatmap = get_gradcam2(model, sample, pred_class)

    # GASF
    axes[i, 0].imshow(sample[:,:,0], cmap='viridis')
    axes[i, 0].set_ylabel(f'{label}', fontsize=9, fontweight='bold')
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([]) 

    # GADF
    axes[i, 1].imshow(sample[:,:,1], cmap='plasma')
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])  




    

    # Grad-CAM
    axes[i, 2].imshow(sample[:,:,0], cmap='viridis')
    axes[i, 2].imshow(heatmap, cmap='jet', alpha=0.6)
    axes[i, 2].set_title(f'{confidence:.1f}%', fontsize=9)
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])

plt.tight_layout()
plt.savefig(f'{data_folder}/gasf_gadf_gradcam_comparison.png', dpi=150)
plt.show()
print("✅ GASF/GADF/Grad-CAM comparison graph done!")