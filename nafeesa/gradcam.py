import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("🔍 Grad-CAM shuru!")

data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'

model = tf.keras.models.load_model(f'{data_folder}/wifi_model_v2_best.h5')
print("✅ Model loaded!")

known_classes = ['ARP_Spoofing', 'Benign', 'TCP_IP-DDoS-ICMP1', 'TCP_IP-DDoS-SYN1']
le = LabelEncoder()
le.fit(known_classes)

y_all = np.load(f'{data_folder}/wifi_labels.npy', allow_pickle=True)
X_all = np.load(f'{data_folder}/wifi_images.npy', allow_pickle=True)

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

def get_gradcam(model, img, class_idx):
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
    
    # Simple resize using numpy
    from PIL import Image
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((32, 32))
    heatmap = np.array(heatmap_img) / 255.0
    return heatmap

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle('Grad-CAM Visualization — WiFi Protocol', fontsize=14)

for i, (sample, label) in enumerate(zip(samples_3ch, sample_labels)):
    pred = model.predict(np.expand_dims(sample, 0), verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = np.max(pred[0]) * 100
    
    heatmap = get_gradcam(model, sample, pred_class)
    
    axes[i, 0].imshow(sample[:,:,0], cmap='viridis')
    axes[i, 0].set_title(f'{label}\nGASF Channel')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(sample[:,:,1], cmap='viridis')
    axes[i, 1].set_title(f'GADF Channel')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(sample[:,:,0], cmap='viridis')
    axes[i, 2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[i, 2].set_title(f'Grad-CAM\nPred: {le.classes_[pred_class]}\n{confidence:.1f}%')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig(f'{data_folder}/gradcam_results.png', dpi=150)
plt.show()
print("✅ Grad-CAM save ho gaya!")