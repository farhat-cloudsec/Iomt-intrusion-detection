import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("🔍 Attack comparison images bana rahe hain!")

data_folder = r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data'

y_all = np.load(f'{data_folder}/wifi_labels.npy', allow_pickle=True)
X_all = np.load(f'{data_folder}/wifi_images.npy', allow_pickle=True)

attack_classes = ['ARP_Spoofing', 'TCP_IP-DDoS-ICMP1', 'TCP_IP-DDoS-SYN1']

benign_idx = np.where(y_all == 'Benign')[0][0]
benign_sample = X_all[benign_idx]
benign_resized = tf.image.resize(np.expand_dims(benign_sample, 0), [32, 32]).numpy()[0]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Normal Traffic vs Attack Traffic\nGASF & GADF Image Comparison — WiFi Protocol',
             fontsize=15, fontweight='bold', y=1.01)

# Column headers
col_headers = ['Benign — GASF', 'Benign — GADF', 'Attack — GASF', 'Attack — GADF']
for col, title in enumerate(col_headers):
    axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)

for i, attack_cls in enumerate(attack_classes):
    attack_idx = np.where(y_all == attack_cls)[0][0]
    attack_sample = X_all[attack_idx]
    attack_resized = tf.image.resize(
        np.expand_dims(attack_sample, 0), [32, 32]).numpy()[0]

    axes[i, 0].imshow(benign_resized[:,:,0], cmap='viridis')
    axes[i, 0].set_ylabel(f'vs\n{attack_cls}', fontsize=10, fontweight='bold', rotation=90)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    axes[i, 1].imshow(benign_resized[:,:,1], cmap='plasma')
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])

    axes[i, 2].imshow(attack_resized[:,:,0], cmap='viridis')
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])

    axes[i, 3].imshow(attack_resized[:,:,1], cmap='plasma')
    axes[i, 3].set_xticks([])
    axes[i, 3].set_yticks([])

plt.tight_layout(pad=2.0)
plt.savefig(f'{data_folder}/attack_vs_benign_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Attack comparison graph save ho gaya!")