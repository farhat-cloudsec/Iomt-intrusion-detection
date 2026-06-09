import numpy as np
import matplotlib.pyplot as plt

# Metrics data
protocols = ['WiFi', 'MQTT', 'Bluetooth']
precision = [0.92, 0.99, 0.83]
recall = [0.88, 0.99, 0.74]
f1 = [0.88, 0.99, 0.72]

x = np.arange(len(protocols))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4dabf7', edgecolor='black')
bars2 = ax.bar(x, recall, width, label='Recall', color='#51cf66', edgecolor='black')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#ff922b', edgecolor='black')

ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Protocol-wise Precision, Recall & F1-Score\nMobileNetV2 on CICIoMT2024', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(protocols, fontsize=12)
ax.legend(fontsize=11)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(r'C:\Users\Nafeesa\Desktop\IOMT\iomt-intrusion-detection\data\IOMT-Project-Data\metrics_graph.png', dpi=150)
plt.show()
print("✅ Metrics graph save ho gaya!")