from pyts.image import GramianAngularField
import numpy as np
import matplotlib.pyplot as plt

# Normal traffic simulate karo
normal = np.array([[
    0.3, 0.3, 0.4, 0.3, 0.3,
    0.4, 0.3, 0.3, 0.4, 0.3,
    0.3, 0.4, 0.3, 0.3, 0.4,
    0.3, 0.3, 0.4, 0.3, 0.3
]])

# Attack traffic simulate karo
attack = np.array([[
    0.3, 0.3, 0.9, 0.9, 0.9,
    0.9, 0.9, 0.3, 0.3, 0.3,
    0.9, 0.9, 0.9, 0.3, 0.3,
    0.9, 0.9, 0.9, 0.3, 0.3
]])

# GASF
gasf = GramianAngularField(
    image_size=20,
    method='summation')

# GADF
gadf = GramianAngularField(
    image_size=20,
    method='difference')

# Convert karo
normal_gasf = gasf.fit_transform(normal)
attack_gasf = gasf.fit_transform(attack)
normal_gadf = gadf.fit_transform(normal)
attack_gadf = gadf.fit_transform(attack)

# Show karo
fig, axes = plt.subplots(
    2, 2, figsize=(12, 10))

axes[0,0].imshow(
    normal_gasf[0], cmap='rainbow')
axes[0,0].set_title(
    'Normal Traffic - GASF',
    fontsize=14)
axes[0,0].axis('off')

axes[0,1].imshow(
    attack_gasf[0], cmap='rainbow')
axes[0,1].set_title(
    'Attack Traffic - GASF',
    fontsize=14)
axes[0,1].axis('off')

axes[1,0].imshow(
    normal_gadf[0], cmap='rainbow')
axes[1,0].set_title(
    'Normal Traffic - GADF',
    fontsize=14)
axes[1,0].axis('off')

axes[1,1].imshow(
    attack_gadf[0], cmap='rainbow')
axes[1,1].set_title(
    'Attack Traffic - GADF',
    fontsize=14)
axes[1,1].axis('off')

plt.suptitle(
    'Normal vs Attack Traffic Images',
    fontsize=16)
plt.tight_layout()
plt.savefig(
    'farhat/results/normal_vs_attack.png')
plt.show()

print("Dekho kitna farq hai!")