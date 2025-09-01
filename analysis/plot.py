import matplotlib.pyplot as plt
import numpy as np

risk_descriptions = [
    'Insufficient telemetry data',
    'Integration difficulties',
    'Regulatory compliance failure',
    'Computational constraints',
    'Performance degradation',
    'Milestone delays'
]

likelihood = [3, 2, 1, 2, 2, 2]  # 1=Low, 2=Medium, 3=High
impact     = [3, 2, 1, 2, 3, 2]

colors = []
for l, i in zip(likelihood, impact):
    score = l * i
    if score >= 6:
        colors.append('red')
    elif score >= 3:
        colors.append('yellow')
    else:
        colors.append('green')

coord_groups = {}
for idx, (l, i) in enumerate(zip(likelihood, impact)):
    coord_groups.setdefault((l, i), []).append(idx)

offset_radius = 0.2
x_offsets = np.zeros(len(likelihood))
y_offsets = np.zeros(len(impact))

for (l, i), idxs in coord_groups.items():
    n = len(idxs)
    if n == 1:
        continue
    for j, idx in enumerate(idxs):
        angle = 2 * np.pi * j / n
        x_offsets[idx] = offset_radius * np.cos(angle)
        y_offsets[idx] = offset_radius * np.sin(angle)

x_plot = [l + dx for l, dx in zip(likelihood, x_offsets)]
y_plot = [i + dy for i, dy in zip(impact, y_offsets)]

plt.figure(figsize=(10, 7))
plt.scatter(x_plot, y_plot, s=400, c=colors, alpha=0.6, edgecolors='w', linewidth=2)

for xi, yi, desc in zip(x_plot, y_plot, risk_descriptions):
    plt.annotate(
        desc,
        (xi, yi),
        textcoords="offset points",
        xytext=(5,5),
        ha='left',
        va='bottom',
        color='black',
        fontsize=10,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
    )

plt.xlabel('Likelihood (1=Low, 3=High)', fontsize=12)
plt.ylabel('Impact (1=Low, 3=High)', fontsize=12)
plt.title('Risk Chart', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0.5, 3.5)
plt.ylim(0.5, 3.5)
plt.tight_layout()
plt.show()