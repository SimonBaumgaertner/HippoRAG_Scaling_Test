import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'scaling_results.json')

df = pd.read_json(json_path)

# Polynomial Fitting
coeffs_poly = np.polyfit(df['document_count'], df['total_indexing_time_s'], 2)
polynomial = np.poly1d(coeffs_poly)
x_fit = np.linspace(df['document_count'].min(), df['document_count'].max(), 500)
y_fit_poly = polynomial(x_fit)

# Linear Reference Line
p1 = df[df['document_count'] == 20].iloc[0]
p2 = df[df['document_count'] == 80].iloc[0]

x1, y1 = p1['document_count'], p1['total_indexing_time_s']
x2, y2 = p2['document_count'], p2['total_indexing_time_s']

slope = (y2 - y1) / (x2 - x1)
intercept = y1 - (slope * x1)
linear_func = lambda x: slope * x + intercept
y_fit_linear = linear_func(x_fit)

plt.figure(figsize=(10, 6))

# 1. Actual Data
plt.plot(df['document_count'], df['total_indexing_time_s'], marker='o', linestyle='-', color='tab:blue', alpha=0.6, label='Actual Data')

# 2. Quadratic Fit
poly_label = f'Quadratic Fit: $y={coeffs_poly[0]:.4f}x^2 + {coeffs_poly[1]:.2f}x + {coeffs_poly[2]:.0f}$'
plt.plot(x_fit, y_fit_poly, color='darkorange', linestyle='--', linewidth=2, label=poly_label)

# 3. Linear Projection
linear_label = f'Linear Projection (based on 20-80 docs)'
plt.plot(x_fit, y_fit_linear, color='green', linestyle=':', linewidth=2, label=linear_label)

plt.title('Total Indexing Time vs Document Count')
plt.xlabel('Document Count')
plt.ylabel('Total Indexing Time (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'indexing_time_vs_document_count.png'))
plt.show()
