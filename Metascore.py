import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Loads the dataset and reads the CSV file into a pandas
df = pd.read_csv("imdb_top_1000.csv")

# Data cleaning
# Convert Runtime to numeric (remove ' min')
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(int)

# Convert Gross to numeric (remove commas and handle missing values)
df['Gross'] = df['Gross'].str.replace(',', '').astype(float)

# rows with missing values in relevant columns
df = df.dropna(subset=['Runtime', 'IMDB_Rating', 'Gross'])

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Create a figure with multiple subplots for different visualizations
# (1 row, 2 columns) = 2 plots total
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Runtime vs IMDB Rating (Scatter Plot)
# Do longer movies have higher ratings?
axes[0].scatter(df['Runtime'], df['IMDB_Rating'], alpha=0.6, color='blue')
axes[0].set_xlabel('Runtime (minutes)')
axes[0].set_ylabel('IMDB Rating')
axes[0].set_title('Runtime vs IMDB Rating')
axes[0].grid(True)


# Add linear line correlation for Runtime vs IMDB_Rating
corr_rating = df['Runtime'].corr(df['IMDB_Rating'])
m, b = np.polyfit(df['Runtime'], df['IMDB_Rating'], 1)
xs = np.linspace(df['Runtime'].min(), df['Runtime'].max(), 100)
axes[0].plot(xs, m * xs + b, color='red', linewidth=1.5, label=f'fit (r={corr_rating:.2f})')
axes[0].legend(loc='best')
axes[0].text(0.02, 0.95, f"r = {corr_rating:.2f}", transform=axes[0].transAxes,
             fontsize=10, verticalalignment='top')





# Plot 2: Runtime vs Gross Revenue (Scatter Plot)
# Do longer movies generate more revenue?
axes[1].scatter(df['Runtime'], df['Gross'], alpha=0.6, color='green')
axes[1].set_xlabel('Runtime (minutes)')
axes[1].set_ylabel('Gross Revenue ($)')
axes[1].set_title('Runtime vs Gross Revenue')
axes[1].grid(True)

# Add linear line correlation for Runtime vs Gross
corr_gross = df['Runtime'].corr(df['Gross'])
m2, b2 = np.polyfit(df['Runtime'], df['Gross'], 1)
xs2 = np.linspace(df['Runtime'].min(), df['Runtime'].max(), 100)
axes[1].plot(xs2, m2 * xs2 + b2, color='red', linewidth=1.5, label=f'fit (r={corr_gross:.2f})')
axes[1].legend(loc='best')
axes[1].text(0.02, 0.95, f"r = {corr_gross:.2f}", transform=axes[1].transAxes,
             fontsize=10, verticalalignment='top')

# Changes layout to prevent overlapping and show all plots
plt.tight_layout()
plt.show()