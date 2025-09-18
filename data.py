import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
print("="*50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("="*50)

try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Explore structure
    print("\nDataset information:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Clean dataset (though Iris dataset typically has no missing values)
    # For demonstration, we'll show how to handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()  # or use df.fillna() for specific strategies
        print("Missing values handled!")
    else:
        print("No missing values found in the dataset.")
        
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50)

# Basic statistics
print("Basic statistics for numerical columns:")
print(df.describe())

# Group by species and compute means
print("\nMean values by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis
print("\nAdditional analysis:")
print(f"Number of samples per species:")
print(df['species'].value_counts())

# Find interesting patterns
max_sepal_length = df.loc[df['sepal length (cm)'].idxmax()]
print(f"\nSample with maximum sepal length:")
print(max_sepal_length)

# Correlation analysis
print("\nCorrelation matrix:")
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print(correlation_matrix)

# Task 3: Data Visualization
print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Visualizations', fontsize=16, fontweight='bold')

# 1. Line chart (simulating trends over time by using index as pseudo-time)
axes[0, 0].plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue', alpha=0.7)
axes[0, 0].plot(df.index, df['petal length (cm)'], label='Petal Length', color='red', alpha=0.7)
axes[0, 0].set_title('Trend of Sepal and Petal Length (by sample index)')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart - Average sepal length by species
species_colors = {'setosa': 'lightcoral', 'versicolor': 'lightgreen', 'virginica': 'lightblue'}
avg_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
bars = axes[0, 1].bar(avg_sepal_length.index, avg_sepal_length.values, 
                     color=[species_colors[sp] for sp in avg_sepal_length.index])
axes[0, 1].set_title('Average Sepal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Average Sepal Length (cm)')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')

# 3. Histogram - Distribution of sepal length
axes[1, 0].hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Add mean line
mean_sepal = df['sepal length (cm)'].mean()
axes[1, 0].axvline(mean_sepal, color='red', linestyle='--', 
                  label=f'Mean: {mean_sepal:.2f} cm')
axes[1, 0].legend()

# 4. Scatter plot - Sepal length vs Petal length
scatter = axes[1, 1].scatter(df['sepal length (cm)'], df['petal length (cm)'], 
                            c=df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}), 
                            cmap='viridis', alpha=0.7)
axes[1, 1].set_title('Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')

# Add colorbar for species
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])
cbar.set_label('Species')

# Add correlation coefficient
corr_coef = np.corrcoef(df['sepal length (cm)'], df['petal length (cm)'])[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', 
               transform=axes[1, 1].transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Additional visualizations
print("\nAdditional Visualizations:")

# Box plot for feature distributions by species
plt.figure(figsize=(12, 6))
df_melted = pd.melt(df, id_vars="species", var_name="features", value_name="value")
sns.boxplot(x="features", y="value", hue="species", data=df_melted)
plt.title('Feature Distributions by Species')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pairplot for comprehensive relationships
sns.pairplot(df, hue='species', palette=species_colors)
plt.suptitle('Pairplot of Iris Dataset Features', y=1.02)
plt.show()

# Findings and Observations
print("\n" + "="*50)
print("FINDINGS AND OBSERVATIONS")
print("="*50)

print("""
Key Findings:

1. Dataset Overview:
   - The Iris dataset contains 150 samples with 4 numerical features
   - Three species: setosa, versicolor, and virginica (50 samples each)
   - No missing values detected

2. Statistical Insights:
   - Setosa species has the smallest average measurements
   - Virginica has the largest average measurements
   - Strong positive correlation (0.87) between sepal length and petal length

3. Visual Patterns:
   - Clear separation between setosa and other species in scatter plots
   - Virginica shows the widest range of measurements
   - Features show approximately normal distributions

4. Interesting Observations:
   - Petal measurements show better species separation than sepal measurements
   - Versicolor appears as an intermediate species between setosa and virginica
   - The dataset is well-balanced across all three species

This analysis demonstrates effective use of pandas for data manipulation and 
matplotlib/seaborn for creating informative visualizations that reveal 
underlying patterns in the data.
""")