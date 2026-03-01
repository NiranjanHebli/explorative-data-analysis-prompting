import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the visual style for the report
sns.set_theme(style="whitegrid", palette="muted")

def perform_titanic_eda(file_path):
    # 1. Load the Dataset
    try:
        df = sns.load_dataset('titanic') # Using seaborn's built-in copy for ease
        # If using a local CSV, use: df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Data Overview ---
    print("## Dataset Overview")
    print(f"Shape: {df.shape}")
    print("\n## Column Info & Data Types")
    print(df.info())
    
    # --- Missing Values ---
    print("\n## Missing Values Count")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0])

    # --- Summary Statistics ---
    print("\n## Numerical Summary")
    print(df.describe().T)
    
    print("\n## Categorical Summary")
    print(df.describe(include=['object', 'category']).T)

    # --- Visualizations ---
    # Create a figure with a layout for multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.4)

    # 1. Overall Survival Counts
    sns.countplot(data=df, x='survived', ax=axes[0, 0], hue='survived', legend=False)
    axes[0, 0].set_title('Overall Survival (0 = No, 1 = Yes)')
    axes[0, 0].set_xlabel('Survived')

    # 2. Survival by Gender
    sns.countplot(data=df, x='sex', hue='survived', ax=axes[0, 1])
    axes[0, 1].set_title('Survival by Gender')
    axes[0, 1].set_xlabel('Sex')

    # 3. Survival by Passenger Class
    sns.countplot(data=df, x='pclass', hue='survived', ax=axes[1, 0])
    axes[1, 0].set_title('Survival by Passenger Class')
    axes[1, 0].set_xlabel('Class (1 = 1st, 2 = 2nd, 3 = 3rd)')

    # 4. Age Distribution
    sns.histplot(data=df, x='age', kde=True, ax=axes[1, 1], color='teal')
    axes[1, 1].set_title('Age Distribution of Passengers')
    axes[1, 1].set_xlabel('Age')

    plt.show()

    # --- Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

if __name__ == "__main__":
    # You can replace 'titanic' with your local CSV path
    perform_titanic_eda('Titanic-Dataset.csv')