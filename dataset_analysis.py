# ==============================================================================
#                      AUTSL Dataset Analysis Script
# ==============================================================================
# This script is designed to analyze the AUTSL (Turkish Sign Language) dataset.
# It loads the dataset labels, generates various plots to understand the data
# distribution, and saves the plots to a specified directory.
#
# The script assumes the dataset has the following structure:
# - A main directory containing the data.
# - CSV files (e.g., train_labels.csv, val_labels.csv, test_labels.csv) that contain
#   information about the video files, their labels, and the signers.
# ==============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#                           Configuration
# ==============================================================================
# --- Paths ---
# NOTE: Update these paths to match your local dataset structure.
DATASET_PATH = 'autsl'  # Root directory of the AUTSL dataset
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, 'train_labels.csv')
VAL_LABELS_PATH = os.path.join(DATASET_PATH, 'val_labels.csv')
TEST_LABELS_PATH = os.path.join(DATASET_PATH, 'test_labels.csv')
OUTPUT_DIR = 'dataset_analysis'

# --- Plotting Style ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 12
# ==============================================================================

# ------------------------------------------------------------------------------
# --- Create Output Directory ---
# ------------------------------------------------------------------------------
def create_output_directory():
    """
    Creates the directory to save the analysis plots if it doesn't exist.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' created successfully.")

# ------------------------------------------------------------------------------
# --- Load Dataset Labels ---
# ------------------------------------------------------------------------------
def load_dataset_labels():
    """
    Loads the training, validation, and testing labels from CSV files into pandas DataFrames.

    Returns:
        tuple: A tuple containing three pandas DataFrames (train_df, val_df, test_df).
               Returns (None, None, None) if files are not found.
    """
    print("Loading dataset labels...")
    try:
        train_df = pd.read_csv(TRAIN_LABELS_PATH, header=None, names=['filename', 'label'])
        val_df = pd.read_csv(VAL_LABELS_PATH, header=None, names=['filename', 'label'])
        test_df = pd.read_csv(TEST_LABELS_PATH, header=None, names=['filename', 'label'])
        print("Labels loaded successfully.")
        return train_df, val_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the label CSV files are at the correct path.")
        return None, None, None

# ------------------------------------------------------------------------------
# --- Extract Signer ID ---
# ------------------------------------------------------------------------------
def extract_signer_info(df):
    """
    Extracts the signer ID from the filename.
    Assumes filename format is 'signerXX_...'.

    Args:
        df (pd.DataFrame): DataFrame containing the 'filename' column.

    Returns:
        pd.DataFrame: The DataFrame with an added 'signer' column.
    """
    df['signer'] = df['filename'].apply(lambda x: x.split('_')[0])
    return df

# ------------------------------------------------------------------------------
# --- Plot Class Distribution ---
# ------------------------------------------------------------------------------
def plot_class_distribution(df, title='Class Distribution'):
    """
    Analyzes and plots the distribution of classes (signs).

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        title (str): The title for the plot.
    """
    print(f"Analyzing and plotting: {title}")
    plt.figure()
    class_counts = df['label'].value_counts()
    
    # Plot only the top 50 classes for readability
    top_n = 50
    if len(class_counts) > top_n:
        class_counts = class_counts.head(top_n)
        plot_title = f'{title} (Top {top_n} Classes)'
    else:
        plot_title = title

    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', hue=class_counts.index, legend=False)
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Sign Label', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")
    plt.close()

# ------------------------------------------------------------------------------
# --- Plot Signer Distribution ---
# ------------------------------------------------------------------------------
def plot_signer_distribution(df, title='Signer Distribution'):
    """
    Analyzes and plots the contribution of each signer.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        title (str): The title for the plot.
    """
    print(f"Analyzing and plotting: {title}")
    plt.figure()
    signer_counts = df['signer'].value_counts()
    
    sns.barplot(x=signer_counts.index, y=signer_counts.values, palette='plasma', hue=signer_counts.index, legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel('Signer ID', fontsize=12)
    plt.ylabel('Number of Samples Contributed', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")
    plt.close()

# ------------------------------------------------------------------------------
# --- Plot Samples per Class Histogram ---
# ------------------------------------------------------------------------------
def plot_samples_per_class_histogram(df, title='Histogram of Samples per Class'):
    """
    Plots a histogram showing the distribution of the number of samples per class.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        title (str): The title for the plot.
    """
    print(f"Analyzing and plotting: {title}")
    plt.figure()
    class_counts = df.groupby('label').size()
    
    sns.histplot(class_counts, bins=30, kde=True, color='darkcyan')
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Samples in a Class', fontsize=12)
    plt.ylabel('Frequency (Number of Classes)', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")
    plt.close()

# ------------------------------------------------------------------------------
# --- Main Execution Block ---
# ------------------------------------------------------------------------------
def main():
    """
    Main function to run the complete dataset analysis pipeline.
    """
    print("Starting AUTSL Dataset Analysis...")
    
    create_output_directory()
    train_df, val_df, test_df = load_dataset_labels()
    
    if train_df is None or val_df is None or test_df is None:
        print("Analysis stopped due to missing label files.")
        return

    # Extract signer information from filenames
    train_df = extract_signer_info(train_df)
    val_df = extract_signer_info(val_df)
    test_df = extract_signer_info(test_df)
    
    # Combine dataframes for overall analysis
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # --- Generate Plots ---
    plot_class_distribution(train_df, title='Training Set Class Distribution')
    plot_class_distribution(val_df, title='Validation Set Class Distribution')
    plot_class_distribution(test_df, title='Test Set Class Distribution')
    plot_class_distribution(combined_df, title='Overall Class Distribution')
    
    plot_signer_distribution(train_df, title='Training Set Signer Distribution')
    plot_signer_distribution(val_df, title='Validation Set Signer Distribution')
    plot_signer_distribution(test_df, title='Test Set Signer Distribution')
    plot_signer_distribution(combined_df, title='Overall Signer Distribution')

    plot_samples_per_class_histogram(combined_df, title='Overall Histogram of Samples per Class')

    print("\nDataset analysis complete.")
    print(f"All plots have been saved in the '{OUTPUT_DIR}' directory.")
    
    # --- Print Summary Statistics ---
    print("\n--- Dataset Summary ---")
    print(f"Total number of samples: {len(combined_df)}")
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")
    print(f"Number of testing samples: {len(test_df)}")
    print(f"Number of unique classes (signs): {combined_df['label'].nunique()}")
    print(f"Number of unique signers: {combined_df['signer'].nunique()}")
    print("-----------------------\n")


if __name__ == '__main__':
    main()

