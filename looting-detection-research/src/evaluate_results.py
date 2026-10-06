import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import pandas as pd
from pathlib import Path

def generate_results(y_true, y_pred, method_name, class_names=['Preserved', 'Looted'], save_dir=None):
    """
    Calculates metrics and plots a confusion matrix for a given set of predictions.
    
    Args:
    y_true: Ground truth labels (e.g., list or numpy array)
    y_pred: Predicted labels by your model
    method_name: Name of the method (for the plot title)
    class_names: List of class names
    save_dir: Optional path to save the resulting image
    """
    
    # 1. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0) 
    
    print(f"\n--- Results for {method_name} ---")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print("-" * 30)
    
    # 2. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # 3. Plot Confusion Matrix
    plt.figure(figsize=(7, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': ''},
                annot_kws={"size": 12})
    
    plt.title(f'Test confusion matrix - {method_name}')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        # Create a clean filename
        clean_name = "".join([c if c.isalnum() else "_" for c in method_name.lower()])
        filename = save_path / f"confusion_matrix_{clean_name}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved REAL confusion matrix plot to: {filename}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # ==========================================
    # Evaluate REAL Results from CSV files
    # ==========================================
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    print("Looking for real test prediction files in your results folder...")
    
    # Find all CSV files that end with 'test_predictions.csv' in all subfolders
    csv_files = list(RESULTS_DIR.rglob("*test_predictions.csv"))
    
    if not csv_files:
        print("\nERROR: No '*test_predictions.csv' files found!")
        print("Please make sure you have successfully run your training scripts (like step6, step7, step9, etc.) first.")
    else:
        for csv_file in csv_files:
            try:
                # Read the real predictions saved by your models
                df = pd.read_csv(csv_file)
                
                # Check if the necessary columns exist
                if "true_label" in df.columns and "predicted_label" in df.columns:
                    y_true = df["true_label"].values
                    y_pred = df["predicted_label"].values
                    
                    # Use the folder or file name to label the chart
                    method_name = csv_file.stem.replace("_test_predictions", "").replace("_", " ").title()
                    
                    # Generate the real result chart!
                    generate_results(y_true=y_true, 
                                     y_pred=y_pred, 
                                     method_name=method_name,
                                     save_dir=RESULTS_DIR)
                else:
                    print(f"Skipping {csv_file.name}: Missing 'true_label' or 'predicted_label' columns.")
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
