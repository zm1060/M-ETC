import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample DataFrame (replace with your actual data)
data = {
    'model_type': ['XGBoost', 'CNN_BiGRU_Attention', 'XGBoost', 'CNN_BiGRU_Attention', 'XGBoost', 'CNN_BiGRU_Attention'],
    'dataset': ['DoH-DGA', 'DoH-DGA', 'CIRA-CIC', 'CIRA-CIC', 'DoH-Tunnel', 'DoH-Tunnel'],
    'train_accuracy': [0.999881288, 0.966761064, 0.979425503, 0.934303058, 1, 0.999622597],
    'val_accuracy': [0.990977535, 0.973293769, 0.970776824, 0.943052984, 1, 0.999595864],
    'train_loss': [0.00110392, 0.116743205, 0.050337104, 0.161254821, 1.54E-05, 0.008476631],
    'val_loss': [0.035489603, 0.082961632, 0.063502154, 0.135964155, 1.55E-05, 0.005831033],
    'aggregated_confusion_matrix': [
        [[832, 4, 0, 4], [4, 733, 2, 5], [2, 7, 1799, 0], [3, 7, 0, 810]],
        [[669, 1, 0, 5], [0, 539, 3, 53], [1, 0, 1436, 8], [0, 19, 0, 636]],
        [[165440, 1046, 1000], [590, 32895, 2285], [447, 1933, 44200]],
        [[130578, 1206, 2206], [349, 24166, 4100], [490, 3031, 33744]],
        [[46080, 0, 0], [0, 30040, 0], [0, 0, 29040]],
        [[36865, 0, 0], [0, 24035, 0], [34, 0, 23196]]
    ]
}

df = pd.DataFrame(data)

# Labels for each dataset
labels = {
    'DoH-DGA': ['padcrypt', 'tinba', 'sisron', 'zloader'],
    'CIRA-CIC': ['dns2tcp', 'iodine', 'dnscat2'],
    'DoH-Tunnel': ['dnstt', 'tcp-over-dns', 'tuns']
}

# Plot Confusion Matrices with Percentages and Labels
for i, row in df.iterrows():
    matrix = np.array(row['aggregated_confusion_matrix'])
    matrix_percentage = (matrix.T / matrix.sum(axis=1)).T * 100  # Normalize by row sums
    dataset_labels = labels[row['dataset']]  # Get labels for the dataset
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                     xticklabels=dataset_labels, yticklabels=dataset_labels)

    # Manually append the '%' symbol to each annotation
    for text in ax.texts:
        text.set_text(f'{text.get_text()}%')

    # plt.title(f'Confusion Matrix (Percentage): {row["model_type"]} ({row["dataset"]})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_percentage_{row["model_type"]}_{row["dataset"]}.jpg')

print("Confusion Matrix plots with percentages and labels saved successfully!")
