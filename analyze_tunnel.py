import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Updated DataFrame with the provided data
data = {
    'model_type': ['XGBoost', 'CNN_BiGRU_Attention', 'XGBoost', 'CNN_BiGRU_Attention', 'XGBoost', 'CNN_BiGRU_Attention'],
    'dataset': ['DoH-Tunnel', 'DoH-Tunnel', 'DoH-DGA', 'DoH-DGA', 'CIRA-CIC', 'CIRA-CIC'],
    'train_accuracy': [1, 0.999622005, 0.999881288, 0.952041876, 0.979425503, 0.936882599],
    'val_accuracy': [1, 0.999619627, 0.990977535, 0.959872698, 0.970776824, 0.945900498],
    'train_loss': [1.54E-05, 0.008298656, 0.00110392, 0.165724178, 0.050337104, 0.156166394],
    'val_loss': [1.55E-05, 0.005793395, 0.035489603, 0.107393979, 0.063502154, 0.132734017],
    'aggregated_confusion_matrix': [
        [[46080, 0, 0],
         [0, 30040, 0], 
         [0, 0, 29040]],
        [[46080, 0, 0], 
         [0, 30040, 0], 
         [40, 0, 29000]],
        [[832, 4, 0, 4], 
         [4, 733, 2, 5], 
         [2, 7, 1799, 0], 
         [3, 7, 0, 810]],
        [[830, 1, 0, 9], 
         [6, 666, 1, 71], 
         [1, 0, 1797, 10], 
         [1, 69, 0, 750]],
        [[165440, 1046, 1000], 
         [590, 32895, 2285], 
         [447, 1933, 44200]],
        [[163518, 1494, 2474], 
         [546, 31108, 4116], 
         [743, 4143, 41694]]
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
    
    # Normalize the confusion matrix by row, so that each row sums to 100
    matrix_percentage = matrix / matrix.sum(axis=1, keepdims=True) * 100
    
    dataset_labels = labels[row['dataset']]  # Get labels for the dataset
    
    # Create a figure for each model and dataset combination
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                     xticklabels=dataset_labels, yticklabels=dataset_labels)

    # Manually append the '%' symbol to each annotation
    for text in ax.texts:
        text.set_text(f'{text.get_text()}%')

    plt.title(f'Confusion Matrix (Percentage): {row["model_type"]} ({row["dataset"]})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f'confusion_matrix_percentage_{row["model_type"]}_{row["dataset"]}.jpg')
    plt.close()  # Close the figure to free up memory

print("Confusion Matrix plots with percentages and labels saved successfully!")
