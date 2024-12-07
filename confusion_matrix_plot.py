import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a nice style and larger font sizes for better readability
sns.set_theme(style="whitegrid", font_scale=1.4)

confusion_matrices = {
    "ViT_Casia": [
        [1025,  100],
        [  178, 810]
    ],
    "ViT_FaceForensics": [
        [1476,   24],
        [ 73,   1427]
    ],
    "Xception_Casia": [
        [833, 279],
        [209, 464]
    ],
    "Xception_FaceForensics": [
        [794, 206],
        [147, 853]
    ]
}

# Define class names (optional)
classes = ["Authentic", "Tampered"]

# Create and save each confusion matrix plot
for model_name, cm in confusion_matrices.items():
    cm_array = np.array(cm)

    # Create a new figure with high DPI for better quality
    plt.figure(figsize=(6, 5), dpi=300)

    # Plot the confusion matrix as a heatmap
    ax = sns.heatmap(cm_array,
                     annot=True,        # Show the numbers in cells
                     fmt='d',           # Format as integers
                     cmap='Blues',       # Color map
                     cbar=True,         # Add colorbar
                     xticklabels=classes,
                     yticklabels=classes,
                     square=True,        # Make cells square
                     linewidths=.5,      # Separate cells with a line
                     linecolor='gray')

    # Add title and labels
    plt.title(f"{model_name}", fontsize=16, pad=12)
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the figure locally with the model name in the filename
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=300)
    plt.close()  # Close the figure to free up memory
