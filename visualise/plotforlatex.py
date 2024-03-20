import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def plot_true_vs_predicted_percentages(true_percentages, predicted_percentages, saveFileName=""):
    print("True & predicted percentages has to be in from [lym%, mon%, eos%, neu%]")
    mae = mean_absolute_error(true_percentages, predicted_percentages)
     # Names of the cell types
    categories = ['Lymphocytes', 'Monocytes', 'Eosinophils', 'Neutrophils']

    # Set the figure size and style
    plt.figure(figsize=(14, 8))  # Adjusted figure size for better aspect ratio
    plt.grid(axis='y')  # Just horizontal grid lines
    plt.gca().set_facecolor('white')

    # Create the bar plot
    bar_width = 0.35
    index = range(len(categories))
    plt.bar(index, predicted_percentages, bar_width, label='Predicted', color='#1f77b4')  # Slightly darker blue
    plt.bar([i + bar_width for i in index], true_percentages, bar_width, label='Ground Truth', color='lightgray')

    # Add labels, title, and ticks with adjusted font size
    plt.xlabel('Cell Types', fontsize=24)
    plt.ylabel('Percentage', fontsize=24)
    plt.title(f'Mean Absolute Error = {mae:.2f}', fontsize=24)

    plt.xticks([i + bar_width/2 for i in index], categories, fontsize=24)
    plt.yticks(fontsize=16)
    plt.legend(frameon=True, edgecolor='gray', fontsize=20)  # Adjusted font size and added rectangle

    # Add percentage values on top of each bar with adjusted font size
    for i, val in enumerate(predicted_percentages):
        plt.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=20, color='#1f77b4')
    for i, val in enumerate(true_percentages):
        plt.text(i + bar_width, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=20, color='gray')

    plt.ylim(0, 100)
    # Display or save the plot
    if saveFileName != "":
        plt.savefig(f"{saveFileName}.pdf")
    plt.tight_layout()