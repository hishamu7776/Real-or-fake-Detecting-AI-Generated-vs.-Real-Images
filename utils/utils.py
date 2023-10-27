
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def plot_line(hist_df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    sns.lineplot(data=hist_df[['train_loss', 'validation_loss']], ax=ax[0])
    sns.lineplot(data=hist_df[['train_acc', 'validation_acc']], ax=ax[1])
    ax[0].set_title('Model loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].set_title('Model accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    plt.show()

def plot_metrics(y_true, y_pred):
    """
    
    Plots f1_score, precision, accuracy, and recall.
    Args:
        y_true: The ground truth labels.
        y_pred_prob: The predicted probabilities.
    Returns:
        None.
        
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = [f1, precision, recall, accuracy]
    labels = ["f1_score", "precision", "recall", "accuracy"]
    
    sns.barplot(x=labels, y=metrics)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred):
    """Plots confusion matrix using SNS.
    
    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        
    Returns:
    None.
    
    """
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='.1f', ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()