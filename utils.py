import matplotlib.pyplot as plt
from tqdm import tqdm

import torch 
import torch.nn 



def plot_train_samples(train_loader, num_samples=6):
    
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"Shape of images: {images.shape}")
    print(f"Shape of labels: {labels.shape}")


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, num_epochs):
    
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_f1_scores, label='Training F1 Score', color='blue')
    plt.plot(epochs, val_f1_scores, label='Validation F1 Score', color='red')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()


def evaluate_model(model, test_loader, criterion, device):
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad(): 
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  
            loss = criterion(outputs, labels) 

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

           
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.2f}')


