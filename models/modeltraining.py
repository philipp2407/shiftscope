from shiftscope.helperfunctions import Colors
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# function to train resnet (with 4 classes)
def trainResnetSimple(training_dataloader, testset_dataloader, save_model_path):
    all_labels = np.array([])
    all_predictions = np.array([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)  # Load Pretrained ResNet50 Model
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Modify the first convolution layer to accommodate the number of channels
    blocks_to_unfreeze = ['layer3','layer4', 'fc']
    print(f"Unfreezing layers in blocks {blocks_to_unfreeze}")
    for name, param in model.named_parameters():
        if any(layer in name for layer in blocks_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False
    num_features = model.fc.in_features # Modify the Final Layer for 4 Classes
    model.fc = nn.Linear(num_features, 4)  # 4 classes
    model = model.to(device) # Move model to the appropriate device
    criterion = nn.CrossEntropyLoss() # 4. Create Loss Function and Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # 5. Training Loop
    for epoch in range(10):
        model.train()
        train_loss = 0
        # Wrap your dataloader with tqdm for a progress bar
        train_loader = tqdm(training_dataloader, desc=f'Epoch {epoch+1}/{10} [Train]',leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader.set_postfix(loss=train_loss/len(train_loader))
        avg_train_loss = train_loss / len(train_loader)
    torch.save(model.state_dict(), save_model_path)
    # evaluate on test set
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testset_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # Accumulate labels and predictions
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
            all_predictions = np.concatenate((all_predictions, predicted.cpu().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    return test_accuracy


# function to train resnet (with 5 classes - including DEBRIS)
def trainResnetSimpleDEBRIS(training_dataloader, testset_dataloader, save_model_path):
    all_labels = np.array([])
    all_predictions = np.array([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)  # Load Pretrained ResNet50 Model
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Modify the first convolution layer to accommodate the number of channels
    blocks_to_unfreeze = ['layer3','layer4', 'fc']
    print(f"Unfreezing layers in blocks {blocks_to_unfreeze}")
    for name, param in model.named_parameters():
        if any(layer in name for layer in blocks_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False
    num_features = model.fc.in_features # Modify the Final Layer for 4 Classes
    model.fc = nn.Linear(num_features, 5)  # 4 classes
    model = model.to(device) # Move model to the appropriate device
    criterion = nn.CrossEntropyLoss() # 4. Create Loss Function and Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # 5. Training Loop
    for epoch in range(10):
        model.train()
        train_loss = 0
        # Wrap your dataloader with tqdm for a progress bar
        train_loader = tqdm(training_dataloader, desc=f'Epoch {epoch+1}/{10} [Train]',leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader.set_postfix(loss=train_loss/len(train_loader))
        avg_train_loss = train_loss / len(train_loader)
    torch.save(model.state_dict(), save_model_path)
    # evaluate on test set
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testset_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # Accumulate labels and predictions
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
            all_predictions = np.concatenate((all_predictions, predicted.cpu().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    return test_accuracy




# function to train resnet in DDM mode (Domain Discriminative Model - distinguish between two datasets)
def trainResnet_DDM_MODE(training_dataloader, val_dataloader, num_classes=2, num_epochs=10, num_channels=3):
    """
        This function trains Resnet in DDM mode to distinguish between two datasets
    """
    all_labels = np.array([])
    all_predictions = np.array([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Pretrained ResNet50 Model
    model = models.resnet50(pretrained=True)
    # Modify the first convolution layer to accommodate the number of channels
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    blocks_to_unfreeze = ['layer3','layer4', 'fc']
    print(f"Unfreezing layers in blocks {blocks_to_unfreeze}")
    for name, param in model.named_parameters():
        if any(layer in name for layer in blocks_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False
    # 3. Modify the Final Layer for 4 Classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes) 
    # Move model to the appropriate device
    model = model.to(device)
    # 4. Create Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        # Wrap dataloader with tqdm for a progress bar
        train_loader = tqdm(training_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]',leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader.set_postfix(loss=train_loss/len(train_loader))
        avg_train_loss = train_loss / len(train_loader)
    print("DONE with training - now calculating test accuracy")
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # Accumulate labels and predictions
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
            all_predictions = np.concatenate((all_predictions, predicted.cpu().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    # return the test accuracy of the last epoch in percent
    return test_accuracy


# OUTSOURCED STUFF
#____________________________________________________________________________________________________________________________________________________________________
# function to train resnet
def trainResnet(training_dataloader, val_dataloader, num_epochs, path_to_save_model_to, filename_for_saved_model, path_to_save_confusion_matrix_to, filename_for_saved_confusion_matrix, num_channels=3):
    # Initialize arrays to store true labels and predictions
    print(f"Using {num_channels} channels for the training of ResNET")
    print(f"{Colors.BLUE}Starting training of ResNET for {num_epochs} epochs{Colors.RESET}")
    all_labels = np.array([])
    all_predictions = np.array([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load Pretrained ResNet50 Model
    model = models.resnet50(pretrained=True)
    # Modify the first convolution layer to accommodate the number of channels
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    blocks_to_unfreeze = ['layer3','layer4', 'fc']
    print(f"Unfreezing layers in blocks {blocks_to_unfreeze}")
    for name, param in model.named_parameters():
        if any(layer in name for layer in blocks_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False
    # 3. Modify the Final Layer for 4 Classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # Adjust for 4 classes
    # Move model to the appropriate device
    model = model.to(device)
    # 4. Create Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    val_accuracies =[]
    print(f"Using Adam optimizer with learning rate 0.001")
    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        # Wrap your dataloader with tqdm for a progress bar
        train_loader = tqdm(training_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]',leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader.set_postfix(loss=train_loss/len(train_loader))
        avg_train_loss = train_loss / len(train_loader)
        # 6. Validation Loop
        model.eval()
        val_loss = 0
        total = 0
        correct = 0
        # Similarly wrap your validation loader
        val_loader = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]',leave=False)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # Accumulate labels and predictions
                all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
                all_predictions = np.concatenate((all_predictions, predicted.cpu().numpy()))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loader.set_postfix(loss=val_loss/len(val_loader))

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_accuracies.append([epoch+1,val_accuracy])
    print(val_accuracies)
    print("DONE")
    # Save the trained model
    new_path = os.path.join(path_to_save_model_to, filename_for_saved_model)
    torch.save(model.state_dict(), new_path)
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions += predicted.tolist()
            val_labels += labels.tolist()

    # Plot confusion matrix
    classes = ['lymphocytes', 'monocytes', 'eosinophils', 'neutrophils']

    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figsize if needed
    cm = confusion_matrix(val_labels, val_predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes).plot(values_format='d', ax=ax)
    ax.set_xlabel('Predicted label', labelpad=10, fontsize=14)  # Adjust label padding if needed
    ax.set_ylabel('True label', labelpad=10, fontsize=16)       # Adjust label padding if needed
    # Increase the font size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45)  # Rotate x labels if needed
    plt.yticks(rotation=45)  # Rotate y labels if needed
    plt.savefig(f'{path_to_save_confusion_matrix_to}/{filename_for_saved_confusion_matrix}.pdf', bbox_inches='tight')
    plt.close(fig)

    # return the validation accuracy of the last epoch in percent
    return val_accuracies[-1][1]*100