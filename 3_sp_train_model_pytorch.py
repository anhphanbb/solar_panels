# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:10:28 2024

@author: Anh

Converted to PyTorch: Utilizing GPU for ResNet-50 training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.8, hue=0.1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root='training_images', transform=transform)
print("Loaded")

# Split dataset into training (70%), validation (20%), and test (10%)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the ResNet-50 model with a modified classification head
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50Model, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.base_model(x)

# Initialize model
model = ResNet50Model().to(device)
print("Initialized")

# Define optimizer, loss function, and metrics
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

# Training loop
def train_model(model, train_loader, val_loader, epochs=200):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}/{len(train_loader)}")
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/resnet50_best.pth')
            print("Model saved!")

    return history

# Train the model
history = train_model(model, train_loader, val_loader, epochs=50)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid()
plt.show()

# Evaluate model on test set
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('models/resnet50_best.pth'))
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = evaluate_model(model, test_loader)

# Compute confusion matrix at different thresholds
confusion_matrix_results = []
for threshold in np.arange(0, 1.01, 0.01):
    y_pred_thresholded = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    confusion_matrix_results.append({
        'Threshold': threshold,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'True_Positives': tp,
        'Accuracy': accuracy,
        'Recall': recall
    })

# Save results to CSV
confusion_matrix_df = pd.DataFrame(confusion_matrix_results)
confusion_matrix_df.to_csv('pytorch_confusion_matrix_with_accuracy_recall.csv', index=False)

# Plot accuracy and recall over thresholds
plt.figure(figsize=(10, 5))
plt.plot(confusion_matrix_df['Threshold'], confusion_matrix_df['Accuracy'], label='Accuracy', color='blue')
plt.plot(confusion_matrix_df['Threshold'], confusion_matrix_df['Recall'], label='Recall', color='orange')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Accuracy and Recall Over Thresholds')
plt.grid()
plt.show()

print("Confusion matrix with accuracy and recall saved to pytorch_confusion_matrix_with_accuracy_recall.csv")
