import os
from datetime import datetime

import kagglehub
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from ASL_DNN import ASL_DNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_dir = "C:\\Users\\horga\\Documents\\GitHub\\ASL_WithGUI\\weights"
weights_path = os.path.join(weights_dir, "5_asd.pth")
batch_size = 128
epochs = 30
#link_to_dataset = "debashishsau/aslamerican-sign-language-aplhabet-dataset" #this needs "ASL_Alphabet_Dataset", "asl_alphabet_train" to be joined
link_to_dataset = "kapillondhe/american-sign-language" # this needs "ASL_Dataset", "Train" or "Test" to be joined

def prRed(s): print("\033[91m{}\033[00m".format(s))
def prGreen(s): print("\033[92m{}\033[00m".format(s))
def prYellow(s): print("\033[93m{}\033[00m".format(s))

def train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler):
    global device, weights_path
    best_val_acc = 0
    patience = 5
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        all_preds = []
        all_labels = []

        print(f"Training model - Epoch {epoch+1}/{epochs} ({datetime.now()})")
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Validating model - Epoch {epoch+1}/{epochs} ({datetime.now()})")
        val_acc, val_f1 = evaluate(model, val_loader)

        scheduler.step(val_acc)

        prGreen(f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {running_loss / len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} | "
                f"Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), weights_path)
        else:
            trigger_times += 1
            prRed(f"No improvement for {trigger_times} epoch(s)")

        if trigger_times >= patience:
            prRed("Early stopping triggered")
            break

        print("-------------------------------------------------------------------------------------------------------")

def evaluate(model, loader):
    global device
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1

if __name__ == "__main__":
    #region preparation
    prYellow("PREPARATIONS\n")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),                                            #grayscale
        transforms.Resize((64, 64)),                                                            #64 x 64
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7),                   #brightness
        transforms.RandomRotation(20),                                                          #rotation
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),   #zoom in/out
        transforms.RandomHorizontalFlip(p=0.5),                                                 #flip
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),                                 #blur
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),                                     #pixel erase
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ds_path = kagglehub.dataset_download(link_to_dataset)
    train_val_ds_path = os.path.join(ds_path, "ASL_Dataset", "Train")
    test_ds_path = os.path.join(ds_path, "ASL_Dataset", "Test")
    print(f"Dataset path: {ds_path}")
    print(f"Dataset train dir path: {train_val_ds_path}")
    print(f"Dataset test dir path: {test_ds_path}\n")

    train_val_dataset = datasets.ImageFolder(root=train_val_ds_path, transform=None)
    test_dataset = datasets.ImageFolder(root=test_ds_path, transform=val_test_transform)
    print(f"Dataset classes: {train_val_dataset.classes}")

    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_val_dataset.classes)
    input_dim = 64*64
    #endregion

    #region training
    model = ASL_DNN(input_dim, num_classes)
    model.to(device)
    print(f"Device: {device}\n")
    prYellow("TRAINING\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3)

    start = datetime.now()
    print(f"Starting training ({start})")
    #train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler)
    end = datetime.now()
    print(f"Finished training ({end})")
    print(f"Total training time: {end - start}\n")
    #endregion

    #region testing
    prYellow("TESTING\n")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_acc, test_f1 = evaluate(model, test_loader)
    print(f"Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")
    #endregion