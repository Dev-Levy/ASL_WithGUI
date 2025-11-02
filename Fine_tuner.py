import os
import argparse
from datetime import datetime

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader

from utils.Transforms import get_train_transforms
from utils.Color_print import prGreen, prYellow
from Models.DNN_model import ASL_DNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 28
input_dim = 64 * 64
batch_size = 128

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, help="The absolute path of the weights file")
    parser.add_argument("--my_imgs", type=str, help="The absolute path of the directory of my images")
    args = parser.parse_args()

    #region preparation
    post_train_dataset = datasets.ImageFolder(root=args.my_imgs, transform=get_train_transforms())
    post_train_loader = DataLoader(post_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ASL_DNN(input_dim, num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    for param in model.parameters():
        param.requires_grad = False

    model.fc4 = nn.Linear(model.fc3.out_features, len(post_train_dataset.classes))
    for param in model.fc4.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    post_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    model.to(device)
    print(f"\nDevice: {device}\n")
    prYellow("TRAINED MODEL LOADED\n")
    #endregion

    #region post-training
    prYellow("POST-TRAINING\n")

    post_epochs = 5
    for epoch in range(post_epochs):
        model.train()
        running_loss = 0
        all_preds, all_labels = [], []

        print(f"Training model - Epoch {epoch + 1}/{post_epochs} ({datetime.now()})")
        for images, labels in post_train_loader:
            images, labels = images.to(device), labels.to(device)

            post_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            post_optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        prGreen(f"Post-Training Epoch {epoch + 1}/{post_epochs} | "
                f"Loss: {running_loss / len(post_train_loader):.4f} | "
                f"Acc: {train_acc:.2f}% |"
                f"F1: {train_f1:.2f}%")
    #endregion

    post_weights = os.path.join(os.path.dirname(args.weights), "post_trained_weights.pth")
    torch.save(model.state_dict(), post_weights)
    prGreen(f"Post-trained model saved to {post_weights}\n")
