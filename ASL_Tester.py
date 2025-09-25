import os

import kagglehub
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from ASL_DNN import ASL_DNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#link_to_dataset = "debashishsau/aslamerican-sign-language-aplhabet-dataset" #this needs "ASL_Alphabet_Dataset", "asl_alphabet_train" to be joined
link_to_dataset = "kapillondhe/american-sign-language" # this needs "ASL_Dataset", "Train" or "Test" to be joined

#if weights will be in the same dir as the script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
weights_path = os.path.join(script_dir, "weights", "5_asd.pth")

#using absolute path
weights_dir = "C:\\Users\\horga\\Documents\\GitHub\\ASL_WithGUI\\weights"
#weights_path = os.path.join(weights_dir, "5_asd.pth")

def prYellow(s): print("\033[93m{}\033[00m".format(s))
def prGreen(s): print("\033[92m{}\033[00m".format(s))

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
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ds_path = kagglehub.dataset_download(link_to_dataset)
    test_ds_path = os.path.join(ds_path, "ASL_Dataset", "Test")
    print(f"Dataset test dir path: {test_ds_path}")

    test_dataset = datasets.ImageFolder(root=test_ds_path, transform=test_transform)
    print(f"Dataset classes: {test_dataset.classes}\n")

    num_classes = len(test_dataset.classes)
    input_dim = 64 * 64

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    model = ASL_DNN(input_dim, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    prYellow("TESTING\n")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_acc, test_f1 = evaluate(model, test_loader)
    prGreen(f"Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")