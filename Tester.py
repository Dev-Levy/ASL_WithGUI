import argparse
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.Transforms import get_test_transforms
from utils.Color_print import prGreen, prYellow
from Models.DNN_model import ASL_DNN
from Models.CNN_model import ASL_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 64 * 64

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="DNN or CNN")
    parser.add_argument("--test", type=str, help="Directory of testing data")
    parser.add_argument("--weights", type=str, help="The absolute path of the weights file")
    args = parser.parse_args()

    test_dataset = datasets.ImageFolder(root=args.test, transform=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    num_classes = len(test_dataset.classes)
    if args.model == "DNN":
        model = ASL_DNN(input_dim, num_classes).to(device)
    elif args.model == "CNN":
        model = ASL_CNN(num_classes).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    prYellow("TESTING\n")
    test_acc, test_f1 = evaluate(model, test_loader)
    prGreen(f"Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")