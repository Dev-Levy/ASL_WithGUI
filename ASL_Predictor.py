import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from torchvision import transforms

from ASL_DNN import ASL_DNN

img_pil = None
img_path_global = None
wraplength = 450

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#i could read these values by loading the dataset, but it's much quicker if i hardcode them
#dataset_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'] #this was used with 'debashishsau' dataset
dataset_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] #this is used with 'kapillondhe' dataset

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Weight Files", "*.pth")])
    if file_path:
        weight_label.config(text=f"Weight path: {file_path}")

def browse_image():
    global img_pil, img_path_global
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg")])
    if img_path:
        img_label.config(text=f"Image path: {img_path}")
        img_path_global = img_path
        img_pil = Image.open(img_path)

        ax.clear()
        ax.imshow(img_pil)
        ax.axis("off")
        canvas.draw()

def on_submit():
    global img_pil, img_path_global
    file_path = weight_label.cget("text").replace("Weight path: ", "")
    img_path = img_path_global

    if not file_path or not img_path:
        print("Select both file and image first!")
        return

    result = predict_asl_dactyl_sign(file_path, img_path)
    print(f"Predicted class: {result} - ({img_path_global})")

    ax.clear()
    ax.imshow(img_pil)
    ax.axis("off")
    ax.set_title(result, fontsize=16, color="white")
    canvas.draw()

def predict_asl_dactyl_sign(weights_path, img_path):
    global transform, dataset_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(dataset_classes)
    input_dim = 64*64*1

    model = ASL_DNN(input_dim,num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()

    return dataset_classes[pred_class_idx]

if __name__ == "__main__":
    root = tk.Tk()
    root.title("ASL Predictor")
    root.geometry("500x600")
    root.configure(bg="#003049")

    title_label = tk.Label(root,
                           text="American Sign Language\nDactyl Sign Predictor",
                           font=("Arial", 16, "bold"),
                           bg="#003049",
                           fg="white")
    title_label.pack(pady=20)

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#003049")
    ax.text(0.5, 0.5,
            "Prediction Preview Here",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold")
    ax.axis("off")
    ax.set_facecolor("none")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(pady=10)

    weight_label = tk.Label(root, text="Weight path: ",bg="#003049", fg="white", wraplength=wraplength)
    weight_label.pack(pady=5)

    img_label = tk.Label(root, text="Image path: ",bg="#003049", fg="white",wraplength=wraplength)
    img_label.pack(pady=5)

    frame = tk.Frame(root, bg="#003049")
    frame.pack(pady=20)

    weight_button = tk.Button(frame, text="Select weight file", command=browse_file,bg="#f77f00")
    weight_button.pack(side="left", padx=5)

    img_button = tk.Button(frame, text="Browse Image", command=browse_image,bg="#f77f00")
    img_button.pack(side="left", padx=5)

    submit_button = tk.Button(frame, text="Predict", command=on_submit,bg="#f77f00")
    submit_button.pack(side="left", padx=5)

    root.mainloop()
