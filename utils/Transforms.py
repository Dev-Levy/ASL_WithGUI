from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
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
def get_test_transforms():
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])