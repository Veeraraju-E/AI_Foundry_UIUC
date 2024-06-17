# MNIST using FCN
# 1. It is always the packages
import torch
from torch import nn, optim
# import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ResNet101
from dataset import ImageFolder
import cv2
import os

# 2. Some global variables, like device, num_epochs, lr, all the hyperparameters and other task-specific variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 10e-3
NUM_EPOCHS = 2
BATCH_SIZE = 32
IN_DIM = 3
NUM_CLASSES = 4
MOMENTUM = 10e-4
WT_DECAY = 0.9

# 3. Load the data
transform = A.Compose(
    [
        A.Resize(width=624, height=624),
        A.RandomCrop(width=592, height=592),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        # A.OneOf(
        #     [
        #         A.Blur(blur_limit=3, p=0.5),
        #         A.ColorJitter(p=0.5),
        #     ],
        #     p=1.0,
        # ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)

train_dataset = ImageFolder(root_dir=os.path.join(os.getcwd(), "train"), transform=transform)
train_dataLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ImageFolder(root_dir=os.path.join(os.getcwd(), "test"), transform=transform)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Load the model, from model.py
model = ResNet101(img_channel=IN_DIM, num_classes=NUM_CLASSES).to(DEVICE)

# 5. Initialize the loss function and optimizers
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WT_DECAY)


# 6. Training Loop
# train_loader gives us the x, viz the features and y, viz the label; quite common to enumerate thru the train_loader
def train(loader, model):
    for epoch in range(NUM_EPOCHS):
        # print(epoch)
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(train_dataLoader)):
            # print(data.shape)
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            # data = data.reshape(data.shape[0], -1)
            # forward pass
            y_pred = model(data)
            loss = loss_fn(y_pred, target)
            print(loss.item())
            # back-prop, using optimizer
            optimizer.zero_grad()  # intialize the grads to 0 for every batch, to prevent memory of previous gradients
            loss.backward()

            # gradient descent, viz the Adam
            optimizer.step()
    print('\nTraining is Done!!')

# 7. Evaluate
def test(loader, model):
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            y_predictions = model(data)
            _, y_pred = y_predictions.max(1)  # along axis=1
            num_correct += (y_pred == target).sum()
            num_samples += data.shape[0]
    print(f'accuracy is {num_correct}/{num_samples} or {float(num_correct)/float(num_samples):.2f}')


if __name__ == '__main__':
    train(train_dataLoader, model)
    test(train_dataLoader, model)
    test(test_dataLoader, model)
