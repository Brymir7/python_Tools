

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # makes feature maps smol
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
# Define the
# Load the trained model
model = Net()
model.load_state_dict(torch.load("2d_handwritten_model.pt"))
model.eval()

# Load the single image
image = Image.open("/home/brymir/PycharmProjects/image_classifier/data/mnist1.png")

# Define the preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Grayscale(num_output_channels=1)

])

# Preprocess the image
input_tensor = preprocess(image)
input_tensor = input_tensor.unsqueeze(0)

# Use the trained model to make a prediction
output_tensor = model(input_tensor)
predicted_class = torch.argmax(output_tensor, dim=1)

# Print the predicted class label
print("Predicted class:", predicted_class.item())
