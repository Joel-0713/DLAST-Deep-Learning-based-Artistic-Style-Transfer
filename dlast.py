import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
def load_image(img_path, max_size=256):
    image = Image.open(img_path).convert('RGB')
    size = max(image.size)
    if size > max_size:
        size = max_size
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().squeeze(0)
    image = torch.clamp(image, 0, 1)
    image = transforms.ToPILImage()(image)
    return image

# Define losses
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(c, h * w)
        G = torch.mm(features, features.t())
        return G / (c * h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Build model
def get_model_and_losses(cnn, style_img, content_img):
    cnn = cnn.features.to(device).eval()

    content_layer = 'conv_4'
    style_layers = ['conv_1', 'conv_2', 'conv_3']

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name == content_layer:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# Style transfer
def run_style_transfer(cnn, content_img, style_img, input_img,
                       num_steps=100, style_weight=1e5, content_weight=1):
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.02)

    print("Optimizing...")
    for step in range(num_steps):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

    input_img.data.clamp_(0, 1)
    return input_img

# Paths to images
content_path = '/Users/sreejareddy/Desktop/Screenshot 2026-03-08 at 6.05.11 PM.png'
style_path = '/Users/sreejareddy/Desktop/Screenshot 2026-03-08 at 6.04.57 PM.png'

content = load_image(content_path)
style = load_image(style_path)
input_img = content.clone()

cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device).eval()
output = run_style_transfer(cnn, content, style, input_img)

# Show and save output
result = im_convert(output)
plt.imshow(result)
plt.axis('off')
plt.show()
result.save("fast_stylized_output.jpg")