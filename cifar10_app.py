import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
import torchvision.models as models

#1. 定义模型结构
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#2. 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # 不需要再下载预训练权重
model.fc = nn.Linear(512, 10)
model = model.to(device)
model.load_state_dict(torch.load('cifar10_resnet_best.pth', map_location=device, weights_only=False))
model.eval()
print("模型加载成功")

#3. 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10 类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

#4. 预测函数（Gradio 会调用它）
def predict_image(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences = probabilities[0].cpu().numpy()
            predicted_class = confidences.argmax()
        
        # 这里返回两个值
        return classes[predicted_class], {classes[i]: float(confidences[i]) for i in range(len(classes))}
    except Exception as e:
        return "错误", {"error": str(e)}

#5. 创建 Gradio 界面
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="上传图片"),
    outputs=[
        gr.Label(label="识别结果"),          # 显示最可能的类别
        gr.JSON(label="所有类别置信度")      # 显示所有类别的概率
    ],
    title="ResNet-18图像分类器",
    description="上传一张 CIFAR-10 类别相关的图片（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车），模型将返回识别结果和置信度。",
    article="模型在 CIFAR-10 测试集上的准确率为 83.36%。",
    examples=[['test.jpg']]
)

# 6. 启动 Web 服务
if __name__ == "__main__":
    iface.launch(share=True)