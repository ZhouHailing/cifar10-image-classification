import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

#1. 设备 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

#2. 超参数 
batch_size = 64
learning_rate = 0.001
num_epochs = 20

#3. 数据增强 + 预处理 
# 训练集：带数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # 随机水平翻转
    transforms.RandomCrop(32, padding=4),        # 随机裁剪（32x32，补4圈0）
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 测试集：只做标准化，不做增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

#4. 加载数据 
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别: {train_dataset.classes}")

#5. 模型（带 BatchNorm） 
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # 卷积块 1：3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 卷积块 2：32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 卷积块 3：64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # 32 → 16
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # 16 → 8
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # 8 → 4

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CIFAR10CNN().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

#6. 损失函数、优化器、学习率调度 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 当验证准确率连续 3 个 epoch 不提升时，学习率乘以 0.5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
writer = SummaryWriter('runs/cifar10_enhanced')     # TensorBoard 初始化 

#7. 训练
print("\n开始训练...")
train_start_time = time.time()

best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] 完成, 训练准确率: {train_accuracy:.2f}%')

    #每个 epoch 结束后在测试集上评估
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f'测试集准确率: {test_acc:.2f}%')
#TensorBoard 记录
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)   # 记录训练 Loss（平均）
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)                  # 记录训练准确率
    writer.add_scalar('Accuracy/test', test_acc, epoch)                         # 记录测试准确率
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)  # 记录学习率变化

    # 学习率调度：根据测试准确率决定是否降低学习率
    scheduler.step(test_acc)

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'cifar10_best.pth')
        print(f'最优模型已保存 (准确率: {best_acc:.2f}%)')

train_end_time = time.time()
print(f"\n训练完成! 总耗时: {train_end_time - train_start_time:.2f} 秒")
print(f"最优测试准确率: {best_acc:.2f}%")
writer.close() #关闭记录器

#8. 加载最优模型，最终测试
print("\n加载最优模型进行最终测试...")
model.load_state_dict(torch.load('cifar10_best.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total
print(f'最终测试集准确率: {final_acc:.2f}%')

#9. 可视化
print("\n显示一个预测示例...")
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

classes = train_dataset.classes

plt.figure(figsize=(4, 4))
plt.imshow(images[0].cpu().permute(1, 2, 0))
plt.title(f'True: {classes[labels[0].item()]}, Pred: {classes[predicted[0].item()]}')
plt.axis('off')
plt.show()

print("所有步骤完成")