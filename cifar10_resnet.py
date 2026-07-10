import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models  
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

#1. 设备 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

#2. 超参数 
batch_size = 64
learning_rate = 0.001
num_epochs = 10

#3. 数据增强 + 预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

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

#5. 模型（ResNet-18 预训练）
# 加载在 ImageNet 上预训练好的 ResNet-18
model = models.resnet18(pretrained=True)

# 因为 CIFAR-10 只有 10 类，替换最后一层
model.fc = nn.Linear(512, 10)

model = model.to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

#6. 损失函数、优化器、调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

#7. TensorBoard 
writer = SummaryWriter('runs/cifar10_resnet')

#8. 训练
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

    # 测试集评估
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

    # TensorBoard 记录
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    scheduler.step(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'cifar10_resnet_best.pth')
        print(f'最优模型已保存 (准确率: {best_acc:.2f}%)')

train_end_time = time.time()
print(f"\n训练完成! 总耗时: {train_end_time - train_start_time:.2f} 秒")
print(f"最优测试准确率: {best_acc:.2f}%")

writer.close()

#9. 加载最优模型，最终测试 
print("\n加载最优模型进行最终测试...")
model.load_state_dict(torch.load('cifar10_resnet_best.pth', weights_only=False))
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

#10. 可视化 
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

print("所有步骤完成!")