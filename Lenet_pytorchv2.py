import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import time
print(" 导入成功!!!")

learning_rate = 0.001    # 学习率
batch_size = 64          # 一次训练的样本个数
epoches = 20            # 训练的轮次
print(" 训练超参数设置完毕!!!")


trans_img = transforms.ToTensor()

trainset = MNIST('./data', train=True, transform=trans_img,download=True)
testset = MNIST('./data', train=False, transform=trans_img,download=True)
print(" 数据集准备完毕!!!")


trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
print(" 数据集加载完毕!!!")

id = 5
(data, label) = trainset[id]
import matplotlib.pyplot as plt
plt.imshow(data.reshape(28, 28), cmap='gray')
plt.title('label is :{}'.format(label))
plt.show()


# build network
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            # 卷积层
            nn.MaxPool2d(2, 2),
            # 最大池化层
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            # 最大池化层
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
        # 全连接层
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
        
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
print(" 网络构建完毕!!!")

#  初始化网络
lenet = Lenet()     # 实例化网络
print(" 网络实例化成功!!!")
lenet.cuda()        # 将网络结构放入GPU计算环境中
print(" 网络已放入GPU计算单元!!!")

#   打印网络结构
print(lenet)

#  设置网络的损失函数及优化器
criterian = nn.CrossEntropyLoss(size_average=False)      # 损失函数
print(" 损失函数已加载 !!!")
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)     # 优化器
print(" 优化器已加载 !!!")

# train
print(" ########## ")
print(" 开始训练啦~ ")
print(" ########## ")

iter = 0
iters = []
iter_loss = []
epoch_id = []
epoch_loss = []
count = 0

for i in range(epoches):
    since = time.time()
    running_loss = 0.
    running_acc = 0.
    for (img, label) in trainloader:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        
        optimizer.zero_grad()
        output = lenet(img)
        loss = criterian(output, label)
        # backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.item()

        iteration_loss = 0
        count += 1
        if iter % 500 == 0:
          iteration_loss = running_loss / count
          iters.append(iter)
          iter_loss.append(iteration_loss)
          print(f"iter:{iter}: loss:{loss.item()} acc:{correct_num.item()/batch_size}" )
        iter += 1
    # iter = 0
    epoch_id.append(i)
    
    running_loss /= len(trainset)
    running_acc /= len(trainset)
    #print("[%d/%d] Loss: %.5f, Acc: %.2f, Time: %.1f s" %(i+1, epoches, running_loss, 100*running_acc, time.time()-since))

    epoch_loss.append(running_loss)
    print("[%d/%d] Loss: %.5f, Acc: %.2f, Time: %.1f s" %(i+1, epoches,
running_loss, 100*running_acc, time.time()-since))

print("#################")
print("本次训练结束!!")
print("#################")

# 绘制loss曲线
# 绘制基于iteration的loss
import matplotlib.pyplot as plt
plt.plot(iters, iter_loss)
plt.ylabel('Loss')
plt.xlabel('iter')
plt.show()
print(iters)
print(iter_loss)

# 绘制基于epoches的loss
plt.plot(epoch_id, epoch_loss)
plt.ylabel('Loss')
plt.xlabel('epoches')
plt.show()
print(epoch_id)
print(epoch_loss)


# evaluate
lenet.eval()
print(" ########## ")
print(" 开始测试 ~ ")
print(" ########## ")

testloss = 0.
testacc = 0.
for (img, label) in testloader:
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    
    output = lenet(img)
    loss = criterian(output, label)
    testloss += loss.item()
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.item()

testloss /= len(testset)
testacc /= len(testset)
print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))

print(" ########## ")
print(" 测试结束~ ")
print(" ########## ")


# 选取一张测试集数据进行测试 id_test 可以自己修改
id_test = 218
(data, label) = testset[id_test]
import matplotlib.pyplot as plt
plt.imshow(data.reshape(28, 28), cmap='gray')
plt.title('label is :{}'.format(label))
plt.show()

# 本段代码将选取的图片输入到网络并进测试
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
count = 0
result = []
for (img, label) in testloader:
    count += 1
    img = Variable(img).cuda()
# 将图片加载进网络并进行预测
    output = lenet(img)
    _, predict = torch.max(output, 1)
    result.append(predict)
    if count==id_test+3:
       break
print("测试结果为:",result[id_test])


# 测试自己手动设计的手写数字 '0.jpg'处需要修改为自己的图片路径
import torchvision.transforms as transforms
from PIL import Image
I = Image.open('0.jpg')
L = I.convert('L')
L = L.resize((28, 28))
plt.imshow(L, cmap='gray')
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
])
im = transform(L)
# [C, H, W]
im = torch.unsqueeze(im, dim=0)
# [N, C, H, W]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
im = im.to(device)
with torch.no_grad():
    outputs = lenet(im)
    print(outputs)
    _, predict = torch.max(outputs.data, 1)
    print(predict)
