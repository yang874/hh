import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# global definitions
BATCH_SIZE = 100
MNIST_PATH = r"C:\Users\yang\Desktop\mnist"


# 把字节数据转化成Tensor，并按照一定标准归一化数据
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

# 准备训练集数据
# training dataset
train_dataset = datasets.MNIST(root=MNIST_PATH,
                               train=True,
                               download=True,
                               transform=transform)
# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=BATCH_SIZE)

#准备测试集数据
# test dataset
test_dataset = datasets.MNIST(root=MNIST_PATH,
                              train=False,
                              download=True,
                              transform=transform)
# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=BATCH_SIZE)


# 定义模型
#简单的三层全连接神经网络
class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


#定义学习率，训练次数，损失函数，优化器
learning_rate = 1e-2
epoches = 20
criterion = nn.CrossEntropyLoss()
model = simpleNet(28*28,128,256,10)
optimizer = optim.SGD(model.parameters(),lr=learning_rate)
num_classes = 10  # 设定类别数量

#定义指标数组
#loss  = []
Acc = []
Pre = []
Rec = []
F1 = []

#模型进行训练

for epoch in range(epoches):
    train_loss = 0
    train_acc = 0

    # 初始化TP, FP, FN计数器
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for img,label in train_loader:
        img = torch.Tensor(img.view(img.size(0),-1))
        label = torch.Tensor(label)
        output = model(img)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        _,pred = output.max(1)
        num_correct = (pred==label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

        for i in range(len(label)):
                    if label[i] == pred[i]:
                        tp[label[i]] += 1
                    else:
                        fp[pred[i]] += 1
                        fn[label[i]] += 1

        precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(num_classes)]
        recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(num_classes)]
        f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i] ) >0 else 0 for i in range(num_classes)]

        # 计算每个类别的平均精确度、召回率和F1分数
        avg_precision = sum(precision) / num_classes
        avg_recall = sum(recall) / num_classes
        avg_f1_score = sum(f1_score) / num_classes

    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Pre: {:.6f},Train Rec: {:.6f}, Train F1_Score: {:.6f}'.\
            format(epoch+1, train_loss/len(train_loader), train_acc/len(train_loader), avg_precision, avg_recall, avg_f1_score))

    #数组填充
    #loss.append(train_loss/len(train_loader))
    Acc.append(train_acc/len(train_loader))
    Pre.append(avg_precision)
    Rec.append(avg_recall)
    F1.append(avg_f1_score)

#测试网络模型
model.eval()
eval_loss = 0
eval_acc = 0
for img,label in test_loader:
    img = torch.Tensor(img.view(img.size(0),-1))
    label = torch.Tensor(label)
    output = model(img)
    loss = criterion(output,label)
    eval_loss += loss.data*img.size(0)
    _ , pred = torch.max(output,1)
    num_correct = (pred==label).sum().item()
    eval_acc += num_correct 
print("Test Loss:{:.6f},Acc:{:.6f}".format(eval_loss/len(test_dataset),eval_acc/len(test_dataset)))

import matplotlib.pyplot as plt

plt.figure(figsize=(20,8),dpi=150) # 设置图片大小
plt.title('指标')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('epoch')  # x轴标题
plt.ylabel('%')  # y轴标题
x = range(1 , epoches + 1)  # x = epoch

#plt.plot(x, loss, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, Acc, marker='o', markersize=3)
plt.plot(x, Pre, marker='o', markersize=3)
plt.plot(x, Rec, marker='o', markersize=3)
plt.plot(x, F1, marker='o', markersize=3)

'''
#for a, b in zip(x, loss):
#    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, Acc):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=1)
for a, b in zip(x, Pre):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=1)
for a, b in zip(x, Rec):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=1)
for a, b in zip(x, F1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=1)
'''
    
plt.legend(['准确率', '精确率', '召回率', 'F1分数'])  # 设置折线名称

plt.show()  # 显示折线图

plt.savefig("C:/Users/yang/Desktop/dian_test/index_image.png")        #将图片保存到本地


