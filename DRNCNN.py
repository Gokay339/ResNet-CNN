import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image # resimleri preprosses yaparken kullanıcaz
import os


#%% 

def read_images(path,num_img):
    array = np.zeros([num_img,64*32])   # 64*32 resim boyutu
    i = 0
    for img in os.listdir(path):  # os.listdir(path): Belirtilen klasördeki dosya isimlerini döner
        img_path = path + "\\" + img  # Görüntünün tam dosya yolunu oluşturur
        img = Image.open(img_path, mode="r")
        data = np.asarray(img,dtype = "uint8") # Görüntüyü Numpy Dizisine Dönüştürme
        data = data.flatten() # data.flatten(): 2 boyutlu bir diziyi tek boyutlu bir diziye dönüştürür 
                              # (örneğin, 64x32'lik bir görüntü 2048 uzunluğunda bir diziye dönüşür).
        array[i,:] = data
        # Her bir görüntü için array'in i'inci satırına, düzleştirilmiş data dizisi yerleştirilir. 
        # Böylece her satır bir görüntüyü temsil eder.
        i += 1
    return array

# TRAİN ALANI
train_neg_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Train\neg"
num_train_neg_img = 43390
train_neg_array = read_images(train_neg_path, num_train_neg_img)

# TORCH VERİLERİNİ NUMPY VERİLERİNE ÇEVİREBİLİRİZ
# 1 boyutlu vektör 2 boyutlu matris , 3 4 5 6 10 boyutlular bunların genel ismi tensor
x_train_neg_tensor = torch.from_numpy(train_neg_array)
print("x : ",x_train_neg_tensor.size())

y_train_neg_tensor = torch.zeros(num_train_neg_img,dtype = torch.long)
print("y : ",y_train_neg_tensor.size())

# POZİTİFLER
train_pos_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Train\pos"
num_train_pos_img = 10208
train_pos_array = read_images(train_pos_path, num_train_pos_img)

x_train_pos_tensor = torch.from_numpy(train_pos_array)
print("x : ",x_train_pos_tensor.size())

y_train_pos_tensor = torch.ones(num_train_pos_img,dtype = torch.long)
print("y : ",y_train_pos_tensor.size())


# concat train  - pytorch kütüphanesinde concatenate = cat
# YLER GENELLİKLE LABEL  XLER GENELLİKLE RESİMLERİMİZ
x_train = torch.cat((x_train_neg_tensor,x_train_pos_tensor),0)
y_train = torch.cat((y_train_neg_tensor,y_train_pos_tensor),0)
print("x_train size : ",x_train.size())
print("y_train size : ",y_train.size())


# TEST ALANI

test_neg_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Test\neg"
num_test_neg_img = 22050
test_neg_array = read_images(test_neg_path, num_test_neg_img)

# TORCH VERİLERİNİ NUMPY VERİLERİNE ÇEVİREBİLİRİZ
# 1 boyutlu vektör 2 boyutlu matris , 3 4 5 6 10 boyutlular bunların genel ismi tensor
x_test_neg_tensor = torch.from_numpy(test_neg_array[:20855,:])
print("x : ",x_test_neg_tensor.size())

y_test_neg_tensor = torch.zeros(20855,dtype = torch.long)
print("y : ",y_test_neg_tensor.size())

# POZİTİFLER
test_pos_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Test\pos"
num_test_pos_img = 5944
test_pos_array = read_images(test_pos_path, num_test_pos_img)

x_test_pos_tensor = torch.from_numpy(test_pos_array)
print("x : ",x_test_pos_tensor.size())

y_test_pos_tensor = torch.ones(num_test_pos_img,dtype = torch.long)
print("y : ",y_test_pos_tensor.size())


# concat train  - pytorch kütüphanesinde concatenate = cat
# YLER GENELLİKLE LABEL  XLER GENELLİKLE RESİMLERİMİZ
x_test = torch.cat((x_test_neg_tensor,x_test_pos_tensor),0)
y_test = torch.cat((y_test_neg_tensor,y_test_pos_tensor),0)
print("x_train size : ",x_test.size())
print("y_train size : ",y_test.size())


#%%

plt.imshow(x_train[45001,:].reshape(64,32),cmap="gray")


#%% hyper parameters

num_epochs = 50
num_classes = 2
batch_size = 8933
learning_rate = 0.0001

    
import torch.utils.data
train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train , batch_size=batch_size, shuffle=True)

test = torch.utils.data.TensorDataset(x_test,y_test)
testloader = torch.utils.data.DataLoader(test , batch_size=batch_size, shuffle=False)


#%%  Deep Resual Network



  # in_planes  input channel sayısı
  # out_planes layerımdaki nöron sayısı
  # stride  1 olunca birer birer tarama yapar 2 olunca ikişer ikişer tarama yapar. Stride değeri yazmazsak her zaman 1 kabul eder
  # padding size arttırıyor mesela 3x3 matrisi padding 1 vererek 5x5 yapıyoruz . ( bilgi kaybını engellemeye yarar )
  

def conv3x3(in_planes,out_planes,stride = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3 ,stride = stride , padding = 1 , bias=False)


def conv1x1(in_planes,out_planes,stride = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=1 ,stride = stride ,bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride=1,downsample = None):
        super(BasicBlock, self).__init__()    
        
        self.conv1 = conv3x3(inplanes, planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)  # inplace=True anlamı Relu fonksiyonunu çağırdıktan sonra sonucu kendisine eşitle
        self.drop = nn.Dropout(0.9)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        
        
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        
        if self.downsample is not None: 
            # bir bloğun girişinin boyutlarını, o bloğun çıkışıyla uyumlu hale getirmek için kullanılan bir yöntem
            identity = self.downsample(x)
            
        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes = num_classes):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride = 2, padding = 3, bias= False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # girdi boyutu 100x100 olsa bile biz 1x1 verdik çıktı 1x1 olarak çıkıcak
        
        self.fc = nn.Linear(256*block.expansion,num_classes)
        
        # döngünün amacı eğitime başlarken un uygun w değerlerini seçmek
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
                
        
    #  aldığı inputlara göre Basicblockları birbirine bağlar
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride),
                    nn.BatchNorm2d(planes*block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
        
    
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x) # haritalarının boyutunu azaltırken, önemli bilgileri korur
        x = x.view(x.size(0),-1)
        # Çıktıyı düzleştirir (flatten). Bu, çok boyutlu tensoru 2D bir tensor haline getirir; 
        # burada x.size(0) batch boyutunu temsil ederken, -1 otomatik olarak kalanı hesaplamak içindir.
        x = self.fc(x)
        return x
    
    
model = ResNet(BasicBlock,[2,2,2])

#%% OPTİMİZE 

criterion = nn.CrossEntropyLoss()          # 1. Kayıp fonksiyonu tanımı
opti = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 2. Optimizer tanımı

# Kayıp Fonksiyonu (criterion): Modelin tahminlerinin ne kadar doğru olduğunu ölçmek için kullanılır.
# Çapraz entropi kaybı, sınıflandırma problemleri için uygun bir seçimdir.

# Crossentropy Modelin tahmin ettiği olasılık dağılımını ve gerçek etiketlerin bir "one-hot"
# vektörü biçimindeki dağılımını kullanarak kaybı hesaplar.




# Optimizer (opti): Modelin ağırlıklarını güncellemek için kullanılır. Adam optimizasyon algoritması, 
# daha hızlı ve stabil öğrenme sağlamak için yaygın olarak tercih edilir.

# model.parameters(): Modelin ağırlıklarının ve bias terimlerinin güncellenmesi için gerekli olan parametreleri döndürür.
    
    
    
    
#%% TRAİN

loss_list = []
train_acc = []
test_acc = []

total_step = len(trainloader)

for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(trainloader):
        images = images.view(batch_size,1,64,32)
        images = images.float()
        output = model(images)
        
        loss = criterion(output,labels)

        #backward and optimization
        
        opti.zero_grad()
        loss.backward()
        opti.step()
        print("resim shape : ",images.shape)  # Giriş verisinin boyutunu yazdır

        if i %2 == 0 :
            print("epoch {} {} / {}".format(epoch,i,total_step))
    
    correct = 0 
    total = 0 
    with torch.no_grad():
        for data in trainloader:
            images,labels = data
            images = images.view(batch_size,1,64,32)
            images = images.float()


            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy train %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)
    
    
    loss_list.append(loss.item())
    
#%% visualize

fig, ax1 = plt.subplots()
plt.plot(loss_list,label = "Loss",color = "black")
ax2 = ax1.twinx()
ax2.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    












