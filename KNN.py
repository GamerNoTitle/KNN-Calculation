import numpy as np  #导入numpy库
import pandas as pd #导入Pandas库
from sklearn import neighbors   #从sklearn导入neighbors函数
from sklearn.model_selection  import train_test_split   #从sklearn.model_selection导入train_test_split函数

features = list()  #存放特征
labels = list()    #存放分类标签
data=pd.read_csv('.\Iris_Data.csv') #利用Pandas读取csv文件
ln=data.shape[0]    #设变量ln为csv文件的行数
data_list=[]    #初始化data_list
for i in range(ln): 
    temp=[] #初始化temp
    for n in data:
        if n == 'species':  #因为特征不读取结果，所以不能读取species的内容，就要break出循环
            break
        a=data.loc[i,n] #设a的值为文件中的特征数
        temp.append(a)  #将特征数a加入到temp中
    data_list.append(temp)  #将temp加入data_list列表中
    #print(temp)    打印temp
label=[]    #初始化label
for i in range(ln):
    for n in data:
        if n == 'species':  #这里是一定要读取species
            label.append(data.loc[i,n]) #将species记录进label中
            break
feature_train,feature_test,label_train,label_test=train_test_split(data_list,label,test_size=0.3,shuffle='species') #使用train_test_split函数随机打乱，test_size测试占比30%,shuffle打乱标准为species
#print(feature_train)
#print(feature_test)
#print(label_train)
#print(label_test)
knn = neighbors.KNeighborsClassifier()  #设定k的值，不输入默认为5
knn.fit(feature_train,label_train)  #开始训练
result=knn.predict(feature_test)    #预测结果
var=np.mean(result==label_test) #获得准确值
print(var)