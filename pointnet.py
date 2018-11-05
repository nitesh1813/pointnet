
# coding: utf-8

# In[1]:


import  utils
import h5py
import numpy as np
import math
import tensorflow as tf


# In[2]:


def rotate(sample,theta):
    c=math.cos(theta)
    s=math.sin(theta)
    affine=np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
    sample=np.matmul(sample,affine)
    affine=np.matrix([[1,0,0],[0,c,-s],[0,s,c]])
    sample=np.matmul(sample,affine)
    affine=np.matrix([[c,0,s],[0,1,0],[-s,0,c]])
    sample=np.matmul(sample,affine)
   
    return sample
def jitter(data):
    noise=np.random.normal(0,0.1,(2048,3))
    return data+noise
def rotateall(data,label):
    list=[]
    labels=[]
    for x,y in zip(data,label):
        list.append(rotate(x,math.pi/4.0))
#         list.append(rotate(x,math.pi/2.0))
        list.append(jitter(x))
        list.append(rotate(x,math.pi*3.0/4))
        labels.extend([y for i in range(3)])
    return list,labels
#         list.append(rotate(x,math.pi))


# In[3]:


def load_h5(filepath,flag):
    """
    Data loader function.
    Input: The path of h5 filename
    Output: A tuple of (data,label)
    """
    data=[]
    label=[]
    file=open(filepath)
    for h5_filename in file:
        h5_filename=h5_filename.strip()
#         print(h5_filename)
        f = h5py.File(h5_filename)
        data.extend( f['data'][:])
        label.extend(f['label'][:])
        if flag:
            points,labels=rotateall(f['data'][:],f['label'][:])
            data.extend(points)
            label.extend(labels)
            
        
    return (data, label)


# In[4]:


data=load_h5("data/modelnet40_ply_hdf5_2048/train_files.txt",True)


# In[5]:


def one_hot(label):
    output=[]
    for x in label:
        temp=[0]*40
#         print(x)
        temp[x[0]]=1
        output.append(temp)
    return output


# In[6]:


def create_batch(data,batch_size):
    batchtrainData=[]
    batchtrainLabel=[]
    trainingData,trainingLabel=data
    trainingLabel=one_hot(trainingLabel)
    n=len(trainingData)
    
    for i in range (int(n/batch_size)):
        batchtrainData.append(trainingData[batch_size*i:min(batch_size*(i+1),n)])
        batchtrainLabel.append(trainingLabel[batch_size*i:min(batch_size*(i+1),n)])
    return np.array(batchtrainData),np.array(batchtrainLabel)
    


# In[7]:


x,y=data
len(x)
# sess=data[0][0]


# In[8]:


def createConvLayer(input_layer,flag,layer,inputsize,size):
    with tf.variable_scope(layer):
        conv1 = tf.layers.conv2d(inputs=input_layer,filters=size,kernel_size=[1, inputsize],padding="Valid",activation=tf.nn.relu)
        conv_bn = tf.contrib.layers.batch_norm(conv1, data_format='NHWC',center=True, scale=True,is_training=flag,scope='bn')

    return conv_bn
def createDenseCell(x,flag,size,layer):
    with tf.variable_scope(layer):
        h1 = tf.contrib.layers.fully_connected(x, size,activation_fn=tf.nn.relu,scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,center=True, scale=True,is_training=flag,scope='bn')
        return h2
def classificationLayer(flag,input,layers):
    output=createDenseCell(input,flag,512,layers[0])
    output2=createDenseCell(output,flag,256,layers[1])
    output3=createDenseCell(output,flag,40,layers[2])
    return output3
def getLayer(input_image,flag,mlp,n):
    new_points=input_image
#         number_points=input_image.shape[1]

    for i, num_out_channel in enumerate(mlp,n):
            inputsize=new_points.shape[2]
            new_points = createConvLayer(new_points,flag,'conv%d'%(i),inputsize,num_out_channel)
#                 print(new_points.shape,number_points)
            new_points=tf.reshape(new_points,[-1,number_points,num_out_channel,1])
    return new_points


def tnet(input_image,flag,size,n):
    mlp1=getLayer(input_image,flag,[64,64,128,1024],n)

#         print(mlp1.shape)

    pooled = tf.nn.max_pool(mlp1,
                             ksize=[1, number_points, 1, 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID',
                             name="pool")
    featurevec=tf.reshape(pooled,[-1,1024])
#         print(featurevec.shape)
    output1=createDenseCell(featurevec,flag,int(512*int(math.sqrt(size))),"layers%d"%(size))
    output2 = tf.contrib.layers.fully_connected(output1, 256*(int(math.sqrt(size))),activation_fn=tf.nn.relu)
    matrix=tf.contrib.layers.fully_connected(output1, size*size,activation_fn=tf.nn.relu)
    matrix=tf.reshape(matrix,[-1,size,size])
    return matrix


# In[ ]:



        


# In[9]:


batch_size=20
number_points=2048
label_number=40

with tf.device('/gpu:0'):
    tf.reset_default_graph()
    
    point_cloud = tf.placeholder(tf.float32,
                                      shape=(batch_size, number_points, 3))
    flag=tf.placeholder(tf.bool,name='flag')
    
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, label_number))
    input_image = tf.expand_dims(point_cloud, -1)
    matrix=tnet(input_image,flag,3,0)
    transformed_input=tf.matmul(point_cloud,matrix)
    input_image = tf.expand_dims(transformed_input, -1)
    mlp1=getLayer(input_image,flag,[64,64],5)
    f_matrix=tnet(mlp1,flag,64,10)
    temp_feat=tf.reshape(mlp1,[-1,number_points,64])
    transformed_feat=tf.matmul(temp_feat,f_matrix)
    transformed_feat=tf.expand_dims(transformed_feat,-1)
    mlp2=getLayer(transformed_feat,flag,[64,128,1024],18)


    pooled = tf.nn.max_pool(mlp2,
                         ksize=[1, number_points, 1, 1],
                         strides=[1, 1, 1, 1],
                         padding='VALID',
                         name="pool")
    featurevec=tf.reshape(pooled,[-1,1024])

#             print(featurevec.shape)
    logits=classificationLayer(flag,featurevec,["layer5","layer6","layer7"])


# In[ ]:


with tf.device('/gpu:0'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf_train_labels, 1), tf.argmax(logits, 1)),'float32'))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


# In[ ]:


epoch=200
# sess=tf.InteractiveSession()
# output=open("output.txt",'w')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
batch_traindata,batch_trainlabel=create_batch(data,batch_size)
# print(len(train_batch))
# x_valid,y_valid=mnist.validation.next_batch(5000)
allLoss=[]
import time
a = time.time()
with sess.as_default():
    tf.global_variables_initializer().run()
    for i in range(epoch):
        running_loss=0
        running_acc=0
        for x_train,y_train in zip(batch_traindata,batch_trainlabel):
    #         print(y_train.shape)
            t,loss_t,acc=sess.run([optimizer,loss,accuracy], feed_dict={point_cloud:x_train,tf_train_labels:y_train,flag:True})
            running_loss+=loss_t
            running_acc+=acc
        print (running_loss,running_acc/len(batch_traindata))

print (time.time() -a)


# In[ ]:


testdata=load_h5("data/modelnet40_ply_hdf5_2048/test_files.txt",False)
batch_testdata,batch_testlabel=create_batch(testdata,batch_size)
print(np.random.random((2048,3)).shape)
# batch_testdata[0][6]-=2*np.ones((2048,3))
def getAccuracy(batchpoints,batchlabel):
    running_acc=0
    for x,y in zip(batchpoints,batchlabel):
    #         print(y_train.shape)
            acc=sess.run(accuracy, feed_dict={point_cloud:x,tf_train_labels:y,flag:False})
           
#             print(acc)
            running_acc+=acc
    return running_acc
print(getAccuracy(batch_testdata,batch_testlabel))


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def Visualize_3D(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()
image=x[6]
print(image.shape)
# image=transform[5]
Visualize_3D(image[:,0],image[:,1],image[:,2])
# print("Test Accuracy: ",getAccuracy(batch_testdata,batch_testlabel))


# In[ ]:


image=transform[6]
# image=x[]
Visualize_3D(image[:,0],image[:,1],image[:,2])


# In[ ]:




