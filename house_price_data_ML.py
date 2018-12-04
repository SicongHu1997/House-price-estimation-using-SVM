from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from numpy import *
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

house_data = []
house_price = []
with open("final.csv") as f:
    f_c = csv.reader(f)
    i = 0
    for row in f_c:
        house_price.append([float(row[0])])
        house_data.append([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[16]),float(row[7])*float(row[16]),float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]),float(row[13]),float(row[14]),
                           float(row[15]),float(row[17])])
        i += 1
        if i >1000:
            break
house_data = array(house_data).reshape(-1,1)
house_price = array(house_price).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(house_data, house_price, test_size = 0.2)
# f(x) = (x - means) / standard deviation
scaler = StandardScaler()
scaler.fit(x_train)
# standardization
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x=tf.placeholder(tf.float32,[None,18])
y=tf.placeholder(tf.float32,[None,1])

Weight_l1=tf.Variable(tf.random_normal([18,10]))
baise_l1 = tf.Variable(tf.zeros([1,10]))

Wx_plus_b_11=tf.matmul(x,Weight_l1)+baise_l1
l1=tf.nn.tanh(Wx_plus_b_11)


Weight_12=tf.Variable(tf.random_normal([10,1]))
biase_12=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_12=tf.matmul(l1,Weight_12)+biase_12
prediction=tf.nn.tanh(Wx_plus_b_12)

loss=tf.reduce_mean(tf.square(y-prediction))


train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_train,y:y_train})

    predict_value=sess.run(prediction,feed_dict={x:x_test})

    for u in predict_value:
        print(u)
