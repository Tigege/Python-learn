from keras.models import Sequential
from keras.layers import Dense,Activation
import baidu_test
from sklearn.model_selection import train_test_split
import numpy as np
data = baidu_test.make_data()
# select features and labels for training
dataset_X = data[["K1K2驱动信号","电子锁驱动信号","急停信号","门禁信号","THDV-M","THDI-M","test1","test2","test3","test4"]].as_matrix()
dataset_Y = data[["label","label2"]].as_matrix()
print("--------------")
print(dataset_X)
# split training data and validation set data
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)


model = Sequential()
model.add(Dense(output_dim=2, input_dim=10))
model.compile(loss='mse', optimizer='sgd')

print('Training -----------')
for step in range(3010):
    cost = model.train_on_batch(X_train, y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

print('\nTesting ------------')
cost = model.evaluate(X_test, y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)
Y_pred = model.predict(X_test)
prediction = np.argmax(Y_pred,1)
print("Y_pred")
print(prediction)



