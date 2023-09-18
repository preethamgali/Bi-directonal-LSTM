# bi-lstm
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as kb

import numpy as np
import matplotlib.pyplot as plt
import os
import  random

import warnings
warnings.filterwarnings('ignore')

def add_to_file(file_path, data):
  """
        Save data to file
  """
  # file_path = './'+directory_name+'/'+file_name
  with open(file_path,'a+') as f:
      f.write(str(data)+'\n')

def get_test_data(path_to_dir):
    """
            takes path of test dataset file and returns normalized data
    """
    shares = []
    path = path_to_dir
    with open(path) as f:
      data_set = f.read().split()

      for data in data_set:
        data =  data.split(',')

        Share_Prices = [float(i) for i in data[3:]]
        if len(Share_Prices) != n_steps:
          Share_Prices = Share_Prices + [0]* (31-len(Share_Prices))

        shares.append(np.asarray(Share_Prices))

    shares = np.asarray(shares)
    n_samples = shares.shape[0]
    dim = [n_samples, n_steps, n_features]
    shares = shares.reshape(dim)
    shares = normalize(shares)
    data = normalize(np.asarray(shares))
    return data


def get_data_set(path_to_dir):
    """
            takes path of training dataset file and returns normalized data
    """
    shares = []; buy = []; sell = [];
    path = path_to_dir
    with open(path) as f:
      data_set = f.read().split()

      for data in data_set:
        data =  data.split(',')

        Share_Prices = [float(i) for i in data[3:-2]]
        if len(Share_Prices) != n_steps:
          Share_Prices = Share_Prices + [0]* (31-len(Share_Prices))

        Buy_Date = [0]*31
        Buy_Date[int(data[-2])-1] = 1

        Sell_Date = [0]*31
        Sell_Date[int(data[-1])-1] = 1

        shares.append(np.asarray(Share_Prices))
        buy.append(Buy_Date)
        sell.append(Sell_Date)

    shares = np.asarray(shares)
    n_samples = shares.shape[0]
    dim = [n_samples, n_steps, n_features]
    shares = shares.reshape(dim)
    shares = normalize(shares)

    return np.asarray(shares), np.asarray(buy), np.asarray(sell)

def weighted_loss(data):
    """
            class weights is used to provide a weight or bias for each output class
    """
    data = np.sum(data, axis= 0)
    data[data==0] = 1
    maximum = np.max(data)
    data = maximum/data
    return np.clip(data, 0,10)


def get_dict(data):
    """
          Class weights are prodived in the form of dictonary in keras
    """
    class_weight = {}
    for i,j in enumerate(data):
      class_weight[i] = j
    return class_weight

def normalize(data):
    """
        normalizing data between 0 to 1
    """
    temp = []

    for d in data:
      min_ = np.min(d)
      max_ = np.max(data)
      max_min = max_ - min_
      d = (d-min_)/max_min
      temp.append(d)

    return np.asarray(temp)

def get_random_data(data, buy, sell, sample_size):
    """
        return randomly sampled data 
    """
    indexes = random.sample(range(data.shape[0]),sample_size)
    temp_data = [data[i] for i in indexes]
    temp_buy = [buy[i] for i in indexes]
    temp_sell = [sell[i] for i in indexes]

    return np.asarray(temp_data), np.asarray(temp_buy), np.asarray(temp_sell)

def get_model():
    """
                                    Bi- directional LSTM model written in keras

                                    __________________________________________________________________________________________________
                                    Layer (type)                    Output Shape         Param #     Connected to                     
                                    ==================================================================================================
                                    input_1 (InputLayer)            [(None, 31, 1)]      0                                            
                                    __________________________________________________________________________________________________
                                    bidirectional (Bidirectional)   (None, 31, 128)      33792       input_1[0][0]                    
                                    __________________________________________________________________________________________________
                                    bidirectional_1 (Bidirectional) (None, 128)          98816       bidirectional[0][0]              
                                    __________________________________________________________________________________________________
                                    dense (Dense)                   (None, 128)          16512       bidirectional_1[0][0]            
                                    __________________________________________________________________________________________________
                                    dropout (Dropout)               (None, 128)          0           dense[0][0]                      
                                    __________________________________________________________________________________________________
                                    dense_1 (Dense)                 (None, 31)           3999        dropout[0][0]                    
                                    __________________________________________________________________________________________________
                                    dense_2 (Dense)                 (None, 31)           3999        dropout[0][0]                    
                                    ==================================================================================================
                                    Total params: 157,118
                                    Trainable params: 157,118
                                    Non-trainable params: 0
                                    __________________________________________________________________________________________________

    """
    # Bi-LSTM dropout
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session(config=tf.ConfigProto())
    kb.set_session(sess)

    train_info = []

    inputs = Input(shape=(n_steps, n_features))
    lstm1 = Bidirectional(LSTM(64,activation='tanh', input_shape=(n_steps, n_features), return_sequences= True, dropout=0.2))(inputs)
    lstm2 = Bidirectional(LSTM(64, activation='tanh', dropout=0.2))(lstm1)
    d1 = Dense(128, activation='relu')(lstm2)
    dr1 = Dropout(0.2)(d1)

    out1 = Dense(n_steps, activation='softmax')(dr1)
    out2 = Dense(n_steps, activation='softmax')(dr1)

    model = Model(inputs=inputs, outputs=[out1, out2])
    # model.summary()
    
    return model

def visualize_data(shares, x1, x2):
    """
        Plot buy and sell date on monthly share price
    """
    temp = [s for s in shares if s !=0 ]

    plt.figure()
    plt.plot(temp)
    plt.axvline(x=x1, linewidth=1, color='g', label= 'BD')
    plt.axvline(x=x2, linewidth=1, color='r', label= 'SD')
    plt.xlabel("Date")
    plt.ylabel("normalized(price)")
    plt.title("Share price")
    plt.legend()
    plt.show()

def train_model(model):
    """
          To train model on train data set
    """

    data, buy, sell = get_data_set(path_of_train_data_set)

    model.compile(optimizer= 'adam',
              loss = 'categorical_crossentropy',
              metrics=['categorical_accuracy'])

    buy_data_prob = weighted_loss(buy)
    sell_data_prob = weighted_loss(sell)

    buy_data_prob_dict= get_dict(buy_data_prob)
    sell_data_prob_dict= get_dict(sell_data_prob)

    for e in range(3):
        print(e, "iterator")
        temp_data, temp_buy, temp_sell = get_random_data(data, buy, sell, 10000)

        history= model.fit(temp_data, 
                  [temp_buy ,temp_sell], 
                  epochs=100, batch_size=32, 
                  class_weight=[buy_data_prob_dict, sell_data_prob_dict])
        
        add_to_file('./train_model_info.txt', history.history)

        if e%10 == 0:
          # add_to_file('/content/drive/My Drive/data_set_infy/train_model_info.txt', history.history)
          file_name = "saved_weights_"+ str(e) + '.h5'
          model.save_weights('./'+ file_name)

    save_model(model)

def save_model(model):
  """
      To save trained model and its weights
  """
  model.save('./model_data/saved_model.h5')
  model.save_weights('./saved_weights.h5')

def load_trained_model(model):
    """
          To load trained weights into the model
    """
    model.load_weights('./saved_weights.h5')

def predict_buy_sell_date(data):
    """
          To predict buy and sell date ; its takes normalized data as input
    """
    a1,a2= model.predict(np.asarray(data))
    return np.argmax(a1), np.argmax(a2)

def show_training_info():
    """
          To show error and accuracy of model for each epoch happened during training model
    """
    import json

    with open('./train_model_info.txt') as json_file:
        s = json_file.read()
        s = s.replace("\'", "\"")
        list_dict = []
        data = s.split('\n')
        list_dict = [json.loads(d) for d in data if d!='']

    all_loss, all_buy_loss, all_sell_loss, all_buy_acc, all_sell_acc = [],[],[],[],[]
    for dic in list_dict:
        loss, buy_loss, sell_loss, buy_acc, sell_acc = dic.values()
        all_loss += loss
        all_buy_loss += buy_loss
        all_sell_loss += sell_loss
        all_buy_acc += buy_acc
        all_sell_acc += sell_acc

    all_loss= np.asarray(all_loss) 
    all_buy_loss= np.asarray(all_buy_loss)
    all_sell_loss= np.asarray(all_sell_loss)
    all_buy_acc= np.asarray(all_buy_acc) 
    all_sell_acc= np.asarray(all_sell_acc)

    plt.figure(1)
    plt.plot(all_buy_loss/np.max(all_buy_loss) * 100)
    plt.plot(all_sell_loss/np.max(all_sell_loss) * 100)
    plt.xlabel("epochs")
    plt.ylabel("error in %")
    plt.title("Error")
    plt.legend({'sell date error','buy date error'})

    plt.figure(0)
    plt.plot(all_buy_acc * 100)
    plt.plot(all_sell_acc* 100)
    plt.xlabel("epochs")
    plt.ylabel("accuracy in %")
    plt.title("Accuracy")
    plt.legend({'sell date acc','buy date acc'})
    
n_steps = 31
n_features = 1

path_of_train_data_set = "/content/share_train_data.csv"
path_of_test_data_set = "/content/share_test_data.csv"


train_data, buy, sell = get_data_set(path_of_train_data_set)
test_data = get_test_data(path_of_test_data_set)

model = get_model()
# train_model(model)
# show_training_info()
load_trained_model(model)

def test_dataset_output(file1, file2, test_output):
    with open(file1,'r') as f:
      lines =  f.readlines()
    
    with open(file2,'a+') as f:
      full_text = ''
      for line, output in zip(lines, test_output):

        temp = line.split()[0]
        buy= str(output[0])
        sell= str(output[1])
        temp += ',{},{}\n'.format(buy, sell)
        full_text += temp

      f.write(full_text)

test_output = []
for i,t in enumerate(test_data):
  inp = np.asarray([test_data[i]])
  buy_date, sell_date= predict_buy_sell_date(inp)
  test_output.append([buy_date, sell_date])
  
file2 = "share_test_data_with_output.csv"
file1 = "share_test_data.csv"

test_dataset_output(file1, file2, test_output)

index = random.randint(1,1200)
inp = np.asarray([train_data[index]])
predicted_buy_date, predicted_sell_date = predict_buy_sell_date(inp)

actual_buy_date , actual_sell_date = np.argmax(buy[index]), np.argmax(sell[index])

visualize_data(inp.reshape(31), actual_buy_date , actual_sell_date)
print('actual_buy_date:',actual_buy_date, 'actual_sell_date:', actual_sell_date)
print('predicted_buy_date:',predicted_buy_date, 'predicted_sell_date:', predicted_sell_date)
