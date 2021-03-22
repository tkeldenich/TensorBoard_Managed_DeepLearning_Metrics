```python
#À utiliser lors si l'on exécute plus d'une fois le notebook
#cela supprimes les logs précédemment enregistrés
#%rm -rf ./logs/
```


```python
%load_ext tensorboard
```


```python
from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
    17465344/17464789 [==============================] - 0s 0us/step


    <string>:6: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/datasets/imdb.py:159: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/datasets/imdb.py:160: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])



```python
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Dropout

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```


```python
import datetime, os
import keras

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
```


```python
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=callbacks)
```

    Epoch 1/10
    625/625 [==============================] - 2s 2ms/step - loss: 0.6896 - acc: 0.5402 - val_loss: 0.6536 - val_acc: 0.6666
    Epoch 2/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.6057 - acc: 0.7390 - val_loss: 0.5472 - val_acc: 0.7192
    Epoch 3/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.4839 - acc: 0.7846 - val_loss: 0.5081 - val_acc: 0.7340
    Epoch 4/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.4228 - acc: 0.8129 - val_loss: 0.4959 - val_acc: 0.7496
    Epoch 5/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.3882 - acc: 0.8324 - val_loss: 0.4962 - val_acc: 0.7506
    Epoch 6/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.3630 - acc: 0.8435 - val_loss: 0.4981 - val_acc: 0.7540
    Epoch 7/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.3382 - acc: 0.8542 - val_loss: 0.5036 - val_acc: 0.7540
    Epoch 8/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.3103 - acc: 0.8722 - val_loss: 0.5125 - val_acc: 0.7490
    Epoch 9/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.2969 - acc: 0.8788 - val_loss: 0.5205 - val_acc: 0.7442
    Epoch 10/10
    625/625 [==============================] - 1s 1ms/step - loss: 0.2770 - acc: 0.8896 - val_loss: 0.5317 - val_acc: 0.7402



```python
model.evaluate(x_test, y_test)
```

    782/782 [==============================] - 1s 867us/step - loss: 0.5251 - acc: 0.7516

    [0.5251463055610657, 0.7515599727630615]


```python
%tensorboard --logdir logs
```

    <IPython.core.display.Javascript object>

