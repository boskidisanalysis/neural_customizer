import streamlit as st

st.title('Customize a Neural Network Classifier')
st.write('You can select the parameters of the Neural Network to classify handwritten digits')

num_neurons = st.sidebar.slider('Number of Neurons in hidden layer:', 1,100, )
num_epochs = st.sidebar.slider('Number of Epochs:', 1,100, )
activation = st.sidebar.text_input('Activation Function:')

st.write('Number of Neurons: ',num_neurons)
st.write('Num of Epochs: ', num_epochs)
st.write('Activation Function: ', activation)

if st.button('Train the model'):
    import tensorflow as tf 
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    def preprocess_image(images):
        images = images/255
        return images

    x_train = preprocess_image(x_train)
    x_test = preprocess_image(x_test)
    model = Sequential([
                    InputLayer(input_shape=(28,28)),
                    Flatten(),
                    Dense(num_neurons, activation),
                    Dense(10, 'softmax')
    ])
    model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    save_cp = ModelCheckpoint('model', save_best_only=True)
    history_cp = tf.keras.callbacks.CSVLogger('history.csv')
    model.fit(x_train, y_train,
          epochs= num_epochs,
          validation_split=0.2,
          callbacks=[save_cp, history_cp])



if st.button('Evaluate the model'):
    import pandas as pd 
    import matplotlib.pyplot as plt
    history = pd.read_csv('history.csv', index_col='epoch')
    fig, ax = plt.subplots(figsize=(10,6));
    history.plot(ax=ax, title='Neural Network Training')
    fig
    