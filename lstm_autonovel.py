#import library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import os

# Download and unzip text if not present
if not os.path.exists('gakumonno_susume.txt'):
    !wget http://www.aozora.gr.jp/cards/000296/files/47061_ruby_28378.zip
    !unzip 47061_ruby_28378.zip

# File read
with open('gakumonno_susume.txt', 'r', encoding='sjis') as f:
    text = f.read()
text = text[:10000]

# Preprocess text data
text = re.split('\-{5,}', text)[2]
text = re.split('底本：', text)[0]
text = text.replace('|', '').replace('\r', '')
text = re.sub('《.+?》', '', text)
text = re.sub('［＃.+?］', '', text)

# Confirm first 1500 characters
print(text[:1500])
print()

# Prepare characters and indices
chars = sorted(list(set(text)))
print('Size of text: ', len(text))
print('Total chars: ', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Parameters
maxlen, step = 28, 1

# Create sequences
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences: ', len(sentences))

# Vectorize text
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Define model
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars)),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

# Plot model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
im = Image.open('model.png')
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(np.array(im))
plt.show()

# Function to generate text
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate text
iter_count = 10
generated_list = []
history = []

for iteration in range(iter_count):
    print('\n' + '-' * 50)
    print('Iteration: ', iteration + 1)
    
    temp = model.fit(X, y, batch_size=128, epochs=1)
    history.append(temp.history['loss'])
    start_index = random.randint(0, len(text) - maxlen - 1)
    
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    print('Seed sentence: ', sentence)
    for i in range(200):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    generated_list.append(generated)
    print('Generated text:')
    print(generated)

# Plot loss
plt.figure(figsize=(6,4))
plt.plot(history)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.xticks(np.arange(0, iter_count+1, 1))
plt.legend(['Loss'], loc='upper left')
plt.show()
