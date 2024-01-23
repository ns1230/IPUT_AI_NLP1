#import library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import re

import os
if not os.path.exists('gakumonno_susume.txt'):
    !wget http://www.aozora.gr.jp/cards/000296/files/47061_ruby_28378.zip
    !unzip 47061_ruby_28378.zip

# File read
f = open('gakumonno_susume.txt', 'r', encoding='sjis')
text = f.read()
f.close()
text = text[:10000]

#青空文庫の場合専用の作業
# 入力テキストを整形する
# ヘッダ削除、- が5つ以上続いた文字の3つ目以降をtextに
text = re.split('\-{5,}',text)[2]
# フッタ削除
text = re.split('底本：',text)[0]
# | 削除
text = text.replace('|', '')
# ルビ削除
text = re.sub('《.+?》', '', text) 
# 入力注削除
text = re.sub('［＃.+?］', '',text)

# 最初のx文字確認
print(text[:1500])
print()

# textの文字の重複を削除し、文字単位に分け、ソートする。
chars = sorted(list(set(text)))

# 文字数を確認、文字単位の辞書サイズを確認
print('Size of text: ', len(text))
print('Total chars: ', len(chars))

# charsの文字に番号をつけ、辞書作成：文字-->番号
char_indices = dict((c,i) for i,c in enumerate(chars))
# 逆引き辞書作成：番号-->文字
indices_char = dict((i,c) for i,c in enumerate(chars))

# 学習用パラメータ
maxlen = 28                                             #VALUE
step = 1                                                #VALUE

# 学習用文字列の作成
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('文字列数: ', len(sentences))

# 小説テキストのベクトル化
# 学習用に0と1だけの要素でできているデータを作成
# X: 学習用入力データ初期化
print("len(sentences), maxlen, len(chars)=",len(sentences), maxlen, len(chars))
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
print(X.shape)

# y: 学習用教師データ初期化
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
print(y.shape)

# sentencesとchar_indicesの値を元に、各要素に1を設定
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1     #各入力sentenceの40の文字の使われているものに1を立てる
    y[i, char_indices[next_chars[i]]] = 1   #sentenceの”正解”に1を立てる

# モデルの定義
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))  
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# モデルのグラフ化
# 作ったモデルのグラフ表示
from tensorflow.keras.utils import plot_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
plot_model(model, to_file='sample-model1.png', show_shapes=True, show_layer_names=True)
im = Image.open('sample-model1.png')
plt.figure(figsize=(8,8))
plt.axis('off')
_ = plt.imshow(np.array(im))

#文字生成用関数の定義
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 小説自動生成
iter_count = 10                                         #VALUE

generated_list = []
history = []
gen_length = 200                                        #VALUE
print("len(chars)=",len(chars))

for iteration in range(iter_count):
    print('')
    print('-' *50)
    print('繰り返し回数: ', iteration+1)
    
    # 学習の実施
    temp = model.fit(X, y, batch_size=128, epochs=1)
    history.append(temp.history['loss'])
    start_index = 9       
    print('')
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    print('Seed文: ',  sentence)
    for i in range(gen_length):
        x = np.zeros((1,maxlen,len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
        preds = model.predict(x, verbose=9)[0]
        next_index = sample(preds)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    generated_list.append(generated)
    print('生成文: ')
    print(generated)
    print('')  

# Lossの推移
plt.figure(figsize=(6,4))
plt.plot(history)
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.xticks(np.arange(0, iter_count+1, 1))
plt.legend(['loss'], loc='upper left')
plt.show()


