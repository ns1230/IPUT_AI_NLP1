!apt install aptitude
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
!pip install mecab-python3==0.7

#import
import zipfile
import os.path
import urllib.request as req
import MeCab
from gensim import models
from gensim.models.doc2vec import TaggedDocument

#Mecabの初期化
mecab = MeCab.Tagger()
mecab.parse("")

#学習対象とする青空文庫の作品リスト --- (*1)
list = [
    {"auther":{
        "name":"石川 啄木",
        "url":"https://www.aozora.gr.jp/cards/000153/files/"}, 
     "books":[
        {"name":"火星の芝居","zipname":"43070_ruby_16276.zip"},
        {"name":"閑天地","zipname":"49677_ruby_49019.zip"},
        {"name":"雲は天才である","zipname":"45462_ruby_32209.zip"},
        {"name":"渋民村より","zipname":"49678_ruby_38267.zip"},
        {"name":"初めて見たる小樽","zipname":"812_ruby_20594.zip"},
    ]},
    {"auther":{
        "name":"泉 鏡花",
        "url":"https://www.aozora.gr.jp/cards/000050/files/"}, 
     "books":[
        {"name":"紫陽花","zipname":"3582_ruby_6277.zip"},
        {"name":"一景話題","zipname":"48327_ruby_32329.zip	"},
        {"name":"浮舟","zipname":"50541_ruby_65937.zip"},
        {"name":"歌行灯","zipname":"3587_ruby_5923.zip"},
        {"name":"瓜の涙","zipname":"48328_ruby_33824.zip"},
    ]},
    {"auther":{
        "name":"伊藤 左千夫",
        "url":"https://www.aozora.gr.jp/cards/000058/files/"}, 
     "books":[
        {"name":"牛舎の日記","zipname":"59060_ruby_65439.zip"},
        {"name":"子規と和歌","zipname":"57469_ruby_65939.zip"},
        {"name":"新万葉物語","zipname":"56532_ruby_67953.zip"},
        {"name":"滝見の旅","zipname":"57991_ruby_65661.zip"},
        {"name":"隣の嫁","zipname":"1196_ruby_32528.zip"},
    ]},
    {"auther":{
        "name":"岩野 泡鳴",
        "url":"https://www.aozora.gr.jp/cards/000137/files/"}, 
     "books":[
        {"name":"戦話","zipname":"2944_ruby_5748.zip"},
        {"name":"耽溺","zipname":"1207_ruby_21583.zip"},
        {"name":"猫八","zipname":"50133_ruby_56453.zip"},
        {"name":"黒き素船","zipname":"60898_ruby_74772.zip"},
        {"name":"札幌の印象","zipname":"56528_ruby_67965.zip"},
    ]},
]

#作品リストを取得してループ処理に渡す --- (*2)
def book_list():
    for novelist in list:
        auther = novelist["auther"]
        for book in novelist["books"]:
            yield auther, book
        
#Zipファイルを開き、中の文書を取得する --- (*3)
def read_book(auther, book):
    zipname = book["zipname"]
    #Zipファイルが無ければ取得する
    if not os.path.exists(zipname):
        req.urlretrieve(auther["url"] + zipname, zipname)
    zipname = book["zipname"]
    #Zipファイルを開く
    with zipfile.ZipFile(zipname,"r") as zf:
        #Zipファイルに含まれるファイルを開く。
        for filename in zf.namelist():
            # テキストファイル以外は処理をスキップ
            if os.path.splitext(filename)[1] != ".txt":
                continue
            with zf.open(filename,"r") as f: 
                #今回読むファイルはShift-JISなので指定してデコードする
                return f.read().decode("shift-jis")

#引数のテキストを分かち書きして配列にする ---(*4)
def split_words(text):
    node = mecab.parseToNode(text)
    wakati_words = []
    while node is not None:
        hinshi = node.feature.split(",")[0]
        if  hinshi in ["名詞"]:
            wakati_words.append(node.surface)
#            print("node.surface=",node.surface)
        elif hinshi in ["動詞", "形容詞"]:
            wakati_words.append(node.feature.split(",")[6])
#            print("node.feature=",node.feature)
#            print("node.surface-2=",node.surface)
        node = node.next
    return wakati_words

#作品リストをDoc2Vecが読めるTaggedDocument形式にし、配列に追加する --- (*5)
documents = []
#作品リストをループで回す
for auther, book in book_list():
    #作品の文字列を取得
    words = read_book(auther, book)
    print("author,book=",auther, book)
    #作品の文字列を分かち書きに
    wakati_words = split_words(words)
    #TaggedDocumentの作成　文書=分かち書きにした作品　タグ=作者:作品名
    document = TaggedDocument(
        wakati_words, [auther["name"] + ":" + book["name"]])
    documents.append(document)

#自分のテキストを学習モデルに入れる
if False:
  author = "上條"
  workname = "サンプル"
  f = open('sample.txt', 'r')
  words = f.read()
  f.close()
  #文章を分かち書きに
  wakati_words = split_words(words)
#  print("wakati=",wakati_words)
  #TaggedDocumentの作成　文書=分かち書きにした作品　タグ=作者:作品名
  document = TaggedDocument(
      wakati_words, [author + ":" + workname])
  documents.append(document)

#TaggedDocumentの配列を使ってDoc2Vecの学習モデルを作成 --- (*6)
model = models.Doc2Vec(
    documents, dm=0, vector_size=300, window=15, min_count=1)

#Doc2Vecの学習モデルを保存
model.save('aozora.model')

print("モデル作成完了")

#作成モデルを使った作業開始
#import
import urllib.request as req
import zipfile
import os.path 
import MeCab
from gensim import models

#Mecabの初期化
mecab = MeCab.Tagger()
mecab.parse("")

#保存したDoc2Vec学習モデルを読み込み --- (*7)
model = models.Doc2Vec.load('aozora.model')

#分類用のZipファイルを開き、中の文書を取得する --- (*8)
def read_book(url, zipname):
    if not os.path.exists(zipname):
        req.urlretrieve(url, zipname)

    with zipfile.ZipFile(zipname,"r") as zf:
        for filename in zf.namelist():
            with zf.open(filename,"r") as f:
                return f.read().decode("shift-jis")

#引数のテキストを分かち書きして配列にする
def split_words(text):
    node = mecab.parseToNode(text)
    wakati_words = []
    while node is not None:
        hinshi = node.feature.split(",")[0]
        if  hinshi in ["名詞"]:
            wakati_words.append(node.surface)
        elif hinshi in ["動詞", "形容詞"]:
            wakati_words.append(node.feature.split(",")[6])
        node = node.next
    return wakati_words

#引数のタイトル、URLの作品を分類する --- (*9)
def similar(title, url):
    zipname = url.split("/")[-1]
        
    words = read_book(url, zipname)
    wakati_words = split_words(words)
    print("wakati_words=",wakati_words)
    vector = model.infer_vector(wakati_words)
 #   print("vector=",vector)
    print("--- 「" + title + '」 と似た作品は? ---')
    print(model.docvecs.most_similar([vector],topn=3))
    print("")

#自分のテキストを分類する
def similar_text(title, fn):
    f = open(fn, 'r')
    words = f.read()
    f.close()
    wakati_words = split_words(words)
    vector = model.infer_vector(wakati_words)
    print("--- 「" + title + '」 と似た作品は? ---')
    print(model.docvecs.most_similar([vector],topn=3))
    print("")    

#similar_text("my file",'sample.txt')

#各作家の作品を１つずつ分類 --- (*10)
similar("江戸川 乱歩:悪魔の紋章",
        "https://www.aozora.gr.jp/cards/001779/files/57240_ruby_60876.zip")

