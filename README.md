****IPUT AI NLP1 - In-Class NLP Work****
This repository contains Python scripts developed for an introductory course on Natural Language Processing (NLP) using AI techniques.

**Contents**
doc2vec.py: This script uses the Doc2Vec model from the gensim library to analyze Japanese texts. It includes code for downloading and processing texts from Aozora Bunko (a Japanese digital library), training a Doc2Vec model on these texts, and using the model to find similar documents.

doc2vec_refined: A refined version of the doc2vec.py script with better code modularization and function documentation. It demonstrates how to download and extract texts, preprocess them, and use them for NLP tasks.

lstm_autonovel.py: An implementation of an LSTM (Long Short-Term Memory) neural network model to generate text. This script demonstrates how to preprocess text data, define and train an LSTM model, and generate new text based on the learned patterns.

**How to Use**
Clone the repository to your local machine.
Install necessary Python packages: gensim, mecab-python3, tensorflow, PIL, and others as required.
Run each script individually in a Python environment. For example, use python doc2vec.py to train and test the Doc2Vec model.
