from Model import Encoder, Feedforward
from process_sms import read_table, get_features_and_labels, remove_punctuations, tokenize, remove_stopwords
from process_sms import create_vocab_with_id, create_dtm, word2vec
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn

path = "sms.tsv"

# read data
dataset = read_table(path=path, names=["labels", "message"])

# Split dataset into train and test arrays
train_arr, test_arr = train_test_split(dataset, random_state=1)

# Preprocess the training and test data
# 1. Separate into features and labels
X_train, y_train = get_features_and_labels(train_arr.to_numpy())
X_test, y_test = get_features_and_labels(test_arr.to_numpy())

# 2. Remove punctuations
X_train = remove_punctuations(X_train)
X_test = remove_punctuations(X_test)

# 3. Tokenization
X_train = tokenize(X_train)
X_test = tokenize(X_test)

# 4. Remove Stopwords
X_train = remove_stopwords(X_train)
X_test = remove_stopwords(X_test)

# 5. Create vocabulary
vocabulary = create_vocab_with_id(X_train)

# 6. Create document-term matrices for train and test
X_train_dtm = create_dtm(X_train, vocabulary)
X_test_dtm = create_dtm(X_test, vocabulary)


# 7. Create the label arrays
def label_array(arr, label_map):
    labels = []
    for label in arr:
        labels.append(label_map[label])
    return labels


mapping = {"ham": 0, "spam": 1}
y_train = label_array(y_train, mapping)
y_test = label_array(y_test, mapping)

STATE_VEC_DIM = 100
NUM_EPOCHS = 2
OUTPUT_DIM = len(mapping)
HIDDEN_SIZE = 1000
LEARNING_RATE = 0.01
INPUT_SIZE = len(vocabulary)

encoder = Encoder(INPUT_SIZE, STATE_VEC_DIM)
feedforward = Feedforward(STATE_VEC_DIM, HIDDEN_SIZE)

criterion = nn.BCELoss()

for epoch in range(NUM_EPOCHS):
    num_iterations = 0
    for X, y in zip(X_train, y_train):
        state_vec = encoder.init_hidden()
        encoder.zero_grad()
        feedforward.zero_grad()
        y = torch.tensor([y], dtype=torch.float32).view(1, 1)
        for word in X:
            word_vec = word2vec(word, vocabulary)
            state_vec = encoder(word_vec, state_vec)
        output = feedforward(state_vec)

        # Calculate loss and calculate gradients
        loss = criterion(output, y)
        loss.backward()

        # update weights
        for param in encoder.parameters():
            param.data.add_(param.grad.data, alpha=-LEARNING_RATE)

        for param in feedforward.parameters():
            param.data.add_(param.grad.data, alpha=-LEARNING_RATE)

        num_iterations += 1

        if num_iterations % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, iteration: {num_iterations}, loss {loss.item():.4f}")







