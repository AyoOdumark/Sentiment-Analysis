import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from process_sms import read_table, get_features_and_labels, remove_punctuations, tokenize, remove_stopwords
from process_sms import create_vocab_with_id, create_dtm

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

# 4. Remove stopwords
X_train = remove_stopwords(X_train)
X_test = remove_stopwords(X_test)

# 5. create vocabulary
vocabulary = create_vocab_with_id(X_train)

# 6. Create document-term matrices for train and test
X_train_dtm = create_dtm(X_train, vocabulary)
X_test_dtm = create_dtm(X_test, vocabulary)


# 7. Create the label arrays
def label_array(arr, label_map):
    labels = []
    for label in arr:
        labels.append(label_map[label])
    labels = np.array(labels, dtype=np.float32)
    labels = labels.reshape(-1, 1)
    return labels


mapping = {"ham": 0, "spam": 1}
y_train = label_array(y_train, mapping)
y_test = label_array(y_test, mapping)


# Create dataloader
class SMSDataset(Dataset):
    def __init__(self, X, y):
        super(SMSDataset, self).__init__()
        self.X = X
        self.y = y
        self.n_samples = len(y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n_samples


# Convert training and test data into torch tensors
X_train_dtm = torch.from_numpy(X_train_dtm)
y_train = torch.from_numpy(y_train)

X_test_dtm = torch.from_numpy(X_test_dtm)
y_test = torch.from_numpy(y_test)

# create dataloaders
BATCH_SIZE = 100

train_data = SMSDataset(X_train_dtm, y_train)
test_data = SMSDataset(X_test_dtm, y_test)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


# Build the Model
class Classifier(nn.Module):
    def __init__(self, num_labels, vocab_length):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(vocab_length, num_labels)

    def forward(self, input):
        return torch.sigmoid(self.linear(input))


OUTPUT_DIM = 1
VOCAB_SIZE = len(vocabulary)
LEARNING_RATE = 0.1
NUM_EPOCHS = 10
NUM_ITERATIONS = len(train_loader)

model = Classifier(OUTPUT_DIM, VOCAB_SIZE)

# define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (texts, labels) in enumerate(train_loader):
        # forward pass
        outputs = model(texts)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f"epoch {epoch+1}/{NUM_EPOCHS}, step {i+1}/{NUM_ITERATIONS}, loss {loss.item():.4f}")


# 5. Evaluate
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for texts, labels in test_loader:
        outputs = model(texts)
        n_samples += labels.shape[0]
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()

    acc = n_correct/n_samples
    print(f"accuracy: {acc:.4f}")
