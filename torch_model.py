
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('data_train.csv', delimiter=',')
X = dataset[:, 0:6]
y = dataset[:, 6]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


# define the model
class GenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(6, 8)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(8, 6)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(6, 1)
        self.act_output = nn.Sigmoid()
# add a few dropout layers to avoid overfitting

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.dropout1(x)
        x = self.act2(self.hidden2(x))
        x = self.dropout2(x)
        x = self.act_output(self.output(x))
        return x


model = GenderClassifier()
print(model)

# train the model
weight = torch.tensor([0.8, 0.2]) # higher weight for female class
criterion = nn.BCEWithLogitsLoss(weight=weight)
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate accuracy after training

dataset = np.loadtxt('data_test.csv', delimiter=',')
X_val = dataset[:, 0:6]
y_val = dataset[:, 6]

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

y_pred = model(X_val)
accuracy = (y_pred.round() == y_val).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# Evaluate using recall, precision, F1 and ROC
y_val = y_val.detach().numpy()
y_pred = y_pred.round().detach().numpy()
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_val, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_val, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_val, y_pred)
print('F1 score: %f' % f1)
# ROC AUC
auc = roc_auc_score(y_val, y_pred)
print('ROC AUC: %f' % auc)

# Examine results for both male and female