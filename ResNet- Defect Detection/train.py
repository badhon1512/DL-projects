import torch as t
from torch.utils.data import DataLoader
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("data.csv", sep=";")
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)


train_data = ChallengeDataset(train_data, "train")
test_data = ChallengeDataset(test_data, "val")

train_dataloader = DataLoader(train_data, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=20, shuffle=False)


model = model.ResNet()  # custom resnet

loss_criterion = t.nn.BCELoss()
optimizer = t.optim.AdamW(model.parameters(), lr=0.01)
trainer = Trainer(model, loss_criterion, optimizer, train_dataloader, test_dataloader, False)


# go, go, go... call fit on trainer
res = trainer.fit(50)  # TODO


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
