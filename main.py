import os
import logging
import warnings

from torch.serialization import save
logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', UserWarning)

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from lib import dataset, models

import numpy as np
import random

def train(logger, device, dataloader, model, loss_fn, optimizer, epochs, scheduler=None):
    model.train()
    loss_sum = 0
    for epoch in np.arange(1,epochs+1):
        loss_sum = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y= X.to(device), y.to(device)

            #compute loss
            pred = model(X)
            onehot = torch.eye(4)[y].to(device)
            loss = (1 - loss_fn(pred, onehot).sum()/len(X))
            l1_lambda = 1e-3
            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += l1_reg * l1_lambda
            loss_sum += loss.item()

            #BackPropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch+1) == len(dataloader):
                logger.info(f"loss: {loss_sum/len(dataloader):>7f}  [{epoch}/{epochs}]")
        if scheduler:
            scheduler.step()

def test(logger, device, dataloader, model):
    size = 0
    model.eval()
    correct = 0
    answer = np.zeros((4,4), dtype=np.int)
    with torch.no_grad():
        for X, y in dataloader:
            X,y  = X.to(device), y.to(device)
            pred = model(X)
            size += len(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i in np.arange(len(X)):
                answer[pred.argmax(1)[i].to('cpu'), y[i].to('cpu')] += 1
    accuracy = correct/size
    logger.info(f"Accuracy: {(100*accuracy):>0.1f}%")
    logger.info(f"Answer Matrix:\n{answer}")

def main(args):
    try:
        os.remove(args.logdir)
    except:
        pass
    #logger
    logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(filename=args.logdir))

    #args
    logger.info(args)

    #datasets & dataloaders
    train_dataset = dataset.EikllxDataset(args.root, 'Train.csv')
    test_dataset = dataset.EikllxDataset(args.root, 'Test.csv')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #define model
    model = models.get_model(args.model).to(device)
    logger.info(model)

    #loss function & optimizer
    #loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion = torch.nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #learning rate scheduler
    class CosineDecayScheduler:
        def __init__(self, max_epochs: int, warmup_lr_limit=1, warmup_epochs=int(args.n_epochs/10)):
            self._max_epochs = max_epochs
            self._warmup_lr_limit = warmup_lr_limit
            self._warmup_epochs = warmup_epochs

        def __call__(self, epoch: int):
            epoch = max(epoch, 1)
            if epoch <= self._warmup_epochs:
                return self._warmup_lr_limit * epoch / self._warmup_epochs
            epoch -= 1
            rad = np.pi * epoch / self._max_epochs
            weight = (np.cos(rad) + 1.) / 2
            return self._warmup_lr_limit * weight
    lr_scheduler_func = CosineDecayScheduler(args.n_epochs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                    lr_lambda=lr_scheduler_func)

    test(logger, device, test_dataloader, model)

    #train model
    train(logger, device, train_dataloader, model, loss_criterion, 
                    optimizer, args.n_epochs, scheduler=lr_scheduler)

    #test model
    test(logger, device, test_dataloader, model)

    torch.save(model.state_dict(), 'model_log')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data', type=str,
                        help='/path/to/dataset')
    parser.add_argument('--logdir', default='./log/console.log', type=str,
                        help='/path/to/log')
    # Model
    parser.add_argument('--model', default='easycnn', type=str)

    # Optimization
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)

    # seed
    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()

    main(args)