import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):

        self._optim.zero_grad()
        y_hat = self._model(x)
        y = y.float()
        loss = self._crit(y_hat, y)
        loss.backward()
        self._optim.step()
        return loss


    def val_test_step(self, x, y):
        y_hat = self._model(x)
        return self._crit(y_hat, y.float())  , y_hat

    def train_epoch(self):
        self._model.train(True)
        loss = 0
        if self._cuda:
            self._train_dl = self._train_dl
        for _, data in tqdm(enumerate(self._train_dl), unit="batch", total=len(self._train_dl)):  # iterate through the
            X, y = data
            loss += self.train_step(X, y) .item()
        avg_loss = loss / len(self._train_dl.dataset)
        print("Training loss:", avg_loss, end="\n")

        return avg_loss

    def val_test(self):
        self._model.train(False)
        loss = 0
        if self._cuda:
            self._val_test_dl = self._val_test_dl.cuda()

        with t.no_grad():
            for _, data in tqdm(enumerate(self._val_test_dl), unit="batch", total=len(self._val_test_dl)):  # iterate through the validation set
                X, y = data
                l, y_hat = self.val_test_step(X, y)
                loss += l.item()
        avg_loss =  loss / len(self._val_test_dl)
        print("Val Average Loss: {:.3f}".format(avg_loss))

        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        dev_losses = []
        c_epoch = 0

        while c_epoch < epochs:
            print('Epoch {}'.format(c_epoch))
            train_losses.append(self.train_epoch())
            dev_losses.append(self.val_test())
            self.save_checkpoint(c_epoch)
            c_epoch += 1
        return train_losses, dev_losses




