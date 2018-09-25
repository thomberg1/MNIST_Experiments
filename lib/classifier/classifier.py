from torch import optim
from .stopping import Stopping
from .checkpoint import Checkpoint
from .logger import TensorboardLogger, PytorchLogger
from .utilities import *
from .scorer import Scorer
from .evaluator import Evaluator
from .trainer import Trainer

#######################################################################################################################

class Classifier(object):
    '''
    Wrapper class for pytorch to train classifier models
    '''
    def __init__(self, H, train_loader, valid_loader):
        self.H = H
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.m = Metric([('train_loss', np.inf), ('train_score', np.inf), ('valid_loss', np.inf),
                         ('valid_score', 0), ('train_lr', 0)])

        self.model = self.H.MODEL(D_in=(1, 28, 28), D_out=10, H=H.HIDDEN_SIZE, dropout=H.DROPOUT,
                                  initialize=H.INITIALIZE, preload=H.PRELOAD if 'PRELOAD' in H else None )
        if H.USE_CUDA:
            self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(list(filter(lambda p:p.requires_grad, self.model.parameters())),
                              lr = self.H.LR, weight_decay = self.H.WEIGHT_DECAY, momentum=self.H.MOMENTUM,
                                   nesterov=self.H.NESTROV)

        self.stopping = Stopping(self.model, patience= self.H.STOPPING_PATIENCE)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[self.H.LR_LAMBDA])

        self.logger = TensorboardLogger(root_dir=self.H.EXPERIMENT, experiment_dir=self.H.TIMESTAMP) #PytorchLogger()

        self.checkpoint = Checkpoint(self.model, self.optimizer, self.stopping, self.m,
                                     root_dir=self.H.EXPERIMENT, experiment_dir=self.H.TIMESTAMP, restore_from=-1,
                                     interval=self.H.CHECKPOINT_INTERVAL, verbose=0)

        self.scorer = Scorer()

        self.trainer = Trainer(self.model, self.train_loader, self.optimizer, self.scheduler, self.criterion, self.scorer)

        self.evaluator = Evaluator(self.model, self.valid_loader, self.criterion, self.scorer)

    def __call__(self):

        epoch_start = self.checkpoint.restore()+1 if self.H.CHECKPOINT_RESTORE else 1
        epoch_itr = self.logger.set_itr(range(epoch_start, self.H.NUM_EPOCHS+1))

        epoch = 0
        for epoch in epoch_itr:

            with DelayedKeyboardInterrupt():

                self.m.train_loss,  self.m.train_score,  self.m.train_lr = self.trainer(epoch)

                self.m.valid_loss, self.m.valid_score = self.evaluator()

                self.checkpoint.step(epoch)

                stopping_flag = self.stopping.step(epoch, self.m.valid_score)

                epoch_itr.log_values( self.m.train_loss, self.m.train_score, self.m.train_lr,
                                      self.m.valid_loss, self.m.valid_score,
                                      self.stopping.best_score_epoch, self.stopping.best_score)

                if stopping_flag:
                    print("Early stopping")
                    break

        self.checkpoint.create(epoch)

#######################################################################################################################