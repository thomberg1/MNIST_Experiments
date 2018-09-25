import torch
from torch.autograd import Variable

#######################################################################################################################

class Evaluator(object):
    '''
    Wrapper to perform the evaluation of a pytorch model
    '''
    def __init__(self, model, loader, criterion, scorer):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.scorer = scorer
        self.use_cuda = next(self.model.parameters()).is_cuda


    def __call__(self):
        self.model.eval()
        with torch.no_grad():

            idx_batch, avg_loss, avg_score = 0, 0.0, 0.0
            for idx_batch,(input_vars, target_vars) in enumerate(self.loader):
                input_vars, target_vars = Variable(input_vars), Variable(target_vars)
                if self.use_cuda:
                    input_vars, target_vars = input_vars.cuda(), target_vars.cuda()

                preds = self.model(input_vars)

                loss = self.criterion(preds, target_vars)
                avg_loss += loss.item()

                y_pred = torch.max(preds, 1)[1]
                avg_score += self.scorer(target_vars, y_pred)

                del preds
                del loss

            avg_loss = avg_loss / (idx_batch + 1)
            avg_score = avg_score / (idx_batch + 1)

            return avg_loss, avg_score

#######################################################################################################################
