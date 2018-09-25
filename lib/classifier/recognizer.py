import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .utilities import *

#######################################################################################################################

class Recognizer(object):
    '''
    Wrapper that encapsulates the logic to predict the output classes based on an input vector using a trained
    pytorch model.
    '''
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.use_cuda = next(self.model.parameters()).is_cuda

    def __call__(self):

        self.model.eval()
        with torch.no_grad():

            y_preds, y_probas = [], None
            for idx_batch, (input_vars, _) in enumerate(self.loader):
                input_vars = Variable(input_vars)
                if self.use_cuda:
                    input_vars= input_vars.cuda()

                preds = self.model(input_vars)

                out = torch.max(preds, 1)[1]
                y_preds.extend( to_np( out) )

                y_probas = preds if y_probas is None else torch.cat( [y_probas, preds], dim=0)

                del preds

        return np.array(y_preds), to_np(F.softmax( y_probas, dim=1))