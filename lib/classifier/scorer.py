from sklearn import metrics

#######################################################################################################################

class Scorer(object):
    '''
    Wrapper for the metrics function used to score a pytorch model.
    '''
    def __init__(self):
        self.metrics = metrics.accuracy_score

    def __call__(self, predicted_vars, target_vars):
        acc = self.metrics(predicted_vars.cpu().numpy(), target_vars.cpu().numpy())
        return acc

#######################################################################################################################

