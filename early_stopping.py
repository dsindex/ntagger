from __future__ import print_function

# ------------------------------------------------------------------------------ #
# base source from http://forensics.tistory.com/29
# ------------------------------------------------------------------------------ #

class EarlyStopping():

    def __init__(self, logger, patience=0, measure='loss', verbose=0):
        """Set early stopping condition

        Args:
          Patience: how many times to be patient before "Early Stopping".
          Measure: checking measure, Loss | F1 | Accuracy.
          Verbose: if 1, enable Verbose mode.
        """
        self._step = 0
        if measure == 'loss': # Loss
            self._value = float('inf')
        else:                 # F1, Accuracy
            self._value = 0
        self.logger = logger
        self.patience  = patience
        self.verbose = verbose

    def reset(self, value):
        self._step = 0
        self._value = value

    def status(self):
        self.logger.info('EarlyStopping Status: _step / patience = %d / %d, value = %f' % (self._step, self.patience, self._value))

    def step(self):
        return self._step

    def validate(self, value, measure='loss'):
        going_worse = False
        if measure == 'loss': # loss
            if self._value < value: going_worse = True
        else:                 # f1, accuracy
            if self._value > value: going_worse = True 
        if going_worse:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    self.logger.info('Training process is halted early!!!')
                return True
        else:
            self.reset(value)
        return False
