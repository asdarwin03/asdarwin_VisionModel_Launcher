import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter


class Logger(object):

    def __init__(self, logdir, resume=None):
        self.logdir = logdir

        handlers = [logging.StreamHandler(sys.stdout)]
        if logdir is not None:
            if resume is None:
                os.makedirs(logdir)
            handlers.append(logging.FileHandler(os.path.join(logdir, 'log.txt')))
            self.writer = SummaryWriter(log_dir=logdir)
        else:
            self.writer = None

        logging.basicConfig(format=f"[%(asctime)s] %(message)s",
                            level=logging.INFO,
                            handlers=handlers)
        logging.info(' '.join(sys.argv))

    def print(self, p):
        logging.info(p)