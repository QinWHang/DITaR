import logging
from torch.utils.tensorboard import SummaryWriter
import os
import time

class Logger(object):
    '''
    A simplified logger that uses the pre-constructed output_dir from args.
    '''

    def __init__(self, args):
        self.args = args
        self._create_logger()

    def _create_logger(self):
        """
        Initialize the logging module.
        """
        main_log_path = self.args.output_dir

        os.makedirs(main_log_path, exist_ok=True)

        now_str = time.strftime("%m%d%H%M%S", time.localtime())
        self.now_str = now_str

        run_instance_dir = os.path.join(main_log_path, now_str)
        os.makedirs(run_instance_dir, exist_ok=True)

        if self.args.log:
            tensorboard_dir = os.path.join(run_instance_dir, 'tensorboard/')
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
        else:
            class DummyWriter:
                def add_scalar(self, *args, **kwargs): pass
            self.writer = DummyWriter()

        self.logger = logging.getLogger(self.args.model_name)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)

        log_file_path = os.path.join(run_instance_dir, 'log.txt')
        self.fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        self.fh.setLevel(logging.DEBUG)
        fm = logging.Formatter("%(asctime)s - %(message)s")
        self.fh.setFormatter(fm)
        self.logger.addHandler(self.fh)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        self.logger.addHandler(self.ch)

        self.logger.info('Parameters are as below:')
        for k, v in vars(self.args).items():
            self.logger.info(f'{k}: {v}')

    def end_log(self):
        if self.fh: self.logger.removeHandler(self.fh)
        if self.ch: self.logger.removeHandler(self.ch)

    def get_logger(self):
        return self.logger, self.writer
    
    def get_now_str(self):
        return self.now_str
