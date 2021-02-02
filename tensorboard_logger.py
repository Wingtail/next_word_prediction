import traceback
from tensorboardX import SummaryWriter
import os
from datetime import datetime

class TensorboardLogger(object):
    def __init__(self, log_dir, model_name):
        self.model_name = model_name

        log_dir = self.create_tb_dir(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.train_stats = {}
        self.eval_stats = {}
    
    def create_tb_dir(self, log_dir):
        if not os.path.exists(log_dir):
            print("made dir")
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = log_dir + '-' + datetime.now().strftime("%B-%d-%Y_%I+%M%p")
            os.makedirs(log_dir, exist_ok=True)
            print("Overlapping Tensorboard directory detected")
            print("Creating new log directory of ", log_dir)
        return log_dir

    def dict_to_tb_scalar(self, scope_name, stats, step):
        for key, value in stats.items():
            self.writer.add_scalar('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_text(self, stats, step):
        for key, value in stats.items():
            self.writer.add_text(key, value, step)

    def dict_to_tb_figure(self, scope_name, figures, step):
        for key, value in figures.items():
            self.writer.add_figure('{}/{}'.format(scope_name, key), value, step)

    def tb_train_iter_stats(self, step, stats):
        self.dict_to_tb_scalar(f"{self.model_name}_TrainStats", stats, step)

    def tb_eval_stats(self, step, stats):
        self.dict_to_tb_scalar(f"{self.model_name}_EvalStats", stats, step)

    def tb_add_text(self, texts, step):
        self.dict_to_tb_text(texts, step)
