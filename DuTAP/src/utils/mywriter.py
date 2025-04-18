from tensorboardX import SummaryWriter
import os

class MyWriter():
    def __init__(self, logdir):
        self.logdir = logdir
        self.train_LOSS = self.create_writer('Train_LOSS')
        self.val_LOSS = self.create_writer('Validate_LOSS')

    def create_writer(self, name):
        writer = SummaryWriter(os.path.join(self.logdir, name))
        return writer

    def log_train(self, Train_LOSS, Validate_LOSS, epoch):
        self.train_LOSS.add_scalar('LOSS', Train_LOSS, epoch)
        self.val_LOSS.add_scalar('LOSS', Validate_LOSS, epoch)

    # def log_train(self, train_RMSE, train_MAE, Train_LOSS, test_RMSE, test_MAE, Validate_LOSS, epoch):
    #     self.train_RMSE.add_scalar('RMSE', train_RMSE, epoch)
    #     self.train_MAE.add_scalar('MAE', train_MAE, epoch)
    #     self.train_LOSS.add_scalar('LOSS', Train_LOSS, epoch)
    #     self.val_RMSE.add_scalar('RMSE', test_RMSE, epoch)
    #     self.val_MAE.add_scalar('MAE', test_MAE, epoch)
    #     self.val_LOSS.add_scalar('LOSS', Validate_LOSS, epoch)





