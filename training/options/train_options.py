from .base_options import BaseOptions
from pytorch_lightning import Trainer

class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()
        parser = self.parser
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--nepoch_decay', type=int, default=100, help='# of epochs to linearly decay learning rate to zero')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer to use')
        parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help="learning rate")
        parser.add_argument('--momentum', default=0, type=float)
        parser.add_argument('--weight_decay', default=0, type=float)
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        #parser.add_argument('--lr_scheduler_interval', type=str, default='step', help='the interval at which to call the lr scheduler. epoch, step')
        parser.add_argument('--lr_scheduler_warmup_iters', type=int, default=10000, help='the number of warmup iters when using lr policy LinearWarmupCosineAnnealing')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_scheduler_frequency', type=int, default=1, help='the number of intervals (epochs or steps) to wait for next lr scheduler step')
        parser.add_argument('--lr_plateau_threshold', type=float, default=0.01, help='amount by which loss shouldnt vary to consider it a plateau')
        parser.add_argument('--plateau_min_lr', type=float, default=0.0, help='minimun learning rate when doing plateaua lr schedule')
        parser.add_argument('--lr_plateau_patience', type=int, default=5, help='the amount of intervals (epochs by default) to wait with no loss improvement when using plateau lr schedule')
        parser.add_argument('--lr_decay_factor', default=0.999, type=float, help="decay factor to use with multiplicative learning rate schedulers")
        parser.add_argument('--lr_decay_milestones', type=str, default='[500,1000]', help='the milestones at which to decay the learning rate, when using the multi step lr policy')
        parser.add_argument('--scheduler_interval', type=str, default='epoch', help='sets the interval at which the scheduler is called. Either every epoch or every step')
        parser = Trainer.add_argparse_args(parser)
        self.parser = parser
        self.is_train = True
