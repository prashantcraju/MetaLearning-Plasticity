import os
import torch
import warnings
import argparse
import datetime

from torch import nn, optim
from random import randrange
from torch.nn.utils import stateless
from torch.utils.data import DataLoader, RandomSampler

from utils import log, Plot, meta_stats
from optim import MyOptimizer, plasticity_rule
from dataset import EmnistDataset, DataProcess

warnings.simplefilter(action='ignore', category=UserWarning)

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        dim_out = 47
        self.fc1 = nn.Linear(784, 170, bias=False)
        self.fc2 = nn.Linear(170, 130, bias=False)
        self.fc3 = nn.Linear(130, 100, bias=False)
        self.fc4 = nn.Linear(100, 70, bias=False)
        self.fc5 = nn.Linear(70, dim_out, bias=False)

        self.fk1 = nn.Linear(784, 170, bias=False)
        self.fk2 = nn.Linear(170, 130, bias=False)
        self.fk3 = nn.Linear(130, 100, bias=False)
        self.fk4 = nn.Linear(100, 70, bias=False)
        self.fk5 = nn.Linear(70, dim_out, bias=False)

        self.a_fwd = nn.Parameter(torch.tensor(args.a).float())
        self.b_fwd = nn.Parameter(torch.tensor(0.).float())
        self.c_fwd = nn.Parameter(torch.tensor(0.).float())
        self.d_fwd = nn.Parameter(torch.tensor(0.).float())
        self.e_fwd = nn.Parameter(torch.tensor(0.).float())

        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

        self.params_fwd = nn.ParameterList()

    def forward(self, x):
        y0 = x.squeeze(1)
        y1 = self.sopl(self.fc1(y0))
        y2 = self.sopl(self.fc2(y1))
        y3 = self.sopl(self.fc3(y2))
        y4 = self.sopl(self.fc4(y3))
        return (y0, y1, y2, y3, y4), self.fc5(y4)

class MetaLearner:
    def __init__(self, metatrain_dataset, args):
        self.device = args.device
        self.K = args.K
        self.M = args.M
        self.database = args.database
        self.metatrain_dataset = metatrain_dataset
        self.data_process = DataProcess(K=self.K, Q=args.Q, dim=args.dim, device=self.device)
        self.model = self.load_model(args).to(self.device)
        self.Theta = nn.ParameterList([*self.model.params_fwd])
        self.fbk = args.fbk
        self.lamb = args.lamb
        self.loss_func = nn.CrossEntropyLoss()
        self.OptimAdpt = MyOptimizer(plasticity_rule, args.vec, args.fbk)
        self.OptimMeta = optim.Adam([{'params': self.model.parameters(), 'lr': args.lr_meta}])
        self.res_dir = args.res_dir
        self.plot = Plot(self.res_dir, len(self.Theta), args.avg_window)

    def load_model(self, args):
        model = MyModel(args)
        for key, val in model.named_parameters():
            if 'fc' in key or 'fk' in key:
                val.meta_fwd, val.adapt, val.requires_grad = True, True, True
            elif 'fwd' in key:
                val.meta_fwd, val.adapt = True, False
            if val.meta_fwd is True:
                model.params_fwd.append(val)
        return model

    def reinitialize(self):
        self.model.apply(self.weights_init)
        if self.fbk == 'sym':
            for fk, fc in zip([self.model.fk1, self.model.fk2, self.model.fk3, self.model.fk4, self.model.fk5],
                              [self.model.fc1, self.model.fc2, self.model.fc3, self.model.fc4, self.model.fc5]):
                fk.weight.data = fc.weight.data.clone()
        params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if '.' in key}
        for key in params:
            params[key].adapt = dict(self.model.named_parameters())[key].adapt
        return params

    def train(self):
        self.model.train()
        for eps, data in enumerate(self.metatrain_dataset):
            params = self.reinitialize()
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, self.M)
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):
                y, logits = _stateless.functional_call(self.model, params, x.unsqueeze(0).unsqueeze(0))
                self.OptimAdpt(params, logits, label, y, self.model.Beta, self.Theta)
            y, logits = _stateless.functional_call(self.model, params, x_qry.unsqueeze(1))
            l1_reg = None
            for T in self.model.params_fwd.parameters():
                if l1_reg is None:
                    l1_reg = T.norm(1)
                else:
                    l1_reg += T.norm(1)
            loss_meta = self.loss_func(logits, y_qry.ravel()) + l1_reg * self.lamb
            acc = meta_stats(logits, params, y_qry.ravel(), y, self.model.Beta, self.res_dir)
            Theta = [p.detach().clone() for p in self.Theta]
            self.OptimMeta.zero_grad()
            loss_meta.backward()
            self.OptimMeta.step()
            log([loss_meta.item()], self.res_dir + '/loss_meta.txt')
            line = 'Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(eps+1, loss_meta.item(), acc)
            for idx, param in enumerate(Theta):
                line += '\tMetaParam_{}: {:.6f}'.format(idx + 1, param.cpu().numpy())
            print(line)
            with open(self.res_dir + '/params.txt', 'a') as f:
                f.writelines(line+'\n')
        self.plot()

# Additional utility and main function remain unchanged.

def parse_args():
    """
        Parses the input arguments for the meta-learning model.

    The function creates an argument parser with various input parameters for
    the model. These parameters include processor, data, meta-training, log,
    and model parameters. After parsing the input arguments, the function sets
    up the storage and GPU settings and returns the validated input arguments
    using the check_args() function.

    :return: argparse.Namespace: The validated input arguments for the
    meta-learning model.
    """
    desc = "Pytorch implementation of meta-learning model for discovering biologically plausible plasticity rules."
    parser = argparse.ArgumentParser(description=desc)

    # -- processor params
    parser.add_argument('--gpu_mode', type=int, default=1, help='Accelerate the script using GPU.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    # -- data params
    parser.add_argument('--database', type=str, default='emnist', help='Meta-training database.')
    parser.add_argument('--dim', type=int, default=28, help='Dimension of the training data.')
    parser.add_argument('--test_name', type=str, default='',
                        help='Name of the folder at the secondary level in the hierarchy of the results directory '
                             'tree.')

    # -- meta-training params
    parser.add_argument('--episodes', type=int, default=600, help='Number of meta-training episodes.')
    parser.add_argument('--K', type=int, default=50, help='Number of training data points per class.')
    parser.add_argument('--Q', type=int, default=10, help='Number of query data points per class.')
    parser.add_argument('--M', type=int, default=5, help='Number of classes per task.')
    parser.add_argument('--lamb', type=float, default=0., help='Meta-loss regularization parameter.')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='Meta-optimization learning rate.')
    parser.add_argument('--a', type=float, default=1e-3,
                        help='Initial learning rate for the pseudo-gradient term at episode 0.')

    # -- log params
    parser.add_argument('--res', type=str, default='results', help='Result directory.')
    parser.add_argument('--avg_window', type=int, default=10,
                        help='The size of moving average window used in the output figures.')

    # -- model params
    parser.add_argument('--vec', nargs='*', default=[],
                        help='Index vector specifying the plasticity terms to be used for model training in '
                             'adaptation.')
    parser.add_argument('--fbk', type=str, default='sym',
                        help='Feedback connection type: 1) sym = Symmetric feedback; 2) fix = Fixed random feedback.')

    args = parser.parse_args()

    # -- storage settings
    s_dir = os.getcwd()
    args.res_dir = os.path.join(s_dir, args.res, args.test_name,
                                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(randrange(40)))
    os.makedirs(args.res_dir)

    # -- GPU settings
    args.device = torch.device('cuda' if (bool(args.gpu_mode) and torch.cuda.is_available()) else 'cpu')

    # -- feedback type
    args.evl = False

    return check_args(args)


def check_args(args):
    """
        Check validity of the input arguments.

    This function checks validity of the input arguments. It also stores the
    settings by writing them to a file named `args.txt` in the `res_dir`
    directory specified in the input arguments.

    :param args: (argparse.Namespace) The command-line arguments.
    :return: (argparse.Namespace) The validated input arguments.
    """
    # -- GPU check
    # If the gpu_mode argument is set to True but no GPUs are
    # available on the device, a message is printed indicating
    # that the program will run on the CPU instead.
    if bool(args.gpu_mode) and not torch.cuda.is_available():
        print('No GPUs on this device! Running on CPU.')

    # -- store settings
    with open(args.res_dir + '/args.txt', 'w') as fp:
        for item in vars(args).items():
            fp.write("{} : {}\n".format(item[0], item[1]))

    return args


def main():
    """
        Main function for Meta-learning the plasticity rule.

    This function serves as the entry point for meta-learning model training
    and performs the following operations:
    1) Loads and parses command-line arguments,
    2) Loads custom EMNIST dataset using meta-training arguments (K, Q),
    3) Creates tasks for meta-training using `RandomSampler` with specified
        number of classes (M) and episodes,
    4) Initializes and trains a MetaLearner object with the set of tasks.

    :return: None
    """
    # -- load arguments
    args = parse_args()

    # -- load data
    dataset = EmnistDataset(K=args.K, Q=args.Q, dim=args.dim)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=args.episodes * args.M)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.M, drop_last=True)

    # -- meta-train
    metalearning_model = MetaLearner(metatrain_dataset, args)
    metalearning_model.train()


if __name__ == '__main__':
    main()
