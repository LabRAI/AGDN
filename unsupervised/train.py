import argparse
import time

import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader  # use pytorch dataloader
from tqdm import tqdm, trange

from utils import TSPLoss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=200, help='Graph Size')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Learning Rate')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='num of layers')
parser.add_argument('--EPOCHS', type=int, default=20,
                    help='epochs to train')
parser.add_argument('--penalty_coefficient', type=float, default=2.,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--rescale', type=float, default=2.,
                    help='rescale for xy plane')
parser.add_argument('--C1_penalty', type=float, default=10.,
                    help='penalty row/column')
parser.add_argument('--topk', type=int, default=10,
                    help='topk')
parser.add_argument('--stepsize', type=int, default=20,
                    help='step size')
parser.add_argument('--diag_loss', type=float, default=0.1,
                    help='penalty on the diag')
parser.add_argument('--model', type=str, default='scat')
parser.add_argument('--device', type=int, default=1)
args = parser.parse_args()
device = args.device
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(device)
np.random.seed(args.seed)
### load train instance

tsp_instances = np.load('./data/train_tsp_instance_%d.npy' % args.num_of_nodes)
NumofTestSample = tsp_instances.shape[0]

Std = np.std(tsp_instances, axis=1)
Mean = np.mean(tsp_instances, axis=1)

tsp_instances = tsp_instances - Mean.reshape((NumofTestSample, 1, 2))
# tsp_instances = np.divide(tsp_instances,Std.reshape((NumofTestSample,1,2)))
tsp_instances = args.rescale * tsp_instances  # 2.0 is the rescale
tsp_sols = np.load('./data/train_tsp_sol_%d.npy' % args.num_of_nodes)

dataset_scale = 1
LENGDATA = tsp_instances.shape[0]
total_samples = int(np.floor(LENGDATA * dataset_scale))

preposs_time = time.time()

from models import GNN
from diffusion_model import DGM
from diffusion_gpr import Diffusion_GPR, Diffusion_GAD
from diffusion_agdn import Encoder_AGD

if args.model == 'scat':
    model = GNN(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
elif args.model == 'dgm':
    model = DGM(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
elif args.model == 'gpr':
    model = Diffusion_GPR(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
elif args.model == 'gad':
    model = Diffusion_GAD(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
elif args.model == 'agd':
    model = Encoder_AGD(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
else:
    raise NotImplementedError(f'Not Implemented {args.model}')

# scattering model
from scipy.spatial import distance_matrix


### count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Total number of parameters:')
print(count_parameters(model))


# dis_mat = distance_matrix(tsp_instances[0],tsp_instances[0])
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html
def coord_to_adj(coord_arr):
    dis_mat = distance_matrix(coord_arr, coord_arr)
    return dis_mat


tsp_instances_adj = np.zeros((LENGDATA, args.num_of_nodes, args.num_of_nodes))
for i in range(LENGDATA):
    tsp_instances_adj[i] = coord_to_adj(tsp_instances[i])


# print(coord_to_adj(tsp_instances[0]))
class TSP_Dataset(Dataset):
    def __init__(self, coord, data, targets):
        self.coord = torch.FloatTensor(coord)
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        xy_pos = self.coord[index]
        x = self.data[index]
        y = self.targets[index]
        #        tsp_instance = Data(coord=x,sol=y)
        return tuple(zip(xy_pos, x, y))

    def __len__(self):
        return len(self.data)


dataset = TSP_Dataset(tsp_instances, tsp_instances_adj, tsp_sols)
# num_trainpoints = int(np.floor(0.6*total_samples))
num_trainpoints = 2000
num_valpoints = total_samples - num_trainpoints
# num_testpoints = total_samples - (num_trainpoints + num_valpoints)
sctdataset = dataset
traindata = sctdataset[0:num_trainpoints]
# traindata = sctdataset[0:1000]
# valdata = sctdataset[num_trainpoints:]
valdata = sctdataset[num_trainpoints:]
batch_size = args.batch_size
train_loader = DataLoader(traindata, batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(valdata, batch_size, shuffle=False, num_workers=8)

from torch.optim.lr_scheduler import StepLR

# optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.wdecay)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
scheduler = StepLR(optimizer, step_size=args.stepsize, gamma=0.8)
model.cuda(device)


def train(epoch):
    scheduler.step()
    model.train()
    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for batch in progress_bar:
        f0 = batch[0].cuda(device)
        distance_m = batch[1].cuda(device)
        output = model(f0, distance_m)
        TSPLoss_constaint, Heat_mat = TSPLoss(SctOutput=output, distance_matrix=distance_m,
                                              num_of_nodes=args.num_of_nodes)
        Heat_mat_diagonals = [torch.diagonal(mat) for mat in Heat_mat]
        Heat_mat_diagonals = torch.stack(Heat_mat_diagonals, dim=0)
        Nrmlzd_constraint = (1. - torch.sum(output, 2)) ** 2
        Nrmlzd_constraint = torch.sum(Nrmlzd_constraint)
        loss = args.C1_penalty * Nrmlzd_constraint + 1. * torch.sum(TSPLoss_constaint) + args.diag_loss * torch.sum(
            Heat_mat_diagonals)
        batchloss = torch.sum(loss) / len(batch[0])
        progress_bar.set_postfix(loss=batchloss.item())
        total_loss += batchloss.item()
        num_batches += 1
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    # 在每个 epoch 结束后单独一行打印平均 Loss
    avg_loss = total_loss / num_batches
    tqdm.write(f'Epoch {epoch} Average Loss: {avg_loss:.5f}')


for i in trange(1, args.EPOCHS + 1, desc='Training Epochs'):
    train(i)
    if (i >= 200) and (i % 10 == 0):
        torch.save(model.state_dict(),
                   f'Saved_Models/TSP_{args.num_of_nodes}/{args.model}_layer_{args.nlayers}_hid_{args.hidden}_{i}_temp_{args.temperature}.pth')
