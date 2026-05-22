from pathlib import Path

import math
import numpy as np
import torch
from torch.utils.data import Dataset

tsplib_collections = {
    'eil51': 426,
    'berlin52': 7542,
    'st70': 675,
    'pr76': 108159,
    'eil76': 538,
    'rat99': 1211,
    'kroA100': 21282,
    'kroE100': 22068,
    'kroB100': 22141,
    'rd100': 7910,
    'kroD100': 21294,
    'kroC100': 20749,
    'eil101': 629,
    'lin105': 14379,
    'pr107': 44303,
    'pr124': 59030,
    'bier127': 118282,
    'ch130': 6110,
    'pr136': 96772,
    'pr144': 58537,
    'kroA150': 26524,
    'kroB150': 26130,
    'ch150': 6528,
    'pr152': 73682,
    'u159': 42080,
    'rat195': 2323,
    'd198': 15780,
    'kroA200': 29368,
    'kroB200': 29437,
    'tsp225': 3916,
    'ts225': 126643,
    'pr226': 80369,
    'gil262': 2378,
    'pr264': 49135,
    'a280': 2579,
    'pr299': 48191,
    'lin318': 42029,
    'rd400': 15281,
    'fl417': 11861,
    'pr439': 107217,
    'pcb442': 50778,
    'd493': 35002,
    'u574': 36905,
    'rat575': 6773,
    'p654': 34643,
    'd657': 48912,
    'u724': 41910,
    'rat783': 8806,
    'pr1002': 259045,
    'u1060': 224094,
    'vm1084': 239297,
    'pcb1173': 56892,
    'd1291': 50801,
    'rl1304': 252948,
    'rl1323': 270199,
    'nrw1379': 56638,
    'fl1400': 20127,
    'u1432': 152970,
    'fl1577': 22249,
    'd1655': 62128,
    'vm1748': 336556,
    'u1817': 57201,
    'rl1889': 316536,
    'd2103': 80450,
    'u2152': 64253,
    'u2319': 234256,
    'pr2392': 378032,
    'pcb3038': 137694,
    'fl3795': 28772,
    'fnl4461': 182566,
    'rl5915': 565530,
    'rl5934': 556045,
    'rl11849': 923288,
    'usa13509': 19982859,
    'brd14051': 469385,
    'd15112': 1573084,
    'd18512': 645238
}


def parse_tsplib_name(tsplib_name):
    return "".join(filter(str.isalpha, tsplib_name)), int("".join(filter(str.isdigit, tsplib_name)))


def read_tsplib_file(file_path):
    """
    The read_tsplib_file function reads a TSPLIB file and returns the nodes and name of the problem.

    :param file_path: Specify the path to the file that is being read
    :return: A list of nodes and a name
    """
    properties = {}
    reading_properties_flag = True
    nodes = []

    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            # read properties
            if reading_properties_flag:
                if ':' in line:
                    key, val = [x.strip() for x in line.split(':')]
                    properties[key] = val
                else:
                    reading_properties_flag = False

            # read node coordinates
            else:
                if line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    pass
                else:
                    line_contents = [x.strip() for x in line.split(" ") if x.strip()]
                    _, x, y = line_contents
                    nodes.append([float(x), float(y)])
            line = read_file.readline()

    return nodes, properties["NAME"]


def load_tsplib_file(root, tsplib_name):
    tsplib_dir = "tsplib"
    file_name = f"{tsplib_name}.tsp"
    file_path = root.joinpath(tsplib_dir).joinpath(file_name)
    instance, name = read_tsplib_file(file_path)

    instance = torch.tensor(instance)
    return instance, name


def normalize_tsp_to_unit_board(tsp_instance):
    """
    normalize a tsp instance to a [0, 1]^2 unit board, prefer to have points on both x=0 and y=0
    :param tsp_instance: a (tsp_size, 2) tensor
    :return: a (tsp_size, 2) tensor, a normalized tsp instance
    """
    normalized_instance = tsp_instance.clone()
    normalization_factor = (normalized_instance.max(dim=0).values - normalized_instance.min(dim=0).values).max()
    normalized_instance = (normalized_instance - normalized_instance.min(dim=0).values) / normalization_factor
    return normalized_instance


def normalize_nodes_to_unit_board(nodes):
    return normalize_tsp_to_unit_board(nodes)


def get_dist_matrix(instance):
    size = instance.shape[0]
    x = instance.unsqueeze(0).repeat((size, 1, 1))
    y = instance.unsqueeze(1).repeat((1, size, 1))
    return torch.norm(x - y, p=2, dim=-1)


def calculate_tour_length_by_dist_matrix(dist_matrix, tours):
    # useful to evaluate one/multiple solutions on one (not-extremely-huge) instance
    if tours.dim() == 1:
        tours = tours.unsqueeze(0)
    tour_shifts = torch.roll(tours, shifts=-1, dims=1)
    tour_lens = dist_matrix[tours, tour_shifts].sum(dim=1)
    return tour_lens


def load_tsp_instances(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} does not exist.")

    tsp_instance_list = []
    opt_tour_list = []
    opt_len_list = []

    with open(path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            line_contents = line.strip().split(" | ")
            tsp_instance_string, opt_tour_string, opt_len_string = line_contents

            tsp_instance = []
            for node_string in tsp_instance_string.split(" "):
                node = node_string.split(",")
                tsp_instance.append([float(node[0]), float(node[1])])
            tsp_instance_list.append(np.array(tsp_instance))

            opt_tour = [int(x) for x in opt_tour_string.split(" ")]
            opt_tour_list.append(np.array(opt_tour))

            opt_len_list.append(float(opt_len_string))

    tsp_instances = np.array(tsp_instance_list)
    opt_tours = np.array(opt_tour_list)
    opt_lens = np.array(opt_len_list)

    num = tsp_instances.shape[0]
    size = tsp_instances.shape[1]

    return tsp_instances, opt_tours, opt_lens, size, num


def choose_bsz(size):
    if size <= 200:
        return 64
    elif size <= 1000:
        return 32
    elif size <= 5000:
        return 16
    else:
        return 4


def run_aug(aug, x, aug_num=None, aug_all=False):
    """
    The run_aug function takes in an augmentation type, a batch of images, and two optional arguments.
    The first optional argument is the number of images to augment per batch. The second is whether or not to
    augment all the images in the batch (defaults to False). It then returns a copy of x with some augmented
    images inserted into it.

    :param aug: Select the augmentation to apply
    :param x: Pass in the data
    :param aug_num: Control the number of augmented images in each batch
    :param aug_all: Decide whether to apply the augmentation on all images or only a subset of them
    :return: A tensor with the same size as x, but with some of its values replaced by augmented data
    """
    x_clone = x.clone()
    if aug == 'rotate':
        x_out, _ = Rotate_aug(x)
    elif aug == 'reflect':
        x_out, _ = Reflect_aug(x)
    elif aug == 'mix':
        x_out, _ = mix_aug(x)
    elif aug == 'noise':
        x_out = x + torch.rand(x.size(), device=x.device) * 1e-5
    else:
        x_out = x
    if not aug_all:
        if aug_num is not None:
            x_out[0::aug_num] = x_clone[0::aug_num]
        else:
            x_out[0] = x_clone[0]
    return x_out


## augmentation function

def Scale(X):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0,
    and then dividing all x-values by the maximum value. It does this for both dimensions.

    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    X = X - torch.reshape(torch.min(X, 1).values, (B, 1, 2)).repeat(1, SIZE, 1)  # translate
    ratio_x = torch.reshape(torch.max(X[:, :, 0], 1).values - torch.min(X[:, :, 0], 1).values, (-1, 1))
    ratio_y = torch.reshape(torch.max(X[:, :, 1], 1).values - torch.min(X[:, :, 1], 1).values, (-1, 1))
    ratio = torch.max(torch.cat((ratio_x, ratio_y), 1), 1).values
    ratio[ratio == 0] = 1
    X = X / (torch.reshape(ratio, (B, 1, 1)).repeat(1, SIZE, 2))
    return X, ratio


def Scale_for_vrp(X, num):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0,
    and then dividing all x-values by the maximum value. It does this for both dimensions.

    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    graph = X[:, :num, :]
    min_values = torch.reshape(torch.min(graph, 1).values, (B, 1, 2)).repeat(1, SIZE, 1)
    X = X - min_values  # translate
    ratio_x = torch.reshape(torch.max(graph[:, :, 0], 1).values - torch.min(graph[:, :, 0], 1).values, (-1, 1))
    ratio_y = torch.reshape(torch.max(graph[:, :, 1], 1).values - torch.min(graph[:, :, 1], 1).values, (-1, 1))
    ratio = torch.max(torch.cat((ratio_x, ratio_y), 1), 1).values
    ratio[ratio == 0] = 1
    X = X / (torch.reshape(ratio, (B, 1, 1)).repeat(1, SIZE, 2))
    X[ratio == 0, :, :] = X[ratio == 0, :, :] + min_values[ratio == 0, :, :]
    return X, ratio


def Rotate_aug(X):
    """
    The Rotate_aug function takes in a batch of points and rotates them by a random angle.
    The function also scales the points to be between 0 and 1.

    :param X: Pass the input data to the function
    :return: The rotated point cloud and the ratio of the bounding box
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B, 1), device=device) * 2 * np.pi
    Theta = Theta.repeat(1, SIZE)
    tmp1 = torch.reshape(X[:, :, 0] * torch.cos(Theta) - X[:, :, 1] * torch.sin(Theta), (B, SIZE, 1))
    tmp2 = torch.reshape(X[:, :, 0] * torch.sin(Theta) + X[:, :, 1] * torch.cos(Theta), (B, SIZE, 1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio


def Reflect_aug(X):
    """
    The Reflect_aug function takes in a batch of points and performs the following operations:
        1. Rotate each point by a random angle between 0 and 2pi radians
        2. Reflect each point across the x-axis (i.e., multiply y coordinate by -2)
        3. Add 10 to all coordinates so that no points are negative anymore (this is for convenience)
        4. Scale all coordinates down to be between 0 and 1

    :param X: Pass the data points to the function
    :return: A reflected point cloud and a scale ratio
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B, 1), device=device) * 2 * np.pi
    Theta = Theta.repeat(1, SIZE)
    tmp1 = torch.reshape(X[:, :, 0] * torch.cos(2 * Theta) + X[:, :, 1] * torch.sin(2 * Theta), (B, SIZE, 1))
    tmp2 = torch.reshape(X[:, :, 0] * torch.sin(2 * Theta) - X[:, :, 1] * torch.cos(2 * Theta), (B, SIZE, 1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio


def mix_aug(X):
    """
    The mix_aug function takes in a batch of images and returns the same batch with half of them rotated and half reflected.
    The function also returns the ratio between the number of pixels that are black after augmentation to before augmentation.

    :param X: Pass in the data
    :return: The augmented images and the ratio of the number of augmented images to original ones
    """
    X_out = X.clone()
    X_out[0::2], ratio = Rotate_aug(X[0::2])
    X_out[1::2], ratio = Reflect_aug(X[1::2])
    return X_out, ratio


def compute_tsp_tour_length(x, tour):
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, nb_nodes, 2) batch of tsp tour instances
             tour of size (bsz, nb_nodes) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    bsz = x.shape[0]
    nb_nodes = tour.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    first_cities = x[arange_vec, tour[:, 0], :]  # size(first_cities)=(bsz,2)
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1, nb_nodes):
            current_cities = x[arange_vec, tour[:, i], :]
            L += torch.sum((current_cities - previous_cities) ** 2, dim=1) ** 0.5  # dist(current, previous node)
            previous_cities = current_cities
        L += torch.sum((current_cities - first_cities) ** 2, dim=1) ** 0.5  # dist(last, first node)
    return L


def avg_list(list_object):
    return sum(list_object) / len(list_object) if len(list_object) > 0 else 0


def run_tsplib_test_knn(model, action_k, state_k, path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(path)
    aug = 'mix'
    # main loop
    st1 = []
    st2 = []
    st3 = []
    st4 = []
    tsplib_names = list(tsplib_collections.keys())
    tsplib_names.sort(key=lambda x: parse_tsplib_name(x)[1])

    print(f"Start evaluation...")
    for i in range(len(tsplib_names)):
        name = tsplib_names[i]
        opt_len = tsplib_collections[name]
        _, size = parse_tsplib_name(name)

        # prepare env
        instance, _ = load_tsplib_file(root, name)
        dist_matrix = get_dist_matrix(instance).to(device)

        # normalize instance for tsplib
        normalized_instance = normalize_nodes_to_unit_board(instance)
        size = normalized_instance.size(0)
        bsz = choose_bsz(size)
        normalized_instance = torch.tensor(normalized_instance).float().to(device)
        normalized_instance = normalized_instance.unsqueeze(0)
        normalized_instance = normalized_instance.repeat((bsz, 1, 1))
        X = run_aug(aug, normalized_instance)
        with torch.no_grad():
            tour, _ = model(X, action_k, state_k, choice_deterministic=True)
        length_by_agent = compute_tsp_tour_length(normalized_instance, tour)
        idx = length_by_agent.min(dim=0).indices.item()
        best_tour = tour[idx, :]

        # evaluate tour length
        tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, best_tour).item()
        tour_len = math.ceil(tour_len)
        gap = tour_len / opt_len - 1

        code = [name, size, tour_len, gap]
        if size <= 100:
            st1.append(code)
        elif size <= 1000:
            st2.append(code)
        elif size <= 10000:
            st3.append(code)
        else:
            st4.append(code)
        print(f"Instance {i:4d} {name:10}: model len {tour_len:.3f} to opt {opt_len:.3f} "
              f"-> gap {gap * 100:.3f}%.")

    # conclusion
    print(f"\n\n")
    print(f"TSP 1~100     : {len(st1)} instances, "
          f"gap {avg_list([x[3] for x in st1]) * 100:.3f}%")
    print(f"TSP 101~1000  : {len(st2)} instances, "
          f"gap {avg_list([x[3] for x in st2]) * 100:.3f}%")
    print(f"TSP 1001~10000: {len(st3)} instances, "
          f"gap {avg_list([x[3] for x in st3]) * 100:.3f}%")
    print(f"TSP >10000    : {len(st4)} instances, "
          f"gap {avg_list([x[3] for x in st4]) * 100:.3f}%")


class TSPLibDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        ...

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
