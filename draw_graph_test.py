import numpy as np
from other_models import *
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from LocalCapsulePoolingNetwork import LocalCapsulePoolingNetwork
from data_processing import get_dataset
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import dense_to_sparse

parser = argparse.ArgumentParser(description='Draw Graph')
parser.add_argument('-data', dest='dataset', default='PROTEINS', type=str,
                    help='dataset type')
parser.add_argument('-model', dest='model', default='LocalCapsulePoolingNetwork', type=str,
                    help='model to test')
parser.add_argument('-seed', dest='seed', type=int, default=7, help='seed')
parser.add_argument('-hidden', dest='hidden', type=int, default=128, help='hidden size')

parser.add_argument("-gpu", dest='gpu', type=int, default=0)
parser.add_argument("-ratio", dest='ratio', default=0.5, help='Pool Ratio')
parser.add_argument("-dropout_att", dest='dropout_att', default=0.5, help='Pool Ratio')
parser.add_argument("-graph_id", dest='graph_id', default=6, help='which graph to plot')
args = parser.parse_args()

if args.gpu != '-1' and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 在dataset的默认顺序中，找到想要的某个graph
def get_data(data_num, loader):
    # 给定id，找到对应id的graph
    print(f"loader {loader}")
    iterator = iter(loader)
    for i in range(int(data_num)+1):
        data = next(iterator)
    return data

def addModel(model_name):
    """
    根据model_name 加载对应的模型
    :param model_name:
    :return:
    """
    if model_name == 'LocalCapsulePoolingNetwork':
        model = LocalCapsulePoolingNetwork(
            dataset=dataset,
            num_layers=3,
            hidden=args.hidden,
            ratio=args.ratio,
            dropout_att=args.dropout_att,
            dataset_name=args.dataset
        )

    elif model_name == 'MinCutPool':
        model = MinCutPool(
            dataset=dataset,
            hidden=args.hidden,
            ratio=args.ratio,
            dropout_att=args.dropout_att
        )
    elif model_name == 'DiffPool':
        model = DiffPool(
            dataset=dataset,
            hidden=args.hidden,
            ratio=args.ratio,
            dropout_att=args.dropout_att
        )

    elif model_name == 'SAGPool':
        model = SAGPool(
            dataset=dataset,
            hidden=args.hidden,
            ratio=args.ratio,
            dropout_att=args.dropout_att
        )
    elif model_name == 'TopKPool':
        model = TopKPool(
            dataset=dataset,
            hidden=args.hidden,
            ratio=args.ratio,
            dropout_att=args.dropout_att
        )
    elif model_name == 'ASAP':
        model = ASAP_Pool(
            dataset=dataset,
            num_layers=3,
            hidden=args.hidden,
            ratio=args.ratio,
            dropout_att=args.dropout_att
        )
    else:
        raise NotImplementedError
    model.to(args.device).reset_parameters()
    return model

def load_model(load_path):
    """
    在空的模型架构中导入模型参数
    :param load_path:
    :return:
    """
    state = torch.load(load_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

def plt_graph(edge_index, seed=7):
    """给定sparse 类型的edge_index，返回用于绘图的nx数据"""
    edge_index_np = edge_index.cpu().numpy().tolist()
    edge_lst = []
    # 将edge_index转换成坐标对的形式
    for item in zip(edge_index_np[0], edge_index_np[1]):
        edge_lst.append(item)
    G = nx.Graph()
    G.add_edges_from(edge_lst)
    pos = nx.spring_layout(G, seed=seed)
    return G, pos

def data_transfer(adj):
    """
    将dense类型的adj转换成sparse类型的edge_index
    :param adj:
    :return:
    """
    # shape of adj: (1, max_num_nodes, max_num_nodes)
    # print(f"-------------------------------")
    # print(f"adj {adj}")
    edge_index, _ = dense_to_sparse(adj)
    # print(f"edge_index {edge_index}")
    return edge_index, _

model_lst = ['DiffPool', 'MinCutPool', 'TopKPool', 'SAGPool', 'ASAP', 'LocalCapsulePoolingNetwork']
color_dict = {
'DiffPool': 'orangered',
'MinCutPool': 'coral',
'TopKPool': 'yellow',
'SAGPool': 'chartreuse',
'ASAP':'mediumorchid',
'LocalCapsulePoolingNetwork':'deeppink'
}
plt.figure(figsize=(4*7, 4))
i = 0
for model_name in model_lst:
    i += 1
    # 根据不同的model_name 判断是否需要dense类型的data
    if model_name in ['LocalCapsulePoolingNetwork', 'SAGPool', 'TopKPool', 'ASAP']:
        sparse = True
    elif model_name in ['MinCutPool', 'DiffPool']:
        sparse = False
    else:
        print(f"Error Model")
    dataset = get_dataset(args.dataset, sparse=sparse)

    if model_name in ['LocalCapsulePoolingNetwork', 'SAGPool', 'TopKPool', 'ASAP']:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    elif model_name in ['MinCutPool', 'DiffPool']:
        # 用dataloader加载数据
        # 每128个图组成一个batch
        loader = DenseLoader(dataset, batch_size=1, shuffle=False)
    else:
        loader = None

    model = addModel(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    load_path = 'torch_saved/' + '{}_{}_run'.format(args.dataset, model_name)

    try:
        # 加载模型
        # model.load_state_dict(torch.load("test_0c111351_DD"))
        load_model(load_path)
        print(f"load_path {load_path}")
    except:
        print("Can't load the model")
        assert False

    # 将同一个graph送入每个训练好的model中，并且在一个plt中绘制图
    model.eval()
    with torch.no_grad():
        model = model.to(args.device)
        # data = get_data(52, loader).to(args.device)
        # ToDo 这里需要测试一下
        data = get_data(args.graph_id, loader).to(args.device)
        # print(f"data {data}")
        # 这边就直接得到了graph的特征，好像不行
        # 理想中得到的应该是经过3层50%池化后的X和A
        graph_representation, aux_loss, pooled_x, pooled_edge_index = model(data)

        if model_name in ['MinCutPool', 'DiffPool']:
            pooled_edge_index, _ = data_transfer(pooled_edge_index.squeeze())
        plt.subplot(1, len(model_lst)+1, i+1)
        G, pos = plt_graph(pooled_edge_index, seed=args.seed)
        nx.draw(G, pos, node_color=color_dict[model_name], node_size=20,  width=0.1)
        plt.axis()
        plt.title(model_name, y=-0.1)

# 在第一个subfigure 绘制original graph
dataset = get_dataset(args.dataset, sparse=True)
data = dataset[int(args.graph_id)]
original_edge_index = data.edge_index

plt.subplot(1, len(model_lst)+1, 1)
G, pos = plt_graph(original_edge_index, seed=args.seed)
nx.draw(G, pos, node_color=[0.3, 0.5, 0.8], node_size=20, width=0.5)
plt.axis()
plt.title('Original', y=-0.1)
# plt.tight_layout()
plt.show()
plt.savefig("./graph_pictures/pooled_graph_{}_{}.jpg".format(args.dataset, args.graph_id))