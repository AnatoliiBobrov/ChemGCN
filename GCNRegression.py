import csv
import json
import time
import itertools
import torch #torch==2.4.1
import torch.nn.functional as F
from torch_geometric.data import Data #torch-geometric==2.6.1
from torch_geometric.nn import GCNConv 
import numpy as np #numpy==1.26.4
from rdkit import Chem #rdkit==2023.9.5
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondDir as BD
from rdkit.Chem.rdchem import ChiralType as CT
from rdkit.Chem.rdchem import HybridizationType as HT
from sklearn.model_selection import train_test_split #scikit-learn==1.4.2

CHIRAL = [
    CT.CHI_ALLENE,
    CT.CHI_OCTAHEDRAL,
    CT.CHI_OTHER,
    CT.CHI_SQUAREPLANAR,
    CT.CHI_TETRAHEDRAL,
    CT.CHI_TETRAHEDRAL_CCW,
    CT.CHI_TETRAHEDRAL_CW,
    CT.CHI_TRIGONALBIPYRAMIDAL,
    CT.CHI_UNSPECIFIED
]
CHIRAL_LEN = len(CHIRAL)

HYBRIDIZATION = [
    HT.OTHER,
    HT.S,
    HT.SP,
    HT.SP2,
    HT.SP2D,
    HT.SP3,
    HT.SP3D,
    HT.SP3D2,
    HT.UNSPECIFIED
]
HYBRIDIZATION_LEN = len(HYBRIDIZATION)

"""
BONDTYPE = [
    BT.AROMATIC,
    BT.DATIVE,
    BT.DATIVEL,
    BT.DATIVEONE,
    BT.DATIVER,
    BT.DOUBLE,
    BT.FIVEANDAHALF,
    BT.FOURANDAHALF,
    BT.HEXTUPLE,
    BT.HYDROGEN,
    BT.IONIC,
    BT.ONEANDAHALF,
    BT.OTHER,
    BT.QUADRUPLE,
    BT.QUINTUPLE,
    BT.SINGLE,
    BT.THREEANDAHALF,
    BT.THREECENTER,
    BT.TRIPLE,
    BT.TWOANDAHALF,
    BT.UNSPECIFIED,
    BT.ZERO
]
BONDTYPE_LEN = len(BONDTYPE)

BONDDIR = [
    BD.BEGINDASH,
    BD.BEGINWEDGE,
    BD.EITHERDOUBLE,
    BD.ENDDOWNRIGHT,
    BD.ENDUPRIGHT,
    BD.NONE,
    BD.UNKNOWN
]
"""

BONDTYPE = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR = [
    BD.NONE,
    BD.ENDUPRIGHT,
    BD.ENDDOWNRIGHT
]
BONDTYPE_LEN = len(BONDTYPE)
BONDDIR_LEN = len(BONDDIR)

DATASETS = [
    ["freesolv.csv", "smiles", "expt", "calc"], 
    ["esol.csv", "smiles", "measured log solubility in mols per litre", "ESOL predicted log solubility in mols per litre"],
    ["qm7.csv", "smiles", "u0_atom", "u0_atom"], ]
DATASET_NUM = 0 #0 freesolv 1 esol 2 qm7
HIDDEN_SIZE = [20 * i for i in range(1, 11)]
LEARNING_RATE = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]
EPOCHS = 100

start = 0
best_index = 0
best_loss = None
params = None
try:
    with open('saved_success.json', 'r') as json_file:
        data = json.load(json_file)
        if data["EPOCHS"] == EPOCHS:
            start = data["start"]
            best_index = data["best_index"]
            best_loss = data["best_loss"]
            params = data["params"]
            print (f"Loaded last success {str(data)}")
        else:
            print ("Loaded data is not for this EPOCHS. Start from the beginning")
except FileNotFoundError:
    pass
    
def set_zero_seed():
    np.random.seed(0)
    torch.manual_seed(0)

def one_hot_encode(current_value, vector_lenght):
    ohe = [0 for i in range (0,vector_lenght)]
    ohe[current_value - 1] = 1
    return ohe

def smiles_to_graph(sm, ohe_edges=False):
    mol = Chem.MolFromSmiles(sm)
    mol = Chem.AddHs(mol)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        feature = one_hot_encode(atom.GetAtomicNum() - 1, 118)
        feature += [atom.GetFormalCharge(),  1 if (atom.GetIsAromatic()) else 0]
        feature += one_hot_encode(CHIRAL.index(atom.GetChiralTag()), CHIRAL_LEN)
        feature += one_hot_encode(HYBRIDIZATION.index(atom.GetHybridization()), HYBRIDIZATION_LEN)
        atom_features.append(feature)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        
        if ohe_edges:
            attrs = one_hot_encode(BONDTYPE.index(bond.GetBondType()), BONDTYPE_LEN)
            attrs += one_hot_encode(BONDDIR.index(bond.GetBondDir()), BONDDIR_LEN)
            attrs += [1 if bond.GetIsConjugated() else 0]
            edge_attrs.extend([attrs, attrs])
        else:
            bond_type = bond.GetBondTypeAsDouble() / 6 + 0.8333
            edge_attrs.extend([bond_type, bond_type])
        
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    if len(edge_indices) == 0:
        edge_attr = None
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def read_dataset(filename, smiles_pos, exp_pos, comp_pos, ohe_edges=False):
    smiles_ = []
    experimental_values_ = []
    computed_values_ = []
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            smiles_.append(smiles_to_graph(row[smiles_pos], ohe_edges))
            experimental_values_.append(float(row[exp_pos]))
            computed_values_.append(float(row[comp_pos]))
    return smiles_, experimental_values_, computed_values_

set_zero_seed()
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, _, computed_energy = train_test_split(*read_dataset(*DATASETS[DATASET_NUM]), test_size=0.2)
Y_TRAIN = [torch.FloatTensor([i]) for i in Y_TRAIN]
TRAIN_SET_LEN = len(Y_TRAIN)
y_test_ = torch.FloatTensor(Y_TEST)
computed_energy = torch.FloatTensor(computed_energy)
Y_TEST = [torch.FloatTensor([i]) for i in Y_TEST]
TEST_LEN_SET = len(Y_TEST)

class GCNConv1(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=121):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.lin1 = torch.nn.Linear(num_node_features, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = torch.mean(x, dim=0)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
def train_and_test(hidden_size, lr):
    set_zero_seed()
    model = GCNConv1(X_TRAIN[0].num_node_features, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(EPOCHS):
        for i in range(TRAIN_SET_LEN):
            optimizer.zero_grad()
            outputs = model(X_TRAIN[i])
            loss = criterion(outputs, Y_TRAIN[i])
            loss.backward()
            optimizer.step()
    
    test_loss = 0
    with torch.no_grad():
        for i in range(TEST_LEN_SET):
            preds = model(X_TEST[i])
            test_loss += criterion(preds, Y_TEST[i]).item()
    return test_loss/TEST_LEN_SET

query = list(itertools.product(HIDDEN_SIZE, LEARNING_RATE))
length = len(query)
print(f"Amoung of hyperparameters' combinations {length}")

def save_success(start, best_index, best_loss, params):
     json_data = {'start': start, 'best_index': best_index, 'best_loss': best_loss, 'params': params, 'EPOCHS': EPOCHS}
     with open('saved_success.json', 'w') as json_file:
        json.dump(json_data, json_file)


#loss = train_and_test(120, 0.0025)
#print(f"Loss: {loss:.4f}")
if best_loss == None:
    best_loss = train_and_test(*query[0])
    params = str(query[0])
    
while start < length:
    begin = time.time()
    loss = train_and_test(*query[start])
    if loss < best_loss:
        best_loss = loss
        best_index = start
        params = str(query[start])
        print (f"New best loss {best_loss:.4f} with params {params}, index {best_index}")
    if start % 10 == 0:
        end = time.time()
        print(f"Excecution time is {end-begin:.2f} seconds, iteration {start}")
        save_success(start+1, best_index, best_loss, params)
    start += 1
save_success(start, best_index, best_loss, params)
