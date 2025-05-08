# All of the code in this file is taken from or adapted from:
# https://github.com/VinceThi/low-rank-hypothesis-complex-systems/tree/v1.0.0
#
# which is the code accompanying the following paper:
# 
# @article{thibeault2024low,
#   title={The low-rank hypothesis of complex systems},
#   author={Thibeault, Vincent and Allard, Antoine and Desrosiers, Patrick},
#   journal={Nature Physics},
#   volume={20},
#   number={2},
#   pages={294--302},
#   year={2024},
#   publisher={Nature Publishing Group UK London}
# }
#
# Please credit the original authors if you use this code.

import networkx as nx
import numpy as np

### qmf model with and without perturbations, and jacobians
def qmf_sis(t, x, W, coupling, D):
    return -D@x + (1 - x)*(coupling*W@x)

def qmf_sis_pert(t, x, W, coupling, D, pert):
    #return -D@x + (1 - x)*(coupling*W@x+pert(t))
    return -D @ x + (1 - x) * (coupling * W @ x)+ pert(t)*x*(1-x)

def jacobian_x_SIS(x, W, coupling, D):
    return -D - coupling*np.diag(W@x)

def jacobian_y_SIS(x, coupling):
    return coupling*(np.eye(len(x)) - np.diag(x))


# Weight matrix for epidemiological graph
def get_epidemiological_weight_matrix(graph_str):
    path_str = f"./"+graph_str+"/"

    if graph_str == "high_school_proximity":
        # Taken from Netzschleuder : https://networks.skewed.de/
        G = nx.read_edgelist(path_str + "edges_no_time.csv", delimiter=',',
                             create_using=nx.Graph)
        A = nx.to_numpy_array(G)

    else:
        raise ValueError("This graph_str epidemiological is not an option. "       
                         "See the documentation of"
                         "get_epidemiological_weight_matrix")

    return A