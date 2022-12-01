#%%
import torch
from deepsnap.hetero_graph import HeteroGraph
import networkx as nx
from torch_sparse import SparseTensor
from collections import Counter
import pandas as pd

def get_prediction(model,H:HeteroGraph) -> dict:
    prediction = {}
    scores = {}
    with torch.no_grad():
        preds = model(H)
        for key,pred in preds.items():
            logits = torch.sigmoid(pred)
            pred_label = torch.round(logits)
            prediction[key] = pred_label
            scores[key] = torch.round(logits,decimals=3)
    return prediction,scores

def get_edge_data(H:HeteroGraph, type:tuple) -> dict:
    """(n1,n2): {attr dict}"""
    edges = list(H.G.edges(data=True))
    edge_mapping = H.edge_to_graph_mapping[type].tolist()
    #data = {(edges[idx][0], edges[idx][1]):edges[idx][2] for j,idx in edge_mapping}
    data = {j:{"graph_idx":idx, "edge": (edges[idx][0],edges[idx][1]), "attr":edges[idx][2]} for j,idx in enumerate(edge_mapping)}
    return data

def get_node_data(H:HeteroGraph, type:str) -> dict:
    """{tensor idx: {attribute dict})}"""
    nodes = dict(H.G.nodes(data=True))
    node_mapping = H.node_to_graph_mapping[type].tolist()
    data = {j:{"data":nodes[idx], "graph_idx":idx} for j,idx in enumerate(node_mapping)}
    return data

# def check_same_edges(H:HeteroGraph, type1:tuple, type2:tuple) -> bool:
#     edges1 = list(get_edge_data(H,type1).keys())
#     edges2 = list(get_edge_data(H,type2).keys())
#     reversed_1 = [tuple(reversed(edge)) for edge in edges1]
#     return set(reversed_1) == set(edges2)

def tensor_to_edgelist(tensor: torch.tensor):
  "Toma un edge_index de shape (2,num_edges) y devuelve una lista de tuplas"
  sources = tensor[0,:].tolist()
  targets = tensor[1,:].tolist()
  edgelist = list(zip(sources,targets))
  return edgelist

def init_node_features(G, mode, size):
  """ Para usar con nx.Graph """
  if mode == "ones":
    feature = torch.ones(size)
    nx.set_node_attributes(G, feature, 'node_feature')
  elif mode == "random":
    feature_dict = {}
    for node in list(G.nodes()):
      feature_dict[node] = torch.rand(size)
    nx.set_node_attributes(G,feature_dict,'node_feature')

def edgeindex_to_sparsematrix(het_graph: HeteroGraph) -> dict : 
    sparse_edge_dict = {}
    for key, index in het_graph.edge_index.items():
        # adj = to_torch_coo_tensor(index).long()
        from_size = het_graph.num_nodes()[key[0]]
        to_size = het_graph.num_nodes()[key[2]]
        adj = SparseTensor.from_edge_index(index,sparse_sizes=(from_size,to_size))
        sparse_edge_dict[key] = adj.t()
    return sparse_edge_dict

def count_edgetypes(G):
  edges = list(G.edges(data=True))
  nodes = dict(G.nodes(data=True))
  typed = [(nodes[edge[0]]["node_type"],edge[2]["edge_type"],nodes[edge[1]]["node_type"]) for edge in edges]
  counts = Counter(typed)

  return counts

def check_symmetry_test(G):
  counts = count_edgetypes(G)
  is_symmetric = []
  failed = []
  for key, val in counts.items():
    inverse_key = tuple(reversed(key))
    test = counts[inverse_key] == val
    is_symmetric.append(test)
    if ~test:
      failed.append(key)
  
  assert(all(is_symmetric)), f"The following edges are not symmetric:{failed}"

def check_symmetry_print(G):
  counts = count_edgetypes(G)
  print("Edge type -----------  Is symmetric \n")
  for key, val in counts.items():
    inverse_key = tuple(reversed(key))
    test = counts[inverse_key] == val
    print(f"{key} ------- {test}")
  
def init_node_feature_map(G, mode, size):
  """ Para usar con nx.Graph """
  if mode == "ones":
    feature = torch.ones(size)
    feature_dict = {node:feature for node in list(G.nodes())}
  elif mode == "random":
    feature_dict = {}
    for node in list(G.nodes()):
      feature_dict[node] = torch.rand(size)
  return feature_dict

def generate_features(G,modes_sizes:dict) -> dict:
  feature_dict = {}
  for mode,sizes in modes_sizes.items():
    for size in sizes:
      feature_map = init_node_feature_map(G,mode,size)
      feature_dict[(mode,size)] = feature_map
  return feature_dict

def init_multiple_features(G,modes_sizes:dict) -> dict:
  for mode,sizes in modes_sizes.items():
    for size in sizes:
      feature_map = init_node_feature_map(G,mode,size)
      nx.set_node_attributes(G,feature_map,f"{mode}_{size}")

def set_feature(split,init_mode,feature_length):
  feature_key = f"{init_mode}_{feature_length}"
  split.__setattr__("node_feature",split[feature_key])


def map_prediction_to_edges(prediction:dict,H:HeteroGraph, edge_to_label = True) -> dict:
  """Dada una predicción, devuelve un diccionario que mapea el edge (u,v) a la etiqueta predecida.
  El mapa corresponde a los enlaces en el dataset usado: si se predijo sobre val, pasar dataset val al argument
  esto es importante porque los indices no se conservan en los splits
  if edge_to_label el dict es {(u,v):label}, sino es {label:[(u,v),...]}"""

  prediction_map = {}
  for edge_type,pred in prediction.items():
    predicted_labels = pred.tolist()
    labeled_edges = tensor_to_edgelist(H.edge_label_index[edge_type])
    pred_map = {edge:label for edge,label in zip(labeled_edges,predicted_labels)}
    if edge_to_label:
      prediction_map[edge_type] = pred_map
    else:
      positives = [edge for edge,val in pred_map.items() if val==1]
      negatives = [edge for edge,val in pred_map.items() if val==0]
      prediction_map[edge_type] = {1:positives, 0:negatives}
  return prediction_map

def get_prediction_dataframe(scores,H)-> dict:
  """Dado un tensor de predicción, devuelve un dataframe con un enlace por fila y su info.
  El tensor de input es una predicción en formato torch.tensor([1,0,1...]). Los índices de ese tensor corresponden a los indices del enlace.
  Esos índices dependen del grafo del cual salieron, por eso hay que también pasar el dataset como argumento
  Esta función traduce los indices a enlaces y busca sus datos, devuelve todo en un dataframe"""

  results = {}
  edge_to_label_dict = map_prediction_to_edges(scores,H)
  for edgetype,pred in edge_to_label_dict.items():
    src_type, relation, trg_type = edgetype
    src_info = get_node_data(H,src_type)
    trg_info = get_node_data(H,trg_type)
    for edge, label in pred.items():
      src,trg = edge
      results[edge] = {"type":relation,"source_index":src_info[src]["data"]["node_dataset_index"],"target_index":trg_info[trg]["data"]["node_dataset_index"],"source_type":src_info[src]["data"]["node_type"],"target_type":trg_info[trg]["data"]["node_type"],"source_name":src_info[src]["data"]["node_name"], "target_name":trg_info[trg]["data"]["node_name"] ,"score":label}
  
  frame = pd.DataFrame.from_dict(results, orient="index")
  frame["label"] = frame["score"].round()
  frame = frame.sort_values(by="score",ascending=False)

  return frame