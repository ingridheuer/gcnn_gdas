import torch
from deepsnap.hetero_graph import HeteroGraph
import pandas as pd

def get_prediction(model,H:HeteroGraph) -> dict:
    prediction = {}
    with torch.no_grad():
        preds = model(H)
        for key,pred in preds.items():
            logits = torch.sigmoid(pred)
            pred_label = torch.round(logits)
            prediction[key] = pred_label
    return prediction

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

def check_same_edges(H:HeteroGraph, type1:tuple, type2:tuple) -> bool:
    edges1 = list(get_edge_data(H,type1).keys())
    edges2 = list(get_edge_data(H,type2).keys())
    reversed_1 = [tuple(reversed(edge)) for edge in edges1]
    return set(reversed_1) == set(edges2)

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

def map_prediction_to_edges(prediction:torch.tensor,H:HeteroGraph, edge_to_label = True) -> dict:
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

def get_edge(data:pd.DataFrame,edge:tuple):
    row = edge_data[(data.a_idx == edge[0]) & (data.b_idx == edge[1])]
    if row.empty:
      row = edge_data[(data.a_idx == edge[1]) & (data.b_idx == edge[0])]
      if row.empty:
        print("Edge does not exist in dataframe")
    return row

def get_prediction_data_dict(prediction,dataset,nodes_data)-> dict:
  """input indexado con tensor indexes"""
  results = {}
  edge_to_label_dict = map_prediction_to_edges(prediction,dataset)
  for edgetype,pred in edge_to_label_dict.items():
    src_type = edgetype[0]
    trg_type = edgetype[2]
    src_info = nodes_data[src_type]
    trg_info = nodes_data[trg_type]
    edgetype_results = {edge:{"source_data":src_info[edge[0]], "target_data":trg_info[edge[1]], "label":label} for edge,label in pred.items()}
    results[edgetype] = edgetype_results
  return results

def get_prediction_dataframe(prediction,dataset)-> dict:
  """input indexado con tensor indexes"""
  #TODO: agregar una columna de score
  results = {}
  edge_to_label_dict = map_prediction_to_edges(prediction,dataset)
  for edgetype,pred in edge_to_label_dict.items():
    src_type, relation, trg_type = edgetype
    src_info = get_node_data(dataset,src_type)
    trg_info = get_node_data(dataset,trg_type)
    for edge, label in pred.items():
      src,trg = edge
      results[edge] = {"type":relation,"source_idx":src_info[src]["data"]["node_dataset_idx"],"target_idx":trg_info[trg]["data"]["node_dataset_idx"],"source_type":src_info[src]["data"]["node_type"],"target_type":trg_info[trg]["data"]["node_type"],"source_name":src_info[src]["data"]["node_name"], "target_name":trg_info[trg]["data"]["node_name"] ,"label":label}
    frame = pd.DataFrame.from_dict(results, orient="index")
  return frame

def edgeindex_to_sparsematrix(het_graph: HeteroGraph) -> dict : 
    sparse_edge_dict = {}
    for key in het_graph.edge_index:
        temp_edge_index = het_graph.edge_index[key]
        from_type = key[0]
        to_type = key[2]
        adj = SparseTensor(row=temp_edge_index[0], col=temp_edge_index[1], sparse_sizes=(het_graph.num_nodes(from_type), het_graph.num_nodes(to_type)))
        sparse_edge_dict[key] = adj.t()
    return sparse_edge_dict