import torch
import torch.nn as nn

class distmult_head(torch.nn.Module):
  def __init__(self, hetero_graph, hidden_size):
    super().__init__()
    self.R_weights = nn.ParameterDict()

    for edge_type in hetero_graph.edge_types:
      self.R_weights[edge_type] = nn.Parameter(torch.rand(hidden_size,hidden_size)*0.01)
  
  def score(self,x,edge_label_index):
    scores = {}
    for message_type in edge_label_index:
      src_type,edge_type,trg_type = message_type[0], message_type[1], message_type[2]
      rel_weights = self.R_weights[edge_type]
      nodes_left = torch.index_select(x[src_type], 0, edge_label_index[message_type][0,:].long())
      nodes_right = torch.index_select(x[trg_type], 0, edge_label_index[message_type][1,:].long())
      mid_product = nodes_right@rel_weights
      scores[message_type] = torch.sum(mid_product@torch.t(nodes_left) , dim=-1)
    return scores
  