import torch
from torch_geometric.nn import SAGEConv, to_hetero

class GraphBlock(torch.nn.Module):
    def __init__(self,input_dim,output_dim,dropout,residual_block):
        super().__init__()

        self.residual_block = residual_block
        self.conv = SAGEConv(input_dim,output_dim,aggr="mean")
        self.bn = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self,x,edge_index):
        identity = x
        out = self.conv(x,edge_index)
        out = self.bn(out)
        out = self.dropout(out)
        out = self.relu(out)

        if self.residual_block:
            out += identity
        
        return out

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.bn.reset_parameters()

    
class PostProcessMLP(torch.nn.Module):
    def __init__(self,input_dim,output_dim,dropout):
        super().__init__()

        self.post_linear_1 = torch.nn.Linear(input_dim,output_dim)
        self.post_linear_bn = torch.nn.BatchNorm1d(output_dim)
        self.post_linear_dropout = torch.nn.Dropout(dropout)
        self.post_linear_act = torch.nn.LeakyReLU()
        self.post_linear_2 = torch.nn.Linear(output_dim,output_dim)

    def forward(self,x):
        x = self.post_linear_1(x)
        x = self.post_linear_bn(x)
        x = self.post_linear_dropout(x)
        x = self.post_linear_act(x)
        x = self.post_linear_2(x)

        return x
    
    def reset_parameters(self):
        self.post_linear_1.reset_parameters()
        self.post_linear_bn.reset_parameters()
        self.post_linear_2.reset_parameters()

class Encoder(torch.nn.Module):
    def __init__(self,metadata):
        super().__init__()
        output_dim = 64
        self.graph_layer_1 = to_hetero(GraphBlock(-1,output_dim,0.4,False),metadata,aggr="mean")
        self.graph_layer_2 = to_hetero(GraphBlock(output_dim,output_dim,0.1,True),metadata,aggr="mean")
        self.graph_layer_3 = to_hetero(GraphBlock(output_dim,output_dim,0.1,True),metadata,aggr="mean")
        self.post_mlp = to_hetero(PostProcessMLP(output_dim,output_dim,0.1),metadata)

    
    def forward(self,x:dict,edge_index:dict):
        x = self.graph_layer_1(x,edge_index)
        x = self.graph_layer_2(x,edge_index)
        x = self.graph_layer_3(x,edge_index)
        x = self.post_mlp(x)
        return x

class InnerProductDecoder(torch.nn.Module):
    def __init__(self,supervision_types):
        super().__init__()
        self.supervision_types = supervision_types

    def forward(self,x:dict,edge_label_index:dict) -> dict:
        pred_dict = {}
        for edge_type in self.supervision_types:
            edge_index = edge_label_index[edge_type]

            source_type, _ , target_type = edge_type
            
            x_source = x[source_type]
            x_target = x[target_type]

            source_index, target_index = edge_index[0], edge_index[1]

            nodes_source = x_source[source_index]
            nodes_target = x_target[target_index]

            pred = (nodes_source * nodes_target).sum(dim=1)

            pred = torch.sigmoid(pred)
            pred_dict[edge_type] = pred
        
        return pred_dict

class Model(torch.nn.Module):
    def __init__(self,metadata,supervision_types):
        super().__init__()
          
        self.encoder = Encoder(metadata)
        self.decoder = InnerProductDecoder(supervision_types)
        self.loss_fn = torch.nn.BCELoss()
        self.supervision_types = supervision_types
    
    
    def forward(self,x:dict,edge_index:dict,edge_label_index:dict) -> dict:
        x = self.encoder(x,edge_index)
        pred = self.decoder(x,edge_label_index)
        return pred
    
    def loss(self, prediction_dict, label_dict):
        loss = 0
        for edge_type,pred in prediction_dict.items():
            y = label_dict[edge_type]
            loss += self.loss_fn(pred, y.type(pred.dtype))
        return loss