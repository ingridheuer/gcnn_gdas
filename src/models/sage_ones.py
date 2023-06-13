import torch
from torch_geometric.nn import SAGEConv, to_hetero

class encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = SAGEConv(-1,32,aggr="sum")
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.conv1_act = torch.nn.LeakyReLU()

        self.conv2 = SAGEConv(32,32,aggr="sum")
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.conv2_act = torch.nn.LeakyReLU()

        self.post_linear = torch.nn.Linear(32,32)
    
    def forward(self,x:dict,edge_index:dict) -> dict:
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv1_act(x)
        x = torch.nn.functional.normalize(x,2,-1)

        x = self.conv2(x,edge_index)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.conv2_act(x)
        x = torch.nn.functional.normalize(x,2,-1)

        x = self.post_linear(x)

        return x

class inner_product_decoder(torch.nn.Module):
    def forward(self,x:dict,edge_label_index:dict,supervision_types,apply_sigmoid=True) -> dict:
        pred_dict = {}
        for edge_type in supervision_types:
            edge_index = edge_label_index[edge_type]

            source_type, _ , target_type = edge_type
            
            x_source = x[source_type]
            x_target = x[target_type]

            source_index, target_index = edge_index[0], edge_index[1]

            nodes_source = x_source[source_index]
            nodes_target = x_target[target_index]

            pred = (nodes_source * nodes_target).sum(dim=1)

            if apply_sigmoid:
                pred = torch.sigmoid(pred)

            pred_dict[edge_type] = pred
        
        return pred_dict

class Model(torch.nn.Module):
    def __init__(self,metadata,supervision_types):
        super().__init__()
          
        self.encoder = to_hetero(encoder(),metadata,aggr="sum")
        self.decoder = inner_product_decoder()
        self.loss_fn = torch.nn.BCELoss()
        self.supervision_types = supervision_types
    
    
    def forward(self,x:dict,edge_index:dict,edge_label_index:dict,return_tensor=None) -> dict:
        x = self.encoder(x,edge_index)
        pred = self.decoder(x,edge_label_index,self.supervision_types)

        if return_tensor is not None:
            pred = pred[self.supervision_types[return_tensor]]

        return pred
    
    def loss(self, prediction_dict, label_dict):
        loss = 0
        for edge_type,pred in prediction_dict.items():
            y = label_dict[edge_type]
            loss += self.loss_fn(pred, y.type(pred.dtype))
        return loss