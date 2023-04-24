import torch
from torch_geometric.nn import SAGEConv,GATConv, to_hetero

class inner_product_decoder(torch.nn.Module):
    def forward(self,x_source,x_target,edge_index,apply_sigmoid=True):
        nodes_src = x_source[edge_index[0]]
        nodes_trg = x_target[edge_index[1]]
        pred = (nodes_src * nodes_trg).sum(dim=-1)

        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        return pred

class base_message_layer(torch.nn.Module):

    def __init__(self, model_params,hidden_layer=True):
        super().__init__()

        # Currently SageConv or GATConv, might have to modify this to support other Convs
        conv_type = model_params["conv_type"]
        self.conv = layer_dict[conv_type]((-1,-1), model_params["hidden_channels"],aggr=model_params["micro_aggregation"],add_self_loops=False)
        self.normalize = model_params["L2_norm"]

        post_conv_modules = []
        if model_params["batch_norm"]:
            bn = torch.nn.BatchNorm1d(model_params["hidden_channels"])
            post_conv_modules.append(bn)
        
        if model_params["dropout"] > 0:    
            dropout = torch.nn.Dropout(p=model_params["dropout"])
            post_conv_modules.append(dropout)
        
        # No activation on final embedding layer
        if hidden_layer:
            activation = model_params["activation"]()
            post_conv_modules.append(activation)
        
        self.post_conv = torch.nn.Sequential(*post_conv_modules)

    def forward(self, x:dict, edge_index:dict) -> dict:
        x = self.conv(x,edge_index)
        x = self.post_conv(x)
        if self.normalize:
            x = torch.nn.functional.normalize(x,2,-1)
        return x

class multilayer_message_passing(torch.nn.Module):
    #TODO: consider input and output dims with skipcat. Currently the two supported convs auto-detect dimensions. Might have to modify this if i add more convs in the future.
    def __init__(self,num_layers,model_params,metadata):
        super().__init__()

        self.skip = model_params["layer_connectivity"]
        self.num_layers = num_layers

        for i in range(self.num_layers):
            hidden_layer = i != self.num_layers-1
            layer = to_hetero(base_message_layer(model_params,hidden_layer),metadata,model_params["macro_aggregation"])
            self.add_module(f"Layer_{i}",layer)
    
    def hetero_skipsum(self,x: dict, x_i:dict) -> dict:
        x_transformed = {}
        for key,x_val in x.items():
            x_i_val = x_i[key]
            transformed_val = x_val + x_i_val
            x_transformed[key] = transformed_val

        return x_transformed

    def hetero_skipcat(self,x: dict, x_i:dict) -> dict:
        x_transformed = {}
        for key,x_val in x.items():
            x_i_val = x_i[key]
            transformed_val = torch.cat([x_val,x_i_val],dim=-1)
            x_transformed[key] = transformed_val

        return x_transformed
    
    def forward(self, x:dict, edge_index:dict) -> dict:
        for i, layer in enumerate(self.children()):
            x_i = x
            x = layer(x,edge_index)
            if self.skip == "skipsum":
                x = self.hetero_skipsum(x,x_i)
            elif self.skip == "skipcat" and i < self.num_layers -1:
                x = self.hetero_skipcat(x,x_i)
        
        return x 

class MLP(torch.nn.Module):
    def __init__(self,num_layers,in_dim,out_dim,model_params,hidden_dim=None):
        super().__init__()

        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        
        modules = []
        if num_layers == 1:
            modules.append(torch.nn.Linear(in_dim,out_dim))
        else:
            for i in range(num_layers):
                final_layer = i == num_layers-1
                first_layer = i == 0
                if first_layer:
                    modules.append(torch.nn.Linear(in_dim,hidden_dim))
                    modules.append(model_params["activation"]())
                elif final_layer:
                    modules.append(torch.nn.Linear(hidden_dim,out_dim))
                else:
                    modules.append(torch.nn.Linear(hidden_dim,hidden_dim))
                    modules.append(model_params["activation"]())
        
        self.model = torch.nn.Sequential(*modules)
    
    def forward(self,x):
        x = self.model(x)
        return x

class base_encoder(torch.nn.Module):
    def __init__(self,model_params,metadata):
        super().__init__()

        self.has_pre_mlp = model_params["pre_process_layers"] > 0
        self.has_post_mlp = model_params["post_process_layers"] > 0

        if self.has_pre_mlp:
            self.pre_mlp = to_hetero(MLP(model_params["pre_process_layers"],model_params["feature_dim"],model_params["hidden_channels"],model_params),metadata)
        
        self.message_passing = multilayer_message_passing(model_params["msg_passing_layers"],model_params,metadata)

        if self.has_post_mlp:
            self.post_mlp = to_hetero(MLP(model_params["post_process_layers"],model_params["hidden_channels"],model_params["hidden_channels"],model_params),metadata)
    
    def forward(self,x:dict,edge_index:dict) -> dict :
        if self.has_pre_mlp:
            x = self.pre_mlp(x)

        x = self.message_passing(x,edge_index)
        
        if self.has_post_mlp:
            x = self.post_mlp(x)

        return x

class base_model(torch.nn.Module):
    def __init__(self, model_params,metadata):
        super().__init__()

        default_model_params = {
            "hidden_channels":32,
            "conv_type":"SAGEConv",
            "batch_norm": True,
            "dropout":0,
            "activation":torch.nn.LeakyReLU,
            "micro_aggregation":"mean",
            "macro_aggregation":"mean",
            "layer_connectivity":None,
            "L2_norm":False,
            "feature_dim": 10,
            "pre_process_layers":0,
            "msg_passing_layers":2,
            "post_process_layers":0,
        }
        
        for arg in default_model_params:
            if arg not in model_params:
                model_params[arg] = default_model_params[arg]
        
        self.encoder = base_encoder(model_params,metadata)
        self.decoder = inner_product_decoder()
        self.loss_fn = torch.nn.BCELoss()
    
    def decode(self,x:dict,edge_label_index:dict,supervision_types):
        pred_dict = {}
        for edge_type in supervision_types:
            edge_index = edge_label_index[edge_type]

            src_type = edge_type[0]
            trg_type = edge_type[2]

            x_src = x[src_type]
            x_trg = x[trg_type]

            pred = self.decoder(x_src,x_trg,edge_index)

            pred_dict[edge_type] = pred
        
        return pred_dict
    
    def encode(self,data):
        x = data.x_dict
        adj_t = data.adj_t_dict

        encodings = self.encoder(x,adj_t)
        return encodings
    
    def forward(self,data,supervision_types):
        x = data.x_dict
        adj_t = data.adj_t_dict
        edge_label_index = data.edge_label_index_dict

        x = self.encoder(x,adj_t)
        pred = self.decode(x,edge_label_index,supervision_types)
        return pred
    
    def loss(self, prediction_dict, label_dict):
        loss = 0
        num_types = len(prediction_dict.keys())
        for edge_type,pred in prediction_dict.items():
            y = label_dict[edge_type]
            loss += self.loss_fn(pred, y.type(pred.dtype))
        return loss/num_types

layer_dict = {
    "GATConv":GATConv,
    "SAGEConv":SAGEConv
}