import pandas as pd
import torch

class MappedDataset():
    def __init__(self,heterodata,node_map,prediction_edge_type):
        self.prediction_edge_type = prediction_edge_type
        self.node_map = node_map
        self.edge_dict = self._reverse_map_heterodata(heterodata)
        self.dataframe = self._edge_dict_to_dataframe()
        
    def _reverse_map_tensor(self,tensor,edge_type):
        """Maps edge dictionary from pyg Heterodata back into the original node indexes from the dataframe"""
        #Tensor to lists [sources], [targets]
        sources = tensor[0,:].tolist()
        targets = tensor[1,:].tolist()

        #Map edge list to node indexes
        src_type, dst_type = edge_type[0], edge_type[2]
        src_map,dst_map = self.node_map[src_type], self.node_map[dst_type]

        mapped_src = [src_map[n] for n in sources]
        mapped_trg = [dst_map[n] for n in targets]

        return {src_type:mapped_src, dst_type:mapped_trg, f"torch_{src_type}_index":sources, f"torch_{dst_type}_index":targets}

    def _reverse_map_heterodata(self,data):
        """Maps full edge data from pyg Heterodata back into the original node indexes from the dataframe"""
        edge_dict = {}
        for edge_type in data.edge_types:
            type_dict = {}
            edge_tensor = data[edge_type]["edge_index"]
            mapped_edge_list = self._reverse_map_tensor(edge_tensor,edge_type)

            type_dict["message_passing_edges"] = mapped_edge_list

            if "edge_label_index" in data[edge_type].keys():
                labeled_edges_tensor = data[edge_type]["edge_label_index"]
                # labeled_edges_list = tensor_to_edgelist(labeled_edges_tensor)
                mapped_labeled_edges_list = self._reverse_map_tensor(labeled_edges_tensor,edge_type)
                edge_labels = data[edge_type]["edge_label"].tolist()

                type_dict["supervision_edges"] = mapped_labeled_edges_list
                type_dict["supervision_labels"] = edge_labels
    
            edge_dict[edge_type] = type_dict
        
        return edge_dict
    
    def _edge_dict_to_dataframe(self):
        edges_df = []
        e_dict = self.edge_dict[self.prediction_edge_type]
        supervision_edges = pd.DataFrame(e_dict["supervision_edges"])

        labeled_edges = pd.concat([supervision_edges,pd.DataFrame(e_dict["supervision_labels"])],axis=1).rename(columns={0:"label"})
        msg_passing_edges = pd.DataFrame(e_dict["message_passing_edges"])

        msg_passing_edges["edge_type"] = "message_passing"
        labeled_edges["edge_type"] = "supervision"


        edges_df.append(labeled_edges)
        edges_df.append(msg_passing_edges)
        total_df = pd.concat(edges_df,axis=0)
        return total_df

class Predictor():
    def __init__(self,node_df, encodings_dict):
        assert node_df.index.name == "node_index", f"df index must be node_index, not {node_df.index.name}."

        self.df = node_df
        self.encodings = encodings_dict
        gene_index = torch.tensor(self.df[node_df.node_type == "gene_protein"]["tensor_index"].index.values)
        disease_index = torch.tensor(self.df[node_df.node_type == "disease"]["tensor_index"].index.values)
        self.node_index_dict = {"gene_protein":gene_index,"disease":disease_index}
    
    def inner_product_decoder(self,x_source,x_target,apply_sigmoid=True):
        pred = (x_source * x_target).sum(dim=1)

        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def prioritize_one_vs_all(self,node_index,return_df=True):
        source_type = self.df.loc[node_index,"node_type"]
        tensor_index = self.df.loc[node_index,"tensor_index"]

        if source_type == "disease":
            target_type = "gene_protein"

        elif source_type == "gene_protein":
            target_type = "disease"

        source_vector = self.encodings[source_type][tensor_index]
        target_matrix = self.encodings[target_type]

        predicted_edges = self.inner_product_decoder(source_vector,target_matrix)
        ranked_scores, ranked_indices = torch.sort(predicted_edges,descending=True)
        ranked_node_index = self.node_index_dict[target_type][ranked_indices]

        
        if return_df:
            results = pd.DataFrame({"score":ranked_scores,"tensor_index":ranked_indices,"node_index":ranked_node_index})
            node_names = self.df[self.df.node_type == target_type]["node_name"]
            ranked_predictions = pd.merge(results,node_names,left_on="node_index",right_index=True)
            ranked_predictions.index.name = "rank"

        else:
            ranked_predictions = [ranked_node_index,ranked_scores]
            
        return ranked_predictions
    
    def predict_supervision_edges(self,data, edge_type, return_dataframe=True):
        """If return_dataframe_==True, returns dataframe with edges, prediction scores and labels. Else, returns predicted scores tensor"""
        src_type, trg_type = edge_type[0],edge_type[2]
        x_source = self.encodings[src_type]
        x_target = self.encodings[trg_type]

        edge_label_index = data.edge_label_index_dict[edge_type]
        source_index, target_index = edge_label_index[0], edge_label_index[1]

        emb_nodes_source = x_source[source_index]
        emb_nodes_target = x_target[target_index]

        pred = self.inner_product_decoder(emb_nodes_source, emb_nodes_target)
        if return_dataframe:
            labels = data.edge_label_dict[edge_type].numpy()
            df = pd.DataFrame({"torch_gene_protein_index":source_index, "torch_disease_index":target_index, "score":pred, "label":labels})
            return df
        else: 
            return pred
    
    def hits_at_k(self,node_index,mapped_train,mapped_val):
      k_list = [5,10,50,100]
      predictions = self.prioritize_one_vs_all(node_index)

      node_type = self.df.loc[node_index,"node_type"]
      y_type = "disease" if node_type == "gene_protein" else "gene_protein"

      new_edges = set(mapped_val[(mapped_val.edge_type == "supervision") & (mapped_val.label == 1) & (mapped_val[node_type] == node_index)][y_type].values)
      seen_edges = set(mapped_train[(mapped_train.label != 0) & (mapped_train[node_type] == node_index)][y_type].values)

      results = {"seen_edges":len(seen_edges),"new_edges":len(new_edges)}

      for k in k_list:
            predicted_top = set(predictions[:k]["node_index"].values)

            seen_hits = len(seen_edges.intersection(predicted_top))
            new_hits = len(new_edges.intersection(predicted_top))

            results[f"{k}_seen"] = seen_hits
            results[f"{k}_new"] = new_hits

      return results