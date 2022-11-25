import pandas as pd 
from statsmodels.stats import multitest

graph_node_data = pd.read_csv("../../data/processed/graph_data_nohubs/nohub_graph_node_data.csv")

def load_and_correct(path,alpha):
    df = pd.read_csv(path, index_col=[0,1])
    df["pvalue"] = multitest.fdrcorrection(df.pvalue.values, alpha=alpha)[1]
    df["reject"] = multitest.fdrcorrection(df.pvalue.values,alpha=alpha)[0]
    df = df[df.reject].drop(columns="reject")

    return df

def get_results(results_df,cluster):
    pathways = results_df.loc[cluster].sort_values(by="odds_ratio",ascending=False).index.values
    result = graph_node_data.set_index("node_index").loc[pathways][["node_name"]]
    return result

def find_pathways(partition,cluster,alpha=0.05):
    path = "../../reports/reports_nohubs/analisis_red_genes/" + partition + "_pathways.csv"
    results_df = load_and_correct(path,alpha)
    pathways = get_results(results_df,cluster)
    return pathways
