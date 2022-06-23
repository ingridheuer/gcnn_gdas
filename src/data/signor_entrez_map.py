#%%
import pandas as pd
import urllib.parse
import urllib.request
from io import StringIO
#%%
data_path = "../data/external/"
signor = pd.read_csv(data_path+"signor_all_data.tsv", sep="\t") #signor all data crudo
#%%
#me quedo solo con los ids que corresponden a datos de UNIPROT
signor_uniprot_ids = pd.concat([signor[signor.DATABASEA == "UNIPROT"].IDA, signor[signor.DATABASEB == "UNIPROT"].IDB]).unique()

#armo un string con todos los ids en el formato que me pide el query
id_query = ''
for id in signor_uniprot_ids:
    id_query += ' ' + id
#%%
#hago el query
url = 'https://www.uniprot.org/uploadlists/'

params = {
'from': 'ACC+ID',
'to': 'P_ENTREZGENEID',
'format': 'tab',
'query': id_query
}

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
with urllib.request.urlopen(req) as f:
   response = f.read()

output = response.decode('utf-8')

#armo un dataframe con el output del query
mapping_table = pd.read_csv(StringIO(output), sep='\t', header=0)
#%%
#agrego al dataframe original dos columnas con los IDS que recuperé del query
signor_entrezid = signor.merge(mapping_table,left_on = "IDA",right_on="From",how="left").rename(columns={'To':'ENTREZ_ID_A'}).drop(['From'],axis=1)
signor_entrezid['ENTREZ_ID_A'] = signor_entrezid['ENTREZ_ID_A'].astype('Int64')

signor_entrezid = signor_entrezid.merge(mapping_table,left_on = "IDB",right_on="From",how="left").rename(columns={'To':'ENTREZ_ID_B'}).drop(['From'],axis=1)
signor_entrezid['ENTREZ_ID_B'] = signor_entrezid['ENTREZ_ID_B'].astype('Int64')

#guardo el csv con los archivos de pasos intermedios
signor_entrezid.to_csv("../data/interim/signor_mapped.csv")