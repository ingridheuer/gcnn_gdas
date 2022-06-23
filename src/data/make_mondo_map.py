#%%
import os
import numpy as np
import pandas as pd
from mondo_obo_parser import OBOReader
#%%
data_path = "../../data/external/mondo.obo"
data_path_interim = "../../data/interim"

data = [*iter(OBOReader(data_path))]
mondo_terms = pd.DataFrame([{'id':x.item_id, 
                             'name':x.name, 
                             'definition':x.definition,
                             'is_obsolete':x.is_obsolete,
                             'replacement_id': x.replaced_by} for x in data])
print(mondo_terms.shape[0], "total terms")
print(mondo_terms.query('is_obsolete==False').shape[0], 'not obsolete')
#%%

print('"is_a" relationships between mondo terms')
mondo_parents = []
for x in data: 
    if x._parents: 
        for parent in x._parents: 
            mondo_parents.append({'parent':parent, 'child':x.item_id})           
mondo_parents = pd.DataFrame(mondo_parents).drop_duplicates()
#%%
#DE ACA SACO LOS MAPPINGS A TODAS LAS ONTOLOGIAS!!!!
#Tiraba error porque algunos xrefs tenian "Nones", agregue unas cosas para manejar eso
print("cross references from mondo to other ontologies")
mondo_xrefs = []
mondo_failed_xrefs = [] #los que tenían error 
for x in data: 
    if x.xrefs:
        for xref in x.xrefs:
            try:
                ont, name = xref.split(':')            
                mondo_xrefs.append({'ontology_id':name, 'ontology':ont, 'mondo_id':x.item_id})           
            except:
                print(x.xrefs)
                print(x.id)
                mondo_failed_xrefs.append({'mondo_id':x.id, 'xrefs':x.xrefs})

#%%
#lo mismo pero en vez de con un except puse algo para evitar el error directamente
mondo_xrefs = []
for x in data: 
    if x.xrefs:
        for xref in x.xrefs:
            if xref != None:
                ont, name = xref.split(':')            
                mondo_xrefs.append({'ontology_id':name, 'ontology':ont, 'mondo_id':x.item_id})

mondo_xrefs = pd.DataFrame(mondo_xrefs).drop_duplicates()
print('references to the following ontologies are available:')
print(np.unique(mondo_xrefs.get('ontology').values))
print('references from mondo to mondo indicate equivalence/synonyms')     
#%%
print("groupings of mondo terms")
mondo_subsets = []
for x in data: 
    if x.subsets: 
        for sub in x.subsets: 
            mondo_subsets.append({'id':x.item_id, 'subset':sub})           
mondo_subsets = pd.DataFrame(mondo_subsets).drop_duplicates()
print('available subsets by count:')
mondo_subsets.groupby('subset').count().sort_values('id',ascending=False)

mondo_def = mondo_terms.get(['id','name','definition']).fillna('').copy()
for x in mondo_def.itertuples(): 
    if x.definition:
        mondo_def.loc[x.Index, 'definition'] =  x.definition.split('\"')[1]
    else: 
        mondo_def.loc[x.Index, 'definition'] = float('nan')
mondo_def = mondo_def.dropna()
mondo_terms = mondo_terms.drop('definition', axis=1)

#mondo_terms.to_csv(data_path+'/mondo_terms.csv', index=False)
#mondo_parents.to_csv(data_path+'/mondo_parents.csv', index=False)
#mondo_xrefs.to_csv(data_path+'/mondo_references.csv', index=False)
#mondo_subsets.to_csv(data_path+'/mondo_subsets.csv', index=False)
#mondo_def.to_csv(data_path+'/mondo_definitions.csv', index=False)

#%%
#Me guardo el mapping table en un csv
mondo_umls_map = mondo_xrefs[mondo_xrefs['ontology'] == 'UMLS'].drop(columns='ontology').rename(columns={'ontology_id':'CUI', 'mondo_id':'mondo'})
mondo_umls_map.to_csv(data_path_interim+"/mondo_cui_map.csv", index=False)