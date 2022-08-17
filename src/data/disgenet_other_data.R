# (1) Libraries ----
library(data.table)
library(glue)
library(tidyverse)
# (2) Load Data ----
data_external = "../../data/external/"
data_interim = "../../data/interim/"
data_processed = "../../data/processed/"
disgnet_gdas = fread(paste0(data_external,"curated_gene_disease_associations.tsv"), header = TRUE)
disgenet_attr = fread(paste0(data_external,"disease_mappings_to_attributes.tsv"), header= TRUE)
disgenet_maps = fread(paste0(data_external,"disease_mappings.tsv"), header= TRUE)
# (3)  Aver ----
disgenet_attr
disgenet_maps

table(disgenet_maps$vocabulary)
disgenet_maps[vocabulary == "MONDO",]
disgenet_attr[hpoClassId == ""]
disgenet_attr[doClassId == ""]
disgenet_attr[(diseaseClassMSH == "") & (hpoClassId != "")]
disgenet_attr[(diseaseClassMSH == "") & (doClassId != "")]
disgenet_attr[hpoClassName != ""]
disgenet_attr[(doClassName != "") & (diseaseClassMSH != "")]

# Aver ----
