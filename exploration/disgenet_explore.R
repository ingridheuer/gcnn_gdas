library(glue)

#----
data_path = '/home/ingrid/Documents/tesis/datos_redes/'
disgenet = read.csv(paste0(data_path,"disgenet/curated_gene_disease_associations.tsv.gz"), sep='\t', header = TRUE)
disgenet_attr = read.csv(paste0(data_path,"disgenet/disease_mappings_to_attributes.tsv"), sep='\t', header= TRUE)

head(disgenet)
unique(disgenet$source)
aver = disgenet[which(disgenet$source == "CGI"),]
head(aver)

head(disgenet_attr)
colnames(disgenet_attr)
unique(disgenet_attr$type)

disgenet_atrr_donly = disgenet_attr[which(disgenet_attr$type == "disease"),]
head(disgenet_atrr_donly)