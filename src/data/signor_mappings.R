#(1) ----
library(glue)
library(data.table)
#(2)----
data_path = "/home/ingrid/Documents/tesis/datos_redes/"
disgenet = fread(paste0(data_path, "disgenet/curated_gene_disease_associations.tsv"))
signor = fread(paste0(data_path,"signor_all_data_06_05_22.tsv"))
#(3)----
colnames(signor)
dim(signor)

#Veo que tipos de entidades hay en signor y como se relacionan entre sí
#Tabla de sources de los datos, en total (tanto para A como para B)
table(c(signor[,DATABASEA], signor[,DATABASEB]))

#Cuantas relaciones hay entre cada tipo A y tipo B, veo que no son simétricas
table(signor[,TYPEA], signor[,TYPEB])

#Voy a mapear los ids de uniprot a entrez, me quedo solo con esos y hago un txt para subir a la página de uniprot
ids_a_mapear = c(signor[DATABASEA=="UNIPROT",IDA], signor[DATABASEB == "UNIPROT",IDB])

write.table(
  ids_a_mapear,
  "signor_uniprot_ids_2.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = ' ',
  quote = FALSE
)

#Veo de que database son los complejos
#los complejos son todos de signor
table(signor[TYPEA == "complex", DATABASEA])
table(signor[TYPEB == "complex", DATABASEB])

#Viendo los que tienen cosas raras
signor[grepl("_", IDA) | grepl("_",IDB), list(IDA,TYPEA,DATABASEA,IDB,TYPEB,DATABASEB)]

#(4) Ya tengo los mapeos, ahora agrego una columna con su id correspondiente (lo hago mergeando las dos dt)
mapeos = fread(paste0(data_path,"signor_to_entrez_2.tab.txt"))

signor =
  merge(x = signor,
        y = mapeos,
        by.x = "IDA",
        by.y = "From",
        all.x = TRUE)[]
setnames(signor, old = "To", new = "ENTREZ_ID_A")

signor =
  merge(x = signor,
        y = mapeos,
        by.x = "IDB",
        by.y = "From",
        all.x = TRUE)[]
setnames(signor, old = "To", new = "ENTREZ_ID_B")

#(5) Probando que esté bien
signor[1,list(IDA,IDB,ENTREZ_ID_A, ENTREZ_ID_B)]
mapeos[From == signor[1,IDA] | From == signor[1,IDB]]

#(6) Viendo que pasó con los que no mapearon
sin_mapear = fread(paste0(data_path, "signor_not_mapped.txt"))
glue("Me quedaron {dim(sin_mapear)[1]} proteinas sin mapear")
signor[IDA == "Q5TG30" | IDB == "Q5TG30", list(TYPEA,TYPEB,DATABASEA,DATABASEB)] #esto está en uniprot y disgenet pero no me mapea el id, nose porque (los dejo??)
signor[IDA %in% sin_mapear[,`not mapped`] | IDB %in% sin_mapear[,`not mapped`]][1:10,list(ENTITYA, ENTITYB, TYPEA, TYPEB, DATABASEA, DATABASEB,EFFECT)]
aver = signor[IDA %in% sin_mapear[,`not mapped`] | IDB %in% sin_mapear[,`not mapped`], list(ENTITYA, ENTITYB, TYPEA, TYPEB, DATABASEA, DATABASEB,EFFECT)]

#LISTO EL MAPEO DE SIGNOR->UNIPROT->ENTREZ ID, lo guardo en un txt
write.table(
  signor,
  "signor_mapeado_a_entrez_2.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = '\t',
  quote = FALSE
)
