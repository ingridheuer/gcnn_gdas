# (0) Cargo librerías ----
#library(DOSE)
#library(DBI)
#library(DO.db)
library(data.table)
library(glue)
#library(AnnotationHub)
#library(biodbUniprot)
#library(tidyverse)
#library(ggplot2)
#library(plotly)
#library(RColorBrewer)
#library(Mes)
# (1) Cargo datos  ----
data_path = '/home/ingrid/Documents/tesis/datos_redes/'
DGN_path <- '/home/ingrid/Documents/tesis/datos_redes/disgenet/'
GDA <-
  read.csv(
    paste0(DGN_path, 'curated_gene_disease_associations.tsv.gz'),
    sep = '\t',
    header = TRUE
  )
DMAP <-
  read.csv(
    paste0(DGN_path, 'disease_mappings_to_attributes.tsv.gz'),
    sep = '\t',
    header = TRUE
  )
hippie <- read.csv(paste0(data_path, 'hippie_current_v2.3.txt'),
                   header = FALSE,
                   sep = '\t')
signor <-
  read.csv(paste0(data_path, 'signor_all_data_06_05_22.tsv'),
           sep = "\t")
signor_to_entrez <-
  read.csv(
    paste0(data_path, 'uniprot_to_entrez_signor.txt'),
    sep = '\t',
    header = TRUE
  )
signor_uniprot_nomap <-
  read.csv(
    paste0(data_path, 'signor_uniprot_sin_mapear.txt'),
    sep = 't',
    header = TRUE
  )
drugbank_uniprot_mappings = read.csv(paste0(data_path, 'drugbank_to_uniprot.tab'),
                                     sep = '\t',
                                     header = TRUE)

# (2) Hago cosas con HIPPIE  ----
setnames(
  hippie,
  old = colnames(hippie),
  new = c(
    "P1_UNIPROT_ID",
    "P1_ENTREZ_ID",
    "P2_UNIPROT_ID",
    "P2_ENTREZ_ID",
    "SCORE",
    "EVIDENCE"
  )
) #esto es "in place"

#Quiero sacar el _HUMAN de la columna de simbolos porque molesta

#reemplacé "_HUMAN" por "" en toda la columna PROT1_SYMBOL
hippie$P1_UNIPROT_ID = lapply(hippie$P1_UNIPROT_ID, function(x) {
  gsub("_HUMAN", "", x)
})
hippie$P2_UNIPROT_ID = lapply(hippie$P2_UNIPROT_ID, function(x) {
  gsub("_HUMAN", "", x)
})

#todas las entradas de hippie tienen prot_id
table(hippie$P1_ENTREZ_ID == "")
table(hippie$P2_ENTREZ_ID == "")

#veo que overlap tengo de proteinas de hippie y datos en disgenet
ids_hippie = unique(c(hippie$P1_ENTREZ_ID, hippie$P2_ENTREZ_ID))
glue("Hay {length(ids_hippie)} ids únicos en HIPPIE") #19679 proteinas distintas

ids_disgenet = unique(GDA$geneId)
glue("Hay {length(ids_disgenet)} ids únicos en DisGeNET") #9703 genes distintos (en disgenet curado)

#IDS que no están en disgenet
not_in_disg_id = which(!(
  hippie$P1_ENTREZ_ID %in% ids_disgenet &
    hippie$P2_ENTREZ_ID %in% ids_disgenet
))

#me fijo si las dos proteínas involucradas están en disgenet
in_disg_id = which(hippie$P1_ENTREZ_ID %in% ids_disgenet &
                     hippie$P2_ENTREZ_ID %in% ids_disgenet)
glue(
  "{length(not_in_disg_id)} interacciones de HIPPIE son entre dos proteínas que NO están en DisGeNET"
)
glue(
  "{length(in_disg_id)} interacciones de HIPPIE son entre dos proteinas de HIPPIE SI están en DisGeNET"
)

hippie[not_in_disg_id, c("P1_UNIPROT_ID", "P1_ENTREZ_ID")]

hippie_in_disg = hippie[in_disg_id,]

unasi_otrano = which((hippie$P1_ENTREZ_ID %in% ids_disgenet) & !(hippie$P2_ENTREZ_ID %in% ids_disgenet)) #en algunos casos está una de las proteínas y la otra no
length(unasi_otrano)
aver = hippie[unasi_otrano,]

#quiero ver como son los scores de los genes que me quedaron
prots_que_quedaron = intersect(ids_hippie, ids_disgenet)
glue("Encontré en HIPPIE enlaces para {length(prots_que_quedaron)} proteínas de las {length(ids_disgenet)} totales en disgenet")

hist(
  hippie_in_disg$SCORE,
  xlab = "Score del enlace de HIPPIE",
  ylab = "Count",
  col = "#79CDCD",
  main = "Score de los enlaces de HIPPIE que quedaron en DisGeNET"
)

#fig = plot_ly(hippie_in_disg, x=hippie_in_disg$SCORE, type = "histogram")
#p <- ggplot(hippie_in_disg, aes(x=hippie_in_disg$SCORE)) +
#  geom_histogram()

#Veo la distribución de scores en el dataset total para ver si cambia mucho
hist(hippie[not_in_disg_id, "SCORE"], main = "Score de los enlaces de HIPPIE que quedaron fuera de DisGeNET")
hist(hippie$SCORE, main = "Score de todos los enlaces de HIPPIE")
hist(GDA$score)

# (3) SIGNOR UNIPROT -> ENTREZ ID ----
signor_types = unique(c(signor$TYPEA, signor$TYPEB))
signor_sources = unique(c(signor$DATABASEA, signor$DATABASEB))
signor_sources

#empiezo mapeando las fuentes de uniprot que se que puedo
signor_uniprot = signor[which(signor$DATABASEA == "UNIPROT" |
                                signor$DATABASEB == "UNIPROT"), ]
signor_uniprot_ids = unique(c(signor_uniprot$IDA, signor_uniprot$IDB))

length(signor_uniprot_ids)

#me armo un archivo para cargar a la pagina de uniprot y que me devuelva el mapeo
write.table(
  signor_uniprot_ids,
  "signor_uniprot_ids.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = ' ',
  quote = FALSE
)

#OBTUVE LOS MAPEOS DE UNIPROT A ENTREZ ID DE LAS PROTS EN SIGNOR CON ID DE UNIPROT
#signor_uniprot_mappings es el txt con los mapeos que me baje, lo cargué mas arriba en la sección cargo datos

#Veo que pasó con los que no mapearon
signor_uniprot_nomap

#chequeando no haberle cargado ids con comas
tienen_comas = which(grepl(",", signor_to_entrez$From))
tienen_comas

#chequeando espacios

tienen_espacios = which(grepl(" ", signor_to_entrez$From))
tienen_espacios


# (4) SIGNOR CHEBI ----

#Estos no existen porque no son proteínas, ver que hago con esto
signor_chebi = signor[which(signor$DATABASEA == "ChEBI" |
                              signor$DATABASEB == "ChEBI"), ]
signor_chebi_ids_1 = unique(c(signor_chebi$IDA, signor_chebi$IDB))
signor_chebi_ids =  signor_chebi_ids_1[which(grepl("CHEBI", signor_chebi_ids_1))]
write.table(
  signor_chebi_ids,
  "signor_chebi_ids.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = ' ',
  quote = FALSE
)

# (5) SIGNOR SIGNOR ----
#los datos de database signor son complejos de proteínas o familias de proteínas, esto no va a mapear a un gen. Pueden ser features
signor[which(signor$DATABASEA == "SIGNOR" |
               signor$DATABASEB == "SIGNOR"), c("TYPEA", "TYPEB", "ENTITYA", "ENTITYB")]

# (6) SIGNOR-PUBCHEM ----
#idem chebi, no son proteínas
signor[which(signor$DATABASEA == "PUBCHEM" |
               signor$DATABASEB == "PUBCHEM"), c("TYPEA", "TYPEB", "ENTITYA", "ENTITYB")]

#(7) SIGNOR- miRBase
#otra vez, no son proteínas, son "mirnas" (jaja)
signor[which(signor$DATABASEA == "miRBase" |
               signor$DATABASEB == "miRBase"), c("TYPEA", "TYPEB")]

# (8) SIGNOR - DRUGBANK ----
#esto pense que iba a mapear pero si mapea!
#son muy poquitos y de tipo "antibody protein"
signor[which(signor$DATABASEA == "DRUGBANK" |
               signor$DATABASEB == "DRUGBANK"), c("TYPEA", "TYPEB")]

signor_drugbank = signor[which(signor$DATABASEA == "DRUGBANK" |
                                 signor$DATABASEB == "DRUGBANK"), ]
signor_drugbank_ids = unique(c(signor_drugbank$IDA, signor_drugbank$IDB))
write.table(
  signor_drugbank_ids,
  "signor_drugbank_ids.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = ' ',
  quote = FALSE
)

setnames(
  drugbank_uniprot_mappings,
  old = c(
    "yourlist.M20220509A084FC58F6BBA219896F365D15F2EB444C5EE2C",
    "Entry"
  ),
  new = c("entry", "uniprot_id")
)
write.table(
  unique(drugbank_uniprot_mappings$uniprot_id),
  "drugbank_uniprot_ids.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = ' ',
  quote = FALSE
)

#OBTUVE EL MAPEO DE DRUGBANK A ENTREZ ID
#IGUAL ESTO BASICAMENTE ME DIO LA COLUMNA DE ENTITY B JAJA, NOSE SI AGREGARLO
drugbank_to_entrez = read.csv(paste0(data_path, 'drugbank_to_entrez.txt'),
                              sep = '\t',
                              header = TRUE)

#entonces: dataframes mapeados a entrez: drugbank_to_entrez,

# (9) Agrego a signor columnas de entrez ids ----


#dif entre cuantas ids tenia que conseguir y cuantas consegui
length(signor_uniprot_ids) - length(signor_to_entrez$From)

#estas son las ids de la lista que son de uniprot y no las que se me filtraron (mas arriba explique porque paso y porque no molesta)
signor_uniprot_ids_filtrado = signor_uniprot_ids[which(
  !grepl("SIGNOR", signor_uniprot_ids) &
    !grepl("CHEBI", signor_uniprot_ids) &
    !grepl("CID", signor_uniprot_ids)
)]
glue(
  "Me faltaron {length(signor_uniprot_ids_filtrado) - length(signor_to_entrez$From)} mapeos de signor->uniprot->entrez"
)

aver = signor_uniprot_nomap[which(
  !grepl("SIGNOR", signor_uniprot_nomap$no) &
    !grepl("CHEBI", signor_uniprot_nomap$no) &
    !grepl("CID", signor_uniprot_nomap$no)
), ]
#estoy viendo los mapeos que faltaron que se que son de uniprot
#veo que lo que pasó es que algunos tienen un guión, por ej "P33316-2", que uniprot toma como "P33316", puedo usar ese mapeo pero nose si esta bien
#otros como P01189_PRO_0000024975, en la pagina me dice que es "P01189", nose si usar ese mapa idem
#los que dicen MI asumo que son mirnas y SID por ahi un signor id
#por ahora me quedo con lo que consegui y lo agrego a la columna nueva

mapeo <- function(entry, mapping_from, mapping_to) {
  map_id = which(mapping_from == entry)
  mapping = mapping_to[map_id]
  return(mapping)
}

#aver si salio
mapeo(signor_uniprot[1, "IDA"], signor_to_entrez[, 1], signor_to_entrez[, 2])
signor_uniprot[1, "IDA"]
signor_to_entrez[which(signor_to_entrez$From == "P52564"), ]
#ok parece que si

#probando
signor2 <-
  merge(x = signor,
        y = signor_to_entrez,
        by.x = "IDA",
        by.y = "From")[]
setnames(signor2, old = "To", new = "ENTREZ_ID_A")

signor3 <-
  merge(x = signor2,
        y = signor_to_entrez,
        by.x = "IDB",
        by.y = "From")[]
setnames(signor3, old = "To", new = "ENTREZ_ID_B")

#probando
signor3[1, c("IDA", "IDB", "ENTREZ_ID_A", "ENTREZ_ID_B")]
signor_to_entrez[which(signor_to_entrez$From == "A0A0B4J2F0"), ]
signor_to_entrez[which(signor_to_entrez$From == "Q96S66"), ]
colnames(signor3)

#LISTO EL MAPEO DE SIGNOR->UNIPROT->ENTREZ ID, lo guardo en un txt
write.table(
  signor3,
  "signor_mapeado_a_entrez.txt",
  na = "",
  row.names = FALSE,
  append = FALSE,
  sep = '\t',
  quote = FALSE
)


#(10) veo cuantos mapeos pude obtener ----
#esto depende si cuento de proteína a proteína o distintos tipos de objetos, ver que hago con los que son químicos etc y de otras bases que no van a mapear a GDA porque... no son genes
signor3_uniprot = signor3[which(signor3$DATABASEA == "UNIPROT" &
                                  signor3$DATABASEB == "UNIPROT"), ]
overlap_count = which(signor3$ENTREZ_ID_A %in% GDA$geneId &
                        signor3$ENTREZ_ID_B %in% GDA$geneId)
glue(
  "Obtuve {length(overlap_count)} mapeos de signor a disgenet, quedaron sin mapear (porque no son genes/proteínas o faltaba el map) {length(signor3$IDA) - length(overlap_count)}"
)
