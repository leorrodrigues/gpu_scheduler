library(ggplot2)
library(lubridate)
theme_set(theme_bw())

ahp <- read.table("AHP.data",header=T,sep=";")

ahpg <- read.table("AHPG.data",header=T,sep=";")

mcl <- read.table("CLUSTER.data",header=T,sep=";")

mcl_ahp <- read.table("MCL_AHP.data",header=T,sep=";")

ahp_clustering <- read.table("AHP_CLUSTER.data",header=T,sep=";")
