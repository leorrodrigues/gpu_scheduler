library(ggplot2)
library(wesanderson)
library(RColorBrewer)

all <- read.table("times_trabalho_ppa",header=T,sep=";")

ggplot(data=all, aes(x=Size, 
                     y=Time, 
                     group=Algorithm, 
                     colour=Algorithm, 
                     shape=Algorithm)
       ) + 
    geom_line(size=1) + 
    geom_point(size=3) + 
    coord_cartesian(ylim = c(0, 900)) + 
    scale_y_continuous(breaks = seq(0,900, by=100)) +
    scale_x_continuous(breaks = seq(4,44, by=4))+
    theme_bw()+ 
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), 
          axis.line = element_line(colour = "black"),
          legend.position="bottom"
          )+
    scale_color_manual(values=wes_palette(n=5, name="Darjeeling1"))