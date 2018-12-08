library(ggplot2)
library(wesanderson)
library(RColorBrewer)

all <- read.table("obj_google_0_flat.txt",header=T,sep=",")

pl<-ggplot(data=all, aes(x=Scheduler_Time,
                            y=RAM
                         )
           )+ 
    geom_line(size=1, color="orange") +
    theme_bw()+ 
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), 
          axis.line = element_line(colour = "black"),
          legend.position="top",
          plot.title = element_text(color="black", size=16, face="bold.italic"))+
    labs(
        x="Tempo(s)",
        y="Percentual de Fragmentação"
    )


    plot(pl)
