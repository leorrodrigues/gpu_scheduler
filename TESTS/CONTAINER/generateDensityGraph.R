library(ggplot2)
library(wesanderson)
library(RColorBrewer)
cols <- c('character', 'integer', 'integer', 'numeric')

all <- read.table("times_total.txt",header=T,sep=";",colClasses = cols)

#pMAHPG <- all[all[,1]=="MCL + AHPG",]
#pM <- all[all[,1]=="Pure MCL",]


names <- c("ahp","ahpg","mcl ahp","mcl ahpg","pure mcl")

for(name in names){
    
    pg <- all[all[,1]==name,]
    #all <- pl
    legend<-"Container clustering with"
    legend<-paste(legend,name,sep=" ")
    legend<-paste(legend,"method",sep=" ")
    pl<-ggplot(data=pg, aes(x=Number.of.Containers,
                            group=factor(Fat.Tree.Size), 
                            fill=factor(Fat.Tree.Size))
    ) + 
        #scale_fill_brewer(palette = "Spectral")+
        geom_histogram(
            bins=4,
            col="black",
            size=.1
                       )+
        #coord_cartesian(xlim=c(0,100)) + 
        theme_bw()+ 
        theme(panel.border = element_blank(), 
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(), 
              axis.line = element_line(colour = "black"),
              legend.position="bottom",
              plot.title = element_text(color="black", size=16, face="bold.italic"))+
        # scale_color_manual(values=wes_palette(n=5, name="FantasticFox1"))+
        labs(colour="Fat Tree Size", shape="Fat Tree Size", x="Number of Containers", y="Time (s)", title=legend)
    plot(pl)
    ggsave(paste(name,".pdf",sep=""),device="pdf")
}
