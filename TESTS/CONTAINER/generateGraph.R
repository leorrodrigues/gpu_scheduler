library(ggplot2)
library(wesanderson)
library(RColorBrewer)
cols <- c('character', 'integer', 'integer', 'numeric')

all <- read.table("times_total.txt",header=T,sep=";",colClasses = cols)

pMAHPG <- all[all[,1]=="MCL + AHPG",]
pM <- all[all[,1]=="Pure MCL",]


names <- c("AHP","AHPG","MCL + AHPG","Pure MCL","AHPG Clustered")

for(name in names){
    
    pg <- all[all[,1]==name,]
    #all <- pl
    legend<-"Container clustering with"
    legend<-paste(legend,name,sep=" ")
    legend<-paste(legend,"method",sep=" ")
    pl<-ggplot(data=pg, aes(x=Number.of.containers, 
                         y=Time, 
                         group=factor(Fat.Tree.Size), 
                         colour=factor(Fat.Tree.Size), 
                         shape=factor(Fat.Tree.Size))
    ) + 
        geom_line(size=1) + 
        geom_point(size=3) + 
        coord_cartesian(ylim = c(0, 900)) + 
        scale_y_continuous(breaks = seq(0,900, by=100)) +
        #scale_x_continuous(breaks = seq(4,48, by=4))+
        theme_bw()+ 
        theme(panel.border = element_blank(), 
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(), 
              axis.line = element_line(colour = "black"),
              legend.position="bottom",
              plot.title = element_text(color="black", size=16, face="bold.italic"))+
        scale_color_manual(values=wes_palette(n=5, name="FantasticFox1"))+
        labs(colour="Fat Tree Size", shape="Fat Tree Size", x="Number of Containers", y="Time (s)", title=legend)
    plot(pl)
    ggsave(paste(name,".pdf",sep=""),device="pdf")
}

sizes <- c(4,16,32,40)

ylimits = c(10,100,900,900)
xlimits = c(1024,1024,8,8)

index=1
for(size in sizes){
    
    pg <- all[all[,2]==size,]
    #all <- pl
    legend<-"Container clustering with fat tree size k="
    legend<-paste(legend,size,sep="")
    pl<-ggplot(data=pg, aes(x=Number.of.containers, 
                            y=Time, 
                            group=factor(Algorithm), 
                            colour=factor(Algorithm), 
                            shape=factor(Algorithm))
    ) + 
        geom_line(size=1) + 
        geom_point(size=3) + 
        coord_cartesian(ylim = c(0, ylimits[index]),xlim=c(0,xlimits[index])) + 
        #scale_y_continuous(breaks = seq(0,900, by=100)) +
        #scale_x_continuous(breaks = seq(4,8, by=4))+
        theme_bw()+ 
        theme(panel.border = element_blank(), 
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(), 
              axis.line = element_line(colour = "black"),
              legend.position="bottom",
              plot.title = element_text(color="black", size=16, face="bold.italic"))+
        scale_color_manual(values=wes_palette(n=5, name="FantasticFox1"))+
        labs(colour="Multicriteria Method", shape="Multicriteria Method", x="Number of Containers", y="Time (s)", title=legend)
    plot(pl)
    ggsave(paste(size,".pdf",sep=""),device="pdf")
    index=index+1
}