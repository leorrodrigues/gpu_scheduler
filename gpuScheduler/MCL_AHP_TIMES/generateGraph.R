library(ggplot2)
library(lubridate)
theme_set(theme_bw())

ahp <- read.table("AHP.data",header=T,sep=";")

mcl_ahp <- read.table("MCL_AHP.data",header=T,sep=";")

total <- read.table("merge.data",header=T,sep=";")

# plot
# p <- ggplot(fill=total$AHP.Time) + 
#     geom_line(data=total, aes(x=total$size,y=total$AHP.Time,color='steelblue'),size=0.5) +
#     geom_line(data=total, aes(x=total$size,y=total$AHP...MCL.Time,color='red'),size=0.5)+
#     labs(title="Host selection algorithm", 
#          y="Time (s)", 
#          x="Fat Tree size",
#          color=NULL)+
#     coord_cartesian(ylim=c(0,2),xlim=c(4,12))+
#     scale_color_discrete(name = "Algorithms", labels = c("AHP", "AHP + MCL"))

# plot
p <- ggplot(fill=total$AHP.Time) + 
    geom_line(data=total, aes(x=total$size,y=total$AHP.Time,color='steelblue'),size=0.5) +
    geom_line(data=total, aes(x=total$size,y=total$AHP...MCL.Time,color='red'),size=0.5)+
    labs(title="Host selection algorithm", 
         y="Time (s)", 
         x="Fat Tree size",
         color=NULL)+
    
    coord_cartesian(ylim=c(0,900),xlim=c(4,46))+
    scale_color_discrete(name = "Algorithms", labels = c("AHP", "AHP + MCL"))+
    scale_x_continuous(breaks = seq(4,48,by=2))

plot(p)
