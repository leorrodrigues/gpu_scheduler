packages <- c("ggplot2","ggsci")

for(package in packages){
    if(!require(package, character.only = TRUE)){
        install.packages(package, dep=TRUE)
        if(!require(package, character.only=TRUE)) stop("Pacote indisponivel")
        
    }
    library(package, character.only=TRUE)
}
rm(packages)
rm(package)

pallet_colors <- c("#00AFBB", "#E7B800", "#FC4E07","#52854C","#FFDB6D","#4E84C4")

is.nan.data.frame <- function(x)
    do.call(cbind, lapply(x, is.nan))

#######################################
## 1070 TI ##
#######################################
gtxDataOnline <- read.table("data/gtx1070ti/dc-none-lyon-50k-fcfs-online-lyon-50k.log",header=FALSE,sep=" ")
gtxDataOffline <- read.table("data/gtx1070ti/dc-none-lyon-50k-fcfs-offline-lyon-50k.log",header=FALSE,sep=" ")

gtxDataOnline$type = "online"
gtxDataOffline$type = "offline"

gtxData <- rbind(gtxDataOnline, gtxDataOffline)

rm(gtxDataOnline)
rm(gtxDataOffline)

names(gtxData) <- c("method","time","dc_fragmentation","footprint_vcpu","footprint_ram","fragmentation_link","footprint_link","footprint_bandwidth","rejected_tasks","accepted_tasks","accepted_performance", "type")
gtxData$gpu <- "GTX 1070 Ti"

#######################################
## Titan XP ##
#######################################
titanXPDataOnline <- read.table("data/titanXP/dc-none-lyon-50k-fcfs-online-lyon-50k.log",header=FALSE,sep=" ")
titanXPDataOffline <- read.table("data/titanXP/dc-none-lyon-50k-fcfs-offline-lyon-50k.log",header=FALSE,sep=" ")

titanXPDataOnline$type = "online"
titanXPDataOffline$type = "offline"

titanXPData <- rbind(titanXPDataOnline, titanXPDataOffline)

rm(titanXPDataOnline)
rm(titanXPDataOffline)

names(titanXPData) <- c("method","time","dc_fragmentation","footprint_vcpu","footprint_ram","fragmentation_link","footprint_link","footprint_bandwidth","rejected_tasks","accepted_tasks","accepted_performance", "type")
titanXPData$gpu <- "Titan XP"

#######################################
## 2080 TI ##
#######################################
rtxDataOnline <- read.table("data/rtx2080ti/dc-none-lyon-50k-fcfs-online-lyon-50k.log",header=FALSE,sep=" ")
rtxDataOffline <- read.table("data/rtx2080ti/dc-none-lyon-50k-fcfs-offline-lyon-50k.log",header=FALSE,sep=" ")

rtxDataOnline$type = "online"
rtxDataOffline$type = "offline"

rtxData <- rbind(rtxDataOnline, rtxDataOffline)

rm(rtxDataOnline)
rm(rtxDataOffline)

names(rtxData) <- c("method","time","dc_fragmentation","footprint_vcpu","footprint_ram","fragmentation_link","footprint_link","footprint_bandwidth","rejected_tasks","accepted_tasks","accepted_performance", "type")
rtxData$gpu <- "RTX 2080 Ti"

#######################################
## Merge all the data ##
#######################################

dt <- rbind(gtxData, titanXPData)
dt <- rbind(dt, rtxData)

rm(gtxData)
rm(titanXPData)
rm(rtxData)

dt$accepted_performance <- NULL
dt$accepted_performance <- with(dt, accepted_tasks / (accepted_tasks+rejected_tasks))
dt[is.nan(dt)] <- 0

#######################################
## Plot the accepted graph ##
#######################################
acceptedPlot <- ggplot()+
    geom_line(data=dt, aes(x=time, y=accepted_performance, color=type)) +
    theme_classic()+
    theme(
        legend.position="top",
        axis.text.x = element_text(
            angle = 0,
            hjust = 0.7,
            size=12,
        ),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.y = element_text(size=12), 
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        legend.text = element_text(size=12),
        legend.title = element_text(size=12)
    )+
    labs(
        x=("Time(s)"),
        y="Accepted Ratio (%)",
        colour="Scheduling Type"
    )+
    scale_color_d3()

#######################################
## Plot the boxplot graph ##
#######################################
footVcpuPlot <- ggplot()+
    geom_line(data=dt, aes(x=time, y=footprint_vcpu, color=type)) +
    theme_classic()+
    theme(
        legend.position="top",
        axis.text.x = element_text(
            angle = 0,
            hjust = 0.7,
            size=12,
        ),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.y = element_text(size=12), 
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        legend.text = element_text(size=12),
        legend.title = element_text(size=12)
    )+
    labs(
        x=("Time(s)"),
        y="Footprint vCPU (%)",
        colour="Scheduling Type"
    )+
    scale_color_d3()

###################################
## Plot all the graphs
###################################

tiff("accepted_performance_50k.tiff",width= 3000, height= 1200, units="px", res=400,compression = 'lzw')
plot(acceptedPlot)
dev.off()

tiff("footprint_vcpu_50k.tiff",width= 3000, height= 1200, units="px", res=400,compression = 'lzw')
plot(footVcpuPlot)
dev.off()
#################################################
## CONVERTING THE FINAL GRAPHS INTO PNG AND EPS##
#################################################
#convert all tiff to png (latex cant use tiff images)
system("for f in *.tiff; do convert -trim $f ${f%.*}.png; done;")
system("for f in *.tiff; do convert -trim $f ${f%.*}.eps; done;")

#move the files to the graphs folder
system("mv *.png graphs")
system("mv *.eps graphs")

#remove all tiff files
system("rm *.tiff")