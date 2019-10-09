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

sizes = c("50k","100k","150k")
for(size in sizes){
    #######################################
    ## 1070 TI ##
    #######################################
    gtxDataOnline <- read.table(sprintf("data/gtx1070ti/request-none-lyon-%s-fcfs-online-lyon-%s.log",size,size),header=FALSE,sep=" ")
    gtxDataOffline <- read.table(sprintf("data/gtx1070ti/request-none-lyon-%s-fcfs-offline-lyon-%s.log",size,size),header=FALSE,sep=" ")
    
    gtxDataOnline$type = "online"
    gtxDataOffline$type = "offline"
    
    gtxData <- rbind(gtxDataOnline, gtxDataOffline)
    
    rm(gtxDataOnline)
    rm(gtxDataOffline)
    
    names(gtxData) <- c("method","submission","id","delay","task_utility","link_utility","walltime","delay_link","footprint_bandwidth","type")
    gtxData$gpu <- "GTX 1070 Ti"
    
    #######################################
    ## Titan XP ##
    #######################################
    titanXPDataOnline <- read.table(sprintf("data/titanXP/request-none-lyon-%s-fcfs-online-lyon-%s.log",size,size),header=FALSE,sep=" ")
    titanXPDataOffline <- read.table(sprintf("data/titanXP/request-none-lyon-%s-fcfs-offline-lyon-%s.log",size,size),header=FALSE,sep=" ")
    
    titanXPDataOnline$type = "online"
    titanXPDataOffline$type = "offline"
    
    titanXPData <- rbind(titanXPDataOnline, titanXPDataOffline)
    
    rm(titanXPDataOnline)
    rm(titanXPDataOffline)
    
    names(titanXPData) <- c("method","submission","id","delay","task_utility","link_utility","walltime","delay_link","footprint_bandwidth","type")
    titanXPData$gpu <- "Titan XP"
    
    #######################################
    ## 2080 TI ##
    #######################################
    rtxDataOnline <- read.table(sprintf("data/rtx2080ti/request-none-lyon-%s-fcfs-online-lyon-%s.log",size,size),header=FALSE,sep=" ")
    rtxDataOffline <- read.table(sprintf("data/rtx2080ti/request-none-lyon-%s-fcfs-offline-lyon-%s.log",size,size),header=FALSE,sep=" ")
    
    rtxDataOnline$type = "online"
    rtxDataOffline$type = "offline"
    
    rtxData <- rbind(rtxDataOnline, rtxDataOffline)
    
    rm(rtxDataOnline)
    rm(rtxDataOffline)
    
    names(rtxData) <- c("method","submission","id","delay","task_utility","link_utility","walltime","delay_link","footprint_bandwidth","type")
    rtxData$gpu <- "RTX 2080 Ti"
    
    #######################################
    ## Merge all the data ##
    #######################################
    
    dt <- rbind(gtxData, titanXPData)
    dt <- rbind(dt, rtxData)
    
    rm(gtxData)
    rm(titanXPData)
    rm(rtxData)
    
    #######################################
    ## Plot the walltime cdf graph ##
    #######################################
    walltimeCDFPlot <- ggplot(data=dt, aes( walltime, color= type, linetype= gpu))+
        stat_ecdf(geom="step") +
        theme_classic()+
        theme(
            legend.position="right",
            axis.text.x = element_text(
                angle = 0,
                hjust = 0.7,
                size=12
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
            x="Walltime",
            y="CDF (%)",
            color="Scheduling Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the delay cdf graph ##
    #######################################
    delayCDFPlot <- ggplot(data=dt, aes( delay, color= type))+
        stat_ecdf(geom="step") +
        theme_classic()+
        theme(
            legend.position="right",
            axis.text.x = element_text(
                angle = 0,
                hjust = 0.7,
                size=12
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
            x="Delay (Events)",
            y="CDF (%)",
            color="Scheduling Type"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the walltime by delay graph ##
    #######################################
    walltimeByDelayPlot <- ggplot(data=dt, aes( x = walltime, color= type, linetype = gpu))+
        stat_ecdf()+
        theme_classic()+
        theme(
            legend.position="right",
            axis.text.x = element_text(
                angle = 0,
                hjust = 0.7,
                size=12
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
            x= "Delay (Events)",
            y="CDF (%)",
            color="Scheduling Type",
            linetype = "GPU"
        )+
        scale_color_d3()
    
    ###################################
    ## Plot all the graphs
    ###################################
    
    tiff(sprintf("walltime_cdf_%s.tiff",size),width= 3500, height= 1200, units="px", res=400,compression = 'lzw')
    plot(walltimeCDFPlot)
    dev.off()
    
    tiff(sprintf("delay_cdf_%s.tiff",size),width= 3500, height= 1200, units="px", res=400,compression = 'lzw')
    plot(delayCDFPlot)
    dev.off()
    
    tiff(sprintf("walltime_delay_%s.tiff",size),width= 3500, height= 1200, units="px", res=400,compression = 'lzw')
    plot(walltimeByDelayPlot)
    dev.off()
}
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