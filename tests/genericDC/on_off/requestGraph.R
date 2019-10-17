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

headers <- c("method","submission","id","delay","task_utility","link_utility","link_footprint","requested_time","start_time","stop_time","scheduling_time","type")

is.nan.data.frame <- function(x)
    do.call(cbind, lapply(x, is.nan))

# sizes = c("50k")
sizes = c("50k","100k","150k")
for(size in sizes){
    #######################################
    ## 1070 TI ##
    #######################################
    gtxDataOnline <- read.table(sprintf("data/gtx1070ti/request-none-lyon-%s-fcfs-online-lyon-%s.log",size,size),header=FALSE,sep=" ")
    gtxDataOffline <- read.table(sprintf("data/gtx1070ti/request-none-lyon-%s-fcfs-offline-lyon-%s.log",size,size),header=FALSE,sep=" ")
    
    gtxDataOnline$type = "Leo w/o resubmission"
    gtxDataOffline$type = "Leo w/ resubmission"
    
    gtxData <- rbind(gtxDataOnline, gtxDataOffline)
    
    rm(gtxDataOnline)
    rm(gtxDataOffline)
    
    names(gtxData) <- headers
    gtxData$gpu <- "GTX 1070 Ti"
    
    #######################################
    ## Titan XP ##
    #######################################
    titanXPDataOnline <- read.table(sprintf("data/titanXP/request-none-lyon-%s-fcfs-online-lyon-%s.log",size,size),header=FALSE,sep=" ")
    titanXPDataOffline <- read.table(sprintf("data/titanXP/request-none-lyon-%s-fcfs-offline-lyon-%s.log",size,size),header=FALSE,sep=" ")
    
    titanXPDataOnline$type = "Leo w/o resubmission"
    titanXPDataOffline$type = "Leo w/ resubmission"
    
    titanXPData <- rbind(titanXPDataOnline, titanXPDataOffline)
    
    rm(titanXPDataOnline)
    rm(titanXPDataOffline)
    
    names(titanXPData) <- headers
    titanXPData$gpu <- "Titan XP"
    
    #######################################
    ## 2080 TI ##
    #######################################
    rtxDataOnline <- read.table(sprintf("data/rtx2080ti/request-none-lyon-%s-fcfs-online-lyon-%s.log",size,size),header=FALSE,sep=" ")
    rtxDataOffline <- read.table(sprintf("data/rtx2080ti/request-none-lyon-%s-fcfs-offline-lyon-%s.log",size,size),header=FALSE,sep=" ")
    
    rtxDataOnline$type = "Leo w/o resubmission"
    rtxDataOffline$type = "Leo w/ resubmission"
    
    rtxData <- rbind(rtxDataOnline, rtxDataOffline)
    
    rm(rtxDataOnline)
    rm(rtxDataOffline)
    
    names(rtxData) <- headers
    rtxData$gpu <- "RTX 2080 Ti"
    
    #######################################
    ## Grid5000 TI ##
    #######################################
    easy <- read.table(sprintf("../easy_traces/easy_trace_%s.log",size),header=FALSE,sep=" ")
    easy$type = "Easy backfilling"
    
    names(easy) <- headers
    easy$gpu <- "N/A"
    
    #######################################
    ## Merge all the data ##
    #######################################
    
    dt <- rbind(gtxData, titanXPData)
    dt <- rbind(dt, rtxData)
    dt <- rbind(dt, easy)
    
    ###############################
    ## Create the new variables ###
    ###############################
    
    dt$waiting_time <- dt$start_time - dt$requested_time
    dt$wall_time <- dt$stop_time - dt$start_time
    
    div_temp <- mapply(max, dt$wall_time, rep(0.001, length(dt$wall_time)))
    dt$slowdown <- (dt$waiting_time + dt$wall_time) / div_temp
    dt$slowdown <- mapply(max, dt$slowdown, 1)
    rm(div_temp)
    
    dt[is.nan(dt)] <- 0
    dt$slowdown[which(!is.finite(dt$slowdown))] <- 0
    
    #############################################
    ## Filter the Data by 10 and 90 percentile ##
    #############################################
    
    quant <- as.numeric(quantile(dt$slowdown, c(.1,.9)))
    dt <- dt[dt$slowdown > quant[1] & dt$slowdown< quant[2],]
    rm(quant)
    
    #######################################
    ## Plot the walltime cdf graph ##
    #######################################
    walltimeCDFPlot <- ggplot(data=dt, aes( wall_time, color= type, linetype= gpu))+
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
            x="Walltime (s)",
            y="CDF (%)",
            color="Scheduling Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the walltime cdf graph ##
    #######################################
    walltimeEventsPlot <- ggplot(data=dt, aes( x=submission, y=wall_time, color= type, linetype= gpu))+
        geom_line() +
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
            x="Submission",
            y="Walltime (Events)",
            color="Scheduling Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the slowdown cdf graph ##
    #######################################
    slowdownCDFPlot <- ggplot(data=dt, aes( slowdown, color= type, linetype= gpu))+
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
            x="Slowdown (s)",
            y="CDF (%)",
            color="Scheduling Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the slowdown events graph ##
    #######################################
    slowdownEventsPlot <- ggplot(data=dt, aes( x=submission , y=slowdown, color= type, linetype= gpu))+
        geom_line() +
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
            x="Submission",
            y="Slowdown (Events)",
            color="Scheduling Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the delay cdf graph ##
    #######################################
    delayCDFPlot <- ggplot(data=dt, aes( delay, color= type, linetype=gpu))+
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
            color="Scheduling Type",
            linetype = "GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the walltime by delay graph ##
    #######################################
    walltimeByDelayPlot <- ggplot(data=dt, aes( x = wall_time, color= type, linetype = gpu))+
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
    
    tiff(sprintf("walltime_events_%s.tiff",size),width= 3500, height= 1200, units="px", res=400,compression = 'lzw')
    plot(walltimeEventsPlot)
    dev.off()
    
    tiff(sprintf("slowdown_cdf_%s.tiff",size),width= 3500, height= 1200, units="px", res=400,compression = 'lzw')
    plot(slowdownCDFPlot)
    dev.off()
    
    tiff(sprintf("slowdown_events_%s.tiff",size),width= 3500, height= 1200, units="px", res=400,compression = 'lzw')
    plot(slowdownEventsPlot)
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
#system("for f in *.tiff; do convert -trim $f ${f%.*}.eps; done;")

#move the files to the graphs folder
system("mv *.png graphs")
#system("mv *.eps graphs")

#remove all tiff files
system("rm *.tiff")