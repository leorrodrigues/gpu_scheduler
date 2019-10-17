packages <- c("ggplot2","ggsci", "dplyr", "purrr")

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

#sizes <- c("1k","10k","50k","100k","150k")
sizes <- c("1k","10k")
queues <- c("fcfs","safmin","spf","sqfmin")
for(size in sizes){
    dt <- data.frame(matrix(vector(), 0, length(headers)+1, dimnames=list(c(), c(headers,"gpu"))), stringsAsFactors=F)
    for(queue in queues){
        #######################################
        ## 1070 TI ##
        #######################################
        # gtxDataOnline <- read.table(sprintf("data/gtx1070ti/request-none-lyon-%s-%s-online-lyon-%s.log",size,queue,size),header=FALSE,sep=" ")
        # gtxDataOffline <- read.table(sprintf("data/gtx1070ti/request-none-lyon-%s-%s-offline-lyon-%s.log",size,queue,size),header=FALSE,sep=" ")
        # 
        # gtxDataOnline$type = "online"
        # gtxDataOffline$type = "offline"
        # 
        # gtxData <- rbind(gtxDataOnline, gtxDataOffline)
        # 
        # rm(gtxDataOnline)
        # rm(gtxDataOffline)
        # 
        # names(gtxData) <- headers
        # gtxData$gpu <- "GTX 1070 Ti"
        # gtxData$queue <- queue
        
        #######################################
        ## Titan XP ##
        #######################################
        # titanXPDataOnline <- read.table(sprintf("data/titanXP/request-none-lyon-%s-%s-online-lyon-%s.log",size,queue,size),header=FALSE,sep=" ")
        # titanXPDataOffline <- read.table(sprintf("data/titanXP/request-none-lyon-%s-%s-offline-lyon-%s.log",size,queue,size),header=FALSE,sep=" ")
        # 
        # titanXPDataOnline$type = "online"
        # titanXPDataOffline$type = "offline"
        # 
        # titanXPData <- rbind(titanXPDataOnline, titanXPDataOffline)
        # 
        # rm(titanXPDataOnline)
        # rm(titanXPDataOffline)
        # 
        # names(titanXPData) <- headers
        # titanXPData$gpu <- "Titan XP"
        # titanXPData$queue <- queue
        
        #######################################
        ## 2080 TI ##
        #######################################
        rtxDataOnline <- read.table(sprintf("data/rtx2080ti/request-none-lyon-%s-%s-online-lyon-%s.log",size,queue,size),header=FALSE,sep=" ")
        # rtxDataOffline <- read.table(sprintf("data/rtx2080ti/request-none-lyon-%s-%s-offline-lyon-%s.log",size,queue,size),header=FALSE,sep=" ")
        
        rtxDataOnline$type = "online"
        # rtxDataOffline$type = "offline"
        
        # rtxData <- rbind(rtxDataOnline, rtxDataOffline)
        rtxData <- rtxDataOnline
            
        rm(rtxDataOnline)
        # rm(rtxDataOffline)
        
        names(rtxData) <- headers
        rtxData$gpu <- "RTX 2080 Ti"
        rtxData$queue <- queue
        
        #######################################
        ## Merge all the data ##
        #######################################
        
        # dt_temp <- rbind(gtxData, titanXPData)
        # dt_temp <- rbind(dt_temp, rtxData)
        
        # rm(gtxData)
        # rm(titanXPData)
        dt <- rbind(dt, rtxData)
        rm(rtxData)
        
        # dt <- rbind(dt, dt_temp)
    }    
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
    # walltimeCDFPlot <- ggplot(data=dt, aes( wall_time, color= queue, linetype= type))+
    
    dens = split(dt, dt$queue) %>%
        map_df(function(d) {
            dens = density(dt$wall_time, adjust=0.1, from=min(dt$wall_time) - 0.05*diff(range(dt$wall_time)), to=max(dt$wall_time) + 0.05*diff(range(dt$wall_time)))
            data.frame(x=dens$x, y=dens$y, cd=cumsum(dens$y)/sum(dens$y), group=d$queue[1])
        })
    
    walltimeCDFPlot <- ggplot()+
        stat_ecdf(data=dt, aes(wall_time, colour=queue), alpha=0.8, lty="11") +
        #geom_line(data=dens, aes(x, cd, colour=group)) +
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
            color="Queue Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the walltime cdf graph ##
    #######################################
    # walltimeEventsPlot <- ggplot(data=dt, aes( x=submission, y=wall_time, color= queue, linetype= type))+
    walltimeEventsPlot <- ggplot(data=dt, aes( x=submission, y=wall_time, color= queue))+
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
            color="Queue Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the slowdown cdf graph ##
    #######################################
    # slowdownCDFPlot <- ggplot(data=dt, aes( slowdown, color= queue, linetype= type))+
    dens = split(dt, dt$queue) %>%
        map_df(function(d) {
            dens = density(dt$slowdown, adjust=0.1, from=min(dt$slowdown) - 0.05*diff(range(dt$slowdown)), to=max(dt$slowdown) + 0.05*diff(range(dt$slowdown)))
            data.frame(x=dens$x, y=dens$y, cd=cumsum(dens$y)/sum(dens$y), group=d$queue[1])
        })
    
    slowdownCDFPlot <- ggplot()+
        stat_ecdf(data=dt, aes(slowdown, colour=queue), alpha=0.8, lty="11") +
        #geom_line(data=dens, aes(x, cd, colour=group)) +
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
            color="Queue Type",
            linetype="GPU"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the slowdown events graph ##
    #######################################
    # slowdownEventsPlot <- ggplot(data=dt, aes( x=submission , y=slowdown, color=  slowdown, color= queue, linetype= type))+
    slowdownEventsPlot <- ggplot(data=dt, aes( x=submission , y=slowdown, color= queue))+
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
            y="Slowdown (s)",
            color="Queue Type"
        )+
        scale_color_d3()
    
    #######################################
    ## Plot the delay cdf graph ##
    #######################################
    # delayCDFPlot <- ggplot(data=dt, aes( delay, color= queue, linetype=type))+
    
    dens = split(dt, dt$queue) %>%
        map_df(function(d) {
            dens = density(dt$delay, adjust=0.1, from=min(dt$delay) - 0.05*diff(range(dt$delay)), to=max(dt$delay) + 0.05*diff(range(dt$delay)))
            data.frame(x=dens$x, y=dens$y, cd=cumsum(dens$y)/sum(dens$y), group=d$queue[1])
        })
    
    delayCDFPlot <- ggplot()+
        stat_ecdf(data=dt, aes(delay, colour=queue), alpha=0.8, lty="11") +
        #geom_line(data=dens, aes(x, cd, colour=group)) +
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
            color="Queue Type"
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
# system("for f in *.tiff; do convert -trim $f ${f%.*}.eps; done;")

#move the files to the graphs folder
system("mv *.png graphs")
# system("mv *.eps graphs")

#remove all tiff files
system("rm *.tiff")