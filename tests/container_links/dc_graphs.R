library(ggplot2)
library(tidyverse)
#Make a loop for pod50 and 100

pallet <- c("#358359FF","#E5BA3AFF","#D86C4FFF" )

setwd(getwd())

names <- c("0","50","100")

metrics <- c("algorithm","clock","server-frag","server-load-cpu","server-load-ram","link-frag","link-load","bw")
metrics_labels <- c("","","Servers Fragmentation","Server Load CPU","Server Load RAM", "Link Fragmentation","Link Load","bw")


for(name in names){
    for(metrics_index in 3:7){
        data1 <- read.table(sprintf("dc/pod%s-bw1.json",name), sep=" ")
        data1$bw <- "<=01Mbps"
        data2 <- read.table(sprintf("dc/pod%s-bw25.json",name), sep= " ")
        data2$bw <- "<=25Mbps"
        data3 <- read.table(sprintf("dc/pod%s-bw50.json",name), sep= " ")
        data3$bw <- "<=50Mbps"
            
        all <- rbind(data1, data2)
        all <- rbind(all, data3)
            
        names(all) <- metrics
        
        df <- split(all,all$algorithm)
        for(alg in df){
            dens = split(alg, alg$bw) %>% map_df(function(d) {
                dens = density(
                    d[,metrics[metrics_index]], adjust=0.1, 
                    from=min(alg[,metrics[metrics_index]]) - 0.05*diff(range(alg[,metrics[metrics_index]])), 
                    to=max(alg[,metrics[metrics_index]]) + 0.05*diff(range(alg[,metrics[metrics_index]]))
                )
                    
                data.frame(x=dens$x, y=dens$y, cd=cumsum(dens$y)/sum(dens$y), group=d$bw[1])
            }
            )
            
            h <- ggplot() +
                stat_ecdf(data=alg, aes(alg[,metrics[metrics_index]]), alpha=0.0, lty="11") +
                geom_hline(yintercept=1, linetype="dashed", color = "grey")+
                geom_hline(yintercept=0, linetype="dashed", color = "grey")+
                geom_line(data=dens, aes(x, cd, colour=factor(group), group=group)) +
                theme_classic()+
                theme_bw()+ 
                theme(
                    legend.position="top",
                    axis.text.x = element_text(
                        angle = 0,
                        hjust = 0.7,
                        size=8,
                    ),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank(),
                    panel.background = element_blank(),
                    axis.line = element_line(colour = "black"),
                    axis.text.y = element_text(size=8), 
                    axis.title.x = element_text(size=10),
                    axis.title.y = element_text(size=10),
                    legend.text = element_text(size=8),
                    legend.title = element_text(size=8)
                )+
                labs(
                    x=metrics_labels[metrics_index],
                    y="CDF",
                    colour="Bandwidth"
                )+
                scale_color_manual(values=pallet)
                
            pdf(sprintf("%s/%s-%s-pod%s.pdf",metrics[metrics_index],alg$algorithm[1],metrics[metrics_index],name),width=4, height=4)
                
            print(h)
            dev.off()
        }
    }
}
