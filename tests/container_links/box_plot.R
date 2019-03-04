library(ggplot2)
library(tidyverse)
#Make a loop for pod50 and 100

pallet <- c("#358359FF","#E5BA3AFF","#D86C4FFF" )

setwd(getwd())

metrics <- c("submission","id","delay","u-container","u-link","runtime","bw","gp")
metrics_labels <- c("","","Delay (events)","Container Utility", "Link Utility","Runtime (s)","","")


data1 <- read.table("../simulation/results-0/requests/pod0-bw0.json", sep=" ")
data1$bw <- "<=01Mbps"
data1$gp <- "Alpha 1"
data2 <- read.table("../simulation/results-0/requests/pod50-bw25.json", sep=" ")
data2$bw <- "<=25Mbps"
data2$gp <- "Alpha 1"
data3 <- read.table("../simulation/results-0/requests/pod100-bw50.json", sep=" ")
data3$bw <- "<=50Mbps"
data3$gp <- "Alpha 1"

data4 <- read.table("../simulation/results-1/requests/pod0-bw0.json", sep=" ")
data4$bw <- "<=01Mbps"
data4$gp <- "Alpha 2"
data5 <- read.table("../simulation/results-1/requests/pod50-bw25.json", sep=" ")
data5$bw <- "<=25Mbps"
data5$gp <- "Alpha 2"
data6 <- read.table("../simulation/results-1/requests/pod100-bw50.json", sep=" ")
data6$bw <- "<=50Mbps"
data6$gp <- "Alpha 2"

data7 <- read.table("../simulation/results-5/requests/pod0-bw0.json", sep=" ")
data7$bw <- "<=01Mbps"
data7$gp <- "Alpha 3"
data8 <- read.table("../simulation/results-5/requests/pod50-bw25.json", sep=" ")
data8$bw <- "<=25Mbps"
data8$gp <- "Alpha 3"
data9 <- read.table("../simulation/results-5/requests/pod100-bw50.json", sep=" ")
data9$bw <- "<=50Mbps"
data9$gp <- "Alpha 3"

all <- rbind(data1, data2)
all <- rbind(all, data3)
all <- rbind(all, data4)
all <- rbind(all, data5)
all <- rbind(all, data6)
all <- rbind(all, data7)
all <- rbind(all, data8)
all <- rbind(all, data9)
            
names(all) <- metrics

pdf("runtime/runtime.pdf",width=4, height=4)

boxplot(
    runtime ~ as.factor(gp), data=all,
    xlab = "",
    ylab = "Runtime", 
    main = "",
    notch = FALSE, 
    varwidth = TRUE, 
    outline=FALSE,
    col = pallet,
    names = c("Alpha 0","Alpha 1","Alpha 5")
)

#print(h)
dev.off()
