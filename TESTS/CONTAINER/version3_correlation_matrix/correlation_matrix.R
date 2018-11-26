#NEED TO PUT LEGEND IN THE X AND Y AXIS
#NEED TO PUT LEGEND IN THE COLOR BAR -  INDENTIFYNG THE TIME(SECONDS)

library(corrplot)
library(RColorBrewer)
library(yarrr)

file_names <- c("ahp","ahpg","mcl_ahp","mcl_ahpg","mcl","ahpg_clusterized")

plot_names <- c("AHP CPU","AHP CUDA","MCL CUDA + AHP CPU","MCL CUDA + AHP CUDA","MCL CUDA","AHP CUDA CLUSTERIZED")

index = 0

for(f_names in file_names){
    index=index+1
    file <- paste("times_",f_names,sep="")
    file <- paste(file,".csv",sep="")
    data <- read.table(file,header=T,sep=",")

    wide = reshape(
        data[,1:3],
        idvar = c("Fat_Tree_Size"),
        timevar="Number_of_Containers", direction = "wide")

    rownames(wide) = wide$Fat_Tree_Size
    wide=wide[-1]
    colnames(wide) = c(1,2,4,8,16,32,64,128,256,512,1024,2048)
    
    #wide<-t(apply(wide, 1, function(x)((x-min(x))/(max(x)-min(x)))))
    #wide<-t(apply(wide, 1, function(x)(2*(x-min(x))/(max(x)-min(x))-1)))
    
    #corrplot(title="\n\n\n\n\n\n\nAHP\nCorrelation Matrix",as.matrix(t(wide)), method="square", is.corr=FALSE,tl.srt=0, tl.col="black", col=brewer.pal(n = 12, name = "info2"))
    
    title_n <- "Correlation Matrix For "
    title_n <- paste(title_n,plot_names[index],sep="")

    files_save <- paste("correlation_matrix_of_",f_names,sep="")
    pdf(paste(files_save,".pdf",sep=""), width=8, height=6.8)

        corrplot(
            as.matrix(t(wide)),
            method="shade",
            type="full",
            is.corr=FALSE,
            tl.srt=0,
            tl.col="black",
            col=yarrr::piratepal("info2",trans=0, length.out = 14),
            cl.pos="r",
            cl.align.text="c",
            cl.ratio=0.2,
            number.digits=7,
            win.asp=1,
            na.label="X",
            addgrid.col="black"
            )+
            title(title_n, font.main=4, cex.main=1.7)+
            mtext("Container Request Size", side=2, line=2.5, cex=1.4)+
            mtext("Fat Tree Size", side=3, line=-1.9, cex=1.4)+
            mtext("                                                                                                                        Time(s)", side=3, line=-2.7 , cex=1.2)+
            mtext("                                                                                                                X values corresponds to execution\n                                                                                                              that needs more than 10 minutes", side=1, line=1)

    dev.off()
}

