packages<- c("corrplot","RColorBrewer","gridExtra")

for(package in packages){
    if (!require(package,character.only = TRUE)){
        install.packages(package,dep=TRUE)
        if(!require(package,character.only = TRUE)) stop("Package not found")
    }
    library(package,character.only=TRUE)
}

pallet <- c("#006A40FF","#75B41EFF","#95828DFF","#708C98FF","#8AB8CFFF","#358359FF","#8BA1BCFF","#5A5895FF","#F2990CFF","#E5BA3AFF","#D86C4FFF" )

file_names <- c("data")

methods <- c("ahpg","topsis","ahpg_mcl","topsis_mcl")

index = 1

data <- read.table("data.csv",header=TRUE,sep=";")

for(method in methods){
    d<- data[data$method==method,]
    
    if(method!="pure_mcl"){
        d<-d[,c(2,3,5)]
    }else{
        d<-d[,c(2,3,4)]
    }
    wide = reshape(
        d[,1:3],
        idvar = c("fat_tree"),
        timevar="request",
        direction = "wide"
    )
    
    rownames(wide) = wide$fat_tree
    wide=wide[-1]
    colnames(wide) = c(1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144)

    wide=t(wide)
        
    files_save <- sprintf("correlation_matrix_of_%s.pdf",method)
    pdf(files_save, width=6.5, height=5)

    corrplot(
        as.matrix(wide),
        col=rep(pallet),
        method="color",
        type="full",
        is.corr=FALSE,
        tl.srt=0,
        tl.col="black",
        cl.pos="r",
        cl.align.text="c",
        cl.ratio=0.2,
        number.digits=4,
        win.asp=1,
        na.label="  ",
        addgrid.col="black"
    )+
    mtext("                                                                                                       Tempo(s)", line=2.4 , cex=1.2)+
    mtext("Tamanho da Requisição", side=2, line=2.5, cex=1.4)+
    mtext("Tamanho da Fat-Tree", side=3, line=2.7, cex=1.4)
    dev.off()
    
    system(sprintf("pdfcrop --margins '0 15 10 -1' %s %s",files_save,files_save))
}

make_summary <- function(consulta1){
    consulta1 <- na.omit(consulta1)
    meanx <- mean(consulta1)
    desviopadrao <- sd(consulta1)
    minx <-min(consulta1)
    listper <- quantile(consulta1, c(.01, .05,.95,.99))
    P1percentil <- listper[1][[1]]
    P5percentil <- listper[2][[1]]
    listquartil<-quantile(consulta1)
    P1quartil <- listquartil[2][[1]]
    medianax <-median(consulta1)
    P3quartil <- listquartil[4][[1]]
    P95percentil <- listper[3][[1]]
    P99percentil <- listper[4][[1]]
    maxx <-max(consulta1)
    
    data <-c(meanx,desviopadrao,minx,P1percentil,P5percentil,P1quartil,medianax,P3quartil,P95percentil,P99percentil,maxx)
    
    return (data)
}

df_mcl <- make_summary(data[data$method=="pure_mcl",4])
names(df_mcl) <- c("meanx","desviopadrao","minx","P1percentil","P5percentil","P1quartil","medianax","P3quartil","P95percentil","P99percentil","maxx")
df_mcl <- t(df_mcl)

pdf("summary-mcl.pdf", height=3, width=20)
grid.table(df_mcl)
dev.off()