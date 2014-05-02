#################
# R code for analyzing output and plot figures
# v3.0 (organized on 2014-05-02) for revision 2
# for Figure 5a (convergence of the checkpoint models) in the paper
# published on https://github.com/tong-wang/NVLearning
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output/Checkpoint_PowerOfTwo")


#read the output files
data_A <- read.table("NVLearning_CheckpointA.PowerOfTwo.txt", header=TRUE)[c("M", "Pi_A")]
data_B <- read.table("NVLearning_CheckpointB.PowerOfTwo.txt", header=TRUE)[c("M", "Pi_B")]
data_Am <- read.table("NVLearning_CheckpointA_myopic.PowerOfTwo.txt", header=TRUE)[c("M", "Pi_Am")]
data_Bm <- read.table("NVLearning_CheckpointB_myopic.PowerOfTwo.txt", header=TRUE)[c("M", "Pi_Bm")]

Pi_F <- 27.80394629
Pi_E <- 27.45002563
Pi_Em <- 27.3749171
Pi_T <- 27.66852677
Pi_Tm <- 27.63839435


#merge into one data frame
data <- merge(data_A, data_B, by=c("M"))
data <- merge(data, data_Am, by=c("M"))
data <- merge(data, data_Bm, by=c("M"))

#calculate profit gaps
data$Gap_A <- (data$Pi_A - Pi_E) / (Pi_F - Pi_E) * 100
data$Gap_B <- (data$Pi_B - Pi_E) / (Pi_F - Pi_E) * 100
data$Gap_Am <- (data$Pi_Am - Pi_E) / (Pi_F - Pi_E) * 100
data$Gap_Bm <- (data$Pi_Bm - Pi_E) / (Pi_F - Pi_E) * 100
Gap_T <- (Pi_T - Pi_E) / (Pi_F - Pi_E) * 100
Gap_Tm <- (Pi_Tm - Pi_E) / (Pi_F - Pi_E) * 100
Gap_Em <- (Pi_Em - Pi_E) / (Pi_F - Pi_E) * 100


xrange = range(log2(data$M))
#    yrange = c(0,100)
yrange = c(-25,100)

#plot figure 5a
pdf('Figure-Checkpoint-PowerOfTwo.pdf', width = 8, height = 8)

    plot(xrange, yrange, type="n", xlab="Number of Checkpoints (M)", ylab="Loss Recovery (%)", xaxt="n", yaxt="n")
    
    #F model as upper bound (100%)
    lines(log2(data$M), rep(100, length(data$M)), lty=5, lwd=0.5, col="grey10")
    text(x=1, y=97, "Full", col="grey50")
    
    #E model as lower bound (0%)
    lines(log2(data$M), rep(0, length(data$M)), lty=5, lwd=0.5, col="grey10")
    text(x=1, y=3, expression(paste("Event (", eta^"E", ")")), col="grey50")

    #E model as lower bound (0%)
    lines(log2(data$M), rep(Gap_Em, length(data$M)), lty=5, lwd=0.5, col="grey10")
    text(x=1, y=-18, expression(paste("Myopic Event (", hat(eta)^"E", ")")), col="grey50")

    #T model as the limit of IC[M] model
    lines(log2(data$M), rep(Gap_T, length(data$M)), lty=5, lwd=0.5, col="grey10")
    text(x=1, y=65, expression(paste("Timing (", eta^"T", ")")), col="grey50")

    #T model as the limit of IC[M] model
    lines(log2(data$M), rep(Gap_Tm, length(data$M)), lty=5, lwd=0.5, col="grey10")
    text(x=1, y=56, expression(paste("Myopic Timing (", hat(eta)^"T", ")")), col="grey50")


    #IC[M] model
    lines(log2(data$M), data$Gap_A, lty=1, lwd=2)
    points(log2(data$M), data$Gap_A, pch=15)
    
    #IC[M]m model
    lines(log2(data$M), data$Gap_Am, lty=2, lwd=1)
    points(log2(data$M), data$Gap_Am, pch=0)
    #text(x=1, y=88, "Inventory Checkpoint [M]")
    
    #SC[M] model
    lines(log2(data$M), data$Gap_B, lty=1, lwd=2)
    points(log2(data$M), data$Gap_B, pch=17)
    
    #SC[M]m model
    lines(log2(data$M), data$Gap_Bm, lty=2, lwd=1)
    points(log2(data$M), data$Gap_Bm, pch=2)
    #text(x=1, y=25, "Stock-out Checkpoint [M]")
    
    axis(side=1, at=log2(data$M), labels=data$M)
#    axis(side=2, at=seq(50, 100, 10), labels=seq(50, 100, 10), las=1)
    axis(side=2, at=seq(-25, 100, 25), labels=seq(-25, 100, 25), las=1)
    
    #legend(x="bottomright", inset=0.04, legend=c("Inventory Checkpoint [M]", "Stock-out Checkpoint [M]"), lty=c(1,1), lwd=c(1,1), pch=c(9,5), x.intersp=1.5, y.intersp=1.5)
    legend(x="bottomright", inset=0, ncol=2,
        legend=c(expression(eta^"IC[M]"), 
                expression(eta^"SC[M]"), 
                expression(hat(eta)^"IC[M]"), 
                expression(hat(eta)^"SC[M]")), 
        lty=c(1,1,2,2), lwd=c(2,2,1,1), pch=c(15,17,0,2), x.intersp=1.5, y.intersp=1.5, cex=1.2, bg="white")

dev.off()
