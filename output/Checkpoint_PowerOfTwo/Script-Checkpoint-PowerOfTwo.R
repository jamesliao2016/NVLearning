#################
# R code for analyzing output and plot figures
# v1.0 (organized on 2013-09-14)
# for Figure 5a in the paper
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
  Pi_T <- 27.66852677
  Pi_Tm <- 27.63839435

#merge into one data frame
data <- merge(data_A, data_B, by=c("M"))
data <- merge(data, data_Am, by=c("M"))
data <- merge(data, data_Bm, by=c("M"))
#data <- data[data$M>1,]

#calculate profit gaps
data$Gap_A <- (data$Pi_A - Pi_E) / (Pi_F - Pi_E) * 100
data$Gap_B <- (data$Pi_B - Pi_E) / (Pi_F - Pi_E) * 100
data$Gap_Am <- (data$Pi_Am - Pi_E) / (Pi_F - Pi_E) * 100
data$Gap_Bm <- (data$Pi_Bm - Pi_E) / (Pi_F - Pi_E) * 100
Gap_T <- (Pi_T - Pi_E) / (Pi_F - Pi_E) * 100


#plot figure
pdf('Figure-Checkpoint-PowerOfTwo.pdf', width = 8, height = 8)

    xrange = range(log2(data$M))
#    yrange = c(0,100)
    yrange = c(-25,100)

    plot(xrange, yrange, type="n", xlab="Number of Checkpoints (M)", ylab="Loss Recovery (%)", xaxt="n", yaxt="n")
    lines(log2(data$M), rep(100, length(data$M)), lty=5, lwd=0.5)
    text(x=1, y=97, "Full")
    lines(log2(data$M), rep(Gap_T, length(data$M)), lty=5, lwd=0.5)
    text(x=1, y=65, "Timing")
    lines(log2(data$M), rep(0, length(data$M)), lty=5, lwd=0.5)
    text(x=1, y=3, "Event")
    lines(log2(data$M), data$Gap_A, lty=1, lwd=1)
    points(log2(data$M), data$Gap_A, pch=15)
    lines(log2(data$M), data$Gap_Am, lty=2, lwd=1)
    points(log2(data$M), data$Gap_Am, pch=0)
    #text(x=1, y=88, "Inventory Checkpoint [M]")
    lines(log2(data$M), data$Gap_B, lty=1, lwd=1)
    points(log2(data$M), data$Gap_B, pch=17)
    lines(log2(data$M), data$Gap_Bm, lty=2, lwd=1)
    points(log2(data$M), data$Gap_Bm, pch=2)
    #text(x=1, y=25, "Stock-out Checkpoint [M]")
    axis(side=1, at=log2(data$M), labels=data$M)
#    axis(side=2, at=seq(50, 100, 10), labels=seq(50, 100, 10), las=1)
    axis(side=2, at=seq(-25, 100, 25), labels=seq(-25, 100, 25), las=1)
    #legend(x="bottomright", inset=0.04, legend=c("Inventory Checkpoint [M]", "Stock-out Checkpoint [M]"), lty=c(1,1), lwd=c(1,1), pch=c(9,5), x.intersp=1.5, y.intersp=1.5)
legend(x="bottomright", inset=0.02, ncol=2,
       legend=c(expression(eta^"IC[M]"), expression(eta^"SC[M]"), expression(hat(eta)^"IC[M]"), expression(hat(eta)^"SC[M]")), lty=c(1,1,2,2), lwd=c(1,1,1,1), pch=c(15,17,0,2), x.intersp=1.5, y.intersp=1.5, cex=1.2)


dev.off()
