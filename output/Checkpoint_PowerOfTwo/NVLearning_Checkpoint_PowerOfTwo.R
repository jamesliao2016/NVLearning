#################
# R code for analyzing output and plot figures
# v1.0 (organized on 2013-04-17)
# for Figure 4 in the paper
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
  setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output/Checkpoint_PowerOfTwo")

#read the output files
  data_A <- read.table("NVLearning_CheckpointA_PowerOfTwo.txt", header=TRUE)[c("M", "Pi_A")]
  data_B <- read.table("NVLearning_CheckpointB_PowerOfTwo.txt", header=TRUE)[c("M", "Pi_B")]

  Pi_E <- 27.450019
  Pi_T <- 27.66852677
  Pi_F <- 27.80394629
  Pi_Tm <- 27.63839435

#merge into one data frame
data <- merge(data_A, data_B, by=c("M"))
data <- data[data$M>1,]

#calculate profit gaps
data$Gap_A <- (data$Pi_A - Pi_E) / (Pi_T - Pi_E) * 100
data$Gap_B <- (data$Pi_B - Pi_E) / (Pi_T - Pi_E) * 100


#plot figure
pdf('Figure-Checkpoint-PowerOfTwo.pdf', width = 8, height = 8)

    xrange = range(log2(data$M))
    yrange = c(50,100)

    plot(xrange, yrange, type="n", xlab="Number of Checkpoints (M)", ylab="Profit Gap (%)", xaxt="n", yaxt="n")
    lines(log2(data$M), rep(100, length(data$M)), lty=5, lwd=1)
    #text(x=1.5, y=98, "Timing")
    #lines(log2(data$M), rep(0, length(data$M)), lty=5, lwd=1)
    #text(x=1, y=3, "Event")
    lines(log2(data$M), data$Gap_A, lty=1, lwd=1)
    points(log2(data$M), data$Gap_A, pch=7)
    #text(x=1.5, y=88, "Checkpoint-a[M]")
    lines(log2(data$M), data$Gap_B, lty=1, lwd=1)
    points(log2(data$M), data$Gap_B, pch=9)
    #text(x=1.5, y=57, "Checkpoint-b[M]")
    axis(side=1, at=log2(data$M), labels=data$M)
    axis(side=2, at=seq(50, 100, 10), labels=seq(50, 100, 10))
    legend(x="bottomright", inset=0, legend=c("Inventory Checkpoint [M]", "Stock-out Checkpoint [M]"), lty=c(1,1), lwd=c(1,1), pch=c(7,9), x.intersp=1.5, y.intersp=1.5)


dev.off()
