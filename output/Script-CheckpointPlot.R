#################
# R code for analyzing output and plot figures
# v1.0 (organized on 2013-04-17)
# for Figure 4 in the paper
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output")

#read the output files
data_F <- read.table("NVLearning_Full.txt", header=TRUE)[c("c", "alpha", "beta", "Q_F", "Pi_F")]
data_E <- read.table("NVLearning_Event.txt", header=TRUE)[c("c", "alpha", "beta", "Q_E", "Pi_E")]
data_T <- read.table("NVLearning_Timing.txt", header=TRUE)[c("c", "alpha", "beta", "Q_T", "Pi_T")]
data_Tm <- read.table("NVLearning_Timing_myopic.txt", header=TRUE)[c("c", "alpha", "beta", "Q_Tm", "Pi_Tm")]


#data_B2 <- read.table("NVLearning_CheckpointB_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Q_B2", "Pi_B2")]
data_B2m <- read.table("NVLearning_CheckpointB_myopic_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Q_B2m", "Pi_B2m")]
#data_B4 <- read.table("NVLearning_CheckpointB_M4.txt", header=TRUE)[c("c", "alpha", "beta", "Q_B4", "Pi_B4")]
data_B4m <- read.table("NVLearning_CheckpointB_myopic_M4.txt", header=TRUE)[c("c", "alpha", "beta", "Q_B4m", "Pi_B4m")]

#data_A2 <- read.table("NVLearning_CheckpointA_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Q_A2", "Pi_A2")]
data_A2m <- read.table("NVLearning_CheckpointA_myopic_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Q_A2m", "Pi_A2m")]
#data_A3m <- read.table("NVLearning_CheckpointA_myopic_M3.txt", header=TRUE)[c("c", "alpha", "beta", "Q_A3m", "Pi_A3m")]
data_A4m <- read.table("NVLearning_CheckpointA_myopic_M4.txt", header=TRUE)[c("c", "alpha", "beta", "Q_A4m", "Pi_A4m")]

#merge into one data frame
data <- merge(data_F, data_E, by=c("c", "alpha", "beta"))
data <- merge(data, data_T, by=c("c", "alpha", "beta"))
data <- merge(data, data_Tm, by=c("c", "alpha", "beta"))
#data <- merge(data, data_B2, by=c("c", "alpha", "beta"))
data <- merge(data, data_B2m, by=c("c", "alpha", "beta"))
#data <- merge(data, data_B4, by=c("c", "alpha", "beta"))
data <- merge(data, data_B4m, by=c("c", "alpha", "beta"))
#data <- merge(data, data_A2, by=c("c", "alpha", "beta"))
data <- merge(data, data_A2m, by=c("c", "alpha", "beta"))
#data <- merge(data, data_A3m, by=c("c", "alpha", "beta"))
data <- merge(data, data_A4m, by=c("c", "alpha", "beta"))





#calculate profit gaps
Profit_Gap <- data[c("c", "alpha", "beta")]
Profit_Gap$Gap_F <- (data$Pi_F - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_E <- (data$Pi_E - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_T <- (data$Pi_T - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_Tm <- (data$Pi_Tm - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
#Profit_Gap$Gap_B2 <- (data$Pi_B2 - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_B2m <- (data$Pi_B2m - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
#Profit_Gap$Gap_B4 <- (data$Pi_B4 - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_B4m <- (data$Pi_B4m - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
#Profit_Gap$Gap_A2 <- (data$Pi_A2 - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_A2m <- (data$Pi_A2m - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
#Profit_Gap$Gap_A3m <- (data$Pi_A3m - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_A4m <- (data$Pi_A4m - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100


#figure 5: aggreage profit gaps
Profit_Gap_agg_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=mean, na.rm=TRUE)
Profit_Gap_agg_c$fractile <- (2-Profit_Gap_agg_c$c)/2

pdf('Figure-Profit-checkpoint.pdf', width = 8, height = 8)

    xrange5 = range(Profit_Gap_agg_c$fractile)
    yrange5 = c(0, 100)

    plot(xrange5, yrange5, type="n", xlab="Newsvendor Ratio", ylab="Loss Recovery (%)", xaxt="n", yaxt="n")
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_F, lty=5, lwd=0.5)
    text(x=0.27, y=98, "Full")
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_E, lty=5, lwd=0.5)
    text(x=0.27, y=2, "Event")
    
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_T, lty=1, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_T, pch=19)
    #text(x=0.27, y=78, expression(eta^T), cex=1.5)
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_Tm, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_Tm, pch=1)
    #text(x=0.27, y=70, expression(hat(eta)^T), cex=1.5)

    #lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_B4, lty=1, lwd=2)
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_B4m, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_B4m, pch=17)
    #text(x=0.28, y=62, expression(hat(eta)^"SC[4]"), cex=1.5)

    #lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_B2, lty=1, lwd=2)
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_B2m, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_B2m, pch=2)
    #text(x=0.28, y=23, expression(hat(eta)^"SC[2]"), cex=1.5)


    #lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A4, lty=1, lwd=2)
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A4m, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A4m, pch=15)
    #text(x=0.28, y=48, expression(hat(eta)^"IC[4]"), cex=1.5)

    #lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A2, lty=1, lwd=2)
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A2m, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A2m, pch=0)
    #text(x=0.28, y=48, expression(hat(eta)^"IC[2]"), cex=1.5)

    #lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_A3m, lty=2, lwd=1)

    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, at=seq(0, 100, 10), labels=seq(0, 100, 10), las=1)

legend(x="topright", inset=0.04, ncol=3,
       legend=c( expression(eta^T), expression(hat(eta)^T), 
                 expression(hat(eta)^"SC[4]"), expression(hat(eta)^"SC[2]"), 
                 expression(hat(eta)^"IC[4]"), expression(hat(eta)^"IC[2]")), 
       lty=c(1,2,2,2,2,2), lwd=c(1,1,1,1,1,1), pch=c(19,1,17,2,15,0), x.intersp=1.3, y.intersp=1.3, cex=1.2)

dev.off()


summary(Profit_Gap)