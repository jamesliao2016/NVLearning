#################
# R code for analyzing output and plot figures
# v1.0 (organized on 2013-04-17)
# for Figure 4 in the paper
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output")
source(file="legend2.R")

#read the output files
data_F <- read.table("NVLearning_Full.txt", header=TRUE)[c("c", "alpha", "beta", "Q_F", "Pi_F")]
data_E <- read.table("NVLearning_Event.txt", header=TRUE)[c("c", "alpha", "beta", "Q_E", "Pi_E")]
data_T <- read.table("NVLearning_Timing.txt", header=TRUE)[c("c", "alpha", "beta", "Q_T", "Pi_T")]
data_Tm <- read.table("NVLearning_Timing_myopic.txt", header=TRUE)[c("c", "alpha", "beta", "Q_Tm", "Pi_Tm")]


#merge into one data frame
data <- merge(data_F, data_E, by=c("c", "alpha", "beta"))
data <- merge(data, data_T, by=c("c", "alpha", "beta"))
data <- merge(data, data_Tm, by=c("c", "alpha", "beta"))


#figure 3: optimal profit in a typical scenario
data_typical = data[data$alpha==0.625&data$beta==0.0625,]
data_typical$fractile = (2-data_typical$c)/2


pdf('Figure-Profit-typical.pdf', width = 8, height = 8)

xrange3 = range(data_typical$fractile)
yrange3 = range(data_typical[c("Pi_F", "Pi_E", "Pi_T")])

plot(xrange3, yrange3, type="n", xlab="Newsvendor Ratio", ylab="Optimal Profit", xaxt="n", yaxt="n")
lines(data_typical$fractile, data_typical$Pi_F, lty=2, lwd=1)
points(data_typical$fractile, data_typical$Pi_F, pch=20)
lines(data_typical$fractile, data_typical$Pi_E, lty=1, lwd=1)
points(data_typical$fractile, data_typical$Pi_E, pch=1)
lines(data_typical$fractile, data_typical$Pi_T, lty=1, lwd=1)
points(data_typical$fractile, data_typical$Pi_T, pch=2)
axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
axis(side=2, las=1)
legend(x="topleft", inset=0, legend=c(expression(V[1]^{E}), expression(V[1]^{T}), expression(V[1]^{F})), lty=c(1,1,2), lwd=c(1,1,1), pch=c(1,2,20), x.intersp=1.5, y.intersp=1.5)

dev.off()





#calculate profit gaps
Profit_Gap <- data[c("c", "alpha", "beta")]
Profit_Gap$Gap_T <- (data$Pi_T - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
#Profit_Gap$Gap_Tm <- (data$Pi_Tm - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100




#figure 4: aggreage profit gaps
Profit_Gap_mean_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=mean, na.rm=TRUE)[c("c", "Gap_T")]
colnames(Profit_Gap_mean_by_c) <- c("c", "Gap_T_mean")
Profit_Gap_95quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.95, na.rm=TRUE)[c("c", "Gap_T")]
colnames(Profit_Gap_95quantile_by_c) <- c("c", "Gap_T_95quantile")
Profit_Gap_05quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.05, na.rm=TRUE)[c("c", "Gap_T")]
colnames(Profit_Gap_05quantile_by_c) <- c("c", "Gap_T_05quantile")
Profit_Gap_75quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.75, na.rm=TRUE)[c("c", "Gap_T")]
colnames(Profit_Gap_75quantile_by_c) <- c("c", "Gap_T_75quantile")
Profit_Gap_25quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.25, na.rm=TRUE)[c("c", "Gap_T")]
colnames(Profit_Gap_25quantile_by_c) <- c("c", "Gap_T_25quantile")

Profit_Gap_agg_c <- merge(Profit_Gap_mean_by_c, Profit_Gap_95quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_05quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_75quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_25quantile_by_c, by=c("c"))
Profit_Gap_agg_c$fractile <- (2-Profit_Gap_agg_c$c)/2


##plot confidence interval as an area plot by using PLOYGON()  
xx <- c(rev(Profit_Gap_agg_c$fractile), Profit_Gap_agg_c$fractile)
yy90 <- c(rev(Profit_Gap_agg_c$Gap_T_05quantile), Profit_Gap_agg_c$Gap_T_95quantile)
yy50 <- c(rev(Profit_Gap_agg_c$Gap_T_25quantile), Profit_Gap_agg_c$Gap_T_75quantile)

pdf('Figure-Profit-aggregate.pdf', width = 8, height = 8)

    xrange4 = range(Profit_Gap_agg_c$fractile)
    yrange4 = c(0, 100)

    plot(xrange4, yrange4, type="n", xlab="Newsvendor Ratio", ylab="Profit Gap (%)", xaxt="n", yaxt="n")
    #lines(Profit_Gap_agg_c$fractile, rep(100, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=1)
    #text(x=0.25, y=97, "Full")
    #lines(Profit_Gap_agg_c$fractile, rep(0, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=1)
    #text(x=0.25, y=3, "Event")
    polygon(xx, yy90, col="grey90", border=NA)
    polygon(xx, yy50, col="grey80", border=NA)
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_T_mean, lty=1, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_T_mean, pch=2)
    #text(x=0.25, y=78, "Timing")
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, at=seq(0, 100, 10), labels=seq(0, 100, 10), las=1)
    legend2(x="bottomright", inset=0,
        legend=c(expression(paste("mean of ", Delta^T, sep="")), 
                 expression(paste("50% interval of ", Delta^T, sep="")),
                 expression(paste("90% interval of ", Delta^T, sep=""))),  
        lty=c(1,NA,NA), lwd=c(1,NA,NA), pch=c(2,NA,NA),
        fill=c(NA, "grey80", "grey90"), border=c(NA,NA,NA), 
        x.intersp=1.5, y.intersp=1.5)

 
dev.off()




#gaps by lambda
Profit_Gap$lambda = Profit_Gap$alpha/Profit_Gap$beta
Profit_Gap_mean_by_lambda <- aggregate(Profit_Gap, by=list(Profit_Gap$lambda), FUN=mean, na.rm=TRUE)[c("lambda", "Gap_T")]
colnames(Profit_Gap_mean_by_lambda) <- c("lambda", "Gap_T_mean")
Profit_Gap_95quantile_by_lambda <- aggregate(Profit_Gap, by=list(Profit_Gap$lambda), FUN=quantile, prob=0.95, na.rm=TRUE)[c("lambda", "Gap_T")]
colnames(Profit_Gap_95quantile_by_lambda) <- c("lambda", "Gap_T_95quantile")
Profit_Gap_05quantile_by_lambda <- aggregate(Profit_Gap, by=list(Profit_Gap$lambda), FUN=quantile, prob=0.05, na.rm=TRUE)[c("lambda", "Gap_T")]
colnames(Profit_Gap_05quantile_by_lambda) <- c("lambda", "Gap_T_05quantile")
Profit_Gap_75quantile_by_lambda <- aggregate(Profit_Gap, by=list(Profit_Gap$lambda), FUN=quantile, prob=0.75, na.rm=TRUE)[c("lambda", "Gap_T")]
colnames(Profit_Gap_75quantile_by_lambda) <- c("lambda", "Gap_T_75quantile")
Profit_Gap_25quantile_by_lambda <- aggregate(Profit_Gap, by=list(Profit_Gap$lambda), FUN=quantile, prob=0.25, na.rm=TRUE)[c("lambda", "Gap_T")]
colnames(Profit_Gap_25quantile_by_lambda) <- c("lambda", "Gap_T_25quantile")

Profit_Gap_agg_lambda <- merge(Profit_Gap_mean_by_lambda, Profit_Gap_95quantile_by_lambda, by=c("lambda"))
Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_05quantile_by_lambda, by=c("lambda"))
Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_75quantile_by_lambda, by=c("lambda"))
Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_25quantile_by_lambda, by=c("lambda"))


##plot confidence interval as an area plot by using PLOYGON()  
xx <- c(rev(Profit_Gap_agg_lambda$lambda), Profit_Gap_agg_lambda$lambda)
yy90 <- c(rev(Profit_Gap_agg_lambda$Gap_T_05quantile), Profit_Gap_agg_lambda$Gap_T_95quantile)
yy50 <- c(rev(Profit_Gap_agg_lambda$Gap_T_25quantile), Profit_Gap_agg_lambda$Gap_T_75quantile)

pdf('Figure-Profit-aggregate-by-lambda.pdf', width = 8, height = 8)

xrange4 = range(Profit_Gap_agg_lambda$lambda)
yrange4 = c(0, 100)

plot(xrange4, yrange4, type="n", xlab="Mean Demand", ylab="Profit Gap (%)", xaxt="n", yaxt="n")
#lines(Profit_Gap_agg_c$fractile, rep(100, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=1)
#text(x=0.25, y=97, "Full")
#lines(Profit_Gap_agg_c$fractile, rep(0, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=1)
#text(x=0.25, y=3, "Event")
polygon(xx, yy90, col="grey90", border=NA)
polygon(xx, yy50, col="grey80", border=NA)
lines(Profit_Gap_agg_lambda$lambda, Profit_Gap_agg_lambda$Gap_T_mean, lty=1, lwd=1)
points(Profit_Gap_agg_lambda$lambda, Profit_Gap_agg_lambda$Gap_T_mean, pch=2)
#text(x=0.25, y=78, "Timing")
axis(side=1, at=seq(10, 50, 10), labels=seq(10, 50, 10))
axis(side=2, at=seq(0, 100, 10), labels=seq(0, 100, 10), las=1)
legend2(x="bottomright", inset=0,
        legend=c(expression(paste("mean of ", Delta^T, sep="")), 
                 expression(paste("50% interval of ", Delta^T, sep="")),
                 expression(paste("90% interval of ", Delta^T, sep=""))),  
        lty=c(1,NA,NA), lwd=c(1,NA,NA), pch=c(2,NA,NA),
        fill=c(NA, "grey80", "grey90"), border=c(NA,NA,NA), 
        x.intersp=1.5, y.intersp=1.5)


dev.off()




summary(Profit_Gap)