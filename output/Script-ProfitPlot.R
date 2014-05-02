#################
# R code for analyzing output and plot figures
# v3.0 (organized on 2014-05-02) for revision 2
# for Figure 4 (plot of profits of E, T, and F models) in the paper
# published on https://github.com/tong-wang/NVLearning
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output")

#read the output files
data_F <- read.table("NVLearning_Full.txt", header=TRUE)[c("c", "alpha", "beta", "Q_F", "Pi_F")]
data_E <- read.table("NVLearning_Event.txt", header=TRUE)[c("c", "alpha", "beta", "Q_E", "Pi_E")]
data_T <- read.table("NVLearning_Timing.txt", header=TRUE)[c("c", "alpha", "beta", "Q_T", "Pi_T")]
data_Tm <- read.table("NVLearning_Timing_myopic.txt", header=TRUE)[c("c", "alpha", "beta", "Q_Tm", "Pi_Tm")]


#merge into one data frame
data <- merge(data_F, data_E, by=c("c", "alpha", "beta"))
data <- merge(data, data_T, by=c("c", "alpha", "beta"))
data <- merge(data, data_Tm, by=c("c", "alpha", "beta"))


##figure 4a: optimal profit in a typical scenario

#subset data, just keep the scenarios with alpha=10/16 and beta=1/16
data_typical = data[data$alpha==0.625&data$beta==0.0625,]
#calculate newsvendor critical fractile
data_typical$fractile = (2-data_typical$c)/2


xrange3 = range(data_typical$fractile)
yrange3 = range(data_typical[c("Pi_F", "Pi_E", "Pi_T")])

pdf('Figure-Profit-typical.pdf', width = 8, height = 8)

    plot(xrange3, yrange3, type="n", xlab="Newsvendor Ratio", ylab="Optimal Profit", xaxt="n", yaxt="n")
    
    #F model
    lines(data_typical$fractile, data_typical$Pi_F, lty=1, lwd=1, col="grey10")
    points(data_typical$fractile, data_typical$Pi_F, pch=8, col="grey10")
    
    #E model
    lines(data_typical$fractile, data_typical$Pi_E, lty=1, lwd=1, col="grey10")
    points(data_typical$fractile, data_typical$Pi_E, pch=4, col="grey10")
    
    #T odel
    lines(data_typical$fractile, data_typical$Pi_T, lty=1, lwd=2)
    points(data_typical$fractile, data_typical$Pi_T, pch=19)
    
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, las=1)
    
    legend(x="topleft", inset=0, legend=c(expression(V[1]^{E}), expression(V[1]^{T}), expression(V[1]^{F})), lty=c(1,1,1), lwd=c(1,2,1), pch=c(4,19,8), col=c("grey10", "black", "grey10"), x.intersp=1.5, y.intersp=1.5, cex=1.2)

dev.off()




##figure 4b: aggreage profit gaps by c

#calculate profit gaps
Profit_Gap <- data[c("c", "alpha", "beta")]
Profit_Gap$Gap_T <- (data$Pi_T - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100
Profit_Gap$Gap_Tm <- (data$Pi_Tm - data$Pi_E) / (data$Pi_F - data$Pi_E) * 100

#aggregate by $c$ and find mean and 50% and 90% intervals of the T model
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

#mean of the Tm model
Profit_Gap_mean_Tm_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=mean, na.rm=TRUE)[c("c", "Gap_Tm")]
colnames(Profit_Gap_mean_Tm_by_c) <- c("c", "Gap_Tm_mean")

#merge into one data.frame for plotting
Profit_Gap_agg_c <- merge(Profit_Gap_mean_by_c, Profit_Gap_95quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_05quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_75quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_25quantile_by_c, by=c("c"))

Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_mean_Tm_by_c, by=c("c"))

#calculate newsvendor critical fractile
Profit_Gap_agg_c$fractile <- (2-Profit_Gap_agg_c$c)/2


#plot confidence interval as an area plot by using PLOYGON()  
xx <- c(rev(Profit_Gap_agg_c$fractile), Profit_Gap_agg_c$fractile)
yy90 <- c(rev(Profit_Gap_agg_c$Gap_T_05quantile), Profit_Gap_agg_c$Gap_T_95quantile)
yy50 <- c(rev(Profit_Gap_agg_c$Gap_T_25quantile), Profit_Gap_agg_c$Gap_T_75quantile)

xrange4 = range(Profit_Gap_agg_c$fractile)
yrange4 = c(0, 100)

source(file="legend2.R")
pdf('Figure-Profit-aggregate.pdf', width = 8, height = 8)

    plot(xrange4, yrange4, type="n", xlab="Newsvendor Ratio", ylab="Loss Recovery (%)", xaxt="n", yaxt="n")
    
    #F model as upper bound (100%)
    lines(Profit_Gap_agg_c$fractile, rep(100, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=0.5, col="grey10")
    text(x=0.25, y=98, "Full", col="grey50")
    
    #E model as lower bound (0%)
    lines(Profit_Gap_agg_c$fractile, rep(0, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=0.5, col="grey10")
    text(x=0.25, y=2, "Event", col="grey50")
    
    #90% inverval of T model
    polygon(xx, yy90, col="grey90", border=NA)
    
    #50% inverval of T model
    polygon(xx, yy50, col="grey80", border=NA)
    
    #mean of T model
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_T_mean, lty=1, lwd=2)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_T_mean, pch=19)
    #text(x=0.25, y=78, "Timing")
    
    #mean of Tm model
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_Tm_mean, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_Tm_mean, pch=1)
    
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, at=seq(0, 100, 10), labels=seq(0, 100, 10), las=1)
    
    legend2(x="bottomright", inset=0, bg="white",
        legend=c(expression(paste("mean of ", eta^T, sep="")), 
                 expression(paste("mean of ", hat(eta)^T, sep="")), 
                 expression(paste("50% interval of ", eta^T, sep="")),
                 expression(paste("90% interval of ", eta^T, sep=""))),  
        lty=c(1,2,NA,NA), lwd=c(2,1,NA,NA), pch=c(19,1,NA,NA),
        fill=c(NA, NA, "grey80", "grey90"), border=c(NA,NA,NA,NA), 
        x.intersp=1.5, y.intersp=1.5, cex=1.2)

 
dev.off()

#summary statistics
summary(Profit_Gap)



##figure 4c: aggreage profit gaps by lambda (not included in the paper)

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

Profit_Gap_mean_Tm_by_lambda <- aggregate(Profit_Gap, by=list(Profit_Gap$lambda), FUN=mean, na.rm=TRUE)[c("lambda", "Gap_Tm")]
colnames(Profit_Gap_mean_Tm_by_lambda) <- c("lambda", "Gap_Tm_mean")

#merge
Profit_Gap_agg_lambda <- merge(Profit_Gap_mean_by_lambda, Profit_Gap_95quantile_by_lambda, by=c("lambda"))
Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_05quantile_by_lambda, by=c("lambda"))
Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_75quantile_by_lambda, by=c("lambda"))
Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_25quantile_by_lambda, by=c("lambda"))

Profit_Gap_agg_lambda <- merge(Profit_Gap_agg_lambda, Profit_Gap_mean_Tm_by_lambda, by=c("lambda"))

##plot confidence interval as an area plot by using PLOYGON()  
xx <- c(rev(Profit_Gap_agg_lambda$lambda), Profit_Gap_agg_lambda$lambda)
yy90 <- c(rev(Profit_Gap_agg_lambda$Gap_T_05quantile), Profit_Gap_agg_lambda$Gap_T_95quantile)
yy50 <- c(rev(Profit_Gap_agg_lambda$Gap_T_25quantile), Profit_Gap_agg_lambda$Gap_T_75quantile)

xrange4 = range(Profit_Gap_agg_lambda$lambda)
yrange4 = c(0, 100)

pdf('Figure-Profit-aggregate-by-lambda.pdf', width = 8, height = 8)

    plot(xrange4, yrange4, type="n", xlab="Mean Demand", ylab="Loss Recovery (%)", xaxt="n", yaxt="n")
    
    lines(Profit_Gap_agg_lambda$lambda, rep(100, length(Profit_Gap_agg_lambda$lambda)), lty=5, lwd=1)
    text(x=20, y=97, "Full")
    
    lines(Profit_Gap_agg_lambda$lambda, rep(0, length(Profit_Gap_agg_lambda$lambda)), lty=5, lwd=1)
    text(x=20, y=3, "Event")
    
    polygon(xx, yy90, col="grey90", border=NA)
    
    polygon(xx, yy50, col="grey80", border=NA)
    
    lines(Profit_Gap_agg_lambda$lambda, Profit_Gap_agg_lambda$Gap_T_mean, lty=1, lwd=1)
    points(Profit_Gap_agg_lambda$lambda, Profit_Gap_agg_lambda$Gap_T_mean, pch=19)
    #text(x=0.25, y=78, "Timing")
    
    lines(Profit_Gap_agg_lambda$lambda, Profit_Gap_agg_lambda$Gap_Tm_mean, lty=2, lwd=1)
    points(Profit_Gap_agg_lambda$lambda, Profit_Gap_agg_lambda$Gap_Tm_mean, pch=1)
    
    axis(side=1, at=seq(10, 50, 10), labels=seq(10, 50, 10))
    axis(side=2, at=seq(0, 100, 10), labels=seq(0, 100, 10), las=1)
    
    legend2(x="bottomright", inset=0.04,
        legend=c(expression(paste("mean of ", eta^T, sep="")), 
                expression(paste("mean of ", hat(eta)^T, sep="")), 
                expression(paste("50% interval of ", eta^T, sep="")),
                expression(paste("90% interval of ", eta^T, sep=""))),  
        lty=c(1,2,NA,NA), lwd=c(1,1,NA,NA), pch=c(19,1,NA,NA),
        fill=c(NA, NA, "grey80", "grey90"), border=c(NA,NA,NA,NA), 
        x.intersp=1.5, y.intersp=1.5, cex=1.2)

dev.off()

