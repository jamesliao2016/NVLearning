#################
# R code for analyzing output and plot figures
# v2.0 (organized on 2013-11-11)
# for Figure 3 (plot of optimal inventory levels) in the paper
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



##figure 3a: optimal inventory level in a typical scenario

#subset data, just keep the scenarios with alpha=10/16 and beta=1/16
data_typical = data[data$alpha==0.625&data$beta==0.0625,]
#calculate newsvendor critical fractile
data_typical$fractile = (2-data_typical$c)/2

xrange1 = range(data_typical$fractile)
yrange1 = range(data_typical[c("Q_F", "Q_E", "Q_T")])

pdf('Figure-Inventory-typical.pdf', width = 8, height = 8)

    plot(xrange1, yrange1, type="n", xlab="Newsvendor Ratio", ylab="Optimal Inventory Level", xaxt="n", yaxt="n")
    
    #F model
    lines(data_typical$fractile, data_typical$Q_F, lty=2, lwd=1)
    points(data_typical$fractile, data_typical$Q_F, pch=8)
    
    #E model
    lines(data_typical$fractile, data_typical$Q_E, lty=1, lwd=1)
    points(data_typical$fractile, data_typical$Q_E, pch=4)
    
    #T model
    lines(data_typical$fractile, data_typical$Q_T, lty=1, lwd=1)
    points(data_typical$fractile, data_typical$Q_T, pch=19)
    
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, las=1)
    
    legend(x="topleft", inset=0, legend=c(expression(y[1]^{E}), expression(y[1]^{T}), expression(y[1]^{F})), lty=c(1,1,2), lwd=c(1,1,1), pch=c(4,19,8), x.intersp=1.5, y.intersp=1.5, cex=1.2)
 
dev.off()




##figure 3b: service level

#calculate Fill Rate from inventory level
ServiceLevel <- data[c("c", "alpha", "beta", "Q_F", "Q_E", "Q_T")]
ServiceLevel$fractile <- (2 - ServiceLevel$c) / 2
ServiceLevel$FR_F <- pnbinom(ServiceLevel$Q_F, ServiceLevel$alpha, 1 - 1/(1+ServiceLevel$beta)) * 100
ServiceLevel$FR_E <- pnbinom(ServiceLevel$Q_E, ServiceLevel$alpha, 1 - 1/(1+ServiceLevel$beta)) * 100
ServiceLevel$FR_T <- pnbinom(ServiceLevel$Q_T, ServiceLevel$alpha, 1 - 1/(1+ServiceLevel$beta)) * 100

#calculate Fill Rate difference
ServiceLevel$Gap_E <- ServiceLevel$FR_E - ServiceLevel$FR_F
ServiceLevel$Gap_T <- ServiceLevel$FR_T - ServiceLevel$FR_F

#aggregate by fractile
ServiceLevel_agg_mean <- aggregate(ServiceLevel[c("Gap_E", "Gap_T")], by=list(ServiceLevel$fractile), FUN=mean)
colnames(ServiceLevel_agg_mean) <- c("fractile", "Gap_E_mean", "Gap_T_mean")
ServiceLevel_agg_Q90 <- aggregate(ServiceLevel[c("Gap_E", "Gap_T")], by=list(ServiceLevel$fractile), FUN=quantile, probs=0.9)
colnames(ServiceLevel_agg_Q90) <- c("fractile", "Gap_E90", "Gap_T90")
ServiceLevel_agg <- merge(ServiceLevel_agg_mean , ServiceLevel_agg_Q90, by="fractile")

#plot confidence interval as an area plot by using PLOYGON()  
xx_T <- c(rev(ServiceLevel_agg$fractile), ServiceLevel_agg$fractile)
yy_T <- c(rep(0, length(ServiceLevel_agg$fractile)), ServiceLevel_agg$Gap_T90)
xx_E <- c(rev(ServiceLevel_agg$fractile), ServiceLevel_agg$fractile)
yy_E <- c(rep(0, length(ServiceLevel_agg$fractile)), ServiceLevel_agg$Gap_E90)



xrange2 = range(ServiceLevel_agg$fractile)
yrange2 = range(yy_E, yy_T)

pdf('Figure-Inventory-ServiceLevel.pdf', width = 8, height = 8)

    plot(xrange2, yrange2, type="n", xlab="Newsvendor Ratio", ylab="Increase in In-stock Probability (%)", xaxt="n", yaxt="n")

    #polygon(xx_E, yy_E, col="grey90", border=NA)
    #polygon(xx_T, yy_T, col="grey80", border=NA)

    #mean of E model
    lines(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_E_mean, lty=1, lwd=1)
    points(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_E_mean, pch=4)
    
    #mean T model
    lines(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_T_mean, lty=1, lwd=1)
    points(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_T_mean, pch=19)
    
    #90% quantilie of E model
    lines(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_E90, lty=2, lwd=1)
    points(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_E90, pch=4)
    
    #90% quantilie of T model
    lines(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_T90, lty=2, lwd=1)
    points(ServiceLevel_agg$fractile, ServiceLevel_agg$Gap_T90, pch=19)
    
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, las=1)

    legend(x="topright", inset=0,
        legend=c(expression(paste("mean of ", delta^E, sep="")), 
                expression(paste("mean of ", delta^T, sep="")), 
                expression(paste("90% quantile of ", delta^E, sep="")),  
                expression(paste("90% quantile of ", delta^T, sep=""))),  
        lty=c(1,1,2,2), lwd=c(1,1,1,1), pch=c(4,19,4,19),
        #fill=c(NA, NA, "grey90", "gray80"), border=c(NA, NA, NA, NA), 
        x.intersp=1.5, y.intersp=1.5, cex=1.2)

dev.off()

#summary statistics
summary(ServiceLevel)





##figure 3c: aggregate analysis of over-ordering (not included in the paper)

#calculate percentage overordering in E and T models
Q_Gap <- data[c("c", "alpha", "beta")]
Q_Gap$Gap_E <- (data$Q_E - data$Q_F) / data$Q_F * 100
Q_Gap$Gap_T <- (data$Q_T - data$Q_F) / data$Q_F * 100

#aggregate by $c$ and calculate median and 90% interval
Q_Gap_agg_c_50 <- aggregate(Q_Gap[c("Gap_E", "Gap_T")], by=list(Q_Gap$c), FUN=quantile, probs=0.5, na.rm=TRUE)
colnames(Q_Gap_agg_c_50) <- c("c", "Gap_E50", "Gap_T50")
Q_Gap_agg_c_05 <- aggregate(Q_Gap[c("Gap_E", "Gap_T")], by=list(Q_Gap$c), FUN=quantile, probs=0.05, na.rm=TRUE)
colnames(Q_Gap_agg_c_05) <- c("c", "Gap_E05", "Gap_T05")
Q_Gap_agg_c_95 <- aggregate(Q_Gap[c("Gap_E", "Gap_T")], by=list(Q_Gap$c), FUN=quantile, probs=0.95, na.rm=TRUE)
colnames(Q_Gap_agg_c_95) <- c("c", "Gap_E95", "Gap_T95")

Q_Gap_agg_c <- merge(Q_Gap_agg_c_50, Q_Gap_agg_c_05, by="c")
Q_Gap_agg_c <- merge(Q_Gap_agg_c, Q_Gap_agg_c_95, by="c")

#calculate newsvendor critical fractile
Q_Gap_agg_c$fractile <- (2-Q_Gap_agg_c$c)/2


#plot confidence interval as an area plot by using PLOYGON()  
xx_T <- c(rev(Q_Gap_agg_c$fractile), Q_Gap_agg_c$fractile)
yy_T <- c(rev(Q_Gap_agg_c$Gap_T05), Q_Gap_agg_c$Gap_T95)
xx_E <- c(rev(Q_Gap_agg_c$fractile), Q_Gap_agg_c$fractile)
yy_E <- c(rev(Q_Gap_agg_c$Gap_E05), Q_Gap_agg_c$Gap_E95)

xrange3 = range(Q_Gap_agg_c$fractile)
yrange3 = range(yy_E, yy_T)

source(file="legend2.R")
pdf('Figure-Inventory-aggregate.pdf', width = 8, height = 8)

plot(xrange3, yrange3, type="n", xlab="Newsvendor Ratio", ylab="Inventory Over-ordering (%)", xaxt="n", yaxt="n")

    #90% interval of E model
    polygon(xx_E, yy_E, col="grey90", border=NA)

    #90% interval of T model
    polygon(xx_T, yy_T, col="grey80", border=NA)
    
    #median of E model
    lines(Q_Gap_agg_c$fractile, Q_Gap_agg_c$Gap_E50, lty=1, lwd=1)
    points(Q_Gap_agg_c$fractile, Q_Gap_agg_c$Gap_E50, pch=4)
    
    #median of T model
    lines(Q_Gap_agg_c$fractile, Q_Gap_agg_c$Gap_T50, lty=1, lwd=1)
    points(Q_Gap_agg_c$fractile, Q_Gap_agg_c$Gap_T50, pch=19)
    
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, las=1)
    
    legend2(x="topright", inset=0,
        legend=c(expression(paste("median of ", delta^E, sep="")), 
                 expression(paste("median of ", delta^T, sep="")), 
                 expression(paste("90% interval of ", delta^E, sep="")),  
                 expression(paste("90% interval of ", delta^T, sep=""))),  
        lty=c(1,1,NA,NA), lwd=c(1,1,NA,NA), pch=c(4,19,NA,NA),
        fill=c(NA, NA, "grey90", "gray80"), border=c(NA, NA, NA, NA), 
        x.intersp=1.5, y.intersp=1.5, cex=1.2)

dev.off()

#remove NA
if (nrow(Q_Gap[is.infinite(Q_Gap$Gap_E),])>0) Q_Gap[is.infinite(Q_Gap$Gap_E),]$Gap_E = NA
if (nrow(Q_Gap[is.infinite(Q_Gap$Gap_T),])>0) Q_Gap[is.infinite(Q_Gap$Gap_T),]$Gap_T = NA

#summary statistics of the gaps
summary(Q_Gap)

