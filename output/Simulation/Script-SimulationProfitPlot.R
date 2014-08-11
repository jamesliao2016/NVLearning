#################
# R code for analyzing output and plot figures
# v3.0 (organized on 2014-05-02) for revision 2
# for Figure 4 (plot of profits of E, T, and F models) in the paper
# published on https://github.com/tong-wang/NVLearning
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output/Simulation")


data <- read.table("NVLearning-Simulation.txt", header=TRUE)
head(data)




##figure: aggreage profit gaps by c

#calculate profit gaps
Profit_Gap <- data[c("c", "alpha", "beta")]
Profit_Gap$Gap_Tm <- (data$Pi_Tm - data$Pi_Em) / (data$Pi_F - data$Pi_Em) * 100

#aggregate by $c$ and find mean and 50% and 90% intervals of the T model
Profit_Gap_mean_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=mean, na.rm=TRUE)[c("c", "Gap_Tm")]
colnames(Profit_Gap_mean_by_c) <- c("c", "Gap_Tm_mean")
Profit_Gap_95quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.95, na.rm=TRUE)[c("c", "Gap_Tm")]
colnames(Profit_Gap_95quantile_by_c) <- c("c", "Gap_Tm_95quantile")
Profit_Gap_05quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.05, na.rm=TRUE)[c("c", "Gap_Tm")]
colnames(Profit_Gap_05quantile_by_c) <- c("c", "Gap_Tm_05quantile")
Profit_Gap_75quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.75, na.rm=TRUE)[c("c", "Gap_Tm")]
colnames(Profit_Gap_75quantile_by_c) <- c("c", "Gap_Tm_75quantile")
Profit_Gap_25quantile_by_c <- aggregate(Profit_Gap, by=list(Profit_Gap$c), FUN=quantile, prob=0.25, na.rm=TRUE)[c("c", "Gap_Tm")]
colnames(Profit_Gap_25quantile_by_c) <- c("c", "Gap_Tm_25quantile")


#merge into one data.frame for plotting
Profit_Gap_agg_c <- merge(Profit_Gap_mean_by_c, Profit_Gap_95quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_05quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_75quantile_by_c, by=c("c"))
Profit_Gap_agg_c <- merge(Profit_Gap_agg_c, Profit_Gap_25quantile_by_c, by=c("c"))


#calculate newsvendor critical fractile
Profit_Gap_agg_c$fractile <- (2-Profit_Gap_agg_c$c)/2


#plot confidence interval as an area plot by using PLOYGON()  
xx <- c(rev(Profit_Gap_agg_c$fractile), Profit_Gap_agg_c$fractile)
yy90 <- c(rev(Profit_Gap_agg_c$Gap_Tm_05quantile), Profit_Gap_agg_c$Gap_Tm_95quantile)
yy50 <- c(rev(Profit_Gap_agg_c$Gap_Tm_25quantile), Profit_Gap_agg_c$Gap_Tm_75quantile)

xrange4 = range(Profit_Gap_agg_c$fractile)
yrange4 = c(0, 100)

source(file="../legend2.R")
pdf('Figure-Simulation-Profit-aggregate.pdf', width = 8, height = 8)

    plot(xrange4, yrange4, type="n", xlab="Newsvendor Ratio", ylab="Loss Recovery (%)", xaxt="n", yaxt="n")
    
    #F model as upper bound (100%)
    lines(Profit_Gap_agg_c$fractile, rep(100, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=0.5, col="grey10")
    text(x=0.25, y=98, "Full", col="grey50")
    
    #E model as lower bound (0%)
    lines(Profit_Gap_agg_c$fractile, rep(0, length(Profit_Gap_agg_c$fractile)), lty=5, lwd=0.5, col="grey10")
    text(x=0.25, y=2, "Event-myopic", col="grey50")
    
    #90% inverval of T model
    polygon(xx, yy90, col="grey90", border=NA)
    
    #50% inverval of T model
    polygon(xx, yy50, col="grey80", border=NA)
    
    
    #mean of Tm model
    lines(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_Tm_mean, lty=2, lwd=1)
    points(Profit_Gap_agg_c$fractile, Profit_Gap_agg_c$Gap_Tm_mean, pch=1)
    
    axis(side=1, at=seq(0.1, 0.9, 0.1), labels=seq(0.1, 0.9, 0.1))
    axis(side=2, at=seq(0, 100, 10), labels=seq(0, 100, 10), las=1)
    
    legend2(x="bottomright", inset=0, bg="white",
        legend=c(
                 expression(paste("mean of ", hat(eta)^T, sep="")), 
                 expression(paste("50% interval of ", hat(eta)^T, sep="")),
                 expression(paste("90% interval of ", hat(eta)^T, sep=""))),  
        lty=c(2,NA,NA), lwd=c(1,NA,NA), pch=c(1,NA,NA),
        fill=c(NA, "grey80", "grey90"), border=c(NA,NA,NA), 
        x.intersp=1.5, y.intersp=1.5, cex=1.2)

 
dev.off()

#summary statistics
summary(Profit_Gap)


