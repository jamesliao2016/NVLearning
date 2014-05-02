#################
# R code for analyzing output and plot figures
# v2.0 (organized on 2014-05-02) for revision 2
# for Figure 6 (plot of simulation results) in the paper
# published on https://github.com/tong-wang/NVLearning
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output/Simulation")


simu <- read.table("NVLearning-Simulation.1.625.txt", header=TRUE)
head(simu)





# plot of per-period profit
simu.profit <- simu[, 1:9]
simu.profit$n <- simu.profit$n+1
n.max <- max(simu.profit$n)
simu.profit$percF <- 100*(1-simu.profit$avgPF/simu.profit$avgPP)
simu.profit$percT <- 100*(1-simu.profit$avgPT/simu.profit$avgPP)
simu.profit$percE <- 100*(1-simu.profit$avgPE/simu.profit$avgPP)
simu.profit$percF.u <- 100*(1-(simu.profit$avgPF-simu.profit$sdPF)/simu.profit$avgPP)
simu.profit$percT.u <- 100*(1-(simu.profit$avgPT-simu.profit$sdPT)/simu.profit$avgPP)
simu.profit$percE.u <- 100*(1-(simu.profit$avgPE-simu.profit$sdPE)/simu.profit$avgPP)
simu.profit$percF.l <- 100*(1-(simu.profit$avgPF+simu.profit$sdPF)/simu.profit$avgPP)
simu.profit$percT.l <- 100*(1-(simu.profit$avgPT+simu.profit$sdPT)/simu.profit$avgPP)
simu.profit$percE.l <- 100*(1-(simu.profit$avgPE+simu.profit$sdPE)/simu.profit$avgPP)

head(simu.profit)




xrange = log10(range(simu.profit$n))
yrange.profit = c(0, 100)

##plot confidence interval as an area plot by using PLOYGON()  
xx <- c(rev(simu.profit$n), simu.profit$n)
CIF <- c(rev(simu.profit$percF.u), simu.profit$percF.l)
CIT <- c(rev(simu.profit$percT.u), simu.profit$percT.l)
CIE <- c(rev(simu.profit$percE.u), simu.profit$percE.l)



pdf('Figure-Simulation-Profit-1.pdf', width = 8, height = 8)

plot(xrange, yrange.profit, type="n", xlab="Period n", ylab="Profit Gap (%)", xaxt="n", yaxt="n")

# shade for confidence intervals
#polygon(log10(xx), CIF, col="grey90", border=NA)
#polygon(log10(xx), CIT, col="grey90", border=NA)
#polygon(log10(xx), CIE, col="grey90", border=NA)

#P model (in dashed line)
lines(xrange, c(0,0), lty=2, lwd=1, col="grey10")
text(x=0.27, y=2, "Perfect", col="grey50")

#F model
lines(log10(simu.profit$n), simu.profit$percF, lty=1, lwd=1, col="grey10")
points(log10(simu.profit$n), simu.profit$percF, pch=8, cex=.8, col="grey10")

#T model
lines(log10(simu.profit$n), simu.profit$percT, lty=1, lwd=2)
points(log10(simu.profit$n), simu.profit$percT, pch=1, cex=.8)

#E model
lines(log10(simu.profit$n), simu.profit$percE, lty=1, lwd=1, col="grey10")
points(log10(simu.profit$n), simu.profit$percE, pch=4, cex=.8, col="grey10")



axis(side=1, at=log10(seq(1, n.max, 25)), labels=c(0, seq(25, n.max, 25)))
axis(side=2, las=1)

legend(x="topright", inset=0, legend=c(expression(hat(zeta)[n]^{E}), expression(hat(zeta)[n]^{T}), expression(hat(zeta)[n]^{F})), lty=c(1,1,1), lwd=c(1,2,1), pch=c(4,1,8), col=c("grey10", "black", "grey10"), x.intersp=1.5, y.intersp=1.5, cex=1.2)

dev.off()






