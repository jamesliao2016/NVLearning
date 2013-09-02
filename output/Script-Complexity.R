#################
# R code for analyzing output and plot figures
# v1.0 (organized on 2013-04-17)
# for Figure 4 in the paper
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output")

#read the output files
data_F <- read.table("NVLearning_Full.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_F")]
data_E <- read.table("NVLearning_Event.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_E")]
data_T <- read.table("NVLearning_Timing.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_T")]
data_Tm <- read.table("NVLearning_Timing_myopic.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_Tm")]


data_B2 <- read.table("NVLearning_CheckpointB_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_B2")]
data_B2m <- read.table("NVLearning_CheckpointB_myopic_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_B2m")]
data_B4 <- read.table("NVLearning_CheckpointB_M4.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_B4")]
data_B4m <- read.table("NVLearning_CheckpointB_myopic_M4.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_B4m")]

data_A2 <- read.table("NVLearning_CheckpointA_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_A2")]
data_A2m <- read.table("NVLearning_CheckpointA_myopic_M2.txt", header=TRUE)[c("c", "alpha", "beta", "Comp_A2m")]


#merge into one data frame
data <- merge(data_F, data_E, by=c("c", "alpha", "beta"))
data <- merge(data, data_Tm, by=c("c", "alpha", "beta"))
data <- merge(data, data_T, by=c("c", "alpha", "beta"))
data <- merge(data, data_B2m, by=c("c", "alpha", "beta"))
data <- merge(data, data_B2, by=c("c", "alpha", "beta"))
data <- merge(data, data_B4m, by=c("c", "alpha", "beta"))
data <- merge(data, data_B4, by=c("c", "alpha", "beta"))
data <- merge(data, data_A2m, by=c("c", "alpha", "beta"))
data <- merge(data, data_A2, by=c("c", "alpha", "beta"))

colnames(data) <- c("c", "alpha", "beta", "F", "E",  "Tm", "T", "SC[2]m", "SC[2]", "SC[4]m", "SC[4]", "IC[2]m", "IC[2]")

require("reshape2")
data.long <- melt(data=data, id.vars=c("c","alpha", "beta"), variable.name="Model", value.name="Comp")



pdf('Figure-Complexity.pdf', width = 8, height = 8)

boxplot(log10(Comp) ~ Model, data=data.long, xlab="Model", ylab="Complexity", 
        at=c(1, 3, 5.25,4.75, 7.25,6.75, 9.25,8.75, 11.25, 10.75), #show.names=c(1,1,0,1,0,1,0,1),
        xaxt="n", yaxt="n", ylim=c(1,8), outline=FALSE,
        boxlty=c(1, 1, 2,1, 2,1, 2,1, 2,1), boxfill=c("grey80", "grey80", "grey98","grey80", "grey98","grey80", "grey98","grey80", "grey98","grey80"), 
        medlty=c(1, 1, 2,1, 2,1, 2,1, 2,1), medlwd=c(1, 1, 1,1, 1,1, 1,1, 1,1),#medcol=c("grey40", "grey40", "grey70","grey40", "grey70","grey40", "grey70","grey40", "grey70","grey40"), 
        whisklty=c(1, 1, 2,1, 2,1, 2,1, 2,1), staplelty=c(1, 1, 2,1, 2,1, 2,1, 2,1) )
axis(side=1, at=c(1, 3, 5, 7, 9, 11), labels=c("F","E","T","SC[2]","SC[4]", "IC[2]"))
axis(side=2, at=seq(1, 8, 1), labels=parse(text=paste("10^", seq(1, 8, 1), sep="")), las=1)

dev.off()


