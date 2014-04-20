#################
# R code for analyzing output and plot figures
# v2.0 (organized on 2013-11-11)
# for Figure A.1b in the paper
# published on https://github.com/tong-wang/NVLearning
#################

#NEED TO FIRST SET R WORKING DIRECTORY TO WHERE THE FILES ARE LOCATED!!!
setwd("~/Dropbox/Research/CensoredDemand/NVLearning.git/output/serial_benchmark/")


#read the output files
data_F <- read.table("NVLearning_Full.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_F")]
data_E <- read.table("NVLearning_Event.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_E")]
data_Em <- read.table("NVLearning_Event_myopic.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_Em")]
data_T <- read.table("NVLearning_Timing.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_T")]
data_Tm <- read.table("NVLearning_Timing_myopic.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_Tm")]
data_B2 <- read.table("NVLearning_CheckpointB.M2.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_B2")]
data_B2m <- read.table("NVLearning_CheckpointB_myopic.M2.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_B2m")]
data_B4 <- read.table("NVLearning_CheckpointB.M4.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_B4")]
data_B4m <- read.table("NVLearning_CheckpointB_myopic.M4.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_B4m")]
data_A2 <- read.table("NVLearning_CheckpointA.M2.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_A2")]
data_A2m <- read.table("NVLearning_CheckpointA_myopic.M2.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_A2m")]
#data_A4 <- read.table("NVLearning_CheckpointA.M4.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_A4")]
data_A4m <- read.table("NVLearning_CheckpointA_myopic.M4.serial.txt", header=TRUE)[c("c", "alpha", "beta", "CPUTime_A4m")]



#merge into one data frame
data <- merge(data_F, data_Em, by=c("c", "alpha", "beta"))
data <- merge(data, data_E, by=c("c", "alpha", "beta"))
data <- merge(data, data_Tm, by=c("c", "alpha", "beta"))
data <- merge(data, data_T, by=c("c", "alpha", "beta"))
data <- merge(data, data_B2m, by=c("c", "alpha", "beta"))
data <- merge(data, data_B2, by=c("c", "alpha", "beta"))
data <- merge(data, data_B4m, by=c("c", "alpha", "beta"))
data <- merge(data, data_B4, by=c("c", "alpha", "beta"))
data <- merge(data, data_A2m, by=c("c", "alpha", "beta"))
data <- merge(data, data_A2, by=c("c", "alpha", "beta"))
data <- merge(data, data_A4m, by=c("c", "alpha", "beta"))
#data <- merge(data, data_A4, by=c("c", "alpha", "beta")) 

colnames(data) <- c("c", "alpha", "beta", "F", "Em", "E",  "Tm", "T", "SC[2]m", "SC[2]", "SC[4]m", "SC[4]", "IC[2]m", "IC[2]", "IC[4]m") #, "IC[4]")

#reshape data
require("reshape2")
data.long <- melt(data=data, id.vars=c("c","alpha", "beta"), variable.name="Model", value.name="CPUTime")


pdf('Figure-CPUTime.pdf', width = 8, height = 8)

    boxplot(log10(CPUTime) ~ Model, data=data.long, xlab="Model", ylab="CPU Time (millisecond)", range=0,
        xaxt="n", yaxt="n", ylim=c(1.7, 10.3),
        at=c(1, 3.25,2.75, 5.25,4.75, 7.25,6.75, 9.25,8.75, 11.25,10.75, 13.25), #,12.75),
        boxlty=c(1, 2,1, 2,1, 2,1, 2,1, 2,1, 2), #,1) 
        boxfill=c("grey80", "grey98","grey80", "grey98","grey80", "grey98","grey80", "grey98","grey80", "grey98","grey80", "grey98"), #,"grey80"), 
        medlty=c(1, 2,1, 2,1, 2,1, 2,1, 2,1,  2), #,1), 
        medlwd=c(1, 1,1, 1,1, 1,1, 1,1, 1,1, 1), #,1),
        #medcol=c("grey40", "grey40", "grey70","grey40", "grey70","grey40", "grey70","grey40", "grey70","grey40"), 
        whisklty=c(1, 2,1, 2,1, 2,1, 2,1, 2,1, 2), #,1), 
        staplelty=c(1, 2,1, 2,1, 2,1, 2,1, 2,1, 2) #,1),
    )

    axis(side=1, at=c(1, 3, 5, 7, 9, 11, 13), labels=c("F","E","T","SC[2]","SC[4]", "IC[2]", "IC[4]"))
    axis(side=2, at=seq(2, 10, 1), labels=parse(text=paste("10^", seq(2, 10, 1), sep="")), las=1)

dev.off()
