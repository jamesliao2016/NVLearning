NVLearning
==========
This repository hosts codes for the numerical experiments for the paper "Demand Estimation and Ordering under Censoring: Stock-out timing is (almost) all you need"


    Repository:     NVLearning
    Author:         Tong WANG
    Email:          tong.wang@nus.edu.sg
    Version:        v8.0 (2013-11-11)
    Copyright:      absolutely free (In case you find the code helpful in any sense and do not know how to express your appreciation, cite our paper below :)
    Description:
                   <NVLearning/src> folder hosts the C++ code for the models developed in the paper: additive MMFE, multiplicative MMFE, additive MMFE with fixed ordering costs, and additive MMFE with order cancelations. The codes essentially solve the dynamic programs developed in the paper to obtain the optimal inventory control policies and expected profits.

                    <NVLearning/output> folder hosts the raw output data files generated by the C++ codes, the R scripts for cleaning, organizing, summarizing, and plotting these data, and the final output (in the format of Tables and Figures) published in the paper.


Reference
=========

    Paper title:    "Demand Estimation and Ordering under Censoring: Stock-out timing is (almost) all you need"
    Authors:        Aditya Jain, Nils Rudi, and Tong Wang
    Year:           2013
    Journal:        NA
    Volume:         NA
    Number:         NA
    Page:           NA
    Journal link:   NA
    BibTeX:         NA
    
    Further resources (such as the original paper in PDF format, its appendix, and presentation slides) are available at http://bschool.nus.edu/staff/bizwt/research.html.


Instructions
============

    The C++ codes are re-written using the new C++11 standard library and the Boost C++ Library (http://www.boost.org/). In order to compile them, you will need:
        (1). An update-to-date C++ compiler (GNU GCC 4.7 or Intel C++ Compiler 13.0) supporting C++11 and OpenMP
        (2). Install the Boost Library (v1.53.0 or newer, check it out at http://www.boost.org/).
        (3). Compile by ICC: "icpc -openmp -std=c++11 -O3 -lboost_serialization -lboost_program_options -I /usr/local/boost_1_53_0 -L /usr/local/boost_1_53_0/stage/lib -o NVLearning_Timing.exe  NVLearning_Timing.cpp", assuming the install directory of the Boost Library is "/usr/local/boost_1_53_0".
	(4). Compile by GCC: “g++ -fopenmp -std=c++11 -O3 -lboost_serialization -lboost_program_options -I /usr/local/boost_1_53_0 -L /usr/local/boost_1_53_0/stage/lib -o NVLearning_Timing.exe  NVLearning_Timing.cpp"



    The R scripts require a working R environment (http://www.r-project.org/). The scripts are only tested on R version 2.15, but should be running smoothly on newer versions. The scripts do not require extra R packages.


