Description
------------

There are five *.cpp* files corresponding to five models considered in the paper:

	NVLearning_Full.cpp			<==> The model with full observation of all demand occurrences (F)
	NVLearning_Event.cpp		<==> The model with observation of sales quantity and stock-out event (E)
	NVLearning_Timing.cpp		<==> The model with observation of sales occurrences (T)
	NVLearning_CheckpointA.cpp	<==> The model with inventory checkpoints (IC)
	NVLearning_CheckpointB.cpp	<==> The model with stock-out checkpoints (SC)

Each file implements the dynamic programming algorithm in finding the optimal inventory level and evaluating expected profit.

The file *NVLearning_Simulation.cpp* is for the simulation study of myopic polices in Section 5.3 of the paper.

The two *.h* header files are required when compiling the *.cpp* files.

Instructions
------------

The C++ code in the **NVLearning/src/** folder is written using OpenMP, C++11 standard library, and Boost C++ Library (http://www.boost.org/). 

In order to compile them, you will need:

1. An update-to-date C++ compiler (GNU GCC 4.7 or Intel C++ Compiler 13.0) supporting C++11 and OpenMP；
2. Install the Boost Library (v1.53.0 or newer, check it out at http://www.boost.org/)；
3. To compile (assuming the install directory of the Boost Library is "/usr/local/boost_1_53_0"):

        by ICC: "icpc -openmp -std=c++11 -O3 -lboost_serialization -lboost_program_options -I /usr/local/boost_1_53_0 -L /usr/local/boost_1_53_0/stage/lib -o NVLearning_Timing.exe  NVLearning_Timing.cpp"
        
        by GCC: “g++ -fopenmp -std=c++11 -O3 -lboost_serialization -lboost_program_options -I /usr/local/boost_1_53_0 -L /usr/local/boost_1_53_0/stage/lib -o NVLearning_Timing.exe  NVLearning_Timing.cpp"

