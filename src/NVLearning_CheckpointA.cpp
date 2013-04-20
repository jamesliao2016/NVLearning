//============================================================================
// Name        : NVLearning_CheckpointA.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v4.0 (2013-04-20)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Checkpoint-A heuristic
//               compile using Intel icc: icpc -std=c++11 -openmp -O3 -fast -o NVLearning_CheckpointA.exe NVLearning_CheckpointA.cpp
//============================================================================

//***********************************************************

#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <set>
#include <map>
#include <tuple>
#include <vector>
#include <omp.h>

using namespace std;

//***********************************************************

#define N 4                     //number of periods
#define M_MAX 4                 //Maximum number of checkpoints to be considered
#define LAMBDA_STEP 1000        //discretize the continuous distribution of lambda into LAMBDA_STEP=1000 pieces

#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10
#define LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 8
#define LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 4

//***********************************************************

int M;                          //Number of checkpoints within a period
double price, cost;             //Newsvendor price and cost parameters
double alpha0, beta0;           //Initial prior of lambda is Gamma(alpha0, beta0)

ofstream file;                  //output files

map<tuple<int, int, multiset<int>>, tuple<double, double, vector<double>>> lambda_map;  //an std::map to store updated distributions of lambda
map<tuple<int, int, int, multiset<int>>, vector<vector<double>> > observation_map;      //an std::map to store predictive distributions of observation
map<tuple<int, int, int, multiset<int>>, double> v_map;                                 //an std::map to store calculated value of the V() function


//***********************************************************
//Functions for calculating various probability distributions


double Poisson(int xx, double lam)
{
    double log_pmf = xx*log(lam) -lam - lgamma(xx+1);

    return exp(log_pmf);
}


double Poisson_CDF(int xx, double lam)
{
    double out=0;
    for (int i=0; i<=xx;i++)
        out += Poisson(i,lam);
    return out;
}


double Gamma(double xx, double aa, double bb)
{
    double log_pdf = 0;

    if ((xx>=0)&&(aa>0)&&(bb>0))
        log_pdf = aa*log(bb) + log(xx)*(aa-1)  -bb*xx - lgamma(aa);

    return exp(log_pdf);
}


double NegBinomial(int kk, double rr, double pp)
{
    double log_pmf = 0;

    if ((kk>=0)&&(rr>0)&&(pp>=0)&&(pp<=1))
        log_pmf = lgamma(kk+rr) - lgamma(kk+1) - lgamma(rr)  + rr*log(1-pp) + kk*log(pp);

    return exp(log_pmf);
}


//***********************************************************
//Bayesian updating implementation (for cases with censored observations)
//Key variables:
//  int fullObs_cumulativeTime: cumulative number of sub-periods (1/M) with full observation
//  int fullObs_cumulativeQuantity: cumulative demand quantity that was fully observed (in terms of the number of sub-periods, for the sake of discreteness)
//  multiset<int> censoredObservations: the set of censored observations in previous sub-periods



//calculate the likelihood of observing all the historical censored observations with a given lambda
//Prob(D(1/M)>=censoredObservations[0]) * Prob(D(1/M)>=censoredObservations[1]) * ...
double Likelihood(multiset<int> censoredObservations, double lambda)
{
    double out=1;

    //for each censored observation $co$ in the set $censoredObservations$, calculate its probability
    for (int co : censoredObservations)
    {
        //Probability of a $co$: Prob{ D(1/M) >= co }
        double F = 0;
        for (int d1=0; d1<co; d1++)
            F += Poisson(d1, lambda/M);

        out *= max(0.0, 1 - F);
    }

    return out;
}


//brute-force Bayesian updating of the pdf of lambda based on historical observations
tuple<double, double, vector<double>> lambda_pdf_update(int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations)
{
    //initialize the output tuple
    tuple<double, double, vector<double>> lambda_tuple;

    //first try to search for existing $lambda$ in $lambda_map$, based on the key $allObservations$
    auto allObservations = make_tuple(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    int count;
    //ALL READ/WRITE ACCESSES TO std::map ARE PROTECTED BY "OMP CRITICAL" TO ENSURE THREAD SAFETY
    #pragma omp critical (lambda_map)
    {count = lambda_map.count(allObservations);}
    
    //load the pdf of lambda if available in lambda_map
    if (count==1)
    {
        //cout << "lambda\t" << "\t" << fullObs_cumulativeTime << "\t" << fullObs_cumulativeQuantity << "\t" << count << endl;
        #pragma omp critical (lambda_map)
        {lambda_tuple = lambda_map[allObservations];}
    
    } else {
        
        //update alpha,beta based on the exact observations
        double alpha_n = alpha0 + fullObs_cumulativeQuantity;
        double beta_n = beta0 + (double)fullObs_cumulativeTime/M;

        
        double lambda_mean = alpha_n/beta_n;
        double lambda_stdev = sqrt(alpha_n)/beta_n;
        double lambda_up = lambda_mean + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
        double lambda_low = fmax(0, lambda_mean - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
        double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;

        //initialize an array for storing the Bayesian kernel of lambda
        vector<double> kernel (LAMBDA_STEP);
        double predictive=0;

        //calculate the kernel and predictive in Bayesian equation at the same time
        //#pragma omp parallel for schedule(static) reduction(+:predictive)
        for (int i=0; i<LAMBDA_STEP; i++)
        {
            double lambda = lambda_low + (i+0.5)*delta_lambda;
            kernel[i] = Likelihood(censoredObservations, lambda) * Gamma(lambda, alpha_n, beta_n); //kernel is equal to likelihood * prior

            predictive += kernel[i]; // predictive is obtained by integrate the kernel
        }

        predictive *= delta_lambda;

        //calculate the Bayesian posterior for lambda, posterior = kernel/predictive
        vector<double> lambda_pdf(LAMBDA_STEP);

        for (int i=0; i<LAMBDA_STEP; i++)
            lambda_pdf[i] = kernel[i]/predictive;

        //encapsule $lambda_low$, $delta_lambda$, and $pdf$ vector into the $lambda$ tuple
        lambda_tuple = make_tuple(lambda_low, delta_lambda, lambda_pdf);
        
        //save the newly obtained $lambda$ into $lambda_map$
        #pragma omp critical (lambda_map)
        {lambda_map.insert(make_pair(allObservations, lambda_tuple));}

    }

    return lambda_tuple;
}


//calculate the predictive distributions of current period observations
vector< vector<double> > observation_pdf_update(int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations)
{
    //initialize the output vector
    vector< vector<double> > observation_pdf(M+1, vector<double>(x));

    //first try to search for existing $observation_pdf$ in $observations_map$, based on the key $allObservations$
    auto allObservations = make_tuple(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    int count;
    #pragma omp critical (observation_map)
    {count = observation_map.count(allObservations);}
    
    //load the pdf if available in observations_map
    if (count==1)
    {
        //cout << "Obs\t" << x << "\t" << fullObs_cumulativeTime << "\t" << fullObs_cumulativeQuantity << "\t" << count << endl;
        #pragma omp critical (observation_map)
        {observation_pdf = observation_map[allObservations];}
    
    } else {
        
        //initialize predictive probability distributions of different kind of observations, with given prior on Lambda ~ Gamma(alpha_n, beta_n)
        //1. m=M, there is an exact observation, so the predictive just updates to NegBin(alpha_n, 1/(1+beta_n))
        //2. 0<m<M, the probability of observing demand d_m at the m-th checkpoint and having stockout in the coming sub-period, , i.e., P(D_m=x and D_1>=x-d_m)
        //3. m=0, keep track of the de-cumulative distribution of demand in the first sub-period of length 1/M, i.e., P(D_1>=x) where D_1 ~ NegBin[alpha_n, 1/(1+beta_n*M)]

        if (censoredObservations.size() == 0)
        {
            //update alpha,beta based on the exact observations
            double alpha_n = alpha0 + fullObs_cumulativeQuantity;
            double beta_n = beta0 + (double)fullObs_cumulativeTime/M;

            double lambda_mean = alpha_n/beta_n;
            double lambda_stdev = sqrt(alpha_n)/beta_n;
            double lambda_up = lambda_mean + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
            double lambda_low = fmax(0, lambda_mean - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
            double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;


            //for 1. m=M, nothing to initialize.
            for (int d=0; d<x; d++)
                observation_pdf[M][d] = NegBinomial(d, alpha_n, 1/(1+beta_n));

            
            //for 2. 0<m<M
            //#pragma omp parallel for schedule(static) collapse(2)
            for (int m=1; m<M; m++)
                for (int d_m=0; d_m<x; d_m++)
                {
                    double intg=0;

                    for (int i=0; i<LAMBDA_STEP; i++)
                    {
                        double lambda = lambda_low + (i+0.5) * delta_lambda;
                        intg += Poisson(d_m, lambda*m/M) * (1-Poisson_CDF(x-d_m-1, lambda/M)) * Gamma(lambda, alpha_n, beta_n);
                    }
                    intg *= delta_lambda;

                    observation_pdf[m][d_m] = intg;
                }

            //for 3. m=0
            observation_pdf[0][0] = 1 - NegBinomial(0, alpha_n, 1/(1+beta_n*M));
            for (int d_1=1; d_1<x; d_1++)
                observation_pdf[0][d_1] = observation_pdf[0][d_1-1] - NegBinomial(d_1, alpha_n, 1/(1+beta_n*M));

            ///////////////////////////////////////
            /*
            //test probability distributions
            double sum=observation_pdf[0][x-1];

            for (int m=1; m<M; m++)
                for (int d_m=0; d_m<x; d_m++)
                    sum +=  observation_pdf[m][d_m];//NegBinomial(d_m, m*alpha_n, 1/(1+beta_n*M)) * F_bar_1[x-d_m-1];

            for (int d=0; d<x; d++)
                sum += NegBinomial(d, alpha_n, 1/(1+beta_n));

             
            printf("%f\t", sum);
            //*/
            //////////////////////////////////////
        
        } else {
            
            //load/update pdf of lambda based on historical observations
            auto lambda_tuple = lambda_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);

            //read lambda_low, delta_lambda, and the pdf vector from the $lambda$ tuple
            double lambda_low = get<0>(lambda_tuple);
            double delta_lambda = get<1>(lambda_tuple);
            vector<double> lambda_pdf = get<2>(lambda_tuple);
            
            //initialize predictive probability distributions of different kind of observations, with given prior on Lambda ~ lambda_pdf[]
            //1. m=M, there is an exact observation, just calculate mixture of Poisson(lambda) and lambda_pdf[]
            //2. 0<m<M, the probability of observing demand d_m at the m-th checkpoint and having stockout in the coming sub-period, , i.e., P(D_m=x and D_1>x-d_m-1)
            //3. m=0, keep track of the de-cumulative distribution of demand in the first sub-period of length 1/M, i.e., P(D_1>x) where D_1 follows the mixture of Poisson(lambda/M) and lambda_pdf[]

            
            //for 1. m=M
            for (int d=0; d<x; d++)
            {
                double intg = 0;

                //#pragma omp parallel for schedule(static) reduction(+:intg)
                for (int i=0; i<LAMBDA_STEP; i++)
                {
                    intg += Poisson(d, lambda_low + (i+0.5) * delta_lambda) * lambda_pdf[i];
                }
                intg *= delta_lambda;

                observation_pdf[M][d] = intg;
            }

            //for 2. 0<m<M
            //#pragma omp parallel for schedule(static) collapse(2)
            for (int m=1; m<M; m++)
                for (int d_m=0; d_m<x; d_m++)
                {
                    double intg = 0;
                    
                    for (int i=0; i<LAMBDA_STEP; i++)
                    {
                        double lambda = lambda_low + (i+0.5) * delta_lambda;
                        intg += Poisson(d_m, lambda * m / M) * (1-Poisson_CDF(x-d_m-1, lambda/M))  * lambda_pdf[i];
                    }
                    intg *= delta_lambda;
                    
                    observation_pdf[m][d_m] = intg;
                }

            //for 3. m=0
            for (int d_1=0; d_1<x; d_1++)
            {
                double intg = 0;

                //#pragma omp parallel for schedule(static) reduction(+:intg)
                for (int i=0; i<LAMBDA_STEP; i++)
                {
                    intg += (1-Poisson_CDF(d_1, (lambda_low + (i+0.5) * delta_lambda)/M)) * lambda_pdf[i];
                }
                intg *= delta_lambda;

                observation_pdf[0][d_1] = intg;
            }

        }

        
        //save the newly calculated pdf vector into observation_map
        #pragma omp critical (observation_map)
        {observation_map.insert(make_pair(allObservations, observation_pdf));}
        
    }

    return observation_pdf;
}



//***********************************************************
//first-order difference of L := L(x+1) - L(x), for both cases with and without censoring
double L_prime(int x, int n, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations)
{
    
    double Phi_x = 0; //$Phi_x$ is for Prob{d <= x} with the given updated belief on lambda
    
    if (censoredObservations.size()==0)
    {
        //without censoring, lambda ~ Gamma(alpha_n, beta_n), and d ~ NegBin(alpha_n, p)
        //update alpha,beta based on the exact observations
        double alpha_n = alpha0 + fullObs_cumulativeQuantity;
        double beta_n = beta0 + (double)fullObs_cumulativeTime/M;
        double p = 1/(1+beta_n);
        
        
        for (int d=0; d<=x; d++)
            Phi_x += NegBinomial(d, alpha_n, p);
        
    } else {
        
        //load/update pdf of lambda based on historical observations
        auto lambda_tuple = lambda_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        //read lambda_low, delta_lambda, and the pdf vector from the $lambda$ tuple
        double lambda_low = get<0>(lambda_tuple);
        double delta_lambda = get<1>(lambda_tuple);
        vector<double> lambda_pdf = get<2>(lambda_tuple);
        
        
        //with censored observations, lambda ~ $lambda_pdf[]$, d|lambda ~ Poisson(lambda)
        //#pragma omp parallel for collapse(2) schedule(static) reduction(+:Phi_x)
        for (int d=0; d<=x; d++)
            for (int i=0; i<LAMBDA_STEP; i++)
            {
                Phi_x += Poisson(d, lambda_low + (i+0.5) * delta_lambda) * lambda_pdf[i];
            }
        
        Phi_x *= delta_lambda;
        
    }
	
    return price * (1 - Phi_x) - cost;

}


//search for myopic inventory level with updated knowledge about lambda
int find_x_myopic(int n, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations)
{
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + (double)fullObs_cumulativeTime/M;

    //bi-sectional search for x such that L_prime is zero

    int x;
    int x_up = (int) X_MYOPIC_UP_MULTIPLE_OF_MEAN * alpha_n / beta_n;
    int x_low = 0;
        
    while (x_up - x_low > 3)
    {
        x = (x_up + x_low)/2;
        
        double temp = L_prime(x, n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        if (temp > 0)
            x_low = x+1;
        else
            x_up = x;
    }

    for (x=x_low; x<=x_up; x++)
    {
        double temp = L_prime(x, n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);

        if (temp < 0)
            break;
    }
    
    return x;

}


//***********************************************************


//============================================
//Dynamic Program recursion
//============================================
//Parameters for function G() and V():
//int n: current period index
//int x: current inventory level

double G_CheckpointA(int n, int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations);

double V_CheckpointA(int n, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations)
{
    double v_max;
    
    if (n==N+1)
        v_max = 0;    //V_{N+1}()=0
    else
    {
        auto parameters = make_tuple(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        int count;
        #pragma omp critical (v_map)
            {count = v_map.count(parameters);}

        if (count==1)
        {
            //cout << "V\t" << n << "\t" << fullObs_cumulativeTime << "\t" << fullObs_cumulativeQuantity << "\t" << count << endl;
            #pragma omp critical (v_map)
            {v_max = v_map[parameters];}
        
        } else {
            
            //search for optimal inventory level x

            //initial the lower bound of x, use myopic inventory level as lower bound
            int x_low = find_x_myopic(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
            int x_opt = x_low;
            
            //evaluate low bound x_low
            v_max = G_CheckpointA(n, x_low, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);

//********************
//comment out the following block for Myopic policies (skipping the search for opitmal inventory level, use myopic inventory level instead) 
            //linear search from x_low onwards
            for (int x=x_low+1;;x++)
            {
                double temp = G_CheckpointA(n, x, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
            
                if (temp > v_max)
                {
                    x_opt = x;
                    v_max = temp;
                }
                else
                    break;
            }
//*******************/
            
            if (n==1)
            {
                cout << x_opt << "\t" << v_max  << "\t";
                file << x_opt << "\t" << v_max  << "\t";
            }
        
            
            #pragma omp critical (v_map)
            { v_map.insert(make_pair(parameters, v_max)); }

        }

    }

    return v_max;
}



double G_CheckpointA(int n, int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity, multiset<int> censoredObservations)
{

    //if x==0, jump to next period without learning or updating the system state (because without keeping inventory, there will be no observation, no cost, and no revenue anyhow)
    if (x==0)
        return V_CheckpointA(n+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    else
    {
    
        //load the predictive pdf of current period observation
        vector< vector<double> > observation_pdf = observation_pdf_update(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        
        //start calculate expected profit-to-go
        double out1 = 0;
        
        //Case I.1: when there is no stockout in the current period...
        //#pragma omp parallel for schedule(dynamic) reduction(+:out1)
        for (int d=0; d<x; d++)
        {
            out1 += ( price*d + V_CheckpointA(n+1, fullObs_cumulativeTime + M, fullObs_cumulativeQuantity + d, censoredObservations) ) * observation_pdf[M][d];
        }
        
        
        //Case I.2&3: when there is stockout in the current period...
        //Case I.2: stockout can happen after any checkpoint m=1, 2, ..., M-1; so iterate over all these checkpoints
        #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:out1)
        for (int m=1; m<M; m++)
        {
            //if stockout happen after checkpoint m, we observe: (1) demand D(m/M) = d_m fully and (2) censored observation D(1/M) >= x-d_m
            //iterate over d_m to calculate expected profit
            for (int d_m=0; d_m<x; d_m++)
            {
                multiset<int> censoredObservations_new = censoredObservations;
                censoredObservations_new.insert(x-d_m);
                
                out1 += ( price*x + V_CheckpointA(n+1, fullObs_cumulativeTime + m, fullObs_cumulativeQuantity + d_m, censoredObservations_new) ) * observation_pdf[m][d_m];
            }
        }
        
        //Case I.3: stockout happen before the first checkpoint
        //we observe demand D(1/M) >= x
        multiset<int> censoredObservations_new = censoredObservations;
        censoredObservations_new.insert(x);
        
        out1 += ( price*x + V_CheckpointA(n+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations_new) ) * observation_pdf[0][x-1];

        
        return out1 - cost*x;
    
    } //end-if

}




int main(void)
{
    //Open output file
    file.open("NVLearning_CheckpointA.txt", fstream::app|fstream::out);
    
    if (! file)
    {
        //if fail to open the file
        cerr << "can't open output file NVLearning_CheckpointA.txt!" << endl;
        exit(EXIT_FAILURE);
    }
	
    file << setprecision(8);
    cout << setprecision(8);
    
    omp_set_num_threads(omp_get_num_procs());
	
    cout << "Num of Procs: " << omp_get_num_procs() << endl;
    cout << "Max Num of Threads: " << omp_get_max_threads() << endl;
    cout << "Num of periods (N): " << N << endl;
    cout << "Max Num of checkpoints (M): " << M_MAX << endl;
    cout << "r\tc\talpha\tbeta\tM\tQ_A\tPi_A\tTime_A\tCPUTime_A" << endl;
        
    
    file << "Num of Procs: " << omp_get_num_procs() << endl;
    file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    file << "Num of periods (N): " << N << endl;
    file << "Max Num of checkpoints (M): " << M_MAX << endl;
    file << "r\tc\talpha\tbeta\tM\tQ_A\tPi_A\tTime_A\tCPUTime_A" << endl;
    
    
    
    //initialize cost parameters
    price = 2;
    cost = 1.0;
    
    //initialize info parameters
    int lambda_mean = 10;
    beta0 = 1;

    //initial observations are null
    multiset<int> nullSet;


    //for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
    //for (beta0=2; beta0>=0.05; beta0/=2)
    {
        alpha0 = beta0*lambda_mean;
    
        
        for (M=1; M<=M_MAX;M++)
        {
            //previously saved distributions of lambda needed to be cleared if there is any change in alpha, beta, or M
            lambda_map.clear();
            observation_map.clear();

            for (cost=1.8;cost>=0.15;cost-=0.1)
            {
                v_map.clear();


                cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t" << M << "\t";
                file << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t" << M << "\t";
            
                clock_t cpu_start = clock();
                auto startTime = chrono::system_clock::now();
                V_CheckpointA(1, 0, 0, nullSet);
                clock_t cpu_end = clock();
                auto endTime = chrono::system_clock::now();
            
                cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << endl;
                file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << endl;
            }
            
        }
        
    }

    
    file.close();

    return 0;
}

