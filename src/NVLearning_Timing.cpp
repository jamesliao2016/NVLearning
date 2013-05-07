//============================================================================
// Name        : NVLearning_Timing.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v4.0 (2013-04-20)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Stock-out Timing Model
//               compile using Intel icc: icpc -std=c++11 -openmp -O3 -fast -o NVLearning_Timing.exe NVLearning_Timing.cpp
//============================================================================

//***********************************************************

#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <numeric>
#include <iomanip> //required by setprecision()
#include <map>
#include <tuple>
#include <omp.h>

using namespace std;

//***********************************************************

#define N 4                                         //number of periods
#define T_STEP 1000                                 //discretize the [0,1] interval into 1000 segments
#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10             //upper bound for X

//***********************************************************

double price, cost;                             //Newsvendor price and cost parameters
double alpha0, beta0;                           //Initial prior of lambda is Gamma(alpha0, beta0)

ofstream file;                                  //output files

map<tuple<int, int, int>, double> v_map;        //an std::map to store calculated value of the V() function

//***********************************************************
//Functions for calculating various probability distributions


double Poisson(int xx, double lam)
{
    double log_pmf = xx*log(lam) -lam - lgamma(xx+1);

    return exp(log_pmf);
}


double NegBinomial(int kk, double rr, double pp)
{
    double log_pmf = 0;

    if ((kk>=0)&&(rr>0)&&(pp>=0)&&(pp<=1))
        log_pmf = lgamma(kk+rr) - lgamma(kk+1) - lgamma(rr)  + rr*log(1-pp) + kk*log(pp);

    return exp(log_pmf);
}


double InvBeta2(double tt, int xx, double aa, double bb)
{
	if (tt==0) return 0;
	else
	{
		double log_pdf = 0;
        
		if ((tt>0)&&(xx>0)&&(aa>0)&&(bb>0))
			log_pdf = (xx-1)*log(tt) + aa*log(bb) - (xx+aa) * log(tt+bb) - ( lgamma(xx) + lgamma(aa) - lgamma(xx+aa) ) ;
        //pdf = pow(tt, xx-1) * pow(bb, aa) * pow(tt+bb, -xx-aa) / Beta(xx,aa);
        
		return exp(log_pdf);
	}
}



//***********************************************************

//first-order difference of L := L(x+1) - L(x)
double L_prime(int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{
    
    double Phi_x = 0; //$Phi_x$ is for Prob{d <= x} with the given updated belief on lambda
    
    //lambda ~ Gamma(alpha_n, beta_n), and d ~ NegBin(alpha_n, p)
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + (double)fullObs_cumulativeTime/T_STEP;
    double p_n = 1/(1+beta_n);
    
    
    for (int d=0; d<=x; d++)
        Phi_x += NegBinomial(d, alpha_n, p_n);
    
	
    return price * (1 - Phi_x) - cost;

}


//search for myopic inventory level with updated knowledge about lambda
int find_x_myopic(int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + (double)fullObs_cumulativeTime/T_STEP;

    //bi-sectional search for x such that L_prime is zero

    int x;
    int x_up = (int) X_MYOPIC_UP_MULTIPLE_OF_MEAN * alpha_n / beta_n;
    int x_low = 0;
        
    while (x_up - x_low > 3)
    {
        x = (x_up + x_low)/2;
        
        double temp = L_prime(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        
        if (temp > 0)
            x_low = x+1;
        else
            x_up = x;
    }

    for (x=x_low; x<=x_up; x++)
    {
        double temp = L_prime(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);

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

double G_T(int n, int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity);

double V_T(int n, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{
    double v_max;
    
    if (n==N+1)
        v_max = 0;    //V_{N+1}()=0
    else
    {
        auto parameters = make_tuple(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        int count;
        #pragma omp critical (v_map)
        {count = v_map.count(parameters);}

        if (count==1)
        {
            #pragma omp critical (v_map)
            {v_max = v_map[parameters];}
        
        } else {
            
            //search for optimal inventory level x
            
            //initial the lower bound of x, use myopic inventory level as lower bound
            int x_low = find_x_myopic(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            int x_opt = x_low;
            
            //evaluate low bound x_low
            v_max = G_T(n, x_low, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            
            //********************
             //comment out the following block for Myopic policies (skipping the search for opitmal inventory level, use myopic inventory level instead)
             //linear search from x_low onwards
             for (int x=x_low+1;;x++)
             {
                 double temp = G_T(n, x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
             
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



double G_T(int n, int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{

    //if x==0, jump to next period without learning or updating the system state (because without keeping inventory, there will be no observation, no cost, and no revenue anyhow)
    if (x==0)
        return V_T(n+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
    else
    {
        double alpha_n = alpha0 + fullObs_cumulativeQuantity;
        double beta_n = beta0 + (double)fullObs_cumulativeTime/T_STEP;
        double r = alpha_n;
        double p = 1/(1 + beta_n);
        
        /***********************
         //test probability distributions
         double prob1=0;
         for (int d=0; d<x; d++)
         prob1 += NegBinomial(d, r, p);
         
         double prob2=0;
         for (int i=1;i<=T_STEP;i+=2)
         prob2 += InvBeta2((double)i/T_STEP, x, alpha_n, beta_n);
         prob2 *= 2.0/T_STEP;
         
         cout << prob1+prob2 << endl;
         //************************/
        

        //start calculate expected profit-to-go
        
        //case 1: no stockout happening
        double out1 = 0;

        //#pragma omp parallel for schedule(dynamic) reduction(+:out1)
        for (int d=0; d<x; d++)
        {
            out1 += ( price * d + V_T(n+1, fullObs_cumulativeTime + T_STEP, fullObs_cumulativeQuantity + d) ) * NegBinomial(d, r, p);
        }
        
        //case 2: stockout at sometime before the end of the period
        double out2 = 0;

        //take stepsize=2 to speed up the calculation of the integral
        #pragma omp parallel for schedule(dynamic) reduction(+:out2)
        for (int i=1;i<=T_STEP;i+=2)
        {
            out2 += ( price * x + V_T(n+1, fullObs_cumulativeTime + i, fullObs_cumulativeQuantity + x) ) * InvBeta2((double)i/T_STEP, x, alpha_n, beta_n); //integral at ticks 1, 3, 5, ..., 999
        }
        out2 *= 2.0/T_STEP; //dt = 2/T_STEP

        
        return out1 + out2 - cost*x;
    }
}




int main(void)
{
    //Open output file
    file.open("NVLearning_Timing.txt", fstream::app|fstream::out);
    
    if (! file)
    {
        //if fail to open the file
        cerr << "can't open output file NVLearning_Timing.txt!" << endl;
        exit(EXIT_FAILURE);
    }
	
    file << setprecision(10);
    cout << setprecision(10);
    
    omp_set_num_threads(omp_get_num_procs());
	
    cout << "Num of Procs: " << omp_get_num_procs() << endl;
    cout << "Max Num of Threads: " << omp_get_max_threads() << endl;
    cout << "Num of periods (N): " << N << endl;
    cout << "r\tc\talpha\tbeta\tQ_T\tPi_T\tTime_T\tCPUTime_T" << endl;
        
    
    file << "Num of Procs: " << omp_get_num_procs() << endl;
    file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    file << "Num of periods (N): " << N << endl;
    file << "r\tc\talpha\tbeta\tQ_T\tPi_T\tTime_T\tCPUTime_T" << endl;
    

    
    //initialize cost parameters
    price = 2;
    cost = 1;
    
    //initialize info parameters
    int lambda_mean = 10;
    beta0 = 1;


    for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
    for (beta0=2; beta0>=0.05; beta0/=2)
    {
        alpha0 = beta0*lambda_mean;
    
        for (cost=1.8;cost>=0.15;cost-=0.1)
        {
            v_map.clear();


            cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t";
            file << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t";
        
            clock_t cpu_start = clock();
            auto startTime = chrono::system_clock::now();
            V_T(1, 0, 0);
            clock_t cpu_end = clock();
            auto endTime = chrono::system_clock::now();
        
            cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << endl;
            file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << endl;
        }
            
    }

    
    file.close();

    return 0;
}

