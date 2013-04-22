//============================================================================
// Name        : NVLearning_Full.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v4.0 (2013-04-20)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Full-observation Model
//               compile using Intel icc: icpc -std=c++11 -openmp -O3 -fast -o NVLearning_Full.exe NVLearning_Full.cpp
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
//#include <omp.h>

using namespace std;

//***********************************************************

#define N 4                                         //number of periods

#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 10  //upper bound for D

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


//***********************************************************

//first-order difference of L := L(x+1) - L(x)
double L_prime(int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{
    
    double Phi_x = 0; //$Phi_x$ is for Prob{d <= x} with the given updated belief on lambda
    
    //lambda ~ Gamma(alpha_n, beta_n), and d ~ NegBin(alpha_n, p)
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + fullObs_cumulativeTime;
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
    double beta_n = beta0 + fullObs_cumulativeTime;

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

double G_F(int n, int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity);

double V_F(int n, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{
    double v_max;
    
    if (n==N+1)
        v_max = 0;    //V_{N+1}()=0
    else
    {
        auto parameters = make_tuple(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        int count;
        //#pragma omp critical (v_map)
        {count = v_map.count(parameters);}

        if (count==1)
        {
            //#pragma omp critical (v_map)
            {v_max = v_map[parameters];}
        
        } else {
            
            //no need to search for optimal inventory level x, the myopic inventory level is optimal
            int x_opt = find_x_myopic(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            
            v_max = G_F(n, x_opt, fullObs_cumulativeTime, fullObs_cumulativeQuantity);

            
            if (n==1)
            {
                cout << x_opt << "\t" << v_max  << "\t";
                file << x_opt << "\t" << v_max  << "\t";
            }
        
            
            //#pragma omp critical (v_map)
            { v_map.insert(make_pair(parameters, v_max)); }

        }

    }

    return v_max;
}



double G_F(int n, int x, int fullObs_cumulativeTime, int fullObs_cumulativeQuantity)
{

    double r = alpha0 + fullObs_cumulativeQuantity;
    double p = 1/(1 + beta0 + fullObs_cumulativeTime);
    
    double d_mean = r*p/(1-p);
    double d_var = r*p/pow(1-p,2.0);
    int d_up = d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*sqrt(d_var);  //upper bound of D is set to be equal to $mean + N*var$

    //start calculate expected profit-to-go
    double out1 = 0;
    
    //#pragma omp parallel for schedule(dynamic) reduction(+:out1)
    for (int d=0; d<=d_up; d++)
    {
        out1 += ( price * min(d,x) + V_F(n+1, fullObs_cumulativeTime + 1, fullObs_cumulativeQuantity + d) ) * NegBinomial(d, r, p);
    }
    
    
    return out1 - cost*x;
    
}




int main(void)
{
    //Open output file
    file.open("NVLearning_Full.txt", fstream::app|fstream::out);
    
    if (! file)
    {
        //if fail to open the file
        cerr << "can't open output file NVLearning_Full.txt!" << endl;
        exit(EXIT_FAILURE);
    }
	
    file << setprecision(10);
    cout << setprecision(10);
    
//    omp_set_num_threads(omp_get_num_procs());
	
//    cout << "Num of Procs: " << omp_get_num_procs() << endl;
//    cout << "Max Num of Threads: " << omp_get_max_threads() << endl;
    cout << "Num of periods (N): " << N << endl;
    cout << "r\tc\talpha\tbeta\tQ_F\tPi_F\tTime_F\tCPUTime_F" << endl;
        
    
//    file << "Num of Procs: " << omp_get_num_procs() << endl;
//    file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    file << "Num of periods (N): " << N << endl;
    file << "r\tc\talpha\tbeta\tQ_F\tPi_F\tTime_F\tCPUTime_F" << endl;
    
    
    
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
            V_F(1, 0, 0);
            clock_t cpu_end = clock();
            auto endTime = chrono::system_clock::now();
            
            cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << endl;
            file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << endl;
        }
        
    }

    
    file.close();

    return 0;
}

