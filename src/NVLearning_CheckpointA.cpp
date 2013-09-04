//============================================================================
// Name        : NVLearning_CheckpointA.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v8.0 (2013-09-03)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Checkpoint-A (Inventory Checkpoint) heuristic
//============================================================================

//***********************************************************

#include <iostream>
#include <fstream>
#include <iomanip> //required by setprecision()

#include <cmath>
#include <numeric>

#include <chrono>
#include <ctime>

#include <set>
#include <tuple>
#include <vector>
#include <boost/unordered_map.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/set.hpp>
#include "serialize_tuple.h"
#include "unordered_map_serialization.h"

#include <omp.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

//***********************************************************

#define N 4                     //number of periods
#define M_MAX 4                 //Maximum number of checkpoints to be considered
#define LAMBDA_STEP 1000        //discretize the continuous distribution of lambda into LAMBDA_STEP=1000 pieces

#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 10  //upper bound for D
#define LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 8
#define LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 4

//***********************************************************

int M;                                              //Number of checkpoints within a period

double price, cost;                                 //Newsvendor price and cost parameters
double alpha0, beta0;                               //Initial prior of lambda is Gamma(alpha0, beta0)
double lambda_mean;                                 //Mean of lambda = alpha0/beta0;
int myopic;

ofstream file;                                      //output files

string path, modelName, scenarioName;               //model name used in naming archive files
string resultFile, archiveFile;                     //output and archive file names


auto startTime = chrono::system_clock::now();       //time point of starting calculation
auto endTime = chrono::system_clock::now();         //time point of finishing calculation
auto lastTime = chrono::system_clock::now();        //time point of last archiving
int lastMapSize;

boost::unordered_map<tuple<int, int, multiset<int>>, tuple<double, double, vector<double>>> lambda_map;     //a boost::unordered_map to store updated distributions of lambda
boost::unordered_map<tuple<int, int, multiset<int>>, vector<double> > demand_map;                           //a boost::unordered_map to store updated demand distributions
boost::unordered_map<tuple<int, int, int, multiset<int>>, vector<vector<double>> > observation_map;         //a boost::unordered_map to store predictive distributions of observations
boost::unordered_map<tuple<int, int, int, multiset<int>>, double> v_map;                                    //a boost::unordered_map to store calculated value of the V() function


//***********************************************************

string dbl_to_str(const double& f)
{
    
    string str = to_string (f);
    
    str.erase ( str.find_last_not_of('0') + 1, std::string::npos );
    
    if (str.back() == '.') str.pop_back();
    
    return str;
}


//Functions for calculating various probability distributions
double Poisson(const int& xx, const double& lam)
{
    double log_pmf = xx*log(lam) -lam - lgamma(xx+1);
    
    return exp(log_pmf);
}


double Poisson_CDF(const int& xx, const double& lam)
{
    double out=0;
    
    for (int i=0; i<=xx; ++i)
        out += Poisson(i,lam);
    
    if (out >1) out = 1;
    if (out <0) out = 0;
    
    return out;
}


double Gamma(const double& xx, const double& aa, const double& bb)
{
    double log_pdf = 0;
    
    if ((xx>=0)&&(aa>0)&&(bb>0))
        log_pdf = aa*log(bb) + log(xx)*(aa-1)  -bb*xx - lgamma(aa);
    
    return exp(log_pdf);
}


double NegBinomial(const int& kk, const double& rr, const double& pp)
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
//  multiset<int> censoredObservations: the set of censored observations in previous sub-periods, values stored are the inventory levels at the previous checkpoint



//calculate the likelihood of observing all the historical censored observations with a given lambda
//Prob(D(1/M)>=censoredObservations[0]) * Prob(D(1/M)>=censoredObservations[1]) * ...
double Likelihood(const multiset<int> & censoredObservations, const double& lambda)
{
    double out = 1;
    double lam = lambda/M;
    
    //for each censored observation $co$ in the set $censoredObservations$, calculate its probability
    for (int co : censoredObservations)
    {
        //Probability of a $co$: Prob{ D(1/M) >= co }
        double F = 0;
        for (int d1=0; d1<co; ++d1)
            F += Poisson(d1, lam);
        
        out *= max(0.0, 1 - F);
    }
    
    return out;
}


//brute-force Bayesian updating of the pdf of lambda based on historical observations
tuple<double, double, vector<double>> lambda_pdf_update(const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    //initialize the output tuple
    tuple<double, double, vector<double>> lambda_tuple;
    
    //first try to search for existing $lambda$ in $lambda_map$, based on the key $allObservations$
    auto allObservations = make_tuple(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    
    bool found = false;
#pragma omp critical (lambda_map)
    {
        auto  it = lambda_map.find(allObservations);
        if (it != lambda_map.end())
        {
            found = true;
            lambda_tuple = it->second;
        }
    }
    
    if (!found)
    {
        
        //update alpha,beta based on the exact observations
        double alpha_n = alpha0 + fullObs_cumulativeQuantity;
        double beta_n = beta0 + double(fullObs_cumulativeTime)/M;
        
        
        double lambda_mu = alpha_n/beta_n;
        double lambda_stdev = sqrt(alpha_n)/beta_n;
        double lambda_up = lambda_mu + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
        double lambda_low = max(0.0, lambda_mu - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
        double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;
        
        //initialize an array for storing the Bayesian kernel of lambda
        vector<double> kernel (LAMBDA_STEP);
        double predictive=0;
        
        //calculate the kernel and predictive in Bayesian equation at the same time
        //#pragma omp parallel for schedule(static) reduction(+:predictive)
        for (int i=0; i<LAMBDA_STEP; ++i)
        {
            double lambda = lambda_low + (i+0.5)*delta_lambda;
            kernel[i] = Likelihood(censoredObservations, lambda) * Gamma(lambda, alpha_n, beta_n); //kernel is equal to likelihood * prior
            
            predictive += kernel[i]; // predictive is obtained by integrate the kernel
        }
        
        predictive *= delta_lambda;
        
        //calculate the Bayesian posterior for lambda, posterior = kernel/predictive
        vector<double> lambda_pdf(LAMBDA_STEP);
        
        for (int i=0; i<LAMBDA_STEP; ++i)
            lambda_pdf[i] = kernel[i]/predictive;
        
        //encapsule $lambda_low$, $delta_lambda$, and $pdf$ vector into the $lambda$ tuple
        lambda_tuple = make_tuple(lambda_low, delta_lambda, lambda_pdf);
        
        //save the newly obtained $lambda$ into $lambda_map$
#pragma omp critical (lambda_map)
        {lambda_map.emplace(allObservations, lambda_tuple);}
        
    }
    
    return lambda_tuple;
}



//calculate the predictive distributions of current period demand
vector<double> demand_pdf_update(const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + double(fullObs_cumulativeTime)/M;
    double p_n = 1 / (1+beta_n);
    
    double d_mean = alpha_n*p_n / (1-p_n);
    double d_var = alpha_n*p_n / pow(1-p_n,2.0);
    int d_up = d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*sqrt(d_var);  //upper bound of D is set to be equal to $mean + N*var$
    
    //initialize the output vector
    vector<double> demand_pdf (d_up+1);
    
    //first try to search for existing $demand_pdf$ in $demand_map$, based on the key $allObservations$
    auto allObservations = make_tuple(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    
    bool found = false;
#pragma omp critical (demand_map)
    {
        auto  it = demand_map.find(allObservations);
        if (it != demand_map.end())
        {
            found = true;
            demand_pdf = it->second;
        }
    }
    
    if (!found)
    {
        
        if (censoredObservations.empty())
        {
            for (int d=0; d<=d_up; ++d)
                demand_pdf[d] = NegBinomial(d, alpha_n, p_n);
            
        } else {
            
            //load/update pdf of lambda based on historical observations
            auto lambda_tuple = lambda_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
            
            //read lambda_low, delta_lambda, and the pdf vector from the $lambda$ tuple
            double lambda_low = get<0>(lambda_tuple);
            double delta_lambda = get<1>(lambda_tuple);
            vector<double> lambda_pdf = get<2>(lambda_tuple);
            
            
            //update demand_pdf
            for (int d=0; d<=d_up; ++d)
            {
                double intg = 0;
                
                //#pragma omp parallel for schedule(static) reduction(+:intg)
                for (int i=0; i<LAMBDA_STEP; ++i)
                {
                    intg += Poisson(d, lambda_low + (i+0.5) * delta_lambda) * lambda_pdf[i];
                }
                intg *= delta_lambda;
                
                demand_pdf[d] = intg;
            }
            
        }
        
        //save the newly calculated pdf into demand_map
#pragma omp critical (demand_map)
        {demand_map.emplace(allObservations, demand_pdf);}
        
    }
    
    
    return demand_pdf;
}


//calculate the predictive distributions of current period checkpoint observations
vector< vector<double> > observation_pdf_update(const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    //initialize the output vector
    vector< vector<double> > observation_pdf(M, vector<double>(x));
    
    //first try to search for existing $observation_pdf$ in $observations_map$, based on the key $allObservations$
    auto allObservations = make_tuple(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    
    bool found = false;
#pragma omp critical (observation_map)
    {
        auto  it = observation_map.find(allObservations);
        if (it != observation_map.end())
        {
            found = true;
            observation_pdf = it->second;
        }
    }
    
    if (!found)
    {
        
        //initialize predictive probability distributions of checkpoint observations, with updated prior on Lambda
        //for 0<m<M, the probability of observing demand d_m at the m-th checkpoint and having stockout in the following sub-period, i.e., P(D_m=x and D_1>=x-d_m)
        //for m=0, keep track of the de-cumulative distribution of demand in the first sub-period of length 1/M, i.e., P(D_1>=x)
        
        if (censoredObservations.empty())
        {
            //update alpha,beta based on the exact observations
            double alpha_n = alpha0 + fullObs_cumulativeQuantity;
            double beta_n = beta0 + double(fullObs_cumulativeTime)/M;
            
            double lambda_mu = alpha_n/beta_n;
            double lambda_stdev = sqrt(alpha_n)/beta_n;
            double lambda_up = lambda_mu + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
            double lambda_low = max(0.0, lambda_mu - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
            double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;
            
            
            //for 0<m<M
            //#pragma omp parallel for schedule(static) collapse(2)
            for (int m=1; m<M; ++m)
            {
                for (int d_m=0; d_m<x; ++d_m)
                {
                    double intg=0;
                    
                    for (int i=0; i<LAMBDA_STEP; ++i)
                    {
                        double lambda = lambda_low + (i+0.5) * delta_lambda;
                        intg += Poisson(d_m, lambda*m/M) * (1-Poisson_CDF(x-d_m-1, lambda/M)) * Gamma(lambda, alpha_n, beta_n);
                    }
                    intg *= delta_lambda;
                    
                    observation_pdf[m][d_m] = intg;
                }
            }
            
            //for m=0
            double p_nm = 1/(1+beta_n*M);
            observation_pdf[0][x-1] = 1;
            for (int d_1=0; d_1<x; ++d_1)
                observation_pdf[0][x-1] -= NegBinomial(d_1, alpha_n, p_nm);
            
        } else {
            
            //load/update pdf of lambda based on historical observations
            auto lambda_tuple = lambda_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
            
            //read lambda_low, delta_lambda, and the pdf vector from the $lambda$ tuple
            double lambda_low = get<0>(lambda_tuple);
            double delta_lambda = get<1>(lambda_tuple);
            vector<double> lambda_pdf = get<2>(lambda_tuple);
            
            //for 0<m<M
            //#pragma omp parallel for schedule(static) collapse(2)
            for (int m=1; m<M; ++m)
            {
                for (int d_m=0; d_m<x; ++d_m)
                {
                    double intg = 0;
                    
                    for (int i=0; i<LAMBDA_STEP; ++i)
                    {
                        double lambda = lambda_low + (i+0.5) * delta_lambda;
                        intg += Poisson(d_m, lambda*m/M) * (1-Poisson_CDF(x-d_m-1, lambda/M))  * lambda_pdf[i];
                    }
                    intg *= delta_lambda;
                    
                    observation_pdf[m][d_m] = intg;
                }
            }
            
            //for m=0
            double intg = 0;
            
            //#pragma omp parallel for schedule(static) reduction(+:intg)
            for (int i=0; i<LAMBDA_STEP; ++i)
            {
                intg += (1-Poisson_CDF(x-1, (lambda_low + (i+0.5) * delta_lambda)/M)) * lambda_pdf[i];
            }
            intg *= delta_lambda;
            
            observation_pdf[0][x-1] = intg;
            
        }
        
        
        //save the newly calculated pdf vector into observation_map
#pragma omp critical (observation_map)
        {observation_map.emplace(allObservations, observation_pdf);}
    }
    
    return observation_pdf;
}



//***********************************************************
//search for myopic inventory level with updated knowledge about lambda
int find_x_myopic(const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    //load the predictive pdf of current period observation
    vector<double> demand_pdf = demand_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    int d_up = demand_pdf.size() - 1;
    
    //bi-sectional search for x such that L_prime is zero
    
    int x;
    int x_up = d_up;
    int x_low = 0;
    
    vector<double> demand_cdf (d_up+1);
    demand_cdf[0] = demand_pdf[0];
    for (int d=1; d<=d_up; ++d)
        demand_cdf[d] = min(1.0, demand_cdf[d-1] + demand_pdf[d]);
    
    
    while (x_up - x_low > 3)
    {
        x = (x_up + x_low)/2;
        
        double temp = price * (1 -  demand_cdf[x]) - cost;
        
        if (temp > 0)
            x_low = x+1;
        else
            x_up = x;
    }
    
    for (x=x_low; x<=x_up; ++x)
    {
        double temp = price * (1 -  demand_cdf[x]) - cost;
        
        if (temp < 0)
            break;
    }
    
    return x;
    
}


//***********************************************************

void archiveVMap(const boost::unordered_map<tuple<int, int, int, multiset<int>>, double> & vMap, const string& ofile)
{
    ofstream ofs(ofile); //open the temporary archive file
    if (ofs.good())
    {
        boost::archive::text_oarchive oa(ofs); //initialize boost::archive::text_oarchive
        oa << vMap; //serialize v_map and save it to file
    }
    ofs.close();
}


void importVMap(boost::unordered_map<tuple<int, int, int, multiset<int>>, double> & vMap, const string& ifile)
{
    ifstream ifs(ifile); //open archive file to read
    if (ifs.good())
    {
        boost::archive::text_iarchive ia(ifs); //initialize text_iarchive
        ia >> vMap; // restore v_map from the archive
        cout << "[INFO:] IMPORTED " << vMap.size() << " records from " << ifile << endl;
    }
    ifs.close();
}


//============================================
//Dynamic Program recursion
//============================================
//Parameters for function G() and V():
//int n: current period index
//int x: current inventory level

double G_CheckpointA(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations);
double G_CheckpointA_UpperBound(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations);

double V_CheckpointA(const int& n, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    double v_max;
    
    if (n == N+1)
        v_max = 0;    //V_{N+1}()=0
    else
    {
        auto parameters = make_tuple(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        bool found = false;
#pragma omp critical (v_map)
        {
            auto  it = v_map.find(parameters);
            if (it != v_map.end()) {
                found = true;
                v_max = it->second;
            }
        }
        
        
        if (!found)
        {
            //search for optimal inventory level x
            
            //initial the lower bound of x, use myopic inventory level as lower bound
            int x_myopic = find_x_myopic(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
            int x_opt = x_myopic;
            int x = x_myopic;
            
            //evaluate low bound x_low
            v_max = G_CheckpointA(n, x_myopic, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
            
            if (myopic == 0)
            {
                //linear search from x_low onwards, until reaches the upper bound of x
                for (x=x_myopic+1; ; ++x)
                {
                    double temp = G_CheckpointA(n, x, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
                    
                    if (temp > v_max)
                    {
                        x_opt = x;
                        v_max = temp;
                    }
                    //else //if assume concavity, searching can stop once the first order condition is met, without checking the bound
                    if (temp >= G_CheckpointA_UpperBound(n, x+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations))
                        break;
                }
            }
            
            if (n > 1) //when n==1, do not save v_max into v_map, calculate it instead so that we will have the value of x_opt (optimal inventory level information is not store in v_map, so have to calculate here)
            {
#pragma omp critical (v_map)
                {v_map.emplace(parameters, v_max);}
            }
            
            //save the whole v_map into $archiveFile$ every hour
            if (omp_get_thread_num() == 0) //only Thread 0 is in charge of archiving
            {
                auto currentTime = chrono::system_clock::now();  //get current time
                
                if (chrono::duration_cast<chrono::hours> (currentTime-lastTime).count() >= 1)
                {
                    int mapSize = v_map.size();
                    if (mapSize > lastMapSize)
                    {
                        
                        lastTime = currentTime; //update time of last archive
                        lastMapSize = mapSize; //update last archive size
                        
                        
                        //#pragma omp critical (archive)
                        {archiveVMap(v_map, archiveFile);}
                        cout << "[INFO:] ARCHIEVED V_map " << mapSize << " records to " << archiveFile << " at Hour " << chrono::duration_cast<chrono::hours> (currentTime-startTime).count() << "." << endl;
                    }
                }
            }
            
            
            
            if (n == 1)
            {
                cout << x << "\t" << x_opt << "\t" << v_max  << "\t";
                file << x << "\t" << x_opt << "\t" << v_max  << "\t";
            }
            
        }
        
    }
    
    return v_max;
}



double G_CheckpointA(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    
    //if x==0, jump to next period without learning or updating the system state (because without keeping inventory, there will be no observation, no cost, and no revenue anyhow)
    if (x==0)
        return V_CheckpointA(n+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    else
    {
        
        //load the predictive pdf of current period demand and observations
        vector<double> demand_pdf = demand_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        vector< vector<double> > observation_pdf = observation_pdf_update(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        
        //start calculate expected profit-to-go
        double out1 = 0;
        
        //Case I.1: when there is no stockout in the current period...
        //#pragma omp parallel for schedule(dynamic) reduction(+:out1)
        for (int d=0; d<x; ++d)
        {
            out1 += ( price*d + V_CheckpointA(n+1, fullObs_cumulativeTime + M, fullObs_cumulativeQuantity + d, censoredObservations) ) * demand_pdf[d];
        }
        
        
        //Case I.2&3: when there is stockout in the current period...
        //Case I.2: stockout can happen after any checkpoint m=1, 2, ..., M-1; so iterate over all these checkpoints
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:out1)
        for (int m=1; m<M; ++m)
        {
            //if stockout happen after checkpoint m, we observe: (1) demand D(m/M) = d_m fully and (2) censored observation D(1/M) >= x-d_m
            //iterate over d_m to calculate expected profit
            for (int d_m=0; d_m<x; ++d_m)
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


double G_CheckpointA_UpperBound(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<int> & censoredObservations)
{
    
    //load the predictive pdf of current period demand
    vector<double> demand_pdf = demand_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    int d_up = demand_pdf.size() - 1;
    
    //start calculate expected profit-to-go
    double out1 = 0;
    
    for (int d=0; d<=d_up; ++d)
        out1 += ( price* min(d,x) + V_CheckpointA(n+1, fullObs_cumulativeTime + M, fullObs_cumulativeQuantity + d, censoredObservations) ) * demand_pdf[d];
    
    return out1 - cost*x;
    
}







int main(int ac, char* av[])
{
    //read and parse command line inputs (using boose::program_options)
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    //scenario parameters
    ("myopic", po::value<int>(&myopic)->default_value(0), "adopt myopic inventory policy?")
    ("lambda,l", po::value<double>(&lambda_mean)->default_value(10), "mean demand (E[lambda])")
    ("beta,b", po::value<double>(&beta0)->default_value(1), "beta")
    ("numberOfCheckpoints,M", po::value<int>(&M)->default_value(2), "number of checkpoints (M>=1)")
    ("cost,c", po::value<double>(&cost)->default_value(1), "unit cost")
    //file names input
    ("path,p", po::value<string>(&path), "path for temporary archive files")
    ("outfile,o", po::value<string>(&resultFile), "file to save result")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);
    
    
    if (vm.count("help")) {
        cout << "Usage: options_description [options]\n";
        cout << desc;
        return 0;
    }
    
    
    
    //initialize the file names by taking all parameters
    modelName = "NVLearning_CheckpointA";
    if (myopic != 0) modelName = modelName + "_myopic";
    if (resultFile == "") resultFile = modelName + ".result.txt";
    
    
    
    //Open output file
    file.open(resultFile, fstream::app|fstream::out);
    
    if (! file)
    {
        //if fail to open the file
        cerr << "can't open output file " << resultFile << endl;
        exit(EXIT_FAILURE);
    }
	
    
    file << setprecision(10);
    cout << setprecision(10);
    
    
    omp_set_num_threads(omp_get_num_procs());
	
    cout << "Num of Procs: " << omp_get_num_procs() << endl;
    cout << "Max Num of Threads: " << omp_get_max_threads() << endl;
    cout << "Num of periods (N): " << N << endl;
    cout << "Max Num of checkpoints (M): " << M_MAX << endl;
    if (myopic==0)
        cout << "r\tc\talpha\tbeta\tM\tQ_A_bar\tQ_A\tPi_A\tTime_A\tCPUTime_A\tComp_A" << endl;
    else
        cout << "r\tc\talpha\tbeta\tM\tQ_Am_bar\tQ_Am\tPi_Am\tTime_Am\tCPUTime_Am\tComp_Am" << endl;
    
    
    //file << "Num of Procs: " << omp_get_num_procs() << endl;
    //file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    //file << "Num of periods (N): " << N << endl;
    //file << "Max Num of checkpoints (M): " << M_MAX << endl;
    if (myopic==0)
        file << "r\tc\talpha\tbeta\tM\tQ_A_bar\tQ_A\tPi_A\tTime_A\tCPUTime_A\tComp_A" << endl;
    else
        file << "r\tc\talpha\tbeta\tM\tQ_Am_bar\tQ_Am\tPi_Am\tTime_Am\tCPUTime_Am\tComp_Am" << endl;
    
    
    
    //initialize cost parameters
    price = 2;
    //lambda_mean = 10;
    //beta0 = 1;
    //M = 2;
    //cost = 1;
    
    //setup initial observations
    multiset<int> initialObservations;
    
    
    
    //for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
    //for (beta0=2; beta0>=0.05; beta0/=2)
    {
        alpha0 = beta0*lambda_mean;
        
        
        //for (M=2; M<=M_MAX; M*=2)
        {
            
            for (cost=1.8; cost>=0.15; cost-=0.1)
            {
				//previously saved distributions of lambda needed to be cleared
                lambda_map.clear();
                demand_map.clear();
                observation_map.clear();
                v_map.clear();
                
                cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t" << M << "\t";
                file << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t" << M << "\t";
                
                
                //try load archived data into v_map
                scenarioName = ".l" + dbl_to_str(lambda_mean) +".b" + dbl_to_str(beta0) + ".M" + to_string(M) + ".c" + dbl_to_str(cost);
                archiveFile = path + modelName + scenarioName + ".oarchive.txt";
                
                
                importVMap(v_map, archiveFile);
                lastMapSize = v_map.size();
                
                
                startTime = chrono::system_clock::now();
                lastTime = startTime;
                clock_t cpu_start = clock();
                
                V_CheckpointA(1, 0, 0, initialObservations);
                
                clock_t cpu_end = clock();
                endTime = chrono::system_clock::now();
                
                cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
                file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
                
                
                //archive again upon finishing
                int mapSize = v_map.size();
                if (mapSize > lastMapSize)
                {
                    archiveVMap(v_map, archiveFile);
                }
                cout << mapSize+1 << endl;
                file << mapSize+1 << endl;
                
            }
            
        }
        
    }
    
    
    file.close();
    
    return 0;
}
