//============================================================================
// Name        : NVLearning_CheckpointB_myopic.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v7.2 (2013-07-16)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Checkpoint-B (Stock-out Checkpoint) myopic heuristic
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
//#include <map>
#include <boost/unordered_map.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/set.hpp>
//#include <boost/serialization/map.hpp>
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

#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10
#define LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 8
#define LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 4

//***********************************************************

int M;                          //Number of checkpoints within a period
vector<double> tau;             //timepoint of each checkpoints

double price, cost;             //Newsvendor price and cost parameters
double alpha0, beta0;           //Initial prior of lambda is Gamma(alpha0, beta0)
double lambda_mean;             //Mean of lambda = alpha0/beta0;
int myopic;

ofstream file;                  //output files

string path, modelName, scenarioName;               //model name used in naming archive files
string resultFile;                                  //result file
vector<string> archiveFile;                         //archive file names


auto startTime = chrono::system_clock::now();       //time point of starting calculation
auto endTime = chrono::system_clock::now();         //time point of finishing calculation
auto lastTime = chrono::system_clock::now();        //time point of last archiving


boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, tuple<double, double, vector<double>>> lambda_map;                 //a boost::unordered_map to store updated distributions of lambda
boost::unordered_map<tuple<int, int, int, multiset<tuple<int, int>>>, tuple<vector<double>, vector<double>> > observation_map;      //a boost::unordered_map to store predictive distributions of observation
//boost::unordered_map<tuple<int, int, int, multiset<tuple<int, int>>>, double> v_map;                                              //a boost::unordered_map to store calculated value of the V() function


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
//  int fullObs_cumulativeTime: cumulative number of periods with full observation
//  int fullObs_cumulativeQuantity: cumulative demand quantity that was fully observed
//  multiset<tuple<int, int>> censoredObservations: the set of censored observations, each observation consists of the inventory level and the last in-stock checkpoint

//Probability of observing stockout after checkpoint $m$ when the initial inventory is $x$ and demand is Poisson($lambda$)
double ProbOfm(const int& m, const int& x, const double& lambda)
{
    double out=0;
    double lambda1 = lambda*tau[m];
    double lambda2 = lambda*(tau[m+1]-tau[m]);
    
    if (m==0)
        out = 1 - Poisson_CDF(x-1, lambda2);
    else
        for (int d1=0; d1<x; ++d1)
            out += Poisson(d1, lambda1)*(1-Poisson_CDF(x-d1-1, lambda2));
    
    return out;
}


//calculate the likelihood of observing all the historical censored observations ($censoredObservations$) with a given lambda
double Likelihood(const multiset<tuple<int, int>> & censoredObservations, const double& lambda)
{
    double out=1;
    
    //for each censored observation $co$ in the set $censoredObservations$, calculate its probability by calling ProbOfm()
    for (tuple<int, int> co : censoredObservations)
    {
        int x = get<0>(co);
        int m = get<1>(co);
        
        out *= ProbOfm(m, x, lambda);
    }
    
    return out;
}



//brute-force Bayesian updating of the pdf of lambda based on historical observations
tuple<double, double, vector<double>> lambda_pdf_update(const int&  fullObs_cumulativeTime, const int&  fullObs_cumulativeQuantity, const multiset<tuple<int, int>> & censoredObservations)
{
    //initialize the output tuple
    tuple<double, double, vector<double>> lambda_tuple;
    
    //first try to search for existing $lambda$ in $lambda_map$, based on the key $allObservations$
    auto allObservations = make_tuple(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
    
    bool found = false;
    #pragma omp critical (lambda_map)     //ALL READ/WRITE ACCESSES TO std::map ARE PROTECTED BY "OMP CRITICAL" TO ENSURE THREAD SAFETY
    {
        auto  it = lambda_map.find(allObservations);
        
        //load the pdf of lambda if available in lambda_map
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
        double beta_n = beta0 + fullObs_cumulativeTime;
        
        
        double lambda_mu = alpha_n/beta_n;
        double lambda_stdev = sqrt(alpha_n)/beta_n;
        double lambda_up = lambda_mu + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
        double lambda_low = max(0.0, lambda_mu - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
        double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;
        
        //initialize an array for storing Bayesian kernel of lambda
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
        
        //calculate the Bayesian posterior for lambda
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



//calculate the predictive distributions of current period observation
tuple<vector<double>, vector<double>> observation_pdf_update(const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<tuple<int, int>> & censoredObservations)
{
    //initialize the output vector
    tuple<vector<double>, vector<double>> observation_pdf;
    
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
        
        //initialize the probability vectors to be saved in $observation_pdf$
        vector<double> prob_m (M); //probability of observation stockout after the $m$-th checkpoint
        vector<double> prob_d (x); //probability of no stockout and observing demand $x$
        
        //initialize predictive probability distributions of different kind of observations, with given prior on Lambda ~ Gamma(alpha_n, beta_n)
        //1. m=M, there is an exact observation, so the predictive just updates to NegBin(alpha_n, 1/(1+beta_n))
        //2. 0<=m<M, the probability of observing stockout after the m-th checkpoint
        
        if (censoredObservations.empty())
        {
            //update alpha,beta based on the exact observations
            double alpha_n = alpha0 + fullObs_cumulativeQuantity;
            double beta_n = beta0 + fullObs_cumulativeTime;
            
            double lambda_mu = alpha_n/beta_n;
            double lambda_stdev = sqrt(alpha_n)/beta_n;
            double lambda_up = lambda_mu + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
            double lambda_low = max(0.0, lambda_mu - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
            double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;
            
            
            //for 1. m=M, nothing to initialize.
            double p_n = 1/(1+beta_n);
            for (int d=0; d<x; ++d)
                prob_d[d] = NegBinomial(d, alpha_n, p_n);
            
            
            //for 2. 0<m<M
            //if stockout happen after checkpoint m, we observe: (1) demand D(m) < x and (2) D(m+1) >= x
            for (int m=1; m<M; ++m)
            {
                double intg=0;
                
                //#pragma omp parallel for schedule(static) reduction(+:intg)
                for (int i=0; i<LAMBDA_STEP; ++i)
                {
                    double lambda = lambda_low + (i+0.5) * delta_lambda;
                    intg += ProbOfm(m, x, lambda) * Gamma(lambda, alpha_n, beta_n);
                }
                intg *= delta_lambda;
                
                prob_m[m] = intg;
            }
            
            //for 3. m=0
            double p_nm = 1/(1+beta_n/tau[1]);
            prob_m[0] = 1;
            for (int d_1=0; d_1<x; ++d_1)
                prob_m[0] -=  NegBinomial(d_1, alpha_n, p_nm);

            ///////////////////////////////////////
            /*
             //test probability distributions
             double sum=0;
             
             for (int m=0; m<M; ++m)
             sum +=  prob_m[m];
             
             for (int d=0; d<x; ++d)
             sum += prob_d[d];
             
             
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
            //2. 0<=m<M, the probability of observing stockout after the m-th checkpoint
            
            //for 1. m=M
            for (int d=0; d<x; ++d)
            {
                double intg = 0;
                
                //#pragma omp parallel for schedule(static) reduction(+:intg)
                for (int i=0; i<LAMBDA_STEP; ++i)
                {
                    intg += Poisson(d, lambda_low + (i+0.5) * delta_lambda) * lambda_pdf[i];
                }
                intg *= delta_lambda;
                
                prob_d[d] = intg;
            }
            
            //for 2. 0<=m<M
            for (int m=0; m<M; ++m)
            {
                double intg=0;
                
                //#pragma omp parallel for schedule(static) reduction(+:intg)
                for (int i=0; i<LAMBDA_STEP; ++i)
                {
                    double lambda = lambda_low + (i+0.5) * delta_lambda;
                    intg += ProbOfm(m, x, lambda) * lambda_pdf[i];
                }
                intg *= delta_lambda;
                
                prob_m[m] = intg;
            }
            
        }
        
        
        observation_pdf = make_tuple(prob_m, prob_d);
        
        //save the newly calculated pdf into observation_map
        #pragma omp critical (observation_map)
        {observation_map.emplace(allObservations, observation_pdf);}
        
    }
    
    return observation_pdf;
}



//***********************************************************
//first-order difference of L := L(x+1) - L(x), for both cases with and without censoring
double L_prime(const int& x, const int& n, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<tuple<int, int>> & censoredObservations)
{
    
    double Phi_x = 0; //$Phi_x$ is for Prob{d <= x} with the given updated belief on lambda
    
    if (censoredObservations.empty())
    {
        //without censoring, lambda ~ Gamma(alpha_n, beta_n), and d ~ NegBin(alpha_n, p)
        //update alpha,beta based on the exact observations
        double alpha_n = alpha0 + fullObs_cumulativeQuantity;
        double beta_n = beta0 + fullObs_cumulativeTime;
        double p_n = 1/(1+beta_n);
        
        for (int d=0; d<=x; ++d)
            Phi_x += NegBinomial(d, alpha_n, p_n);
        
    } else {
        
        //load/update pdf of lambda based on historical observations
        auto lambda_tuple = lambda_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        //read lambda_low, delta_lambda, and the pdf vector from the $lambda$ tuple
        double lambda_low = get<0>(lambda_tuple);
        double delta_lambda = get<1>(lambda_tuple);
        vector<double> lambda_pdf = get<2>(lambda_tuple);
        
        
        //with censored observations, lambda ~ $lambda_pdf[]$, d|lambda ~ Poisson(lambda)
        //#pragma omp parallel for collapse(2) schedule(static) reduction(+:Phi_x)
        for (int d=0; d<=x; ++d)
            for (int i=0; i<LAMBDA_STEP; ++i)
            {
                Phi_x += Poisson(d, lambda_low + (i+0.5) * delta_lambda) * lambda_pdf[i];
            }
        
        Phi_x *= delta_lambda;
        
    }

    Phi_x = min(1.0, Phi_x); //prob should not go beyond 1

    return price * (1 - Phi_x) - cost;
    
}


//search for myopic inventory level with updated knowledge about lambda
int find_x_myopic(const int& n, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity, const multiset<tuple<int, int>> & censoredObservations)
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
        
        double temp = L_prime(x, n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        if (temp > 0)
            x_low = x+1;
        else
            x_up = x;
    }
    
    for (x=x_low; x<=x_up; ++x)
    {
        double temp = L_prime(x, n, fullObs_cumulativeTime, fullObs_cumulativeQuantity, censoredObservations);
        
        if (temp < 0)
            break;
    }
    
    
    return x;
    
}



void archiveVMap(const boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, double> & vMap, const string& ofile)
{
    ofstream ofs(ofile); //open the temporary archive file named $archiveFile$
    if (ofs.good())
    {
        boost::archive::text_oarchive oa(ofs); //initialize boost::archive::text_oarchive
        oa << vMap; //serialize v_map and save it to file
    }
    ofs.close();
}


void importVMap(boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, double> & vMap, const string& ifile)
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


void generateJobsToList(const int& n, const vector<tuple<int, int, multiset<tuple<int, int>>>> & currentJobList, const boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, double> & currentVMap, vector<tuple<int, int, multiset<tuple<int, int>>>> & nextJobList)
{
    set<tuple<int, int, multiset<tuple<int, int>>>> nextJobSet;
    
    for (auto ii = currentJobList.begin(); ii != currentJobList.end(); ++ii)
    {
        int tt = get<0>(*ii);
        int qq = get<1>(*ii);
        multiset<tuple<int, int>> co = get<2>(*ii);
        
        
        if (currentVMap.count(make_tuple(tt, qq, co))==0)
        {
            
            int xx = find_x_myopic(n, tt, qq, co);
            
            if (xx==0)
                nextJobSet.insert(make_tuple(tt, qq, co));
            else
            {
                //case 2
                for (int m=0; m<M; ++m)
                {
                    multiset<tuple<int, int>> co_new = co;
                    co_new.insert(make_tuple(xx, m));
                    nextJobSet.insert(make_tuple(tt, qq, co_new));
                }
                
                //case 1
                for (int d=0; d<xx; ++d)
                    nextJobSet.insert(make_tuple(tt + 1, qq + d, co));
            }
		}
	}
    
    nextJobList.resize(nextJobSet.size());
    copy(nextJobSet.begin(), nextJobSet.end(), nextJobList.begin());
    
}



void solveJobsInList(const int& n, const vector<tuple<int, int, multiset<tuple<int, int>>>> & currentJobList, const boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, double> & nextVMap, boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, double> & currentVMap)
{
    //cout << "Solving " << currentJobList.size() << " jobs for period " << n << endl;
    auto lastMapSize = currentVMap.size();
    
    #pragma omp parallel for schedule(dynamic)
    for (int ii=0; ii<currentJobList.size(); ++ii)
    {
        
        int tt = get<0>(currentJobList[ii]);
        int qq = get<1>(currentJobList[ii]);
        multiset<tuple<int, int>> co = get<2>(currentJobList[ii]);
        
        auto parameters = make_tuple(tt, qq, co);
        
		//DEBUG
        //cout << n << " " << tt << " " << qq << " " << co.size() << " " ;
        
        int count;
        #pragma omp critical (v_map)
        {count = currentVMap.count(parameters);}
        
        //cout << count << " " ;
        
        if (count==0)
        {
            //start calculate expected profit-to-go
            int xx = find_x_myopic(n, tt, qq, co);
            double out1 = 0;
            double vv = 0;
            
            //cout << xx << " ";
            
            if (xx==0)
            {
                if (n<N) out1 = nextVMap.at(parameters);
            }
            else
            {
                //load the predictive pdf of current period observation
                auto observation_pdf = observation_pdf_update(xx, tt, qq, co);
                auto prob_m = get<0>(observation_pdf);
                auto prob_d = get<1>(observation_pdf);
                
                
                //Case I.2
                for (int m=0; m<M; ++m)
                {
                    if (n<N) {
                        multiset<tuple<int, int>> co_new = co;
                        co_new.insert(make_tuple(xx, m));
                        vv = nextVMap.at(make_tuple(tt, qq, co_new));
                    }
                    
                    out1 += ( price*xx + vv) * prob_m[m];
                }
                
                
                //Case I.1: when there is no stockout in the current period...
                for (int d=0; d<xx; ++d)
                {
                    if (n<N) vv = nextVMap.at(make_tuple(tt+1, qq+d, co));
                    out1 += ( price*d + vv ) * prob_d[d];
                }
                
                out1 = out1 - cost*xx;
            }
            
            
            #pragma omp critical (v_map)
            {currentVMap.emplace(parameters, out1);}
            
            //DEBUG
            //cout << ii << "/" << currentJobList.size() << " " << out1 << endl;
        }
        
        
        
        //save the whole v_map into $archiveFile$ every hour
        if (omp_get_thread_num() == 0) //#pragma omp master
        {
            auto currentTime = chrono::system_clock::now();  //get current time
            
            if (chrono::duration_cast<chrono::hours> (currentTime-lastTime).count() >= 1)
            {
                int mapSize = currentVMap.size();
                if (mapSize > lastMapSize)
                {
                    
                    lastTime = currentTime; //update time of last archive
                    lastMapSize = mapSize; //update last archive size
                    
                    
                    //#pragma omp critical (archive)
                    {archiveVMap(currentVMap, archiveFile[n]);}
                    cout << "[INFO:] ARCHIEVED V_map(" << n << ") " << mapSize << " records to " << archiveFile[n] << " at Hour " << chrono::duration_cast<chrono::hours> (currentTime-startTime).count() << "." << endl;
                }
            }
        }
        
    }
    
    
    //archive again upon finishing
    if (currentVMap.size() > lastMapSize)
    {
        archiveVMap(currentVMap, archiveFile[n]);
        cout << "[INFO:] ARCHIEVED V_map(" << n << ") " << currentVMap.size() << " records to " << archiveFile[n] << " at the end." << endl;
    }
    
}







int main(int ac, char* av[])
{
    //read and parse command line inputs (using boose::program_options)
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    //scenario parameters
    ("myopic", po::value<int>(&myopic)->default_value(1), "adopt myopic inventory policy?")
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
    modelName = "NVLearning_CheckpointB";
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
	
    
    file << setprecision(8);
    cout << setprecision(8);
    
    
    omp_set_num_threads(omp_get_num_procs());
	
    cout << "Num of Procs: " << omp_get_num_procs() << endl;
    cout << "Max Num of Threads: " << omp_get_max_threads() << endl;
    cout << "Num of periods (N): " << N << endl;
    cout << "Max Num of checkpoints (M): " << M_MAX << endl;
    if (myopic==0)
        cout << "r\tc\talpha\tbeta\tM\tQ_B\tPi_B\tTime_B\tCPUTime_B\tComp_B" << endl;
    else
        cout << "r\tc\talpha\tbeta\tM\tQ_Bm\tPi_Bm\tTime_Bm\tCPUTime_Bm\tComp_Bm" << endl;
    
    
    //file << "Num of Procs: " << omp_get_num_procs() << endl;
    //file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    //file << "Num of periods (N): " << N << endl;
    //file << "Max Num of checkpoints (M): " << M_MAX << endl;
    if (myopic==0)
        file << "r\tc\talpha\tbeta\tM\tQ_B\tPi_B\tTime_B\tCPUTime_B\tComp_B" << endl;
    else
        file << "r\tc\talpha\tbeta\tM\tQ_Bm\tPi_Bm\tTime_Bm\tCPUTime_Bm\tComp_Bm" << endl;
    
    
    
    //initialize cost parameters
    price = 2;
    //lambda_mean = 10;
    //beta0 = 1;
    //M = 2;
    //cost = 1;
    
    //setup initial observations
    multiset<tuple<int, int>> initialObservations;
    

    
    //for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
    //for (beta0=2; beta0>=0.05; beta0/=2)
    {
        alpha0 = beta0*lambda_mean;
        
        
        //for (M=2; M<=M_MAX; M*=2)
        {
            
            //initialize the timepoint of the M checkpoints, assuming they are evenly spaced on the interval [0, 1]
            tau.clear();
            tau.resize(M+1);
            for (int i=0; i<=M; i++)
                tau[i] = double(i) / M;

            for (cost=1.8; cost>=0.15; cost-=0.1)
            {
				//previously saved distributions of lambda needed to be cleared
                lambda_map.clear();
                observation_map.clear();
                archiveFile.clear();
                
                cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t" << M << "\t";
                file << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t" << M << "\t";
                
                
                
                //try load archived data into v_map
                scenarioName = ".l" + dbl_to_str(lambda_mean) +".b" + dbl_to_str(beta0) + ".M" + to_string(M) + ".c" + dbl_to_str(cost);
                for (int n=0; n<=N; ++n)
                    archiveFile.emplace_back(path + modelName + scenarioName + ".n" + to_string(n) + ".oarchive.txt");
                
				boost::unordered_map<tuple<int, int, multiset<tuple<int, int>>>, double> vMap1, vMap2, vMap3, vMap4;
                vector<tuple<int, int, multiset<tuple<int, int>>>> jobList1, jobList2, jobList3, jobList4;
                
                importVMap(vMap4, archiveFile[4]);
                importVMap(vMap3, archiveFile[3]);
                importVMap(vMap2, archiveFile[2]);
                importVMap(vMap1, archiveFile[1]);
                
                jobList1.emplace_back(0, 0, initialObservations);
                
                
                startTime = chrono::system_clock::now();
                lastTime = startTime;
                clock_t cpu_start = clock();
                
                generateJobsToList(1, jobList1, vMap1, jobList2);
                generateJobsToList(2, jobList2, vMap2, jobList3);
                generateJobsToList(3, jobList3, vMap3, jobList4);
                
                solveJobsInList(4, jobList4, vMap1, vMap4);
                solveJobsInList(3, jobList3, vMap4, vMap3);
                solveJobsInList(2, jobList2, vMap3, vMap2);
                solveJobsInList(1, jobList1, vMap2, vMap1);
                
                int x_myopic = find_x_myopic(1,0,0,initialObservations);
                double v_max = vMap1.at(make_tuple(0, 0, initialObservations));
                cout << x_myopic << "\t" << v_max << "\t";
                file  << x_myopic << "\t" << v_max  << "\t";
                
                
                clock_t cpu_end = clock();
                endTime = chrono::system_clock::now();
                
                
                
                cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
                file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
                
                auto vMapSize = vMap1.size() + vMap2.size() + vMap3.size() + vMap4.size();
                
                cout << vMapSize << endl;
                file << vMapSize << endl;
                
                
            }
            
        }
        
    }
    
    
    file.close();
    
    return 0;
}
