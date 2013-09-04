//============================================================================
// Name        : NVLearning_Full.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v8.0 (2013-09-03)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Full-observation Model
//============================================================================

//***********************************************************

#include <iostream>
#include <fstream>
#include <iomanip> //required by setprecision()

#include <cmath>
#include <numeric>

#include <chrono>
#include <ctime>

#include <tuple>
#include <boost/unordered_map.hpp>


#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "serialize_tuple.h"
#include "unordered_map_serialization.h"


//#include <omp.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;


using namespace std;

//***********************************************************

#define N 4                                         //number of periods

#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 10  //upper bound for D

#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10             //upper bound for X

//***********************************************************

double price, cost;                             //Newsvendor price and cost parameters
double alpha0, beta0;                           //Initial prior of lambda is Gamma(alpha0, beta0)
double lambda_mean;                                //Mean of lambda = alpha0/beta0;

ofstream file;                                  //output files

string path, modelName, scenarioName;               //model name used in naming archive files
string resultFile, archiveFile;                     //output and archive file names


boost::unordered_map<tuple<int, int, int>, double> v_map;        //an std::map to store calculated value of the V() function


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


double NegBinomial(const int& kk, const double& rr, const double& pp)
{
    double log_pmf = 0;
    
    if ((kk>=0)&&(rr>0)&&(pp>=0)&&(pp<=1))
        log_pmf = lgamma(kk+rr) - lgamma(kk+1) - lgamma(rr)  + rr*log(1-pp) + kk*log(pp);
    
    return exp(log_pmf);
}


//***********************************************************

//first-order difference of L := L(x+1) - L(x)
double L_prime(const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    
    double Phi_x = 0; //$Phi_x$ is for Prob{d <= x} with the given updated belief on lambda
    
    //lambda ~ Gamma(alpha_n, beta_n), and d ~ NegBin(alpha_n, p)
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + fullObs_cumulativeTime;
    double p_n = 1/(1+beta_n);
    
    
    for (int d=0; d<=x; ++d)
        Phi_x += NegBinomial(d, alpha_n, p_n);
    
    Phi_x = min(1.0, Phi_x); //prob should not go beyond 1
	
    return price * (1 - Phi_x) - cost;
    
}


//search for myopic inventory level with updated knowledge about lambda
int find_x_myopic(const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
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
    
    for (x=x_low; x<=x_up; ++x)
    {
        double temp = L_prime(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        
        if (temp < 0)
            break;
    }
    
    return x;
    
}


void archiveVMap(const boost::unordered_map<tuple<int, int, int>, double> & vMap, const string& ofile)
{
    ofstream ofs(ofile); //open the temporary archive file named $archiveFile$
    if (ofs.good())
    {
        boost::archive::text_oarchive oa(ofs); //initialize boost::archive::text_oarchive
        oa << vMap; //serialize v_map and save it to file
    }
    ofs.close();
}

//***********************************************************


//============================================
//Dynamic Program recursion
//============================================
//Parameters for function G() and V():
//int n: current period index
//int x: current inventory level

double G_F(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity);

double V_F(const int& n, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    double v_max;
    
    if (n==N+1)
        v_max = 0;    //V_{N+1}()=0
    else
    {
        auto parameters = make_tuple(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        
        bool found = false;
        //#pragma omp critical (v_map)
        {
            auto  it = v_map.find(parameters);
            if (it != v_map.end()) {
                found = true;
                v_max = it->second;
            }
        }
        
        
        if (!found)
        {
            
            //no need to search for optimal inventory level x, the myopic inventory level is optimal
            int x_opt = find_x_myopic(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            
            v_max = G_F(n, x_opt, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            
            
            if (n==1)
            {
                cout << x_opt << "\t" << v_max  << "\t";
                file << x_opt << "\t" << v_max  << "\t";
            }
            
            
            //#pragma omp critical (v_map)
            { v_map.emplace(parameters, v_max); }
            
        }
        
    }
    
    return v_max;
}



double G_F(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    
    double r = alpha0 + fullObs_cumulativeQuantity;
    double p = 1/(1 + beta0 + fullObs_cumulativeTime);
    
    double d_mean = r*p/(1-p);
    double d_var = r*p/pow(1-p,2.0);
    int d_up = d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*sqrt(d_var);  //upper bound of D is set to be equal to $mean + N*var$
    
    //start calculate expected profit-to-go
    double out1 = 0;
    
    //#pragma omp parallel for schedule(dynamic) reduction(+:out1)
    for (int d=0; d<=d_up; ++d)
        out1 += ( price * min(d,x) + V_F(n+1, fullObs_cumulativeTime + 1, fullObs_cumulativeQuantity + d) ) * NegBinomial(d, r, p);
    
    
    return out1 - cost*x;
    
}




int main(int ac, char* av[])
{
    //read and parse command line inputs (using boose::program_options)
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    //scenario parameters
    ("lambda,l", po::value<double>(&lambda_mean)->default_value(10), "mean demand (E[lambda])")
    ("beta,b", po::value<double>(&beta0)->default_value(1), "beta")
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
    modelName = "NVLearning_Full";
    if (resultFile == "") resultFile = modelName + ".result.txt";
    
    
    //Open output file
    file.open(resultFile, fstream::app|fstream::out);
    
    if (! file)
    {
        //if fail to open the file
        cerr << "can't open output file NVLearning_Full_result.txt!" << endl;
        exit(EXIT_FAILURE);
    }
	
    file << setprecision(10);
    cout << setprecision(10);
    
    //omp_set_num_threads(omp_get_num_procs());
	
    //cout << "Num of Procs: " << omp_get_num_procs() << endl;
    //cout << "Max Num of Threads: " << omp_get_max_threads() << endl;
    cout << "Num of periods (N): " << N << endl;
    cout << "r\tc\talpha\tbeta\tQ_F\tPi_F\tTime_F\tCPUTime_F\tComp_F" << endl;
    
    //file << "Num of Procs: " << omp_get_num_procs() << endl;
    //file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    //file << "Num of periods (N): " << N << endl;
    file << "r\tc\talpha\tbeta\tQ_F\tPi_F\tTime_F\tCPUTime_F\tComp_F" << endl;
    
    
    
    //initialize cost parameters
    price = 2;
    //lambda_mean = 10;
    //cost = 1;
    //beta0 = 1;
    
    
    for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
    for (beta0=2; beta0>=0.05; beta0/=2)
    {
        alpha0 = beta0*lambda_mean;
        
        for (cost=1.8; cost>=0.15; cost-=0.1)
        {
            v_map.clear();
            
            
            cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t";
            file << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t";
            
            
            auto startTime = chrono::system_clock::now();
            clock_t cpu_start = clock();
            
            V_F(1, 0, 0);
            
            clock_t cpu_end = clock();
            auto endTime = chrono::system_clock::now();
            
            cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
            file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
            
            
            scenarioName = ".l" + dbl_to_str(lambda_mean) +".b" + dbl_to_str(beta0) + ".c" + dbl_to_str(cost);
            archiveFile = path + modelName + scenarioName + ".oarchive.txt";
            
            archiveVMap(v_map, archiveFile);
            auto mapSize = v_map.size();
            cout << mapSize+1 << endl;
            file << mapSize+1 << endl;
        }
        
    }

    
    
    
    file.close();
    
    return 0;
}

