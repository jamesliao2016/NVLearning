//============================================================================
// Name        : NVLearning_Timing.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v8.0 (2013-09-03)
// Copyright   : ...
// Description : general code for newsvendor with censored demand --- the Stock-out Timing Model
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

#include <omp.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

//***********************************************************

#define N 4                                         //number of periods

#define T_STEP 1000                                 //discretize the [0,1] interval into 1000 segments

#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 10  //upper bound for D

//***********************************************************

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

//boost::unordered_map<tuple<int, int>, vector<double> > demand_map;             //a boost::unordered_map to store predictive distributions of demand
//boost::unordered_map<tuple<int, int, int>, vector<double> > observation_map;        //a boost::unordered_map to store predictive distributions of timing observations
boost::unordered_map<tuple<int, int, int>, double> v_map;                           //a boost::unordered_map to store calculated value of the V() function


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


double InvBeta2(const double& tt, const int& xx, const double& aa, const double& bb)
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
//Bayesian updating implementation
//Key variables:
//  int fullObs_cumulativeTime: cumulative number of periods with effective observation
//  int fullObs_cumulativeQuantity: cumulative demand quantity that was effectively observed (in terms of the number of sub-periods, for the sake of discreteness)

//calculate the predictive distributions of current period demand
vector<double> demand_pdf_update(const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + fullObs_cumulativeQuantity;
    double beta_n = beta0 + double(fullObs_cumulativeTime)/T_STEP;
    double p_n = 1/(1+beta_n);
    
    double d_mean = alpha_n*p_n/(1-p_n);
    double d_var = alpha_n*p_n/pow(1-p_n,2.0);
    int d_up = d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*sqrt(d_var);  //upper bound of D is set to be equal to $mean + N*var$
    
    //initialize the output vector
    vector<double> demand_pdf (d_up+1);
    
//    //first try to search for existing $demand_pdf$ in $demand_map$, based on the key $allObservations$
//    auto allObservations = make_tuple(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
//    
//    bool found = false;
//#pragma omp critical (demand_map)
//    {
//        auto  it = demand_map.find(allObservations);
//        if (it != demand_map.end())
//        {
//            found = true;
//            demand_pdf = it->second;
//        }
//    }
//    
//    if (!found)
//    {
        for (int d=0; d<d_up; ++d)
            demand_pdf[d] = NegBinomial(d, alpha_n, p_n);
        
//        
//        //save the newly calculated pdf into demand_map
//#pragma omp critical (demand_map)
//        {demand_map.emplace(allObservations, demand_pdf);}
//        
//        
//    }
//    
    return demand_pdf;
    
}

//calculate the predictive distributions of current period timing observation
vector<double> observation_pdf_update(const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    //initialize the output vector
    vector<double> observation_pdf (T_STEP/2);
    
//    //first try to search for existing $observation_pdf$ in $observations_map$, based on the key $allObservations$
//    auto allObservations = make_tuple(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
//    
//    bool found = false;
//#pragma omp critical (observation_map)
//    {
//        auto  it = observation_map.find(allObservations);
//        if (it != observation_map.end())
//        {
//            found = true;
//            observation_pdf = it->second;
//        }
//    }
//    
//    if (!found)
//    {
//        
        //initialize predictive probability distributions of timing observations, with given prior on Lambda ~ Gamma(alpha_n, beta_n)
        
        //update alpha,beta based on the effective observations
        double alpha_n = alpha0 + fullObs_cumulativeQuantity;
        double beta_n = beta0 + double(fullObs_cumulativeTime)/T_STEP;
        
        //the probability of observing stockout at time $t$ ($t$ is in [0,1], which is discretized into T_STEP points)
        for (int t=0; t<T_STEP/2; ++t)
            observation_pdf[t] = InvBeta2((2.0*t+1)/T_STEP, x, alpha_n, beta_n);
        
//        
//        
//        //save the newly calculated pdf into observation_map
//#pragma omp critical (observation_map)
//        {observation_map.emplace(allObservations, observation_pdf);}
//        
//    }
//    
    
    return observation_pdf;
    
}


//search for myopic inventory level with updated distribution
int find_x_myopic(const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    //load the predictive pdf of current period demand
    vector<double> demand_pdf = demand_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
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


void importVMap(boost::unordered_map<tuple<int, int, int>, double> & vMap, const string& ifile)
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


//***********************************************************


//============================================
//Dynamic Program recursion
//============================================
//Parameters for function G() and V():
//int n: current period index
//int x: current inventory level

double G_T(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity);
double G_T_UpperBound(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity);


double V_T(const int& n, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    double v_max;
    
    if (n==N+1)
        v_max = 0;    //V_{N+1}()=0
    else
    {
        auto parameters = make_tuple(n, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        
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
            int x_myopic = find_x_myopic(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            int x_opt = x_myopic;
            int x = x_myopic;
            
            //evaluate low bound x_low
            v_max = G_T(n, x_myopic, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
            
            if (myopic == 0)
            {
                //linear search from x_low onwards, until reaches the upper bound of x
                for (x=x_myopic+1; ; ++x)
                {
                    double temp = G_T(n, x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
                    
                    if (temp > v_max)
                    {
                        x_opt = x;
                        v_max = temp;
                    }
                    //else //if assume concavity, searching can stop once the first order condition is met, without checking the bound
                    if (temp >= G_T_UpperBound(n, x+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity))
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
            
            
            if (n==1)
            {
                cout << x << "\t" << x_opt << "\t" << v_max  << "\t";
                file << x << "\t" << x_opt << "\t" << v_max  << "\t";
            }
            
            
        }
        
    }
    
    return v_max;
}



double G_T(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    
    //if x==0, jump to next period without learning or updating the system state (because without keeping inventory, there will be no observation, no cost, and no revenue anyhow)
    if (x==0)
        return V_T(n+1, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
    else
    {
        //load the predictive pdf of current period demand and timing observations
        vector<double> demand_pdf = demand_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        vector<double> observation_pdf = observation_pdf_update(x, fullObs_cumulativeTime, fullObs_cumulativeQuantity);
        
        
        //start calculate expected profit-to-go
        
        //case 1: no stockout happening
        double out1 = 0;
        
        //#pragma omp parallel for schedule(dynamic) reduction(+:out1)
        for (int d=0; d<x; ++d)
        {
            out1 += ( price * d + V_T(n+1, fullObs_cumulativeTime + T_STEP, fullObs_cumulativeQuantity + d) ) * demand_pdf[d];
        }
        
        //case 2: stockout at sometime before the end of the period
        double out2 = 0;
        
        //take stepsize=2 to speed up the calculation of the integral
#pragma omp parallel for schedule(dynamic) reduction(+:out2)
        for (int i=1; i<=T_STEP; i+=2)
        {
            out2 += ( price * x + V_T(n+1, fullObs_cumulativeTime + i, fullObs_cumulativeQuantity + x) ) * observation_pdf[i/2]; //integral at ticks 1, 3, 5, ..., 999
        }
        out2 *= 2.0/T_STEP; //dt = 2/T_STEP
        
        
        return out1 + out2 - cost*x;
    }
}


double G_T_UpperBound(const int& n, const int& x, const int& fullObs_cumulativeTime, const int& fullObs_cumulativeQuantity)
{
    
    //load the predictive pdf of current period demand
    vector<double> demand_pdf = demand_pdf_update(fullObs_cumulativeTime, fullObs_cumulativeQuantity);
    int d_up = demand_pdf.size() - 1;
    
    //start calculate expected profit-to-go
    double out1 = 0;
    
    for (int d=0; d<=d_up; ++d)
        out1 += ( price * min(d, x) + V_T(n+1, fullObs_cumulativeTime + T_STEP, fullObs_cumulativeQuantity + d) ) * demand_pdf[d];
    
    
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
    modelName = "NVLearning_Timing";
    if (myopic != 0) modelName = modelName + "_myopic";
    if (resultFile == "") resultFile = modelName + ".result.txt";
    
    
    //Open output file
    file.open(resultFile, fstream::app|fstream::out);
    
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
    if (myopic==0)
        cout << "r\tc\talpha\tbeta\tQ_T_bar\tQ_T\tPi_T\tTime_T\tCPUTime_T\tComp_T" << endl;
    else
        cout << "r\tc\talpha\tbeta\tQ_Tm_bar\tQ_Tm\tPi_Tm\tTime_Tm\tCPUTime_Tm\tComp_Tm" << endl;
    
    
    //file << "Num of Procs: " << omp_get_num_procs() << endl;
    //file << "Max Num of Threads: " << omp_get_max_threads() << endl;
    //file << "Num of periods (N): " << N << endl;
    if (myopic==0)
        file << "r\tc\talpha\tbeta\tQ_T_bar\tQ_T\tPi_T\tTime_T\tCPUTime_T\tComp_T" << endl;
    else
        file << "r\tc\talpha\tbeta\tQ_Tm_bar\tQ_Tm\tPi_Tm\tTime_Tm\tCPUTime_Tm\tComp_Tm" << endl;
    
    
    
    
    //initialize cost parameters
    price = 2;
    //lambda_mean = 10;
    //beta0 = 1;
    //cost = 1;
    
    
    
    //for (lambda_mean=10; lambda_mean<=50; lambda_mean+=10)
    for (beta0=2; beta0>=0.05; beta0/=2)
    {
        alpha0 = beta0*lambda_mean;
        
        for (cost=1.8; cost>=0.15; cost-=0.1)
        {
            //demand_map.clear();
            //observation_map.clear();
            v_map.clear();
            
            cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t";
            file << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << "\t";
            
            //try load archived data into v_map
            scenarioName = ".l" + dbl_to_str(lambda_mean) +".b" + dbl_to_str(beta0) + ".c" + dbl_to_str(cost);
            archiveFile = path + modelName + scenarioName + ".oarchive.txt";
            
            importVMap(v_map, archiveFile);
            lastMapSize = v_map.size();
            
            
            //start solving DP recursively
            startTime = chrono::system_clock::now();
            lastTime = startTime;
            clock_t cpu_start = clock();
            
            V_T(1, 0, 0);
            
            clock_t cpu_end = clock();
            endTime = chrono::system_clock::now();
            
            cout << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
            file << chrono::duration_cast<chrono::milliseconds> (endTime-startTime).count() << "\t" << 1000.0*(cpu_end-cpu_start)/CLOCKS_PER_SEC << "\t";
            
            
            //archive again upon finishing
            int mapSize = v_map.size();
            if (mapSize > lastMapSize)
                archiveVMap(v_map, archiveFile);
            cout << mapSize+1 << endl;
            file << mapSize+1 << endl;
            
        }
        
    }
    
    
    file.close();
    
    return 0;
}

