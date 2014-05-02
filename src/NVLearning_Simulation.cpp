//============================================================================
// Name        : NVLearning_Simulation.cpp
// Author      : Tong WANG
// Email       : tong.wang@nus.edu.sg
// Version     : v1.0 (2014-04-16)
// Copyright   : ...
// Description : simulation for comparing models:
//                  P Model with known demand distribution
//                  F Model with unknown distribution and full demand observations
//                  T-M Model with unknown distribution, stock-out timing observations, and myopic inventory decision
//                  E-M Model with unknown distribution, stock-out event observations, and myopic inventory decision
//============================================================================

//***********************************************************

#include <iostream>
#include <fstream>
#include <iomanip> //required by setprecision()

#include <cmath>
#include <numeric>
#include <random>

#include <chrono>
#include <ctime>

#include <vector>
#include <set>


#include <omp.h>


using namespace std;

//***********************************************************

#define N 101                                                        //number of periods
#define RUN 1000000                                                  //number of simulation runs

#define D_MAX 1000
#define D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 10                  //upper bound for D

#define LAMBDA_STEP 1000                                            //discretize the continuous distribution of lambda into LAMBDA_STEP=1000 pieces
#define LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 8
#define LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN 4

#define X_MYOPIC_UP_MULTIPLE_OF_MEAN 10                             //upper bound for X

//***********************************************************

double price, cost;                                                 //Newsvendor price and cost parameters
double alpha0, beta0;                                               //Initial prior of lambda is Gamma(alpha0, beta0)

ofstream file;                                                      //output files

//***********************************************************

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

// Newsvendor profit calculation with given inventory and realized demand
double pi(const int& inv, const int& demand) {
    
    return price * min(inv, demand) - cost * inv;
    
}


//optimal inventory in the P (perfect knowledge) model (this is just the standard newsvendor solution with Poisson demand)
int find_yP(const double& lam) {
    
    unsigned int y;
    double pmf = 0;
    double cdf = 0;
    
    for (y=0; ; ++y) {
        pmf = Poisson(y, lam);
        cdf += pmf;
        
        if (cdf >= (price-cost)/price)
            break;
    }
    
    return y;
}



//first-order difference of L := L(x+1) - L(x)
double L_prime(const int& x, const double& cumulativeTime, const int& cumulativeQuantity)
{
    
    double Phi_x = 0; //$Phi_x$ is for Prob{d <= x} with the given updated belief on lambda
    
    //lambda ~ Gamma(alpha_n, beta_n), and d ~ NegBin(alpha_n, p)
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + cumulativeQuantity;
    double beta_n = beta0 + cumulativeTime;
    double p_n = 1/(1+beta_n);
    
    
    for (int d=0; d<=x; ++d)
        Phi_x += NegBinomial(d, alpha_n, p_n);
    
    Phi_x = min(1.0, Phi_x); //prob should not go beyond 1
	
    return price * (1 - Phi_x) - cost;
    
}


//search for myopic inventory level with updated knowledge about lambda (for F and T models)
int find_yF(const double& cumulativeTime, const int& cumulativeQuantity)
{
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + cumulativeQuantity;
    double beta_n = beta0 + cumulativeTime;
    
    //bi-sectional search for x such that L_prime is zero
    
    int x;
    int x_up = (int) X_MYOPIC_UP_MULTIPLE_OF_MEAN * alpha_n / beta_n;
    int x_low = 0;
    
    while (x_up - x_low > 3)
    {
        x = (x_up + x_low)/2;
        
        double temp = L_prime(x, cumulativeTime, cumulativeQuantity);
        
        if (temp > 0)
            x_low = x+1;
        else
            x_up = x;
    }
    
    for (x=x_low; x<=x_up; ++x)
    {
        double temp = L_prime(x, cumulativeTime, cumulativeQuantity);
        
        if (temp < 0)
            break;
    }
    
    return x;
    
}


//***********************************************************
//For the E model: (functions taken from NVLearning-Event code)
//Bayesian updating implementation
//Key variables:
//  int cumulativeTime: cumulative number of periods with full observation
//  int cumulativeQuantity: cumulative demand quantity that was fully observed
//  multiset<int> censoredObservations: the set of censored observations in previous periods, values stored are the intial inventory levels


//calculate the likelihood of observing all the historical censored observations with a given lambda
//Prob(D>=censoredObservations[0]) * Prob(D>=censoredObservations[1]) * ...
double Likelihood(const multiset<int> & censoredObservations, const double& lambda)
{
    double out=1;
    
    //for each censored observation $co$ in the set $censoredObservations$, calculate its probability
    for (int co : censoredObservations)
    {
        //Probability of a $co$: Prob{ D >= co }
        out *= 1-Poisson_CDF(co-1, lambda);
    }
    
    return out;
}


//calculate the predictive distributions of current period demand
vector<double> demand_pdf_update(const int& cumulativeTime, const int& cumulativeQuantity, const multiset<int> & censoredObservations)
{
    //update alpha,beta based on the exact observations
    double alpha_n = alpha0 + cumulativeQuantity;
    double beta_n = beta0 + cumulativeTime;
    double p_n = 1/(1+beta_n);
    
    double d_mean = alpha_n*p_n/(1-p_n);
    double d_var = alpha_n*p_n/pow(1-p_n,2.0);
    int d_up = d_mean + D_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*sqrt(d_var);  //upper bound of D is set to be equal to $mean + N*var$
    
    
    //initialize the output vector
    vector<double> demand_pdf (d_up+1);
    
    //initialize predictive probability distributions of demand, with given posterior on Lambda
    if (censoredObservations.empty())
    {
        for (int d=0; d<=d_up; ++d)
            demand_pdf[d] = NegBinomial(d, alpha_n, p_n);
        
    } else {
        
        //brute-force Bayesian updating of the pdf of lambda based on historical observations
        double lambda_mean = alpha_n/beta_n;
        double lambda_stdev = sqrt(alpha_n)/beta_n;
        double lambda_up = lambda_mean + LAMBDA_UP_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev;
        double lambda_low = max(0.0, lambda_mean - LAMBDA_LOW_NUMBER_OF_STDEV_AWAY_FROM_THE_MEAN*lambda_stdev);
        double delta_lambda = (lambda_up-lambda_low) / LAMBDA_STEP;
        
        //initialize an array for storing the Bayesian kernel of lambda
        vector<double> kernel (LAMBDA_STEP);
        
        double predictive=0;
        
        //calculate the kernel and predictive in Bayesian equation at the same time
        for (int i=0; i<LAMBDA_STEP; ++i)
        {
            double lambda = lambda_low + (i+0.5)*delta_lambda;
            kernel[i] = Likelihood(censoredObservations, lambda) * Gamma(lambda, alpha_n, beta_n); //kernel is equal to likelihood * prior
            
            predictive += kernel[i]; // predictive is obtained by integrating the kernel
        }
        predictive *= delta_lambda;
        
        //calculate the Bayesian posterior for lambda, posterior = kernel/predictive
        vector<double> lambda_pdf(LAMBDA_STEP);
        
        for (int i=0; i<LAMBDA_STEP; ++i)
            lambda_pdf[i] = kernel[i]/predictive;
        
        
        //update demand_pdf
        for (int d=0; d<=d_up; ++d)
        {
            double intg = 0;
            
            for (int i=0; i<LAMBDA_STEP; ++i)
            {
                intg += Poisson(d, lambda_low + (i+0.5) * delta_lambda) * lambda_pdf[i];
            }
            intg *= delta_lambda;
            
            demand_pdf[d] = intg;
        }
        
        
    }
    
    return demand_pdf;
}


//search for myopic inventory level with updated distribution
int find_yE(const int& cumulativeTime, const int& cumulativeQuantity, const multiset<int> & censoredObservations)
{
    //load the predictive pdf of current period demand
    vector<double> demand_pdf = demand_pdf_update(cumulativeTime, cumulativeQuantity, censoredObservations);
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



int main(int ac, char* av[])
{
    
    //initialize cost parameters
    price = 2;
    cost = 1.5;
    
    //initial prior parameters
    alpha0 = 0.625;
    beta0 = 0.0625;
    
    
    
    //Open output file
    file.open("NVLearning-Simulation.15.625.txt", fstream::app|fstream::out);
    
    if (! file)
    {
        //if fail to open the file
        cerr << "can't open output file NVLearning_Full_result.txt!" << endl;
        exit(EXIT_FAILURE);
    }
	
    file << setprecision(10);
    cout << setprecision(10);

    cout << "Price\tCost\talpha0\tbeta0" << endl;
    cout << price << "\t" << cost << "\t" << alpha0 << "\t" << beta0 << endl;
    
    
    
    //======================
    //start simulation
    //======================
    
    
    // Initialize random number generator.
    random_device rd;                                           //defind random device
    knuth_b re(12345);                                          //define a knuth_b random engine with a fixed seed
    gamma_distribution<> rgamma(alpha0, 1/beta0);               //initialize a Gamma(alpha0, beta0) distribution (for simulating lambda)
    
    
    // prepare vectors storing simulation results
    vector<vector<double>> profitP(RUN, vector<double>(N));     // profit[run][period]
    vector<vector<double>> profitF(RUN, vector<double>(N));
    vector<vector<double>> profitT(RUN, vector<double>(N));
    vector<vector<double>> profitE(RUN, vector<double>(N));
    vector<int> inventoryP(RUN);                                // inventoryP[run] --- inventory in P model does not change over time
    vector<vector<int>> inventoryF(RUN, vector<int>(N));        // inventory[run][period]
    vector<vector<int>> inventoryT(RUN, vector<int>(N));
    vector<vector<int>> inventoryE(RUN, vector<int>(N));
    
    
    cout << "Starting simulation runs ..." << endl;
    
    
    //the big RUN loop, r = 0, ..., RUN-1
#pragma omp parallel for schedule(static) shared(profitP, profitF, profitT, profitE, inventoryP, inventoryF, inventoryT, inventoryE)
    for (unsigned int r=0; r<RUN; ++r)
    {
        //in each run, first simulate a lambda_true from the Gamma(alpha0, beta0) distribution
        double lambda_true = rgamma(re);
        
        //initialize an Exp(lambda_true) distribution for simulating inter-arrival times
        exponential_distribution<> rexp(lambda_true);
        
        
        if (r % 100 == 0)
            cout << "Run " << r << " ... (lambda = " << lambda_true << ") " << endl;
        
        
        //observation state of the F, T, and E models
        double cumQuantityF = 0;
        double cumTimeF = 0;
        double cumQuantityT = 0;
        double cumTimeT = 0;
        double cumQuantityE = 0;
        double cumTimeE = 0;
        multiset<int> censoredObservationsE;
        
        
        //solve the newsvendor quantity when lambda_true is known (this remains the same across all the periods, so just keep one copy)
        int yP = find_yP(lambda_true);
        inventoryP[r] = yP;
        
        
        //loop over all the periods n = 0, ..., N-1
        for (unsigned int n=0; n<N; ++n)
        {
            //***********************************************************
            //in each period, first simulate demand arrivals and store arrival times in the vector
            double demandInPeriod;
            vector<double> arrivalsInPeriod(D_MAX);
            arrivalsInPeriod[0] = 0;
            
            
            //simulate arrivals one by one
            for (unsigned int i=1;; ++i) {
                
                double tau = rexp(re);  //inter-arrival time follows Exp(lambda_true)
                arrivalsInPeriod[i] = arrivalsInPeriod[i-1] + tau;
                
                
                if (arrivalsInPeriod[i] > 1) //stop simulation once time reaches 1
                {
                    // demand is the number of occurrances within time interval (0, 1]
                    demandInPeriod = i-1;
                    break;
                }
            }
            
            
            //***********************************************************
            //Evaluate each model, and save the inventory decision and final profit
            
            //[P Model] just calculate profit
            profitP[r][n] = pi(yP, demandInPeriod);
            
            
            //[F Model]
            //find inventory with current belief
            int yF = find_yF(cumTimeF, cumQuantityF);
            inventoryF[r][n] = yF;
            
            //calculate and save profit
            profitF[r][n] = pi(yF, demandInPeriod);
            
            //update new observation
            cumQuantityF += demandInPeriod;
            cumTimeF++;
            
            
            //[T Model]
            //find the myopic inventory level
            int yT = find_yF(cumTimeT, cumQuantityT);
            inventoryT[r][n] = yT;
            
            //calculate and save profit
            profitT[r][n] = pi(yT, demandInPeriod);
            
            //update new observation
            if (yT > demandInPeriod)
            {
                cumQuantityT += demandInPeriod;             //if not censored, update like in the F model
                cumTimeT++;
            } else {
                cumQuantityT += yT;                         //if censored, update using stock-out timing
                cumTimeT += arrivalsInPeriod[yT];
            }
            
            
            //[E Model]
            //find the myopic inventory level
            int yE = find_yE(cumTimeE, cumQuantityE, censoredObservationsE);
            inventoryE[r][n] = yE;
            
            //calculate and save profit
            profitE[r][n] = pi(yE, demandInPeriod);
            
            //update new observation
            if (yE > demandInPeriod)
            {
                cumQuantityE += demandInPeriod;             //if not censored, update like in the F model
                cumTimeE++;
            } else {
                censoredObservationsE.insert(yE);           //if censored, add a censoring event (D>=yE)
            }
            
        }
        
        
        
    }
    
    cout << "Simulations are done!" << endl;
    
    
    
    //statistics of the simulation results
    
    cout << "n\tavgPP\tavgPF\tavgPT\tavgPE\tsdPP\tsdPF\tsdPT\tsdPE\tavgIP\tavgIF\tavgIT\tavgIE\tsdIP\tsdIF\tsdIT\tsdIE" << endl;
    file << "n\tavgPP\tavgPF\tavgPT\tavgPE\tsdPP\tsdPF\tsdPT\tsdPE\tavgIP\tavgIF\tavgIT\tavgIE\tsdIP\tsdIF\tsdIT\tsdIE" << endl;
    
    
    //vectors for mean and stdev of Profit
    vector<double> avgProfitPByPeriod(N);
    vector<double> avgProfitFByPeriod(N);
    vector<double> avgProfitTByPeriod(N);
    vector<double> avgProfitEByPeriod(N);
    vector<double> sdProfitPByPeriod(N);
    vector<double> sdProfitFByPeriod(N);
    vector<double> sdProfitTByPeriod(N);
    vector<double> sdProfitEByPeriod(N);
    
    //vectors for mean and stdev of Inventory
    vector<double> avgInventoryPByPeriod(N);
    vector<double> avgInventoryFByPeriod(N);
    vector<double> avgInventoryTByPeriod(N);
    vector<double> avgInventoryEByPeriod(N);
    vector<double> sdInventoryPByPeriod(N);
    vector<double> sdInventoryFByPeriod(N);
    vector<double> sdInventoryTByPeriod(N);
    vector<double> sdInventoryEByPeriod(N);
    
    
    //run stat for each period
    for (unsigned int n=0; n<N; ++n)
    {
        double sumPP = 0;
        double sumPF = 0;
        double sumPT = 0;
        double sumPE = 0;
        
        double sqrsumPP = 0;
        double sqrsumPF = 0;
        double sqrsumPT = 0;
        double sqrsumPE = 0;
        
        double sumIP = 0;
        double sumIF = 0;
        double sumIT = 0;
        double sumIE = 0;
        
        double sqrsumIP = 0;
        double sqrsumIF = 0;
        double sqrsumIT = 0;
        double sqrsumIE = 0;
        
        for (unsigned int r=0; r<RUN; ++r)
        {
            //sum of profits of all the runs
            sumPP += profitP[r][n];
            sumPF += profitF[r][n];
            sumPT += profitT[r][n];
            sumPE += profitE[r][n];
            
            //sum of squared profits of all the runs
            sqrsumPP += pow(profitP[r][n], 2.0);
            sqrsumPF += pow(profitF[r][n], 2.0);
            sqrsumPT += pow(profitT[r][n], 2.0);
            sqrsumPE += pow(profitE[r][n], 2.0);
            
            //sum of inventory of all the runs
            sumIP += inventoryP[r];
            sumIF += inventoryF[r][n];
            sumIT += inventoryT[r][n];
            sumIE += inventoryE[r][n];
            
            //sum of squared inventory of all the runs
            sqrsumIP += pow(inventoryP[r], 2.0);
            sqrsumIF += pow(inventoryF[r][n], 2.0);
            sqrsumIT += pow(inventoryT[r][n], 2.0);
            sqrsumIE += pow(inventoryE[r][n], 2.0);
        }
        
        //mean profits
        avgProfitPByPeriod[n] = sumPP/RUN;
        avgProfitFByPeriod[n] = sumPF/RUN;
        avgProfitTByPeriod[n] = sumPT/RUN;
        avgProfitEByPeriod[n] = sumPE/RUN;
        
        //95% conficendence range of the mean profit
        sdProfitPByPeriod[n] = 1.96 * sqrt((sqrsumPP/RUN - pow(avgProfitPByPeriod[n], 2.0))/(RUN-1));
        sdProfitFByPeriod[n] = 1.96 * sqrt((sqrsumPF/RUN - pow(avgProfitFByPeriod[n], 2.0))/(RUN-1));
        sdProfitTByPeriod[n] = 1.96 * sqrt((sqrsumPT/RUN - pow(avgProfitTByPeriod[n], 2.0))/(RUN-1));
        sdProfitEByPeriod[n] = 1.96 * sqrt((sqrsumPE/RUN - pow(avgProfitEByPeriod[n], 2.0))/(RUN-1));
        
        //mean inventory
        avgInventoryPByPeriod[n] = sumIP/RUN;
        avgInventoryFByPeriod[n] = sumIF/RUN;
        avgInventoryTByPeriod[n] = sumIT/RUN;
        avgInventoryEByPeriod[n] = sumIE/RUN;
        
        //95% conficendence range of the mean inventory
        sdInventoryPByPeriod[n] = 1.96 * sqrt((sqrsumIP/RUN - pow(avgInventoryPByPeriod[n], 2.0))/(RUN-1));
        sdInventoryFByPeriod[n] = 1.96 * sqrt((sqrsumIF/RUN - pow(avgInventoryFByPeriod[n], 2.0))/(RUN-1));
        sdInventoryTByPeriod[n] = 1.96 * sqrt((sqrsumIT/RUN - pow(avgInventoryTByPeriod[n], 2.0))/(RUN-1));
        sdInventoryEByPeriod[n] = 1.96 * sqrt((sqrsumIE/RUN - pow(avgInventoryEByPeriod[n], 2.0))/(RUN-1));
        
        
        //output the means and confidence intervals
        cout << n << "\t" << avgProfitPByPeriod[n] << "\t" << avgProfitFByPeriod[n] << "\t" << avgProfitTByPeriod[n] << "\t" << avgProfitEByPeriod[n] << "\t" << sdProfitPByPeriod[n] << "\t" << sdProfitFByPeriod[n] << "\t" << sdProfitTByPeriod[n] << "\t" << sdProfitEByPeriod[n] << "\t" << avgInventoryPByPeriod[n] << "\t" << avgInventoryFByPeriod[n] << "\t" << avgInventoryTByPeriod[n] << "\t" << avgInventoryEByPeriod[n] << "\t" << sdInventoryPByPeriod[n] << "\t" << sdInventoryFByPeriod[n] << "\t" << sdInventoryTByPeriod[n] << "\t" << sdInventoryEByPeriod[n]<< endl;
        file << n << "\t" << avgProfitPByPeriod[n] << "\t" << avgProfitFByPeriod[n] << "\t" << avgProfitTByPeriod[n] << "\t" << avgProfitEByPeriod[n] << "\t" << sdProfitPByPeriod[n] << "\t" << sdProfitFByPeriod[n] << "\t" << sdProfitTByPeriod[n] << "\t" << sdProfitEByPeriod[n] << "\t" << avgInventoryPByPeriod[n] << "\t" << avgInventoryFByPeriod[n] << "\t" << avgInventoryTByPeriod[n] << "\t" << avgInventoryEByPeriod[n] << "\t" << sdInventoryPByPeriod[n] << "\t" << sdInventoryFByPeriod[n] << "\t" << sdInventoryTByPeriod[n] << "\t" << sdInventoryEByPeriod[n]<< endl;
    }
    
    
    
    
    return 0;
}

