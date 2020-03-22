#include "Eigen/Eigen"
#define VOLESTI_DEBUG
#include <fstream>
#include "random.hpp"
#include "random/uniform_int.hpp"
#include "random/normal_distribution.hpp"
#include "random/uniform_real_distribution.hpp"
#include "volume.h"
#include "rotating.h"
#include "misc.h"
#include "linear_extensions.h"
#include "cooling_balls.h"
#include "cooling_hpoly.h"
#include "sample_only.h"
#include "exact_vols.h"
#include "known_polytope_generators.h"
#include "h_polytopes_gen.h"

using namespace std;

typedef boost::mt19937 RNGType;
typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope <Point> Hpolytope;

// auxilary variables 
int walk_len = 10, nsam = 1000;
int n, N = 20, W = 20, n_threads = 1;
int rnum = pow(0.1,-2) * 400 * n * log(n);
double a = 0.5;

bool user_W=false, user_N=false, user__ratio=false, user_NN = false, set_algo = false, set_error = false;
NT ball_radius=0.0, diameter = -1.0, lb = 0.1, ub = 0.15, p = 0.75, rmax = 0.0, alpha = 0.2, _round_val = 1.0;
NT C=2.0,_ratio,frac=0.1,delta=-1.0,error=0.1;
bool birk=false, rotate=false, _ball_walk=false, ball_rad=false, experiments=true, CG = false, Vpoly=false, Zono=false, cdhr=false, 
rdhr=false,user_randwalk = false, exact_zono = false, gaussian_sam = false, hpoly = false, billiard=false, win2 = false,
boundary = false, verbose=false, rand_only=false, _round_only=false, file=false, _round=false, NN=false, user_walk_len=false, 
linear_extensions=false;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
RNGType rng(seed);
boost::normal_distribution<> rdist(0,1);
boost::random::uniform_real_distribution<>(urdist);
boost::random::uniform_real_distribution<> urdist1(-1, 1);


double func(Point p1, Point p2, double a) {
    int dim = p1.dimension();
    Point diff = p1 - p2;

    double sum = 0;
    
    for (int i = 0; i < dim; ++i) {
        sum += diff[i] * diff[i];
    }
    
    return exp(-a * sum);
}

double monteCarlo(Hpolytope HP, pair<Point, NT> InnerBall, double vol) {
    double radius = InnerBall.second;
    
    vars<NT, RNGType> var1(0, n, walk_len, 1, 0, 0, 0, 0.0, 0, radius, diameter, rng,
            urdist, urdist1, delta, verbose, rand_only, _round, NN, birk, _ball_walk, cdhr, rdhr,billiard);
    vars_g<NT, RNGType> var2(n, walk_len, N, W, 1, 0, radius, rng, C, frac, _ratio, delta,
          verbose, rand_only, _round, NN, birk, _ball_walk, cdhr, rdhr);

    // Monte Carlo implementation
    Point x0 = InnerBall.first;
    
    list<Point> randPoints;
    sampling_only<Point>(randPoints, HP, walk_len, nsam, false, a, false, x0, var1, var2);
    
    double sum = 0;

    for (auto pnt: randPoints) {
        sum += func(pnt, x0, 0.5);
    }
      
    return sum * vol / (double)nsam;
}


int main(const int argc, const char** argv) {
    // Computation of the Integral using Monte-Carlo
    Hpolytope HP = random_hpoly<Hpolytope, RNGType>(10, 500);
    pair<Point, NT> InnerBall = HP.ComputeInnerBall();
    
    n = HP.dimension();

    vars_g <NT, RNGType> _var1(n, walk_len, N, W, 1, error, InnerBall.second, rng, C, frac, _ratio, delta,
                              verbose, rand_only, _round, NN, birk, _ball_walk, cdhr, rdhr);
    vars <NT, RNGType> _var2(rnum, n, 10 + n / 10, n_threads, error, error, 0, 0.0, 0, InnerBall.second, diameter, rng,
                             urdist, urdist1, delta, verbose, rand_only, _round, NN, birk, _ball_walk, cdhr,
                             rdhr,billiard);

    double vol = volume_gaussian_annealing(HP, _var1, _var2, InnerBall);
    
    int repet = 20;
    double mean = 0;
    vector<double> values;
    
    for (int t = 1; t <= repet; ++t) {
        values.push_back(monteCarlo(HP, InnerBall, vol));
        mean += values.back();
    }
    
    mean /= repet;
    double sum = 0;

    for (auto val: values) {
        sum += (val - mean) * (val - mean);
    }
    
    sum /= (repet - 1);
    double std = sqrt(sum);

    cout << "integral = " << mean << "  standard deviation = " << std << endl;
    return 0;
}
