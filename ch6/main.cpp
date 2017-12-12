#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
using namespace std;
struct CURVE_FITTING_COST{
	CURVE_FITTING_COST(double x, double y): _x(x), _y(y){}
	
	template <typename T>
	bool operator() (const T* const abc, T* residual) const
	{
		residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
		return true;
	}
	const double _x, _y;
};

int main(int argc, char** argv)
{
	double a = 1.0, b = 2.0, c = 1.0;
	int N = 100;
	double w_sigma = 1.0;
	cv::RNG rng;
	double abc[3] = {0, 0, 0};
	vector<double> x_data, y_data;
	cout << "Genarating data: " << endl;
	
	for (int i = 0; i < N; i++)
	{
		double x = i/100.0;
		x_data.push_back(x);
		y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
		cout << x_data[i] << ", " << y_data[i] << endl;
	}
	
	ceres::Problem problem;
	for (int i = 0; i < N; i++)
		
	{
        ceres::CostFunction* costfunction = new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i]));
		problem.AddResidualBlock(costfunction, nullptr, abc);
	}
	
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	
	ceres::Solver::Summary summary;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "Time used: " << time_used.count() << "seconds. " << endl;
	
	cout << summary.BriefReport() << endl;
	cout << "estimated a, b, c = ";
	for (auto a:abc) cout << a << ", ";
	cout << endl;
	
	return 0;
}
