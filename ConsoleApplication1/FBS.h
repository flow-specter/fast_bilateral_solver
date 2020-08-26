#include <cmath>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>
#include <omp.h>
using namespace std;
using namespace cv;

typedef std::unordered_map<long long /* hash */, int /* vert id */>  mapId;

struct bs_params
{
	float lam;
	float A_diag_min;
	float cg_tol;
	int cg_maxiter;
	bs_params()
	{
		lam = 128.0;
		A_diag_min = 1e-5f;
		cg_tol = 1e-5f;
		cg_maxiter = 25;
	}
};

struct grid_params
{
	float spatialSigma;
	float lumaSigma;
	grid_params()
	{
		spatialSigma = 8.0;
		lumaSigma = 8.0;
	}
};

class FastBilateralSolverFilter {

public:

	// initial; construct the bilateral grid.
	FastBilateralSolverFilter(Mat& guide, float sigma_spatial, float sigma_luma, float lambda, int num_iter, float max_tol){
		CV_Assert(guide.type() == CV_8UC1 || guide.type() == CV_8UC3);
		grid_param.spatialSigma = sigma_spatial;
		grid_param.lumaSigma = sigma_luma;
		bs_param.cg_maxiter = num_iter;
		bs_param.cg_tol = max_tol;
		bs_param.lam = lambda;
		ConstructBilateralGrid(guide); 
	}

	void solve(cv::Mat& src, cv::Mat& confidence, cv::Mat& dst);
	void pcg(Eigen::SparseMatrix<float, Eigen::ColMajor> A, Eigen::VectorXf b, Eigen::VectorXf c, Eigen::VectorXf t, int max_iter, int err_threshold, Eigen::VectorXf &y_output);
	void showGrid(Mat &output_grid);

private:
	int rows, cols;
	int dim;

	void ConstructBilateralGrid(cv::Mat& reference);
	void Splat(Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void Blur(Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void Slice(Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void diagonal(Eigen::VectorXf& v, Eigen::SparseMatrix<float>& mat);

	int npixels, nvertices;
	grid_params grid_param;
	bs_params bs_param;
	std::vector<std::pair<int, int> > blur_idx;
	Eigen::VectorXf m;
	Eigen::VectorXf n;
	Eigen::SparseMatrix<float, Eigen::ColMajor> Dn;
	Eigen::SparseMatrix<float, Eigen::ColMajor> Dm;
	Eigen::SparseMatrix<float, Eigen::ColMajor> blurs; 
	std::vector<int> splat_idx;
};


