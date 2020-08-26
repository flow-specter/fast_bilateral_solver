#include <chrono>
#include <opencv2/ximgproc/edge_filter.hpp>
#include "FBS.h"
#include "PreProcessing.h"
#include "Evaluate.h"

#define TEST_REAL_TOF false // 测试真实场景的tof数据
#define TEST_TEDDY_TOF false // 测试teddy场景的tof数据
#define TEST_THINOUT_MB_Pfm true // 测试Middlebury数据抽稀后的补全实验
	#define PARA_TUNE false //循环调参
		#define PARA_TUNE_LAMBDA false
		#define PARA_TUNE_SPATIAL false
		#define PARA_TUNE_LUMA false
#define ANALYSIS_ROI false  // 选取小块的ROI图像进行分析
#define EMPTY_AREA_TEST false // 测试大面积的空洞

clock_t start;
clock_t end1;

int main()
{	
	/* Test: Tof data for real-world scenarios */
	if (TEST_REAL_TOF) {
		chrono::steady_clock::time_point t1, t2;
		chrono::duration<double> time_used;
		PreProcessing myPreProcessing;
		Evaluate MyEvaluate;
		fstream targetTxt;
		targetTxt.open("data_real_tof/target_real_tof.txt");
		Mat guide = imread("data_real_tof/target_real_tof.png");

		Mat structure_element = getStructuringElement(MORPH_RECT, Size(1, 1)); 
		Mat src_original = Mat::zeros(guide.rows, guide.cols, CV_32F);
		Mat src = Mat::zeros(guide.rows, guide.cols, CV_32F);
		Mat guide_gray;
		Mat confidence = Mat::zeros(guide.rows, guide.cols, CV_32F);
		myPreProcessing.readTargetConfiRef(targetTxt, guide, src_original, src, structure_element, guide_gray, confidence);

		Mat disp_src = src_original.clone();
		float focal = 640;
		float baseline = 50; 
		myPreProcessing.depth2Disp(src_original, disp_src, focal, baseline);

		Mat disp_thinout = disp_src.clone();
		myPreProcessing.thinoutGT(disp_src, 0.99, disp_thinout);
		Mat target_dilate = Mat::zeros(guide.rows, guide.cols, CV_32F);

		myPreProcessing.readTargetConfiRef(disp_thinout, guide, target_dilate, structure_element, guide_gray, confidence);

		Mat dst_whole = Mat::zeros(guide_gray.rows, guide_gray.cols, CV_32FC1);
		Mat dst_whole_DT = Mat::zeros(guide_gray.rows, guide_gray.cols, CV_32FC1);;
		
		float sigma_spatial = 10;
		float sigma_luma = 12;
		float lambda = 2;
		int max_iter = 100;
		int max_tol = 0.00001;

		t1 = chrono::steady_clock::now();  
		FastBilateralSolverFilter myfbs_simulate(guide_gray, sigma_spatial, sigma_luma, lambda, max_iter, max_tol);
		myfbs_simulate.solve(disp_thinout, confidence, dst_whole);
		t2 = chrono::steady_clock::now();
		time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
		dtFilter(guide_gray, dst_whole, dst_whole_DT, 16, 16, cv::ximgproc::DTF_NC, 3);
		Mat dst_whole_DT_median = dst_whole_DT.clone();

		chrono::steady_clock::time_point t3, t4;
		chrono::duration<double> time_used_median;
		t3 = chrono::steady_clock::now();  
		cv::medianBlur(dst_whole_DT, dst_whole_DT_median, 5);
		t4 = chrono::steady_clock::now();  
		time_used_median = chrono::duration_cast <chrono::duration < double >> (t4 - t3);
		cout << "median_filter time_used: " << endl;

		float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, disp_src, 2);
		float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, disp_src);
		cout << "耗时： " << time_used.count() << endl;
		cout << "bad2:" << bad2 << endl;
		cout << "rmse:" << rmse << endl;

	}

	/* Test: depth-inpainiting */
	if (TEST_THINOUT_MB_Pfm) {
		// 1. thinout 
		PreProcessing myPreProcessing;
		Mat dispGT = myPreProcessing.loadPFM("MiddEval3-trainingH/Playtable/disp0GT.pfm");
		Mat reference = imread("MiddEval3-trainingH/Playtable/im0.png");

		Mat GT_thinout = dispGT.clone();
		myPreProcessing.thinoutGT(dispGT, 0.99, GT_thinout);
		Mat target_dilate = Mat::zeros(reference.rows, reference.cols, CV_32F);

		// 1*1 = no dilate
		Mat structure_element = getStructuringElement(MORPH_RECT, Size(1,1)); 
		Mat reference_gray;
		Mat confidence = Mat::zeros(reference.rows, reference.cols, CV_32F);
		myPreProcessing.readTargetConfiRef(GT_thinout, reference, target_dilate, structure_element, reference_gray, confidence);
		
		// 2. solve
		Mat dst_whole = Mat::zeros(reference_gray.rows, reference_gray.cols, CV_32FC1);
		Mat dst_whole_DT = Mat::zeros(reference_gray.rows, reference_gray.cols, CV_32FC1);;
		
		chrono::steady_clock::time_point t1, t2,t3;
		chrono::duration<double> time_used, time_used_1;
		t1 = chrono::steady_clock::now();  
		FastBilateralSolverFilter myfbs_simulate(reference_gray, 12, 20, 1, 50, 0.0000001);
		myfbs_simulate.solve(target_dilate, confidence, dst_whole);
		t2 = chrono::steady_clock::now();
		time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
		cout << "Total time:" << time_used.count() << endl;	

		// 3. post processing; DT filter
		dtFilter(reference_gray, dst_whole, dst_whole_DT, 16, 16, cv::ximgproc::DTF_NC, 3);
		t3 = chrono::steady_clock::now();
		time_used_1 = chrono::duration_cast <chrono::duration < double >> (t3 - t1);
		cout << "Total time(with DT):" << time_used_1.count() << endl;	//ms为单位

		// 4. eval： bad2 & rmse
		Evaluate MyEvaluate;
		float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, dispGT, 2);
		float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, dispGT);
		cout << "bad2:" << bad2 << endl;
		cout << "rmse:" << rmse << endl;		
	
		// 5. finetune
		if (PARA_TUNE) {

			if(PARA_TUNE_LAMBDA){
				//0.01
				for (float lambda = 0; lambda < 10; lambda += 1)
				{
					t1 = chrono::steady_clock::now();  // 计时
					FastBilateralSolverFilter myfbs_simulate(reference_gray, 10, 8, lambda, 60, 0.000001);
					myfbs_simulate.solve(target_dilate, confidence, dst_whole);
					t2 = chrono::steady_clock::now();
					time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
					dtFilter(reference_gray, dst_whole, dst_whole_DT, 16, 16, cv::ximgproc::DTF_NC, 3);
					cout << "lambda  " << lambda << "  " << "耗时： " << time_used.count() << endl;;
					float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, dispGT, 2);
					float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, dispGT);
					cout << "bad2:" << bad2 << endl;
					cout << "rmse:" << rmse << endl;
					end1 = clock();
					double endtime = (double)(end1 - start) / CLOCKS_PER_SEC;
					cout << "Total time:" << endtime * 1000 << "ms" << endl;	//ms为单位
					cout << endl;
				}
			}

			if (PARA_TUNE_SPATIAL) {
				for (float sigma_spatial = 8; sigma_spatial < 18; sigma_spatial += 1)
				{
					t1 = chrono::steady_clock::now();  
					FastBilateralSolverFilter myfbs_simulate(reference_gray, sigma_spatial, 8, 0.01, 60, 0.000001);
					myfbs_simulate.solve(target_dilate, confidence, dst_whole);
					t2 = chrono::steady_clock::now();
					time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
					dtFilter(reference_gray, dst_whole, dst_whole_DT, 16, 16, cv::ximgproc::DTF_NC, 3);
					cout << "sigma_spatial  " << sigma_spatial << "  " << "耗时： " << time_used.count() << endl;;
					float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, dispGT, 2);
					float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, dispGT);
					cout << "bad2:" << bad2 << endl;
					cout << "rmse:" << rmse << endl;
					end1 = clock();
					double endtime = (double)(end1 - start) / CLOCKS_PER_SEC;
					cout << "Total time:" << endtime * 1000 << "ms" << endl;	
					cout << endl;
				}
			}

			if (PARA_TUNE_LUMA) {
				for (float sigma_luma = 4; sigma_luma < 256; sigma_luma += 4)
				{
					t1 = chrono::steady_clock::now();  
					FastBilateralSolverFilter myfbs_simulate(reference_gray, 10, sigma_luma, 0.01, 60, 0.000001);
					myfbs_simulate.solve(target_dilate, confidence, dst_whole);
					t2 = chrono::steady_clock::now();
					time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
					dtFilter(reference_gray, dst_whole, dst_whole_DT, 16, 16, cv::ximgproc::DTF_NC, 3);
					cout << "sigma_luma  " << sigma_luma << "  " << "耗时： " << time_used.count() << endl;;
					float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, dispGT, 2);
					float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, dispGT);
					cout << "bad2:" << bad2 << endl;
					cout << "rmse:" << rmse << endl;
					end1 = clock();
					double endtime = (double)(end1 - start) / CLOCKS_PER_SEC;
					cout << "Total time:" << endtime * 1000 << "ms" << endl;	
					cout << endl;
				}
			}
		}
	}

	/* Test: Tof data for TEDDY*/
	if (TEST_TEDDY_TOF) {
		chrono::steady_clock::time_point t1, t2;
		chrono::duration<double> time_used;
		PreProcessing myPreProcessing;
		Evaluate MyEvaluate;
		fstream targetTxt;
		targetTxt.open("sr_d/disp_sparse.txt");

		Mat guide = imread("sr_d/sx_und.png");
		Mat structure_element = getStructuringElement(MORPH_RECT, Size(1, 1)); 

		Mat src_original = Mat::zeros(guide.rows, guide.cols, CV_32F);
		Mat src = Mat::zeros(guide.rows, guide.cols, CV_32F);
		Mat guide_gray;
		Mat confidence = Mat::zeros(guide.rows, guide.cols, CV_32F);
		myPreProcessing.readTargetConfiRef(targetTxt, guide, src_original, src, structure_element, guide_gray, confidence);
		
		Mat dst_whole = Mat::zeros(guide_gray.rows, guide_gray.cols, CV_32FC1);
		Mat dst_whole_DT = Mat::zeros(guide_gray.rows, guide_gray.cols, CV_32FC1);;

		t1 = chrono::steady_clock::now(); 
		
		float sigma_spatial = 10;
		float sigma_luma = 10;
		float lambda = 1;
		int num_iter = 50;
		float max_tol = 0.000001;

		FastBilateralSolverFilter myfbs_simulate(guide_gray, sigma_spatial, sigma_luma, lambda, num_iter, max_tol);
		myfbs_simulate.solve(src, confidence, dst_whole);
		t2 = chrono::steady_clock::now();
		time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
		
		dtFilter(guide_gray, dst_whole, dst_whole_DT, 16, 16, cv::ximgproc::DTF_NC, 3);
		float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, src_original, 2);
		float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, src_original);
		cout << "耗时： " << time_used.count() << endl;
		cout << "bad2:" << bad2 << endl;		
		cout << "rmse:" << rmse << endl;
	}

	/* Test: large area blank */ 
	if (EMPTY_AREA_TEST) {
		PreProcessing myPreProcessing;
		
		Mat dispGT = myPreProcessing.loadPFM("MiddEval3-trainingH/Playtable/disp0GT.pfm");
		Mat reference = imread("MiddEval3-trainingH/Playtable/im0.png");

		Mat GT_thinout = dispGT.clone();
		myPreProcessing.thinoutGT(dispGT, 0.99, GT_thinout);

		Mat target_dilate = Mat::zeros(reference.rows, reference.cols, CV_32F);

		// 1*1 = no dilate
		Mat structure_element = getStructuringElement(MORPH_RECT, Size(1, 1));
		Mat reference_gray;
		Mat confidence = Mat::zeros(reference.rows, reference.cols, CV_32F);

		myPreProcessing.readTargetConfiRef(GT_thinout, reference, target_dilate, structure_element, reference_gray, confidence);

		Mat target_empty_area = target_dilate.clone();
		Mat confidence_empty_area = confidence.clone();
		myPreProcessing.emptyArea(10,10,700,700, target_empty_area, confidence_empty_area);

		Mat dst_whole = Mat::zeros(reference_gray.rows, reference_gray.cols, CV_32FC1);
		Mat dst_whole_DT = Mat::zeros(reference_gray.rows, reference_gray.cols, CV_32FC1);;

		chrono::steady_clock::time_point t1, t2, t3;
		chrono::duration<double> time_used, time_used_1;
		t1 = chrono::steady_clock::now();  // 计时

		FastBilateralSolverFilter myfbs_simulate(reference_gray, 10, 30, 2, 20, 0.0000001);
		myfbs_simulate.solve(target_empty_area, confidence_empty_area, dst_whole);

		t2 = chrono::steady_clock::now();
		time_used = chrono::duration_cast <chrono::duration < double >> (t2 - t1);
		cout << "Total time:" << time_used.count() << endl;	//ms为单位

		dtFilter(reference_gray, dst_whole, dst_whole_DT, 64, 32, cv::ximgproc::DTF_NC, 3);
		t3 = chrono::steady_clock::now();
		time_used_1 = chrono::duration_cast <chrono::duration < double >> (t3 - t1);
		cout << "Total time(with DT):" << time_used_1.count() << endl;	//ms为单位

		Evaluate MyEvaluate;
		float bad2 = MyEvaluate.eval_bad_tof(dst_whole_DT, dispGT, 2);
		float rmse = MyEvaluate.eval_rmse_tof(dst_whole_DT, dispGT);
		cout << "bad2:" << bad2 << endl;
		cout << "rmse:" << rmse << endl;	
	}

	/* Analysis */
	if (ANALYSIS_ROI) {
		PreProcessing myPreProcessing;
		Mat guide = imread("sr_d/sx_und.png");

		//  1. 读取confidence，target， reference的roi图块
		fstream targetTxt;
		targetTxt.open("sr_d/disp_sparse.txt");
		Mat target = Mat::zeros(guide.rows, guide.cols, CV_32F);
		myPreProcessing.txt2Mat32F(targetTxt, guide.rows, guide.cols, target);

		Mat confidence = 0.6*Mat::ones(guide.rows, guide.cols, CV_32FC1);
		myPreProcessing.generateConfi(target, target, confidence);

		int x = 0;
		int y = 0;
		int width = 800;
		int height = 700;
		Rect area(x, y, width, height);
		Mat target_roi = target(area);
		Mat confidence_roi = confidence(area);
		Mat guide_gray = Mat::zeros(guide.rows, guide.cols, CV_32FC1);
		cv::cvtColor(guide, guide_gray, cv::COLOR_BGR2GRAY);
		Mat guide_roi_gray = guide_gray(area);

		FastBilateralSolverFilter myfbs_simulate(guide_roi_gray, 5, 30, 1, 20, 0.0001);
		Mat dst_patch = Mat::zeros(guide_roi_gray.rows, guide_roi_gray.cols, CV_32FC1);
		myfbs_simulate.solve(target_roi, confidence_roi, dst_patch);

		Mat res = dst_patch.clone();
		cv::medianBlur(dst_patch, res, 3);
		Mat output_grid = target_roi.clone();
		myfbs_simulate.showGrid(output_grid);
	}

	waitKey(0);
	return 0;
}
