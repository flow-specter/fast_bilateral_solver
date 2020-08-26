#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Evaluate
{
public:
	Evaluate() {};
	~Evaluate() {};

	/* 计算所得图像与GT的rmse指标 */
	float eval_rmse(Mat img, Mat gt) {
		float rmse = 0;
		Mat tmp;
		gt.convertTo(gt, CV_32F);
		absdiff(img, gt, tmp);
		tmp = tmp.mul(tmp);
		rmse = float(sqrt(mean(tmp)[0]));
		return rmse;
	}

	/* 计算所得图像与GT的bad指标 */
	float eval_bad(Mat disp, Mat gt, float bad_thresh = 2) {
		int h = disp.rows;
		int w = disp.cols;
		int bad_nums = 0;
		gt.convertTo(gt, CV_32F);

		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) {
				float d = disp.at<float>(r, c);
				float d_gt = gt.at<float>(r, c);
				float error = fabs(d - d_gt);
				if (error > bad_thresh) {
					bad_nums++;
				}
			}
		}
		float bad_percent = 100.0f * bad_nums / (h * w);
		return bad_percent;
	}

	/* 计算所得图像与tof点之间的bad指标 */
	float eval_bad_tof(Mat disp, Mat gt, float bad_thresh = 2) {
		int h = disp.rows;
		int w = disp.cols;
		int bad_nums = 0;

		for (int r = 0; r < h; r++) {
			float* ptr_d_gt = gt.ptr<float>(r);
			for (int c = 0; c < w; c++) {
				float d_gt = ptr_d_gt[c];
				if(d_gt>0 ){ //对tof点中有数据的点进行计算
					float d = disp.at<float>(r, c);
					float error = fabs(d - d_gt);
					if (error > bad_thresh) {
						bad_nums++;
					}
				}
			}
		}
		float bad_percent = 100.0f * bad_nums / (h * w);
		return bad_percent;
	}

	/* 计算所得图像与tof点之间的rmse指标 */
	float eval_rmse_tof(Mat img, Mat gt) {
		int h = img.rows;
		int w = img.cols;
		int valid_nums = 0;
		float res = 0;

		for (int r = 0; r < h; r++) {
			float* ptr_d_gt = gt.ptr<float>(r);
			float* ptr_img = img.ptr<float>(r);
			for (int c = 0; c < w; c++) {
				float d_gt = ptr_d_gt[c];
				if (d_gt != 0) { //对tof点中有数据的点进行计算
					float delta = ptr_img[c] - d_gt;
					res += delta * delta;
					valid_nums++;
				}
			}
		}
		res = sqrt(res / valid_nums);
		return res;
	}
};

