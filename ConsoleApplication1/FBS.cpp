#include "FBS.h"
#define EIGEN_CG false  // Eigen API

// ��ʼ��ʱ
void StartCalTime(float &t)
{
	t = (float)cvGetTickCount();//��ʱ��ʼ
}

// ֹͣ��ʱ�������ʱ���
void StopCalTime(float &t, char* textout)
{
	float tnow;
	tnow = (float)cvGetTickCount() - t;//��ʱֹͣ,�����ʱ��

	tnow = (float)(tnow / ((float)cvGetTickFrequency()*1000.0));

	printf(textout);
	printf("  time %.1f ms\n\n", tnow);
}

/* ������Ϊ�Խ��������ɶԽ��� */
void FastBilateralSolverFilter::diagonal(Eigen::VectorXf& v, Eigen::SparseMatrix<float>& mat)
{
	mat = Eigen::SparseMatrix<float>(v.size(), v.size());
	for (int i = 0; i < int(v.size()); i++)
	{
		mat.insert(i, i) = v(i);
	}
	//cout << "diag(Sc)" << endl;
	//cout << mat << endl;
}

/* splat�����������ؿռ������ת��Ϊ˫�߿ռ������ */
void FastBilateralSolverFilter::Splat(Eigen::VectorXf& input, Eigen::VectorXf& output)
{
	output.setZero();
	for (int i = 0; i < int(splat_idx.size()); i++)
	{
		output(splat_idx[i]) += input(i);
	}
}

/* blur��������˫�߿ռ�����������˲� */
void FastBilateralSolverFilter::Blur(Eigen::VectorXf& input, Eigen::VectorXf& output)
{
	output.setZero();
	output = input * 6; 

	for (int i = 0; i < int(blur_idx.size()); i++)
	{
		output(blur_idx[i].first) += input(blur_idx[i].second);
	}

}

/* slice��������˫�߿ռ��е������任Ϊ���ؿռ��е����� */
void FastBilateralSolverFilter::Slice(Eigen::VectorXf& input, Eigen::VectorXf& output)
{
	output.setZero();
	for (int i = 0; i < int(splat_idx.size()); i++)
	{
		output(i) = input(splat_idx[i]);
	}
}

/* ����˫�߸��� */
void FastBilateralSolverFilter::ConstructBilateralGrid(cv::Mat& reference)
{
	if (reference.channels() == 1)
	{
		dim = 3;
		cols = reference.cols;
		rows = reference.rows;
		npixels = cols*rows;
		long long hash_vec[3];
		for (int i = 0; i < 3; ++i)
			hash_vec[i] = static_cast<long long>(std::pow(255, i));
		mapId hashed_coords;
#if __cplusplus <= 199711L
#else
		hashed_coords.reserve(cols*rows);
#endif
		const unsigned char* pref = (const unsigned char*)reference.data;
		int vert_idx = 0;
		int pix_idx = 0;

		float t_splat=0;
		StartCalTime(t_splat);

		// ����splat/slice����
		splat_idx.resize(npixels);
		for (int y = 0; y < rows; ++y)
		{
			for (int x = 0; x < cols; ++x)
			{
				long long coord[3];
				coord[0] = int(x / grid_param.spatialSigma);
				coord[1] = int(y / grid_param.spatialSigma);
				coord[2] = int(pref[0] / grid_param.lumaSigma);

				// ������ֵתΪΨһ�Ĺ�ϣֵ
				long long hash_coord = 0;
				for (int i = 0; i < 3; ++i)
					hash_coord += coord[i] * hash_vec[i];

				mapId::iterator it = hashed_coords.find(hash_coord);
				if (it == hashed_coords.end())
				{
					hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
					splat_idx[pix_idx] = vert_idx;
					++vert_idx;
				}
				else
				{
					splat_idx[pix_idx] = it->second;
				}
				pref += 1; 
				++pix_idx;
			}
		}

		StopCalTime(t_splat,"����splat����");

		nvertices = static_cast<int>(hashed_coords.size());

		// construct Blur matrices
		float t_blur;
		StartCalTime(t_blur);
		Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
		diagonal(ones_nvertices, blurs);
		blurs *= 6;

		for (int offset = -1; offset <= 1;++offset)
		{
			if (offset == 0) continue;
			for (int i = 0; i < dim; ++i)
			{
				Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
				blur_temp.reserve(Eigen::VectorXi::Constant(nvertices, 6));

				long long offset_hash_coord = offset * hash_vec[i];
				
				for (mapId::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
				{
					long long neighb_coord = it->first + offset_hash_coord;
					mapId::iterator it_neighb = hashed_coords.find(neighb_coord);
					if (it_neighb != hashed_coords.end())
					{
						blur_temp.insert(it->second, it_neighb->second) = 1.0f;
						blur_idx.push_back(std::pair<int, int>(it->second, it_neighb->second));
					}
				}	
				blurs += blur_temp;
			}
		}
		StopCalTime(t_blur, "����blur����");

		//bistochastize
		float t_bisto;
		StartCalTime(t_bisto);
		int maxiter = 10;
		n = ones_nvertices;
		m = Eigen::VectorXf::Zero(nvertices);

		for (int i = 0; i < int(splat_idx.size()); i++)
		{
			m(splat_idx[i]) += 1.0f;
		}

		Eigen::VectorXf bluredn(nvertices);
		for (int i = 0; i < maxiter; i++)
		{
			Blur(n, bluredn);
			n = ((n.array()*m.array()).array() / bluredn.array()).array().sqrt();
		}

		Blur(n, bluredn);
		m = n.array() * (bluredn).array();
		diagonal(m, Dm);
		diagonal(n, Dn);
		StopCalTime(t_bisto, "bisto");

	}
	else {
		cout << "reference��Ԥ����Ϊ��ͨ��" << endl;
	}
}

/* ���A��b����������pcg�Ż������������ */
void FastBilateralSolverFilter::solve(cv::Mat& target,
	cv::Mat& confidence,
	cv::Mat& output)
{
	Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices, nvertices);
	Eigen::VectorXf b(nvertices);
	Eigen::VectorXf y(nvertices);
	Eigen::VectorXf y0(nvertices);
	Eigen::VectorXf y1(nvertices);
	Eigen::VectorXf c_splat(nvertices);
	Eigen::VectorXf x(npixels);
	Eigen::VectorXf c(npixels);

	float prepare_cx = 0;
	StartCalTime(prepare_cx);
	if (target.depth() == CV_32F)
	{
		for (int i = 0;i < target.rows;i++) {
			const float *pft = target.ptr<float>(i);
			for (int j = 0; j < target.cols; j++) {
				x(j+target.cols*i)= pft[j];
			}
		}
	}
	else {
		cout << "Ӧ����CV_32F��targetӰ��" << endl;
		system("pause");
	}

	if (confidence.depth() == CV_32F)
	{
		for (int i = 0;i < confidence.rows;i++) {
			const float *pft = confidence.ptr<float>(i);
			for (int j = 0; j < confidence.cols; j++) {
				c(j + confidence.cols*i) = pft[j];
			}
		}
	}
	else {
		cout << "Ӧ����CV_32F��confidenceӰ��,�ұ�֤��ֵ��0-1֮��" << endl;
		system("pause");
	}
	StopCalTime(prepare_cx, "׼��cx");

	float t_construct_Ab;

	StartCalTime(t_construct_Ab);

	//�������� A
	Splat(c, c_splat);
	diagonal(c_splat, A_data);

	A = bs_param.lam * (Dm - Dn * (blurs*Dn)) + A_data;

	//�������� b
	b.setZero();

	for (int i = 0; i < int(splat_idx.size()); i++)
	{
		b(splat_idx[i]) += x(i) * c(i);
	}

	StopCalTime(t_construct_Ab,"����Ab");

	float t_pcg = 0;
	StartCalTime(t_pcg);
	pcg(A, b, c, x, bs_param.cg_maxiter, bs_param.cg_tol, y);
	StopCalTime(t_pcg, "PCG ��ʱ");

	// ��Eigen�е�API���бȶ�
	if (EIGEN_CG) {
		//construct guess for y
		float t_Eigen_API = 0;
		StartCalTime(t_Eigen_API);
		y0.setZero();
		for (int i = 0; i < int(splat_idx.size()); i++) {
			y0(splat_idx[i]) += x(i);
		}

		y1.setZero();

		for (int i = 0; i < int(splat_idx.size()); i++) {
			y1(splat_idx[i]) += 1.0f;
		}

		for (int i = 0; i < nvertices; i++) {
			y0(i) = y0(i) / y1(i);
		}

		// solve Ay = b
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
		cg.compute(A);
		cg.setMaxIterations(bs_param.cg_maxiter);
		cg.setTolerance(bs_param.cg_tol);

		//y = cg.solve(b);
		y = cg.solveWithGuess(b, y0);
		StopCalTime(t_Eigen_API, "cg");

		std::cout << "#iterations:     " << cg.iterations() << std::endl;
		std::cout << "estimated error: " << cg.error() << std::endl;
	}

	//slice
	float t_slice = 0;
	StartCalTime(t_slice);
	float *pftar = (float*)(output.data);
	for (int i = 0; i < int(splat_idx.size()); i++)
	{
		pftar[i] = y(splat_idx[i]); 
	}
	StopCalTime(t_slice, "slice");
}

/* �����ſɱ�Ԥ�����ӵ�CG�㷨��������ϡ�����Է�����*/
void FastBilateralSolverFilter::pcg(Eigen::SparseMatrix<float, Eigen::ColMajor> A,Eigen::VectorXf b, Eigen::VectorXf c, Eigen::VectorXf t,int max_iter, int err_threshold, Eigen::VectorXf &y_output) {

	/* 0. construct Jacobi preconditioner */
	float lambda = bs_param.lam;
	Eigen::VectorXf A_diag(b.size());
	Eigen::VectorXf c_splat(b.size());
	Eigen::VectorXf blur_diag(b.size());

	blur_diag = 6 * Eigen::VectorXf::Ones(b.size());
	Splat(c, c_splat);
	A_diag = lambda * (m.array() - n.array() * blur_diag.array() * n.array()) + c_splat.array();

	Eigen::SparseMatrix<float> M_jacobi_inv(b.size(), b.size());
	for (int i = 0; i < int(b.size()); i++)
	{
		M_jacobi_inv.insert(i, i) = (1+0.00001)/ (A_diag(i)+ 0.00001); //avoid NaN
	}

	/* 1. construct y_flat or y_initial */
	Eigen::VectorXf ct_splat(b.size());
	Eigen::VectorXf ct(b.size());

	ct = c.array()*t.array();
	Splat(ct, ct_splat);
	Eigen::VectorXf y_flat = ct_splat.array() / (c_splat.array()+ 0.00001);
	
	/* 2. iter */
	int i = 0;
	Eigen::VectorXf r(b.size());
	Eigen::VectorXf d(b.size());
	Eigen::VectorXf q(b.size());
	Eigen::VectorXf s(b.size());

	r = b - A*y_flat;
	d = M_jacobi_inv * r;
	float lambda_new = r.transpose()*d;
	y_output = y_flat;

	while (i < max_iter) {
		q = A*d;
		float alpha = lambda_new / (d.transpose()*q);
		if (isinf(alpha) || isnan(alpha) || isnan(-alpha)) break;

		y_output = y_output + alpha*d;
		r = r - alpha*q;
		if (r.norm() <= err_threshold)
			break;
		s = M_jacobi_inv * r;
		float lambda_old = lambda_new;
		lambda_new = r.transpose() * s;
		float beta = lambda_new / lambda_old;
		d = s + beta * d;
		++i;
	}

	cout << "����������" << i << endl;
	cout << "������" << r.norm() << endl;
	cout << "˫�߸�������ά��" << b.size() << endl;
};

/* ��ʾÿ�������������vertex */
void FastBilateralSolverFilter::showGrid(Mat &output_grid) {
	int rows = output_grid.rows;
	int cols = output_grid.cols;
	for (int i = 0;i < rows;++i) {
		float* ptr_output_grid = output_grid.ptr<float>(i);
		for (int j = 0; j < cols;++j) {
			ptr_output_grid[j] = splat_idx[i*cols + j];
		}
	}
};
