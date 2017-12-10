#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#define MATRIX_SIZE 3

using namespace std;

int main(int argc, char **argv) {
    Eigen::Matrix<float, 2, 3> matrix_23;
	Eigen::Vector3d v_3d;
	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
	Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
	Eigen::MatrixXd matrix_x;
	matrix_23 << 1, 2, 3, 4, 5, 6;
	cout << "Matrix2_3: \n" << matrix_23 << endl << endl;
	for (int i=0; i<2; i++)
	{
		for(int j=0; j<3; j++)
			cout << matrix_23(i, j) << ' ';
		cout << endl;
	}
	v_3d << 3, 2, 1;
	// Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << "Matrix2_3 * v_3d: \n" << result << endl;
	//Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<doulbe>() * v_3d;
	
	matrix_33 = Eigen::Matrix3d::Random();
	cout << "A 3*3 matrix: \n" << matrix_33 << endl << endl;
	cout << "Transpose: \n" << matrix_33.transpose() << endl;
	cout << "Sum: \n" << matrix_33.sum() << endl;
	cout << "10 * Matrix: \n" << 10*matrix_33 << endl;
	cout << "Inverse: \n" << matrix_33.inverse() << endl;
	cout << "Determinant: \n" << matrix_33.determinant() << endl;
	
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
	cout << "Eigen values: \n" << eigen_solver.eigenvalues() << endl;
	cout << "Eigen vectors: \n" << eigen_solver.eigenvectors() <<endl;
	
	Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
	Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
	v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);
	
	clock_t time_stt = clock();
	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
	cout << "time use in normal inverse is " << 1000 * (clock() - time_stt)/(double)CLOCKS_PER_SEC
	<< "ms" << endl;
	cout << "x: \n" << x << endl;
	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << "time use in Qr composition is " << 1000 * (clock() - time_stt)/(double)CLOCKS_PER_SEC
	<< "ms" << endl;
	cout << "x: \n" << x << endl;
	
    return 0;
}
