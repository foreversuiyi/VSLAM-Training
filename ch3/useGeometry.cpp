#include <iostream>
#include <cmath>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char** argv)
{
	Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
	Eigen::AngleAxisd rotation_vector(M_PI/4, Eigen::Vector3d(0, 0, 1));
	cout.precision(3);
	cout<<"rotation matrix = \n" << rotation_vector.matrix() << endl;
	
	rotation_matrix = rotation_vector.toRotationMatrix();
	Eigen::Vector3d v(1, 0, 0);
	Eigen::Vector3d v_rotated = rotation_vector * v;
	cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << endl;
	v_rotated = rotation_matrix * v;
	cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << endl;
	
	Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
	cout << "yaw pitch roll = " << euler_angles.transpose() << endl;
	
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	T.rotate(rotation_vector);
	T.pretranslate(Eigen::Vector3d(1, 3, 4));
	cout << "Transform matrix = \n" << T.matrix() << endl;
	Eigen::Vector3d v_transformed = T * v;
	cout << "v_tranformed = " << v_transformed.transpose() << endl;
	
	Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
	cout << "Quaternion = \n " << q.coeffs() << endl;
	q = Eigen::Quaterniond(rotation_matrix);
	cout << "Quaternion = \n " << q.coeffs() << endl;
	v_rotated = q * v;
	cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << endl;
	
	return 0;
}