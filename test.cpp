#include "Matrix.h"

int main()
{
    using namespace algebra;

    vector<vector<double>> vec{{ 2, -1 ,3},{1,3,2},{5,3,0}};
    Matrix<double> m(vec);
    pair<Matrix<double>,Matrix<double>> p = m.QR();
    cout<<"------Q------"<<endl;
    p.first.print_Matrix();
    cout<<endl;
    cout<<"------R------"<<endl;
    p.second.print_Matrix();
    cout<<endl;
    cout<<"------Q*R------"<<endl;
    Matrix<double> r = p.first*p.second;
    cout<<endl;
    r.trimMatrix();
    r.print_Matrix();

    cout<<endl<<endl;

    array<Matrix<double>, 3> arr = m.svd();
	Matrix<double> & U = arr[0], & E = arr[1], & VT=arr[2];
	cout << "------U------" << endl;
	U.print_Matrix();
	cout << endl;
	cout << "------E------" << endl;
	E.print_Matrix();
	cout << endl;
	cout << "------VT------" << endl;
	VT.print_Matrix();
	cout << endl;
	cout << "------Moriginal------" << endl;
	m.print_Matrix();
	cout << endl;
	U.trimMatrix();
	E.trimMatrix();
	VT.trimMatrix();
	Matrix res = U * E * VT;
	res.trimMatrix();
	cout << "------U*E*VT------" << endl;
	res.print_Matrix();

    cout<<endl;
    pair<Matrix<double>,Matrix<double>> lu = m.LU();
    cout<<"------L------"<<endl;
    lu.first.print_Matrix();
    cout<<endl;
    cout<<"------U------"<<endl;
    lu.second.print_Matrix();
    cout<<endl;
    Matrix<double> o = lu.first*lu.second;
    cout<<"------original------"<<endl;
    o.print_Matrix();
}
