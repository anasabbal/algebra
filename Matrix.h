#ifndef _MATRIX_H
#define _MATRIX_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <array>
#include <climits>
#include <string>
#include <type_traits>


namespace algebra
{
    using namespace std;

    template<typename T>
    concept floating = is_floating_point_v<T>;

    //forward declaration

    template<floating T>
    class Matrix;

    template<floating T>
    Matrix<T> concat_cols(Matrix<T> a,Matrix<T> b);

    template<floating T>
    Matrix<T>  concat_rows(Matrix<T> a,Matrix<T> b);

    template<floating T>
    T max(Matrix<T> m);

    template<floating T>
    bool cmp(pair<int,T>& p1,pair<int,T>& p2);

    template<floating T>
    void jacobi(T alpha,T beta,T gamma,Matrix<T>& G,T& t);



    template<floating T>
    class Matrix
    {

        private:

        T** data;
        int rows;
        int cols;

        public:

        Matrix(){
            rows = 0;
            cols = 0;
            data = nullptr;
        }
        Matrix(int r,int c,T element=0);
        Matrix(const Matrix& M);
        Matrix(T** M,int r,int c);
        Matrix(const vector<vector<T>>& M);
        Matrix(T* vec,int n,string type = "column",bool remove_vec = true);
        Matrix(int n);

        ~Matrix();


        void clean();
        void print_Matrix() const;

        int get_rows() const {return rows;}
        int get_cols() const {return cols;}

        T* row_idx(int idx) const;
        T* col_idx(int idx) const;


        Matrix<T> operator+(const Matrix<T>& r) const;
        Matrix<T> operator-(const Matrix<T>& r) const;
        Matrix<T> operator*(const Matrix<T>& r) const;

        void operator*=(const Matrix<T> & r);
	    void operator+=(const Matrix<T> & r);
	    void operator-=(const Matrix<T> & r);
        //void operator/=(const Matrix<T> & r);

        Matrix<T> operator*(T num) const;
        Matrix<T> operator+(T num) const;
        Matrix<T> operator-(T num) const;
        Matrix<T> operator/(T num) const;

        void operator*=(T num);
	    void operator+=(T num);
	    void operator-=(T num);
	    void operator/=(T num);




        bool operator==(const Matrix<T>& r)const;
	    bool operator!=(const Matrix<T>& r)const;

	    Matrix<T>& operator=(const Matrix<T>& r);
	    T* operator[](int i)const { return data[i]; }


        void swap_rows(int idx_1,int idx_2);
        void swap_cols(int idx_1,int idx_2);

        Matrix<T> remove_row_col(int r,int c);
        Matrix<T> cofactor_Matrix();
        T determinant();
        Matrix<T> elewisesquare();

        Matrix<T> sum(int axis=0);
        int argmax(double* col, double& max, int n);

        Matrix<T> cliprows(int start,int end);
        Matrix<T> clipcols(int start,int end);

        void assigncol(Matrix<T> m,int col);
        void assignrow(Matrix<T> m,int row);

        void trimMatrix();
        T frobeniusNorm();

        Matrix<T> transpose();
        Matrix<T> inverse();

       
        pair<Matrix<T>,Matrix<T>> QR();
        pair<Matrix<T>,Matrix<T>> LU();
        array<Matrix<T>,3> svd();

    };

    template<floating T>
    Matrix<T>::Matrix(int n)
    {
        rows = n;
        cols = n;
        data = new T*[n];
        for(int i=0;i<n;i++)
        {
            data[i] = new T[n];
        }
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(i!= j)
                {
                    data[i][j] = 0;
                }else{
                    data[i][j] = 1;
                }
            }
        }
    }

    template<floating T>
    Matrix<T>::Matrix(int r,int c,T element)
    {
        rows = r;
        cols = c;

        data = new T*[r];

        for(int i=0;i<r;++i)
        {
            data[i] = new T[c];
        }

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                data[i][j] = element;
            }
        }
    }

    template<floating T>
    Matrix<T>::Matrix(T** M,int r,int c)
    {
        rows = r;
        cols = c;
        data = M;
    }

    template<floating T>
    Matrix<T>::Matrix(const vector<vector<T>>& M)
    {
        rows = M.size();
        cols = M[0].size();

        data = new T*[rows];

        for(int i=0;i<rows;++i)
        {
            data[i] = new T[cols];
        }

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                data[i][j] = M[i][j];
            }
        }
    }

    template<floating T>
    Matrix<T>::Matrix(const Matrix& M)
    {
        rows = M.get_rows();
        cols = M.get_cols();

        data = new T*[rows];

        for(int i=0;i<rows;++i)
        {
            data[i] = new T[cols];
        }

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                data[i][j] = M[i][j];
            }
        }
    }

    template<floating T>
    Matrix<T>::Matrix(T* vec,int n,string type,bool remove_vec)
    {
        if(type == "column")
        {
            data = new T*[n];

            for(int i=0;i<n;++i)
            {
                data[i] = new T[1];
                data[i][0] = vec[i];
            }



            rows = n;
            cols = 1;
        }else{

            data = new T*[1];
            data[0] = new T[n];

            for(int i=0;i<n;++i)
            {
                data[0][i]= vec[i];
            }

            rows = 1;
            cols = n;

        }

        if(remove_vec)
        {
            delete [] vec;
            vec = nullptr;
        }
    }

    template<floating T>
    T* Matrix<T>::row_idx(int idx) const
    {
        assert(idx <rows);
        T* vec = new T[cols];

        for(int i=0;i<cols;++i)
        {
            vec[i] = data[idx][i];
        }
        return vec;
    }

    template<floating T>
    T* Matrix<T>::col_idx(int idx) const
    {
        assert(idx < cols);

        T* vec = new T[rows];

        for(int i=0;i<rows;++i)
        {
            vec[i] = data[i][idx];
        }
        return vec;
    }

    template<floating T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& r) const
    {
        assert(rows == r.get_rows() && cols == r.get_cols());
        Matrix<T> result(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = data[i][j] + r[i][j];
            }
        }
        return result;
    }

     template<floating T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& r) const
    {
        assert(rows == r.get_rows() && cols == r.get_cols());
        Matrix<T> result(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = data[i][j] - r[i][j];
            }
        }
        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& r) const
    {
        assert(cols == r.get_rows());
        Matrix<T> result(rows,r.get_cols());

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<r.get_cols();++j)
            {

                for(int k=0;k<cols;++k)
                {
                    result[i][j] += data[i][k] * r[k][j];
                }
            }
        }
        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::operator*(T num) const
    {
        Matrix<T> result(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = num * data[i][j];
            }
        }

        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::operator+(T num) const
    {
        Matrix<T> result(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = num + data[i][j];
            }
        }

        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::operator-(T num) const
    {
        Matrix<T> result(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = data[i][j] - num ;
            }
        }

        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::operator/(T num) const
    {
        assert(num != 0);
        Matrix<T> result(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = data[i][j]/num;
            }
        }

        return result;
    }

    template<floating T>
    void Matrix<T>::operator+=(T num)
    {
        *this = *this +num;
    }
    template<floating T>
    void Matrix<T>::operator-=(T num)
    {
        *this = *this -num;
    }
    template<floating T>
    void Matrix<T>::operator*=(T num)
    {
        *this = *this *num;
    }
    template<floating T>
    void Matrix<T>::operator/=(T num)
    {
        *this = *this /num;
    }

    template<floating T>
    void Matrix<T>::operator*=(const Matrix<T>& r)
    {
        *this =*this * r;
    }

    template<floating T>
    void Matrix<T>::operator+=(const Matrix<T>& r)
    {
        *this += r;
    }

    template<floating T>
    void Matrix<T>::operator-=(const Matrix<T>& r)
    {
        *this = *this -r;
    }

    template<floating T>
    Matrix<T>& Matrix<T>::operator=(const Matrix<T>& m)
    {
        if(!(rows == m.get_rows() && cols == m.get_cols()))
        {
            clean();
            rows = m.get_rows();
            cols = m.get_cols();
            data = new T*[rows];
            for(int i=0;i<rows;++i)
            {
                data[i] = new T[cols];
            }
        }

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                data[i][j] = m[i][j];
            }
        }
        return *this;
    }



    template<floating T>
    bool Matrix<T>::operator==(const Matrix<T>& r)const
    {
        assert(rows == r.get_rows() && cols == r.get_cols());

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                if(data[i][j] != r[i][j])
                {
                    return false;
                }
            }
        }
        return false;
    }

    template<floating T>
    bool Matrix<T>::operator!=(const Matrix<T>& r) const
    {
        return !(*this == r);
    }

    template<floating T>
    void Matrix<T>::clean()
    {
        if(data != nullptr)return;

        for(int i=0;i<rows;++i)
        {
            delete[] data[i];
        }

        delete[] data;

        data = nullptr;
    }

    template<floating T>
    Matrix<T>::~Matrix()
    {
        clean();
    }

    template<floating T>
    void Matrix<T>::print_Matrix()const
    {
        for(int i =0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                cout << setprecision(2) << data[i][j] << "\t";
            }
            cout<<endl;
        }
    }

    template<floating T>
    void Matrix<T>::swap_rows(int idx_1,int idx_2)
    {

        assert(idx_1 < rows && idx_2 <rows);

        for(int i=0;i<cols;++i)
        {
            swap(data[idx_1][i] ,data[idx_2][i]);
        }
    }

    template<floating T>
    void Matrix<T>::swap_cols(int idx_1,int idx_2)
    {
        assert(idx_1 < cols && idx_2 < cols);
        for(int i=0;i<rows;++i)
        {
            swap(data[i][idx_1] , data[i][idx_2]);
        }
    }

    template<floating T>
    Matrix<T> Matrix<T>::remove_row_col(int r,int c)
    {
        assert(r<rows && c<cols);

        Matrix<T> result(rows-1,cols-1);
        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                if(i!= r && j!= c)
                {
                    int idx_i = i,idx_j = j;
                    if(i> r) idx_i--;
                    if(j>c) idx_j--;

                    result[idx_i][idx_j] = data[i][j];
                }

            }
        }

        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::cofactor_Matrix()
    {
        Matrix<T> result(rows,cols);
        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                Matrix<T> tmp = remove_row_col(i,j);
                result[i][j] = tmp.determinant()*pow(-1,i+j);
            }
        }
        return result;
    }

    //https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants

    template<floating T>
    T Matrix<T>::determinant()
    {
        assert(rows == cols);

        if(rows == 1)
        {
            return data[0][0];
        }else if(rows == 2)
        {
            return data[0][0]*data[1][1] - data[0][1]*data[1][0];
        }

        T det = T();
        int sign = 1;

        for(int i=0;i<cols;++i)
        {
            Matrix<T> mat = remove_row_col(0,i);
            det += sign * data[0][i]*mat.determinant();
            sign *= -1;
        }
        return det;


    }

    template<floating T>
    Matrix<T> Matrix<T>::elewisesquare()
    {
        Matrix<T> result(*this);
        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                result[i][j] = data[i][j] * data[i][j];
            }
        }
        return result;
    }

    template<floating T>
    Matrix<T> Matrix<T>::sum(int axis)
    {
        Matrix tmp;


        if(axis == 0)
        {
            tmp = Matrix(rows,1);
            for(int i=0;i<rows;++i)
            {
                for(int j=0;j<cols;++j)
                {
                    tmp[i][0] += data[i][j];
                }
            }
        }
        else if(axis == 1)
        {
            tmp = Matrix(1,cols);

            for(int i=0;i<rows;++i)
            {
                for(int j=0;j<cols;++j)
                {
                    tmp[0][j] += data[i][j];
                }
            }
        }
        else{

            tmp = Matrix(1,1);

            for(int i=0;i<rows;++i)
            {
                for(int j=0;j<cols;++j)
                {
                    tmp[0][0] += data[i][j];
                }
            }
        }
        return tmp;
    }

    template<floating T>
    Matrix<T> Matrix<T>::cliprows(int start,int end)
    {
        assert(start < rows && end<rows);

        Matrix<T> tmp(end-start+1,cols);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                tmp[i-start][j] = data[i][j];
            }
        }

        return tmp;

    }

    template<floating T>
    Matrix<T> Matrix<T>::clipcols(int start,int end)
    {
        assert(start < cols && end<cols);

        Matrix<T> tmp(rows,end-start+1);

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                tmp[i][j-start] = data[i][j];
            }
        }

        return tmp;

    }

    template<floating T>
    void Matrix<T>::assignrow(Matrix<T> m,int row)
    {
        if(m.get_rows() != 1)
        {
            cout<< "Warning, mat is not a row vector" << endl;
            return;
        }

        for(int i=0;i<cols;++i)
        {
            data[row][i] = m[0][i];
        }
    }

    template<floating T>
    void Matrix<T>::assigncol(Matrix<T> m,int col)
    {
        if(m.get_cols() != 1)
        {
            cout<< "Warning, mat is not a column vector" << endl;
            return;
        }

        for(int i=0;i<rows;++i)
        {
            data[i][col] = m[i][0];
        }
    }

    template<floating T>
    void Matrix<T>::trimMatrix()
    {
        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                double s = abs(static_cast<double>(data[i][j]));
                if(s<1e-11)
                {
                    data[i][j] = 0;
                }
            }
        }
    }

    template<floating T>
    T Matrix<T>::frobeniusNorm()
    {
        return sqrt(elewisesquare().sum(-1)[0][0]);
    }


    template<floating T>
    Matrix<T> Matrix<T>::transpose()
    {
        Matrix<T> t(cols,rows);
        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                t[j][i] = data[i][j];
            }
        }
        return t;
    }

    template<floating T>
    Matrix<T> Matrix<T>::inverse()
    {
        double  det = static_cast<double>(determinant());
        if(det == 0)
        {
            cout<< "Matrix is singular" << endl;
            return Matrix<T>();
        }

        Matrix<T> inv = cofactor_Matrix().transpose() / (det);
        return inv;
    }

    

    template<floating T>
    int Matrix<T>::argmax(double* col, double & max, int n)
    {
        int idx = -1;
        for(int i=0;i<n;++i)
        {
            if(max < abs(col[i]))
            {
                max = abs(col[i]);
                idx = i;
            }
        }
        return idx;
    }

    //http://www.math.iit.edu/~fass/477577_Chapter_4.pdf

    template<floating T>
    pair<Matrix<T>,Matrix<T>> Matrix<T>::QR()
    {
        assert(determinant() != 0);
        Matrix<T> Q(rows,cols);
        Matrix<T> R(rows,cols);

        vector<Matrix<T>> us(cols);
        vector<Matrix<T>> es(cols);

        for(int i=0;i<cols;++i)
        {
            us[i] = Matrix<T>(col_idx(i),rows,string("row"));
            for(int j=i;j>0;--j)
            {
                Matrix<T> tmp = Matrix<T>(col_idx(i),rows,string("row"))*es[j-1].transpose();
                us[i] -= es[j-1] * tmp[0][0];
            }
            es[i] = us[i];
            es[i] /= sqrt((us[i] * us[i].transpose())[0][0]);
        }

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                Q[i][j] = es[j][0][i];
            }
        }

        for(int i=0;i<rows;++i)
        {
            for(int j=0;j<cols;++j)
            {
                Matrix<T> tmp = Matrix<T>(col_idx(j),rows,string("row"))*es[i].transpose();
                R[i][j] = tmp[0][0];
            }
        }
        Q.trimMatrix();
        R.trimMatrix();
        return make_pair(Q,R);


    }



    template<floating T>
    Matrix<T> Diag(Matrix<T>& m)
    {

        if(m.get_rows() != 1 && m.get_cols() != 1)
        {
            cout<< "Warning, matrix is not one dimensional" << endl;
        }

        int n = std::max(m.get_rows(),m.get_cols());
        Matrix<T> tmp(n,n);
        int k =0;

        for(int i=0;i<m.get_rows();++i)
        {
            for(int j=0;j<m.get_cols();++j)
            {
                tmp[k][k] = m[i][j];
                k++;
            }
        }
        return tmp;

    }

    template<floating T>
    Matrix<T> Diag(vector<T>& vec)
    {
        int n = vec.size();
        Matrix<T> tmp(n,n);
        for(int i=0;i<n;++i)
        {
            tmp[i][i] = vec[i];
        }
        return tmp;
    }



    template<floating T>
    Matrix<T> concat_rows(Matrix<T> a,Matrix<T> b)
    {
        assert(a.get_cols() == b.get_cols());
        Matrix<T> result(a.get_rows() + b.get_rows(),a.get_cols());

        for(int i=0;i<a.get_rows();++i)
        {
            for(int j=0;j<a.get_cols();++j)
            {
                result[i][j] = a[i][j];
            }
        }

        for(int i=a.get_rows();i<a.get_rows() + b.get_rows();++i)
        {
            for(int j=0;j<a.get_cols();++j)
            {
                result[i][j] = b[i-a.get_rows()][j];
            }
        }
        return result;
    }

    template<floating T>
    Matrix<T> concat_cols(Matrix<T> a,Matrix<T> b)
    {
        assert(a.get_rows() == b.get_rows());
        Matrix<T> result(a.get_rows(),a.get_cols() + b.get_cols());
        for(int i=0;i<a.get_rows();++i)
        {
            for(int j=0;j<a.get_cols();++j)
            {
                result[i][j] = a[i][j];
            }
            for(int j=a.get_cols();j<a.get_cols() + b.get_cols();++j)
            {
                result[i][j] = b[i][j-a.get_cols()];
            }

        }

        return result;
    }

    template<floating T>
    T max(Matrix<T> m)
    {
        T d = INT8_MIN;;

        for(int i=0;i<m.get_rows();++i)
        {
            for(int j=0;j<m.get_cols();++j)
            {
                if(d < m[i][j])d=m[i][j];

            }
        }
        return d;
    }



    template<floating T>
    pair<Matrix<T>,Matrix<T>> Matrix<T>::LU()
    {
        assert(rows == cols);
        Matrix<T> L(rows,cols);
        Matrix<T> U(rows,cols);

        for(int i=0;i<rows;++i)
        {
            for(int k=i;k<rows;++k)
            {
                T sum = T();
                for(int j=0;j<i;++j)
                {
                    sum += L[i][j]* U[j][k];
                }
                U[i][k] = data[i][k] - sum;
            }

            for(int k=i;k<rows;++k)
            {
                if(i==k)
                {
                    L[i][i] = 1;
                }else{
                    T sum = T();
                    for(int j=0;j<i;++j)
                    {
                        sum += L[k][j] * U[j][i];
                    }
                    L[k][i] = (data[k][i] -sum)/U[i][i];
                }
            }
        }
        return make_pair(L,U);

    }

    //https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf

    template<floating T>
    array<Matrix<T>,3> Matrix<T>::svd()
	{
		Matrix<T> copy = *this;
		Matrix<T> U(rows,cols),sigma = elewisesquare().sum(1);
        Matrix<T> V(cols);
		double eps = 1e-15;
		double rots = 1;
		double tols = eps*frobeniusNorm();

		while(rots >= 1)
		{
			rots = 0;

			for(int p=0;p<cols;++p)
			{
				for(int q=p+1;q<cols;++q)
				{
					double beta = (Matrix<T>(copy.col_idx(p),rows,"rows")*Matrix<T>(copy.col_idx(q),rows,"column"))[0][0];
					if(sigma[0][p]*sigma[0][q]>tols && abs(beta)>=eps*sqrt(sigma[0][p] * sigma[0][q]))
					{
						rots++;
						Matrix<T> G;
						double t;
						jacobi(sigma[0][p],beta,sigma[0][q],G,t);
						sigma[0][p] = sigma[0][p] - beta*t;
						sigma[0][q] = sigma[0][q] + beta*t;
						Matrix<T> temp = concat_cols(Matrix<T>(copy.col_idx(p), rows),Matrix<T>(copy.col_idx(q), rows))*G;

						copy.assigncol(Matrix(temp.col_idx(0),copy.get_rows()),p);
						copy.assigncol(Matrix(temp.col_idx(1),copy.get_rows()),q);
						Matrix<T> temp2 = concat_cols(Matrix<T>(V.col_idx(p),rows),Matrix<T>(V.col_idx(q),rows))*G;
						V.assigncol(Matrix(temp2.col_idx(0), V.get_rows()),p);
					    V.assigncol(Matrix<T>(temp2.col_idx(1), V.get_rows()),q);
					}
				}
			}
		}

		Matrix<T> copy_v = V;
		vector<pair<int,T>> id(sigma.get_cols());

	    for(int i=0;i<sigma.get_cols();++i)
		{
			id[i] = make_pair(i,sigma[0][i]);
		}
		std::sort(id.begin(),id.end(),cmp<T>);

		for(int i=0;i<U.get_rows();++i)
		{
			for(int j=0;j<id.size();++j)
			{
				sigma[0][j] = id[j].second;
				U[i][j] = copy[i][id[j].first];
			}
		}

		for(int i=0;i<copy_v.get_rows();++i)
		{
			for(int j=0;j<id.size();++j)
			{
				V[i][j] = copy_v[i][id[j].first];
			}
		}

		for(int k=0;k<cols;++k)
		{
			if(sigma[0][k] == 0)
			{
				for(int j=k;j<cols;++j)
				{
					sigma[0][j] = 0;
				}
			}
			sigma[0][k] = sqrt(abs(sigma[0][k]));
			for(int i =0;i<rows;++i)
			{
				U[i][k] /= sigma[0][k];
			}
		}

		array<Matrix<T>,3> result;
		result[0] = U;
		result[1] = Diag(sigma);
		result[2] = V.transpose();
		return result;
	}

    template<floating T>
    bool cmp(pair<int,T>& p1,pair<int,T>& p2)
    {
        return p1.second>p2.second;
    }

    template<floating T>
    void jacobi(T alpha,T beta,T gamma,Matrix<T>& G,T& t)
    {
        T c = 1, s=0;
	    t = 0;
	    G = Matrix<T>(2, 2);
	    if (beta != 0) {
		    T tau = (gamma - alpha) / (2 * beta);
		    if (tau >= 0) {
			   t = 1 / (tau + sqrt(1 + tau * tau));
		    }
		    else {
			   t = -1 / (-tau + sqrt(1 + tau * tau));
		    }
		    c = 1 / sqrt(1 + t * t);
		    s = t*c;
	    }
	    G[0][0] = c;
	    G[0][1] = s;
	    G[1][0] = -s;
	    G[1][1] = c;
	    s = t;
    }

}

#endif
