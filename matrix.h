#pragma once
#include <string.h>
#include <vector>
#include <valarray>
#include <iostream>
#include <assert.h>
using namespace std;
//define a vector: provide storage and some convenient initialization
//similarly to matlab usage
//new operator @ unary; //new operators are not allowed in C++
template <typename T> class matrix;
template <typename T> class vec;
const float eps=1.0e-6;
const double deps=1.0e-14;
#define SIGN(a) ((a)>=0?1:-1)
//vec is just a vector, mainly used for indexing
//if you want to use a math vector, may use it as a matrix

//slice matrix provide left/right usage, it reference elements to the matrix it points to
//follow convention: dim1 is number of row, dim2 is num_col
//this is contradictory to the C arrange [m]x[n], m is dim1, n is dim2
template <typename T> 
class slice_matrix
{
	matrix<T>& m;
	vec<int> row_indx;
	vec<int> col_indx;
public:
	slice_matrix(matrix<T>& mm,const vec<int>& rd,const vec<int>& cd);
	slice_matrix(matrix<T>& mm,const vec<int>& rd, const char* s);
	slice_matrix(matrix<T>& mm,const char* s, const vec<int>& cd);
	slice_matrix(matrix<T>& mm,const char* s, const char* s1);
	//supporting: a(ind1,ind2) as a whole
	//1. b=a(ind1,ind2); 2. a(ind1,ind2)=scalar; 3: a(ind1,ind2)=matrix; 4: a(ind1,ind2)=b(ind10,ind20);
	//a(ind1,ind2) in arithmetics 
	slice_matrix<T>& operator=(T t);
	slice_matrix<T>& operator=(const matrix<T>& mm);
	int dim1() const {return row_indx.size();}
	int dim2() const {return col_indx.size();}
	const T& operator()(int i,int j) const {return m(row_indx[i],col_indx[j]);} //only RHS

	const matrix<T>& get_matrix() const {return m;}

	//also slice matrix needs arithmetics operation, can only return matrix
	
	matrix<T> operator+(T val);
	matrix<T> operator-(T val);
	matrix<T> operator*(T val);
	matrix<T> operator/(T val);
	matrix<T> operator%(T val);
	
	//slice +-*/ can only return a matrix!!!
	matrix<T> operator+(const slice_matrix<T>& sm);
	matrix<T> operator-(const slice_matrix<T>& sm);
	matrix<T> operator*(const slice_matrix<T>& sm);
	matrix<T> operator/(const slice_matrix<T>& sm);
	matrix<T> operator%(const slice_matrix<T>& sm);

	slice_matrix<T>& operator+=(T val);
	slice_matrix<T>& operator-=(T val);
	slice_matrix<T>& operator*=(T val);
	slice_matrix<T>& operator/=(T val);
	slice_matrix<T>& operator%=(T val);

	void print();
	template <typename T> friend ostream& operator<<(ostream&,const slice_matrix<T>&); //have to add template, it is not a member function!
};
//define a matrix
template <typename T>
class ind_array
{
	matrix<T>& m;
	vector<int> ind; //better use matrix<T>, cannot use reference since the ind might be temp object
public:
	ind_array(matrix<T>& mm, const vector<int>& indx):m(mm),ind(indx){}
	int size() const {return ind.size();}
	int operator[](int i) {return ind[i];}
	ind_array<T>& operator=(T t);
};

template <typename T>
class bool_array
{
	matrix<T>& m;
	//vector<bool> ind;
	matrix<bool>& ind;
public:
	bool_array(matrix<T>& mm, const matrix<bool>& indx):m(mm),ind(indx){}
	int size() const {return ind.size();}
	int operator[](int i) {return ind[i];}
	bool_array<T>& operator=(T t);
};

template <class T>
class matrix
{
protected:
	int nrow,ncol;
	T *pdata;
public:
	//ctor & d_ctor
	matrix() {nrow=ncol=0;pdata=0;}
	explicit matrix(int len) {nrow=len;ncol=1;pdata=new T[len];memset(pdata,0,ncol*nrow*sizeof(T));}
	matrix(int r,int c) {nrow=r;ncol=c;pdata=new T[nrow*ncol];memset(pdata,0,ncol*nrow*sizeof(T));}
	matrix(T* p,int nr,int nc=1) {nrow=nr;ncol=nc;pdata=new T[ncol*nrow];memcpy(pdata,p,nrow*ncol*sizeof(T));}
	matrix(T* p,int nr,int nc,int step);
	matrix(const matrix<T>& m) {nrow=m.dim1();ncol=m.dim2();pdata=new T[ncol*nrow];memcpy(pdata,m.data(),ncol*nrow*sizeof(T));}
	matrix(const slice_matrix<T>& sm); //assign a slice matrix to another matrix
	matrix(const vec<T> &v); //convert a vector to a matrix
	matrix(const vector<vector<T>>& v2d);
	matrix(const valarray<valarray<T>>& v2d);
	template <typename C> matrix(const matrix<C>& m);
	template <typename T, int m,int n>
	matrix(T (&arr)[m][n]) {nrow=m;ncol=n;pdata=new T[nrow*ncol];memcpy(pdata,arr,nrow*ncol*sizeof(T));}
	~matrix() {delete []pdata;}
	matrix<T>& operator=(const matrix<T>&); //assignment
	matrix<T>& operator=(const slice_matrix<T>& sm);
	matrix<T>& operator=(T);
	matrix<T>& operator=(const vector<vector<T>>& v2d);
	matrix<T>& operator=(const valarray<valarray<T>>& v2d);
	template <typename C> matrix<T>& operator=(const matrix<C>& m);
	//matrix info
	int dim1() const {return nrow;}
	int dim2() const {return ncol;}
	int length() const {return nrow*ncol;}
	const T* data() const {return pdata;}
	T* data() {return pdata;}
	//element, row, col access
	T& operator()(int i,int j) {return pdata[i*ncol+j];}
	const T& operator()(int i,int j) const {return pdata[i*ncol+j];}
	//random element access using ind_array
	ind_array<T> operator()(const vector<int>& ind) {return ind_array<T>(*this,ind);}
	bool_array<T> operator()(const vector<bool>& ind) {return bool_array<T>(*this,ind);}
	//slice a matrix, generally a fixed sized to a dynamic sized matrix
	//using two vector<int> for the row, col selection
	slice_matrix<T> operator()(const vec<int>& ri,const vec<int>& ci);
	slice_matrix<T> operator()(const vec<int>& ri,const char*);
	slice_matrix<T> operator()(const char*,const vec<int>& ci);
	slice_matrix<T> operator()(const char*,const char*);
	matrix<T> operator()(const char*) //directly convert to a row vector
	{
		if(is_vec(*this)) return *this;
		matrix<T> tmp=*this;
		return tmp.reshape(1,length());
	}
	//const slice_matrix<T>& operator(const vec<ind>& ri,const vec<ind>& ci) const;

	//row and column access: they are special cases of slice_matrix
	slice_matrix<T> col(int i) {assert(i<dim2());return slice_matrix<T>(*this,"",i);} 
	slice_matrix<T> row(int i) {assert(i<dim1());return slice_matrix<T>(*this,i,"");}
	matrix<T>& row_swap(int i,int j) {assert(i<dim1() && j<dim1() && i!=j);slice_matrix<T> s=row(i);row(i)=row(j);row(j)=s;}
	matrix<T>& col_swap(int i,int j) {assert(i<dim2() && j<dim2() && i!=j);slice_matrix<T> s=col(i);col(i)=col(j);col(j)=s;}
	//bool operations, index operations, such as find, and relational expressions

	//matrix arithmetic operations
	matrix<T>& operator-();
	//elementwise operations
	matrix<T> operator+(const matrix<T>& m); //better include hyrbid operations here
	matrix<T> operator-(const matrix<T>& m);
	matrix<T> operator*(const matrix<T>& m);
	matrix<T> operator/(const matrix<T>& m);
	matrix<T> operator%(const matrix<T>& m);
	matrix<T>& operator+=(const matrix<T>& m);
	matrix<T>& operator-=(const matrix<T>& m);
	matrix<T>& operator*=(const matrix<T>& m);
	matrix<T>& operator/=(const matrix<T>& m);
	matrix<T>& operator%=(const matrix<T>& m);
	//matrix scalar operations
	matrix<T> operator+(T t) const;
	matrix<T> operator-(T t) const;
	matrix<T> operator*(T t) const;
	matrix<T> operator/(T t) const;
	matrix<T> operator%(T t);
	matrix<T>& operator+=(T t);
	matrix<T>& operator-=(T t);
	matrix<T>& operator*=(T t);
	matrix<T>& operator/=(T t);
	matrix<T>& operator%=(T t);

	//logical relation
	matrix<bool> operator>(T t);
	matrix<bool> operator<(T t);
	matrix<bool> operator==(T t);
	matrix<bool> operator!=(T t);
	matrix<bool> operator>=(T t);
	matrix<bool> operator<=(T t);

	//matrix operations use the ^ for matrix multiplication
	matrix<T> operator^(const matrix<T>& m);
	//matrix<T>& opertor^=(const matrix<T>& m); //not needed since it will have to create a new matrix
	//matrix<T>& operator^=(const matrix<T>& m); //this is not allowed
	matrix<T>& reshape(int nr,int nc) 
	{
		if(nr*nc!=length()) throw "size does not match";
		nrow=nr;ncol=nc;
	}
	void print();

};

//vec is a special matrix with 1xn or nx1 dimensions
//can use matrix all operations to avoid duplicate codes
//main purpose to provide some convenient constructing tools
//default vector is a row vector (one row, multiple cols)
//template <typename T,int n> int size(T (&arr)[n]) {return n;} //calculate the regular c++ array size passing an array
template <typename T> class vec: public matrix<T>
{
private:
	vec<T>& reshape(int m,int n) {return *this;}
public:
	vec(){}
	vec(T t) {nrow=ncol=1;pdata=new T[1];pdata[0]=t;} //implicit convert a scalar to an vector
	vec(T* p,int m) {nrow=1;ncol=m;pdata=new T[nrow*ncol];memcpy(pdata,p,m*sizeof(T));}
	vec(const vec<T>& v) {nrow=v.dim1();ncol=v.dim2();pdata=new T[nrow*ncol];memcpy(pdata,v.data(),ncol*nrow*sizeof(T));}
	vec(const matrix<T>& m) //implicit convert a matrix into a vector
	{
		if(is_vec(m)) {nrow=m.dim1();ncol=m.dim2();}
		else {nrow=1;ncol=m.length();}
		pdata=new T[nrow*ncol];memcpy(pdata,m.data(),nrow*ncol*sizeof(T));
	} //also used for implicit conversion
	vec(const vector<T>& v);
	vec(const valarray<T>& v);
	vec(int len,T t) {nrow=1;ncol=len;pdata=new T[nrow*ncol];for(int i=0;i<len;i++) pdata[i]=t;}
	~vec() {} //call base automatically
	template <typename T,int n> 
	vec(T (&arr)[n]) {nrow=1;ncol=n;pdata=new T[nrow*ncol];memcpy(pdata,&arr,nrow*ncol*sizeof(T));}
	vec<T>& operator=(const vec<T>& v) 
	{
		if(this==&v) return *this;
		delete []pdata;
		nrow=v.dim1();
		ncol=v.dim2();
		pdata=new T[nrow*ncol];
		memcpy(pdata,v.data(),nrow*ncol*sizeof(T));
		return *this;
	}
	
	vec(T start,T step,T end)
	{
		//int num=(end-start+step+eps)/step;
		T dist=end-start+step;
		dist+=SIGN(dist)*eps;
		int num=dist/step;
		nrow=1;ncol=num;
		if(num>0)
		{
			pdata=new T[num];
			for(int i=0;i<num;i++) pdata[i]=start+i*step;
		}
	}
	int size() const {return length();}
	T& operator[](int i) {return pdata[i];} //LHS
	const T& operator[](int i) const {return pdata[i];} //RHS
	//vector operations, note vector is a special matrix. 
	//it can inherit matrix's element-wise and matrix-wise operations
	vec<T>& transpose() {int t=nrow;nrow=ncol;ncol=t;return *this;} //no storage change
	//slice operations
	slice_matrix<T> operator()(const vec<int>& ri) 
	{
		if(ncol==1)	return slice_matrix<T>(*this,ri,0);
		else return slice_matrix<T>(*this,vec<int>(0),ri); //0 can be pointer or integer type, will not call implicit conversion
	}

	//slice_matrix<T> operator()(const char*);	//for all, not needed
	//void print();
};
//matrix operations: inverse, rotate, transpose, fliplr, flipud
template <typename T> matrix<T> operator+(T, const matrix<T>&);
template <typename T> matrix<T> operator-(T, const matrix<T>&);
template <typename T> matrix<T> operator*(T, const matrix<T>&);
template <typename T> matrix<T> operator/(T, const matrix<T>&);
template <typename T> matrix<T> operator%(T, const matrix<T>&);
template <typename T> matrix<T> abs(const matrix<T>& m);
//friend matrix<T> operator^(const matrix<T>& m,int i);
template <typename T> matrix<T> transpose(const matrix<T>& m);
vector<int> find(const matrix<bool>& m);
template <typename T> matrix<T> reshape(const matrix<T>& m,int nr,int nc);
template <typename T> matrix<T> operator^(const matrix<T>& m1,const matrix<T>& m2);
template <typename T> double dot(const vec<T>& v1, const vec<T>& v2);
template <typename T> bool is_vec(const matrix<T>& m) {return m.dim1()==1 || m.dim2()==1;}
template <typename T> ostream& operator<<(ostream& out, const matrix<T>& m);
template <typename T> matrix<T> conj(const matrix<T>&m);
template <typename T> matrix<T> eye(T val,int n); //build an I matrix
template <typename T> matrix<T> power(const matrix<T>& m, int n); //only for square matrix
template <typename T> matrix<T> cofactor(const matrix<T>& m,int p,int q,int n);
template <typename T> double det(const matrix<T>& m,int n);
template <typename T> matrix<T> adjoint(const matrix<T>& m);
template <typename T> matrix<T> inv(const matrix<T>& A);
template <typename T> matrix<cplx> complex(const matrix<T>& m);
//template <typename T> ostream& operator<<(ostream&,const slice_matrix<T>&);
#include "matrix.cpp"