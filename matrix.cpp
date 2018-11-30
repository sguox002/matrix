
//****************Slice_matrix**********************************
template <typename T>
slice_matrix<T>::slice_matrix(matrix<T>& mm,const vec<int>& rd,const vec<int>& cd):
    m(mm),row_indx(rd),col_indx(cd) {}

template <typename T>
slice_matrix<T>::slice_matrix(matrix<T>& mm,const vec<int>& rd, const char* s):
	m(mm),row_indx(rd),col_indx(vec<int>(0,1,mm.dim2()-1)){}

template <typename T>
slice_matrix<T>::slice_matrix(matrix<T>& mm,const char* s, const vec<int>& cd):
	m(mm),row_indx(vec<int>(0,1,mm.dim1()-1)),col_indx(cd){}

template <typename T>
slice_matrix<T>::slice_matrix(matrix<T>& mm,const char* s, const char* s1):
	m(mm),row_indx(vec<int>(0,1,mm.dim1()-1)),col_indx(vec<int>(0,mm.dim2()-1)){}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator=(T t)
{
    for(int i=0; i<dim1(); i++) //repeat row
    {
        for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
            m(row_indx[i],col_indx[j])=t;
    }

    return *this;
}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator=(const matrix<T>& mm)
{
    if(dim1()!=mm.dim1() || dim2()!=mm.dim2())
        throw "matrix size does not match";

    for(int i=0; i<dim1(); i++) //repeat row
    {
        for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
            m(row_indx[i],col_indx[j])=mm(i,j);
    }
}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator+=(T t)
{
    //slice matrix could be rows, cols, or sub-matrix
    for(int i=0; i<dim1(); i++) //repeat row
    {
        for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
            m(row_indx[i],col_indx[j])+=t;
    }

    return *this;
}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator-=(T t)
{
    //slice matrix could be rows, cols, or sub-matrix
    for(int i=0; i<dim1(); i++) //repeat row
    {
        for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
            m(row_indx[i],col_indx[j])-=t;
    }

    return *this;
}

//******************Implementation for ind_array************************

template <typename T>
ind_array<T>& ind_array<T>::operator=(T t)
{
    for(int i=0; i<size(); i++) m.data()[ind[i]]=t;

    return *this;
}

///////////////////Implementation for bool_array*************************
template <typename T>
bool_array<T>& bool_array<T>::operator=(T t)
{
    for(int i=0; i<size(); i++)
        if(ind[i]) m.data()[i]=t;

    return *this;
}

template <typename T>
matrix<T> slice_matrix<T>::operator+(T val) 
{
    matrix<T> tmp=*this;
    return tmp+=val;
}
template <typename T>
matrix<T> slice_matrix<T>::operator-(T val) 
{
    matrix<T> tmp=*this;
    return tmp-=val;
}

template <typename T>
matrix<T> slice_matrix<T>::operator*(T val) 
{
    matrix<T> tmp=*this;
    return tmp*=val;
}

template <typename T>
matrix<T> slice_matrix<T>::operator/(T val) 
{
    matrix<T> tmp=*this;
    return tmp/=val;
}

template <typename T>
matrix<T> slice_matrix<T>::operator%(T val)
{
    slice_matrix<T> tmp=*this;
    return tmp%=val;
}

template <typename T>
matrix<T> slice_matrix<T>::operator+(const slice_matrix<T>& sm)
{
    if(dim1()!=sm.dim1() || dim2()!=sm.dim2())
        throw "dimension mismatch";

    matrix<T> tmp(*this);
	for(int i=0;i<dim1();i++)
		for(int j=0;j<dim2();j++) tmp(i,j)+=sm(i,j);

	return tmp;
}
template <typename T>
matrix<T> slice_matrix<T>::operator-(const slice_matrix<T>& sm)
{
    if(dim1()!=sm.dim1() || dim2()!=sm.dim2())
        throw "dimension mismatch";

    matrix<T> tmp(*this);
	for(int i=0;i<dim1();i++)
		for(int j=0;j<dim2();j++) tmp(i,j)-=sm(i,j);
    return tmp;
}
template <typename T>
matrix<T> slice_matrix<T>::operator*(const slice_matrix<T>& sm)
{
    if(dim1()!=sm.dim1() || dim2()!=sm.dim2())
        throw "dimension mismatch";

    matrix<T> tmp(*this);
	for(int i=0;i<dim1();i++)
		for(int j=0;j<dim2();j++) tmp(i,j)*=sm(i,j);

    return tmp;
}
template <typename T>
matrix<T> slice_matrix<T>::operator/(const slice_matrix<T>& sm)
{
    if(dim1()!=sm.dim1() || dim2()!=sm.dim2())
        throw "dimension mismatch";

    matrix<T> tmp(*this);
	for(int i=0;i<dim1();i++)
		for(int j=0;j<dim2();j++) tmp(i,j)/=sm(i,j);

    return tmp;
}
template <typename T>
matrix<T> slice_matrix<T>::operator%(const slice_matrix<T>& sm)
{
    if(dim1()!=sm.dim1() || dim2()!=sm.dim2())
        throw "dimension mismatch";

    slice_matrix<T> tmp(*this);
	for(int i=0;i<dim1();i++)
		for(int j=0;j<dim2();j++) tmp(i,j)%=sm(i,j);

    return tmp;
}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator*=(T t)
{
    //slice matrix could be rows, cols, or sub-matrix

        for(int i=0; i<dim1(); i++) //repeat row
        {
            for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
                m(row_indx[i],col_indx[j])*=t;
        }

    return *this;
}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator/=(T t)
{
    //slice matrix could be rows, cols, or sub-matrix
        for(int i=0; i<dim1(); i++) //repeat row
        {
            for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
                m(row_indx[i],col_indx[j])/=t;
        }

    return *this;
}

template <typename T>
slice_matrix<T>& slice_matrix<T>::operator%=(T t)
{
    //slice matrix could be rows, cols, or sub-matrix
        for(int i=0; i<dim1(); i++) //repeat row
        {
            for(int j=0; j<dim2(); j++) //m[i*m.dim1()+j]=t;
                m(row_indx[i],col_indx[j])%=t;
        }

    return *this;
}

template <typename T>
template <typename C>
matrix<T>::matrix(const matrix<C>& m)
{
	nrow=m.dim1();
	ncol=m.dim2();
	pdata=new T[nrow*ncol];
	for(int i=0;i<nrow*ncol;i++) pdata[i]=m.data()[i];
}

template <typename T>
template <typename C>
matrix<T>& matrix<T>::operator=(const matrix<C>& m)
{
	if(length()!=m.length())
	{
		delete []pdata;
	}
	nrow=m.dim1();
	ncol=m.dim2();
	pdata=new T[nrow*ncol];
	for(int i=0;i<nrow*ncol;i++) pdata[i]=m.data()[i];
	return *this;
}

template <typename T>
matrix<T>::matrix(const slice_matrix<T>& sm)
{
    //if(*this==sm.get_matrix()) return;
    //if(this==&sm.get_matrix()) return;
    //delete []pdata;
    nrow=sm.dim1();//note nrow, ncol could be zero
    ncol=sm.dim2();
    pdata=new T[nrow*ncol];

    for(int i=0; i<nrow; i++)
    {
        for(int j=0; j<ncol; j++)
            pdata[i*ncol+j]=sm(i,j);
    }
}
template <typename T>
matrix<T>::matrix(const vector<vector<T>>& v2d)
{
	nrow=v2d.size();
	ncol=v2d[0].size();
	pdata=new T[nrow*ncol];
	memcpy(pdata,&v2d[0][0],nrow*ncol*sizeof(T));
}
template <typename T>
matrix<T>::matrix(const valarray<valarray<T>>& v2d)
{
	nrow=v2d.size();
	ncol=v2d[0].size();
	pdata=new T[nrow*ncol];
	memcpy(pdata,&v2d[0][0],nrow*ncol*sizeof(T));
}

template <typename T>
matrix<T>& matrix<T>::operator=(const matrix<T>& mm)
{
    if(this==&mm) return *this;
	if(length()!=mm.length()) //if length the same, avoid delete and new
	{
		delete []pdata;
		pdata=new T[mm.length()];
	}
	nrow=mm.dim1();
    ncol=mm.dim2();
    memcpy(pdata,mm.data(),nrow*ncol*sizeof(T));
    return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator=(const slice_matrix<T>& sm)
{
    //if(*this==sm.get_matrix()) return;
    if(this==&sm.get_matrix()) return *this;

    delete []pdata;
    nrow=sm.dim1();
    ncol=sm.dim2();
    pdata=new T[nrow*ncol];

    for(int i=0; i<nrow; i++)
    {
        for(int j=0; j<ncol; j++)
            pdata[i*ncol+j]=sm(i,j);
    }

    return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator=(T t)
{
	T* p=data();
	for(int i=0;i<length();i++) p[i]=t;
}

template <typename T>
matrix<T>& matrix<T>::operator=(const vector<vector<T>>& v2d)
{
	if(length()!=v2d.size()*v2d[0].size())
	{
		delete []pdata;
		pdata=new T[nrow*ncol];
	}
	memcpy(pdata,&v[0][0],nrow*ncol*sizeof(T));
	return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator=(const valarray<valarray<T>>& v2d)
{
	if(length()!=v2d.size()*v2d[0].size())
	{
		delete []pdata;
		pdata=new T[nrow*ncol];
	}
	memcpy(pdata,&v[0][0],nrow*ncol*sizeof(T));
	return *this;
}
template <typename T>
slice_matrix<T> matrix<T>::operator()(const vec<int>& ri,const vec<int>& ci)
{
    return slice_matrix<T>(*this,ri,ci);
}
template <typename T>
slice_matrix<T> matrix<T>::operator()(const vec<int>& ri,const char* s)
{
	return slice_matrix<T>(*this,ri,s);
}
template <typename T>
slice_matrix<T> matrix<T>::operator()(const char* s,const vec<int>& ci)
{
	return slice_matrix<T>(*this,s,ci);
}
template <typename T>
slice_matrix<T> matrix<T>::operator()(const char*s,const char*s1)
{
	return slice_matrix<T>(*this,s,s1);
}

/*
template <typename T>
void vec<T>::print()
{
    //for(int i=0; i<length(); i++) cout<<pdata[i]<<"\t";
	for(int i=0;i<dim1();i++)
	{
		for(int j=0;j<dim2();j++)
			cout<<pdata[i*dim2()+j]<<"\t";
		cout<<endl;
	}
    cout<<endl;
}
*/

template <typename T>
void matrix<T>::print()
{
    for(int i=0; i<nrow; i++)
    {
        for(int j=0; j<ncol; j++)
            cout<<operator()(i,j)<<"\t";
        cout<<endl;
    }
	cout<<endl;
}

template <typename T>
matrix<T> matrix<T>::operator^(const matrix<T>& m)
{
	if(dim2()!=m.dim1()) throw "matrix dimension not match";
	matrix<T> tmp(dim1(),m.dim2());
	for(int i=0;i<tmp.dim1();i++)
	{
		for(int j=0;j<tmp.dim2();j++)
		{
			T sum=0;
			for(int k=0;k<dim2();k++)
				sum+=(*this)(i,k)*m(k,j);
			tmp(i,j)=sum;
		}
	}
	return tmp;
}

template <typename T>
void slice_matrix<T>::print()
{
        for(int i=0; i<dim1(); i++)
        {
            for(int j=0; j<dim2(); j++)
                cout<<m(row_indx[i],col_indx[j])<<"\t";

            cout<<endl;
        }
}

template <typename T>
matrix<T>::matrix(T* p,int nr,int nc,int step)
{
    nrow=nr;
    ncol=nc;
    pdata=new T[nrow*ncol];

    for(int i=0; i<nrow*ncol; i++) pdata[i]=p[i*step];
}

template <typename T>
matrix<T>& matrix<T>::operator-()
{
    for(int i=0; i<nrow*ncol; i++) pdata[i]=-pdata[i];

    return *this;
}

template <typename T>
matrix<T> matrix<T>::operator+(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        matrix<T> t(*this);
        return t+=m;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T> matrix<T>::operator-(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        matrix<T> t(*this);
        return t-=m;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        matrix<T> t(*this);
        return t*=m;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T> matrix<T>::operator/(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        matrix<T> t(*this);
        return t/=m;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T> matrix<T>::operator%(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        matrix<T> t(*this);
        return t%=m;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T>& matrix<T>::operator+=(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        const T* p=m.data();

        for(int i=0; i<nrow*ncol; i++) pdata[i]+=p[i];

        return *this;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T>& matrix<T>::operator-=(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        const T* p=m.data();

        for(int i=0; i<nrow*ncol; i++) pdata[i]-=p[i];

        return *this;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T>& matrix<T>::operator*=(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        const T* p=m.data();

        for(int i=0; i<nrow*ncol; i++) pdata[i]*=p[i];

        return *this;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T>& matrix<T>::operator/=(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        const T* p=m.data();

        for(int i=0; i<nrow*ncol; i++) pdata[i]/=p[i];

        return *this;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T>& matrix<T>::operator%=(const matrix<T>& m)
{
    if(nrow==m.dim1() && ncol==m.dim2())
    {
        const T* p=m.data();

        for(int i=0; i<nrow*ncol; i++) pdata[i]%=p[i];

        return *this;
    }
	else throw "matrix dimension mismatch";
}

template <typename T>
matrix<T> matrix<T>::operator+(T t0) const
{
    matrix<T> t(*this);
    return t+=t0;
}

template <typename T>
matrix<T> matrix<T>::operator-(T t0) const
{
    matrix<T> t(*this);
    return t-=t0;
}

template <typename T>
matrix<T> matrix<T>::operator*(T t0) const
{
    matrix<T> t(*this);
    return t*=t0;
}

template <typename T>
matrix<T> matrix<T>::operator/(T t0) const
{
    matrix<T> t(*this);
    return t/=t0;
}

template <typename T>
matrix<T> matrix<T>::operator%(T t0)
{
    matrix<T> t(*this);
    return t%=t0;
}

template <typename T>
matrix<T>& matrix<T>::operator+=(T t)
{
    for(int i=0; i<nrow*ncol; i++) pdata[i]+=t;

    return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator-=(T t)
{
    for(int i=0; i<nrow*ncol; i++) pdata[i]-=t;

    return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator*=(T t)
{
    for(int i=0; i<nrow*ncol; i++) pdata[i]*=t;

    return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator/=(T t)
{
    for(int i=0; i<nrow*ncol; i++) pdata[i]/=t;

    return *this;
}

template <typename T>
matrix<T>& matrix<T>::operator%=(T t)
{
    for(int i=0; i<nrow*ncol; i++) pdata[i]%=t;

    return *this;
}

template <typename T>
matrix<bool> matrix<T>::operator>(T t)
{
    //vector<bool> tmp(length());
	matrix<bool> tmp(dim1(),dim2());

    for(int i=0; i<length(); i++) 
		tmp.data()[i]=pdata[i]>t;

    return tmp;
}

template <typename T>
matrix<bool> matrix<T>::operator<(T t)
{
   //vector<bool> tmp(length());
	matrix<bool> tmp(dim1(),dim2());

    for(int i=0; i<length(); i++) 
		tmp.data()[i]=pdata[i]<t;

    return tmp;
}
template <typename T>
matrix<bool> matrix<T>::operator==(T t)
{
   //vector<bool> tmp(length());
	matrix<bool> tmp(dim1(),dim2());

    for(int i=0; i<length(); i++) 
		tmp.data()[i]=pdata[i]==t;
    return tmp;
}

template <typename T>
matrix<bool> matrix<T>::operator!=(T t)
{
   //vector<bool> tmp(length());
	matrix<bool> tmp(dim1(),dim2());

    for(int i=0; i<length(); i++) 
		tmp.data()[i]=pdata[i]!=t;
    return tmp;
}

//supporting functions
template <typename T>
matrix<T> operator+(T t,const matrix<T>& m)
{
    return m+t;
}

template <typename T>
matrix<T> operator-(T t,const matrix<T>& m)
{
    return -m+t;
}

template <typename T>
matrix<T> operator*(T t,const matrix<T>& m)
{
    return m*t;
}

template <typename T>
matrix<T> operator/(T t,const matrix<T>& m)
{
    matrix<T> temp(m);

    for(int i=0; i<m.dim1(); i++)
        for(int j=0; j<m.dim2(); j++)
            m(i,j)=t/m(i,j);
}

template <typename T>
matrix<T> operator%(T t,const matrix<T>& m)
{
    matrix<T> temp(m);

    for(int i=0; i<m.dim1(); i++)
        for(int j=0; j<m.dim2(); j++)
            m(i,j)=t%m(i,j);
}

template <typename T>
matrix<T> abs(const matrix<T>& m)
{
    matrix<T> tmp(m);
    T* p=tmp.data();

    for(int i=0; i<m.length(); i++) p[i]=abs(p[i]);

    return tmp;
}

template <typename T>
matrix<T> transpose(const matrix<T>& m)
{
    matrix<T> tmp(m.dim2(),m.dim1());

    for(int i=0; i<m.dim1(); i++)
        for(int j=0; j<m.dim2(); j++)
            tmp(j,i)=m(i,j);

    return tmp;
}

template <typename T>
matrix<T> reshape(const matrix<T>& m,int nr,int nc)
{
	matrix<T> tmp=m;
	return tmp.reshape(nr,nc); //use reshape to convert different matrix or vec-matrix
}

template <typename T>
matrix<T> operator^(const matrix<T>& M,const matrix<T>& N)
{
	return M^N;
}

template <typename T> 
double dot(const vec<T>& v1, const vec<T>& v2)
{
	if(!is_vec(v1) || !is_vec(v2)) throw "not a vec";
	if(v1.length()!=v2.length()) throw "vec size not the same";
	double sum=0;
	for(int i=0;i<v1.length();i++)
		sum+=v1[i]*v2[i];
	return sum;
}

template <typename T> 
ostream& operator<<(ostream& out, const matrix<T>& m)
{
	out<<endl;
	//complx needs its own format output
	for(int i=0;i<m.dim1();i++)
	{
		for(int j=0;j<m.dim2();j++)
			out<<m(i,j)<<"\t";
		out<<endl;
	}
	out<<endl;
	return out;
}

template <typename T>
ostream& operator<<(ostream& out,const slice_matrix<T>& sm)
{
	out<<endl;
	const matrix<T>& m=sm.get_matrix();
	for(int i=0; i<sm.dim1(); i++)
	{
		for(int j=0; j<sm.dim2(); j++)
			out<<m(sm.row_indx[i],sm.col_indx[j])<<"\t";
		out<<endl;
	}
	out<<endl;
	return out;
}


template <typename T> matrix<T> conj(const matrix<T>& m)
{
	matrix<T> tmp=m;
	//not done yet
	return tmp;
}

template <typename T> matrix<T> eye(T v,int n)
{
	matrix<T> m(n,n);
	for(int i=0;i<n;i++) m(i,i)=1;
	return m;
}

template <typename T> matrix<T> power(const matrix<T>& m, int n)
{
	matrix<T> mt=eye(T(0),n);
	//n>=0 has to be
	return mt;
}

template <typename T> matrix<T> cofactor(const matrix<T>& A,int p,int q,int n)
{
	int N=A.dim1();
    int i = 0, j = 0;
 
	matrix<T> temp(N,N);
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q)
            {
                //temp[i][j++] = A[row][col];
				temp(i,j++)=A(row,col);
 
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
	return temp;
}

template <typename T> double det(const matrix<T>& A,int n)
{
    double D = 0; // Initialize result
 
    //  Base case : if matrix contains single element
    if (n == 1)
        return A(0,0);
 
    matrix<T> temp(A.dim1(),A.dim2()); // To store cofactors
 
    int sign = 1;  // To store sign multiplier
 
     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        temp=cofactor(A, 0, f, n);
        D += sign * A(0,f) * det(temp, n - 1);
 
        // terms are to be added with alternate sign
        sign = -sign;
    }
 
    return D;
}

template <typename T> matrix<T> adjoint(const matrix<T>& A)
{
	int N=A.dim1();
	matrix<T> adj(N,N);
	if (N == 1)
    {
        adj(0,0) = 1;
        return adj;
    }
 
    // temp is used to store cofactors of A[][]
    int sign = 1;//, temp[N][N];
	matrix<T> temp(N,N);
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            // Get cofactor of A[i][j]
            temp=cofactor(A, i, j, N);
 
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;
 
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj(j,i) = (sign)*(det(temp, N-1));
        }
    }
	return adj;
}

template <typename T> matrix<T> inv(const matrix<T>& A)
{
	int N=A.dim1();
    double det0 = det(A, N);
	
    if (det0<1e-15)
    {
        cout << "Singular matrix, can't find its inverse";
        return matrix<T>();// false;
    }
 
    // Find adjoint
    //int adj[N][N];
    matrix<T> adj=adjoint(A);
	matrix<T> inverse(N,N);
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse(i,j) = adj(i,j)/det0;
	T* pdata=inverse.data();
    return inverse;
}

template <typename T> matrix<cplx> complex(const matrix<T>& m)
{
	matrix<cplx> mat=m;
	return mat;
}