
#if   defined(BLAS)
#if   defined(r8)
double ddot (int n, double *x, int incx, double *y, int incy);
void   daxpy(long int n, double da, double *dx, long int incx, 
	     double *dy, long int incy);
void   dcopy(int n, double *x, int incx, double *y, int incy);
#define dot  ddot
#define axpy daxpy
#define copy dcopy
#else
float sdot (int n, float *x, int incx, float *y, int incy);
void  saxpy(long int n, float da, float *dx, long int incx, 
	    float *dy, long int incy);
float scopy(int n, float *x, int incx, float *y, int incy);
#define dot  sdot
#define axpy saxpy
#define copy scopy
#endif
#elif defined(CBLAS)
#if   defined(r8)
double cblas_ddot (int n, double *x, int incx, double *y, int incy);
void   cblas_daxpy(long int n, double da, double *dx, long int incx, 
		   double *dy, long int incy);
void   cblas_dcopy(int n, double *x, int incx, double *y, int incy);
#define dot  cblas_ddot
#define axpy cblas_daxpy
#define copy cblas_dcopy
#else
float cblas_sdot (int n, float *x, int incx, float *y, int incy);
void  cblas_saxpy(long int n, float da, float *dx, long int incx, 
           float *dy, long int incy);
float cblas_scopy(int n, float *x, int incx, float *y, int incy);
#define dot  cblas_sdot
#define axpy cblas_saxpy
#define copy cblas_scopy
#endif
#endif







