/*
   Here are macros for the blas, with some special cases (such as
   increments of 1).  Similar conventions are used, including
   a leading D (double) and Z (double-complex).  Note that double-complex
   is ALWAYS supported, since C is not as broken as Fortran (complex is
   just another structure, and structure values may be returned).

   A few routines have been "added".  These are SET and AYPX (y <- x + a*y),
   the latter is needed by interative solvers.
 */

/* For the routines, we need a way to declare the routines as Fortran.
   We could use CDECL, but we don't need that much.  Mostly, we need a
   way to define whether a routine has a trailing underscore, and
   how character strings are passed. */

#ifndef _BLAS1
#define _BLAS1

#if defined(cray)
#define ddot_  SDOT
#define daxpy_ SAXPY
#define dswap_ SSWAP
#define dcopy_ SCOPY
#elif !defined(FORTRANUNDERSCORE)
#define ddot_  ddot
#define daxpy_ daxpy
#define dswap_ dswap
#define dcopy_ dcopy
#endif

extern double ddot_();
#ifndef _DOUBLECOMPLEX
#define PETSC_COMPLEX
#define _DOUBLECOMPLEX
typedef struct {
    double r, i;
    } dcomplex;
#endif
extern dcomplex zdotc_(), zdotu_();

#define DSETRT(n,a,x)    DSETIL(n,a,x)
#define DSWAPRT(n,x,y) { int _One=1;dswap_(&(n),x,&_One,y,&_One);}
#define DSCALRT(n,a,x) { int _One=1;dscal_(&(n),&(a),x,&_One);}
#define DCOPYRT(n,x,y) { int _One=1;dcopy_(&(n),x,&_One,y,&_One);}
#define DAXPYRT(n,a,x,y) { int _One=1;double _v=a;\
daxpy_(&(n),&(_v),x,&_One,y,&_One);}
#define DDOTRT(n,x,y,sum)  { int _One=1;sum+=ddot_(&(n),x,&_One,y,&_One);}
#define DAYPXRT(n,a,x,y) DAYPXIL(n,a,x,y)

#define ZSETRT(n,a,x)  ZSETIL(n,a,x)
#define ZSWAPRT(n,x,y) { int _One=1;zswap_(&(n),x,&_One,y,&_One);}
#define ZSCALRT(n,a,x) { int _One=1;zscal_(&(n),&(a),x,&_One);}
#define ZCOPYRT(n,x,y) { int _One=1;zcopy_(&(n),x,&_One,y,&_One);}
#define ZAXPYRT(n,a,x,y) { int _One=1;zaxpy_(&(n),&(a),x,&_One,y,&_One);}
#define ZDOTRT(n,x,y,sum)  { int _One=1;sum+=zdotc_(&(n),x,&_One,y,&_One);}
#define ZDOTTRT(n,x,y,sum) { int _One=1;sum+=zdotu_(&(n),x,&_One,y,&_One);}
#define ZAYPXRT(n,a,x,y) ZAYPXIL(n,a,x,y)

/* Inline versions.  For now, these will draw on the existing inline
   libraries (not done yet).
   These are unrolled to a depth of 4. 
 */
#define DSETIL(n,a,x) {register int _j; for(_j=0; _j<n; _j++) x[_j]=a;}
#define DSWAPIL(n,x,y) { register int _j; register double *_x=x,*_y=y;\
    register double _y1, _y2, _y3, _y4;\
    for (_j=0; _j<n-3; _j+=4) {_y1 = _x[_j];_y2 = _x[_j+1];\
	_y3 = _x[_j+2]; _y4 = _x[_j+3];\
        _x[_j] = _y[_j]; _x[_j+1] = _y[_j+1]; _x[_j+2] = _y[_j+2];\
	_x[_j+3] = _y[_j+3];\
	_y[_j]=_y1;_y[_j+1]=_y2;_y[_j+2]=_y3;_y[_j+3]=_y4;}\
    for (; _j<n; _j++){_y1=_x[_j];_x[_j]=_y[_j];_y[_j]=_y1;}}
#define DSCALIL(n,a,x) { register int _j; register double _a=a,*_x=x;\
    register double _x1, _x2, _x3, _x4;\
    for (_j=0; _j<n-3; _j+=4) {_x1 = _x[_j]*_a;_x2 = _x[_j+1]*_a;\
	_x3 = _x[_j+2]*_a; _x4 = _x[_j+3]*_a;\
        _x[_j]=_x1;_x[_j+1]=_x2;_x[_j+2]=_x3;_x[_j+3]=_x4;}\
    for (; _j<n; _j++){_x[_j]=_x[_j]*_a;}}
#define DCOPYIL(n,x,y) { register int _j; register double *_x=x,*_y=y;\
    register double _y1, _y2, _y3, _y4;\
    for (_j=0; _j<n-3; _j+=4) {_y1 = _x[_j];_y2 = _x[_j+1];\
	_y3 = _x[_j+2]; _y4 = _x[_j+3];\
        _y[_j]=_y1;_y[_j+1]=_y2;_y[_j+2]=_y3;_y[_j+3]=_y4;}\
    for (; _j<n; _j++){_y[_j]=_x[_j];}}
#define DAXPYIL(n,a,x,y) { register int _j; register double _a=a,*_x=x,*_y=y;\
    register double _y1, _y2, _y3, _y4;\
    for (_j=0; _j<n-3; _j+=4) {_y1 = _x[_j]*_a;_y2 = _x[_j+1]*_a;\
	_y3 = _x[_j+2]*_a; _y4 = _x[_j+3]*_a;\
        _y1+=_y[_j];_y2+=_y[_j+1];_y3+=_y[_j+2];_y4+=_y[_j+3];\
        _y[_j]=_y1;_y[_j+1]=_y2;_y[_j+2]=_y3;_y[_j+3]=_y4;}\
    for (; _j<n; _j++){_y1=_x[_j]*_a;_y1+=_y[_j];_y[_j]=_y1;}}
#define DDOTIL(n,x,y,sum) { register int _j; register double _sum,*_x=x,*_y=y;\
    register double _x1, _x2, _x3, _x4; _sum=0.0;\
    for (_j=0; _j<n-3; _j+=4) {_x1 = _x[_j];_x2 = _x[_j+1];\
	_x3 = _x[_j+2]; _x4 = _x[_j+3];\
        _sum += _y[_j]*_x1+_y[_j+1]*_x2+_y[_j+2]*_x3+_y[_j+3]*_x4;}\
    for (; _j<n; _j++){_sum+=_x[_j]*_y[_j];}sum=_sum;}
#define DAYPXIL(n,a,x,y) { register int _j; register double _a=a,*_x=x,*_y=y;\
    register double _y1, _y2, _y3, _y4;\
    for (_j=0; _j<n-3; _j+=4) {_y1 = _y[_j]*_a;_y2 = _y[_j+1]*_a;\
	_y3 = _y[_j+2]*_a; _y4 = _y[_j+3]*_a;\
        _y1+=_x[_j];_y2+=_x[_j+1];_y3+=_x[_j+2];_y4+=_x[_j+3];\
        _y[_j]=_y1;_y[_j+1]=_y2;_y[_j+2]=_y3;_y[_j+3]=_y4;}\
    for (; _j<n; _j++){_y1=_y[_j]*_a;_y1+=_x[_j];_y[_j]=_y1;}\

/* Finally, choice of the default meaning.  For now, that means the
   routines unless inline is selected */

#ifndef NO_BLAS
#define DSET(n,a,x)  DSETRT(n,a,x)
#define DSWAP(n,x,y) DSWAPRT(n,x,y)
#define DSCAL(n,a,x) DSCALRT(n,a,x)
#define DCOPY(n,x,y) DCOPYRT(n,x,y)
#define DAXPY(n,a,x,y) DAXPYRT(n,a,x,y)
#define DDOT(n,x,y,sum)  DDOTRT(n,x,y,sum)
#define DAYPX(n,a,x,y) DAYPXIL(n,a,x,y)

#define ZSET(n,a,x) ZSETRT(n,a,x)
#define ZSWAP(n,x,y) ZSWAPRT(n,x,y)
#define ZSCAL(n,a,x) ZSCALRT(n,a,x)
#define ZCOPY(n,x,y) ZCOPYRT(n,x,y)
#define ZAXPY(n,a,x,y) ZAXPYRT(n,a,x,y)
#define ZDOT(n,x,y,sum)  ZDOTRT(n,x,y,sum)
#define ZDOTT(n,x,y,sum) ZDOTTRT(n,x,y,sum)
#define ZAYPX(n,a,x,y) ZAYPXIL(n,a,x,y)

#else
#define DSET(n,a,x)  DSETIL(n,a,x)
#define DSWAP(n,x,y) DSWAPIL(n,x,y)
#define DSCAL(n,a,x) DSCALIL(n,a,x)
#define DCOPY(n,x,y) DCOPYIL(n,x,y)
#define DAXPY(n,a,x,y) DAXPYIL(n,a,x,y)
#define DDOT(n,x,y,sum)  DDOTIL(n,x,y,sum)
#define DAYPX(n,a,x,y) DAYPXIL(n,a,x,y)
#endif

#endif
