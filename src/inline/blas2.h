/*D
   Level2Blas - Macros for providing level 2 blas to C programmers

   Here are macros for some BLAS2 operations.  This is a partial set;
   only ones that have been needed in the tools routines are provided.
   Further, they are organized differently from the BLAS: there are 
   different routines/macros for different operations, simplifying the
   use of common special cases.

   Most of the other level 2 BLAS involve triangular, banded, or symmetric
   matrics, or rank-1 updates to a matrix.

   Notes on implementation:
   Since the appropriate version can depend on run-time parameters, such as
   the size of the block, both the routine version and inline code versions
   are available.  The routine version ends with RT, the inline with
   IL.  
   The generic version makes some guess at the appropriate tradeoff.
   (Currently, this is to use the routine version always.)

   Definitions:
   Vectors and scalars in lower case, Matrices in upper case.
   "size?" is the size of a vector.  Matrices may be rectangular.
   
   We will need Complex routines for FMM, as well as triangular routines.

   We use a blas-like notation:
   _MV      a <- B*c
   _VPMV    a <- a + B*c
   _VPMgV   a <- a + B*c, but B may have a declared row dimension
   _VPAMV   a <- a + alpha * B * c
   _VPAMgV  a <- a + alpha * B * c, but B may have a declared row dimension
   _TRMVU   a <- U * a
   _TRMVL   a <- L * a
   _TRSLU   a <- U^-1 * a
   _TRSLL   a <- L^-1 * a
   Note the lack of routines for triangular matrices that are NOT in-place.
   This is unfortunate, since an in-place solve is NOT always what is
   wanted.  I'll add non-inplace versions; one implementation may
   use a copy and the in-place routine.

   For doing the interpolation and restriction operations in domain
   decomposition, it would be useful to have versions of these routines
   for non-square matrices, and for their transposes.  Also for the
   matrix-matrix versions.
 D*/

/*M
  DMV - Matrix-Vector product

  Input Parameters:
. b  - matrix
. nr,nc - size of matrix
. c  - vector to multiply

  Output Parameters
. a - Result of B * c

  Synopsis:
  void DMV(a,b,nr,nc,c)
  double *a, *b, *c;
  int    nr, nc
M*/

/*M
  DVPMV - Matrix-Vector product plus vector

  Input Parameters:
. b  - matrix
. nr,nc - size of matrix
. c  - vector to multiply

  Output Parameters
. a - Result of a + B * c

  Synopsis:
  void DVPMV(a,b,nr,nc,c)
  double *a, *b, *c;
  int    nr, nc
M*/

/*M
  DVPMgV - Matrix-Vector product plus vector

  Input Parameters:
. b  - matrix
. nr,nc - size of matrix
. nrd - declared row dimension of b
. c  - vector to multiply

  Output Parameters
. a - Result of a + B * c

  Synopsis:
  void DVPMgV(a,b,nr,nrd,nc,c)
  double *a, *b, *c;
  int    nr, nrd, nc
M*/

/*M
  DVPAMV - Scaled Matrix-Vector product plus vector

  Input Parameters:
. b  - matrix
. nr,nc - size of matrix
. alpha - scaling
. c  - vector to multiply

  Output Parameters
. a - Result of a + alpha * B * c

  Synopsis:
  void DVPAMV(a,b,nr,nc,alpha,c)
  double *a, *b, *c, alpha;
  int    nr, nc
M*/

/*M
  DVPAMgV - Scaled Matrix-Vector product plus vector

  Input Parameters:
. b  - matrix
. nr,nc - size of matrix
. nrd - declared row dimension of b
. alpha - scaling
. c  - vector to multiply

  Output Parameters
. a - Result of a + alpha * B * c

  Synopsis:
  void DVPAMgV(a,b,nr,nrd,nc,alpha,c)
  double *a, *b, *c, alpha;
  int    nr, nc, nrd
M*/

/*M
  DTRMVU - Multiply vector by upper triangular matrix

  Input Parameters:
. U  - matrix (stored as dense)
. n  - size of U
. a  - a = U * a
 
  Synopsis:
  void DTRMVU( a, u, n )
  double *a, *u;
  int    n;
M*/

/*M
  DTRMVL - Multiply vector by lower triangular matrix

  Input Parameters:
. L  - matrix (stored as dense)
. n  - size of L
. a  - a = L * a
 
  Synopsis:
  void DTRMVL( a, u, l )
  double *a, *l;
  int    n;
M*/

/*M
  DTRSLU - Solve upper-triangular system

  Input Parameters:
. U  - matrix (stored as dense)
. n  - size of U
. a  - a = inverse(U) * a
 
  Synopsis:
  void DTRSLU( a, u, n )
  double *a, *u;
  int    n;
M*/

/*M
  DTRSLL - Solve lower-triangular system

  Input Parameters:
. L  - matrix (stored as dense)
. n  - size of U
. a  - a = inverse(U) * a
 
  Synopsis:
  void DTRSLL( a, l, n )
  double *a, *l;
  int    n;
M*/

#ifndef _BLAS2
#define _BLAS2

/* Definitions mapping BLAS to proper Fortran sequences is in tools.h */

#define DMVRT(a,b,nr,nc,c) {int _One=1;double _DOne=1.0, _DZero=0.0;\
LAgemv_("N", &(nr), &(nc), &_DOne, b, &(nr), c, &_One, &_DZero, a, &_One, 1 );}
#define DVPMVRT(a,b,nr,nc,c) {int _One=1; double _DOne=1.0;\
LAgemv_("N", &(nr), &(nc), &_DOne, b, &(nr), c, &_One, &_DOne, a, &_One, 1 );} 
#define DVPMgVRT(a,b,nr,nrd,nc,c) {int _One=1; double _DOne=1.0;\
LAgemv_("N", &(nr), &(nc), &_DOne, b, &(nrd), c, &_One,&_DOne, a, &_One, 1 );} 
#define DVPAMVRT(a,b,nr,nc,al,c) {int _One=1; double _DOne=1.0;\
LAgemv_( "N", &(nr), &(nc), &(al), b, &(nr), c, &_One, &_DOne, a, &_One, 1 );} 
#define DVPAMgVRT(a,b,nr,nrd,nc,al,c) {int _One=1; double _DOne=1.0;\
LAgemv_("N", &(nr), &(nc), &(al), b, &(nrd), c, &_One, &_DOne, a, &_One, 1 );} 
#define DTRMVURT(a,u,n) {int _One=1;\
LAtrmv_( "U", "N", "N", &(n), u, &(n), a, &_One, 1, 1, 1 );}
#define DTRMVLRT(a,l,n) {int _One=1;\
LAtrmv_( "L", "N", "N", &(n), l, &(n), a, &_One, 1, 1, 1 );}
#define DTRSLURT(a,u,n) {int _One=1;\
LAtrsl_( "U", "N", "N", &(n), u, &(n), a, &_One, 1, 1, 1 );}
#define DTRSLLRT(a,l,n) {int _One=1;\
LAtrsl_( "L", "N", "N", &(n), l, &(n), a, &_One, 1, 1, 1 );}

/* These may need some fixes for doublecomplex constants */
#define ZMVRT(a,b,nr,nc,c) {int _One=1;double _DOne=1.0, _DZero=0.0;\
LAZgemv_("N",&(nr), &(nc), &_DOne, b, &(nr), c, &_One, &_DZero, a, &_One, 1 );}
#define ZVPMVRT(a,b,nr,nc,c) {int _One=1; double _DOne=1.0;\
LAZgemv_("N",&(nr), &(nc), &_DOne, b, &(nr), c, &_One, &_DOne, a, &_One, 1 );} 
#define ZVPAMVRT(a,b,nr,nc,al,c) {int _One=1; double _DOne=1.0;\
LAZgemv_("N", &(nr), &(nc), &(al), b, &(nr), c, &_One, &_DOne, a, &_One, 1 );} 
#define ZTRMVURT(a,u,n) {int _One=1;\
LAZtrmv_( "U", "N", "N", &(n), u, &(n), a, &_One, 1, 1, 1 );}
#define ZTRMVLRT(a,l,n) {int _One=1;\
LAZtrmv_( "L", "N", "N", &(n), l, &(n), a, &_One, 1, 1, 1 );}
#define ZTRSLURT(a,u,n) {int _One=1;\
LAZtrsl_( "U", "N", "N", &(n), u, &(n), a, &_One, 1, 1, 1 );}
#define ZTRSLLRT(a,l,n) {int _One=1;\
LAZtrsl_( "L", "N", "N", &(n), l, &(n), a, &_One, 1, 1, 1 );}

/*
   These are very straight-forward versions.  More sophisticated ones
   can do things like those in the other inline libraries, including:
   copy several items into registers (from consequtive locations to
   maximize cache/far-near memory bandwidth), do inner or outer product
   operations, and use different forms of loop unrolling (for, while).
   With any luck, the basic BLAS implemenations will be fast enough that
   it will not be necessary to do these by hand.

   For starters, stay with the routines (inline only those cases that
   are a problem; these might be small blocks. 

   To really get good performance from these routines, it is necessary
   to work with a block of the matrix and of the vectors.  For example,
   a 2x2 matrix block, and blocks of 2 vectors.  Note that with level-2,
   only the data in the vectors is reused; the elements of the matrix 
   are accessed only once.  Further, particularly in C, it is necessary to
   manually unroll the loops because C MUST assume that any store can
   conflict with a subsequent load.
 */   
#define DMVIL(a,b,nr,nc,c){\
int _i, _j, _k; register double _sum, *_bb=b, *_aa=a, *_cc=(c); \
for (_i=0; _i<nr; _i++) {\
    _sum = 0.0;  _k = _i; \
    for (_j=0; _j<nc; _j++) {\
        _sum += _bb[_k] * _cc[_j]; _k += nr;}\
    _aa[_i] = _sum; \
    }}
#define DMV2IL(a,b,nr,nc,c){\
int _i, _j; register double *_bb=b, *_aa=a, *_cc=(c); \
register double _tmp, _c1, _c2;\
for (_j=0; _j<nr; _j++) _aa[_j] = 0.0;\
for (_i=0; _i<nc-1; _i+=2) {\
    _c1 = _cc[_i]; _c2 = _cc[_i+1]; \
    for (_j=0; _j<nr; _j++) {_tmp = _aa[_j];\
        _aa[_j] = _tmp + _bb[_j] * _c1 + _bb[_j+nr] * _c2;}_bb+=nr+nr;}\
if (0x1&(nc)) {\
    _c1 = _cc[_i]; for (_j=0; _j<nr; _j++) _aa[_j] += _bb[_j] * _c1;}}

/* These just do inlined daxpy's */
#define DMV3aIL(a,b,nr,nc,c){\
int _i, _j; register double *_bb=b, *_aa=a, *_cc=(c); \
register double _c1, _a1, _a2;\
for (_j=0; _j<nr; _j++) _aa[_j] = 0.0;\
for (_i=0; _i<nc; _i++) {\
    _c1 = _cc[_i];\
    for (_j=0; _j<nr-1; _j+=2) {_a1 = _aa[_j];_a2 = _aa[_j+1];\
        _a1+=_bb[_j]*_c1;_a2+=_bb[_j+1]*_c1;_aa[_j]=_a1;_aa[_j+1]=_a2;}\
    if (0x1&(nr)) {_a1=_aa[_j];_a1+=_bb[_j]*_c1;_aa[_j]=_a1;}\
    _bb+=nr;}}
#define DVPMV3aIL(a,b,nr,nc,c){\
int _i, _j; register double *_bb=b, *_aa=a, *_cc=(c); \
register double _c1, _a1, _a2;\
for (_i=0; _i<nc; _i++) {\
    _c1 = _cc[_i];\
    for (_j=0; _j<nr-1; _j+=2) {_a1 = _aa[_j];_a2 = _aa[_j+1];\
        _a1+=_bb[_j]*_c1;_a2+=_bb[_j+1]*_c1;_aa[_j]=_a1;_aa[_j+1]=_a2;}\
    if (0x1&(nr)) {_a1=_aa[_j];_a1+=_bb[_j]*_c1;_aa[_j]=_a1;}\
    _bb+=nr;}}

/* These just do inlined daxpy's */
#define DMV2aIL(a,b,nr,nc,c){\
int _i, _j; register double *_bb=b, *_aa=a, *_cc=(c); \
register double _c1, _a1, _a2, _a3, _a4;\
for (_j=0; _j<nr; _j++) _aa[_j] = 0.0;\
for (_i=0; _i<nc; _i++) {\
    _c1 = _cc[_i];\
    for (_j=0; _j<nr-3; _j+=4) {_a1 = _bb[_j]*_c1;_a2 = _bb[_j+1]*_c1;\
	_a3 = _bb[_j+2]*_c1; _a4 = _bb[_j+3]*_c1;\
        _a1+=_aa[_j];_a2+=_aa[_j+1];_a3+=_aa[_j+2];_a4+=_aa[_j+3];\
        _aa[_j]=_a1;_aa[_j+1]=_a2;_aa[_j+2]=_a3;_aa[_j+3]=_a4;}\
    for (; _j<nr; _j++){_a1=_bb[_j]*_c1;_a1+=_aa[_j];_aa[_j]=_a1;}\
    _bb+=nr;}}
#define DVPMV2aIL(a,b,nr,nc,c){\
int _i, _j; register double *_bb=b, *_aa=a, *_cc=(c); \
register double _c1, _a1, _a2,_a3,_a4;\
for (_i=0; _i<nc; _i++) {\
    _c1 = _cc[_i];\
    for (_j=0; _j<nr-3; _j+=4) {_a1 = _bb[_j]*_c1;_a2 = _bb[_j+1]*_c1;\
	_a3 = _bb[_j+2]*_c1; _a4 = _bb[_j+3]*_c1;\
        _a1+=_aa[_j];_a2+=_aa[_j+1];_a3+=_aa[_j+2];_a4+=_aa[_j+3];\
        _aa[_j]=_a1;_aa[_j+1]=_a2;_aa[_j+2]=_a3;_aa[_j+3]=_a4;}\
    for (; _j<nr; _j++){_a1=_bb[_j]*_c1;_a1+=_aa[_j];_aa[_j]=_a1;}\
    _bb+=nr;}}

#define DVPMVIL(a,b,nr,nc,c){\
int _i, _j, _k; register double _sum, *_bb=b, *_aa=a, *_cc=(c); \
for (_i=0; _i<nr; _i++) {\
    _sum = _aa[_i];    _k = _i; \
    for (_j=0; _j<nc; _j++) {\
        _sum += _bb[_k] * _cc[_j]; _k += nr;}\
    _aa[_i] = _sum;  \
    }}
/* This version uses an unrolled daxpy operation */
#define DVPMV2IL(a,b,nr,nc,c){\
int _i, _j; register double *_bb=b, *_aa=a, *_cc=(c); \
register double _tmp, _c1, _c2;\
for (_i=0; _i<nc-1; _i+=2) {\
    _c1 = _cc[_i]; _c2 = _cc[_i+1]; \
    for (_j=0; _j<nr; _j++) {_tmp = _aa[_j];\
        _aa[_j] = _tmp + _bb[_j] * _c1 + _bb[_j+nr] * _c2;}_bb+=nr+nr;}\
if (0x1&(nc)) {\
    _c1 = _cc[_i]; for (_j=0; _j<nr; _j++) _aa[_j] += _bb[_j] * _c1;}}

#define DVPAMVIL(a,b,nr,nc,al,c) {\
int _i, _j, _k; register double _sum, *_bb=b, *_aa=a, *_cc=c; \
for (_i=0; _i<nr; _i++) {\
    _sum = 0.0;    _k = _i; \
    for (_j=0; _j<nc; _j++) {\
        _sum += _bb[_k] * _cc[_j]; _k += nr;}\
    _aa[_i] += al * _sum; \
    }}

/* Here are the default definitions */
#ifndef NO_BLAS
#define DMV(a,b,nr,nc,c)        DMVRT(a,b,nr,nc,c)
#define DVPMV(a,b,nr,nc,c)      DVPMVRT(a,b,nr,nc,c)
#define DVPMgV(a,b,nr,nrd,nc,c) DVPMgVRT(a,b,nr,nrd,nc,c)
#define DVPAMV(a,b,nr,nc,al,c)  DVPAMVRT(a,b,nr,nc,al,c)
#define DVPAMgV(a,b,nr,nrd,nc,al,c) DVPAMgVRT(a,b,nr,nrd,nc,al,c)
#define DTRMVU(a,u,n)           DTRMVURT(a,u,n)
#define DTRMVL(a,l,n)           DTRMVLRT(a,l,n)
#define DTRSLU(a,u,n)           DTRSLURT(a,u,n) 
#define DTRSLL(a,l,n)           DTRSLLRT(a,l,n)

#define ZMV(a,b,nr,nc,c)        ZMVRT(a,b,nr,nc,c)
#define ZVPMV(a,b,nr,nc,c)      ZVPMVRT(a,b,nr,nc,c) 
#define ZVPAMV(a,b,nr,nc,al,c)  ZVPAMVRT(a,b,nr,nc,al,c) 
#define ZTRMVU(a,u,n)           ZTRMVURT(a,u,n) 
#define ZTRMVL(a,l,n)           ZTRMVLRT(a,l,n)
#define ZTRSLU(a,u,n)           ZTRSLURT(a,u,n)
#define ZTRSLL(a,l,n)           ZTRSLLRT(a,l,n) 

#else
/* BLAS not available - use the inline versions */
#define DMV(a,b,nr,nc,c)        DMVIL(a,b,nr,nc,c)
#define DVPMV(a,b,nr,nc,c)      DVPMVIL(a,b,nr,nc,c)
#define DVPMgV(a,b,nr,nrd,nc,c) DVPMgVIL(a,b,nr,nrd,nc,c)
#define DVPAMV(a,b,nr,nc,al,c)  DVPAMVIL(a,b,nr,nc,al,c)
#define DVPAMgV(a,b,nr,nrd,nc,al,c) DVPAMgVIL(a,b,nr,nrd,nc,al,c)
#define DTRMVU(a,u,n)           DTRMVUIL(a,u,n)
#define DTRMVL(a,l,n)           DTRMVLIL(a,l,n)
#define DTRSLU(a,u,n)           DTRSLUIL(a,u,n) 
#define DTRSLL(a,l,n)           DTRSLLIL(a,l,n)
#endif

#endif
