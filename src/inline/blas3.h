/* Here are macros for some BLAS3 operations.  This is a partial set;
   only ones that have been needed in the tools routines are provided.
   Further, they are organized differently from the BLAS: there are 
   different routines/macros for different operations, simplifying the
   use of common special cases.

   Most of the other level 3 BLAS involve triangular, banded, or symmetric
   matrics.

   Notes on implementation:
   Since the appropriate version can depend on run-time parameters, such as
   the size of the block, both the routine version and inline code versions
   are available.  The routine version ends with RT, the inline with
   IL.  
   The generic version makes some guess at the appropriate tradeoff.

   These all use Fortran ordering for the matrices, that is, stored by
   columns.  Stored by rows can be handled by using the "T" versions
   (Transpose) rather than the "N" (NoTranspose) versions.

   It might also be useful to have a "scattered-block" version.
   This would be the analogue of the sparse axpy, but on blocks.
   The advantage is that one of the matrices (the "alpha") could be used
   on multiple "x" matrics.  Note that for runs of consequtive blocks,
   I can just do a matrix-matrix product with a longer "x" matrix.
   Somewhat more generally, the same thing can be done when the source and
   destination have the same "column" indices.

   Here are the names in a more blas-like fashion:
   _MM    C <- A * B
   _MPMM  C <- C + A * B
   _MPAMM C <- C + a * A * B
   _TRMML B <- L * B
   _TRMMU B <- U * B
   _TRMSLL B <- L^-1 * B
   _TRMSLU B <- U^-1 * B

   I need to develop inline versions of these.  
 */
#ifndef _BLAS3
#define _BLAS3

#define DMMRT(C,nr,nc,nca,A,B) {int _One=1; double _DZero=0.0, _DOne=1.0;\
dgemm_("N","N",&(nr),&(nc),&(nca),&_DOne,A,&(nr),B,&(nca),&_DZero,C,&(nr),1,1);}
#define DMPMMRT(C,nr,nc,nca,A,B) {int _One=1; double _DOne=1.0;\
dgemm_("N","N",&(nr),&(nc),&(nca),&_DOne,A,&(nr),B,&(nca),&_DOne,C,&(nr),1,1);}
#define DMPAMMRT(C,nr,nc,nca,al,A,B) {int _One=1; double _DOne=1.0;\
dgemm_("N","N",&(nr),&(nc),&(nca),&(al),A,&(nr),B,&(nca),&_DOne,C,&(nr),1,1);}
#define DTRMML(B,l,nr,nc) {int _One=1;double _DOne=1.0;\
dtrmm_("L","L","N","N",&(nr),&(nc),&_DOne,l,&(nr),B,&(nr),1,1,1,1);}
#define DTRMMU(B,u,nr,nc) {int _One=1;double _DOne=1.0;\
dtrmm_("L","U","N","N",&(nr),&(nc),&_DOne,u,&(nr),B,&(nr),1,1,1,1);}
#define DTRMSLL(B,l,nr,nc) {int _One=1;double _DOne=1.0;\
dtrsm_("L","L","N","N",&(nr),&(nc),&_DOne,l,&(nr),B,&(nr),1,1,1,1);}
#define DTRMSLU(B,u,nr,nc) {int _One=1;double _DOne=1.0;\
dtrsm_("L","U","N","N",&(nr),&(nc),&_DOne,u,&(nr),B,&(nr),1,1,1,1);}

#define ZMMRT(C,nr,nc,nca,A,B) {int _One=1; double _DZero=0.0, _DOne=1.0;\
zgemm_("N","N",&(nr),&(nc),&(nca),&_DOne,A,&(nr),B,&(nca),&_DZero,C,&(nr),1,1);}
#define ZMPMMRT(C,nr,nc,nca,A,B) {int _One=1; double _DOne=1.0;\
zgemm_("N","N",&(nr),&(nc),&(nca),&_DOne,A,&(nr),B,&(nca),&_DOne,C,&(nr),1,1);}
#define ZMPAMMRT(C,nr,nc,nca,al,A,B) {int _One=1; double _DOne=1.0;\
zgemm_("N","N",&(nr),&(nc),&(nca),&(al),A,&(nr),B,&(nca),&_DOne,C,&(nr),1,1);}
#define ZTRMML(B,l,nr,nc) {int _One=1;double _DOne=1.0;\
ztrmm_("L","L","N","N",&(nr),&(nc),&_DOne,l,&(nr),B,&(nr),1,1,1,1);}
#define ZTRMMU(B,u,nr,nc) {int _One=1;double _DOne=1.0;\
ztrmm_("L","U","N","N",&(nr),&(nc),&_DOne,u,&(nr),B,&(nr),1,1,1,1);}
#define ZTRMSLL(B,l,nr,nc) {int _One=1;double _DOne=1.0;\
ztrsm_("L","L","N","N",&(nr),&(nc),&_DOne,l,&(nr),B,&(nr),1,1,1,1);}
#define ZTRMSLU(B,u,nr,nc) {int _One=1;double _DOne=1.0;\
ztrsm_("L","U","N","N",&(nr),&(nc),&_DOne,u,&(nr),B,&(nr),1,1,1,1);}

/* Matrix matrix product: mout <- m1 * m2 */
#define DMMPRODRT(mout,m1,m2,nr1,nc1,nc2) {\
double dzero=0.0, dOne=1.0;\
dgemm_( "N", "N", &(nr1), &(nc2), &(nc1), &dOne, m1, &(nr1), m2, &(nc1), \
       &dzero, vout, &(nr1), 1, 1 );}
/* Matrix matrix product with add: mout <- mout + m1*m2 . */
#define DMMPRODADDRT(mout,m1,m2,nr1,nc1,nc2) {\
double dOne=1.0;\
dgemm_( "N", "N", &(nr1), &(nc2), &(nc1), &dOne, m1, &(nr1), m2, &(nc1), \
       &dOne, vout, &(nr1), 1, 1 );}
#define DUPAMMRT(mout,alpha,m1,m2,nr1,nc1,nc2) {\
double dOne=1.0;\
dgemm_( "N", "N", &(nr1), &(nc2), &(nc1), &alpha, m1, &(nr1), m2, &(nc1), \
       &dOne, vout, &(nr1), 1, 1 );}

/* Here I need inline routines for these. This is not correct (and
   besides, I want one that is FAST) */
#define DMMPRODIL(mout,m1,m2,nr1,nc1,nc2) {\
int i, j, k, l; double sum; \
for (j=0; j<nc2; j++) { \
    for (i=0; i<nr1; i++) { \
    	sum = 0.0; l = 0; \
    	for (k=0; k<nc1; k++) { \
    	    sum += m1[l] * m2[k]; l += nr1; \
    	    } \
    	mout[i] = sum; \
        } \
    mout += nr1; \
    } \
}

#define DMMPRODADDIL(mout,m1,m2,nr1,nc1,nc2) {\
int i, j, k, l; double sum; \
for (j=0; j<nc2; j++) { \
    for (i=0; i<nr1; i++) { \
    	sum = mout[i]; l = 0; \
    	for (k=0; k<nc1; k++) { \
    	    sum += m1[l] * m2[k]; l += nr1; \
    	    } \
    	mout[i] = sum; \
        } \
    mout += nr1; \
    } \
}

#ifndef NO_BLAS
#define DMMPROD(mout,m1,m2,nr1,nc1,nc2)    DMMPRODRT(mout,m1,m2,nr1,nc1,nc2)
#define DMMPRODADD(mout,m1,m2,nr1,nc1,nc2) DMMPRODADDRT(mout,m1,m2,nr1,nc1,nc2)
#else
#define DMMPROD(mout,m1,m2,nr1,nc1,nc2)    DMMPRODIL(mout,m1,m2,nr1,nc1,nc2)
#define DMMPRODADD(mout,m1,m2,nr1,nc1,nc2) DMMPRODADDIL(mout,m1,m2,nr1,nc1,nc2)
#endif

#endif
