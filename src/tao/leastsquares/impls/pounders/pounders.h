#ifndef __TAO_MFQNLS_H
#define __TAO_MFQNLS_H
#include <petsc/private/taoimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt npmax;  /* Max number of interpolation points (>n+1) (def: 2n+1) */
  PetscInt nmax; /* Max(n*(n+1)/2, 5*npmax) */
  PetscInt m,n;
  Vec *Xhist;
  Vec *Fhist;
  PetscReal *Fres; /* (nfmax) */
  PetscReal *RES; /* npxm */
  PetscReal *work; /* (n) */
  PetscReal *work2; /* (n) */
  PetscReal *work3; /* (n) */
  PetscReal *xmin; /* (n) */
  PetscReal *mwork; /* (m) */
  PetscReal *Disp; /* nxn */
  PetscReal *Fdiff;/* nxm */
  PetscReal *H; /* model hessians (mxnxn) */
  PetscReal *Hres;  /* nxn */
  PetscReal *Gres;  /* n */
  PetscReal *Gdel; /* mxn */
  PetscReal *Hdel; /* mxnxn */
  PetscReal *Gpoints; /* nxn */
  PetscReal *C; /* m */
  PetscReal *Xsubproblem; /* n */
  PetscInt *indices; /* 1,2,3...m */
  PetscInt minindex;
  PetscInt nmodelpoints;
  PetscInt *model_indices; /* n */
  PetscInt last_nmodelpoints;
  PetscInt *last_model_indices; /* n */
  PetscInt *interp_indices; /* n */
  PetscBLASInt *iwork; /* n */
  PetscReal *w; /* nxn */
  PetscInt nHist;
  VecScatter scatterf,scatterx;
  Vec localf, localx, localfmin, localxmin;
  Vec workxvec,workfvec;
  PetscMPIInt size;


  PetscReal delta; /* Trust region radius (>0) */
  PetscReal delta0;
  PetscBool usegqt;
  Mat Hs;
  Vec b;

  PetscReal deltamax;
  PetscReal deltamin;
  PetscReal c1; /* Factor for checking validity */
  PetscReal c2; /* Factor for linear poisedness */
  PetscReal theta1; /* Pivot threshold for validity */
  PetscReal theta2; /* Pivot threshold for additional points */
  PetscReal gamma0; /* parameter for shrinking trust region (<1) */
  PetscReal gamma1; /* parameter for enlarging trust region (>2) */
  PetscReal eta0;   /* parameter 1 for accepting point (0 <= eta0 < eta1)*/
  PetscReal eta1;   /* parameter 2 for accepting point (eta0 < eta1 < 1)*/
  PetscReal gqt_rtol;   /* parameter used by gqt */
  PetscInt gqt_maxits; /* parameter used by gqt */
  /* QR factorization data */
  PetscInt q_is_I;
  PetscReal *Q; /* npmax x npmax */
  PetscReal *Q_tmp; /* npmax x npmax */
  PetscReal *tau; /* scalar factors of H(i) */
  PetscReal *tau_tmp; /* scalar factors of H(i) */
  PetscReal *npmaxwork; /* work vector of length npmax */
  PetscBLASInt *npmaxiwork; /* integer work vector of length npmax */
  /* morepoints and getquadnlsmfq */
  PetscReal *L;   /* n*(n+1)/2 x npmax */
  PetscReal *L_tmp;   /* n*(n+1)/2 x npmax */
  PetscReal *L_save;   /* n*(n+1)/2 x npmax */
  PetscReal *Z;   /* npmax x npmax-(n+1) */
  PetscReal *M;   /* npmax x n+1 */
  PetscReal *N;   /* npmax x n*(n+1)/2  */
  PetscReal *alpha; /* n+1 */
  PetscReal *beta; /*  r(n+1)/2 */
  PetscReal *omega; /* npmax - np - 1 */

  Tao subtao;
  Vec       subxl,subxu,subx,subpdel,subndel,subb;
  Mat       subH;

} TAO_POUNDERS;


PetscErrorCode gqt(PetscInt n, PetscReal *a, PetscInt lda, PetscReal *b, PetscReal delta, PetscReal rtol, PetscReal atol, PetscInt itmax, PetscReal *par, PetscReal *f, PetscReal *x, PetscInt *info, PetscInt *its, PetscReal *z, PetscReal *wa1, PetscReal *wa2);

PetscErrorCode gqtwrap(Tao tao,PetscReal *gnorm, PetscReal *qmin);
PetscErrorCode phi2eval(PetscReal *x, PetscInt n, PetscReal *phi);
PetscErrorCode getquadpounders(TAO_POUNDERS *mfqP);
PetscErrorCode morepoints(TAO_POUNDERS *mfqP);
PetscErrorCode addpoint(Tao tao, TAO_POUNDERS *mfqP, PetscInt index);
PetscErrorCode modelimprove(Tao tao, TAO_POUNDERS *mfqP, PetscInt addallpoints);
PetscErrorCode affpoints(TAO_POUNDERS *mfqP, PetscReal *xmin, PetscReal c);

EXTERN_C_BEGIN
void dgqt_(PetscInt *n, PetscReal *a, PetscInt *lda, PetscReal *b, PetscReal *delta, PetscReal *rtol, PetscReal *atol, PetscInt *itmax, PetscReal *par, PetscReal *f, PetscReal *x, PetscInt *info, int *its, PetscReal *z, PetscReal *wa1, PetscReal *wa2);
EXTERN_C_END
#endif /* ifndef __TAO_MFQNLS */
