#ifndef __TAO_MFQNLS_H
#define __TAO_MFQNLS_H
#include "include/private/taosolver_impl.h"
#include "petsc.h"
#include "petscblaslapack.h"
#include "taolapack.h"


typedef struct {
  /* Solver Parameters */
  PetscInt npmax;  /* Max number of interpolation points (>n+1) (def: 2n+1) */
  PetscReal delta0; /* Initial trust region radius (def: 0.1) */
  PetscReal gamma0; /* parameter for shrinking trust region in (0,1) */
  PetscReal gamma1; /* parameter for enlarging trust region (>1) */
  PetscReal eta1;   /* parameter 2 for accepting point in (0,1) */
  PetscReal par1; /* delta multiplier for checking validity */
  PetscReal par2; /* delta multiplier for all interp. points */
  PetscReal par3; /* Pivot threshold for validity */
  PetscReal par4; /* Pivot threshold for additional points */
  


  /* Workspace */
  PetscInt m; /* Number of components of residual function */
  PetscInt n; /* Number of parameters */
  PetscInt nfs; /* Number of pre-computed function evaluations */
  PetscReal delta; /* Trust region radius */
  Vec *Xhist;
  Vec *Fhist;
  Mat *Hres; /* array of nxn matrices,  length m */
  Mat *Hresdel; /* array of nxn matrices,  length m */
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
  PetscReal *Hdel_array; /* mxnxn */
  PetscReal *Gres;  /* n */
  PetscReal *Gdel_array; /* mxn */
  PetscReal *Gpoints; /* nxn */
  PetscReal *C; /* m */
  PetscReal *D; /* n */
  PetscReal *Xsubproblem; /* n */
  PetscInt *indices; /* 1,2,3...m */
  PetscInt minindex;
  PetscInt nmodelpoints;
  PetscInt *model_indices; /* n */
  PetscInt *interp_indices; /* n */
  PetscBLASInt *iwork; /* n */
  PetscInt nHist;
  VecScatter scatterf,scatterx; 
  Vec *Mdir;
  PetscInt sizemdir;
  Vec localf, localx, localfmin, localxmin;
  Vec workxvec;
  PetscMPIInt mpisize;


  Mat Hs;
  Vec b;
  
  PetscReal deltamax;
  PetscReal deltamin;
  PetscReal subproblem_rtol;   /* parameter used by quadratic subproblem */
  PetscInt subproblem_maxits; /* parameter used by quadratic subproblem */
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


    
       
} TAO_POUNDERS;



PetscErrorCode TaoPounders_formquad(TAO_POUNDERS *mfqP,PetscBool checkonly)
PetscErrorCode TaoPounders_solvequadratic(TaoSolver tao,PetscReal *gnorm, PetscReal *qmin);
//PetscErrorCode phi2eval(PetscReal *x, PetscInt n, PetscReal *phi);
//PetscErrorCode getquadpounders(TAO_POUNDERS *mfqP);
//PetscErrorCode morepoints(TAO_POUNDERS *mfqP);
//PetscErrorCode addpoint(TaoSolver tao, TAO_POUNDERS *mfqP, PetscInt index);
//PetscErrorCode modelimprove(TaoSolver tao, TAO_POUNDERS *mfqP, PetscInt addallpoints);
//PetscErrorCode affpoints(TAO_POUNDERS *mfqP, PetscReal *xmin, PetscReal c);

#endif /* ifndef __TAO_MFQNLS */
