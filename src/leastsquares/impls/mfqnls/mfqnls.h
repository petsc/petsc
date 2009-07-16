#ifndef __TAO_MFQNLS_H
#define __TAO_MFQNLS_H
#include "include/private/taosolver_impl.h"
#include "petscblaslapack.h"
#include "taolapack.h"

typedef struct {
  PetscInt npmax;  /* Max number of interpolation points (>n+1) (def: 2n+1) */
  PetscInt m,n;
  Vec *Xhist;
  Vec *Fhist;
    PetscReal *Fres; //(np)
    PetscReal *RES; //npxm
    PetscReal *work; //(n)
    PetscReal *work2; //(n)
    PetscReal *mwork; //(m)
    PetscReal *Disp; //nxn
    PetscReal *Fdiff;//nxm
    PetscReal *H; /* model hessians */   //mxnxn
    PetscReal *Hres;  //nxn
    PetscReal *Gres;  //n
    PetscReal *Gdel; //n
    PetscReal *Hdel; //mxnxn
    PetscReal *Gpoints; //nxn
    PetscReal *C; //m
    PetscReal *Xsubproblem; //n
    PetscInt *model_indices; //n
    PetscInt *interp_indices; //n
  VecScatter scatterf,scatterx; 
  Vec localf, localx, localfmin, localxmin;
  PetscMPIInt mpisize;

  PetscReal delta; /* Trust region radius (>0) */
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
    PetscReal *Q; //nxn
    PetscReal *tau; //scalar factors of H(i)
} TAO_MFQNLS;




#endif /* ifndef __TAO_MFQNLS */
