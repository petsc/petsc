
/*  
    Private Krylov Context Structure (KSP) for the Anderson mixing method aka nonlinear Krylov applied to a linear equation

*/

#if !defined(__NGMRESIMPL_H)
#define __NGMRESIMPL_H

/*
        Defines the basic KSP object
*/
#include <private/kspimpl.h>

typedef struct {
  /* the coefficient matrix and orthogonalization information Hh holds the coefficient matrix */
  PetscScalar *hh_origin;
  PetscScalar *orthogwork; /*holds dot products computed in orthogonalization */ 
  Vec       *v,*w;
  PetscReal *f2;     /* 2-norms of function (residual) at each stage */
  PetscInt  msize;   /* maximum size of space */
  PetscInt  csize;   /* current size of space */
  PetscScalar beta; /* relaxation parameter */
  PetscScalar *nrs;            /* temp that holds the coefficients of the Krylov vectors that form the minimum residual solution */
} KSP_NGMRES;

#define HH(a,b)  (ngmres->hh_origin + (a)*(ngmres->msize)+(b))

#endif
