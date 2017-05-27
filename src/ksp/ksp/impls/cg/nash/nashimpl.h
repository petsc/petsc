/*****************************************************************************/
/* Context for using preconditioned conjugate gradient method to minimized a */
/* quadratic function subject to a trust region constraint.  If the matrix   */
/* is indefinite, a direction of negative curvature may be encountered.  If  */
/* a direction of negative curvature is found during the first iteration,    */
/* then it is a preconditioned gradient direction and we follow it to the    */
/* boundary of the trust region.  If a direction of negative curvature is    */
/* encountered on subsequent iterations, then we terminate the algorithm.    */
/*                                                                           */
/* This method is described in:                                              */
/*   S. Nash, "Newton-type Minimization via the Lanczos Method", SIAM        */
/*     Journal on Numerical Analysis, 21, pages 553-572, 1984.               */
/*****************************************************************************/

#if !defined(__CG_NASH)
#define __CG_NASH

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal radius;
  PetscReal norm_d;
  PetscReal o_fcn;
  PetscInt  dtype;
} KSPCG_NASH;

#endif

