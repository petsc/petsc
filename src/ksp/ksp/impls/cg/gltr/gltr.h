/*****************************************************************************/
/* Context for using preconditioned conjugate gradient method to minimized a */
/* quadratic function subject to a trust region constraint.  If the matrix   */
/* is indefinite, a direction of negative curvature may be encountered.  If  */
/* a direction of negative curvature is found, we continue to build the      */
/* tridiagonal Lanczos matrix for a fixed number of iterations.  After this  */
/* matrix is computed, we compute a global solution to solve the trust-      */
/* region problem with the tridiagonal approximation by using a variant of   */
/* the More'-Sorenson algorithm.  The direction is then constructed from     */
/* this solution.                                                            */
/*                                                                           */
/* This method is described in:                                              */
/*   N. Gould, S. Lucidi, M. Roma, and Ph. Toint, "Solving the Trust-Region  */
/*     Subproblem using the Lanczos Method", SIAM Journal on Optimization,   */
/*     9, pages 504-525, 1999.                                               */
/*****************************************************************************/

#ifndef __GLTR
#define __GLTR

typedef struct {
  PetscReal *diag;		/* Diagonal part of Lanczos matrix           */
  PetscReal *offd;		/* Off-diagonal part of Lanczos matrix       */ 
  PetscReal *alpha;		/* Record of alpha values from CG            */
  PetscReal *beta;		/* Record of beta values from CG             */
  PetscReal *norm_r;		/* Record of residual values from CG         */

  PetscReal *rwork;		/* Real workspace for solver computations    */
  PetscBLASInt *iwork;		/* Integer workspace for solver computations */

  PetscReal radius;
  PetscReal norm_d;
  PetscReal e_min;
  PetscReal o_fcn;
  PetscReal lambda;

  PetscReal init_pert;		/* Initial perturbation for solve            */
  PetscReal eigen_tol;		/* Tolerance used when computing eigenvalue  */
  PetscReal newton_tol;		/* Tolerance used for newton method          */

  PetscInt alloced;		/* Size of workspace vectors allocated	     */
  PetscInt init_alloc;		/* Initial size for workspace vectors        */

  PetscInt max_lanczos_its;	/* Maximum lanczos iterations		     */
  PetscInt max_newton_its;	/* Maximum newton iterations                 */
  PetscInt dtype;		/* Method used to measure the norm of step   */
} KSP_GLTR;

#endif

