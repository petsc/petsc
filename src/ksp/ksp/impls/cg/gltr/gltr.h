/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#ifndef __GLTR
#define __GLTR

typedef struct {
  PetscReal *diag;		/* Diagonal part of Lanczos matrix           */
  PetscReal *offd;		/* Off-diagonal part of Lanczos matrix       */ 
  PetscReal *alpha;		/* Record of alpha values from CG            */
  PetscReal *beta;		/* Record of beta values from CG             */
  PetscReal *norm_r;		/* Record of residual values from CG         */

  PetscReal *rwork;		/* Real workspace for solver computations    */
  PetscInt  *iwork;		/* Integer workspace for solver computations */

  PetscReal radius;
  PetscReal norm_d;

  PetscReal init_pert;		/* Initial perturbation for solve            */
  PetscReal eigen_tol;		/* Tolerance used when computing eigenvalue  */
  PetscReal newton_tol;		/* Tolerance used for newton method          */

  PetscInt  alloced;		/* Size of workspace vectors allocated	     */
  PetscInt  init_alloc;		/* Initial size for workspace vectors        */

  PetscInt  max_its;		/* Maximum cg and lanczos iterations         */
  PetscInt  max_cg_its;		/* Maximum conjugate gradient iterations     */
  PetscInt  max_lanczos_its;	/* Maximum lanczos iterations		     */
  PetscInt  max_newton_its;	/* Maximum newton iterations                 */
} KSP_GLTR;

#endif

