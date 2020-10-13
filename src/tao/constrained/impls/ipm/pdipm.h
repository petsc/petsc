
#ifndef __TAO_PDIPM_H
#define __TAO_PDIPM_H
#include <petsc/private/taoimpl.h>

/*
 Context for Primal-Dual Interior-Point Method
 See the document pdipm.pdf
*/

typedef struct {
  /* Sizes (n = local, N = global) */
  PetscInt nx,Nx;           /* Decision variables nx = nxfixed + nxub + nxlb + nxbox + nxfree */
  PetscInt nxfixed,Nxfixed; /* Fixed decision variables */
  PetscInt nxlb,Nxlb;       /* Decision variables with lower bounds only */
  PetscInt nxub,Nxub;       /* Decision variables with upper bounds only */
  PetscInt nxbox,Nxbox;     /* Decision variables with box constraints */
  PetscInt nxfree,Nxfree;   /* Free variables */
  PetscInt ng,Ng;           /* user equality constraints g(x) = 0. */
  PetscInt nh,Nh;           /* user inequality constraints h(x) >= 0. */
  PetscInt nce,Nce;         /* total equality constraints. nce = ng + nxfixed */
  PetscInt nci,Nci;         /* total inequality constraints nci = nh + nxlb + nxub + 2*nxbox */
  PetscInt n,N;             /* Big KKT system size n = nx + nce + 2*nci */

  /* Vectors */
  Vec      X;               /* R^n   - Big KKT system vector [x; lambdae; lambdai; z] */
  Vec      x;               /* R^nx - work vector, same layout as tao->solution */
  Vec      lambdae;         /* R^nce - vector, shares local arrays with X */
  Vec      lambdai;         /* R^nci - vector, shares local arrays with X */
  Vec      z;               /* R^nci - vector, shares local arrays with X */

  /* Work vectors */
  Vec      lambdae_xfixed; /* Equality constraints lagrangian multipler vector for fixed variables */
  Vec      lambdai_xb;     /* User inequality constraints lagrangian multipler vector */

  /* Lagrangian equality and inequality Vec */
  Vec      ce,ci; /* equality and inequality constraints */

  /* Offsets for subvectors */
  PetscInt  off_lambdae,off_lambdai,off_z;

  /* Scalars */
  PetscReal L;     /* Lagrangian = f(x) - lambdae^T*ce(x) - lambdai^T*(ci(x) - z) - mu*sum_{i=1}^{Nci}(log(z_i)) */
  PetscReal gradL; /* gradient of L w.r.t. x */

  /* Matrices */
  Mat Jce_xfixed; /* Jacobian of equality constraints cebound(x) = J(nxfixed) */
  Mat Jci_xb;     /* Jacobian of inequality constraints Jci = [tao->jacobian_inequality ; J(nxub); J(nxlb); J(nxbx)] */
  Mat K;          /* KKT matrix */

  /* Parameters */
  PetscReal mu;               /* Barrier parameter */
  PetscReal mu_update_factor; /* Multiplier for mu update */

  /* Tolerances */

  /* Index sets for types of bounds on variables */
  IS  isxub;    /* Finite upper bound only -inf < x < ub   */
  IS  isxlb;    /* Finite lower bound only  lb <= x < inf  */
  IS  isxfixed; /* Fixed variables          lb  = x = ub   */
  IS  isxbox;   /* Boxed variables          lb <= x <= ub  */
  IS  isxfree;  /* Free variables         -inf <= x <= inf */

  /* Index sets for PC fieldsplit */
  IS  is1,is2;

  /* Options */
  PetscBool monitorkkt;           /* Monitor KKT */
  PetscReal push_init_slack;      /* Push initial slack variables (z) away from bounds */
  PetscReal push_init_lambdai;    /* Push initial inequality variables (lambdai) away from bounds */
  PetscBool solve_reduced_kkt;    /* Solve Reduced KKT with fieldsplit */
  PetscBool solve_symmetric_kkt;  /* Solve non-reduced symmetric KKT system */

  SNES           snes;                                    /* Nonlinear solver */
  Mat            jac_equality_trans,jac_inequality_trans; /* working matrices */

  PetscReal      obj;  /* Objective function */

  /* Offsets for parallel assembly */
  PetscInt       *nce_all;
} TAO_PDIPM;

PETSC_INTERN PetscErrorCode TaoSNESFunction_PDIPM(SNES,Vec,Vec,void*);

#endif /* ifndef __TAO_PDIPM_H */
