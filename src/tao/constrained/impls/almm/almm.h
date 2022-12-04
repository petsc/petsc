#ifndef TAOALMM_H
#define TAOALMM_H
#include <petsc/private/taoimpl.h>

typedef struct {
  Tao subsolver, parent;          /* subsolver for aug-lag subproblem */
  PetscErrorCode (*sub_obj)(Tao); /* subsolver objective function */
  TaoALMMType type;               /* subsolver objective type */

  IS         *Pis, *Yis;           /* index sets to separate primal and dual vector spaces */
  VecScatter *Pscatter, *Yscatter; /* scatter objects to write into combined vector spaces */

  Mat  Ae, Ai;                              /* aliased constraint Jacobians (do not destroy!) */
  Vec  Px, LgradX, Ce, Ci, G;               /* aliased vectors (do not destroy!) */
  Vec  Ps, LgradS, Yi, Ye;                  /* sub-vectors for primal variables */
  Vec *Parr, P, PL, PU, *Yarr, Y, C;        /* arrays and vectors for combined vector spaces */
  Vec  Psub, Xwork, Cework, Ciwork, Cizero; /* work vectors */

  PetscReal Lval, fval, gnorm, cnorm, cenorm, cinorm, cnorm_old; /* scalar variables */
  PetscReal mu0, mu, mu_fac, mu_pow_good, mu_pow_bad;            /* penalty parameters */
  PetscReal ytol0, ytol, gtol0, gtol;                            /* convergence parameters */
  PetscReal mu_max, ye_min, yi_min, ye_max, yi_max;              /* parameter safeguards */

  PetscBool info;
} TAO_ALMM;

PETSC_INTERN PetscErrorCode TaoALMMGetType_Private(Tao, TaoALMMType *);
PETSC_INTERN PetscErrorCode TaoALMMSetType_Private(Tao, TaoALMMType);
PETSC_INTERN PetscErrorCode TaoALMMGetSubsolver_Private(Tao, Tao *);
PETSC_INTERN PetscErrorCode TaoALMMSetSubsolver_Private(Tao, Tao);
PETSC_INTERN PetscErrorCode TaoALMMGetMultipliers_Private(Tao, Vec *);
PETSC_INTERN PetscErrorCode TaoALMMSetMultipliers_Private(Tao, Vec);
PETSC_INTERN PetscErrorCode TaoALMMGetPrimalIS_Private(Tao, IS *, IS *);
PETSC_INTERN PetscErrorCode TaoALMMGetDualIS_Private(Tao, IS *, IS *);
PETSC_INTERN PetscErrorCode TaoALMMSubsolverObjective_Private(Tao, Vec, PetscReal *, void *);
PETSC_INTERN PetscErrorCode TaoALMMSubsolverObjectiveAndGradient_Private(Tao, Vec, PetscReal *, Vec, void *);

#endif
