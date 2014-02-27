#ifndef __TAODM_IMPL_H
#define __TAODM_IMPL_H
#include <petscdm.h>
#include <petscdmda.h>
#include <petsctaodm.h>

struct _TaoDMOps {
  PetscErrorCode (*computeobjectiveandgradientlocal)(DMDALocalInfo*,PetscReal **, PetscReal *, PetscReal **, void*);
  PetscErrorCode (*computeobjectivelocal)(DMDALocalInfo*,PetscReal**, PetscReal*, void*);
  PetscErrorCode (*computegradientlocal)(DMDALocalInfo*,PetscReal**, PetscReal**, void*);
  PetscErrorCode (*computehessianlocal)(DMDALocalInfo*,PetscReal**,Mat,void*);
  PetscErrorCode (*computeobjectiveandgradient)(Tao,Vec,PetscReal*,Vec,void*);
  PetscErrorCode (*computeobjective)(Tao,Vec,PetscReal*,void*);
  PetscErrorCode (*computegradient)(Tao,Vec,Vec,void*);
  PetscErrorCode (*computehessian)(Tao,Vec,Mat,Mat,MatStructure*,void*);
  PetscErrorCode (*computebounds)(TaoDM, Vec, Vec);
  PetscErrorCode (*computeinitialguess)(TaoDM, Vec);
};
#define MAXTAODMMONITORS 10
struct _p_TaoDM {
  PETSCHEADER(struct _TaoDMOps);
  void           *coarselevel;
  DM             dm;  /* Grid information at this level */
  Vec            x;   /* solution on this level */
  Mat            hessian;   /* Hessian on this level */
  Mat            hessian_pre;
  Mat            R;   /* Restriction to next coarser level  (not defined for level 0) */
  Vec            Rscale;
  PetscInt       nlevels; /* # of levels above this one (== total levels for level 0) */
  void           *user; /* user context */
  MatType        mtype;
  TaoType        ttype;
  ISColoringType isctype;
  Tao            tao; /* Tao at this level */
  PetscReal      fatol;
  PetscReal      frtol;
  PetscReal      gatol;
  PetscReal      grtol;
  PetscReal      gttol;
  PetscInt       ksp_its;
  void           *userfctx;
  void           *usergctx;
  void           *userhctx;
  void           *userfgctx;
  void           *userpremonitor[MAXTAODMMONITORS];
  void           *userpostmonitor[MAXTAODMMONITORS];
  PetscErrorCode (*prelevelmonitor[MAXTAODMMONITORS])(TaoDM,PetscInt,void*);
  PetscErrorCode (*postlevelmonitor[MAXTAODMMONITORS])(TaoDM,PetscInt,void*);
  PetscInt       npremonitors;
  PetscInt       npostmonitors;
};

#endif
