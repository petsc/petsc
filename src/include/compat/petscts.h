#ifndef _COMPAT_PETSC_TS_H
#define _COMPAT_PETSC_TS_H

#include "private/tsimpl.h"

#if PETSC_VERSION_(3,1,0)
#define TSCN TSCRANK_NICHOLSON
#define TSRK TSRUNGE_KUTTA
#endif

#if (PETSC_VERSION_(3,0,0))
#define TSEULER           TS_EULER
#define TSBEULER          TS_BEULER
#define TSPSEUDO          TS_PSEUDO
#define TSCN              TS_CRANK_NICHOLSON
#define TSSUNDIALS        TS_SUNDIALS
#define TSRK              TS_RUNGE_KUTTA
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSGL              "gl"
#define TSSSP             "ssp"
#endif

#if (PETSC_VERSION_(3,0,0))
#define PetscTS_ERR_SUP                                                     \
  PetscFunctionBegin;                                                       \
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version"); \
  PetscFunctionReturn(PETSC_ERR_SUP);

typedef PetscErrorCode (*TSIFunction)(TS,PetscReal,Vec,Vec,Vec,void*);
typedef PetscErrorCode (*TSIJacobian)(TS,PetscReal,Vec,Vec,PetscReal,
                                      Mat*,Mat*,MatStructure*,void*);
#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction"
static PETSC_UNUSED
PetscErrorCode TSSetIFunction(TS ts,TSIFunction f,void *ctx)
{PetscTS_ERR_SUP}
#undef __FUNCT__
#define __FUNCT__ "TSSetIJacobian"
static PETSC_UNUSED
PetscErrorCode TSSetIJacobian(TS ts,Mat A,Mat B,TSIJacobian j,void *ctx)
{PetscTS_ERR_SUP}
#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction"
static PETSC_UNUSED
PetscErrorCode TSComputeIFunction(TS ts,PetscReal t,Vec x,Vec Xdot,Vec f)
{PetscTS_ERR_SUP}
#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian"
static PETSC_UNUSED
PetscErrorCode TSComputeIJacobian(TS ts,
                                  PetscReal t,Vec x,Vec Xdot,PetscReal a,
                                  Mat *A,Mat *B,MatStructure *flag)
{PetscTS_ERR_SUP}
#undef __FUNCT__
#define __FUNCT__ "TSGetIJacobian"
static PETSC_UNUSED
PetscErrorCode TSGetIJacobian(TS ts,Mat *A,Mat *B,
                              TSIJacobian *j,void **ctx)
{PetscTS_ERR_SUP}
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "TSThetaSetTheta"
static PETSC_UNUSED
PetscErrorCode TSThetaSetTheta(TS ts,PetscReal theta)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(theta,2);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSThetaSetTheta_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ts,theta);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSThetaGetTheta"
static PETSC_UNUSED
PetscErrorCode TSThetaGetTheta(TS ts,PetscReal *theta)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal*);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(theta,2);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSThetaGetTheta_C",(void(**)(void))&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"TS type %s",((PetscObject)ts)->type_name);
  ierr = (*f)(ts,theta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#endif /* _COMPAT_PETSC_TS_H */
