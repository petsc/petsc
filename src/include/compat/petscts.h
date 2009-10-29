#ifndef _COMPAT_PETSC_TS_H
#define _COMPAT_PETSC_TS_H

#include "private/tsimpl.h"

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define TSEULER           TS_EULER
#define TSBEULER          TS_BEULER
#define TSPSEUDO          TS_PSEUDO
#define TSCRANK_NICHOLSON TS_CRANK_NICHOLSON
#define TSSUNDIALS        TS_SUNDIALS
#define TSRUNGE_KUTTA     TS_RUNGE_KUTTA
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSGL              "gl"
#define TSSSP             "ssp"
#endif



#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "TSGetType"
static PETSC_UNUSED
PetscErrorCode TSGetType_Compat(TS ts, const TSType *type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSGetType(ts,(TSType *)type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define TSGetType TSGetType_Compat
#endif

#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "TSSetMatrices"
static PETSC_UNUSED
PetscErrorCode TSSetMatrices(TS ts,
                             Mat Arhs,PetscErrorCode (*frhs)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),
                             Mat Alhs,PetscErrorCode (*flhs)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),
                             MatStructure flag,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (Arhs) {
    PetscValidHeaderSpecific(Arhs,MAT_COOKIE,2);
    PetscCheckSameComm(ts,1,Arhs,2);
    ierr = TSSetRHSMatrix(ts,Arhs,Arhs,frhs,ctx); CHKERRQ(ierr);
  }
  if (Alhs) {
    PetscValidHeaderSpecific(Alhs,MAT_COOKIE,4);
    PetscCheckSameComm(ts,1,Arhs,4);
    ierr = TSSetLHSMatrix(ts,Alhs,Alhs,flhs,ctx); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "TSSolve"
static PETSC_UNUSED
PetscErrorCode TSSolve(TS ts, Vec u)
{
  PetscInt       steps;
  PetscReal      ptime;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  /* set solution vector if provided */
  if (u) { ierr = TSSetSolution(ts, u); CHKERRQ(ierr); }
  /* reset time step and iteration counters */
  ts->steps = 0; ts->linear_its = 0; ts->nonlinear_its = 0;
  /* steps the requested number of timesteps. */
  ierr = TSStep(ts, &steps, &ptime);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "TSSetTime"
static PETSC_UNUSED
PetscErrorCode TSSetTime(TS ts, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->ptime = t;
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(2,3,2)
#define TSMonitorSet TSSetMonitor
#define TSMonitorCancel TSClearMonitor
#define TSMonitorDefault TSDefaultMonitor
#define TSMonitorSolution TSVecViewMonitor
#define TSMonitorLG TSLGMonitor
#endif

#endif /* _COMPAT_PETSC_TS_H */
