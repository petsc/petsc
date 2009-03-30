#ifndef _PETSC_COMPAT_TS_H
#define _PETSC_COMPAT_TS_H

#include "private/tsimpl.h"

#undef __FUNCT__
#define __FUNCT__ "TSGetType_232"
static PETSC_UNUSED
PetscErrorCode TSGetType_232(TS ts, const TSType *type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSGetType(ts,(TSType *)type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define TSGetType TSGetType_232

#undef __FUNCT__
#define __FUNCT__ "TSSetMatrices_232"
static PETSC_UNUSED
PetscErrorCode TSSetMatrices_232(TS ts,
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
#define TSSetMatrices TSSetMatrices_232

#undef __FUNCT__
#define __FUNCT__ "TSSolve_232"
static PETSC_UNUSED
PetscErrorCode TSSolve_232(TS ts, Vec u)
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
#define TSSolve TSSolve_232

#undef __FUNCT__
#define __FUNCT__ "TSSetTime_232"
static PETSC_UNUSED
PetscErrorCode TSSetTime_232(TS ts, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->ptime = t;
  PetscFunctionReturn(0);
}
#define TSSetTime TSSetTime_232

#define TSMonitorSet TSSetMonitor
#define TSMonitorCancel TSClearMonitor
#define TSMonitorDefault TSDefaultMonitor
#define TSMonitorSolution TSVecViewMonitor
#define TSMonitorLG TSLGMonitor


#endif /* _PETSC_COMPAT_TS_H */
