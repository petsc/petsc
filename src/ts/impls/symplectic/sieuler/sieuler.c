/*
  Code for Timestepping with semi-implicit Euler for separable Hamiltonian systems

  pdot = -dH(q,p,t)/dq
  qdot = dH(q,p,t)/dp

  where the Hamiltonian can be split into the sum of kinetic energy and potential energy

  H(q,p,t) = T(p,t) + V(q,t).

  As a result, the system can be represented by

  pdot = f(q,t) = -dV(q,t)/dq
  qdot = g(p,t) = dT(p,t)/dp

  and solved with

  p_{n+1} = p_n + h*f(q_n,t_n)
  q+{n+1} = q_n + h*g(p_{n+1},t_n).

  The solution is represented by a nest vec [p,q].
  f and g are provided by RHSFunction1 and RHSFunction2 respectively.

  Reference: wikipedia
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
  Vec update;   /* a nest work vector for generalized coordinates */
} TS_SIEuler;

#undef __FUNCT__
#define __FUNCT__ "TSStep_SIEuler"
static PetscErrorCode TSStep_SIEuler(TS ts)
{
  TS_SIEuler     *sieuler = (TS_SIEuler*)ts->data;
  Vec            solution = ts->vec_sol,update = sieuler->update,q,p,q_update,p_update;
  PetscBool      stageok;
  PetscReal      next_time_step = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVec(solution,0,&p);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(solution,1,&q);CHKERRQ(ierr);

  ierr = VecNestGetSubVec(update,0,&p_update);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(update,1,&q_update);CHKERRQ(ierr);

  ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);
  ierr = TSComputeRHSFunctionSplit2w(ts,ts->ptime,q,p_update,1);CHKERRQ(ierr);
  /* update p */
  ierr = VecAXPY(p,ts->time_step,p_update);CHKERRQ(ierr);
  ierr = TSComputeRHSFunctionSplit2w(ts,ts->ptime,p,q_update,2);CHKERRQ(ierr);
  /* update q */
  ierr = VecAXPY(q,ts->time_step,q_update);CHKERRQ(ierr);
  ierr = TSPostStage(ts,ts->ptime,0,&solution);CHKERRQ(ierr);

  ierr = TSAdaptCheckStage(ts->adapt,ts,ts->ptime,solution,&stageok);CHKERRQ(ierr);
  if(!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
  ierr = TSFunctionDomainError(ts,ts->ptime+ts->time_step,update,&stageok);CHKERRQ(ierr);
  if(!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}

  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_SIEuler"
static PetscErrorCode TSSetUp_SIEuler(TS ts)
{
  TS_SIEuler           *sieuler = (TS_SIEuler*)ts->data;
  PetscErrorCode       ierr;
  VecType              vtype;
  PetscBool            isNestVec;
  TSRHSFunctionSplit2w rhsfunction1,rhsfunction2;

  PetscFunctionBegin;
  ierr =  TSGetRHSFunctionSplit2w(ts,NULL,&rhsfunction1,&rhsfunction2,NULL);CHKERRQ(ierr);
  if (!rhsfunction1 || !rhsfunction2) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSSetRHSFunctionSplit2w() in order to use -ts_type sieuler");
  ierr = VecGetType(ts->vec_sol,&vtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(vtype,VECNEST,&isNestVec);CHKERRQ(ierr);
  if (!isNestVec) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must use nest vector for the solution in order to use -ts_type sieuler");
  ierr = VecDuplicate(ts->vec_sol,&sieuler->update);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_SIEuler"
static PetscErrorCode TSReset_SIEuler(TS ts)
{
  TS_SIEuler     *sieuler = (TS_SIEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&sieuler->update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_SIEuler"
static PetscErrorCode TSDestroy_SIEuler(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_SIEuler(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_SIEuler"
static PetscErrorCode TSSetFromOptions_SIEuler(PetscOptionItems *PetscOptionsObject,TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_SIEuler"
static PetscErrorCode TSView_SIEuler(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_SIEuler"
static PetscErrorCode TSInterpolate_SIEuler(TS ts,PetscReal t,Vec X)
{
  TS_SIEuler     *sieuler = (TS_SIEuler*)ts->data;
  Vec            update = sieuler->update;
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWAXPY(X,-ts->time_step,update,ts->vec_sol);CHKERRQ(ierr);
  ierr = VecAXPBY(X,1.0-alpha,alpha,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeLinearStability_SIEuler"
static PetscErrorCode TSComputeLinearStability_SIEuler(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

/*MC
      TSSIEULER - Symplectic ODE solver using the semi implicit Euler method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSSIEULER

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_SIEuler"
PETSC_EXTERN PetscErrorCode TSCreate_SIEuler(TS ts)
{
  TS_SIEuler     *sieuler;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ts,&sieuler);CHKERRQ(ierr);
  ts->data = (void*)sieuler;

  ts->ops->setup           = TSSetUp_SIEuler;
  ts->ops->step            = TSStep_SIEuler;
  ts->ops->reset           = TSReset_SIEuler;
  ts->ops->destroy         = TSDestroy_SIEuler;
  ts->ops->setfromoptions  = TSSetFromOptions_SIEuler;
  ts->ops->view            = TSView_SIEuler;
  ts->ops->interpolate     = TSInterpolate_SIEuler;
  ts->ops->linearstability = TSComputeLinearStability_SIEuler;
  PetscFunctionReturn(0);
}
