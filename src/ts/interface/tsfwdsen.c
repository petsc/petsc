#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

PetscLogEvent TS_ForwardStep;

/*@
  TSForwardSetUp - Sets up the internal data structures for the later use
  of forward sensitivity analysis

  Collective on TS

  Input Parameter:
. ts - the TS context obtained from TSCreate()

  Level: advanced

.keywords: TS, forward sensitivity, setup

.seealso: TSCreate(), TSDestroy(), TSSetUp()
@*/
PetscErrorCode  TSForwardSetUp(TS ts)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->forwardsetupcalled) PetscFunctionReturn(0);

  if (ts->vec_costintegral && !ts->vecs_integral_sensip ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TSForwardSetIntegralGradients() before TSSetCostIntegrand()");

  if (ts->vecs_integral_sensip) {
    ierr = VecDuplicateVecs(ts->vec_sol,ts->numcost,&ts->vecs_drdy);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vecs_integral_sensip[0],ts->numcost,&ts->vecs_drdp);CHKERRQ(ierr);
  }
  if (ts->ops->forwardsetup) {
    ierr = (*ts->ops->forwardsetup)(ts);CHKERRQ(ierr);
  }
  ts->forwardsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetIntegralGradients - Set the vectors holding forward sensitivities of the integral term.

  Input Parameter:
. ts- the TS context obtained from TSCreate()
. numfwdint- number of integrals
. vp = the vectors containing the gradients for each integral w.r.t. parameters

  Level: intermediate

.keywords: TS, forward sensitivity

.seealso: TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardSetIntegralGradients(TS ts,PetscInt numfwdint,Vec *vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->numcost && ts->numcost!=numfwdint) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2rd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand()");
  if (!ts->numcost) ts->numcost = numfwdint;

  ts->vecs_integral_sensip = vp;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetIntegralGradients - Returns the forward sensitivities ofthe integral term.

  Input Parameter:
. ts- the TS context obtained from TSCreate()

  Output Parameter:
. vp = the vectors containing the gradients for each integral w.r.t. parameters

  Level: intermediate

.keywords: TS, forward sensitivity

.seealso: TSForwardSetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardGetIntegralGradients(TS ts,PetscInt *numfwdint,Vec **vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(vp,3);
  if (numfwdint) *numfwdint = ts->numcost;
  if (vp) *vp = ts->vecs_integral_sensip;
  PetscFunctionReturn(0);
}

/*@
  TSForwardStep - Compute the forward sensitivity for one time step.

  Collective on TS

  Input Arguments:
. ts - time stepping context

  Level: advanced

  Notes:
  This function cannot be called until TSStep() has been completed.

.keywords: TS, forward sensitivity

.seealso: TSForwardSetSensitivities(), TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardSetUp()
@*/
PetscErrorCode TSForwardStep(TS ts)
{
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->forwardstep) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide forward sensitivity analysis",((PetscObject)ts)->type_name);
  ierr = PetscLogEventBegin(TS_ForwardStep,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*ts->ops->forwardstep)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_ForwardStep,ts,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetSensitivities - Sets the initial value of the trajectory sensitivities of solution  w.r.t. the problem parameters and initial values.

  Logically Collective on TS and Vec

  Input Parameters:
+ ts - the TS context obtained from TSCreate()
. nump - number of parameters
- Smat - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

  Level: beginner

  Notes:
  Forward sensitivity is also called 'trajectory sensitivity' in some fields such as power systems.
  This function turns on a flag to trigger TSSolve() to compute forward sensitivities automatically.
  You must call this function before TSSolve().
  The entries in the sensitivity matrix must be correctly initialized with the values S = dy/dp|startingtime.

.keywords: TS, timestep, set, forward sensitivity, initial values

.seealso: TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardSetSensitivities(TS ts,PetscInt nump,Mat Smat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(Smat,MAT_CLASSID,3);
  ts->forward_solve  = PETSC_TRUE;
  ts->num_parameters = nump;
  ierr = PetscObjectReference((PetscObject)Smat);CHKERRQ(ierr);
  ierr = MatDestroy(&ts->mat_sensip);CHKERRQ(ierr);
  ts->mat_sensip = Smat;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetSensitivities - Returns the trajectory sensitivities

  Not Collective, but Vec returned is parallel if TS is parallel

  Output Parameter:
+ ts - the TS context obtained from TSCreate()
. nump - number of parameters
- Smat - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

  Level: intermediate

.keywords: TS, forward sensitivity

.seealso: TSForwardSetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardGetSensitivities(TS ts,PetscInt *nump,Mat *Smat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (nump) *nump = ts->num_parameters;
  if (Smat) *Smat = ts->mat_sensip;
  PetscFunctionReturn(0);
}
