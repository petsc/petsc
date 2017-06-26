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

  if (ts->vec_costintegral && !ts->vecs_integral_sensi && !ts->vecs_integral_sensip ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TSForwardSetIntegralGradients() before TSSetCostIntegrand()");

  if (ts->vecs_integral_sensi || ts->vecs_integral_sensip) {
    ierr = VecDuplicateVecs(ts->vec_sol,ts->numcost,&ts->vecs_drdy);CHKERRQ(ierr);
  }
  if (ts->vecs_integral_sensip) {
    ierr = VecDuplicateVecs(ts->vecs_integral_sensip[0],ts->numcost,&ts->vecs_drdp);CHKERRQ(ierr);
  }
  if (ts->ops->forwardsetup) {
    ierr = (*ts->ops->forwardsetup)(ts);CHKERRQ(ierr);
  }
  ts->forwardsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  TSForwardSetRHSJacobianP - Sets the function that computes the Jacobian of G w.r.t. the parameters p where y_t = G(y,p,t), as well as the location to store the vector array.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
$ func (TS ts,PetscReal t,Vec y,Vec* a,void *ctx);
+   t - current timestep
.   y - input vector (current ODE solution)
.   a - output vector array
-   ctx - [optional] user-defined function context

  Level: intermediate

  Notes: the number of vectors in a is the same as the number of parameters and each vector is of the same size as the system dimension.

.keywords: TS, forward sensitivity

.seealso: TSForwardSetSensitivities(), TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode  TSForwardSetRHSJacobianP(TS ts,Vec* a,PetscErrorCode (*func)(TS,PetscReal,Vec,Vec*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidPointer(a,2);

  ts->vecsrhsjacobianp    = func;
  ts->vecsrhsjacobianpctx = ctx;
  if(a) ts->vecs_jacp = a;
  PetscFunctionReturn(0);
}

/*@C
  TSForwardComputeRHSJacobianP - Runs the user-defined JacobianP function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Level: developer

.keywords: TS, forward sensitivity

.seealso: TSForwardSetRHSJacobianP()
@*/
PetscErrorCode  TSForwardComputeRHSJacobianP(TS ts,PetscReal t,Vec X,Vec* A )
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidPointer(A,4);
  if (ts->vecsrhsjacobianp) {
    PetscStackPush("TS user JacobianP function for sensitivity analysis");
    ierr = (*ts->vecsrhsjacobianp)(ts,t,X,A,ts->vecsrhsjacobianpctx); CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

/*@
  TSForwardSetIntegralGradients - Set the vectors holding forward sensitivities of the integral term.

  Input Parameter:
. ts- the TS context obtained from TSCreate()
. numfwdint- number of integrals
. v  = the vectors containing the gradients for each integral wrt initial values
. vp = the vectors containing the gradients for each integral wrt parameters

  Level: intermediate

.keywords: TS, forward sensitivity

.seealso: TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardSetIntegralGradients(TS ts,PetscInt numfwdint,Vec *vp,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->numcost && ts->numcost!=numfwdint) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2rd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand()");
  if (!ts->numcost) ts->numcost = numfwdint;

  ts->vecs_integral_sensi  = v;
  ts->vecs_integral_sensip = vp;
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetIntegralGradients - Returns the forward sensitivities ofthe integral term.

  Input Parameter:
. ts- the TS context obtained from TSCreate()

  Output Parameter:
. v  = the vectors containing the gradients for each integral wrt initial values
. vp = the vectors containing the gradients for each integral wrt parameters

  Level: intermediate

.keywords: TS, forward sensitivity

.seealso: TSForwardSetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardGetIntegralGradients(TS ts,PetscInt *numfwdint,Vec **v,Vec **vp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(v,2);
  if (numfwdint) *numfwdint = ts->numcost;
  if (v) *v = ts->vecs_integral_sensi;
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
. sp - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters
. num - number of initial values
- s - sensitivities with respect to the (selected) initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector

  Level: beginner

  Notes:
  Forward sensitivity is also called 'trajectory sensitivity' in some fields such as power systems.
  This function turns on a flag to trigger TSSolve() to compute forward sensitivities automatically.
  You must call this function before TSSolve().
  The entries in these vectors must be correctly initialized with the values s_i = dy/dp|startingtime.
  The two user-provided sensitivity vector arrays will be packed into one big array to simplify implementation.

.keywords: TS, timestep, set, forward sensitivity, initial values

.seealso: TSForwardGetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardSetSensitivities(TS ts,PetscInt nump,Vec *sp,PetscInt num,Vec *s)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->forward_solve     = PETSC_TRUE;
  ts->num_parameters    = sp ? nump:0;
  ts->num_initialvalues = s ? num:0;
  /* pack fwdsensi and fwdsensip into a big array */
  if (!ts->vecs_fwdsensipacked) {
    ierr = PetscMalloc1(num+nump,&ts->vecs_fwdsensipacked);CHKERRQ(ierr);
  }
  for (i=0; i<num; i++) ts->vecs_fwdsensipacked[i] = s[i];
  for (i=0; i<nump; i++) ts->vecs_fwdsensipacked[i+num] = sp[i];
  PetscFunctionReturn(0);
}

/*@
  TSForwardGetSensitivities - Returns the trajectory sensitivities

  Not Collective, but Vec returned is parallel if TS is parallel

  Output Parameter:
+ ts - the TS context obtained from TSCreate()
. nump - number of parameters
. sp - sensitivities with respect to the parameters, the number of entries in these vectors is the same as the number of parameters
. num - number of initial values
- s - sensitivities with respect to the (selected) initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector

  Level: intermediate

.keywords: TS, forward sensitivity

.seealso: TSForwardSetSensitivities(), TSForwardSetIntegralGradients(), TSForwardGetIntegralGradients(), TSForwardStep()
@*/
PetscErrorCode TSForwardGetSensitivities(TS ts,PetscInt *nump,Vec **sp,PetscInt *num,Vec **s)
{
  PetscFunctionBegin;
  if (nump) *nump = ts->num_parameters;
  if (num) *num   = ts->num_initialvalues;
  if (sp) *sp     = &ts->vecs_fwdsensipacked[(*num)];
  if (s) *s       = ts->vecs_fwdsensipacked;
  PetscFunctionReturn(0);
}
