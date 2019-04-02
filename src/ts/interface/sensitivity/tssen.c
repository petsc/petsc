#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscdraw.h>

PetscLogEvent TS_AdjointStep, TS_ForwardStep;

/* ------------------------ Sensitivity Context ---------------------------*/

/*@C
  TSSetRHSJacobianP - Sets the function that computes the Jacobian of G w.r.t. the parameters p where y_t = G(y,p,t), as well as the location to store the matrix.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
$ func (TS ts,PetscReal t,Vec y,Mat A,void *ctx);
+   t - current timestep
.   y - input vector (current ODE solution)
.   A - output matrix
-   ctx - [optional] user-defined function context

  Level: intermediate

  Notes:
    Amat has the same number of rows and the same row parallel layout as u, Amat has the same number of columns and parallel layout as p

.keywords: TS, sensitivity
.seealso:
@*/
PetscErrorCode TSSetRHSJacobianP(TS ts,Mat Amat,PetscErrorCode (*func)(TS,PetscReal,Vec,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);

  ts->rhsjacobianp    = func;
  ts->rhsjacobianpctx = ctx;
  if(Amat) {
    ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Jacp);CHKERRQ(ierr);
    ts->Jacp = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeRHSJacobianP - Runs the user-defined JacobianP function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Level: developer

.keywords: TS, sensitivity
.seealso: TSSetRHSJacobianP()
@*/
PetscErrorCode TSComputeRHSJacobianP(TS ts,PetscReal t,Vec X,Mat Amat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidPointer(Amat,4);

  PetscStackPush("TS user JacobianP function for sensitivity analysis");
  ierr = (*ts->rhsjacobianp)(ts,t,X,Amat,ts->rhsjacobianpctx); CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
    TSSetCostIntegrand - Sets the routine for evaluating the integral term in one or more cost functions

    Logically Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   numcost - number of gradients to be computed, this is the number of cost functions
.   costintegral - vector that stores the integral values
.   rf - routine for evaluating the integrand function
.   drdyf - function that computes the gradients of the r's with respect to y
.   drdpf - function that computes the gradients of the r's with respect to p, can be NULL if parametric sensitivity is not desired (mu=NULL)
.   fwd - flag indicating whether to evaluate cost integral in the forward run or the adjoint run
-   ctx - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

    Calling sequence of rf:
$   PetscErrorCode rf(TS ts,PetscReal t,Vec y,Vec f,void *ctx);

    Calling sequence of drdyf:
$   PetscErroCode drdyf(TS ts,PetscReal t,Vec y,Vec *drdy,void *ctx);

    Calling sequence of drdpf:
$   PetscErroCode drdpf(TS ts,PetscReal t,Vec y,Vec *drdp,void *ctx);

    Level: intermediate

    Notes:
    For optimization there is usually a single cost function (numcost = 1). For sensitivities there may be multiple cost functions

.keywords: TS, sensitivity analysis, timestep, set, quadrature, function

.seealso: TSSetRHSJacobianP(), TSGetCostGradients(), TSSetCostGradients()
@*/
PetscErrorCode TSSetCostIntegrand(TS ts,PetscInt numcost,Vec costintegral,PetscErrorCode (*rf)(TS,PetscReal,Vec,Vec,void*),
                                                          PetscErrorCode (*drdyf)(TS,PetscReal,Vec,Vec*,void*),
                                                          PetscErrorCode (*drdpf)(TS,PetscReal,Vec,Vec*,void*),
                                                          PetscBool fwd,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (costintegral) PetscValidHeaderSpecific(costintegral,VEC_CLASSID,3);
  if (ts->numcost && ts->numcost!=numcost) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2rd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostGradients() or TSForwardSetIntegralGradients()");
  if (!ts->numcost) ts->numcost=numcost;

  if (costintegral) {
    ierr = PetscObjectReference((PetscObject)costintegral);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vec_costintegral);CHKERRQ(ierr);
    ts->vec_costintegral = costintegral;
  } else {
    if (!ts->vec_costintegral) { /* Create a seq vec if user does not provide one */
      ierr = VecCreateSeq(PETSC_COMM_SELF,numcost,&ts->vec_costintegral);CHKERRQ(ierr);
    } else {
      ierr = VecSet(ts->vec_costintegral,0.0);CHKERRQ(ierr);
    }
  }
  if (!ts->vec_costintegrand) {
    ierr = VecDuplicate(ts->vec_costintegral,&ts->vec_costintegrand);CHKERRQ(ierr);
  } else {
    ierr = VecSet(ts->vec_costintegrand,0.0);CHKERRQ(ierr);
  }
  ts->costintegralfwd  = fwd; /* Evaluate the cost integral in forward run if fwd is true */
  ts->costintegrand    = rf;
  ts->costintegrandctx = ctx;
  ts->drdyfunction     = drdyf;
  ts->drdpfunction     = drdpf;
  PetscFunctionReturn(0);
}

/*@
   TSGetCostIntegral - Returns the values of the integral term in the cost functions.
   It is valid to call the routine after a backward run.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  v - the vector containing the integrals for each cost function

   Level: intermediate

.seealso: TSSetCostIntegrand()

.keywords: TS, sensitivity analysis
@*/
PetscErrorCode  TSGetCostIntegral(TS ts,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(v,2);
  *v = ts->vec_costintegral;
  PetscFunctionReturn(0);
}

/*@
   TSComputeCostIntegrand - Evaluates the integral function in the cost functions.

   Input Parameters:
+  ts - the TS context
.  t - current time
-  y - state vector, i.e. current solution

   Output Parameter:
.  q - vector of size numcost to hold the outputs

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the sensitivity analysis context.

   Level: developer

.keywords: TS, compute

.seealso: TSSetCostIntegrand()
@*/
PetscErrorCode TSComputeCostIntegrand(TS ts,PetscReal t,Vec y,Vec q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(q,VEC_CLASSID,4);

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,y,q,0);CHKERRQ(ierr);
  if (ts->costintegrand) {
    PetscStackPush("TS user integrand in the cost function");
    ierr = (*ts->costintegrand)(ts,t,y,q,ts->costintegrandctx);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    ierr = VecZeroEntries(q);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(TS_FunctionEval,ts,y,q,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSComputeDRDYFunction - Runs the user-defined DRDY function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSAdjointComputeDRDYFunction() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.keywords: TS, sensitivity
.seealso: TSSetCostIntegrand()
@*/
PetscErrorCode TSComputeDRDYFunction(TS ts,PetscReal t,Vec y,Vec *drdy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscStackPush("TS user DRDY function for sensitivity analysis");
  ierr = (*ts->drdyfunction)(ts,t,y,drdy,ts->costintegrandctx); CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@
  TSComputeDRDPFunction - Runs the user-defined DRDP function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSDRDPFunction() is typically used for sensitivity implementation,
  so most users would not generally call this routine themselves.

  Level: developer

.keywords: TS, sensitivity
.seealso: TSSetCostIntegrand()
@*/
PetscErrorCode TSComputeDRDPFunction(TS ts,PetscReal t,Vec y,Vec *drdp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscStackPush("TS user DRDP function for sensitivity analysis");
  ierr = (*ts->drdpfunction)(ts,t,y,drdp,ts->costintegrandctx); CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/* --------------------------- Adjoint sensitivity ---------------------------*/

/*@
   TSSetCostGradients - Sets the initial value of the gradients of the cost function w.r.t. initial values and w.r.t. the problem parameters
      for use by the TSAdjoint routines.

   Logically Collective on TS and Vec

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  lambda - gradients with respect to the initial condition variables, the dimension and parallel layout of these vectors is the same as the ODE solution vector
-  mu - gradients with respect to the parameters, the number of entries in these vectors is the same as the number of parameters

   Level: beginner

   Notes:
    the entries in these vectors must be correctly initialized with the values lamda_i = df/dy|finaltime  mu_i = df/dp|finaltime

   After TSAdjointSolve() is called the lamba and the mu contain the computed sensitivities

.keywords: TS, timestep, set, sensitivity, initial values
@*/
PetscErrorCode TSSetCostGradients(TS ts,PetscInt numcost,Vec *lambda,Vec *mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(lambda,2);
  ts->vecs_sensi  = lambda;
  ts->vecs_sensip = mu;
  if (ts->numcost && ts->numcost!=numcost) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The number of cost functions (2rd parameter of TSSetCostIntegrand()) is inconsistent with the one set by TSSetCostIntegrand");
  ts->numcost  = numcost;
  PetscFunctionReturn(0);
}

/*@
   TSGetCostGradients - Returns the gradients from the TSAdjointSolve()

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
+  lambda - vectors containing the gradients of the cost functions with respect to the ODE/DAE solution variables
-  mu - vectors containing the gradients of the cost functions with respect to the problem parameters

   Level: intermediate

.seealso: TSGetTimeStep()

.keywords: TS, timestep, get, sensitivity
@*/
PetscErrorCode TSGetCostGradients(TS ts,PetscInt *numcost,Vec **lambda,Vec **mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (numcost) *numcost = ts->numcost;
  if (lambda)  *lambda  = ts->vecs_sensi;
  if (mu)      *mu      = ts->vecs_sensip;
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSetUp - Sets up the internal data structures for the later use
   of an adjoint solver

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: advanced

.keywords: TS, timestep, setup

.seealso: TSCreate(), TSAdjointStep(), TSSetCostGradients()
@*/
PetscErrorCode TSAdjointSetUp(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->adjointsetupcalled) PetscFunctionReturn(0);
  if (!ts->vecs_sensi) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetCostGradients() first");
  if (ts->vecs_sensip && !ts->Jacp) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"Must call TSAdjointSetRHSJacobian() first");

  if (ts->vec_costintegral) { /* if there is integral in the cost function */
    ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&ts->vecs_drdy);CHKERRQ(ierr);
    if (ts->vecs_sensip){
      ierr = VecDuplicateVecs(ts->vecs_sensip[0],ts->numcost,&ts->vecs_drdp);CHKERRQ(ierr);
    }
  }

  if (ts->ops->adjointsetup) {
    ierr = (*ts->ops->adjointsetup)(ts);CHKERRQ(ierr);
  }
  ts->adjointsetupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSetSteps - Sets the number of steps the adjoint solver should take backward in time

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  steps - number of steps to use

   Level: intermediate

   Notes:
    Normally one does not call this and TSAdjointSolve() integrates back to the original timestep. One can call this
          so as to integrate back to less than the original timestep

.keywords: TS, timestep, set, maximum, iterations

.seealso: TSSetExactFinalTime()
@*/
PetscErrorCode  TSAdjointSetSteps(TS ts,PetscInt steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,steps,2);
  if (steps < 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Cannot step back a negative number of steps");
  if (steps > ts->steps) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Cannot step back more than the total number of forward steps");
  ts->adjoint_max_steps = steps;
  PetscFunctionReturn(0);
}

/*@C
  TSAdjointSetRHSJacobian - Deprecated, use TSSetRHSJacobianP()

  Level: deprecated

@*/
PetscErrorCode TSAdjointSetRHSJacobian(TS ts,Mat Amat,PetscErrorCode (*func)(TS,PetscReal,Vec,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);

  ts->rhsjacobianp    = func;
  ts->rhsjacobianpctx = ctx;
  if(Amat) {
    ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Jacp);CHKERRQ(ierr);
    ts->Jacp = Amat;
  }
  PetscFunctionReturn(0);
}

/*@C
  TSAdjointComputeRHSJacobian - Deprecated, use TSComputeRHSJacobianP()

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidPointer(Amat,4);

  PetscStackPush("TS user JacobianP function for sensitivity analysis");
  ierr = (*ts->rhsjacobianp)(ts,t,X,Amat,ts->rhsjacobianpctx); CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointComputeDRDYFunction - Deprecated, use TSComputeDRDYFunction()

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeDRDYFunction(TS ts,PetscReal t,Vec y,Vec *drdy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscStackPush("TS user DRDY function for sensitivity analysis");
  ierr = (*ts->drdyfunction)(ts,t,y,drdy,ts->costintegrandctx); CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@
  TSAdjointComputeDRDPFunction - Deprecated, use TSComputeDRDPFunction()

  Level: deprecated

@*/
PetscErrorCode TSAdjointComputeDRDPFunction(TS ts,PetscReal t,Vec y,Vec *drdp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscStackPush("TS user DRDP function for sensitivity analysis");
  ierr = (*ts->drdpfunction)(ts,t,y,drdp,ts->costintegrandctx); CHKERRQ(ierr);
  PetscStackPop;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSensi - monitors the first lambda sensitivity

   Level: intermediate

.keywords: TS, set, monitor

.seealso: TSAdjointMonitorSet()
@*/
PetscErrorCode TSAdjointMonitorSensi(TS ts,PetscInt step,PetscReal ptime,Vec v,PetscInt numcost,Vec *lambda,Vec *mu,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = VecView(lambda[0],viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on TS

   Input Parameters:
+  ts - TS object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
.  monitor - the monitor function
-  monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the TS or PetscViewer objects

   Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode TSAdjointMonitorSetFromOptions(TS ts,const char name[],const char help[], const char manual[],PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,PetscViewerAndFormat*),PetscErrorCode (*monitorsetup)(TS,PetscViewerAndFormat*))
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ts),((PetscObject) ts)->options,((PetscObject)ts)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewerAndFormat *vf;
    ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    if (monitorsetup) {
      ierr = (*monitorsetup)(ts,vf);CHKERRQ(ierr);
    }
    ierr = TSAdjointMonitorSet(ts,(PetscErrorCode (*)(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorSet - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  adjointmonitor - monitoring routine
.  adjointmctx - [optional] user-defined context for private data for the
             monitor routine (use NULL if no context is desired)
-  adjointmonitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Calling sequence of monitor:
$    int adjointmonitor(TS ts,PetscInt steps,PetscReal time,Vec u,PetscInt numcost,Vec *lambda, Vec *mu,void *adjointmctx)

+    ts - the TS context
.    steps - iteration number (after the final time step the monitor routine is called with a step of -1, this is at the final time which may have
                               been interpolated to)
.    time - current time
.    u - current iterate
.    numcost - number of cost functionos
.    lambda - sensitivities to initial conditions
.    mu - sensitivities to parameters
-    adjointmctx - [optional] adjoint monitoring context

   Notes:
   This routine adds an additional monitor to the list of monitors that
   already has been loaded.

   Fortran Notes:
    Only a single monitor function can be set for each TS object

   Level: intermediate

.keywords: TS, timestep, set, adjoint, monitor

.seealso: TSAdjointMonitorCancel()
@*/
PetscErrorCode TSAdjointMonitorSet(TS ts,PetscErrorCode (*adjointmonitor)(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,void*),void *adjointmctx,PetscErrorCode (*adjointmdestroy)(void**))
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numbermonitors;i++) {
    ierr = PetscMonitorCompare((PetscErrorCode (*)(void))adjointmonitor,adjointmctx,adjointmdestroy,(PetscErrorCode (*)(void))ts->adjointmonitor[i],ts->adjointmonitorcontext[i],ts->adjointmonitordestroy[i],&identical);CHKERRQ(ierr);
    if (identical) PetscFunctionReturn(0);
  }
  if (ts->numberadjointmonitors >= MAXTSMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many adjoint monitors set");
  ts->adjointmonitor[ts->numberadjointmonitors]          = adjointmonitor;
  ts->adjointmonitordestroy[ts->numberadjointmonitors]   = adjointmdestroy;
  ts->adjointmonitorcontext[ts->numberadjointmonitors++] = (void*)adjointmctx;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorCancel - Clears all the adjoint monitors that have been set on a time-step object.

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.keywords: TS, timestep, set, adjoint, monitor

.seealso: TSAdjointMonitorSet()
@*/
PetscErrorCode TSAdjointMonitorCancel(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numberadjointmonitors; i++) {
    if (ts->adjointmonitordestroy[i]) {
      ierr = (*ts->adjointmonitordestroy[i])(&ts->adjointmonitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ts->numberadjointmonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorDefault - the default monitor of adjoint computations

   Level: intermediate

.keywords: TS, set, monitor

.seealso: TSAdjointMonitorSet()
@*/
PetscErrorCode TSAdjointMonitorDefault(TS ts,PetscInt step,PetscReal ptime,Vec v,PetscInt numcost,Vec *lambda,Vec *mu,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D TS dt %g time %g%s",step,(double)ts->time_step,(double)ptime,ts->steprollback ? " (r)\n" : "\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitorDrawSensi - Monitors progress of the adjoint TS solvers by calling
   VecView() for the sensitivities to initial states at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current state
.  numcost - number of cost functions
.  lambda - sensitivities to initial conditions
.  mu - sensitivities to parameters
-  dummy - either a viewer or NULL

   Level: intermediate

.keywords: TS,  vector, adjoint, monitor, view

.seealso: TSAdjointMonitorSet(), TSAdjointMonitorDefault(), VecView()
@*/
PetscErrorCode TSAdjointMonitorDrawSensi(TS ts,PetscInt step,PetscReal ptime,Vec u,PetscInt numcost,Vec *lambda,Vec *mu,void *dummy)
{
  PetscErrorCode   ierr;
  TSMonitorDrawCtx ictx = (TSMonitorDrawCtx)dummy;
  PetscDraw        draw;
  PetscReal        xl,yl,xr,yr,h;
  char             time[32];

  PetscFunctionBegin;
  if (!(((ictx->howoften > 0) && (!(step % ictx->howoften))) || ((ictx->howoften == -1) && ts->reason))) PetscFunctionReturn(0);

  ierr = VecView(lambda[0],ictx->viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(ictx->viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscSNPrintf(time,32,"Timestep %d Time %g",(int)step,(double)ptime);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  h    = yl + .95*(yr - yl);
  ierr = PetscDrawStringCentered(draw,.5*(xl+xr),h,PETSC_DRAW_BLACK,time);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSAdjointSetFromOptions - Sets various TSAdjoint parameters from user options.

   Collective on TSAdjoint

   Input Parameter:
.  ts - the TS context

   Options Database Keys:
+  -ts_adjoint_solve <yes,no> After solving the ODE/DAE solve the adjoint problem (requires -ts_save_trajectory)
.  -ts_adjoint_monitor - print information at each adjoint time step
-  -ts_adjoint_monitor_draw_sensi - monitor the sensitivity of the first cost function wrt initial conditions (lambda[0]) graphically

   Level: developer

   Notes:
    This is not normally called directly by users

.keywords: TS, trajectory, timestep, set, options, database

.seealso: TSSetSaveTrajectory(), TSTrajectorySetUp()
*/
PetscErrorCode TSAdjointSetFromOptions(PetscOptionItems *PetscOptionsObject,TS ts)
{
  PetscBool      tflg,opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  ierr = PetscOptionsHead(PetscOptionsObject,"TS Adjoint options");CHKERRQ(ierr);
  tflg = ts->adjoint_solve ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscOptionsBool("-ts_adjoint_solve","Solve the adjoint problem immediately after solving the forward problem","",tflg,&tflg,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
    ts->adjoint_solve = tflg;
  }
  ierr = TSAdjointMonitorSetFromOptions(ts,"-ts_adjoint_monitor","Monitor adjoint timestep size","TSAdjointMonitorDefault",TSAdjointMonitorDefault,NULL);CHKERRQ(ierr);
  ierr = TSAdjointMonitorSetFromOptions(ts,"-ts_adjoint_monitor_sensi","Monitor sensitivity in the adjoint computation","TSAdjointMonitorSensi",TSAdjointMonitorSensi,NULL);CHKERRQ(ierr);
  opt  = PETSC_FALSE;
  ierr = PetscOptionsName("-ts_adjoint_monitor_draw_sensi","Monitor adjoint sensitivities (lambda only) graphically","TSAdjointMonitorDrawSensi",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscInt         howoften = 1;

    ierr = PetscOptionsInt("-ts_adjoint_monitor_draw_sensi","Monitor adjoint sensitivities (lambda only) graphically","TSAdjointMonitorDrawSensi",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts),0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSAdjointMonitorSet(ts,TSAdjointMonitorDrawSensi,ctx,(PetscErrorCode (*)(void**))TSMonitorDrawCtxDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdjointStep - Steps one time step backward in the adjoint run

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: intermediate

.keywords: TS, adjoint, step

.seealso: TSAdjointSetUp(), TSAdjointSolve()
@*/
PetscErrorCode TSAdjointStep(TS ts)
{
  DM               dm;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = TSAdjointSetUp(ts);CHKERRQ(ierr);

  ierr = VecViewFromOptions(ts->vec_sol,(PetscObject)ts,"-ts_view_solution");CHKERRQ(ierr);

  ts->reason = TS_CONVERGED_ITERATING;
  ts->ptime_prev = ts->ptime;
  if (!ts->ops->adjointstep) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed because the adjoint of  %s has not been implemented, try other time stepping methods for adjoint sensitivity analysis",((PetscObject)ts)->type_name);
  ierr = PetscLogEventBegin(TS_AdjointStep,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*ts->ops->adjointstep)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_AdjointStep,ts,0,0,0);CHKERRQ(ierr);
  ts->adjoint_steps++; ts->steps--;

  if (ts->reason < 0) {
    if (ts->errorifstepfailed) {
      if (ts->reason == TS_DIVERGED_NONLINEAR_SOLVE) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s, increase -ts_max_snes_failures or make negative to attempt recovery",TSConvergedReasons[ts->reason]);
      else if (ts->reason == TS_DIVERGED_STEP_REJECTED) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s, increase -ts_max_reject or make negative to attempt recovery",TSConvergedReasons[ts->reason]);
      else SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s",TSConvergedReasons[ts->reason]);
    }
  } else if (!ts->reason) {
    if (ts->adjoint_steps >= ts->adjoint_max_steps) ts->reason = TS_CONVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdjointSolve - Solves the discrete ajoint problem for an ODE/DAE

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database:
. -ts_adjoint_view_solution <viewerinfo> - views the first gradient with respect to the initial values

   Level: intermediate

   Notes:
   This must be called after a call to TSSolve() that solves the forward problem

   By default this will integrate back to the initial time, one can use TSAdjointSetSteps() to step back to a later time

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetCostGradients(), TSSetSolution(), TSAdjointStep()
@*/
PetscErrorCode TSAdjointSolve(TS ts)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSAdjointSetUp(ts);CHKERRQ(ierr);

  /* reset time step and iteration counters */
  ts->adjoint_steps     = 0;
  ts->ksp_its           = 0;
  ts->snes_its          = 0;
  ts->num_snes_failures = 0;
  ts->reject            = 0;
  ts->reason            = TS_CONVERGED_ITERATING;

  if (!ts->adjoint_max_steps) ts->adjoint_max_steps = ts->steps;
  if (ts->adjoint_steps >= ts->adjoint_max_steps) ts->reason = TS_CONVERGED_ITS;

  while (!ts->reason) {
    ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
    ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
    ierr = TSAdjointEventHandler(ts);CHKERRQ(ierr);
    ierr = TSAdjointStep(ts);CHKERRQ(ierr);
    if (ts->vec_costintegral && !ts->costintegralfwd) {
      ierr = TSAdjointCostIntegral(ts);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
  ts->solvetime = ts->ptime;
  ierr = TSTrajectoryViewFromOptions(ts->trajectory,NULL,"-ts_trajectory_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ts->vecs_sensi[0],(PetscObject) ts, "-ts_adjoint_view_solution");CHKERRQ(ierr);
  ts->adjoint_max_steps = 0;
  PetscFunctionReturn(0);
}

/*@C
   TSAdjointMonitor - Runs all user-provided adjoint monitor routines set using TSAdjointMonitorSet()

   Collective on TS

   Input Parameters:
+  ts - time stepping context obtained from TSCreate()
.  step - step number that has just completed
.  ptime - model time of the state
.  u - state at the current model time
.  numcost - number of cost functions (dimension of lambda  or mu)
.  lambda - vectors containing the gradients of the cost functions with respect to the ODE/DAE solution variables
-  mu - vectors containing the gradients of the cost functions with respect to the problem parameters

   Notes:
   TSAdjointMonitor() is typically used automatically within the time stepping implementations.
   Users would almost never call this routine directly.

   Level: developer

.keywords: TS, timestep
@*/
PetscErrorCode TSAdjointMonitor(TS ts,PetscInt step,PetscReal ptime,Vec u,PetscInt numcost,Vec *lambda, Vec *mu)
{
  PetscErrorCode ierr;
  PetscInt       i,n = ts->numberadjointmonitors;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,4);
  ierr = VecLockReadPush(u);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = (*ts->adjointmonitor[i])(ts,step,ptime,u,numcost,lambda,mu,ts->adjointmonitorcontext[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 TSAdjointCostIntegral - Evaluate the cost integral in the adjoint run.

 Collective on TS

 Input Arguments:
 .  ts - time stepping context

 Level: advanced

 Notes:
 This function cannot be called until TSAdjointStep() has been completed.

 .seealso: TSAdjointSolve(), TSAdjointStep
 @*/
PetscErrorCode TSAdjointCostIntegral(TS ts)
{
    PetscErrorCode ierr;
    PetscValidHeaderSpecific(ts,TS_CLASSID,1);
    if (!ts->ops->adjointintegral) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide integral evaluation in the adjoint run",((PetscObject)ts)->type_name);
    ierr = (*ts->ops->adjointintegral)(ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* ------------------ Forward (tangent linear) sensitivity  ------------------*/

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
PetscErrorCode TSForwardSetUp(TS ts)
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

/*@
   TSForwardCostIntegral - Evaluate the cost integral in the forward run.

   Collective on TS

   Input Arguments:
.  ts - time stepping context

   Level: advanced

   Notes:
   This function cannot be called until TSStep() has been completed.

.seealso: TSSolve(), TSAdjointCostIntegral()
@*/
PetscErrorCode TSForwardCostIntegral(TS ts)
{
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->forwardintegral) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide integral evaluation in the forward run",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->forwardintegral)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
