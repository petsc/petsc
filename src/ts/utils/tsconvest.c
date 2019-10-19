#include <petscconvest.h>            /*I "petscconvest.h" I*/
#include <petscts.h>

#include <petsc/private/petscconvestimpl.h>

static PetscErrorCode PetscConvEstSetTS_Private(PetscConvEst ce, PetscObject solver)
{
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetClassId(ce->solver, &id);CHKERRQ(ierr);
  if (id != TS_CLASSID) SETERRQ(PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "Solver was not a TS");
  ierr = TSGetDM((TS) ce->solver, &ce->idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstInitGuessTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSComputeInitialGuess((TS) ce->solver, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstComputeErrorTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u, PetscReal errors[])
{
  Vec            e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(u, &e);CHKERRQ(ierr);
  ierr = TSComputeExactError((TS) ce->solver, u, e);CHKERRQ(ierr);
  ierr = VecNorm(e, NORM_2, errors);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstGetConvRateTS_Private(PetscConvEst ce, PetscReal alpha[])
{
  TS             ts = (TS) ce->solver;
  Vec            u;
  PetscReal     *dt, *x, *y, slope, intercept;
  PetscInt       Ns, oNs, Nf = ce->Nf, f, Nr = ce->Nr, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nr+1, &dt);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt[0]);CHKERRQ(ierr);
  ierr = TSGetMaxSteps(ts, &oNs);CHKERRQ(ierr);
  Ns   = oNs;
  for (r = 0; r <= Nr; ++r) {
    if (r > 0) {
      dt[r] = dt[r-1]/2.0;
      Ns   *= 2;
    }
    ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts, 0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, dt[r]);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts, Ns);CHKERRQ(ierr);
    ierr = PetscConvEstComputeInitialGuess(ce, r, NULL, u);CHKERRQ(ierr);
    ierr = TSSolve(ts, u);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ce->event, ce, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscConvEstComputeError(ce, r, NULL, u, &ce->errors[r*Nf]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ce->event, ce, 0, 0, 0);CHKERRQ(ierr);
    for (f = 0; f < ce->Nf; ++f) {
      ierr = PetscLogEventSetDof(ce->event, f, 1.0/dt[r]);CHKERRQ(ierr);
      ierr = PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]);CHKERRQ(ierr);
    }
  }
  /* Fit convergence rate */
  if (Nr) {
    ierr = PetscMalloc2(Nr+1, &x, Nr+1, &y);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      for (r = 0; r <= Nr; ++r) {
        x[r] = PetscLog10Real(dt[r*Nf+f]);
        y[r] = PetscLog10Real(ce->errors[r*Nf+f]);
      }
      ierr = PetscLinearRegression(Nr+1, x, y, &slope, &intercept);CHKERRQ(ierr);
      /* Since lg err = s lg dt + b */
      alpha[f] = slope;
    }
    ierr = PetscFree2(x, y);CHKERRQ(ierr);
  }
  /* Reset solver */
  ierr = TSSetConvergedReason(ts, TS_CONVERGED_ITERATING);CHKERRQ(ierr);
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts, 0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, dt[0]);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, oNs);CHKERRQ(ierr);
  ierr = PetscConvEstComputeInitialGuess(ce, 0, NULL, u);CHKERRQ(ierr);
  ierr = PetscFree(dt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscConvEstUseTS(PetscConvEst ce)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  ce->ops->setsolver    = PetscConvEstSetTS_Private;
  ce->ops->initguess    = PetscConvEstInitGuessTS_Private;
  ce->ops->computeerror = PetscConvEstComputeErrorTS_Private;
  ce->ops->getconvrate  = PetscConvEstGetConvRateTS_Private;
  PetscFunctionReturn(0);
}
