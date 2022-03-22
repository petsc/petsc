#include <petscconvest.h>            /*I "petscconvest.h" I*/
#include <petscts.h>
#include <petscdmplex.h>

#include <petsc/private/petscconvestimpl.h>

static PetscErrorCode PetscConvEstSetTS_Private(PetscConvEst ce, PetscObject solver)
{
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetClassId(ce->solver, &id);CHKERRQ(ierr);
  PetscCheck(id == TS_CLASSID,PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "Solver was not a TS");
  ierr = TSGetDM((TS) ce->solver, &ce->idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstInitGuessTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSComputeInitialCondition((TS) ce->solver, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstComputeErrorTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u, PetscReal errors[])
{
  TS               ts = (TS) ce->solver;
  PetscErrorCode (*exactError)(TS, Vec, Vec);
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = TSGetComputeExactError(ts, &exactError);CHKERRQ(ierr);
  if (exactError) {
    Vec      e;
    PetscInt f;

    ierr = VecDuplicate(u, &e);CHKERRQ(ierr);
    ierr = TSComputeExactError(ts, u, e);CHKERRQ(ierr);
    ierr = VecNorm(e, NORM_2, errors);CHKERRQ(ierr);
    for (f = 1; f < ce->Nf; ++f) errors[f] = errors[0];
    ierr = VecDestroy(&e);CHKERRQ(ierr);
  } else {
    PetscReal t;

    ierr = TSGetSolveTime(ts, &t);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm, t, ce->exactSol, ce->ctxs, u, errors);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstGetConvRateTS_Temporal_Private(PetscConvEst ce, PetscReal alpha[])
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
      dt[r] = dt[r-1]/ce->r;
      Ns    = PetscCeilReal(Ns*ce->r);
    }
    ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts, 0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, dt[r]);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts, Ns);CHKERRQ(ierr);
    ierr = PetscConvEstComputeInitialGuess(ce, r, NULL, u);CHKERRQ(ierr);
    ierr = TSSolve(ts, u);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ce->event, ce, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscConvEstComputeError(ce, r, ce->idm, u, &ce->errors[r*Nf]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ce->event, ce, 0, 0, 0);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      ce->dofs[r*Nf+f] = 1.0/dt[r];
      ierr = PetscLogEventSetDof(ce->event, f, ce->dofs[r*Nf+f]);CHKERRQ(ierr);
      ierr = PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]);CHKERRQ(ierr);
    }
    /* Monitor */
    ierr = PetscConvEstMonitorDefault(ce, r);CHKERRQ(ierr);
  }
  /* Fit convergence rate */
  if (Nr) {
    ierr = PetscMalloc2(Nr+1, &x, Nr+1, &y);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      for (r = 0; r <= Nr; ++r) {
        x[r] = PetscLog10Real(dt[r]);
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

static PetscErrorCode PetscConvEstGetConvRateTS_Spatial_Private(PetscConvEst ce, PetscReal alpha[])
{
  TS             ts = (TS) ce->solver;
  Vec            uInitial;
  DM            *dm;
  PetscObject    disc;
  PetscReal     *x, *y, slope, intercept;
  PetscInt       Nr = ce->Nr, r, Nf = ce->Nf, f, dim, oldlevel, oldnlev;
  void          *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheck(ce->r == 2.0,PetscObjectComm((PetscObject) ce), PETSC_ERR_SUP, "Only refinement factor 2 is currently supported (not %g)", (double) ce->r);
  ierr = DMGetDimension(ce->idm, &dim);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(ce->idm, &ctx);CHKERRQ(ierr);
  ierr = DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetRefineLevel(ce->idm, &oldlevel);CHKERRQ(ierr);
  ierr = PetscMalloc1((Nr+1), &dm);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &uInitial);CHKERRQ(ierr);
  /* Loop over meshes */
  dm[0] = ce->idm;
  for (r = 0; r <= Nr; ++r) {
    Vec           u;
#if defined(PETSC_USE_LOG)
    PetscLogStage stage;
#endif
    char          stageName[PETSC_MAX_PATH_LEN];
    const char   *dmname, *uname;

    ierr = PetscSNPrintf(stageName, PETSC_MAX_PATH_LEN-1, "ConvEst Refinement Level %D", r);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogStageGetId(stageName, &stage);CHKERRQ(ierr);
    if (stage < 0) {ierr = PetscLogStageRegister(stageName, &stage);CHKERRQ(ierr);}
#endif
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    if (r > 0) {
      if (!ce->noRefine) {
        ierr = DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]);CHKERRQ(ierr);
        ierr = DMSetCoarseDM(dm[r], dm[r-1]);CHKERRQ(ierr);
      } else {
        DM cdm, rcdm;

        ierr = DMClone(dm[r-1], &dm[r]);CHKERRQ(ierr);
        ierr = DMCopyDisc(dm[r-1], dm[r]);CHKERRQ(ierr);
        ierr = DMGetCoordinateDM(dm[r-1], &cdm);CHKERRQ(ierr);
        ierr = DMGetCoordinateDM(dm[r],   &rcdm);CHKERRQ(ierr);
        ierr = DMCopyDisc(cdm, rcdm);CHKERRQ(ierr);
      }
      ierr = DMCopyTransform(ce->idm, dm[r]);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) dm[r-1], &dmname);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) dm[r], dmname);CHKERRQ(ierr);
      for (f = 0; f <= Nf; ++f) {
        PetscErrorCode (*nspconstr)(DM, PetscInt, PetscInt, MatNullSpace *);

        ierr = DMGetNullSpaceConstructor(dm[r-1], f, &nspconstr);CHKERRQ(ierr);
        ierr = DMSetNullSpaceConstructor(dm[r],   f,  nspconstr);CHKERRQ(ierr);
      }
    }
    ierr = DMViewFromOptions(dm[r], NULL, "-conv_dm_view");CHKERRQ(ierr);
    /* Create solution */
    ierr = DMCreateGlobalVector(dm[r], &u);CHKERRQ(ierr);
    ierr = DMGetField(dm[r], 0, NULL, &disc);CHKERRQ(ierr);
    ierr = PetscObjectGetName(disc, &uname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, uname);CHKERRQ(ierr);
    /* Setup solver */
    ierr = TSReset(ts);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dm[r]);CHKERRQ(ierr);
    ierr = DMTSSetBoundaryLocal(dm[r], DMPlexTSComputeBoundary, ctx);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm[r], DMPlexTSComputeIFunctionFEM, ctx);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm[r], DMPlexTSComputeIJacobianFEM, ctx);CHKERRQ(ierr);
    ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts, 0);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    /* Create initial guess */
    ierr = PetscConvEstComputeInitialGuess(ce, r, dm[r], u);CHKERRQ(ierr);
    ierr = TSSolve(ts, u);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ce->event, ce, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscConvEstComputeError(ce, r, dm[r], u, &ce->errors[r*Nf]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ce->event, ce, 0, 0, 0);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscSection s, fs;
      PetscInt     lsize;

      /* Could use DMGetOutputDM() to add in Dirichlet dofs */
      ierr = DMGetLocalSection(dm[r], &s);CHKERRQ(ierr);
      ierr = PetscSectionGetField(s, f, &fs);CHKERRQ(ierr);
      ierr = PetscSectionGetConstrainedStorageSize(fs, &lsize);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&lsize, &ce->dofs[r*Nf+f], 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) ts));CHKERRMPI(ierr);
      ierr = PetscLogEventSetDof(ce->event, f, ce->dofs[r*Nf+f]);CHKERRQ(ierr);
      ierr = PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]);CHKERRQ(ierr);
    }
    /* Monitor */
    ierr = PetscConvEstMonitorDefault(ce, r);CHKERRQ(ierr);
    if (!r) {
      /* PCReset() does not wipe out the level structure */
      SNES snes;
      KSP  ksp;
      PC   pc;

      ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
      ierr = PCMGGetLevels(pc, &oldnlev);CHKERRQ(ierr);
    }
    /* Cleanup */
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  for (r = 1; r <= Nr; ++r) {
    ierr = DMDestroy(&dm[r]);CHKERRQ(ierr);
  }
  /* Fit convergence rate */
  ierr = PetscMalloc2(Nr+1, &x, Nr+1, &y);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    for (r = 0; r <= Nr; ++r) {
      x[r] = PetscLog10Real(ce->dofs[r*Nf+f]);
      y[r] = PetscLog10Real(ce->errors[r*Nf+f]);
    }
    ierr = PetscLinearRegression(Nr+1, x, y, &slope, &intercept);CHKERRQ(ierr);
    /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
    alpha[f] = -slope * dim;
  }
  ierr = PetscFree2(x, y);CHKERRQ(ierr);
  ierr = PetscFree(dm);CHKERRQ(ierr);
  /* Restore solver */
  ierr = TSReset(ts);CHKERRQ(ierr);
  {
    /* PCReset() does not wipe out the level structure */
    SNES snes;
    KSP  ksp;
    PC   pc;

    ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCMGSetLevels(pc, oldnlev, NULL);CHKERRQ(ierr);
    ierr = DMSetRefineLevel(ce->idm, oldlevel);CHKERRQ(ierr); /* The damn DMCoarsen() calls in PCMG can reset this */
  }
  ierr = TSSetDM(ts, ce->idm);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(ce->idm, DMPlexTSComputeBoundary, ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(ce->idm, DMPlexTSComputeIFunctionFEM, ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(ce->idm, DMPlexTSComputeIJacobianFEM, ctx);CHKERRQ(ierr);
  ierr = TSSetConvergedReason(ts, TS_CONVERGED_ITERATING);CHKERRQ(ierr);
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts, 0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, uInitial);CHKERRQ(ierr);
  ierr = PetscConvEstComputeInitialGuess(ce, 0, NULL, uInitial);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscConvEstUseTS(PetscConvEst ce, PetscBool checkTemporal)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  ce->ops->setsolver     = PetscConvEstSetTS_Private;
  ce->ops->initguess     = PetscConvEstInitGuessTS_Private;
  ce->ops->computeerror  = PetscConvEstComputeErrorTS_Private;
  if (checkTemporal) {
    ce->ops->getconvrate = PetscConvEstGetConvRateTS_Temporal_Private;
  } else {
    ce->ops->getconvrate = PetscConvEstGetConvRateTS_Spatial_Private;
  }
  PetscFunctionReturn(0);
}
