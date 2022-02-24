#include <petscconvest.h>            /*I "petscconvest.h" I*/
#include <petscts.h>
#include <petscdmplex.h>

#include <petsc/private/petscconvestimpl.h>

static PetscErrorCode PetscConvEstSetTS_Private(PetscConvEst ce, PetscObject solver)
{
  PetscClassId   id;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetClassId(ce->solver, &id));
  PetscCheck(id == TS_CLASSID,PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "Solver was not a TS");
  CHKERRQ(TSGetDM((TS) ce->solver, &ce->idm));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstInitGuessTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u)
{
  PetscFunctionBegin;
  CHKERRQ(TSComputeInitialCondition((TS) ce->solver, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstComputeErrorTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u, PetscReal errors[])
{
  TS               ts = (TS) ce->solver;
  PetscErrorCode (*exactError)(TS, Vec, Vec);

  PetscFunctionBegin;
  CHKERRQ(TSGetComputeExactError(ts, &exactError));
  if (exactError) {
    Vec      e;
    PetscInt f;

    CHKERRQ(VecDuplicate(u, &e));
    CHKERRQ(TSComputeExactError(ts, u, e));
    CHKERRQ(VecNorm(e, NORM_2, errors));
    for (f = 1; f < ce->Nf; ++f) errors[f] = errors[0];
    CHKERRQ(VecDestroy(&e));
  } else {
    PetscReal t;

    CHKERRQ(TSGetSolveTime(ts, &t));
    CHKERRQ(DMComputeL2FieldDiff(dm, t, ce->exactSol, ce->ctxs, u, errors));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstGetConvRateTS_Temporal_Private(PetscConvEst ce, PetscReal alpha[])
{
  TS             ts = (TS) ce->solver;
  Vec            u;
  PetscReal     *dt, *x, *y, slope, intercept;
  PetscInt       Ns, oNs, Nf = ce->Nf, f, Nr = ce->Nr, r;

  PetscFunctionBegin;
  CHKERRQ(TSGetSolution(ts, &u));
  CHKERRQ(PetscMalloc1(Nr+1, &dt));
  CHKERRQ(TSGetTimeStep(ts, &dt[0]));
  CHKERRQ(TSGetMaxSteps(ts, &oNs));
  Ns   = oNs;
  for (r = 0; r <= Nr; ++r) {
    if (r > 0) {
      dt[r] = dt[r-1]/ce->r;
      Ns    = PetscCeilReal(Ns*ce->r);
    }
    CHKERRQ(TSSetTime(ts, 0.0));
    CHKERRQ(TSSetStepNumber(ts, 0));
    CHKERRQ(TSSetTimeStep(ts, dt[r]));
    CHKERRQ(TSSetMaxSteps(ts, Ns));
    CHKERRQ(PetscConvEstComputeInitialGuess(ce, r, NULL, u));
    CHKERRQ(TSSolve(ts, u));
    CHKERRQ(PetscLogEventBegin(ce->event, ce, 0, 0, 0));
    CHKERRQ(PetscConvEstComputeError(ce, r, ce->idm, u, &ce->errors[r*Nf]));
    CHKERRQ(PetscLogEventEnd(ce->event, ce, 0, 0, 0));
    for (f = 0; f < Nf; ++f) {
      ce->dofs[r*Nf+f] = 1.0/dt[r];
      CHKERRQ(PetscLogEventSetDof(ce->event, f, ce->dofs[r*Nf+f]));
      CHKERRQ(PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]));
    }
    /* Monitor */
    CHKERRQ(PetscConvEstMonitorDefault(ce, r));
  }
  /* Fit convergence rate */
  if (Nr) {
    CHKERRQ(PetscMalloc2(Nr+1, &x, Nr+1, &y));
    for (f = 0; f < Nf; ++f) {
      for (r = 0; r <= Nr; ++r) {
        x[r] = PetscLog10Real(dt[r]);
        y[r] = PetscLog10Real(ce->errors[r*Nf+f]);
      }
      CHKERRQ(PetscLinearRegression(Nr+1, x, y, &slope, &intercept));
      /* Since lg err = s lg dt + b */
      alpha[f] = slope;
    }
    CHKERRQ(PetscFree2(x, y));
  }
  /* Reset solver */
  CHKERRQ(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING));
  CHKERRQ(TSSetTime(ts, 0.0));
  CHKERRQ(TSSetStepNumber(ts, 0));
  CHKERRQ(TSSetTimeStep(ts, dt[0]));
  CHKERRQ(TSSetMaxSteps(ts, oNs));
  CHKERRQ(PetscConvEstComputeInitialGuess(ce, 0, NULL, u));
  CHKERRQ(PetscFree(dt));
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

  PetscFunctionBegin;
  PetscCheck(ce->r == 2.0,PetscObjectComm((PetscObject) ce), PETSC_ERR_SUP, "Only refinement factor 2 is currently supported (not %g)", (double) ce->r);
  CHKERRQ(DMGetDimension(ce->idm, &dim));
  CHKERRQ(DMGetApplicationContext(ce->idm, &ctx));
  CHKERRQ(DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE));
  CHKERRQ(DMGetRefineLevel(ce->idm, &oldlevel));
  CHKERRQ(PetscMalloc1((Nr+1), &dm));
  CHKERRQ(TSGetSolution(ts, &uInitial));
  /* Loop over meshes */
  dm[0] = ce->idm;
  for (r = 0; r <= Nr; ++r) {
    Vec           u;
#if defined(PETSC_USE_LOG)
    PetscLogStage stage;
#endif
    char          stageName[PETSC_MAX_PATH_LEN];
    const char   *dmname, *uname;

    CHKERRQ(PetscSNPrintf(stageName, PETSC_MAX_PATH_LEN-1, "ConvEst Refinement Level %D", r));
#if defined(PETSC_USE_LOG)
    CHKERRQ(PetscLogStageGetId(stageName, &stage));
    if (stage < 0) CHKERRQ(PetscLogStageRegister(stageName, &stage));
#endif
    CHKERRQ(PetscLogStagePush(stage));
    if (r > 0) {
      if (!ce->noRefine) {
        CHKERRQ(DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]));
        CHKERRQ(DMSetCoarseDM(dm[r], dm[r-1]));
      } else {
        DM cdm, rcdm;

        CHKERRQ(DMClone(dm[r-1], &dm[r]));
        CHKERRQ(DMCopyDisc(dm[r-1], dm[r]));
        CHKERRQ(DMGetCoordinateDM(dm[r-1], &cdm));
        CHKERRQ(DMGetCoordinateDM(dm[r],   &rcdm));
        CHKERRQ(DMCopyDisc(cdm, rcdm));
      }
      CHKERRQ(DMCopyTransform(ce->idm, dm[r]));
      CHKERRQ(PetscObjectGetName((PetscObject) dm[r-1], &dmname));
      CHKERRQ(PetscObjectSetName((PetscObject) dm[r], dmname));
      for (f = 0; f <= Nf; ++f) {
        PetscErrorCode (*nspconstr)(DM, PetscInt, PetscInt, MatNullSpace *);

        CHKERRQ(DMGetNullSpaceConstructor(dm[r-1], f, &nspconstr));
        CHKERRQ(DMSetNullSpaceConstructor(dm[r],   f,  nspconstr));
      }
    }
    CHKERRQ(DMViewFromOptions(dm[r], NULL, "-conv_dm_view"));
    /* Create solution */
    CHKERRQ(DMCreateGlobalVector(dm[r], &u));
    CHKERRQ(DMGetField(dm[r], 0, NULL, &disc));
    CHKERRQ(PetscObjectGetName(disc, &uname));
    CHKERRQ(PetscObjectSetName((PetscObject) u, uname));
    /* Setup solver */
    CHKERRQ(TSReset(ts));
    CHKERRQ(TSSetDM(ts, dm[r]));
    CHKERRQ(DMTSSetBoundaryLocal(dm[r], DMPlexTSComputeBoundary, ctx));
    CHKERRQ(DMTSSetIFunctionLocal(dm[r], DMPlexTSComputeIFunctionFEM, ctx));
    CHKERRQ(DMTSSetIJacobianLocal(dm[r], DMPlexTSComputeIJacobianFEM, ctx));
    CHKERRQ(TSSetTime(ts, 0.0));
    CHKERRQ(TSSetStepNumber(ts, 0));
    CHKERRQ(TSSetFromOptions(ts));
    /* Create initial guess */
    CHKERRQ(PetscConvEstComputeInitialGuess(ce, r, dm[r], u));
    CHKERRQ(TSSolve(ts, u));
    CHKERRQ(PetscLogEventBegin(ce->event, ce, 0, 0, 0));
    CHKERRQ(PetscConvEstComputeError(ce, r, dm[r], u, &ce->errors[r*Nf]));
    CHKERRQ(PetscLogEventEnd(ce->event, ce, 0, 0, 0));
    for (f = 0; f < Nf; ++f) {
      PetscSection s, fs;
      PetscInt     lsize;

      /* Could use DMGetOutputDM() to add in Dirichlet dofs */
      CHKERRQ(DMGetLocalSection(dm[r], &s));
      CHKERRQ(PetscSectionGetField(s, f, &fs));
      CHKERRQ(PetscSectionGetConstrainedStorageSize(fs, &lsize));
      CHKERRMPI(MPI_Allreduce(&lsize, &ce->dofs[r*Nf+f], 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) ts)));
      CHKERRQ(PetscLogEventSetDof(ce->event, f, ce->dofs[r*Nf+f]));
      CHKERRQ(PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]));
    }
    /* Monitor */
    CHKERRQ(PetscConvEstMonitorDefault(ce, r));
    if (!r) {
      /* PCReset() does not wipe out the level structure */
      SNES snes;
      KSP  ksp;
      PC   pc;

      CHKERRQ(TSGetSNES(ts, &snes));
      CHKERRQ(SNESGetKSP(snes, &ksp));
      CHKERRQ(KSPGetPC(ksp, &pc));
      CHKERRQ(PCMGGetLevels(pc, &oldnlev));
    }
    /* Cleanup */
    CHKERRQ(VecDestroy(&u));
    CHKERRQ(PetscLogStagePop());
  }
  for (r = 1; r <= Nr; ++r) {
    CHKERRQ(DMDestroy(&dm[r]));
  }
  /* Fit convergence rate */
  CHKERRQ(PetscMalloc2(Nr+1, &x, Nr+1, &y));
  for (f = 0; f < Nf; ++f) {
    for (r = 0; r <= Nr; ++r) {
      x[r] = PetscLog10Real(ce->dofs[r*Nf+f]);
      y[r] = PetscLog10Real(ce->errors[r*Nf+f]);
    }
    CHKERRQ(PetscLinearRegression(Nr+1, x, y, &slope, &intercept));
    /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
    alpha[f] = -slope * dim;
  }
  CHKERRQ(PetscFree2(x, y));
  CHKERRQ(PetscFree(dm));
  /* Restore solver */
  CHKERRQ(TSReset(ts));
  {
    /* PCReset() does not wipe out the level structure */
    SNES snes;
    KSP  ksp;
    PC   pc;

    CHKERRQ(TSGetSNES(ts, &snes));
    CHKERRQ(SNESGetKSP(snes, &ksp));
    CHKERRQ(KSPGetPC(ksp, &pc));
    CHKERRQ(PCMGSetLevels(pc, oldnlev, NULL));
    CHKERRQ(DMSetRefineLevel(ce->idm, oldlevel)); /* The damn DMCoarsen() calls in PCMG can reset this */
  }
  CHKERRQ(TSSetDM(ts, ce->idm));
  CHKERRQ(DMTSSetBoundaryLocal(ce->idm, DMPlexTSComputeBoundary, ctx));
  CHKERRQ(DMTSSetIFunctionLocal(ce->idm, DMPlexTSComputeIFunctionFEM, ctx));
  CHKERRQ(DMTSSetIJacobianLocal(ce->idm, DMPlexTSComputeIJacobianFEM, ctx));
  CHKERRQ(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING));
  CHKERRQ(TSSetTime(ts, 0.0));
  CHKERRQ(TSSetStepNumber(ts, 0));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetSolution(ts, uInitial));
  CHKERRQ(PetscConvEstComputeInitialGuess(ce, 0, NULL, uInitial));
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
