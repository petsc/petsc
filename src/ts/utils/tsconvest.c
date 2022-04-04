#include <petscconvest.h>            /*I "petscconvest.h" I*/
#include <petscts.h>
#include <petscdmplex.h>

#include <petsc/private/petscconvestimpl.h>

static PetscErrorCode PetscConvEstSetTS_Private(PetscConvEst ce, PetscObject solver)
{
  PetscClassId   id;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetClassId(ce->solver, &id));
  PetscCheck(id == TS_CLASSID,PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "Solver was not a TS");
  PetscCall(TSGetDM((TS) ce->solver, &ce->idm));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstInitGuessTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSComputeInitialCondition((TS) ce->solver, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstComputeErrorTS_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u, PetscReal errors[])
{
  TS               ts = (TS) ce->solver;
  PetscErrorCode (*exactError)(TS, Vec, Vec);

  PetscFunctionBegin;
  PetscCall(TSGetComputeExactError(ts, &exactError));
  if (exactError) {
    Vec      e;
    PetscInt f;

    PetscCall(VecDuplicate(u, &e));
    PetscCall(TSComputeExactError(ts, u, e));
    PetscCall(VecNorm(e, NORM_2, errors));
    for (f = 1; f < ce->Nf; ++f) errors[f] = errors[0];
    PetscCall(VecDestroy(&e));
  } else {
    PetscReal t;

    PetscCall(TSGetSolveTime(ts, &t));
    PetscCall(DMComputeL2FieldDiff(dm, t, ce->exactSol, ce->ctxs, u, errors));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstGetConvRateTS_Temporal_Private(PetscConvEst ce, PetscReal alpha[])
{
  TS             ts = (TS) ce->solver;
  Vec            u, u0;
  PetscReal     *dt, *x, *y, slope, intercept;
  PetscInt       Ns, oNs, Nf = ce->Nf, f, Nr = ce->Nr, r;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(Nr+1, &dt));
  PetscCall(TSGetTimeStep(ts, &dt[0]));
  PetscCall(TSGetMaxSteps(ts, &oNs));
  PetscCall(TSGetSolution(ts, &u0));
  PetscCall(PetscObjectReference((PetscObject) u0));
  Ns   = oNs;
  for (r = 0; r <= Nr; ++r) {
    if (r > 0) {
      dt[r] = dt[r-1]/ce->r;
      Ns    = PetscCeilReal(Ns*ce->r);
    }
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetStepNumber(ts, 0));
    PetscCall(TSSetTimeStep(ts, dt[r]));
    PetscCall(TSSetMaxSteps(ts, Ns));
    PetscCall(TSGetSolution(ts, &u));
    PetscCall(PetscConvEstComputeInitialGuess(ce, r, NULL, u));
    PetscCall(TSSolve(ts, NULL));
    PetscCall(TSGetSolution(ts, &u));
    PetscCall(PetscLogEventBegin(ce->event, ce, 0, 0, 0));
    PetscCall(PetscConvEstComputeError(ce, r, ce->idm, u, &ce->errors[r*Nf]));
    PetscCall(PetscLogEventEnd(ce->event, ce, 0, 0, 0));
    for (f = 0; f < Nf; ++f) {
      ce->dofs[r*Nf+f] = 1.0/dt[r];
      PetscCall(PetscLogEventSetDof(ce->event, f, ce->dofs[r*Nf+f]));
      PetscCall(PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]));
    }
    /* Monitor */
    PetscCall(PetscConvEstMonitorDefault(ce, r));
  }
  /* Fit convergence rate */
  if (Nr) {
    PetscCall(PetscMalloc2(Nr+1, &x, Nr+1, &y));
    for (f = 0; f < Nf; ++f) {
      for (r = 0; r <= Nr; ++r) {
        x[r] = PetscLog10Real(dt[r]);
        y[r] = PetscLog10Real(ce->errors[r*Nf+f]);
      }
      PetscCall(PetscLinearRegression(Nr+1, x, y, &slope, &intercept));
      /* Since lg err = s lg dt + b */
      alpha[f] = slope;
    }
    PetscCall(PetscFree2(x, y));
  }
  /* Reset solver */
  PetscCall(TSReset(ts));
  PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTimeStep(ts, dt[0]));
  PetscCall(TSSetMaxSteps(ts, oNs));
  PetscCall(TSSetSolution(ts, u0));
  PetscCall(PetscConvEstComputeInitialGuess(ce, 0, NULL, u0));
  PetscCall(VecDestroy(&u0));
  PetscCall(PetscFree(dt));
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
  PetscCall(DMGetDimension(ce->idm, &dim));
  PetscCall(DMGetApplicationContext(ce->idm, &ctx));
  PetscCall(DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE));
  PetscCall(DMGetRefineLevel(ce->idm, &oldlevel));
  PetscCall(PetscMalloc1((Nr+1), &dm));
  PetscCall(TSGetSolution(ts, &uInitial));
  /* Loop over meshes */
  dm[0] = ce->idm;
  for (r = 0; r <= Nr; ++r) {
    Vec           u;
#if defined(PETSC_USE_LOG)
    PetscLogStage stage;
#endif
    char          stageName[PETSC_MAX_PATH_LEN];
    const char   *dmname, *uname;

    PetscCall(PetscSNPrintf(stageName, PETSC_MAX_PATH_LEN-1, "ConvEst Refinement Level %D", r));
#if defined(PETSC_USE_LOG)
    PetscCall(PetscLogStageGetId(stageName, &stage));
    if (stage < 0) PetscCall(PetscLogStageRegister(stageName, &stage));
#endif
    PetscCall(PetscLogStagePush(stage));
    if (r > 0) {
      if (!ce->noRefine) {
        PetscCall(DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]));
        PetscCall(DMSetCoarseDM(dm[r], dm[r-1]));
      } else {
        DM cdm, rcdm;

        PetscCall(DMClone(dm[r-1], &dm[r]));
        PetscCall(DMCopyDisc(dm[r-1], dm[r]));
        PetscCall(DMGetCoordinateDM(dm[r-1], &cdm));
        PetscCall(DMGetCoordinateDM(dm[r],   &rcdm));
        PetscCall(DMCopyDisc(cdm, rcdm));
      }
      PetscCall(DMCopyTransform(ce->idm, dm[r]));
      PetscCall(PetscObjectGetName((PetscObject) dm[r-1], &dmname));
      PetscCall(PetscObjectSetName((PetscObject) dm[r], dmname));
      for (f = 0; f <= Nf; ++f) {
        PetscErrorCode (*nspconstr)(DM, PetscInt, PetscInt, MatNullSpace *);

        PetscCall(DMGetNullSpaceConstructor(dm[r-1], f, &nspconstr));
        PetscCall(DMSetNullSpaceConstructor(dm[r],   f,  nspconstr));
      }
    }
    PetscCall(DMViewFromOptions(dm[r], NULL, "-conv_dm_view"));
    /* Create solution */
    PetscCall(DMCreateGlobalVector(dm[r], &u));
    PetscCall(DMGetField(dm[r], 0, NULL, &disc));
    PetscCall(PetscObjectGetName(disc, &uname));
    PetscCall(PetscObjectSetName((PetscObject) u, uname));
    /* Setup solver */
    PetscCall(TSReset(ts));
    PetscCall(TSSetDM(ts, dm[r]));
    PetscCall(DMTSSetBoundaryLocal(dm[r], DMPlexTSComputeBoundary, ctx));
    PetscCall(DMTSSetIFunctionLocal(dm[r], DMPlexTSComputeIFunctionFEM, ctx));
    PetscCall(DMTSSetIJacobianLocal(dm[r], DMPlexTSComputeIJacobianFEM, ctx));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetStepNumber(ts, 0));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSetSolution(ts, u));
    PetscCall(VecDestroy(&u));
    /* Create initial guess */
    PetscCall(TSGetSolution(ts, &u));
    PetscCall(PetscConvEstComputeInitialGuess(ce, r, dm[r], u));
    PetscCall(TSSolve(ts, NULL));
    PetscCall(TSGetSolution(ts, &u));
    PetscCall(PetscLogEventBegin(ce->event, ce, 0, 0, 0));
    PetscCall(PetscConvEstComputeError(ce, r, dm[r], u, &ce->errors[r*Nf]));
    PetscCall(PetscLogEventEnd(ce->event, ce, 0, 0, 0));
    for (f = 0; f < Nf; ++f) {
      PetscSection s, fs;
      PetscInt     lsize;

      /* Could use DMGetOutputDM() to add in Dirichlet dofs */
      PetscCall(DMGetLocalSection(dm[r], &s));
      PetscCall(PetscSectionGetField(s, f, &fs));
      PetscCall(PetscSectionGetConstrainedStorageSize(fs, &lsize));
      PetscCallMPI(MPI_Allreduce(&lsize, &ce->dofs[r*Nf+f], 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) ts)));
      PetscCall(PetscLogEventSetDof(ce->event, f, ce->dofs[r*Nf+f]));
      PetscCall(PetscLogEventSetError(ce->event, f, ce->errors[r*Nf+f]));
    }
    /* Monitor */
    PetscCall(PetscConvEstMonitorDefault(ce, r));
    if (!r) {
      /* PCReset() does not wipe out the level structure */
      SNES snes;
      KSP  ksp;
      PC   pc;

      PetscCall(TSGetSNES(ts, &snes));
      PetscCall(SNESGetKSP(snes, &ksp));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCMGGetLevels(pc, &oldnlev));
    }
    /* Cleanup */
    PetscCall(PetscLogStagePop());
  }
  for (r = 1; r <= Nr; ++r) {
    PetscCall(DMDestroy(&dm[r]));
  }
  /* Fit convergence rate */
  PetscCall(PetscMalloc2(Nr+1, &x, Nr+1, &y));
  for (f = 0; f < Nf; ++f) {
    for (r = 0; r <= Nr; ++r) {
      x[r] = PetscLog10Real(ce->dofs[r*Nf+f]);
      y[r] = PetscLog10Real(ce->errors[r*Nf+f]);
    }
    PetscCall(PetscLinearRegression(Nr+1, x, y, &slope, &intercept));
    /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
    alpha[f] = -slope * dim;
  }
  PetscCall(PetscFree2(x, y));
  PetscCall(PetscFree(dm));
  /* Restore solver */
  PetscCall(TSReset(ts));
  {
    /* PCReset() does not wipe out the level structure */
    SNES snes;
    KSP  ksp;
    PC   pc;

    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCMGSetLevels(pc, oldnlev, NULL));
    PetscCall(DMSetRefineLevel(ce->idm, oldlevel)); /* The damn DMCoarsen() calls in PCMG can reset this */
  }
  PetscCall(TSSetDM(ts, ce->idm));
  PetscCall(DMTSSetBoundaryLocal(ce->idm, DMPlexTSComputeBoundary, ctx));
  PetscCall(DMTSSetIFunctionLocal(ce->idm, DMPlexTSComputeIFunctionFEM, ctx));
  PetscCall(DMTSSetIJacobianLocal(ce->idm, DMPlexTSComputeIJacobianFEM, ctx));
  PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetSolution(ts, uInitial));
  PetscCall(PetscConvEstComputeInitialGuess(ce, 0, NULL, uInitial));
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
