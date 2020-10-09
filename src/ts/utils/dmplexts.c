#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/tsimpl.h>     /*I "petscts.h" I*/
#include <petsc/private/snesimpl.h>
#include <petscds.h>
#include <petscfv.h>

static PetscErrorCode DMTSConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (isPlex) {
    *plex = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex);CHKERRQ(ierr);
    if (!*plex) {
      ierr = DMConvert(dm,DMPLEX,plex);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex);CHKERRQ(ierr);
      if (copy) {
        PetscInt    i;
        PetscObject obj;
        const char *comps[3] = {"A","dmAux","dmCh"};

        ierr = DMCopyDMTS(dm, *plex);CHKERRQ(ierr);
        ierr = DMCopyDMSNES(dm, *plex);CHKERRQ(ierr);
        for (i = 0; i < 3; i++) {
          ierr = PetscObjectQuery((PetscObject) dm, comps[i], &obj);CHKERRQ(ierr);
          ierr = PetscObjectCompose((PetscObject) *plex, comps[i], obj);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = PetscObjectReference((PetscObject) *plex);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeRHSFunctionFVM - Form the local forcing F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
- user - The user context

  Output Parameter:
. F  - Global output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *user)
{
  Vec            locF;
  IS             cellIS;
  DM             plex;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {
    ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(plex, &locF);CHKERRQ(ierr);
  ierr = VecZeroEntries(locF);CHKERRQ(ierr);
  ierr = DMPlexComputeResidual_Internal(plex, cellIS, time, locX, NULL, time, locF, user);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(plex, &locF);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeBoundary - Insert the essential boundary values for the local input X and/or its time derivative X_t using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
- user - The user context

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeBoundary(DM dm, PetscReal time, Vec locX, Vec locX_t, void *user)
{
  DM             plex;
  Vec            faceGeometryFVM = NULL;
  PetscInt       Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm, &plex, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetNumFields(plex, &Nf);CHKERRQ(ierr);
  if (!locX_t) {
    /* This is the RHS part */
    for (f = 0; f < Nf; f++) {
      PetscObject  obj;
      PetscClassId id;

      ierr = DMGetField(plex, f, NULL, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFV_CLASSID) {
        ierr = DMPlexGetGeometryFVM(plex, &faceGeometryFVM, NULL, NULL);CHKERRQ(ierr);
        break;
      }
    }
  }
  ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, time, faceGeometryFVM, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexInsertTimeDerivativeBoundaryValues(plex, PETSC_TRUE, locX_t, time, faceGeometryFVM, NULL, NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeIFunctionFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
- user - The user context

  Output Parameter:
. locF  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeIFunctionFEM(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *user)
{
  DM             plex;
  IS             cellIS;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {
    ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
  }
  ierr = DMPlexComputeResidual_Internal(plex, cellIS, time, locX, locX_t, time, locF, user);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeIJacobianFEM - Form the local Jacobian J from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
. X_tshift - The multiplicative parameter for dF/du_t
- user - The user context

  Output Parameter:
. locF  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeIJacobianFEM(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  DM             plex;
  PetscDS        prob;
  PetscBool      hasJac, hasPrec;
  IS             cellIS;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  if (hasJac && hasPrec) {ierr = MatZeroEntries(Jac);CHKERRQ(ierr);}
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex,&depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);}
  ierr = DMPlexComputeJacobian_Internal(plex, cellIS, time, X_tShift, locX, locX_t, Jac, JacP, user);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckResidual - Check the residual of the exact solution

  Input Parameters:
+ ts  - the TS object
. dm  - the DM
. t   - the time
. u   - a DM vector
. u_t - a DM vector
- tol - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
. residual - The residual norm of the exact solution, or NULL

  Level: developer

.seealso: DNTSCheckFromOptions(), DMTSCheckJacobian(), DNSNESCheckFromOptions(), DMSNESCheckDiscretization(), DMSNESCheckJacobian()
@*/
PetscErrorCode DMTSCheckResidual(TS ts, DM dm, PetscReal t, Vec u, Vec u_t, PetscReal tol, PetscReal *residual)
{
  MPI_Comm       comm;
  Vec            r;
  PetscReal      res;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (residual) PetscValidRealPointer(residual, 5);
  ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
  ierr = DMComputeExactSolution(dm, t, u, u_t);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts, t, u, u_t, r, PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  if (tol >= 0.0) {
    if (res > tol) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double) res, (double) tol);
  } else if (residual) {
    *residual = res;
  } else {
    ierr = PetscPrintf(comm, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) r, "Initial Residual");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)r,"res_");CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-vec_view");CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckJacobian - Check the Jacobian of the exact solution against the residual using the Taylor Test

  Input Parameters:
+ ts  - the TS object
. dm  - the DM
. t   - the time
. u   - a DM vector
. u_t - a DM vector
- tol - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
+ isLinear - Flag indicaing that the function looks linear, or NULL
- convRate - The rate of convergence of the linear model, or NULL

  Level: developer

.seealso: DNTSCheckFromOptions(), DMTSCheckResidual(), DNSNESCheckFromOptions(), DMSNESCheckDiscretization(), DMSNESCheckResidual()
@*/
PetscErrorCode DMTSCheckJacobian(TS ts, DM dm, PetscReal t, Vec u, Vec u_t, PetscReal tol, PetscBool *isLinear, PetscReal *convRate)
{
  MPI_Comm       comm;
  PetscDS        ds;
  Mat            J, M;
  MatNullSpace   nullspace;
  PetscReal      dt, shift, slope, intercept;
  PetscBool      hasJac, hasPrec, isLin = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (isLinear) PetscValidBoolPointer(isLinear, 5);
  if (convRate) PetscValidRealPointer(convRate, 5);
  ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
  ierr = DMComputeExactSolution(dm, t, u, u_t);CHKERRQ(ierr);
  /* Create and view matrices */
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  shift = 1.0/dt;
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(ds, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(ds, &hasPrec);CHKERRQ(ierr);
  if (hasJac && hasPrec) {
    ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(ts, t, u, u_t, shift, J, M, PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) M, "Preconditioning Matrix");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) M, "jacpre_");CHKERRQ(ierr);
    ierr = MatViewFromOptions(M, NULL, "-mat_view");CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobian(ts, t, u, u_t, shift, J, J, PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) J, "Jacobian");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) J, "jac_");CHKERRQ(ierr);
  ierr = MatViewFromOptions(J, NULL, "-mat_view");CHKERRQ(ierr);
  /* Check nullspace */
  ierr = MatGetNullSpace(J, &nullspace);CHKERRQ(ierr);
  if (nullspace) {
    PetscBool isNull;
    ierr = MatNullSpaceTest(nullspace, J, &isNull);CHKERRQ(ierr);
    if (!isNull) SETERRQ(comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  /* Taylor test */
  {
    PetscRandom rand;
    Vec         du, uhat, uhat_t, r, rhat, df;
    PetscReal   h;
    PetscReal  *es, *hs, *errors;
    PetscReal   hMax = 1.0, hMin = 1e-6, hMult = 0.1;
    PetscInt    Nv, v;

    /* Choose a perturbation direction */
    ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &du);CHKERRQ(ierr);
    ierr = VecSetRandom(du, rand);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &df);CHKERRQ(ierr);
    ierr = MatMult(J, du, df);CHKERRQ(ierr);
    /* Evaluate residual at u, F(u), save in vector r */
    ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
    ierr = TSComputeIFunction(ts, t, u, u_t, r, PETSC_FALSE);CHKERRQ(ierr);
    /* Look at the convergence of our Taylor approximation as we approach u */
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv);
    ierr = PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &uhat);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &uhat_t);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &rhat);CHKERRQ(ierr);
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      ierr = VecWAXPY(uhat, h, du, u);CHKERRQ(ierr);
      ierr = VecWAXPY(uhat_t, h*shift, du, u_t);CHKERRQ(ierr);
      /* F(\hat u, \hat u_t) \approx F(u, u_t) + J(u, u_t) (uhat - u) + J_t(u, u_t) (uhat_t - u_t) = F(u) + h * J(u) du + h * shift * J_t(u) du = F(u) + h F' du */
      ierr = TSComputeIFunction(ts, t, uhat, uhat_t, rhat, PETSC_FALSE);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(rhat, -1.0, -h, 1.0, r, df);CHKERRQ(ierr);
      ierr = VecNorm(rhat, NORM_2, &errors[Nv]);CHKERRQ(ierr);

      es[Nv] = PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    ierr = VecDestroy(&uhat);CHKERRQ(ierr);
    ierr = VecDestroy(&uhat_t);CHKERRQ(ierr);
    ierr = VecDestroy(&rhat);CHKERRQ(ierr);
    ierr = VecDestroy(&df);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&du);CHKERRQ(ierr);
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL)    break;
    }
    if (v == Nv) isLin = PETSC_TRUE;
    ierr = PetscLinearRegression(Nv, hs, es, &slope, &intercept);CHKERRQ(ierr);
    ierr = PetscFree3(es, hs, errors);CHKERRQ(ierr);
    /* Slope should be about 2 */
    if (tol >= 0) {
      if (!isLin && PetscAbsReal(2 - slope) > tol) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double) slope);
    } else if (isLinear || convRate) {
      if (isLinear) *isLinear = isLin;
      if (convRate) *convRate = slope;
    } else {
      if (!isLin) {ierr = PetscPrintf(comm, "Taylor approximation converging at order %3.2f\n", (double) slope);CHKERRQ(ierr);}
      else        {ierr = PetscPrintf(comm, "Function appears to be linear\n");CHKERRQ(ierr);}
    }
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckFromOptions - Check the residual and Jacobian functions using the exact solution by outputting some diagnostic information

  Input Parameters:
+ ts - the TS object
- u  - representative TS vector

  Note: The user must call PetscDSSetExactSolution() beforehand

  Level: developer
@*/
PetscErrorCode DMTSCheckFromOptions(TS ts, Vec u)
{
  DM             dm;
  SNES           snes;
  Vec            sol, u_t;
  PetscReal      t;
  PetscBool      check;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(((PetscObject)ts)->options,((PetscObject)ts)->prefix, "-dmts_check", &check);CHKERRQ(ierr);
  if (!check) PetscFunctionReturn(0);
  ierr = VecDuplicate(u, &sol);CHKERRQ(ierr);
  ierr = VecCopy(u, sol);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, u);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes, u);CHKERRQ(ierr);

  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMSNESCheckDiscretization(snes, dm, t, sol, -1.0, NULL);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u_t);CHKERRQ(ierr);
  ierr = DMTSCheckResidual(ts, dm, t, sol, u_t, -1.0, NULL);CHKERRQ(ierr);
  ierr = DMTSCheckJacobian(ts, dm, t, sol, u_t, -1.0, NULL, NULL);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u_t);CHKERRQ(ierr);

  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
