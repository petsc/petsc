#include <petsc-private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc-private/tsimpl.h>     /*I "petscts.h" I*/
#include <petscds.h>
#include <petscfv.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSGetGeometryFVM"
/*@
  DMPlexTSGetGeometryFVM - Return precomputed geometric data

  Input Parameter:
. dm - The DM

  Output Parameters:
+ facegeom - The values precomputed from face geometry
. cellgeom - The values precomputed from cell geometry
- minRadius - The minimum radius over the mesh of an inscribed sphere in a cell

  Level: developer

.seealso: DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexTSGetGeometryFVM(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DMTS           dmts;
  PetscObject    obj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm, &dmts);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_facegeom_fvm", &obj);CHKERRQ(ierr);
  if (!obj) {
    Vec cellgeom, facegeom;

    ierr = DMPlexComputeGeometryFVM(dm, &cellgeom, &facegeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmts, "DMPlexTS_facegeom_fvm", (PetscObject) facegeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmts, "DMPlexTS_cellgeom_fvm", (PetscObject) cellgeom);CHKERRQ(ierr);
    ierr = VecDestroy(&facegeom);CHKERRQ(ierr);
    ierr = VecDestroy(&cellgeom);CHKERRQ(ierr);
  }
  if (facegeom) {PetscValidPointer(facegeom, 2); ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_facegeom_fvm", (PetscObject *) facegeom);CHKERRQ(ierr);}
  if (cellgeom) {PetscValidPointer(cellgeom, 3); ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_cellgeom_fvm", (PetscObject *) cellgeom);CHKERRQ(ierr);}
  if (minRadius) {ierr = DMPlexGetMinRadius(dm, minRadius);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSGetGradientDM"
/*@C
  DMPlexTSGetGradientDM - Return gradient data layout

  Input Parameters:
+ dm - The DM
- fv - The PetscFV

  Output Parameter:
. dmGrad - The layout for gradient values

  Level: developer

.seealso: DMPlexTSGetGeometryFVM(), DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexTSGetGradientDM(DM dm, PetscFV fv, DM *dmGrad)
{
  DMTS           dmts;
  PetscObject    obj;
  PetscBool      computeGradients;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(fv,PETSCFV_CLASSID,2);
  PetscValidPointer(dmGrad,3);
  ierr = PetscFVGetComputeGradients(fv, &computeGradients);CHKERRQ(ierr);
  if (!computeGradients) {*dmGrad = NULL; PetscFunctionReturn(0);}
  ierr = DMGetDMTS(dm, &dmts);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_dmgrad_fvm", &obj);CHKERRQ(ierr);
  if (!obj) {
    DM  dmGrad;
    Vec faceGeometry, cellGeometry;

    ierr = DMPlexTSGetGeometryFVM(dm, &faceGeometry, &cellGeometry, NULL);CHKERRQ(ierr);
    ierr = DMPlexComputeGradientFVM(dm, fv, faceGeometry, cellGeometry, &dmGrad);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmts, "DMPlexTS_dmgrad_fvm", (PetscObject) dmGrad);CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);CHKERRQ(ierr);
  }
  ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_dmgrad_fvm", (PetscObject *) dmGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSComputeRHSFunctionFVM"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &locF);CHKERRQ(ierr);
  ierr = VecZeroEntries(locF);CHKERRQ(ierr);
  ierr = DMPlexComputeResidual_Internal(dm, time, locX, NULL, locF, user);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, locF, INSERT_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, locF, INSERT_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSComputeIFunctionFEM"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeResidual_Internal(dm, time, locX, locX_t, locF, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSCheckFromOptions"
PetscErrorCode DMTSCheckFromOptions(TS ts, Vec u, void (**exactFuncs)(const PetscReal x[], PetscScalar *u, void *ctx), void **ctxs)
{
  DM             dm;
  SNES           snes;
  Mat            J, M;
  Vec            sol, r, b;
  MatNullSpace   nullSpace;
  PetscReal     *error, res = 0.0;
  PetscInt       numFields;
  PetscBool      check;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(ts->hdr.prefix, "-dmts_check", &check);CHKERRQ(ierr);
  if (!check) PetscFunctionReturn(0);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &sol);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, sol);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes, sol);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  M    = J;
#if 0
  {
    ierr = CreatePressureNullSpace(dm, user, &nullSpace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);
    if (M != J) {ierr = MatSetNullSpace(M, nullSpace);CHKERRQ(ierr);}
  }
#endif
  /* Check discretization error */
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(1, numFields), &error);CHKERRQ(ierr);
  if (numFields > 1) {
    PetscInt f;

    ierr = DMPlexComputeL2FieldDiff(dm, exactFuncs, ctxs, u, error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: [");CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      if (f) {ierr = PetscPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
      if (error[f] >= 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "%g", error[f]);CHKERRQ(ierr);}
      else                     {ierr = PetscPrintf(PETSC_COMM_WORLD, "< 1.0e-11");CHKERRQ(ierr);}
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "]\n");CHKERRQ(ierr);
  } else {
    ierr = DMPlexComputeL2Diff(dm, exactFuncs, ctxs, u, &error[0]);CHKERRQ(ierr);
    if (error[0] >= 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error[0]);CHKERRQ(ierr);}
    else                     {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
  }
  ierr = PetscFree(error);CHKERRQ(ierr);
  /* Check residual */
  ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
  ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "Initial Residual");CHKERRQ(ierr);
  ierr = VecViewFromOptions(r, "res_", "-vec_view");CHKERRQ(ierr);
  /* Check Jacobian */
  ierr = SNESComputeJacobian(snes, u, M, M);CHKERRQ(ierr);
  ierr = MatGetNullSpace(J, &nullSpace);CHKERRQ(ierr);
  if (nullSpace) {
    PetscBool isNull;
    ierr = MatNullSpaceTest(nullSpace, J, &isNull);CHKERRQ(ierr);
    if (!isNull) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
  ierr = VecSet(r, 0.0);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
  ierr = MatMult(M, u, r);CHKERRQ(ierr);
  ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
  ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "Au - b = Au + F(0)");CHKERRQ(ierr);
  ierr = VecViewFromOptions(r, "linear_res_", "-vec_view");CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
