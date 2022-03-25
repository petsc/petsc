
#include <petsc/private/snesimpl.h>    /*I  "petscsnes.h"  I*/
#include <petscdm.h>

/*@C
   SNESComputeJacobianDefault - Computes the Jacobian using finite differences.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  x1 - compute Jacobian at this point
-  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
+  J - Jacobian matrix (not altered in this routine)
-  B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)

   Options Database Key:
+  -snes_fd - Activates SNESComputeJacobianDefault()
.  -snes_test_err - Square root of function error tolerance, default square root of machine
                    epsilon (1.e-8 in double, 3.e-4 in single)
-  -mat_fd_type - Either wp or ds (see MATMFFD_WP or MATMFFD_DS)

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   SNESComputeJacobianDefault() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Jacobian.

   An alternative routine that uses coloring to exploit matrix sparsity is
   SNESComputeJacobianDefaultColor().

   This routine ignores the maximum number of function evaluations set with SNESSetTolerances() and the function
   evaluations it performs are not counted in what is returned by of SNESGetNumberFunctionEvals().

   Level: intermediate

.seealso: SNESSetJacobian(), SNESComputeJacobianDefaultColor(), MatCreateSNESMF()
@*/
PetscErrorCode  SNESComputeJacobianDefault(SNES snes,Vec x1,Mat J,Mat B,void *ctx)
{
  Vec               j1a,j2a,x2;
  PetscErrorCode    ierr;
  PetscInt          i,N,start,end,j,value,root,max_funcs = snes->max_funcs;
  PetscScalar       dx,*y,wscale;
  const PetscScalar *xx;
  PetscReal         amax,epsilon = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal         dx_min = 1.e-16,dx_par = 1.e-1,unorm;
  MPI_Comm          comm;
  PetscBool         assembled,use_wp = PETSC_TRUE,flg;
  const char        *list[2] = {"ds","wp"};
  PetscMPIInt       size;
  const PetscInt    *ranges;
  DM                dm;
  DMSNES            dms;

  PetscFunctionBegin;
  snes->max_funcs = PETSC_MAX_INT;
  /* Since this Jacobian will possibly have "extra" nonzero locations just turn off errors for these locations */
  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PetscOptionsGetReal(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_test_err",&epsilon,NULL));

  PetscCall(PetscObjectGetComm((PetscObject)x1,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(MatAssembled(B,&assembled));
  if (assembled) {
    PetscCall(MatZeroEntries(B));
  }
  if (!snes->nvwork) {
    if (snes->dm) {
      PetscCall(DMGetGlobalVector(snes->dm,&j1a));
      PetscCall(DMGetGlobalVector(snes->dm,&j2a));
      PetscCall(DMGetGlobalVector(snes->dm,&x2));
    } else {
      snes->nvwork = 3;
      PetscCall(VecDuplicateVecs(x1,snes->nvwork,&snes->vwork));
      PetscCall(PetscLogObjectParents(snes,snes->nvwork,snes->vwork));
      j1a = snes->vwork[0]; j2a = snes->vwork[1]; x2 = snes->vwork[2];
    }
  }

  PetscCall(VecGetSize(x1,&N));
  PetscCall(VecGetOwnershipRange(x1,&start,&end));
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&dms));
  if (dms->ops->computemffunction) {
    PetscCall(SNESComputeMFFunction(snes,x1,j1a));
  } else {
    PetscCall(SNESComputeFunction(snes,x1,j1a));
  }

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->prefix,"Differencing options","SNES");PetscCall(ierr);
  PetscCall(PetscOptionsEList("-mat_fd_type","Algorithm to compute difference parameter","SNESComputeJacobianDefault",list,2,"wp",&value,&flg));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  if (flg && !value) use_wp = PETSC_FALSE;

  if (use_wp) {
    PetscCall(VecNorm(x1,NORM_2,&unorm));
  }
  /* Compute Jacobian approximation, 1 column at a time.
      x1 = current iterate, j1a = F(x1)
      x2 = perturbed iterate, j2a = F(x2)
   */
  for (i=0; i<N; i++) {
    PetscCall(VecCopy(x1,x2));
    if (i>= start && i<end) {
      PetscCall(VecGetArrayRead(x1,&xx));
      if (use_wp) dx = PetscSqrtReal(1.0 + unorm);
      else        dx = xx[i-start];
      PetscCall(VecRestoreArrayRead(x1,&xx));
      if (PetscAbsScalar(dx) < dx_min) dx = (PetscRealPart(dx) < 0. ? -1. : 1.) * dx_par;
      dx    *= epsilon;
      wscale = 1.0/dx;
      PetscCall(VecSetValues(x2,1,&i,&dx,ADD_VALUES));
    } else {
      wscale = 0.0;
    }
    PetscCall(VecAssemblyBegin(x2));
    PetscCall(VecAssemblyEnd(x2));
    if (dms->ops->computemffunction) {
      PetscCall(SNESComputeMFFunction(snes,x2,j2a));
    } else {
      PetscCall(SNESComputeFunction(snes,x2,j2a));
    }
    PetscCall(VecAXPY(j2a,-1.0,j1a));
    /* Communicate scale=1/dx_i to all processors */
    PetscCall(VecGetOwnershipRanges(x1,&ranges));
    root = size;
    for (j=size-1; j>-1; j--) {
      root--;
      if (i>=ranges[j]) break;
    }
    PetscCallMPI(MPI_Bcast(&wscale,1,MPIU_SCALAR,root,comm));
    PetscCall(VecScale(j2a,wscale));
    PetscCall(VecNorm(j2a,NORM_INFINITY,&amax)); amax *= 1.e-14;
    PetscCall(VecGetArray(j2a,&y));
    for (j=start; j<end; j++) {
      if (PetscAbsScalar(y[j-start]) > amax || j == i) {
        PetscCall(MatSetValues(B,1,&j,1,&i,y+j-start,INSERT_VALUES));
      }
    }
    PetscCall(VecRestoreArray(j2a,&y));
  }
  if (snes->dm) {
    PetscCall(DMRestoreGlobalVector(snes->dm,&j1a));
    PetscCall(DMRestoreGlobalVector(snes->dm,&j2a));
    PetscCall(DMRestoreGlobalVector(snes->dm,&x2));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (B != J) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  snes->max_funcs = max_funcs;
  snes->nfuncs    -= N;
  PetscFunctionReturn(0);
}
