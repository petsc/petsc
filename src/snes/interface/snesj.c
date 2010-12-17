#define PETSCSNES_DLL

#include "private/snesimpl.h"    /*I  "petscsnes.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultComputeJacobian"
/*@C
   SNESDefaultComputeJacobian - Computes the Jacobian using finite differences. 

   Collective on SNES

   Input Parameters:
+  x1 - compute Jacobian at this point
-  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
+  J - Jacobian matrix (not altered in this routine)
.  B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-  flag - flag indicating whether the matrix sparsity structure has changed

   Options Database Key:
+  -snes_fd - Activates SNESDefaultComputeJacobian()
.  -snes_test_err - Square root of function error tolerance, default square root of machine
                    epsilon (1.e-8 in double, 3.e-4 in single)
-  -mat_fd_type - Either wp or ds (see MATMFFD_WP or MATMFFD_DS)

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   SNESDefaultComputeJacobian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Jacobian.

   An alternative routine that uses coloring to exploit matrix sparsity is
   SNESDefaultComputeJacobianColor().

   Level: intermediate

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESDefaultComputeJacobianColor(), MatCreateSNESMF()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultComputeJacobian(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec            j1a,j2a,x2;
  PetscErrorCode ierr;
  PetscInt       i,N,start,end,j,value,root;
  PetscScalar    dx,*y,*xx,wscale;
  PetscReal      amax,epsilon = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal      dx_min = 1.e-16,dx_par = 1.e-1,unorm;
  MPI_Comm       comm;
  PetscErrorCode (*eval_fct)(SNES,Vec,Vec)=0;
  PetscTruth     assembled,use_wp = PETSC_TRUE,flg;
  const char     *list[2] = {"ds","wp"};
  PetscMPIInt    size;
  const PetscInt *ranges;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(((PetscObject)snes)->prefix,"-snes_test_err",&epsilon,0);CHKERRQ(ierr);
  eval_fct = SNESComputeFunction;

  ierr = PetscObjectGetComm((PetscObject)x1,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MatAssembled(*B,&assembled);CHKERRQ(ierr);
  if (assembled) {
    ierr = MatZeroEntries(*B);CHKERRQ(ierr);
  }
  if (!snes->nvwork) {
    snes->nvwork = 3;
    ierr = VecDuplicateVecs(x1,snes->nvwork,&snes->vwork);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(snes,snes->nvwork,snes->vwork);CHKERRQ(ierr);
  }
  j1a = snes->vwork[0]; j2a = snes->vwork[1]; x2 = snes->vwork[2];

  ierr = VecGetSize(x1,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end);CHKERRQ(ierr);
  ierr = (*eval_fct)(snes,x1,j1a);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-mat_fd_type","Algorithm to compute difference parameter","SNESDefaultComputeJacobian",list,2,"wp",&value,&flg);CHKERRQ(ierr);
  if (flg && !value) {
    use_wp = PETSC_FALSE;
  }
  if (use_wp) {
    ierr = VecNorm(x1,NORM_2,&unorm);CHKERRQ(ierr);
  }
  /* Compute Jacobian approximation, 1 column at a time. 
      x1 = current iterate, j1a = F(x1)
      x2 = perturbed iterate, j2a = F(x2)
   */
  for (i=0; i<N; i++) {
    ierr = VecCopy(x1,x2);CHKERRQ(ierr);
    if (i>= start && i<end) {
      ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
      if (use_wp) {
        dx = 1.0 + unorm;
      } else {
        dx = xx[i-start];
      }
      ierr = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      if (dx < dx_min && dx >= 0.0) dx = dx_par;
      else if (dx < 0.0 && dx > -dx_min) dx = -dx_par;
#else
      if (PetscAbsScalar(dx) < dx_min && PetscRealPart(dx) >= 0.0) dx = dx_par;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < dx_min) dx = -dx_par;
#endif
      dx *= epsilon;
      wscale = 1.0/dx;
      ierr = VecSetValues(x2,1,&i,&dx,ADD_VALUES);CHKERRQ(ierr);
    } else {
      wscale = 0.0;
    }
    ierr = VecAssemblyBegin(x2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x2);CHKERRQ(ierr);
    ierr = (*eval_fct)(snes,x2,j2a);CHKERRQ(ierr);
    ierr = VecAXPY(j2a,-1.0,j1a);CHKERRQ(ierr);
    /* Communicate scale=1/dx_i to all processors */
    ierr = VecGetOwnershipRanges(x1,&ranges);CHKERRQ(ierr);
    root = size;
    for (j=size-1; j>-1; j--){
      root--;
      if (i>=ranges[j]) break;
    }
    ierr = MPI_Bcast(&wscale,1,MPIU_SCALAR,root,comm);CHKERRQ(ierr);

    ierr = VecScale(j2a,wscale);CHKERRQ(ierr);
    ierr = VecNorm(j2a,NORM_INFINITY,&amax);CHKERRQ(ierr); amax *= 1.e-14;
    ierr = VecGetArray(j2a,&y);CHKERRQ(ierr);
    for (j=start; j<end; j++) {
      if (PetscAbsScalar(y[j-start]) > amax || j == i) {
        ierr = MatSetValues(*B,1,&j,1,&i,y+j-start,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(j2a,&y);CHKERRQ(ierr);
  }
  ierr  = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*B != *J) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag =  DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}


