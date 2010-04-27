#define PETSCTS_DLL

#include "private/tsimpl.h"        /*I  "petscts.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultComputeJacobianColor"
/*@C
    TSDefaultComputeJacobianColor - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity.  
  
    Collective on TS, Vec and Mat

    Input Parameters:
+   ts - nonlinear solver object
.   t - current time
.   x1 - location at which to evaluate Jacobian
-   ctx - coloring context, where ctx must have type MatFDColoring, 
          as created via MatFDColoringCreate()

    Output Parameters:
+   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-   flag - flag indicating whether the matrix sparsity structure has changed

   Level: intermediate

.keywords: TS, finite differences, Jacobian, coloring, sparse

.seealso: TSSetJacobian(), MatFDColoringCreate(), MatFDColoringSetFunction()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultComputeJacobianColor(TS ts,PetscReal t,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring  color = (MatFDColoring) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFDColoringApplyTS(*B,color,t,x1,flag,ts);CHKERRQ(ierr);
  
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSDefaultComputeJacobian"
/*@C
   TSDefaultComputeJacobian - Computes the Jacobian using finite differences.

   Input Parameters:
+  ts - TS context
.  xx1 - compute Jacobian at this point
-  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
+  J - Jacobian
.  B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-  flag - matrix flag

   Notes:
   This routine is slow and expensive, and is not optimized.

   Sparse approximations using colorings are also available and
   would be a much better alternative!

   Level: intermediate

.seealso: TSDefaultComputeJacobianColor()
@*/
PetscErrorCode TSDefaultComputeJacobian(TS ts,PetscReal t,Vec xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec            f1,f2,xx2;
  PetscErrorCode ierr;
  PetscInt       i,N,start,end,j;
  PetscScalar    dx,*y,*xx,wscale;
  PetscReal      amax,epsilon = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal      dx_min = 1.e-16,dx_par = 1.e-1;
  MPI_Comm       comm;
  PetscTruth     assembled;
  PetscMPIInt    size;
  const PetscInt *ranges;
  PetscMPIInt    root;

  PetscFunctionBegin;
  ierr = VecDuplicate(xx1,&f1);CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&f2);CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&xx2);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xx1,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MatAssembled(*B,&assembled);CHKERRQ(ierr);
  if (assembled) {
    ierr = MatZeroEntries(*B);CHKERRQ(ierr);
  }

  ierr = VecGetSize(xx1,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xx1,&start,&end);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,ts->ptime,xx1,f1);CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time.
      xx1 = current iterate, f1 = F(xx1)
      xx2 = perturbed iterate, f2 = F(xx2)
   */
  for (i=0; i<N; i++) {
    ierr = VecCopy(xx1,xx2);CHKERRQ(ierr);
    if (i>= start && i<end) {
      ierr =  VecGetArray(xx1,&xx);CHKERRQ(ierr);
      dx   = xx[i-start];
      ierr =  VecRestoreArray(xx1,&xx);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      if (dx < dx_min && dx >= 0.0) dx = dx_par;
      else if (dx < 0.0 && dx > -dx_min) dx = -dx_par;
#else
      if (PetscAbsScalar(dx) < dx_min && PetscRealPart(dx) >= 0.0) dx = dx_par;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < dx_min) dx = -dx_par;
#endif
      dx *= epsilon;
      wscale = 1.0/dx;
      ierr =  VecSetValues(xx2,1,&i,&dx,ADD_VALUES);CHKERRQ(ierr);
    } else {
      wscale = 0.0;
    }
    ierr = TSComputeRHSFunction(ts,t,xx2,f2);CHKERRQ(ierr);
    ierr = VecAXPY(f2,-1.0,f1);CHKERRQ(ierr);
    /* Communicate scale=1/dx_i to all processors */
    ierr = VecGetOwnershipRanges(xx1,&ranges);CHKERRQ(ierr);
    root = size;
    for (j=size-1; j>-1; j--){
      root--;
      if (i>=ranges[j]) break;
    }
    ierr = MPI_Bcast(&wscale,1,MPIU_SCALAR,root,comm);CHKERRQ(ierr);

    ierr = VecScale(f2,wscale);CHKERRQ(ierr);
    ierr = VecNorm(f2,NORM_INFINITY,&amax);CHKERRQ(ierr); amax *= 1.e-14;
    ierr = VecGetArray(f2,&y);CHKERRQ(ierr);
    for (j=start; j<end; j++) {
      if (PetscAbsScalar(y[j-start]) > amax || j == i) {
        ierr = MatSetValues(*B,1,&j,1,&i,y+j-start,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(f2,&y);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*B != *J) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag =  DIFFERENT_NONZERO_PATTERN;

  ierr = VecDestroy(f1);CHKERRQ(ierr);
  ierr = VecDestroy(f2);CHKERRQ(ierr);
  ierr = VecDestroy(xx2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


