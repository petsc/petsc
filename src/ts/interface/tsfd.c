#define PETSCTS_DLL

#include "src/mat/matimpl.h"      /*I  "petscmat.h"  I*/
#include "src/ts/tsimpl.h"        /*I  "petscts.h"  I*/

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

   Options Database Keys:
$  -mat_fd_coloring_freq <freq> 

   Level: intermediate

.keywords: TS, finite differences, Jacobian, coloring, sparse

.seealso: TSSetJacobian(), MatFDColoringCreate(), MatFDColoringSetFunction()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultComputeJacobianColor(TS ts,PetscReal t,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring  color = (MatFDColoring) ctx;
  SNES           snes;
  PetscErrorCode ierr;
  PetscInt       freq,it;

  PetscFunctionBegin;
  /*
       If we are not using SNES we have no way to know the current iteration.
  */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (snes) {
    ierr = MatFDColoringGetFrequency(color,&freq);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);

    if ((freq > 1) && ((it % freq) != 1)) {
      ierr = PetscInfo2(color,"Skipping Jacobian, it %D, freq %D\n",it,freq);CHKERRQ(ierr);
      *flag = SAME_PRECONDITIONER;
      goto end;
    } else {
      ierr = PetscInfo2(color,"Computing Jacobian, it %D, freq %D\n",it,freq);CHKERRQ(ierr);
      *flag = SAME_NONZERO_PATTERN;
    }
  }
  ierr = MatFDColoringApplyTS(*B,color,t,x1,flag,ts);CHKERRQ(ierr);
  end:
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
.  B - preconditioner, same as Jacobian
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
  Vec            jj1,jj2,xx2;
  PetscErrorCode ierr;
  PetscInt       i,N,start,end,j;
  PetscScalar    dx,*y,scale,*xx,wscale;
  PetscReal      amax,epsilon = PETSC_SQRT_MACHINE_EPSILON;
  PetscReal      dx_min = 1.e-16,dx_par = 1.e-1;
  MPI_Comm       comm;
  PetscTruth     assembled;

  PetscFunctionBegin;
  ierr = VecDuplicate(xx1,&jj1);CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&jj2);CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&xx2);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xx1,&comm);CHKERRQ(ierr);
  ierr = MatAssembled(*J,&assembled);CHKERRQ(ierr);
  if (assembled) {
    ierr = MatZeroEntries(*J);CHKERRQ(ierr);
  }

  ierr = VecGetSize(xx1,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xx1,&start,&end);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,ts->ptime,xx1,jj1);CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time.
      xx1 = current iterate, jj1 = F(xx1)
      xx2 = perturbed iterate, jj2 = F(xx2)
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
    ierr = TSComputeRHSFunction(ts,t,xx2,jj2);CHKERRQ(ierr);
    ierr = VecAXPY(jj2,-1.0,jj1);CHKERRQ(ierr);
    /* Communicate scale to all processors */
    ierr = MPI_Allreduce(&wscale,&scale,1,MPIU_SCALAR,PetscSum_Op,comm);CHKERRQ(ierr);
    ierr = VecScale(jj2,scale);CHKERRQ(ierr);
    ierr = VecNorm(jj2,NORM_INFINITY,&amax);CHKERRQ(ierr); amax *= 1.e-14;
    ierr = VecGetArray(jj2,&y);CHKERRQ(ierr);
    for (j=start; j<end; j++) {
      if (PetscAbsScalar(y[j-start]) > amax) {
        ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(jj2,&y);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag =  DIFFERENT_NONZERO_PATTERN;

  ierr = VecDestroy(jj1);CHKERRQ(ierr);
  ierr = VecDestroy(jj2);CHKERRQ(ierr);
  ierr = VecDestroy(xx2);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


