
/*
   This file contains utility routines for finite difference
   approximation of Jacobian matrices.  This functionality for
   the TS component will eventually be incorporated as part of
   the base PETSc libraries.
*/
#include <petsc-private/tsimpl.h>
#include <petsc-private/snesimpl.h>
#include <petsc-private/fortranimpl.h>

PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
PetscErrorCode RHSJacobianFD(TS,PetscReal,Vec,Mat*,Mat*,MatStructure *,void*);

/* -------------------------------------------------------------------*/

/* Temporary interface routine; this will be eliminated soon! */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define setcroutinefromfortran_ SETCROUTINEFROMFORTRAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define setcroutinefromfortran_ setcroutinefromfortran
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL setcroutinefromfortran_(TS *ts,Mat *A,Mat *B,PetscErrorCode *__ierr)
{
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,RHSJacobianFD,PETSC_NULL);
}

EXTERN_C_END


/* -------------------------------------------------------------------*/
/*
   RHSJacobianFD - Computes the Jacobian using finite differences.

   Input Parameters:
.  ts - TS context
.  xx1 - compute Jacobian at this point
.  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian
.  flag - matrix flag

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.

   Sparse approximations using colorings are also available and
   would be a much better alternative!
*/
PetscErrorCode RHSJacobianFD(TS ts,PetscReal t,Vec xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec            jj1,jj2,xx2;
  PetscInt       i,N,start,end,j;
  PetscErrorCode ierr;
  PetscScalar    dx,*y,scale,wscale;
  PetscReal      amax,epsilon = 1.e-8; /* assumes PetscReal precision */
  PetscReal      dx_min = 1.e-16,dx_par = 1.e-1;
  MPI_Comm       comm;

  ierr = VecDuplicate(xx1,&jj1);CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&jj2);CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&xx2);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xx1,&comm);CHKERRQ(ierr);
  ierr = MatZeroEntries(*B);CHKERRQ(ierr);

  ierr = VecGetSize(xx1,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xx1,&start,&end);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,t,xx1,jj1);CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time.
      xx1 = current iterate, jj1 = F(xx1)
      xx2 = perturbed iterate, jj2 = F(xx2)
   */
  for (i=0; i<N; i++) {
    ierr = VecCopy(xx1,xx2);CHKERRQ(ierr);
    if (i>= start && i<end) {
      ierr = VecGetValues(xx1,1,&i,&dx);CHKERRQ(ierr);
      if (PetscAbsScalar(dx) < dx_min && PetscRealPart(dx) >= 0.0) dx = dx_par;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < dx_min) dx = -dx_par;
      dx *= epsilon;
      wscale = 1.0/dx;
      ierr = VecSetValues(xx2,1,&i,&dx,ADD_VALUES);CHKERRQ(ierr);
    } else {
      wscale = 0.0;
    }
    ierr = VecAssemblyBegin(xx2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(xx2);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,t,xx2,jj2);CHKERRQ(ierr);
    ierr = VecAXPY(jj2,-1.0,jj1);CHKERRQ(ierr);
    /* Communicate scale to all processors */
    ierr = MPI_Allreduce(&wscale,&scale,1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
    ierr = VecScale(jj2,scale);CHKERRQ(ierr);
    ierr = VecGetArray(jj2,&y);CHKERRQ(ierr);
    ierr = VecNorm(jj2,NORM_INFINITY,&amax);CHKERRQ(ierr);
    amax *= 1.e-14;
    for (j=start; j<end; j++) {
      if (PetscAbsScalar(y[j-start]) > amax) {
        ierr = MatSetValues(*B,1,&j,1,&i,y+j-start,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(jj2,&y);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag =  DIFFERENT_NONZERO_PATTERN;

  ierr = VecDestroy(&jj1);CHKERRQ(ierr);
  ierr = VecDestroy(&jj2);CHKERRQ(ierr);
  ierr = VecDestroy(&xx2);CHKERRQ(ierr);

  return 0;
}
