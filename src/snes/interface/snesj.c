
#ifndef lint
static char vcid[] = "$Id: snesj.c,v 1.31 1996/08/05 19:34:50 curfman Exp curfman $";
#endif

#include "draw.h"    /*I  "draw.h"  I*/
#include "snesimpl.h"    /*I  "snes.h"  I*/

/*@C
   SNESDefaultComputeJacobian - Computes the Jacobian using finite 
   differences. 

   Input Parameters:
.  x1 - compute Jacobian at this point
.  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian
.  flag - matrix flag

   Options Database Key:
$  -snes_fd

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   SNESDefaultComputeJacobian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Jacobian.

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESTestJacobian()
@*/
int SNESDefaultComputeJacobian(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec      j1,j2,x2;
  int      i,ierr,N,start,end,j;
  Scalar   dx, mone = -1.0,*y,scale,*xx,wscale;
  double   amax, epsilon = 1.e-8; /* assumes double precision */
  MPI_Comm comm;
  int      (*eval_fct)(SNES,Vec,Vec);

  if (snes->method_class == SNES_NONLINEAR_EQUATIONS)
    eval_fct = SNESComputeFunction;
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)
    eval_fct = SNESComputeGradient;
  else SETERRQ(1,"SNESDefaultComputeJacobian: Invalid method class");

  PetscObjectGetComm((PetscObject)x1,&comm);
  MatZeroEntries(*J);
  if (!snes->nvwork) {
    ierr = VecDuplicateVecs(x1,3,&snes->vwork); CHKERRQ(ierr);
    snes->nvwork = 3;
    PLogObjectParents(snes,3,snes->vwork);
  }
  j1 = snes->vwork[0]; j2 = snes->vwork[1]; x2 = snes->vwork[2];

  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);
  VecGetArray(x1,&xx);
  ierr = eval_fct(snes,x1,j1); CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time. 
      x1 = current iterate, j1 = F(x1)
      x2 = perturbed iterate, j2 = F(x2)
   */
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(x1,x2); CHKERRQ(ierr);
    if ( i>= start && i<end) {
      dx = xx[i-start];
#if !defined(PETSC_COMPLEX)
      if (dx < 1.e-16 && dx >= 0.0) dx = 1.e-1;
      else if (dx < 0.0 && dx > -1.e-16) dx = -1.e-1;
#else
      if (abs(dx) < 1.e-16 && real(dx) >= 0.0) dx = 1.e-1;
      else if (real(dx) < 0.0 && abs(dx) < 1.e-16) dx = -1.e-1;
#endif
      dx *= epsilon;
      wscale = 1.0/dx;
      VecSetValues(x2,1,&i,&dx,ADD_VALUES); 
    } 
    else {
      wscale = 0.0;
    }
    ierr = eval_fct(snes,x2,j2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,j1,j2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(PETSC_COMPLEX)
    MPI_Allreduce(&wscale,&scale,1,MPI_DOUBLE,MPI_SUM,comm);
#else
    MPI_Allreduce(&wscale,&scale,2,MPI_DOUBLE,MPI_SUM,comm);
#endif
    VecScale(&scale,j2);
    VecGetArray(j2,&y);
    VecNorm(j2,NORM_INFINITY,&amax); amax *= 1.e-14;
    for ( j=start; j<end; j++ ) {
      if (PetscAbsScalar(y[j-start]) > amax) {
        ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    VecRestoreArray(j2,&y);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

/*@C
   SNESDefaultComputeHessian - Computes the Hessian using finite 
   differences. 

   Input Parameters:
.  x1 - compute Hessian at this point
.  ctx - application's gradient context, as set with SNESSetGradient()

   Output Parameters:
.  J - Hessian
.  B - preconditioner, same as Hessian
.  flag - matrix flag

   Options Database Key:
$  -snes_fd

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   SNESDefaultComputeHessian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Hessian.

.keywords: SNES, finite differences, Hessian

.seealso: SNESSetHessian(), SNESTestHessian()
@*/
int SNESDefaultComputeHessian(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  return SNESDefaultComputeJacobian(snes,x1,J,B,flag,ctx);
}
