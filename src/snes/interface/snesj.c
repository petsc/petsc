
#ifndef lint
static char vcid[] = "$Id: snesj.c,v 1.17 1995/06/29 23:54:14 bsmith Exp bsmith $";
#endif

#include "draw.h"    /*I  "draw.h"  I*/
#include "snes.h"    /*I  "snes.h"  I*/

/*@
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
int SNESDefaultComputeJacobian(SNES snes,Vec x1,Mat *J,Mat *B,
                               MatStructure *flag,void *ctx)
{
  Vec      j1,j2,x2;
  int      i,ierr,N,start,end,j;
  Scalar   dx, mone = -1.0,*y,scale,*xx,wscale;
  double   epsilon = 1.e-8,amax; /* assumes double precision */
  MPI_Comm comm;

  PetscObjectGetComm((PetscObject)x1,&comm);
  MatZeroEntries(*J);
  ierr = VecDuplicate(x1,&j1); CHKERRQ(ierr);
  ierr = VecDuplicate(x1,&j2); CHKERRQ(ierr);
  ierr = VecDuplicate(x1,&x2); CHKERRQ(ierr);

  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);
  VecGetArray(x1,&xx);
  ierr = SNESComputeFunction(snes,x1,j1); CHKERRQ(ierr);
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(x1,x2); CHKERRQ(ierr);
    if ( i>= start && i<end) {
      dx = xx[i-start];
#if !defined(PETSC_COMPLEX)
      if (dx < 1.e-16 && dx >= 0.0) dx = 1.e-1;
      else if (dx < 0.0 && dx > -1.e-16) dx = -1.e-1;
#else
      if (abs(dx) < 1.e-16 && real(dx) >= 0.0) dx = 1.e-1;
      else if (real(dx) < 0.0 && abs(dx) > -1.e-16) dx = -1.e-1;
#endif
      dx *= epsilon;
      wscale = -1.0/dx;
      VecSetValues(x2,1,&i,&dx,ADDVALUES); 
    } 
    else {
      wscale = 0.0;
    }
    ierr = SNESComputeFunction(snes,x2,j2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,j1,j2); CHKERRQ(ierr);
/* communicate scale to all processors */
#if !defined(PETSC_COMPLEX)
    MPI_Allreduce(&wscale,&scale,1,MPI_DOUBLE,MPI_SUM,comm);
#else
    MPI_Allreduce(&wscale,&scale,2,MPI_DOUBLE,MPI_SUM,comm);
#endif
    VecScale(&scale,j2);
    VecGetArray(j2,&y);
    VecAMax(j2,0,&amax); amax *= 1.e-14;
    for ( j=start; j<end; j++ ) {
#if defined(PETSC_COMPLEX)
      if (abs(y[j-start]) > amax) {
#else
      if (y[j-start] > amax || y[j-start] < -amax) {
#endif
        ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERTVALUES); CHKERRQ(ierr);
      }
    }
    VecRestoreArray(j2,&y);
  }
  MatAssemblyBegin(*J,FINAL_ASSEMBLY);
  VecDestroy(x2); VecDestroy(j1); VecDestroy(j2);
  MatAssemblyEnd(*J,FINAL_ASSEMBLY);
  return 0;
}

