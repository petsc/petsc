

#ifndef lint
static char vcid[] = "$Id: snesj.c,v 1.9 1995/05/11 17:53:49 curfman Exp bsmith $";
#endif

#include "draw.h"
#include "snes.h"
#include "options.h"

/*@
   SNESDefaultComputeJacobian - Computes the Jacobian using finite 
   differences. 

   Input Parameters:
.  x - compute Jacobian at this point
.  ctx - application's Function context

   Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   SNESDefaultComputeJacobian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Jacobian.

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESTestJacobian()
@*/
int SNESDefaultComputeJacobian(SNES snes, Vec x1,Mat *J,Mat *B,int *flag,
                               void *ctx)
{
  Vec    j1,j2,x2;
  int    i,ierr,N,start,end,j;
  Scalar dx, mone = -1.0,*y,scale,*xx;
  double epsilon = 1.e-8,amax; /* assumes double precision */

  MatZeroEntries(*J);
  ierr = VecDuplicate(x1,&j1); CHKERR(ierr);
  ierr = VecDuplicate(x1,&j2); CHKERR(ierr);
  ierr = VecDuplicate(x1,&x2); CHKERR(ierr);

  ierr = VecGetSize(x1,&N); CHKERR(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERR(ierr);
  VecGetArray(x1,&xx);
  ierr = SNESComputeFunction(snes,x1,j1); CHKERR(ierr);
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(x1,x2); CHKERR(ierr);
    if ( i>= start && i<end) {
      dx = xx[i-start];
      if (dx < 1.e-16 && dx >= 0.0) dx = 1.e-1;
      else if (dx < 0.0 && dx > -1.e-16) dx = -1.e-1;
      dx *= epsilon;
      scale = -1.0/dx;
      VecSetValues(x2,1,&i,&dx,ADDVALUES); 
    } 
    ierr = SNESComputeFunction(snes,x2,j2); CHKERR(ierr);
    ierr = VecAXPY(&mone,j1,j2); CHKERR(ierr);
    VecScale(&scale,j2);
    VecGetArray(j2,&y);
    VecAMax(j2,0,&amax); amax *= 1.e-14;
    for ( j=start; j<end; j++ ) {
      if (y[j-start] > amax || y[j-start] < -amax) {
        ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERTVALUES); CHKERR(ierr);
      }
    }
    VecRestoreArray(j2,&y);
  }
  MatAssemblyBegin(*J,FINAL_ASSEMBLY);
  VecDestroy(x2); VecDestroy(j1); VecDestroy(j2);
  MatAssemblyEnd(*J,FINAL_ASSEMBLY);
  return 0;
}

