
#ifndef lint
static char vcid[] = "$Id: snesj.c,v 1.4 1995/05/02 23:39:40 bsmith Exp bsmith $";
#endif

#include "draw.h"
#include "snesimpl.h"
#include "options.h"

/*@
   SNESDefaultComputeJacobian - Computes Jacobian using finite 
       differences. Slow and expensive.

 Input Parameters:
.  x - compute Jacobian at this point
.  ctx - applications Function context

  Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian

.keywords: finite differences, Jacobian

.seealso: SNESSetJacobian, SNESTestJacobian
@*/
int SNESDefaultComputeJacobian(SNES snes, Vec x1,Mat *J,Mat *B,int *flag,
                               void *ctx)
{
  Vec    j1,j2,x2;
  int    i,ierr,N,start,end,j;
  Scalar dx, mone = -1.0,*y,scale;
  double norm;

  ierr = VecCreate(x1,&j1); CHKERR(ierr);
  ierr = VecCreate(x1,&j2); CHKERR(ierr);
  ierr = VecCreate(x1,&x2); CHKERR(ierr);
  ierr = VecNorm(x1,&norm); CHKERR(ierr);
  dx = 1.e-8*norm; /* assumes double precision */
  scale = -1.0/dx;

  ierr = VecGetSize(x1,&N); CHKERR(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERR(ierr);
  ierr = SNESComputeFunction(snes,x1,j1); CHKERR(ierr);
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(x1,x2); CHKERR(ierr);
    if ( i>= start && i<end) {
      VecSetValues(x2,1,&i,&dx,ADDVALUES); 
    } 
    ierr = SNESComputeFunction(snes,x2,j2); CHKERR(ierr);
    ierr = VecAXPY(&mone,j1,j2); CHKERR(ierr);
    VecScale(&scale,j2);
    VecGetArray(j2,&y);
    for ( j=start; j<end; j++ ) {
      if (y[j-start]) {
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

/*@
     SNESTestJacobian - Tests whether a hand computed Jacobian 
        matches one compute via finite differences

  Input Parameters:

  Output Parameters:

@*/
int SNESTestJacobian(SNES snes)
{

  /* compute both versions of Jacobian */

  /* compare */

  return 0;
}
