#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesj2.c,v 1.7 1997/07/09 20:59:37 balay Exp bsmith $";
#endif

#include "src/mat/matimpl.h"      /*I  "mat.h"  I*/
#include "src/snes/snesimpl.h"    /*I  "snes.h"  I*/


#undef __FUNC__  
#define __FUNC__ "SNESDefaultComputeJacobianWithColoring"
/*@C
     SNESDefaultComputeJacobianWithColoring
  
   Input Parameters:
.    snes - nonlinear solver object
.    x1 - location at which to evaluate Jacobian
.    ctx - MatFDColoring contex

   Output Parameters:
.    J - Jacobian matrix
.    B - Jacobian preconditioner
.    flag - flag indicating if the matrix nonzero structure has changed

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESTestJacobian()
@*/
int SNESDefaultComputeJacobianWithColoring(SNES snes,Vec x1,Mat *JJ,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  int           ierr;

  ierr = MatFDColoringApply(*B,color,x1,flag,snes); CHKERRQ(ierr);
  return 0;
}



