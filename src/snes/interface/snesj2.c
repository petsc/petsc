#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesj2.c,v 1.9 1997/09/25 22:40:16 curfman Exp curfman $";
#endif

#include "src/mat/matimpl.h"      /*I  "mat.h"  I*/
#include "src/snes/snesimpl.h"    /*I  "snes.h"  I*/

#undef __FUNC__  
#define __FUNC__ "SNESDefaultComputeJacobianWithColoring"
/*@C
    SNESDefaultComputeJacobianWithColoring - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity. 
  
    Input Parameters:
.   snes - nonlinear solver object
.   x1 - location at which to evaluate Jacobian
.   ctx - coloring context, where
$      ctx must have type MatFDColoring, 
$      as created via MatFDColoringCreate()

    Output Parameters:
.   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
.   flag - flag indicating whether the matrix sparsity structure has changed

.keywords: SNES, finite differences, Jacobian, coloring, sparse

.seealso: SNESSetJacobian(), SNESTestJacobian(), SNESDefaultComputeJacobian()
@*/
int SNESDefaultComputeJacobianWithColoring(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  int           ierr;

  ierr = MatFDColoringApply(*B,color,x1,flag,snes); CHKERRQ(ierr);
  return 0;
}



