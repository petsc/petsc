#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesj2.c,v 1.6 1997/01/06 20:29:45 balay Exp balay $";
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
  Vec           w1,w2,w3;
  int           (*f)(void *,Vec,Vec,void *) = (int (*)(void *,Vec,Vec,void *))snes->computefunction;
  int           ierr;

  if (!snes->nvwork) {
    ierr = VecDuplicateVecs(x1,3,&snes->vwork); CHKERRQ(ierr);
    snes->nvwork = 3;
    PLogObjectParents(snes,3,snes->vwork);
  }
  w1 = snes->vwork[0]; w2 = snes->vwork[1]; w3 = snes->vwork[2];
  ierr = MatFDColoringApply(*B,color,x1,w1,w2,w3,f,snes,snes->funP); CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}



