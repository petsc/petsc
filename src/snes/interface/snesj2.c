#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesj2.c,v 1.13 1997/11/28 16:21:38 bsmith Exp bsmith $";
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

   Collective on SNES

   Options Database Keys:
$  -mat_fd_coloring_freq <freq> 

.keywords: SNES, finite differences, Jacobian, coloring, sparse

.seealso: SNESSetJacobian(), SNESTestJacobian(), SNESDefaultComputeJacobian()
@*/
int SNESDefaultComputeJacobianWithColoring(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  int           ierr,freq,it;

  PetscFunctionBegin;
  ierr = MatFDColoringGetFrequency(color,&freq);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it); CHKERRQ(ierr);

  if ((freq > 1) && ((it % freq) != 1)) {
    PLogInfo(color,"SNESDefaultComputeJacobianWithColoring:Skipping Jacobian, it %d, freq %d\n",it,freq);
    *flag = SAME_PRECONDITIONER;
    PetscFunctionReturn(0);
  } else {
    PLogInfo(color,"SNESDefaultComputeJacobianWithColoring:Computing Jacobian, it %d, freq %d\n",it,freq);
    *flag = SAME_NONZERO_PATTERN;
  }


  PLogEventBegin(SNES_FunctionEval,snes,x1,0,0);
  PetscStackPush("SNES user function");
  ierr = MatFDColoringApply(*B,color,x1,flag,snes); CHKERRQ(ierr);
  PetscStackPop;
  snes->nfuncs++;
  PLogEventEnd(SNES_FunctionEval,snes,x1,0,0);
  PetscFunctionReturn(0);
}



