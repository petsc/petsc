#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsfd.c,v 1.1 1997/10/12 22:00:31 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"      /*I  "mat.h"  I*/
#include "src/ts/tsimpl.h"        /*I  "ts.h"  I*/

#undef __FUNC__  
#define __FUNC__ "TSDefaultComputeJacobianWithColoring"
/*@C
    TSDefaultComputeJacobianWithColoring - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity.  DOES NOT HANDLE
    TIME CORRECTLY!
  
    Input Parameters:
.   ts - nonlinear solver object
.   t - current time
.   x1 - location at which to evaluate Jacobian
.   ctx - coloring context, where
$      ctx must have type MatFDColoring, 
$      as created via MatFDColoringCreate()

    Output Parameters:
.   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
.   flag - flag indicating whether the matrix sparsity structure has changed

   Options Database Keys:
$  -mat_fd_coloring_freq <freq> 

.keywords: TS, finite differences, Jacobian, coloring, sparse

.seealso: TSSetJacobian()
@*/
int TSDefaultComputeJacobianWithColoring(TS ts,double t,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  SNES          snes;
  int           ierr,freq,it;

  ierr = MatFDColoringGetFrequency(color,&freq);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it); CHKERRQ(ierr);

  if ((freq > 1) && ((it % freq) != 1)) {
    PLogInfo(color,"TSDefaultComputeJacobianWithColoring:Skipping Jacobian, it %d, freq %d\n",it,freq);
    *flag = SAME_PRECONDITIONER;
    return 0;
  } else {
    PLogInfo(color,"TSDefaultComputeJacobianWithColoring:Computing Jacobian, it %d, freq %d\n",it,freq);
    *flag = SAME_NONZERO_PATTERN;
  }

  ierr = MatFDColoringApply(*B,color,x1,flag,ts); CHKERRQ(ierr);
  return 0;
}



