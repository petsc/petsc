/*$Id: snesj2.c,v 1.31 2001/06/21 21:18:37 bsmith Exp $*/

#include "src/mat/matimpl.h"      /*I  "petscmat.h"  I*/
#include "src/snes/snesimpl.h"    /*I  "petscsnes.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultComputeJacobianColor"
/*@C
    SNESDefaultComputeJacobianColor - Computes the Jacobian using
    finite differences and coloring to exploit matrix sparsity. 
  
    Collective on SNES

    Input Parameters:
+   snes - nonlinear solver object
.   x1 - location at which to evaluate Jacobian
-   ctx - coloring context, where ctx must have type MatFDColoring, 
          as created via MatFDColoringCreate()

    Output Parameters:
+   J - Jacobian matrix (not altered in this routine)
.   B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)
-   flag - flag indicating whether the matrix sparsity structure has changed

    Options Database Keys:
.  -mat_fd_coloring_freq <freq> - Activates SNESDefaultComputeJacobianColor()

    Level: intermediate

.keywords: SNES, finite differences, Jacobian, coloring, sparse

.seealso: SNESSetJacobian(), SNESTestJacobian(), SNESDefaultComputeJacobian()
          TSDefaultComputeJacobianColor(), MatFDColoringCreate(),
          MatFDColoringSetFunction()

@*/
int SNESDefaultComputeJacobianColor(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  int           ierr,freq,it;
  Vec           f;

  PetscFunctionBegin;
  ierr = MatFDColoringGetFrequency(color,&freq);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);

  if ((freq > 1) && ((it % freq))) {
    PetscLogInfo(color,"SNESDefaultComputeJacobianColor:Skipping Jacobian recomputation, it %d, freq %d\n",it,freq);
    *flag = SAME_PRECONDITIONER;
  } else {
    PetscLogInfo(color,"SNESDefaultComputeJacobianColor:Computing Jacobian, it %d, freq %d\n",it,freq);
    *flag = SAME_NONZERO_PATTERN;
    ierr  = SNESGetFunction(snes,&f,0,0);CHKERRQ(ierr);
    ierr  = MatFDColoringSetF(color,f);CHKERRQ(ierr);
    ierr  = MatFDColoringApply(*B,color,x1,flag,snes);CHKERRQ(ierr);
  }
  if (J != B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



