/*$Id: matioall.c,v 1.17 2000/05/05 22:16:35 balay Exp bsmith $*/

#include "petscmat.h"

EXTERN int MatLoad_MPIRowbs(Viewer,MatType,Mat*);
EXTERN int MatLoad_SeqAIJ(Viewer,MatType,Mat*);
EXTERN int MatLoad_MPIAIJ(Viewer,MatType,Mat*);
EXTERN int MatLoad_SeqBDiag(Viewer,MatType,Mat*);
EXTERN int MatLoad_MPIBDiag(Viewer,MatType,Mat*);
EXTERN int MatLoad_SeqDense(Viewer,MatType,Mat*);
EXTERN int MatLoad_MPIDense(Viewer,MatType,Mat*);
EXTERN int MatLoad_SeqBAIJ(Viewer,MatType,Mat*);
EXTERN int MatLoad_SeqAdj(Viewer,MatType,Mat*);
EXTERN int MatLoad_MPIBAIJ(Viewer,MatType,Mat*);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatLoadRegisterAll"
/*@C
    MatLoadRegisterAll - Registers all standard matrix type routines to load
        matrices from a binary file.

  Not Collective

  Level: developer

  Notes: To prevent registering all matrix types; copy this routine to 
         your source code and comment out the versions below that you do not need.

.seealso: MatLoadRegister(), MatLoad()

@*/
int MatLoadRegisterAll(void)
{
  int ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatLoadRegister(MATMPIROWBS,MatLoad_MPIRowbs);CHKERRQ(ierr);
#endif
  ierr = MatLoadRegister(MATSEQAIJ,MatLoad_SeqAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIAIJ,MatLoad_MPIAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQBDIAG,MatLoad_SeqBDiag);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIBDIAG,MatLoad_MPIBDiag);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQDENSE,MatLoad_SeqDense);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIDENSE,MatLoad_MPIDense);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQBAIJ,MatLoad_SeqBAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIBAIJ,MatLoad_MPIBAIJ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}  

