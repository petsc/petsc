#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: matioall.c,v 1.9 1998/04/13 17:43:46 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "mat.h"

extern int MatLoad_MPIRowbs(Viewer,MatType,Mat*);
extern int MatLoad_SeqAIJ(Viewer,MatType,Mat*);
extern int MatLoad_MPIAIJ(Viewer,MatType,Mat*);
extern int MatLoad_SeqBDiag(Viewer,MatType,Mat*);
extern int MatLoad_MPIBDiag(Viewer,MatType,Mat*);
extern int MatLoad_SeqDense(Viewer,MatType,Mat*);
extern int MatLoad_MPIDense(Viewer,MatType,Mat*);
extern int MatLoad_SeqBAIJ(Viewer,MatType,Mat*);
extern int MatLoad_SeqAdj(Viewer,MatType,Mat*);
extern int MatLoad_MPIBAIJ(Viewer,MatType,Mat*);

#undef __FUNC__  
#define __FUNC__ "MatLoadRegisterAll"
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
#if defined(HAVE_BLOCKSOLVE) && !defined(USE_PETSC_COMPLEX)
  ierr = MatLoadRegister(MATMPIROWBS,MatLoad_MPIRowbs); CHKERRQ(ierr);
#endif
  ierr = MatLoadRegister(MATSEQAIJ,MatLoad_SeqAIJ); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIAIJ,MatLoad_MPIAIJ); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQBDIAG,MatLoad_SeqBDiag); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIBDIAG,MatLoad_MPIBDiag); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQDENSE,MatLoad_SeqDense); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIDENSE,MatLoad_MPIDense); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQBAIJ,MatLoad_SeqBAIJ); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATMPIBAIJ,MatLoad_MPIBAIJ); CHKERRQ(ierr);
  ierr = MatLoadRegister(MATSEQADJ,MatLoad_SeqAdj); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}  

