/*$Id: matioall.c,v 1.23 2001/03/23 23:22:45 balay Exp $*/

#include "petscmat.h"

EXTERN_C_BEGIN
EXTERN int MatLoad_MPIRowbs(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_SeqAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_MPIAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_SeqBDiag(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_MPIBDiag(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_SeqDense(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_MPIDense(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_SeqBAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_SeqAdj(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_MPIBAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_SeqSBAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_MPISBAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_MPIRowbs(PetscViewer,MatType,Mat*);
EXTERN int MatLoad_ESI(PetscViewer,MatType,Mat*);
#if defined(PETSC_HAVE_SUPERLUDIST) && !defined(PETSC_USE_SINGLE)
EXTERN int MatLoad_MPIAIJ_SuperLU_DIST(PetscViewer,MatType,Mat*);
#endif
EXTERN_C_END
extern PetscTruth MatLoadRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "MatLoadRegisterAll"
/*@C
    MatLoadRegisterAll - Registers all standard matrix type routines to load
        matrices from a binary file.

  Not Collective

  Level: developer

  Notes: To prevent registering all matrix types; copy this routine to 
         your source code and comment out the versions below that you do not need.

.seealso: MatLoadRegister(), MatLoad()

@*/
int MatLoadRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  MatLoadRegisterAllCalled = PETSC_TRUE;
  ierr = MatLoadRegisterDynamic(MATSEQAIJ,path,"MatLoad_SeqAIJ",MatLoad_SeqAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPIAIJ,path,"MatLoad_MPIAIJ",MatLoad_MPIAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATSEQBDIAG,path,"MatLoad_SeqBDiag",MatLoad_SeqBDiag);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPIBDIAG,path,"MatLoad_MPIBDiag",MatLoad_MPIBDiag);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATSEQDENSE,path,"MatLoad_SeqDense",MatLoad_SeqDense);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPIDENSE,path,"MatLoad_MPIDense",MatLoad_MPIDense);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATSEQBAIJ,path,"MatLoad_SeqBAIJ",MatLoad_SeqBAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPIBAIJ,path,"MatLoad_MPIBAIJ",MatLoad_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATSEQSBAIJ,path,"MatLoad_SeqSBAIJ",MatLoad_SeqSBAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPISBAIJ,path,"MatLoad_MPISBAIJ",MatLoad_MPISBAIJ);CHKERRQ(ierr);
#if defined(__cplusplus) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && defined(PETSC_HAVE_CXX_NAMESPACE)
  ierr = MatLoadRegisterDynamic(MATESI,path,"MatLoad_ESI",MatLoad_ESI);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATPETSCESI,path,"MatLoad_ESI",MatLoad_ESI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatLoadRegisterDynamic(MATMPIROWBS,path,"MatLoad_MPIRowbs",MatLoad_MPIRowbs);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
  ierr = MatLoadRegisterDynamic(MATSEQAIJSPOOLES,  path,"MatLoad_SeqAIJ",  MatLoad_SeqAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATSEQSBAIJSPOOLES,path,"MatLoad_SeqSBAIJ",MatLoad_SeqSBAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPIAIJSPOOLES,  path,"MatLoad_MPIAIJ",  MatLoad_MPIAIJ);CHKERRQ(ierr);
  ierr = MatLoadRegisterDynamic(MATMPISBAIJSPOOLES,path,"MatLoad_MPISBAIJ",MatLoad_MPISBAIJ);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU) && !defined(PETSC_USE_SINGLE)
  ierr = MatLoadRegisterDynamic(MATSUPERLU,path,"MatLoad_SeqAIJ",MatLoad_SeqAIJ);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLUDIST) && !defined(PETSC_USE_SINGLE)
  ierr = MatLoadRegisterDynamic(MATSUPERLUDIST,path,"MatLoad_MPIAIJ_SuperLU_DIST",MatLoad_MPIAIJ_SuperLU_DIST);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}  

EXTERN_C_BEGIN
EXTERN int MatConvertTo_MPIAdj(Mat,MatType,Mat*);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatConvertRegisterAll"
/*@C
    MatConvertRegisterAll - Registers all standard matrix type routines to convert to

  Not Collective

  Level: developer

  Notes: To prevent registering all matrix types; copy this routine to 
         your source code and comment out the versions below that you do not need.

.seealso: MatLoadRegister(), MatLoad()

@*/
int MatConvertRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  MatConvertRegisterAllCalled = PETSC_TRUE;
  ierr = MatConvertRegisterDynamic(MATMPIADJ,path,"MatConvertTo_MPIAdj",MatConvertTo_MPIAdj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  
