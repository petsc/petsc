/*$Id: matregis.c,v 1.10 2001/06/21 21:16:17 bsmith Exp $*/

#include "petscmat.h"  /*I "petscmat.h" I*/

EXTERN_C_BEGIN
EXTERN int MatCreate_MAIJ(Mat);
EXTERN int MatCreate_IS(Mat);
EXTERN int MatCreate_MPIRowbs(Mat);
EXTERN int MatCreate_SeqAIJ(Mat);
EXTERN int MatCreate_MPIAIJ(Mat);
EXTERN int MatCreate_SeqBAIJ(Mat);
EXTERN int MatCreate_MPIBAIJ(Mat);
EXTERN int MatCreate_SeqSBAIJ(Mat);
EXTERN int MatCreate_MPISBAIJ(Mat);
EXTERN int MatCreate_SeqBDiag(Mat);
EXTERN int MatCreate_MPIBDiag(Mat);
EXTERN int MatCreate_SeqDense(Mat);
EXTERN int MatCreate_MPIDense(Mat);
EXTERN int MatCreate_MPIAdj(Mat);
EXTERN int MatCreate_Shell(Mat);
#if defined(__cplusplus)
EXTERN int MatCreate_ESI(Mat);
EXTERN int MatCreate_PetscESI(Mat);
#endif
#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
EXTERN int MatCreate_SeqAIJ_Spooles(Mat);
EXTERN int MatCreate_SeqSBAIJ_Spooles(Mat);
EXTERN int MatCreate_MPIAIJ_Spooles(Mat);
EXTERN int MatCreate_MPISBAIJ_Spooles(Mat);
#endif
EXTERN_C_END
  
/*
    This is used by MatSetType() to make sure that at least one 
    MatRegisterAll() is called. In general, if there is more than one
    DLL, then MatRegisterAll() may be called several times.
*/
EXTERN PetscTruth MatRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "MatRegisterAll"
/*@C
  MatRegisterAll - Registers all of the matrix types in PETSc

  Not Collective

  Level: advanced

.keywords: KSP, register, all

.seealso:  MatRegisterDestroy()
@*/
int MatRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  MatRegisterAllCalled = PETSC_TRUE;

  ierr = MatRegisterDynamic(MATMPIMAIJ, path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQMAIJ, path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATIS,      path,"MatCreate_IS",      MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSHELL,   path,"MatCreate_Shell",   MatCreate_Shell);CHKERRQ(ierr);
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATMPIROWBS,path,"MatCreate_MPIRowbs",MatCreate_MPIRowbs);CHKERRQ(ierr);
#endif

  ierr = MatRegisterDynamic(MATMPIAIJ,  path,"MatCreate_MPIAIJ",  MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJ,  path,"MatCreate_SeqAIJ",  MatCreate_SeqAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIBAIJ,  path,"MatCreate_MPIBAIJ",  MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBAIJ,  path,"MatCreate_SeqBAIJ",  MatCreate_SeqBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPISBAIJ,  path,"MatCreate_MPISBAIJ",  MatCreate_MPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJ,  path,"MatCreate_SeqSBAIJ",  MatCreate_SeqSBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIBDIAG,  path,"MatCreate_MPIBDiag",  MatCreate_MPIBDiag);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBDIAG,  path,"MatCreate_SeqBDiag",  MatCreate_SeqBDiag);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIDENSE,  path,"MatCreate_MPIDense",  MatCreate_MPIDense);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQDENSE,  path,"MatCreate_SeqDense",  MatCreate_SeqDense);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIADJ,    path,"MatCreate_MPIAdj",    MatCreate_MPIAdj);CHKERRQ(ierr);
#if defined(__cplusplus) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && defined(PETSC_HAVE_CXX_NAMESPACE)
  ierr = MatRegisterDynamic(MATESI,       path,"MatCreate_ESI",    MatCreate_ESI);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATPETSCESI,  path,"MatCreate_PetscESI",    MatCreate_PetscESI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
  ierr = MatRegisterDynamic(MATSEQAIJSPOOLES,  path,"MatCreate_SeqAIJ_Spooles",  MatCreate_SeqAIJ_Spooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJSPOOLES,path,"MatCreate_SeqSBAIJ_Spooles",MatCreate_SeqSBAIJ_Spooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJSPOOLES,  path,"MatCreate_MPIAIJ_Spooles",  MatCreate_MPIAIJ_Spooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPISBAIJSPOOLES,path,"MatCreate_MPISBAIJ_Spooles",MatCreate_MPISBAIJ_Spooles);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


