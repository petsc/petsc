
#include "petscmat.h"  /*I "petscmat.h" I*/

EXTERN_C_BEGIN
EXTERN int MatCreate_MAIJ(Mat);
EXTERN int MatCreate_IS(Mat);
EXTERN int MatCreate_MPIRowbs(Mat);
EXTERN int MatCreate_SeqAIJ(Mat);
EXTERN int MatCreate_MPIAIJ(Mat);
EXTERN int MatCreate_AIJ(Mat);
EXTERN int MatCreate_SeqBAIJ(Mat);
EXTERN int MatCreate_MPIBAIJ(Mat);
EXTERN int MatCreate_BAIJ(Mat);
EXTERN int MatCreate_SeqSBAIJ(Mat);
EXTERN int MatCreate_MPISBAIJ(Mat);
EXTERN int MatCreate_SBAIJ(Mat);
EXTERN int MatCreate_SeqBDiag(Mat);
EXTERN int MatCreate_MPIBDiag(Mat);
EXTERN int MatCreate_BDiag(Mat);
EXTERN int MatCreate_SeqDense(Mat);
EXTERN int MatCreate_MPIDense(Mat);
EXTERN int MatCreate_Dense(Mat);
EXTERN int MatCreate_MPIAdj(Mat);
EXTERN int MatCreate_Shell(Mat);
#if defined(__cplusplus)
EXTERN int MatCreate_ESI(Mat);
EXTERN int MatCreate_PetscESI(Mat);
#endif
#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
EXTERN int MatCreate_SeqAIJSpooles(Mat);
EXTERN int MatCreate_SeqSBAIJSpooles(Mat);
EXTERN int MatCreate_MPIAIJSpooles(Mat);
EXTERN int MatCreate_MPISBAIJSpooles(Mat);
EXTERN int MatCreate_AIJSpooles(Mat);
EXTERN int MatCreate_SBAIJSpooles(Mat);
#endif
#if defined(PETSC_HAVE_SUPERLU) && !defined(PETSC_USE_SINGLE)
EXTERN int MatCreate_SuperLU(Mat);
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST) && !defined(PETSC_USE_SINGLE)
EXTERN int MatCreate_SuperLU_DIST(Mat);
#endif
#if defined(PETSC_HAVE_UMFPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
EXTERN int MatCreate_UMFPACK(Mat);
#endif
#if defined(PETSC_HAVE_ESSL) && !defined(__cplusplus)
EXTERN int MatCreate_Essl(Mat);
#endif
#if defined(PETSC_HAVE_LUSOL) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
EXTERN int MatCreate_LUSOL(Mat);
#endif
#if defined(PETSC_HAVE_MUMPS) && !defined(PETSC_USE_SINGLE)
EXTERN int MatCreate_AIJMUMPS(Mat);
EXTERN int MatCreate_SBAIJMUMPS(Mat);
#endif
#if defined(PETSC_HAVE_DSCPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
EXTERN int MatCreate_DSCPACK(Mat);
#endif
#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
EXTERN int MatCreate_Matlab(Mat);
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
int MatRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  MatRegisterAllCalled = PETSC_TRUE;

  ierr = MatRegisterDynamic(MATMPIMAIJ, path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQMAIJ, path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMAIJ,    path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATIS,      path,"MatCreate_IS",      MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSHELL,   path,"MatCreate_Shell",   MatCreate_Shell);CHKERRQ(ierr);
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATMPIROWBS,path,"MatCreate_MPIRowbs",MatCreate_MPIRowbs);CHKERRQ(ierr);
#endif

  ierr = MatRegisterDynamic(MATMPIAIJ,  path,"MatCreate_MPIAIJ",  MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJ,  path,"MatCreate_SeqAIJ",  MatCreate_SeqAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATAIJ,     path,"MatCreate_AIJ",     MatCreate_AIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIBAIJ,  path,"MatCreate_MPIBAIJ",  MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBAIJ,  path,"MatCreate_SeqBAIJ",  MatCreate_SeqBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATBAIJ,     path,"MatCreate_BAIJ",     MatCreate_BAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPISBAIJ,  path,"MatCreate_MPISBAIJ",  MatCreate_MPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJ,  path,"MatCreate_SeqSBAIJ",  MatCreate_SeqSBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSBAIJ,     path,"MatCreate_SBAIJ",     MatCreate_SBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIBDIAG,  path,"MatCreate_MPIBDiag",  MatCreate_MPIBDiag);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBDIAG,  path,"MatCreate_SeqBDiag",  MatCreate_SeqBDiag);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATBDIAG,     path,"MatCreate_BDiag",     MatCreate_BDiag);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIDENSE,  path,"MatCreate_MPIDense",  MatCreate_MPIDense);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQDENSE,  path,"MatCreate_SeqDense",  MatCreate_SeqDense);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATDENSE,     path,"MatCreate_Dense",     MatCreate_Dense);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIADJ,    path,"MatCreate_MPIAdj",    MatCreate_MPIAdj);CHKERRQ(ierr);
#if defined(__cplusplus) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && defined(PETSC_HAVE_CXX_NAMESPACE)
  ierr = MatRegisterDynamic(MATESI,       path,"MatCreate_ESI",    MatCreate_ESI);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATPETSCESI,  path,"MatCreate_PetscESI",    MatCreate_PetscESI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
  ierr = MatRegisterDynamic(MATSEQAIJSPOOLES,  path,"MatCreate_SeqAIJSpooles",  MatCreate_SeqAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJSPOOLES,path,"MatCreate_SeqSBAIJSpooles",MatCreate_SeqSBAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJSPOOLES,  path,"MatCreate_MPIAIJSpooles",  MatCreate_MPIAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPISBAIJSPOOLES,path,"MatCreate_MPISBAIJSpooles",MatCreate_MPISBAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATAIJSPOOLES,  path,"MatCreate_AIJSpooles",MatCreate_AIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSBAIJSPOOLES,path,"MatCreate_SBAIJSpooles",MatCreate_SBAIJSpooles);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU) && !defined(PETSC_USE_SINGLE)
  ierr = MatRegisterDynamic(MATSUPERLU,path,"MatCreate_SuperLU",MatCreate_SuperLU);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST) && !defined(PETSC_USE_SINGLE)
  ierr = MatRegisterDynamic(MATSUPERLU_DIST,path,"MatCreate_SuperLU_DIST",MatCreate_SuperLU_DIST);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_UMFPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATUMFPACK,path,"MatCreate_UMFPACK",MatCreate_UMFPACK);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ESSL) && !defined(__cplusplus)
  ierr = MatRegisterDynamic(MATESSL,path,"MatCreate_Essl",MatCreate_Essl);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_LUSOL) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATLUSOL,path,"MatCreate_LUSOL",MatCreate_LUSOL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MUMPS) && !defined(PETSC_USE_SINGLE)
  ierr = MatRegisterDynamic(MATAIJMUMPS,  path,"MatCreate_AIJMUMPS",MatCreate_AIJMUMPS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSBAIJMUMPS,path,"MatCreate_SBAIJMUMPS",MatCreate_SBAIJMUMPS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_DSCPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATDSCPACK,path,"MatCreate_DSCPACK",MatCreate_DSCPACK);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATMATLAB,path,"MatCreate_Matlab",MatCreate_Matlab);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


