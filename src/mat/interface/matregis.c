#define PETSCMAT_DLL

#include "petscmat.h"  /*I "petscmat.h" I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_IS(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIRowbs(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_BAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqSBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPISBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqBDiag(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIBDiag(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_BDiag(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqDense(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIDense(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Dense(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAdj(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Shell(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCSRPERM(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCRL(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPICSRPERM(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_CSRPERM(Mat);
#if defined(PETSC_HAVE_SPOOLES)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqAIJSpooles(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqSBAIJSpooles(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAIJSpooles(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPISBAIJSpooles(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJSpooles(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SBAIJSpooles(Mat);
#endif
#if defined(PETSC_HAVE_SUPERLU)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SuperLU(Mat);
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SuperLU_DIST(Mat);
#endif
#if defined(PETSC_HAVE_UMFPACK)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_UMFPACK(Mat);
#endif
#if defined(PETSC_HAVE_ESSL) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Essl(Mat);
#endif
#if defined(PETSC_HAVE_LUSOL)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_LUSOL(Mat);
#endif
#if defined(PETSC_HAVE_MUMPS)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJMUMPS(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SBAIJMUMPS(Mat);
#endif
#if defined(PETSC_HAVE_DSCPACK)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_DSCPACK(Mat);
#endif
#if defined(PETSC_HAVE_MATLAB)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Matlab(Mat);
#endif
#if defined(PETSC_HAVE_PLAPACK)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Plapack(Mat);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatRegisterAllCalled = PETSC_TRUE;

  ierr = MatRegisterDynamic(MATMPIMAIJ, path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQMAIJ, path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMAIJ,    path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATIS,      path,"MatCreate_IS",      MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSHELL,   path,"MatCreate_Shell",   MatCreate_Shell);CHKERRQ(ierr);
#if defined(PETSC_HAVE_BLOCKSOLVE)
  ierr = MatRegisterDynamic(MATMPIROWBS,path,"MatCreate_MPIRowbs",MatCreate_MPIRowbs);CHKERRQ(ierr);
#endif
  ierr = MatRegisterDynamic(MATMPIAIJ,  path,"MatCreate_MPIAIJ",      MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJ,  path,"MatCreate_SeqAIJ",      MatCreate_SeqAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATCSRPERM,  path,"MatCreate_CSRPERM",  MatCreate_CSRPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPICSRPERM,  path,"MatCreate_MPICSRPERM",  MatCreate_MPICSRPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQCSRPERM,  path,"MatCreate_SeqCSRPERM",  MatCreate_SeqCSRPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQCRL,  path,"MatCreate_SeqCRL",      MatCreate_SeqCRL);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPICRL,  path,"MatCreate_MPICRL",      MatCreate_MPICRL);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATAIJ,     path,"MatCreate_AIJ",         MatCreate_AIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIBAIJ,  path,"MatCreate_MPIBAIJ",    MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBAIJ,  path,"MatCreate_SeqBAIJ",    MatCreate_SeqBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATBAIJ,     path,"MatCreate_BAIJ",       MatCreate_BAIJ);CHKERRQ(ierr);

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
#if defined(PETSC_HAVE_SPOOLES)
  ierr = MatRegisterDynamic(MATSEQAIJSPOOLES,  path,"MatCreate_SeqAIJSpooles",  MatCreate_SeqAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJSPOOLES,path,"MatCreate_SeqSBAIJSpooles",MatCreate_SeqSBAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJSPOOLES,  path,"MatCreate_MPIAIJSpooles",  MatCreate_MPIAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPISBAIJSPOOLES,path,"MatCreate_MPISBAIJSpooles",MatCreate_MPISBAIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATAIJSPOOLES,     path,"MatCreate_AIJSpooles",     MatCreate_AIJSpooles);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSBAIJSPOOLES,   path,"MatCreate_SBAIJSpooles",   MatCreate_SBAIJSpooles);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU)
  ierr = MatRegisterDynamic(MATSUPERLU,path,"MatCreate_SuperLU",MatCreate_SuperLU);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
  ierr = MatRegisterDynamic(MATSUPERLU_DIST,path,"MatCreate_SuperLU_DIST",MatCreate_SuperLU_DIST);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_UMFPACK)
  ierr = MatRegisterDynamic(MATUMFPACK,path,"MatCreate_UMFPACK",MatCreate_UMFPACK);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ESSL) && !defined(PETSC_USE_COMPLEX)
  ierr = MatRegisterDynamic(MATESSL,path,"MatCreate_Essl",MatCreate_Essl);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_LUSOL)
  ierr = MatRegisterDynamic(MATLUSOL,path,"MatCreate_LUSOL",MatCreate_LUSOL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatRegisterDynamic(MATAIJMUMPS,  path,"MatCreate_AIJMUMPS",MatCreate_AIJMUMPS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSBAIJMUMPS,path,"MatCreate_SBAIJMUMPS",MatCreate_SBAIJMUMPS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_DSCPACK)
  ierr = MatRegisterDynamic(MATDSCPACK,path,"MatCreate_DSCPACK",MatCreate_DSCPACK);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB)
  ierr = MatRegisterDynamic(MATMATLAB,path,"MatCreate_Matlab",MatCreate_Matlab);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PLAPACK)
  ierr = MatRegisterDynamic(MATPLAPACK,path,"MatCreate_Plapack",MatCreate_Plapack);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


