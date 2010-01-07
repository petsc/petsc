#define PETSCMAT_DLL

#include "petscmat.h"  /*I "petscmat.h" I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MFFD(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_IS(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAIJ(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_BAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIBAIJ(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqSBAIJ(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPISBAIJ(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Dense(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqDense(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIDense(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAdj(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Shell(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Composite(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_CSRPERM(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCSRPERM(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPICSRPERM(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_CRL(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCRL(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPICRL(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Scatter(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_BlockMat(Mat);


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

  ierr = MatRegisterDynamic(MATMFFD,           path,"MatCreate_MFFD",    MatCreate_MFFD);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIMAIJ,        path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQMAIJ,        path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMAIJ,           path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATIS,             path,"MatCreate_IS",      MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSHELL,          path,"MatCreate_Shell",   MatCreate_Shell);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATCOMPOSITE,      path,"MatCreate_Composite",   MatCreate_Composite);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATAIJ,            path,"MatCreate_AIJ",         MatCreate_AIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJ,         path,"MatCreate_MPIAIJ",      MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJ,         path,"MatCreate_SeqAIJ",      MatCreate_SeqAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATCSRPERM,        path,"MatCreate_CSRPERM",  MatCreate_CSRPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPICSRPERM,     path,"MatCreate_MPICSRPERM",  MatCreate_MPICSRPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQCSRPERM,     path,"MatCreate_SeqCSRPERM",  MatCreate_SeqCSRPERM);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATCRL,            path,"MatCreate_CRL",         MatCreate_CRL);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQCRL,         path,"MatCreate_SeqCRL",      MatCreate_SeqCRL);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPICRL,         path,"MatCreate_MPICRL",      MatCreate_MPICRL);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATBAIJ,           path,"MatCreate_BAIJ",       MatCreate_BAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIBAIJ,        path,"MatCreate_MPIBAIJ",    MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBAIJ,        path,"MatCreate_SeqBAIJ",    MatCreate_SeqBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATSBAIJ,          path,"MatCreate_SBAIJ",     MatCreate_SBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPISBAIJ,       path,"MatCreate_MPISBAIJ",  MatCreate_MPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJ,       path,"MatCreate_SeqSBAIJ",  MatCreate_SeqSBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATDENSE,          path,"MatCreate_Dense",     MatCreate_Dense);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIDENSE,       path,"MatCreate_MPIDense",  MatCreate_MPIDense);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQDENSE,       path,"MatCreate_SeqDense",  MatCreate_SeqDense);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIADJ,         path,"MatCreate_MPIAdj",    MatCreate_MPIAdj);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSCATTER,        path,"MatCreate_Scatter",   MatCreate_Scatter);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATBLOCKMAT,       path,"MatCreate_BlockMat",   MatCreate_BlockMat);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


