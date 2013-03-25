
#include <petscmat.h>  /*I "petscmat.h" I*/

PETSC_EXTERN PetscErrorCode MatCreate_MFFD(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_IS(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqSBAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPISBAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBSTRM(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIBSTRM(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqSBSTRM(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPISBSTRM(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqDense(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIDense(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_MPIAdj(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Shell(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Composite(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJPERM(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJPERM(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCRL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCRL(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_Scatter(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_BlockMat(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Nest(Mat);

#if defined PETSC_HAVE_CUSP
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSP(Mat);
#endif

#if defined PETSC_HAVE_TXPETSCGPU
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat);
#endif

#if defined PETSC_HAVE_FFTW
PETSC_EXTERN PetscErrorCode MatCreate_FFTW(Mat);
#endif
PETSC_EXTERN PetscErrorCode MatCreate_Elemental(Mat);

/*
    This is used by MatSetType() to make sure that at least one
    MatRegisterAll() is called. In general, if there is more than one
    DLL, then MatRegisterAll() may be called several times.
*/
extern PetscBool MatRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "MatRegisterAll"
/*@C
  MatRegisterAll - Registers all of the matrix types in PETSc

  Not Collective

  Level: advanced

.keywords: KSP, register, all

.seealso:  MatRegisterDestroy()
@*/
PetscErrorCode  MatRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatRegisterAllCalled = PETSC_TRUE;

  ierr = MatRegister(MATMFFD,           path,"MatCreate_MFFD",    MatCreate_MFFD);CHKERRQ(ierr);

  ierr = MatRegister(MATMPIMAIJ,        path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQMAIJ,        path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMAIJ,           path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegister(MATIS,             path,"MatCreate_IS",      MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegister(MATSHELL,          path,"MatCreate_Shell",   MatCreate_Shell);CHKERRQ(ierr);
  ierr = MatRegister(MATCOMPOSITE,      path,"MatCreate_Composite",   MatCreate_Composite);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJ,MATSEQAIJ,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJ,         path,"MatCreate_MPIAIJ",      MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJ,         path,"MatCreate_SeqAIJ",      MatCreate_SeqAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJPERM,MATSEQAIJPERM,MATMPIAIJPERM);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJPERM,     path,"MatCreate_MPIAIJPERM", MatCreate_MPIAIJPERM);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJPERM,     path,"MatCreate_SeqAIJPERM", MatCreate_SeqAIJPERM);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJCRL,      path,"MatCreate_SeqAIJCRL",  MatCreate_SeqAIJCRL);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJCRL,      path,"MatCreate_MPIAIJCRL",  MatCreate_MPIAIJCRL);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATBAIJ,MATSEQBAIJ,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIBAIJ,        path,"MatCreate_MPIBAIJ",    MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQBAIJ,        path,"MatCreate_SeqBAIJ",    MatCreate_SeqBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMPISBAIJ,       path,"MatCreate_MPISBAIJ",  MatCreate_MPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQSBAIJ,       path,"MatCreate_SeqSBAIJ",  MatCreate_SeqSBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATBSTRM,MATSEQBSTRM,MATMPIBSTRM);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIBSTRM,        path,"MatCreate_MPIBSTRM",    MatCreate_MPIBSTRM);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQBSTRM,        path,"MatCreate_SeqBSTRM",    MatCreate_SeqBSTRM);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATSBSTRM,MATSEQSBSTRM,MATMPISBSTRM);CHKERRQ(ierr);
  ierr = MatRegister(MATMPISBSTRM,       path,"MatCreate_MPISBSTRM",  MatCreate_MPISBSTRM);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQSBSTRM,       path,"MatCreate_SeqSBSTRM",  MatCreate_SeqSBSTRM);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATDENSE,MATSEQDENSE,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIDENSE,       path,"MatCreate_MPIDense",  MatCreate_MPIDense);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQDENSE,       path,"MatCreate_SeqDense",  MatCreate_SeqDense);CHKERRQ(ierr);

  ierr = MatRegister(MATMPIADJ,         path,"MatCreate_MPIAdj",    MatCreate_MPIAdj);CHKERRQ(ierr);
  ierr = MatRegister(MATSCATTER,        path,"MatCreate_Scatter",   MatCreate_Scatter);CHKERRQ(ierr);
  ierr = MatRegister(MATBLOCKMAT,       path,"MatCreate_BlockMat",  MatCreate_BlockMat);CHKERRQ(ierr);
  ierr = MatRegister(MATNEST,           path,"MatCreate_Nest",      MatCreate_Nest);CHKERRQ(ierr);


#if defined PETSC_HAVE_CUSP
  ierr = MatRegisterBaseName(MATAIJCUSP,MATSEQAIJCUSP,MATMPIAIJCUSP);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJCUSP,     path,"MatCreate_SeqAIJCUSP",  MatCreate_SeqAIJCUSP);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJCUSP,     path,"MatCreate_MPIAIJCUSP",  MatCreate_MPIAIJCUSP);CHKERRQ(ierr);
#endif

#if defined PETSC_HAVE_TXPETSCGPU
  ierr = MatRegisterBaseName(MATAIJCUSPARSE,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJCUSPARSE,     path,"MatCreate_SeqAIJCUSPARSE",  MatCreate_SeqAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJCUSPARSE,     path,"MatCreate_MPIAIJCUSPARSE",  MatCreate_MPIAIJCUSPARSE);CHKERRQ(ierr);
#endif

#if defined PETSC_HAVE_FFTW
  ierr = MatRegister(MATFFTW,           path,"MatCreate_FFTW",        MatCreate_FFTW);CHKERRQ(ierr);
#endif
#if defined PETSC_HAVE_ELEMENTAL
  ierr = MatRegister(MATELEMENTAL,      path,"MatCreate_Elemental",    MatCreate_Elemental);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


