
#include <petsc/private/matimpl.h>  /*I "petscmat.h" I*/

PETSC_EXTERN PetscErrorCode MatCreate_MFFD(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_IS(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqSBAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPISBAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqDense(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIDense(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_MPIAdj(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Shell(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Composite(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJPERM(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJPERM(Mat);

#if defined PETSC_HAVE_MKL
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJMKL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJMKL(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJMKL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJMKL(Mat);
#endif

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCRL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCRL(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_Scatter(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_BlockMat(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Nest(Mat);

#if defined PETSC_HAVE_CUSP
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSP(Mat);
#endif

#if defined PETSC_HAVE_VECCUDA
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat);
#endif

#if defined PETSC_HAVE_VIENNACL
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJViennaCL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJViennaCL(Mat);
#endif

#if defined PETSC_HAVE_FFTW
PETSC_EXTERN PetscErrorCode MatCreate_FFTW(Mat);
#endif
PETSC_EXTERN PetscErrorCode MatCreate_Elemental(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_Preallocator(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Dummy(Mat);

#if defined PETSC_HAVE_HYPRE
PETSC_EXTERN PetscErrorCode MatCreate_HYPRE(Mat);
#endif

/*@C
  MatRegisterAll - Registers all of the matrix types in PETSc

  Not Collective

  Level: advanced

.keywords: KSP, register, all

.seealso:  MatRegister()
@*/
PetscErrorCode  MatRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatRegisterAllCalled) PetscFunctionReturn(0);
  MatRegisterAllCalled = PETSC_TRUE;

  ierr = MatRegister(MATMFFD,           MatCreate_MFFD);CHKERRQ(ierr);

  ierr = MatRegister(MATMPIMAIJ,        MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQMAIJ,        MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMAIJ,           MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegister(MATIS,             MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegister(MATSHELL,          MatCreate_Shell);CHKERRQ(ierr);
  ierr = MatRegister(MATCOMPOSITE,      MatCreate_Composite);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJ,MATSEQAIJ,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJ,         MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJ,         MatCreate_SeqAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJPERM,MATSEQAIJPERM,MATMPIAIJPERM);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJPERM,     MatCreate_MPIAIJPERM);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJPERM,     MatCreate_SeqAIJPERM);CHKERRQ(ierr);

#if defined PETSC_HAVE_MKL
  ierr = MatRegisterBaseName(MATAIJMKL, MATSEQAIJMKL,MATMPIAIJMKL);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJMKL,      MatCreate_MPIAIJMKL);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJMKL,      MatCreate_SeqAIJMKL);CHKERRQ(ierr);
  
  ierr = MatRegisterBaseName(MATBAIJMKL,MATSEQBAIJMKL,MATMPIBAIJMKL);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIBAIJMKL,      MatCreate_MPIBAIJMKL);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQBAIJMKL,      MatCreate_SeqBAIJMKL);CHKERRQ(ierr);
#endif

  ierr = MatRegisterBaseName(MATAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJCRL,      MatCreate_SeqAIJCRL);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJCRL,      MatCreate_MPIAIJCRL);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATBAIJ,MATSEQBAIJ,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIBAIJ,        MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQBAIJ,        MatCreate_SeqBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATMPISBAIJ,       MatCreate_MPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQSBAIJ,       MatCreate_SeqSBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATDENSE,MATSEQDENSE,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIDENSE,       MatCreate_MPIDense);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQDENSE,       MatCreate_SeqDense);CHKERRQ(ierr);

  ierr = MatRegister(MATMPIADJ,         MatCreate_MPIAdj);CHKERRQ(ierr);
  ierr = MatRegister(MATSCATTER,        MatCreate_Scatter);CHKERRQ(ierr);
  ierr = MatRegister(MATBLOCKMAT,       MatCreate_BlockMat);CHKERRQ(ierr);
  ierr = MatRegister(MATNEST,           MatCreate_Nest);CHKERRQ(ierr);

#if defined PETSC_HAVE_CUSP
  ierr = MatRegisterBaseName(MATAIJCUSP,MATSEQAIJCUSP,MATMPIAIJCUSP);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJCUSP,     MatCreate_SeqAIJCUSP);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJCUSP,     MatCreate_MPIAIJCUSP);CHKERRQ(ierr);
#endif

#if defined PETSC_HAVE_VECCUDA
  ierr = MatRegisterBaseName(MATAIJCUSPARSE,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJCUSPARSE, MatCreate_SeqAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJCUSPARSE, MatCreate_MPIAIJCUSPARSE);CHKERRQ(ierr);
#endif

#if defined PETSC_HAVE_VIENNACL
  ierr = MatRegisterBaseName(MATAIJVIENNACL,MATSEQAIJVIENNACL,MATMPIAIJVIENNACL);CHKERRQ(ierr);
  ierr = MatRegister(MATSEQAIJVIENNACL, MatCreate_SeqAIJViennaCL);CHKERRQ(ierr);
  ierr = MatRegister(MATMPIAIJVIENNACL, MatCreate_MPIAIJViennaCL);CHKERRQ(ierr);
#endif

#if defined PETSC_HAVE_FFTW
  ierr = MatRegister(MATFFTW,           MatCreate_FFTW);CHKERRQ(ierr);
#endif
#if defined PETSC_HAVE_ELEMENTAL
  ierr = MatRegister(MATELEMENTAL,      MatCreate_Elemental);CHKERRQ(ierr);
#endif

  ierr = MatRegister(MATPREALLOCATOR,   MatCreate_Preallocator);CHKERRQ(ierr);
  ierr = MatRegister(MATDUMMY,          MatCreate_Dummy);CHKERRQ(ierr);

#if defined PETSC_HAVE_HYPRE
  ierr = MatRegister(MATHYPRE,          MatCreate_HYPRE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


