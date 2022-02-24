
#include <petsc/private/matimpl.h>  /*I "petscmat.h" I*/

PETSC_EXTERN PetscErrorCode MatCreate_MFFD(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_KAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_IS(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqSBAIJ(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPISBAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqDense(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIDense(Mat);
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseCUDA(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIDenseCUDA(Mat);
#endif

PETSC_EXTERN PetscErrorCode MatCreate_MPIAdj(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Shell(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Composite(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJPERM(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJPERM(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJSELL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJSELL(Mat);

#if defined(PETSC_HAVE_MKL_SPARSE)
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJMKL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJMKL(Mat);
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJMKL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJMKL(Mat);
#endif

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCRL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCRL(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_Scatter(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_BlockMat(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Nest(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPISELL(Mat);

#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat);
#endif

#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJViennaCL(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJViennaCL(Mat);
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJKokkos(Mat);
#endif

#if defined(PETSC_HAVE_FFTW)
PETSC_EXTERN PetscErrorCode MatCreate_FFTW(Mat);
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_EXTERN PetscErrorCode MatCreate_Elemental(Mat);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_EXTERN PetscErrorCode MatCreate_ScaLAPACK(Mat);
#endif

PETSC_EXTERN PetscErrorCode MatCreate_Preallocator(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_Dummy(Mat);

#if defined(PETSC_HAVE_HYPRE)
PETSC_EXTERN PetscErrorCode MatCreate_HYPRE(Mat);
#endif

PETSC_EXTERN PetscErrorCode MatCreate_ConstantDiagonal(Mat);

#if defined(PETSC_HAVE_H2OPUS)
PETSC_EXTERN PetscErrorCode MatCreate_H2OPUS(Mat);
#endif

#if defined(PETSC_HAVE_HTOOL)
PETSC_EXTERN PetscErrorCode MatCreate_Htool(Mat);
#endif

/*@C
  MatRegisterAll - Registers all of the matrix types in PETSc

  Not Collective

  Level: advanced

.seealso:  MatRegister()
@*/
PetscErrorCode  MatRegisterAll(void)
{
  PetscFunctionBegin;
  if (MatRegisterAllCalled) PetscFunctionReturn(0);
  MatRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(MatRegister(MATMFFD,           MatCreate_MFFD));

  CHKERRQ(MatRegister(MATMPIMAIJ,        MatCreate_MAIJ));
  CHKERRQ(MatRegister(MATSEQMAIJ,        MatCreate_MAIJ));
  CHKERRQ(MatRegister(MATMAIJ,           MatCreate_MAIJ));

  CHKERRQ(MatRegister(MATMPIKAIJ,        MatCreate_KAIJ));
  CHKERRQ(MatRegister(MATSEQKAIJ,        MatCreate_KAIJ));
  CHKERRQ(MatRegister(MATKAIJ,           MatCreate_KAIJ));

  CHKERRQ(MatRegister(MATIS,             MatCreate_IS));
  CHKERRQ(MatRegister(MATSHELL,          MatCreate_Shell));
  CHKERRQ(MatRegister(MATCOMPOSITE,      MatCreate_Composite));

  CHKERRQ(MatRegisterRootName(MATAIJ,MATSEQAIJ,MATMPIAIJ));
  CHKERRQ(MatRegister(MATMPIAIJ,         MatCreate_MPIAIJ));
  CHKERRQ(MatRegister(MATSEQAIJ,         MatCreate_SeqAIJ));

  CHKERRQ(MatRegisterRootName(MATAIJPERM,MATSEQAIJPERM,MATMPIAIJPERM));
  CHKERRQ(MatRegister(MATMPIAIJPERM,     MatCreate_MPIAIJPERM));
  CHKERRQ(MatRegister(MATSEQAIJPERM,     MatCreate_SeqAIJPERM));

  CHKERRQ(MatRegisterRootName(MATAIJSELL,MATSEQAIJSELL,MATMPIAIJSELL));
  CHKERRQ(MatRegister(MATMPIAIJSELL,     MatCreate_MPIAIJSELL));
  CHKERRQ(MatRegister(MATSEQAIJSELL,     MatCreate_SeqAIJSELL));

#if defined(PETSC_HAVE_MKL_SPARSE)
  CHKERRQ(MatRegisterRootName(MATAIJMKL, MATSEQAIJMKL,MATMPIAIJMKL));
  CHKERRQ(MatRegister(MATMPIAIJMKL,      MatCreate_MPIAIJMKL));
  CHKERRQ(MatRegister(MATSEQAIJMKL,      MatCreate_SeqAIJMKL));
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  CHKERRQ(MatRegisterRootName(MATBAIJMKL,MATSEQBAIJMKL,MATMPIBAIJMKL));
  CHKERRQ(MatRegister(MATMPIBAIJMKL,      MatCreate_MPIBAIJMKL));
  CHKERRQ(MatRegister(MATSEQBAIJMKL,      MatCreate_SeqBAIJMKL));
#endif

  CHKERRQ(MatRegisterRootName(MATAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL));
  CHKERRQ(MatRegister(MATSEQAIJCRL,      MatCreate_SeqAIJCRL));
  CHKERRQ(MatRegister(MATMPIAIJCRL,      MatCreate_MPIAIJCRL));

  CHKERRQ(MatRegisterRootName(MATBAIJ,MATSEQBAIJ,MATMPIBAIJ));
  CHKERRQ(MatRegister(MATMPIBAIJ,        MatCreate_MPIBAIJ));
  CHKERRQ(MatRegister(MATSEQBAIJ,        MatCreate_SeqBAIJ));

  CHKERRQ(MatRegisterRootName(MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ));
  CHKERRQ(MatRegister(MATMPISBAIJ,       MatCreate_MPISBAIJ));
  CHKERRQ(MatRegister(MATSEQSBAIJ,       MatCreate_SeqSBAIJ));

  CHKERRQ(MatRegisterRootName(MATDENSE,MATSEQDENSE,MATMPIDENSE));
  CHKERRQ(MatRegister(MATMPIDENSE,       MatCreate_MPIDense));
  CHKERRQ(MatRegister(MATSEQDENSE,       MatCreate_SeqDense));
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(MatRegisterRootName(MATDENSECUDA,MATSEQDENSECUDA,MATMPIDENSECUDA));
  CHKERRQ(MatRegister(MATSEQDENSECUDA,   MatCreate_SeqDenseCUDA));
  CHKERRQ(MatRegister(MATMPIDENSECUDA,   MatCreate_MPIDenseCUDA));
#endif

  CHKERRQ(MatRegister(MATMPIADJ,         MatCreate_MPIAdj));
  CHKERRQ(MatRegister(MATSCATTER,        MatCreate_Scatter));
  CHKERRQ(MatRegister(MATBLOCKMAT,       MatCreate_BlockMat));
  CHKERRQ(MatRegister(MATNEST,           MatCreate_Nest));

  CHKERRQ(MatRegisterRootName(MATSELL,MATSEQSELL,MATMPISELL));
  CHKERRQ(MatRegister(MATMPISELL,         MatCreate_MPISELL));
  CHKERRQ(MatRegister(MATSEQSELL,         MatCreate_SeqSELL));

#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(MatRegisterRootName(MATAIJCUSPARSE,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE));
  CHKERRQ(MatRegister(MATSEQAIJCUSPARSE, MatCreate_SeqAIJCUSPARSE));
  CHKERRQ(MatRegister(MATMPIAIJCUSPARSE, MatCreate_MPIAIJCUSPARSE));
#endif

#if defined(PETSC_HAVE_VIENNACL)
  CHKERRQ(MatRegisterRootName(MATAIJVIENNACL,MATSEQAIJVIENNACL,MATMPIAIJVIENNACL));
  CHKERRQ(MatRegister(MATSEQAIJVIENNACL, MatCreate_SeqAIJViennaCL));
  CHKERRQ(MatRegister(MATMPIAIJVIENNACL, MatCreate_MPIAIJViennaCL));
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  CHKERRQ(MatRegisterRootName(MATAIJKOKKOS,MATSEQAIJKOKKOS,MATMPIAIJKOKKOS));
  CHKERRQ(MatRegister(MATSEQAIJKOKKOS,   MatCreate_SeqAIJKokkos));
  CHKERRQ(MatRegister(MATMPIAIJKOKKOS,   MatCreate_MPIAIJKokkos));
#endif

#if defined(PETSC_HAVE_FFTW)
  CHKERRQ(MatRegister(MATFFTW,           MatCreate_FFTW));
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  CHKERRQ(MatRegister(MATELEMENTAL,      MatCreate_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  CHKERRQ(MatRegister(MATSCALAPACK,      MatCreate_ScaLAPACK));
#endif

  CHKERRQ(MatRegister(MATPREALLOCATOR,   MatCreate_Preallocator));
  CHKERRQ(MatRegister(MATDUMMY,          MatCreate_Dummy));

  CHKERRQ(MatRegister(MATCONSTANTDIAGONAL,MatCreate_ConstantDiagonal));

#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(MatRegister(MATHYPRE,          MatCreate_HYPRE));
#endif

#if defined(PETSC_HAVE_H2OPUS)
  CHKERRQ(MatRegister(MATH2OPUS,         MatCreate_H2OPUS));
#endif

#if defined(PETSC_HAVE_HTOOL)
  CHKERRQ(MatRegister(MATHTOOL,          MatCreate_Htool));
#endif
  PetscFunctionReturn(0);
}
