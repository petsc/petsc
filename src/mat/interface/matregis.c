
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

.seealso: `MatRegister()`
@*/
PetscErrorCode  MatRegisterAll(void)
{
  PetscFunctionBegin;
  if (MatRegisterAllCalled) PetscFunctionReturn(0);
  MatRegisterAllCalled = PETSC_TRUE;

  PetscCall(MatRegister(MATMFFD,           MatCreate_MFFD));

  PetscCall(MatRegister(MATMPIMAIJ,        MatCreate_MAIJ));
  PetscCall(MatRegister(MATSEQMAIJ,        MatCreate_MAIJ));
  PetscCall(MatRegister(MATMAIJ,           MatCreate_MAIJ));

  PetscCall(MatRegister(MATMPIKAIJ,        MatCreate_KAIJ));
  PetscCall(MatRegister(MATSEQKAIJ,        MatCreate_KAIJ));
  PetscCall(MatRegister(MATKAIJ,           MatCreate_KAIJ));

  PetscCall(MatRegister(MATIS,             MatCreate_IS));
  PetscCall(MatRegister(MATSHELL,          MatCreate_Shell));
  PetscCall(MatRegister(MATCOMPOSITE,      MatCreate_Composite));

  PetscCall(MatRegisterRootName(MATAIJ,MATSEQAIJ,MATMPIAIJ));
  PetscCall(MatRegister(MATMPIAIJ,         MatCreate_MPIAIJ));
  PetscCall(MatRegister(MATSEQAIJ,         MatCreate_SeqAIJ));

  PetscCall(MatRegisterRootName(MATAIJPERM,MATSEQAIJPERM,MATMPIAIJPERM));
  PetscCall(MatRegister(MATMPIAIJPERM,     MatCreate_MPIAIJPERM));
  PetscCall(MatRegister(MATSEQAIJPERM,     MatCreate_SeqAIJPERM));

  PetscCall(MatRegisterRootName(MATAIJSELL,MATSEQAIJSELL,MATMPIAIJSELL));
  PetscCall(MatRegister(MATMPIAIJSELL,     MatCreate_MPIAIJSELL));
  PetscCall(MatRegister(MATSEQAIJSELL,     MatCreate_SeqAIJSELL));

#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(MatRegisterRootName(MATAIJMKL, MATSEQAIJMKL,MATMPIAIJMKL));
  PetscCall(MatRegister(MATMPIAIJMKL,      MatCreate_MPIAIJMKL));
  PetscCall(MatRegister(MATSEQAIJMKL,      MatCreate_SeqAIJMKL));
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  PetscCall(MatRegisterRootName(MATBAIJMKL,MATSEQBAIJMKL,MATMPIBAIJMKL));
  PetscCall(MatRegister(MATMPIBAIJMKL,      MatCreate_MPIBAIJMKL));
  PetscCall(MatRegister(MATSEQBAIJMKL,      MatCreate_SeqBAIJMKL));
#endif

  PetscCall(MatRegisterRootName(MATAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL));
  PetscCall(MatRegister(MATSEQAIJCRL,      MatCreate_SeqAIJCRL));
  PetscCall(MatRegister(MATMPIAIJCRL,      MatCreate_MPIAIJCRL));

  PetscCall(MatRegisterRootName(MATBAIJ,MATSEQBAIJ,MATMPIBAIJ));
  PetscCall(MatRegister(MATMPIBAIJ,        MatCreate_MPIBAIJ));
  PetscCall(MatRegister(MATSEQBAIJ,        MatCreate_SeqBAIJ));

  PetscCall(MatRegisterRootName(MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ));
  PetscCall(MatRegister(MATMPISBAIJ,       MatCreate_MPISBAIJ));
  PetscCall(MatRegister(MATSEQSBAIJ,       MatCreate_SeqSBAIJ));

  PetscCall(MatRegisterRootName(MATDENSE,MATSEQDENSE,MATMPIDENSE));
  PetscCall(MatRegister(MATMPIDENSE,       MatCreate_MPIDense));
  PetscCall(MatRegister(MATSEQDENSE,       MatCreate_SeqDense));
#if defined(PETSC_HAVE_CUDA)
  PetscCall(MatRegisterRootName(MATDENSECUDA,MATSEQDENSECUDA,MATMPIDENSECUDA));
  PetscCall(MatRegister(MATSEQDENSECUDA,   MatCreate_SeqDenseCUDA));
  PetscCall(MatRegister(MATMPIDENSECUDA,   MatCreate_MPIDenseCUDA));
#endif

  PetscCall(MatRegister(MATMPIADJ,         MatCreate_MPIAdj));
  PetscCall(MatRegister(MATSCATTER,        MatCreate_Scatter));
  PetscCall(MatRegister(MATBLOCKMAT,       MatCreate_BlockMat));
  PetscCall(MatRegister(MATNEST,           MatCreate_Nest));

  PetscCall(MatRegisterRootName(MATSELL,MATSEQSELL,MATMPISELL));
  PetscCall(MatRegister(MATMPISELL,         MatCreate_MPISELL));
  PetscCall(MatRegister(MATSEQSELL,         MatCreate_SeqSELL));

#if defined(PETSC_HAVE_CUDA)
  PetscCall(MatRegisterRootName(MATAIJCUSPARSE,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE));
  PetscCall(MatRegister(MATSEQAIJCUSPARSE, MatCreate_SeqAIJCUSPARSE));
  PetscCall(MatRegister(MATMPIAIJCUSPARSE, MatCreate_MPIAIJCUSPARSE));
#endif

#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(MatRegisterRootName(MATAIJVIENNACL,MATSEQAIJVIENNACL,MATMPIAIJVIENNACL));
  PetscCall(MatRegister(MATSEQAIJVIENNACL, MatCreate_SeqAIJViennaCL));
  PetscCall(MatRegister(MATMPIAIJVIENNACL, MatCreate_MPIAIJViennaCL));
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(MatRegisterRootName(MATAIJKOKKOS,MATSEQAIJKOKKOS,MATMPIAIJKOKKOS));
  PetscCall(MatRegister(MATSEQAIJKOKKOS,   MatCreate_SeqAIJKokkos));
  PetscCall(MatRegister(MATMPIAIJKOKKOS,   MatCreate_MPIAIJKokkos));
#endif

#if defined(PETSC_HAVE_FFTW)
  PetscCall(MatRegister(MATFFTW,           MatCreate_FFTW));
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(MatRegister(MATELEMENTAL,      MatCreate_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(MatRegister(MATSCALAPACK,      MatCreate_ScaLAPACK));
#endif

  PetscCall(MatRegister(MATPREALLOCATOR,   MatCreate_Preallocator));
  PetscCall(MatRegister(MATDUMMY,          MatCreate_Dummy));

  PetscCall(MatRegister(MATCONSTANTDIAGONAL,MatCreate_ConstantDiagonal));

#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatRegister(MATHYPRE,          MatCreate_HYPRE));
#endif

#if defined(PETSC_HAVE_H2OPUS)
  PetscCall(MatRegister(MATH2OPUS,         MatCreate_H2OPUS));
#endif

#if defined(PETSC_HAVE_HTOOL)
  PetscCall(MatRegister(MATHTOOL,          MatCreate_Htool));
#endif
  PetscFunctionReturn(0);
}
