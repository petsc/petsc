
#if !defined(__BAIJ_H)
#define __BAIJ_H
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/baij/seq/ftn-kernels/fsolvebaij.h>

/*
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[]
  arrays start at 0.
*/

/* This header is shared by the SeqSBAIJ matrix */
#define SEQBAIJHEADER \
  PetscInt    bs2;                      /*  square of block size */                                  \
  PetscInt    mbs,nbs;               /* rows/bs, columns/bs */                                       \
  PetscScalar *mult_work;            /* work array for matrix vector product*/                       \
  PetscScalar *sor_workt;            /* work array for SOR */                                        \
  PetscScalar *sor_work;             /* work array for SOR */                                        \
  MatScalar   *saved_values;                                                                    \
                                                                                                     \
  Mat         sbaijMat;                      /* mat in sbaij format */                                       \
                                                                                                     \
                                                                                                     \
  MatScalar     *idiag;            /* inverse of block diagonal  */                                \
  PetscBool     idiagvalid         /* if above has correct/current values */

typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQBAIJHEADER;
} Mat_SeqBAIJ;

PETSC_INTERN PetscErrorCode MatSeqBAIJSetPreallocation_SeqBAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz);
PETSC_INTERN PetscErrorCode MatAXPY_SeqBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str);

PETSC_INTERN PetscErrorCode MatGetColumnIJ_SeqBAIJ(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscBool*);
PETSC_INTERN PetscErrorCode MatRestoreColumnIJ_SeqBAIJ(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscBool*);
PETSC_INTERN PetscErrorCode MatGetColumnIJ_SeqBAIJ_Color(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscInt *[],PetscBool*);
PETSC_INTERN PetscErrorCode MatRestoreColumnIJ_SeqBAIJ_Color(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscInt *[],PetscBool*);

PETSC_INTERN PetscErrorCode MatILUFactorSymbolic_SeqBAIJ_inplace(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatILUFactorSymbolic_SeqBAIJ(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatICCFactorSymbolic_SeqBAIJ(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqBAIJ(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatDuplicate_SeqBAIJ(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatMissingDiagonal_SeqBAIJ(Mat,PetscBool*,PetscInt*);
PETSC_INTERN PetscErrorCode MatMarkDiagonal_SeqBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatILUDTFactor_SeqBAIJ(Mat,IS,IS,const MatFactorInfo*,Mat*);

PETSC_INTERN PetscErrorCode MatLUFactorSymbolic_SeqBAIJ_inplace(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorSymbolic_SeqBAIJ(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactor_SeqBAIJ(Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_SeqBAIJ(Mat,PetscInt,IS*,PetscInt);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_SeqBAIJ(Mat,IS,IS,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_SeqBAIJ(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqBAIJ(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultHermitianTranspose_SeqBAIJ(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqBAIJ(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultHermitianTransposeAdd_SeqBAIJ(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatScale_SeqBAIJ(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatNorm_SeqBAIJ(Mat,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode MatEqual_SeqBAIJ(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatFDColoringApply_BAIJ(Mat,MatFDColoring,Vec,void*);
PETSC_INTERN PetscErrorCode MatGetDiagonal_SeqBAIJ(Mat,Vec);
PETSC_INTERN PetscErrorCode MatDiagonalScale_SeqBAIJ(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatGetInfo_SeqBAIJ(Mat,MatInfoType,MatInfo*);
PETSC_INTERN PetscErrorCode MatZeroEntries_SeqBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SeqBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatAssemblyEnd_SeqBAIJ(Mat,MatAssemblyType);

PETSC_INTERN PetscErrorCode MatSeqBAIJ_UpdateFactorNumeric_NaturalOrdering(Mat);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_1_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_2_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_3_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_3(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
#if defined(PETSC_HAVE_SSE)
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4_SSE_Demotion(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion_usj(Mat,Vec,Vec);
#endif
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_5_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_5(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_6_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_6(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_7_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_7(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_9_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_11_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_12_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_13_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_14_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_15_NaturalOrdering_ver1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_15_NaturalOrdering_ver2(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_N_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_N(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqBAIJ_N_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_N_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_N(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
#if defined(PETSC_HAVE_SSE)
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj(Mat,Mat,const MatFactorInfo*);
#else
#endif
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_9_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_15_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N_inplace(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_3(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_4(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_5(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_6(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_7(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_9_AVX2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_11(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_12_ver1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_12_ver2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_12_AVX2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_12_ver1(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_12_ver2(Mat,Vec,Vec,Vec);

PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_15_ver1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_15_ver2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_15_ver3(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_15_ver4(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatMult_SeqBAIJ_N(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_1(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_2(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_3(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_4(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_5(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_6(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_7(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_9_AVX2(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_11(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqBAIJ_N(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSeqBAIJSetNumericFactorization_inplace(Mat,PetscBool);
PETSC_INTERN PetscErrorCode MatSeqBAIJSetNumericFactorization(Mat,PetscBool);

PETSC_INTERN PetscErrorCode MatGetRow_SeqBAIJ_private(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**,PetscInt*,PetscInt*,PetscScalar*);
PETSC_INTERN PetscErrorCode MatAXPYGetPreallocation_SeqBAIJ(Mat,Mat,PetscInt*);

PETSC_INTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqBAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPIBAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);

PETSC_INTERN PetscErrorCode MatView_SeqBAIJ(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_SeqBAIJ(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatView_SeqBAIJ_Binary(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_SeqBAIJ_Binary(Mat,PetscViewer);

/* used by mpibaij.c */
PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPIBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDisAssemble_MPIBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatGetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt [],PetscScalar []);
PETSC_INTERN PetscErrorCode MatSetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
PETSC_INTERN PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatGetRow_SeqBAIJ(Mat,PetscInt,PetscInt*,PetscInt*[],PetscScalar*[]);
PETSC_INTERN PetscErrorCode MatRestoreRow_SeqBAIJ(Mat,PetscInt,PetscInt*,PetscInt*[],PetscScalar*[]);
PETSC_INTERN PetscErrorCode MatZeroRows_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscScalar,Vec,Vec);

PETSC_INTERN PetscErrorCode MatDestroySubMatrix_SeqBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDestroySubMatrices_SeqBAIJ(PetscInt,Mat*[]);

/*
  PetscKernel_A_gets_A_times_B_2: A = A * B with size bs=2

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_2(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,4);CHKERRQ(ierr);
  A[0] = W[0]*B[0] + W[2]*B[1];
  A[1] = W[1]*B[0] + W[3]*B[1];
  A[2] = W[0]*B[2] + W[2]*B[3];
  A[3] = W[1]*B[2] + W[3]*B[3];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_2: A = A - B * C with size bs=2

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_2(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0] -= B[0]*C[0] + B[2]*C[1];
  A[1] -= B[1]*C[0] + B[3]*C[1];
  A[2] -= B[0]*C[2] + B[2]*C[3];
  A[3] -= B[1]*C[2] + B[3]*C[3];
  return 0;
}

/*
  PetscKernel_A_gets_A_times_B_3: A = A * B with size bs=3

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_3(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,9);CHKERRQ(ierr);
  A[0] = W[0]*B[0] + W[3]*B[1] + W[6]*B[2];
  A[1] = W[1]*B[0] + W[4]*B[1] + W[7]*B[2];
  A[2] = W[2]*B[0] + W[5]*B[1] + W[8]*B[2];
  A[3] = W[0]*B[3] + W[3]*B[4] + W[6]*B[5];
  A[4] = W[1]*B[3] + W[4]*B[4] + W[7]*B[5];
  A[5] = W[2]*B[3] + W[5]*B[4] + W[8]*B[5];
  A[6] = W[0]*B[6] + W[3]*B[7] + W[6]*B[8];
  A[7] = W[1]*B[6] + W[4]*B[7] + W[7]*B[8];
  A[8] = W[2]*B[6] + W[5]*B[7] + W[8]*B[8];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_3: A = A - B * C with size bs=3

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_3(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0] -= B[0]*C[0] + B[3]*C[1] + B[6]*C[2];
  A[1] -= B[1]*C[0] + B[4]*C[1] + B[7]*C[2];
  A[2] -= B[2]*C[0] + B[5]*C[1] + B[8]*C[2];
  A[3] -= B[0]*C[3] + B[3]*C[4] + B[6]*C[5];
  A[4] -= B[1]*C[3] + B[4]*C[4] + B[7]*C[5];
  A[5] -= B[2]*C[3] + B[5]*C[4] + B[8]*C[5];
  A[6] -= B[0]*C[6] + B[3]*C[7] + B[6]*C[8];
  A[7] -= B[1]*C[6] + B[4]*C[7] + B[7]*C[8];
  A[8] -= B[2]*C[6] + B[5]*C[7] + B[8]*C[8];
  return 0;
}

/*
  PetscKernel_A_gets_A_times_B_4: A = A * B with size bs=4

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_4(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,16);CHKERRQ(ierr);
  A[0]  =  W[0]*B[0]  + W[4]*B[1]  + W[8]*B[2]   + W[12]*B[3];
  A[1]  =  W[1]*B[0]  + W[5]*B[1]  + W[9]*B[2]   + W[13]*B[3];
  A[2]  =  W[2]*B[0]  + W[6]*B[1]  + W[10]*B[2]  + W[14]*B[3];
  A[3]  =  W[3]*B[0]  + W[7]*B[1]  + W[11]*B[2]  + W[15]*B[3];
  A[4]  =  W[0]*B[4]  + W[4]*B[5]  + W[8]*B[6]   + W[12]*B[7];
  A[5]  =  W[1]*B[4]  + W[5]*B[5]  + W[9]*B[6]   + W[13]*B[7];
  A[6]  =  W[2]*B[4]  + W[6]*B[5]  + W[10]*B[6]  + W[14]*B[7];
  A[7]  =  W[3]*B[4]  + W[7]*B[5]  + W[11]*B[6]  + W[15]*B[7];
  A[8]  =  W[0]*B[8]  + W[4]*B[9]  + W[8]*B[10]  + W[12]*B[11];
  A[9]  =  W[1]*B[8]  + W[5]*B[9]  + W[9]*B[10]  + W[13]*B[11];
  A[10] = W[2]*B[8]  + W[6]*B[9]  + W[10]*B[10] + W[14]*B[11];
  A[11] = W[3]*B[8]  + W[7]*B[9]  + W[11]*B[10] + W[15]*B[11];
  A[12] = W[0]*B[12] + W[4]*B[13] + W[8]*B[14]  + W[12]*B[15];
  A[13] = W[1]*B[12] + W[5]*B[13] + W[9]*B[14]  + W[13]*B[15];
  A[14] = W[2]*B[12] + W[6]*B[13] + W[10]*B[14] + W[14]*B[15];
  A[15] = W[3]*B[12] + W[7]*B[13] + W[11]*B[14] + W[15]*B[15];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_4: A = A - B * C with size bs=4

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_4(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0]  -=  B[0]*C[0]  + B[4]*C[1]  + B[8]*C[2]   + B[12]*C[3];
  A[1]  -=  B[1]*C[0]  + B[5]*C[1]  + B[9]*C[2]   + B[13]*C[3];
  A[2]  -=  B[2]*C[0]  + B[6]*C[1]  + B[10]*C[2]  + B[14]*C[3];
  A[3]  -=  B[3]*C[0]  + B[7]*C[1]  + B[11]*C[2]  + B[15]*C[3];
  A[4]  -=  B[0]*C[4]  + B[4]*C[5]  + B[8]*C[6]   + B[12]*C[7];
  A[5]  -=  B[1]*C[4]  + B[5]*C[5]  + B[9]*C[6]   + B[13]*C[7];
  A[6]  -=  B[2]*C[4]  + B[6]*C[5]  + B[10]*C[6]  + B[14]*C[7];
  A[7]  -=  B[3]*C[4]  + B[7]*C[5]  + B[11]*C[6]  + B[15]*C[7];
  A[8]  -=  B[0]*C[8]  + B[4]*C[9]  + B[8]*C[10]  + B[12]*C[11];
  A[9]  -=  B[1]*C[8]  + B[5]*C[9]  + B[9]*C[10]  + B[13]*C[11];
  A[10] -= B[2]*C[8]  + B[6]*C[9]  + B[10]*C[10] + B[14]*C[11];
  A[11] -= B[3]*C[8]  + B[7]*C[9]  + B[11]*C[10] + B[15]*C[11];
  A[12] -= B[0]*C[12] + B[4]*C[13] + B[8]*C[14]  + B[12]*C[15];
  A[13] -= B[1]*C[12] + B[5]*C[13] + B[9]*C[14]  + B[13]*C[15];
  A[14] -= B[2]*C[12] + B[6]*C[13] + B[10]*C[14] + B[14]*C[15];
  A[15] -= B[3]*C[12] + B[7]*C[13] + B[11]*C[14] + B[15]*C[15];
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_5(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,25);CHKERRQ(ierr);
  A[0]  =  W[0]*B[0]  + W[5]*B[1]  + W[10]*B[2]   + W[15]*B[3] + W[20]*B[4];
  A[1]  =  W[1]*B[0]  + W[6]*B[1]  + W[11]*B[2]   + W[16]*B[3] + W[21]*B[4];
  A[2]  =  W[2]*B[0]  + W[7]*B[1]  + W[12]*B[2]  + W[17]*B[3]  + W[22]*B[4];
  A[3]  =  W[3]*B[0]  + W[8]*B[1]  + W[13]*B[2]  + W[18]*B[3]  + W[23]*B[4];
  A[4]  =  W[4]*B[0]  + W[9]*B[1]  + W[14]*B[2]   + W[19]*B[3] + W[24]*B[4];
  A[5]  =  W[0]*B[5]  + W[5]*B[6]  + W[10]*B[7]   + W[15]*B[8] + W[20]*B[9];
  A[6]  =  W[1]*B[5]  + W[6]*B[6]  + W[11]*B[7]   + W[16]*B[8] + W[21]*B[9];
  A[7]  =  W[2]*B[5]  + W[7]*B[6]  + W[12]*B[7]  + W[17]*B[8]  + W[22]*B[9];
  A[8]  =  W[3]*B[5]  + W[8]*B[6]  + W[13]*B[7]  + W[18]*B[8]  + W[23]*B[9];
  A[9]  =  W[4]*B[5]  + W[9]*B[6]  + W[14]*B[7]   + W[19]*B[8] + W[24]*B[9];
  A[10] =  W[0]*B[10]  + W[5]*B[11]  + W[10]*B[12]   + W[15]*B[13] + W[20]*B[14];
  A[11] =  W[1]*B[10]  + W[6]*B[11]  + W[11]*B[12]   + W[16]*B[13] + W[21]*B[14];
  A[12] =  W[2]*B[10]  + W[7]*B[11]  + W[12]*B[12]  + W[17]*B[13]  + W[22]*B[14];
  A[13] =  W[3]*B[10]  + W[8]*B[11]  + W[13]*B[12]  + W[18]*B[13]  + W[23]*B[14];
  A[14] =  W[4]*B[10]  + W[9]*B[11]  + W[14]*B[12]   + W[19]*B[13] + W[24]*B[14];
  A[15] =  W[0]*B[15]  + W[5]*B[16]  + W[10]*B[17]   + W[15]*B[18] + W[20]*B[19];
  A[16] =  W[1]*B[15]  + W[6]*B[16]  + W[11]*B[17]   + W[16]*B[18] + W[21]*B[19];
  A[17] =  W[2]*B[15]  + W[7]*B[16]  + W[12]*B[17]  + W[17]*B[18]  + W[22]*B[19];
  A[18] =  W[3]*B[15]  + W[8]*B[16]  + W[13]*B[17]  + W[18]*B[18]  + W[23]*B[19];
  A[19] =  W[4]*B[15]  + W[9]*B[16]  + W[14]*B[17]   + W[19]*B[18] + W[24]*B[19];
  A[20] =  W[0]*B[20]  + W[5]*B[21]  + W[10]*B[22]   + W[15]*B[23] + W[20]*B[24];
  A[21] =  W[1]*B[20]  + W[6]*B[21]  + W[11]*B[22]   + W[16]*B[23] + W[21]*B[24];
  A[22] =  W[2]*B[20]  + W[7]*B[21]  + W[12]*B[22]  + W[17]*B[23]  + W[22]*B[24];
  A[23] =  W[3]*B[20]  + W[8]*B[21]  + W[13]*B[22]  + W[18]*B[23]  + W[23]*B[24];
  A[24] =  W[4]*B[20]  + W[9]*B[21]  + W[14]*B[22]   + W[19]*B[23] + W[24]*B[24];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_5: A = A - B * C with size bs=5

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_5(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0]  -=  B[0]*C[0]  + B[5]*C[1]  + B[10]*C[2]   + B[15]*C[3] + B[20]*C[4];
  A[1]  -=  B[1]*C[0]  + B[6]*C[1]  + B[11]*C[2]   + B[16]*C[3] + B[21]*C[4];
  A[2]  -=  B[2]*C[0]  + B[7]*C[1]  + B[12]*C[2]  + B[17]*C[3]  + B[22]*C[4];
  A[3]  -=  B[3]*C[0]  + B[8]*C[1]  + B[13]*C[2]  + B[18]*C[3]  + B[23]*C[4];
  A[4]  -=  B[4]*C[0]  + B[9]*C[1]  + B[14]*C[2]   + B[19]*C[3] + B[24]*C[4];
  A[5]  -=  B[0]*C[5]  + B[5]*C[6]  + B[10]*C[7]   + B[15]*C[8] + B[20]*C[9];
  A[6]  -=  B[1]*C[5]  + B[6]*C[6]  + B[11]*C[7]   + B[16]*C[8] + B[21]*C[9];
  A[7]  -=  B[2]*C[5]  + B[7]*C[6]  + B[12]*C[7]  + B[17]*C[8]  + B[22]*C[9];
  A[8]  -=  B[3]*C[5]  + B[8]*C[6]  + B[13]*C[7]  + B[18]*C[8]  + B[23]*C[9];
  A[9]  -=  B[4]*C[5]  + B[9]*C[6]  + B[14]*C[7]   + B[19]*C[8] + B[24]*C[9];
  A[10] -=  B[0]*C[10]  + B[5]*C[11]  + B[10]*C[12]   + B[15]*C[13] + B[20]*C[14];
  A[11] -=  B[1]*C[10]  + B[6]*C[11]  + B[11]*C[12]   + B[16]*C[13] + B[21]*C[14];
  A[12] -=  B[2]*C[10]  + B[7]*C[11]  + B[12]*C[12]  + B[17]*C[13]  + B[22]*C[14];
  A[13] -=  B[3]*C[10]  + B[8]*C[11]  + B[13]*C[12]  + B[18]*C[13]  + B[23]*C[14];
  A[14] -=  B[4]*C[10]  + B[9]*C[11]  + B[14]*C[12]   + B[19]*C[13] + B[24]*C[14];
  A[15] -=  B[0]*C[15]  + B[5]*C[16]  + B[10]*C[17]   + B[15]*C[18] + B[20]*C[19];
  A[16] -=  B[1]*C[15]  + B[6]*C[16]  + B[11]*C[17]   + B[16]*C[18] + B[21]*C[19];
  A[17] -=  B[2]*C[15]  + B[7]*C[16]  + B[12]*C[17]  + B[17]*C[18]  + B[22]*C[19];
  A[18] -=  B[3]*C[15]  + B[8]*C[16]  + B[13]*C[17]  + B[18]*C[18]  + B[23]*C[19];
  A[19] -=  B[4]*C[15]  + B[9]*C[16]  + B[14]*C[17]   + B[19]*C[18] + B[24]*C[19];
  A[20] -=  B[0]*C[20]  + B[5]*C[21]  + B[10]*C[22]   + B[15]*C[23] + B[20]*C[24];
  A[21] -=  B[1]*C[20]  + B[6]*C[21]  + B[11]*C[22]   + B[16]*C[23] + B[21]*C[24];
  A[22] -=  B[2]*C[20]  + B[7]*C[21]  + B[12]*C[22]  + B[17]*C[23]  + B[22]*C[24];
  A[23] -=  B[3]*C[20]  + B[8]*C[21]  + B[13]*C[22]  + B[18]*C[23]  + B[23]*C[24];
  A[24] -=  B[4]*C[20]  + B[9]*C[21]  + B[14]*C[22]   + B[19]*C[23] + B[24]*C[24];
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_6(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,36);CHKERRQ(ierr);
  A[0]  =  W[0]*B[0]   + W[6]*B[1]   + W[12]*B[2]   + W[18]*B[3]  + W[24]*B[4]  + W[30]*B[5];
  A[1]  =  W[1]*B[0]   + W[7]*B[1]   + W[13]*B[2]   + W[19]*B[3]  + W[25]*B[4]  + W[31]*B[5];
  A[2]  =  W[2]*B[0]   + W[8]*B[1]   + W[14]*B[2]   + W[20]*B[3]  + W[26]*B[4]  + W[32]*B[5];
  A[3]  =  W[3]*B[0]   + W[9]*B[1]   + W[15]*B[2]   + W[21]*B[3]  + W[27]*B[4]  + W[33]*B[5];
  A[4]  =  W[4]*B[0]   + W[10]*B[1]  + W[16]*B[2]   + W[22]*B[3]  + W[28]*B[4]  + W[34]*B[5];
  A[5]  =  W[5]*B[0]   + W[11]*B[1]  + W[17]*B[2]   + W[23]*B[3]  + W[29]*B[4]  + W[35]*B[5];
  A[6]  =  W[0]*B[6]   + W[6]*B[7]   + W[12]*B[8]   + W[18]*B[9]  + W[24]*B[10] + W[30]*B[11];
  A[7]  =  W[1]*B[6]   + W[7]*B[7]   + W[13]*B[8]   + W[19]*B[9]  + W[25]*B[10] + W[31]*B[11];
  A[8]  =  W[2]*B[6]   + W[8]*B[7]   + W[14]*B[8]   + W[20]*B[9]  + W[26]*B[10] + W[32]*B[11];
  A[9]  =  W[3]*B[6]   + W[9]*B[7]   + W[15]*B[8]   + W[21]*B[9]  + W[27]*B[10] + W[33]*B[11];
  A[10] =  W[4]*B[6]   + W[10]*B[7]  + W[16]*B[8]   + W[22]*B[9]  + W[28]*B[10] + W[34]*B[11];
  A[11] =  W[5]*B[6]   + W[11]*B[7]  + W[17]*B[8]   + W[23]*B[9]  + W[29]*B[10] + W[35]*B[11];
  A[12] =  W[0]*B[12]  + W[6]*B[13]  + W[12]*B[14]  + W[18]*B[15] + W[24]*B[16] + W[30]*B[17];
  A[13] =  W[1]*B[12]  + W[7]*B[13]  + W[13]*B[14]  + W[19]*B[15] + W[25]*B[16] + W[31]*B[17];
  A[14] =  W[2]*B[12]  + W[8]*B[13]  + W[14]*B[14]  + W[20]*B[15] + W[26]*B[16] + W[32]*B[17];
  A[15] =  W[3]*B[12]  + W[9]*B[13]  + W[15]*B[14]  + W[21]*B[15] + W[27]*B[16] + W[33]*B[17];
  A[16] =  W[4]*B[12]  + W[10]*B[13] + W[16]*B[14]  + W[22]*B[15] + W[28]*B[16] + W[34]*B[17];
  A[17] =  W[5]*B[12]  + W[11]*B[13] + W[17]*B[14]  + W[23]*B[15] + W[29]*B[16] + W[35]*B[17];
  A[18] =  W[0]*B[18]  + W[6]*B[19]  + W[12]*B[20]  + W[18]*B[21] + W[24]*B[22] + W[30]*B[23];
  A[19] =  W[1]*B[18]  + W[7]*B[19]  + W[13]*B[20]  + W[19]*B[21] + W[25]*B[22] + W[31]*B[23];
  A[20] =  W[2]*B[18]  + W[8]*B[19]  + W[14]*B[20]  + W[20]*B[21] + W[26]*B[22] + W[32]*B[23];
  A[21] =  W[3]*B[18]  + W[9]*B[19]  + W[15]*B[20]  + W[21]*B[21] + W[27]*B[22] + W[33]*B[23];
  A[22] =  W[4]*B[18]  + W[10]*B[19] + W[16]*B[20]  + W[22]*B[21] + W[28]*B[22] + W[34]*B[23];
  A[23] =  W[5]*B[18]  + W[11]*B[19] + W[17]*B[20]  + W[23]*B[21] + W[29]*B[22] + W[35]*B[23];
  A[24] =  W[0]*B[24]  + W[6]*B[25]  + W[12]*B[26]  + W[18]*B[27] + W[24]*B[28] + W[30]*B[29];
  A[25] =  W[1]*B[24]  + W[7]*B[25]  + W[13]*B[26]  + W[19]*B[27] + W[25]*B[28] + W[31]*B[29];
  A[26] =  W[2]*B[24]  + W[8]*B[25]  + W[14]*B[26]  + W[20]*B[27] + W[26]*B[28] + W[32]*B[29];
  A[27] =  W[3]*B[24]  + W[9]*B[25]  + W[15]*B[26]  + W[21]*B[27] + W[27]*B[28] + W[33]*B[29];
  A[28] =  W[4]*B[24]  + W[10]*B[25] + W[16]*B[26]  + W[22]*B[27] + W[28]*B[28] + W[34]*B[29];
  A[29] =  W[5]*B[24]  + W[11]*B[25] + W[17]*B[26]  + W[23]*B[27] + W[29]*B[28] + W[35]*B[29];
  A[30] =  W[0]*B[30]  + W[6]*B[31]  + W[12]*B[32]  + W[18]*B[33] + W[24]*B[34] + W[30]*B[35];
  A[31] =  W[1]*B[30]  + W[7]*B[31]  + W[13]*B[32]  + W[19]*B[33] + W[25]*B[34] + W[31]*B[35];
  A[32] =  W[2]*B[30]  + W[8]*B[31]  + W[14]*B[32]  + W[20]*B[33] + W[26]*B[34] + W[32]*B[35];
  A[33] =  W[3]*B[30]  + W[9]*B[31]  + W[15]*B[32]  + W[21]*B[33] + W[27]*B[34] + W[33]*B[35];
  A[34] =  W[4]*B[30]  + W[10]*B[31] + W[16]*B[32]  + W[22]*B[33] + W[28]*B[34] + W[34]*B[35];
  A[35] =  W[5]*B[30]  + W[11]*B[31] + W[17]*B[32]  + W[23]*B[33] + W[29]*B[34] + W[35]*B[35];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_6: A = A - B * C with size bs=6

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_6(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0]  -=  B[0]*C[0]   + B[6]*C[1]   + B[12]*C[2]   + B[18]*C[3]  + B[24]*C[4]  + B[30]*C[5];
  A[1]  -=  B[1]*C[0]   + B[7]*C[1]   + B[13]*C[2]   + B[19]*C[3]  + B[25]*C[4]  + B[31]*C[5];
  A[2]  -=  B[2]*C[0]   + B[8]*C[1]   + B[14]*C[2]   + B[20]*C[3]  + B[26]*C[4]  + B[32]*C[5];
  A[3]  -=  B[3]*C[0]   + B[9]*C[1]   + B[15]*C[2]   + B[21]*C[3]  + B[27]*C[4]  + B[33]*C[5];
  A[4]  -=  B[4]*C[0]   + B[10]*C[1]  + B[16]*C[2]   + B[22]*C[3]  + B[28]*C[4]  + B[34]*C[5];
  A[5]  -=  B[5]*C[0]   + B[11]*C[1]  + B[17]*C[2]   + B[23]*C[3]  + B[29]*C[4]  + B[35]*C[5];
  A[6]  -=  B[0]*C[6]   + B[6]*C[7]   + B[12]*C[8]   + B[18]*C[9]  + B[24]*C[10] + B[30]*C[11];
  A[7]  -=  B[1]*C[6]   + B[7]*C[7]   + B[13]*C[8]   + B[19]*C[9]  + B[25]*C[10] + B[31]*C[11];
  A[8]  -=  B[2]*C[6]   + B[8]*C[7]   + B[14]*C[8]   + B[20]*C[9]  + B[26]*C[10] + B[32]*C[11];
  A[9]  -=  B[3]*C[6]   + B[9]*C[7]   + B[15]*C[8]   + B[21]*C[9]  + B[27]*C[10] + B[33]*C[11];
  A[10] -=  B[4]*C[6]   + B[10]*C[7]  + B[16]*C[8]   + B[22]*C[9]  + B[28]*C[10] + B[34]*C[11];
  A[11] -=  B[5]*C[6]   + B[11]*C[7]  + B[17]*C[8]   + B[23]*C[9]  + B[29]*C[10] + B[35]*C[11];
  A[12] -=  B[0]*C[12]  + B[6]*C[13]  + B[12]*C[14]  + B[18]*C[15] + B[24]*C[16] + B[30]*C[17];
  A[13] -=  B[1]*C[12]  + B[7]*C[13]  + B[13]*C[14]  + B[19]*C[15] + B[25]*C[16] + B[31]*C[17];
  A[14] -=  B[2]*C[12]  + B[8]*C[13]  + B[14]*C[14]  + B[20]*C[15] + B[26]*C[16] + B[32]*C[17];
  A[15] -=  B[3]*C[12]  + B[9]*C[13]  + B[15]*C[14]  + B[21]*C[15] + B[27]*C[16] + B[33]*C[17];
  A[16] -=  B[4]*C[12]  + B[10]*C[13] + B[16]*C[14]  + B[22]*C[15] + B[28]*C[16] + B[34]*C[17];
  A[17] -=  B[5]*C[12]  + B[11]*C[13] + B[17]*C[14]  + B[23]*C[15] + B[29]*C[16] + B[35]*C[17];
  A[18] -=  B[0]*C[18]  + B[6]*C[19]  + B[12]*C[20]  + B[18]*C[21] + B[24]*C[22] + B[30]*C[23];
  A[19] -=  B[1]*C[18]  + B[7]*C[19]  + B[13]*C[20]  + B[19]*C[21] + B[25]*C[22] + B[31]*C[23];
  A[20] -=  B[2]*C[18]  + B[8]*C[19]  + B[14]*C[20]  + B[20]*C[21] + B[26]*C[22] + B[32]*C[23];
  A[21] -=  B[3]*C[18]  + B[9]*C[19]  + B[15]*C[20]  + B[21]*C[21] + B[27]*C[22] + B[33]*C[23];
  A[22] -=  B[4]*C[18]  + B[10]*C[19] + B[16]*C[20]  + B[22]*C[21] + B[28]*C[22] + B[34]*C[23];
  A[23] -=  B[5]*C[18]  + B[11]*C[19] + B[17]*C[20]  + B[23]*C[21] + B[29]*C[22] + B[35]*C[23];
  A[24] -=  B[0]*C[24]  + B[6]*C[25]  + B[12]*C[26]  + B[18]*C[27] + B[24]*C[28] + B[30]*C[29];
  A[25] -=  B[1]*C[24]  + B[7]*C[25]  + B[13]*C[26]  + B[19]*C[27] + B[25]*C[28] + B[31]*C[29];
  A[26] -=  B[2]*C[24]  + B[8]*C[25]  + B[14]*C[26]  + B[20]*C[27] + B[26]*C[28] + B[32]*C[29];
  A[27] -=  B[3]*C[24]  + B[9]*C[25]  + B[15]*C[26]  + B[21]*C[27] + B[27]*C[28] + B[33]*C[29];
  A[28] -=  B[4]*C[24]  + B[10]*C[25] + B[16]*C[26]  + B[22]*C[27] + B[28]*C[28] + B[34]*C[29];
  A[29] -=  B[5]*C[24]  + B[11]*C[25] + B[17]*C[26]  + B[23]*C[27] + B[29]*C[28] + B[35]*C[29];
  A[30] -=  B[0]*C[30]  + B[6]*C[31]  + B[12]*C[32]  + B[18]*C[33] + B[24]*C[34] + B[30]*C[35];
  A[31] -=  B[1]*C[30]  + B[7]*C[31]  + B[13]*C[32]  + B[19]*C[33] + B[25]*C[34] + B[31]*C[35];
  A[32] -=  B[2]*C[30]  + B[8]*C[31]  + B[14]*C[32]  + B[20]*C[33] + B[26]*C[34] + B[32]*C[35];
  A[33] -=  B[3]*C[30]  + B[9]*C[31]  + B[15]*C[32]  + B[21]*C[33] + B[27]*C[34] + B[33]*C[35];
  A[34] -=  B[4]*C[30]  + B[10]*C[31] + B[16]*C[32]  + B[22]*C[33] + B[28]*C[34] + B[34]*C[35];
  A[35] -=  B[5]*C[30]  + B[11]*C[31] + B[17]*C[32]  + B[23]*C[33] + B[29]*C[34] + B[35]*C[35];
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_7(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,49);CHKERRQ(ierr);
  A[0]  =  W[0]*B[0]   + W[7]*B[1]   + W[14]*B[2]   + W[21]*B[3]  + W[28]*B[4]  + W[35]*B[5]  + W[42]*B[6];
  A[1]  =  W[1]*B[0]   + W[8]*B[1]   + W[15]*B[2]   + W[22]*B[3]  + W[29]*B[4]  + W[36]*B[5]  + W[43]*B[6];
  A[2]  =  W[2]*B[0]   + W[9]*B[1]   + W[16]*B[2]   + W[23]*B[3]  + W[30]*B[4]  + W[37]*B[5]  + W[44]*B[6];
  A[3]  =  W[3]*B[0]   + W[10]*B[1]   + W[17]*B[2]   + W[24]*B[3]  + W[31]*B[4]  + W[38]*B[5]  + W[45]*B[6];
  A[4]  =  W[4]*B[0]   + W[11]*B[1]  + W[18]*B[2]   + W[25]*B[3]  + W[32]*B[4]  + W[39]*B[5]  + W[46]*B[6];
  A[5]  =  W[5]*B[0]   + W[12]*B[1]  + W[19]*B[2]   + W[26]*B[3]  + W[33]*B[4]  + W[40]*B[5]  + W[47]*B[6];
  A[6]  =  W[6]*B[0]   + W[13]*B[1]   + W[20]*B[2]   + W[27]*B[3]  + W[34]*B[4] + W[41]*B[5]  + W[48]*B[6];
  A[7]  =  W[0]*B[7]   + W[7]*B[8]   + W[14]*B[9]   + W[21]*B[10]  + W[28]*B[11]  + W[35]*B[12]  + W[42]*B[13];
  A[8]  =  W[1]*B[7]   + W[8]*B[8]   + W[15]*B[9]   + W[22]*B[10]  + W[29]*B[11]  + W[36]*B[12]  + W[43]*B[13];
  A[9]  =  W[2]*B[7]   + W[9]*B[8]   + W[16]*B[9]   + W[23]*B[10]  + W[30]*B[11]  + W[37]*B[12]  + W[44]*B[13];
  A[10] =  W[3]*B[7]   + W[10]*B[8]   + W[17]*B[9]   + W[24]*B[10]  + W[31]*B[11]  + W[38]*B[12]  + W[45]*B[13];
  A[11] =  W[4]*B[7]   + W[11]*B[8]  + W[18]*B[9]   + W[25]*B[10]  + W[32]*B[11]  + W[39]*B[12]  + W[46]*B[13];
  A[12] =  W[5]*B[7]   + W[12]*B[8]  + W[19]*B[9]   + W[26]*B[10]  + W[33]*B[11]  + W[40]*B[12]  + W[47]*B[13];
  A[13] =  W[6]*B[7]   + W[13]*B[8]   + W[20]*B[9]   + W[27]*B[10]  + W[34]*B[11] + W[41]*B[12]  + W[48]*B[13];
  A[14] =  W[0]*B[14]   + W[7]*B[15]   + W[14]*B[16]   + W[21]*B[17]  + W[28]*B[18]  + W[35]*B[19]  + W[42]*B[20];
  A[15] =  W[1]*B[14]   + W[8]*B[15]   + W[15]*B[16]   + W[22]*B[17]  + W[29]*B[18]  + W[36]*B[19]  + W[43]*B[20];
  A[16] =  W[2]*B[14]   + W[9]*B[15]   + W[16]*B[16]   + W[23]*B[17]  + W[30]*B[18]  + W[37]*B[19]  + W[44]*B[20];
  A[17] =  W[3]*B[14]   + W[10]*B[15]   + W[17]*B[16]   + W[24]*B[17]  + W[31]*B[18]  + W[38]*B[19]  + W[45]*B[20];
  A[18] =  W[4]*B[14]   + W[11]*B[15]  + W[18]*B[16]   + W[25]*B[17]  + W[32]*B[18]  + W[39]*B[19]  + W[46]*B[20];
  A[19] =  W[5]*B[14]   + W[12]*B[15]  + W[19]*B[16]   + W[26]*B[17]  + W[33]*B[18]  + W[40]*B[19]  + W[47]*B[20];
  A[20] =  W[6]*B[14]   + W[13]*B[15]   + W[20]*B[16]   + W[27]*B[17]  + W[34]*B[18] + W[41]*B[19]  + W[48]*B[20];
  A[21] =  W[0]*B[21]   + W[7]*B[22]   + W[14]*B[23]   + W[21]*B[24]  + W[28]*B[25]  + W[35]*B[26]  + W[42]*B[27];
  A[22] =  W[1]*B[21]   + W[8]*B[22]   + W[15]*B[23]   + W[22]*B[24]  + W[29]*B[25]  + W[36]*B[26]  + W[43]*B[27];
  A[23] =  W[2]*B[21]   + W[9]*B[22]   + W[16]*B[23]   + W[23]*B[24]  + W[30]*B[25]  + W[37]*B[26]  + W[44]*B[27];
  A[24] =  W[3]*B[21]   + W[10]*B[22]   + W[17]*B[23]   + W[24]*B[24]  + W[31]*B[25]  + W[38]*B[26]  + W[45]*B[27];
  A[25] =  W[4]*B[21]   + W[11]*B[22]  + W[18]*B[23]   + W[25]*B[24]  + W[32]*B[25]  + W[39]*B[26]  + W[46]*B[27];
  A[26] =  W[5]*B[21]   + W[12]*B[22]  + W[19]*B[23]   + W[26]*B[24]  + W[33]*B[25]  + W[40]*B[26]  + W[47]*B[27];
  A[27] =  W[6]*B[21]   + W[13]*B[22]   + W[20]*B[23]   + W[27]*B[24]  + W[34]*B[25] + W[41]*B[26]  + W[48]*B[27];
  A[28] =  W[0]*B[28]   + W[7]*B[29]   + W[14]*B[30]   + W[21]*B[31]  + W[28]*B[32]  + W[35]*B[33]  + W[42]*B[34];
  A[29] =  W[1]*B[28]   + W[8]*B[29]   + W[15]*B[30]   + W[22]*B[31]  + W[29]*B[32]  + W[36]*B[33]  + W[43]*B[34];
  A[30] =  W[2]*B[28]   + W[9]*B[29]   + W[16]*B[30]   + W[23]*B[31]  + W[30]*B[32]  + W[37]*B[33]  + W[44]*B[34];
  A[31] =  W[3]*B[28]   + W[10]*B[29]   + W[17]*B[30]   + W[24]*B[31]  + W[31]*B[32]  + W[38]*B[33]  + W[45]*B[34];
  A[32] =  W[4]*B[28]   + W[11]*B[29]  + W[18]*B[30]   + W[25]*B[31]  + W[32]*B[32]  + W[39]*B[33]  + W[46]*B[34];
  A[33] =  W[5]*B[28]   + W[12]*B[29]  + W[19]*B[30]   + W[26]*B[31]  + W[33]*B[32]  + W[40]*B[33]  + W[47]*B[34];
  A[34] =  W[6]*B[28]   + W[13]*B[29]   + W[20]*B[30]   + W[27]*B[31]  + W[34]*B[32] + W[41]*B[33]  + W[48]*B[34];
  A[35] =  W[0]*B[35]   + W[7]*B[36]   + W[14]*B[37]   + W[21]*B[38]  + W[28]*B[39]  + W[35]*B[40]  + W[42]*B[41];
  A[36] =  W[1]*B[35]   + W[8]*B[36]   + W[15]*B[37]   + W[22]*B[38]  + W[29]*B[39]  + W[36]*B[40]  + W[43]*B[41];
  A[37] =  W[2]*B[35]   + W[9]*B[36]   + W[16]*B[37]   + W[23]*B[38]  + W[30]*B[39]  + W[37]*B[40]  + W[44]*B[41];
  A[38] =  W[3]*B[35]   + W[10]*B[36]   + W[17]*B[37]   + W[24]*B[38]  + W[31]*B[39]  + W[38]*B[40]  + W[45]*B[41];
  A[39] =  W[4]*B[35]   + W[11]*B[36]  + W[18]*B[37]   + W[25]*B[38]  + W[32]*B[39]  + W[39]*B[40]  + W[46]*B[41];
  A[40] =  W[5]*B[35]   + W[12]*B[36]  + W[19]*B[37]   + W[26]*B[38]  + W[33]*B[39]  + W[40]*B[40]  + W[47]*B[41];
  A[41] =  W[6]*B[35]   + W[13]*B[36]   + W[20]*B[37]   + W[27]*B[38]  + W[34]*B[39] + W[41]*B[40]  + W[48]*B[41];
  A[42] =  W[0]*B[42]   + W[7]*B[43]   + W[14]*B[44]   + W[21]*B[45]  + W[28]*B[46]  + W[35]*B[47]  + W[42]*B[48];
  A[43] =  W[1]*B[42]   + W[8]*B[43]   + W[15]*B[44]   + W[22]*B[45]  + W[29]*B[46]  + W[36]*B[47]  + W[43]*B[48];
  A[44] =  W[2]*B[42]   + W[9]*B[43]   + W[16]*B[44]   + W[23]*B[45]  + W[30]*B[46]  + W[37]*B[47]  + W[44]*B[48];
  A[45] =  W[3]*B[42]   + W[10]*B[43]   + W[17]*B[44]   + W[24]*B[45]  + W[31]*B[46]  + W[38]*B[47]  + W[45]*B[48];
  A[46] =  W[4]*B[42]   + W[11]*B[43]  + W[18]*B[44]   + W[25]*B[45]  + W[32]*B[46]  + W[39]*B[47]  + W[46]*B[48];
  A[47] =  W[5]*B[42]   + W[12]*B[43]  + W[19]*B[44]   + W[26]*B[45]  + W[33]*B[46]  + W[40]*B[47]  + W[47]*B[48];
  A[48] =  W[6]*B[42]   + W[13]*B[43]   + W[20]*B[44]   + W[27]*B[45]  + W[34]*B[46] + W[41]*B[47]  + W[48]*B[48];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_7: A = A - B * C with size bs=7

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_7(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0]  -=  B[0]*C[0]   + B[7]*C[1]   + B[14]*C[2]   + B[21]*C[3]  + B[28]*C[4]  + B[35]*C[5]  + B[42]*C[6];
  A[1]  -=  B[1]*C[0]   + B[8]*C[1]   + B[15]*C[2]   + B[22]*C[3]  + B[29]*C[4]  + B[36]*C[5]  + B[43]*C[6];
  A[2]  -=  B[2]*C[0]   + B[9]*C[1]   + B[16]*C[2]   + B[23]*C[3]  + B[30]*C[4]  + B[37]*C[5]  + B[44]*C[6];
  A[3]  -=  B[3]*C[0]   + B[10]*C[1]   + B[17]*C[2]   + B[24]*C[3]  + B[31]*C[4]  + B[38]*C[5]  + B[45]*C[6];
  A[4]  -=  B[4]*C[0]   + B[11]*C[1]  + B[18]*C[2]   + B[25]*C[3]  + B[32]*C[4]  + B[39]*C[5]  + B[46]*C[6];
  A[5]  -=  B[5]*C[0]   + B[12]*C[1]  + B[19]*C[2]   + B[26]*C[3]  + B[33]*C[4]  + B[40]*C[5]  + B[47]*C[6];
  A[6]  -=  B[6]*C[0]   + B[13]*C[1]   + B[20]*C[2]   + B[27]*C[3]  + B[34]*C[4] + B[41]*C[5]  + B[48]*C[6];
  A[7]  -=  B[0]*C[7]   + B[7]*C[8]   + B[14]*C[9]   + B[21]*C[10]  + B[28]*C[11]  + B[35]*C[12]  + B[42]*C[13];
  A[8]  -=  B[1]*C[7]   + B[8]*C[8]   + B[15]*C[9]   + B[22]*C[10]  + B[29]*C[11]  + B[36]*C[12]  + B[43]*C[13];
  A[9]  -=  B[2]*C[7]   + B[9]*C[8]   + B[16]*C[9]   + B[23]*C[10]  + B[30]*C[11]  + B[37]*C[12]  + B[44]*C[13];
  A[10] -=  B[3]*C[7]   + B[10]*C[8]   + B[17]*C[9]   + B[24]*C[10]  + B[31]*C[11]  + B[38]*C[12]  + B[45]*C[13];
  A[11] -=  B[4]*C[7]   + B[11]*C[8]  + B[18]*C[9]   + B[25]*C[10]  + B[32]*C[11]  + B[39]*C[12]  + B[46]*C[13];
  A[12] -=  B[5]*C[7]   + B[12]*C[8]  + B[19]*C[9]   + B[26]*C[10]  + B[33]*C[11]  + B[40]*C[12]  + B[47]*C[13];
  A[13] -=  B[6]*C[7]   + B[13]*C[8]   + B[20]*C[9]   + B[27]*C[10]  + B[34]*C[11] + B[41]*C[12]  + B[48]*C[13];
  A[14] -=  B[0]*C[14]   + B[7]*C[15]   + B[14]*C[16]   + B[21]*C[17]  + B[28]*C[18]  + B[35]*C[19]  + B[42]*C[20];
  A[15] -=  B[1]*C[14]   + B[8]*C[15]   + B[15]*C[16]   + B[22]*C[17]  + B[29]*C[18]  + B[36]*C[19]  + B[43]*C[20];
  A[16] -=  B[2]*C[14]   + B[9]*C[15]   + B[16]*C[16]   + B[23]*C[17]  + B[30]*C[18]  + B[37]*C[19]  + B[44]*C[20];
  A[17] -=  B[3]*C[14]   + B[10]*C[15]   + B[17]*C[16]   + B[24]*C[17]  + B[31]*C[18]  + B[38]*C[19]  + B[45]*C[20];
  A[18] -=  B[4]*C[14]   + B[11]*C[15]  + B[18]*C[16]   + B[25]*C[17]  + B[32]*C[18]  + B[39]*C[19]  + B[46]*C[20];
  A[19] -=  B[5]*C[14]   + B[12]*C[15]  + B[19]*C[16]   + B[26]*C[17]  + B[33]*C[18]  + B[40]*C[19]  + B[47]*C[20];
  A[20] -=  B[6]*C[14]   + B[13]*C[15]   + B[20]*C[16]   + B[27]*C[17]  + B[34]*C[18] + B[41]*C[19]  + B[48]*C[20];
  A[21] -=  B[0]*C[21]   + B[7]*C[22]   + B[14]*C[23]   + B[21]*C[24]  + B[28]*C[25]  + B[35]*C[26]  + B[42]*C[27];
  A[22] -=  B[1]*C[21]   + B[8]*C[22]   + B[15]*C[23]   + B[22]*C[24]  + B[29]*C[25]  + B[36]*C[26]  + B[43]*C[27];
  A[23] -=  B[2]*C[21]   + B[9]*C[22]   + B[16]*C[23]   + B[23]*C[24]  + B[30]*C[25]  + B[37]*C[26]  + B[44]*C[27];
  A[24] -=  B[3]*C[21]   + B[10]*C[22]   + B[17]*C[23]   + B[24]*C[24]  + B[31]*C[25]  + B[38]*C[26]  + B[45]*C[27];
  A[25] -=  B[4]*C[21]   + B[11]*C[22]  + B[18]*C[23]   + B[25]*C[24]  + B[32]*C[25]  + B[39]*C[26]  + B[46]*C[27];
  A[26] -=  B[5]*C[21]   + B[12]*C[22]  + B[19]*C[23]   + B[26]*C[24]  + B[33]*C[25]  + B[40]*C[26]  + B[47]*C[27];
  A[27] -=  B[6]*C[21]   + B[13]*C[22]   + B[20]*C[23]   + B[27]*C[24]  + B[34]*C[25] + B[41]*C[26]  + B[48]*C[27];
  A[28] -=  B[0]*C[28]   + B[7]*C[29]   + B[14]*C[30]   + B[21]*C[31]  + B[28]*C[32]  + B[35]*C[33]  + B[42]*C[34];
  A[29] -=  B[1]*C[28]   + B[8]*C[29]   + B[15]*C[30]   + B[22]*C[31]  + B[29]*C[32]  + B[36]*C[33]  + B[43]*C[34];
  A[30] -=  B[2]*C[28]   + B[9]*C[29]   + B[16]*C[30]   + B[23]*C[31]  + B[30]*C[32]  + B[37]*C[33]  + B[44]*C[34];
  A[31] -=  B[3]*C[28]   + B[10]*C[29]   + B[17]*C[30]   + B[24]*C[31]  + B[31]*C[32]  + B[38]*C[33]  + B[45]*C[34];
  A[32] -=  B[4]*C[28]   + B[11]*C[29]  + B[18]*C[30]   + B[25]*C[31]  + B[32]*C[32]  + B[39]*C[33]  + B[46]*C[34];
  A[33] -=  B[5]*C[28]   + B[12]*C[29]  + B[19]*C[30]   + B[26]*C[31]  + B[33]*C[32]  + B[40]*C[33]  + B[47]*C[34];
  A[34] -=  B[6]*C[28]   + B[13]*C[29]   + B[20]*C[30]   + B[27]*C[31]  + B[34]*C[32] + B[41]*C[33]  + B[48]*C[34];
  A[35] -=  B[0]*C[35]   + B[7]*C[36]   + B[14]*C[37]   + B[21]*C[38]  + B[28]*C[39]  + B[35]*C[40]  + B[42]*C[41];
  A[36] -=  B[1]*C[35]   + B[8]*C[36]   + B[15]*C[37]   + B[22]*C[38]  + B[29]*C[39]  + B[36]*C[40]  + B[43]*C[41];
  A[37] -=  B[2]*C[35]   + B[9]*C[36]   + B[16]*C[37]   + B[23]*C[38]  + B[30]*C[39]  + B[37]*C[40]  + B[44]*C[41];
  A[38] -=  B[3]*C[35]   + B[10]*C[36]   + B[17]*C[37]   + B[24]*C[38]  + B[31]*C[39]  + B[38]*C[40]  + B[45]*C[41];
  A[39] -=  B[4]*C[35]   + B[11]*C[36]  + B[18]*C[37]   + B[25]*C[38]  + B[32]*C[39]  + B[39]*C[40]  + B[46]*C[41];
  A[40] -=  B[5]*C[35]   + B[12]*C[36]  + B[19]*C[37]   + B[26]*C[38]  + B[33]*C[39]  + B[40]*C[40]  + B[47]*C[41];
  A[41] -=  B[6]*C[35]   + B[13]*C[36]   + B[20]*C[37]   + B[27]*C[38]  + B[34]*C[39] + B[41]*C[40]  + B[48]*C[41];
  A[42] -=  B[0]*C[42]   + B[7]*C[43]   + B[14]*C[44]   + B[21]*C[45]  + B[28]*C[46]  + B[35]*C[47]  + B[42]*C[48];
  A[43] -=  B[1]*C[42]   + B[8]*C[43]   + B[15]*C[44]   + B[22]*C[45]  + B[29]*C[46]  + B[36]*C[47]  + B[43]*C[48];
  A[44] -=  B[2]*C[42]   + B[9]*C[43]   + B[16]*C[44]   + B[23]*C[45]  + B[30]*C[46]  + B[37]*C[47]  + B[44]*C[48];
  A[45] -=  B[3]*C[42]   + B[10]*C[43]   + B[17]*C[44]   + B[24]*C[45]  + B[31]*C[46]  + B[38]*C[47]  + B[45]*C[48];
  A[46] -=  B[4]*C[42]   + B[11]*C[43]  + B[18]*C[44]   + B[25]*C[45]  + B[32]*C[46]  + B[39]*C[47]  + B[46]*C[48];
  A[47] -=  B[5]*C[42]   + B[12]*C[43]  + B[19]*C[44]   + B[26]*C[45]  + B[33]*C[46]  + B[40]*C[47]  + B[47]*C[48];
  A[48] -=  B[6]*C[42]   + B[13]*C[43]   + B[20]*C[44]   + B[27]*C[45]  + B[34]*C[46] + B[41]*C[47]  + B[48]*C[48];
  return 0;
}

#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES) && !defined(PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND)
#include <immintrin.h>
PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_9(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;
  PetscInt        i;
  __m256d         S0,S1,S2,S3,S4,S5,S6,S7,S8,B0,B1,B2,B6,B7,B8,A0,A1,A2,A3,A4,A5,A6,A7,A8;

  ierr = PetscArraycpy(W,A,81);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    S0 = _mm256_setzero_pd(); S1 = _mm256_setzero_pd(); S2 = _mm256_setzero_pd();
    S3 = _mm256_setzero_pd(); S4 = _mm256_setzero_pd(); S5 = _mm256_setzero_pd();
    S6 = _mm256_setzero_pd(); S7 = _mm256_setzero_pd(); S8 = _mm256_setzero_pd();

    A0 = _mm256_loadu_pd  (W+ 0); A1 = _mm256_loadu_pd  (W+ 4); A2 = _mm256_loadu_pd  (W+ 8);
    B0 = _mm256_broadcast_sd(B+ 0); B1 = _mm256_broadcast_sd(B+ 9); B2 = _mm256_broadcast_sd(B+18);
    S0 = _mm256_fmadd_pd(A0,B0,S0); S1 = _mm256_fmadd_pd(A1,B0,S1); S2 = _mm256_fmadd_pd(A2,B0,S2);
    S3 = _mm256_fmadd_pd(A0,B1,S3); S4 = _mm256_fmadd_pd(A1,B1,S4); S5 = _mm256_fmadd_pd(A2,B1,S5);
    S6 = _mm256_fmadd_pd(A0,B2,S6); S7 = _mm256_fmadd_pd(A1,B2,S7); S8 = _mm256_fmadd_pd(A2,B2,S8);

    A3 = _mm256_loadu_pd  (W+ 9); A4 = _mm256_loadu_pd  (W+13); A5 = _mm256_loadu_pd  (W+17);
    B6 = _mm256_broadcast_sd(B+ 1); B7 = _mm256_broadcast_sd(B+10); B8 = _mm256_broadcast_sd(B+19);
    S0 = _mm256_fmadd_pd(A3,B6,S0); S1 = _mm256_fmadd_pd(A4,B6,S1); S2 = _mm256_fmadd_pd(A5,B6,S2);
    S3 = _mm256_fmadd_pd(A3,B7,S3); S4 = _mm256_fmadd_pd(A4,B7,S4); S5 = _mm256_fmadd_pd(A5,B7,S5);
    S6 = _mm256_fmadd_pd(A3,B8,S6); S7 = _mm256_fmadd_pd(A4,B8,S7); S8 = _mm256_fmadd_pd(A5,B8,S8);

    A6 = _mm256_loadu_pd  (W+18); A7 = _mm256_loadu_pd  (W+22); A8 = _mm256_loadu_pd  (W+26);
    B0 = _mm256_broadcast_sd(B+ 2); B1 = _mm256_broadcast_sd(B+11); B2 = _mm256_broadcast_sd(B+20);
    S0 = _mm256_fmadd_pd(A6,B0,S0); S1 = _mm256_fmadd_pd(A7,B0,S1); S2 = _mm256_fmadd_pd(A8,B0,S2);
    S3 = _mm256_fmadd_pd(A6,B1,S3); S4 = _mm256_fmadd_pd(A7,B1,S4); S5 = _mm256_fmadd_pd(A8,B1,S5);
    S6 = _mm256_fmadd_pd(A6,B2,S6); S7 = _mm256_fmadd_pd(A7,B2,S7); S8 = _mm256_fmadd_pd(A8,B2,S8);

    A0 = _mm256_loadu_pd  (W+27); A1 = _mm256_loadu_pd  (W+31); A2 = _mm256_loadu_pd  (W+35);
    B6 = _mm256_broadcast_sd(B+ 3); B7 = _mm256_broadcast_sd(B+12); B8 = _mm256_broadcast_sd(B+21);
    S0 = _mm256_fmadd_pd(A0,B6,S0); S1 = _mm256_fmadd_pd(A1,B6,S1); S2 = _mm256_fmadd_pd(A2,B6,S2);
    S3 = _mm256_fmadd_pd(A0,B7,S3); S4 = _mm256_fmadd_pd(A1,B7,S4); S5 = _mm256_fmadd_pd(A2,B7,S5);
    S6 = _mm256_fmadd_pd(A0,B8,S6); S7 = _mm256_fmadd_pd(A1,B8,S7); S8 = _mm256_fmadd_pd(A2,B8,S8);

    A3 = _mm256_loadu_pd  (W+36); A4 = _mm256_loadu_pd  (W+40); A5 = _mm256_loadu_pd  (W+44);
    B0 = _mm256_broadcast_sd(B+ 4); B1 = _mm256_broadcast_sd(B+13); B2 = _mm256_broadcast_sd(B+22);
    S0 = _mm256_fmadd_pd(A3,B0,S0); S1 = _mm256_fmadd_pd(A4,B0,S1); S2 = _mm256_fmadd_pd(A5,B0,S2);
    S3 = _mm256_fmadd_pd(A3,B1,S3); S4 = _mm256_fmadd_pd(A4,B1,S4); S5 = _mm256_fmadd_pd(A5,B1,S5);
    S6 = _mm256_fmadd_pd(A3,B2,S6); S7 = _mm256_fmadd_pd(A4,B2,S7); S8 = _mm256_fmadd_pd(A5,B2,S8);

    A0 = _mm256_loadu_pd  (W+45); A1 = _mm256_loadu_pd  (W+49); A2 = _mm256_loadu_pd  (W+53);
    B6 = _mm256_broadcast_sd(B+ 5); B7 = _mm256_broadcast_sd(B+14); B8 = _mm256_broadcast_sd(B+23);
    S0 = _mm256_fmadd_pd(A0,B6,S0); S1 = _mm256_fmadd_pd(A1,B6,S1); S2 = _mm256_fmadd_pd(A2,B6,S2);
    S3 = _mm256_fmadd_pd(A0,B7,S3); S4 = _mm256_fmadd_pd(A1,B7,S4); S5 = _mm256_fmadd_pd(A2,B7,S5);
    S6 = _mm256_fmadd_pd(A0,B8,S6); S7 = _mm256_fmadd_pd(A1,B8,S7); S8 = _mm256_fmadd_pd(A2,B8,S8);

    A3 = _mm256_loadu_pd  (W+54); A4 = _mm256_loadu_pd  (W+58); A5 = _mm256_loadu_pd  (W+62);
    B0 = _mm256_broadcast_sd(B+ 6); B1 = _mm256_broadcast_sd(B+15); B2 = _mm256_broadcast_sd(B+24);
    S0 = _mm256_fmadd_pd(A3,B0,S0); S1 = _mm256_fmadd_pd(A4,B0,S1); S2 = _mm256_fmadd_pd(A5,B0,S2);
    S3 = _mm256_fmadd_pd(A3,B1,S3); S4 = _mm256_fmadd_pd(A4,B1,S4); S5 = _mm256_fmadd_pd(A5,B1,S5);
    S6 = _mm256_fmadd_pd(A3,B2,S6); S7 = _mm256_fmadd_pd(A4,B2,S7); S8 = _mm256_fmadd_pd(A5,B2,S8);

    A6 = _mm256_loadu_pd  (W+63); A7 = _mm256_loadu_pd  (W+67); A8 = _mm256_loadu_pd  (W+71);
    B6 = _mm256_broadcast_sd(B+ 7); B7 = _mm256_broadcast_sd(B+16); B8 = _mm256_broadcast_sd(B+25);
    S0 = _mm256_fmadd_pd(A6,B6,S0); S1 = _mm256_fmadd_pd(A7,B6,S1); S2 = _mm256_fmadd_pd(A8,B6,S2);
    S3 = _mm256_fmadd_pd(A6,B7,S3); S4 = _mm256_fmadd_pd(A7,B7,S4); S5 = _mm256_fmadd_pd(A8,B7,S5);
    S6 = _mm256_fmadd_pd(A6,B8,S6); S7 = _mm256_fmadd_pd(A7,B8,S7); S8 = _mm256_fmadd_pd(A8,B8,S8);

    A0 = _mm256_loadu_pd  (W+72); A1 = _mm256_loadu_pd  (W+76); A2 = _mm256_broadcast_sd(W+80);
    B0 = _mm256_broadcast_sd(B+ 8); B1 = _mm256_broadcast_sd(B+17); B2 = _mm256_broadcast_sd(B+26);
    S0 = _mm256_fmadd_pd(A0,B0,S0); S1 = _mm256_fmadd_pd(A1,B0,S1); S2 = _mm256_fmadd_pd(A2,B0,S2);
    S3 = _mm256_fmadd_pd(A0,B1,S3); S4 = _mm256_fmadd_pd(A1,B1,S4); S5 = _mm256_fmadd_pd(A2,B1,S5);
    S6 = _mm256_fmadd_pd(A0,B2,S6); S7 = _mm256_fmadd_pd(A1,B2,S7); S8 = _mm256_fmadd_pd(A2,B2,S8);

    _mm256_storeu_pd(&A[ 0+i*27], S0); _mm256_storeu_pd(&A[ 4+i*27], S1); _mm256_maskstore_pd(&A[ 8+i*27], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), S2);
    _mm256_storeu_pd(&A[ 9+i*27], S3); _mm256_storeu_pd(&A[13+i*27], S4); _mm256_maskstore_pd(&A[17+i*27], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), S5);
    _mm256_storeu_pd(&A[18+i*27], S6); _mm256_storeu_pd(&A[22+i*27], S7); _mm256_maskstore_pd(&A[26+i*27], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), S8);

    B += 27;
  }
  return 0;
}
#else
PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_9(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,81);CHKERRQ(ierr);
  A[ 0] = W[0]*B[ 0] + W[ 9]*B[ 1] + W[18]*B[ 2] + W[27]*B[ 3] + W[36]*B[ 4] + W[45]*B[ 5] + W[54]*B[ 6] + W[63]*B[ 7] + W[72]*B[ 8];
  A[ 1] = W[1]*B[ 0] + W[10]*B[ 1] + W[19]*B[ 2] + W[28]*B[ 3] + W[37]*B[ 4] + W[46]*B[ 5] + W[55]*B[ 6] + W[64]*B[ 7] + W[73]*B[ 8];
  A[ 2] = W[2]*B[ 0] + W[11]*B[ 1] + W[20]*B[ 2] + W[29]*B[ 3] + W[38]*B[ 4] + W[47]*B[ 5] + W[56]*B[ 6] + W[65]*B[ 7] + W[74]*B[ 8];
  A[ 3] = W[3]*B[ 0] + W[12]*B[ 1] + W[21]*B[ 2] + W[30]*B[ 3] + W[39]*B[ 4] + W[48]*B[ 5] + W[57]*B[ 6] + W[66]*B[ 7] + W[75]*B[ 8];
  A[ 4] = W[4]*B[ 0] + W[13]*B[ 1] + W[22]*B[ 2] + W[31]*B[ 3] + W[40]*B[ 4] + W[49]*B[ 5] + W[58]*B[ 6] + W[67]*B[ 7] + W[76]*B[ 8];
  A[ 5] = W[5]*B[ 0] + W[14]*B[ 1] + W[23]*B[ 2] + W[32]*B[ 3] + W[41]*B[ 4] + W[50]*B[ 5] + W[59]*B[ 6] + W[68]*B[ 7] + W[77]*B[ 8];
  A[ 6] = W[6]*B[ 0] + W[15]*B[ 1] + W[24]*B[ 2] + W[33]*B[ 3] + W[42]*B[ 4] + W[51]*B[ 5] + W[60]*B[ 6] + W[69]*B[ 7] + W[78]*B[ 8];
  A[ 7] = W[7]*B[ 0] + W[16]*B[ 1] + W[25]*B[ 2] + W[34]*B[ 3] + W[43]*B[ 4] + W[52]*B[ 5] + W[61]*B[ 6] + W[70]*B[ 7] + W[79]*B[ 8];
  A[ 8] = W[8]*B[ 0] + W[17]*B[ 1] + W[26]*B[ 2] + W[35]*B[ 3] + W[44]*B[ 4] + W[53]*B[ 5] + W[62]*B[ 6] + W[71]*B[ 7] + W[80]*B[ 8];

  A[ 9] = W[0]*B[ 9] + W[ 9]*B[10] + W[18]*B[11] + W[27]*B[12] + W[36]*B[13] + W[45]*B[14] + W[54]*B[15] + W[63]*B[16] + W[72]*B[17];
  A[10] = W[1]*B[ 9] + W[10]*B[10] + W[19]*B[11] + W[28]*B[12] + W[37]*B[13] + W[46]*B[14] + W[55]*B[15] + W[64]*B[16] + W[73]*B[17];
  A[11] = W[2]*B[ 9] + W[11]*B[10] + W[20]*B[11] + W[29]*B[12] + W[38]*B[13] + W[47]*B[14] + W[56]*B[15] + W[65]*B[16] + W[74]*B[17];
  A[12] = W[3]*B[ 9] + W[12]*B[10] + W[21]*B[11] + W[30]*B[12] + W[39]*B[13] + W[48]*B[14] + W[57]*B[15] + W[66]*B[16] + W[75]*B[17];
  A[13] = W[4]*B[ 9] + W[13]*B[10] + W[22]*B[11] + W[31]*B[12] + W[40]*B[13] + W[49]*B[14] + W[58]*B[15] + W[67]*B[16] + W[76]*B[17];
  A[14] = W[5]*B[ 9] + W[14]*B[10] + W[23]*B[11] + W[32]*B[12] + W[41]*B[13] + W[50]*B[14] + W[59]*B[15] + W[68]*B[16] + W[77]*B[17];
  A[15] = W[6]*B[ 9] + W[15]*B[10] + W[24]*B[11] + W[33]*B[12] + W[42]*B[13] + W[51]*B[14] + W[60]*B[15] + W[69]*B[16] + W[78]*B[17];
  A[16] = W[7]*B[ 9] + W[16]*B[10] + W[25]*B[11] + W[34]*B[12] + W[43]*B[13] + W[52]*B[14] + W[61]*B[15] + W[70]*B[16] + W[79]*B[17];
  A[17] = W[8]*B[ 9] + W[17]*B[10] + W[26]*B[11] + W[35]*B[12] + W[44]*B[13] + W[53]*B[14] + W[62]*B[15] + W[71]*B[16] + W[80]*B[17];

  A[18] = W[0]*B[18] + W[ 9]*B[19] + W[18]*B[20] + W[27]*B[21] + W[36]*B[22] + W[45]*B[23] + W[54]*B[24] + W[63]*B[25] + W[72]*B[26];
  A[19] = W[1]*B[18] + W[10]*B[19] + W[19]*B[20] + W[28]*B[21] + W[37]*B[22] + W[46]*B[23] + W[55]*B[24] + W[64]*B[25] + W[73]*B[26];
  A[20] = W[2]*B[18] + W[11]*B[19] + W[20]*B[20] + W[29]*B[21] + W[38]*B[22] + W[47]*B[23] + W[56]*B[24] + W[65]*B[25] + W[74]*B[26];
  A[21] = W[3]*B[18] + W[12]*B[19] + W[21]*B[20] + W[30]*B[21] + W[39]*B[22] + W[48]*B[23] + W[57]*B[24] + W[66]*B[25] + W[75]*B[26];
  A[22] = W[4]*B[18] + W[13]*B[19] + W[22]*B[20] + W[31]*B[21] + W[40]*B[22] + W[49]*B[23] + W[58]*B[24] + W[67]*B[25] + W[76]*B[26];
  A[23] = W[5]*B[18] + W[14]*B[19] + W[23]*B[20] + W[32]*B[21] + W[41]*B[22] + W[50]*B[23] + W[59]*B[24] + W[68]*B[25] + W[77]*B[26];
  A[24] = W[6]*B[18] + W[15]*B[19] + W[24]*B[20] + W[33]*B[21] + W[42]*B[22] + W[51]*B[23] + W[60]*B[24] + W[69]*B[25] + W[78]*B[26];
  A[25] = W[7]*B[18] + W[16]*B[19] + W[25]*B[20] + W[34]*B[21] + W[43]*B[22] + W[52]*B[23] + W[61]*B[24] + W[70]*B[25] + W[79]*B[26];
  A[26] = W[8]*B[18] + W[17]*B[19] + W[26]*B[20] + W[35]*B[21] + W[44]*B[22] + W[53]*B[23] + W[62]*B[24] + W[71]*B[25] + W[80]*B[26];

  A[27] = W[0]*B[27] + W[ 9]*B[28] + W[18]*B[29] + W[27]*B[30] + W[36]*B[31] + W[45]*B[32] + W[54]*B[33] + W[63]*B[34] + W[72]*B[35];
  A[28] = W[1]*B[27] + W[10]*B[28] + W[19]*B[29] + W[28]*B[30] + W[37]*B[31] + W[46]*B[32] + W[55]*B[33] + W[64]*B[34] + W[73]*B[35];
  A[29] = W[2]*B[27] + W[11]*B[28] + W[20]*B[29] + W[29]*B[30] + W[38]*B[31] + W[47]*B[32] + W[56]*B[33] + W[65]*B[34] + W[74]*B[35];
  A[30] = W[3]*B[27] + W[12]*B[28] + W[21]*B[29] + W[30]*B[30] + W[39]*B[31] + W[48]*B[32] + W[57]*B[33] + W[66]*B[34] + W[75]*B[35];
  A[31] = W[4]*B[27] + W[13]*B[28] + W[22]*B[29] + W[31]*B[30] + W[40]*B[31] + W[49]*B[32] + W[58]*B[33] + W[67]*B[34] + W[76]*B[35];
  A[32] = W[5]*B[27] + W[14]*B[28] + W[23]*B[29] + W[32]*B[30] + W[41]*B[31] + W[50]*B[32] + W[59]*B[33] + W[68]*B[34] + W[77]*B[35];
  A[33] = W[6]*B[27] + W[15]*B[28] + W[24]*B[29] + W[33]*B[30] + W[42]*B[31] + W[51]*B[32] + W[60]*B[33] + W[69]*B[34] + W[78]*B[35];
  A[34] = W[7]*B[27] + W[16]*B[28] + W[25]*B[29] + W[34]*B[30] + W[43]*B[31] + W[52]*B[32] + W[61]*B[33] + W[70]*B[34] + W[79]*B[35];
  A[35] = W[8]*B[27] + W[17]*B[28] + W[26]*B[29] + W[35]*B[30] + W[44]*B[31] + W[53]*B[32] + W[62]*B[33] + W[71]*B[34] + W[80]*B[35];

  A[36] = W[0]*B[36] + W[ 9]*B[37] + W[18]*B[38] + W[27]*B[39] + W[36]*B[40] + W[45]*B[41] + W[54]*B[42] + W[63]*B[43] + W[72]*B[44];
  A[37] = W[1]*B[36] + W[10]*B[37] + W[19]*B[38] + W[28]*B[39] + W[37]*B[40] + W[46]*B[41] + W[55]*B[42] + W[64]*B[43] + W[73]*B[44];
  A[38] = W[2]*B[36] + W[11]*B[37] + W[20]*B[38] + W[29]*B[39] + W[38]*B[40] + W[47]*B[41] + W[56]*B[42] + W[65]*B[43] + W[74]*B[44];
  A[39] = W[3]*B[36] + W[12]*B[37] + W[21]*B[38] + W[30]*B[39] + W[39]*B[40] + W[48]*B[41] + W[57]*B[42] + W[66]*B[43] + W[75]*B[44];
  A[40] = W[4]*B[36] + W[13]*B[37] + W[22]*B[38] + W[31]*B[39] + W[40]*B[40] + W[49]*B[41] + W[58]*B[42] + W[67]*B[43] + W[76]*B[44];
  A[41] = W[5]*B[36] + W[14]*B[37] + W[23]*B[38] + W[32]*B[39] + W[41]*B[40] + W[50]*B[41] + W[59]*B[42] + W[68]*B[43] + W[77]*B[44];
  A[42] = W[6]*B[36] + W[15]*B[37] + W[24]*B[38] + W[33]*B[39] + W[42]*B[40] + W[51]*B[41] + W[60]*B[42] + W[69]*B[43] + W[78]*B[44];
  A[43] = W[7]*B[36] + W[16]*B[37] + W[25]*B[38] + W[34]*B[39] + W[43]*B[40] + W[52]*B[41] + W[61]*B[42] + W[70]*B[43] + W[79]*B[44];
  A[44] = W[8]*B[36] + W[17]*B[37] + W[26]*B[38] + W[35]*B[39] + W[44]*B[40] + W[53]*B[41] + W[62]*B[42] + W[71]*B[43] + W[80]*B[44];

  A[45] = W[0]*B[45] + W[ 9]*B[46] + W[18]*B[47] + W[27]*B[48] + W[36]*B[49] + W[45]*B[50] + W[54]*B[51] + W[63]*B[52] + W[72]*B[53];
  A[46] = W[1]*B[45] + W[10]*B[46] + W[19]*B[47] + W[28]*B[48] + W[37]*B[49] + W[46]*B[50] + W[55]*B[51] + W[64]*B[52] + W[73]*B[53];
  A[47] = W[2]*B[45] + W[11]*B[46] + W[20]*B[47] + W[29]*B[48] + W[38]*B[49] + W[47]*B[50] + W[56]*B[51] + W[65]*B[52] + W[74]*B[53];
  A[48] = W[3]*B[45] + W[12]*B[46] + W[21]*B[47] + W[30]*B[48] + W[39]*B[49] + W[48]*B[50] + W[57]*B[51] + W[66]*B[52] + W[75]*B[53];
  A[49] = W[4]*B[45] + W[13]*B[46] + W[22]*B[47] + W[31]*B[48] + W[40]*B[49] + W[49]*B[50] + W[58]*B[51] + W[67]*B[52] + W[76]*B[53];
  A[50] = W[5]*B[45] + W[14]*B[46] + W[23]*B[47] + W[32]*B[48] + W[41]*B[49] + W[50]*B[50] + W[59]*B[51] + W[68]*B[52] + W[77]*B[53];
  A[51] = W[6]*B[45] + W[15]*B[46] + W[24]*B[47] + W[33]*B[48] + W[42]*B[49] + W[51]*B[50] + W[60]*B[51] + W[69]*B[52] + W[78]*B[53];
  A[52] = W[7]*B[45] + W[16]*B[46] + W[25]*B[47] + W[34]*B[48] + W[43]*B[49] + W[52]*B[50] + W[61]*B[51] + W[70]*B[52] + W[79]*B[53];
  A[53] = W[8]*B[45] + W[17]*B[46] + W[26]*B[47] + W[35]*B[48] + W[44]*B[49] + W[53]*B[50] + W[62]*B[51] + W[71]*B[52] + W[80]*B[53];

  A[54] = W[0]*B[54] + W[ 9]*B[55] + W[18]*B[56] + W[27]*B[57] + W[36]*B[58] + W[45]*B[59] + W[54]*B[60] + W[63]*B[61] + W[72]*B[62];
  A[55] = W[1]*B[54] + W[10]*B[55] + W[19]*B[56] + W[28]*B[57] + W[37]*B[58] + W[46]*B[59] + W[55]*B[60] + W[64]*B[61] + W[73]*B[62];
  A[56] = W[2]*B[54] + W[11]*B[55] + W[20]*B[56] + W[29]*B[57] + W[38]*B[58] + W[47]*B[59] + W[56]*B[60] + W[65]*B[61] + W[74]*B[62];
  A[57] = W[3]*B[54] + W[12]*B[55] + W[21]*B[56] + W[30]*B[57] + W[39]*B[58] + W[48]*B[59] + W[57]*B[60] + W[66]*B[61] + W[75]*B[62];
  A[58] = W[4]*B[54] + W[13]*B[55] + W[22]*B[56] + W[31]*B[57] + W[40]*B[58] + W[49]*B[59] + W[58]*B[60] + W[67]*B[61] + W[76]*B[62];
  A[59] = W[5]*B[54] + W[14]*B[55] + W[23]*B[56] + W[32]*B[57] + W[41]*B[58] + W[50]*B[59] + W[59]*B[60] + W[68]*B[61] + W[77]*B[62];
  A[60] = W[6]*B[54] + W[15]*B[55] + W[24]*B[56] + W[33]*B[57] + W[42]*B[58] + W[51]*B[59] + W[60]*B[60] + W[69]*B[61] + W[78]*B[62];
  A[61] = W[7]*B[54] + W[16]*B[55] + W[25]*B[56] + W[34]*B[57] + W[43]*B[58] + W[52]*B[59] + W[61]*B[60] + W[70]*B[61] + W[79]*B[62];
  A[62] = W[8]*B[54] + W[17]*B[55] + W[26]*B[56] + W[35]*B[57] + W[44]*B[58] + W[53]*B[59] + W[62]*B[60] + W[71]*B[61] + W[80]*B[62];

  A[63] = W[0]*B[63] + W[ 9]*B[64] + W[18]*B[65] + W[27]*B[66] + W[36]*B[67] + W[45]*B[68] + W[54]*B[69] + W[63]*B[70] + W[72]*B[71];
  A[64] = W[1]*B[63] + W[10]*B[64] + W[19]*B[65] + W[28]*B[66] + W[37]*B[67] + W[46]*B[68] + W[55]*B[69] + W[64]*B[70] + W[73]*B[71];
  A[65] = W[2]*B[63] + W[11]*B[64] + W[20]*B[65] + W[29]*B[66] + W[38]*B[67] + W[47]*B[68] + W[56]*B[69] + W[65]*B[70] + W[74]*B[71];
  A[66] = W[3]*B[63] + W[12]*B[64] + W[21]*B[65] + W[30]*B[66] + W[39]*B[67] + W[48]*B[68] + W[57]*B[69] + W[66]*B[70] + W[75]*B[71];
  A[67] = W[4]*B[63] + W[13]*B[64] + W[22]*B[65] + W[31]*B[66] + W[40]*B[67] + W[49]*B[68] + W[58]*B[69] + W[67]*B[70] + W[76]*B[71];
  A[68] = W[5]*B[63] + W[14]*B[64] + W[23]*B[65] + W[32]*B[66] + W[41]*B[67] + W[50]*B[68] + W[59]*B[69] + W[68]*B[70] + W[77]*B[71];
  A[69] = W[6]*B[63] + W[15]*B[64] + W[24]*B[65] + W[33]*B[66] + W[42]*B[67] + W[51]*B[68] + W[60]*B[69] + W[69]*B[70] + W[78]*B[71];
  A[70] = W[7]*B[63] + W[16]*B[64] + W[25]*B[65] + W[34]*B[66] + W[43]*B[67] + W[52]*B[68] + W[61]*B[69] + W[70]*B[70] + W[79]*B[71];
  A[71] = W[8]*B[63] + W[17]*B[64] + W[26]*B[65] + W[35]*B[66] + W[44]*B[67] + W[53]*B[68] + W[62]*B[69] + W[71]*B[70] + W[80]*B[71];

  A[72] = W[0]*B[72] + W[ 9]*B[73] + W[18]*B[74] + W[27]*B[75] + W[36]*B[76] + W[45]*B[77] + W[54]*B[78] + W[63]*B[79] + W[72]*B[80];
  A[73] = W[1]*B[72] + W[10]*B[73] + W[19]*B[74] + W[28]*B[75] + W[37]*B[76] + W[46]*B[77] + W[55]*B[78] + W[64]*B[79] + W[73]*B[80];
  A[74] = W[2]*B[72] + W[11]*B[73] + W[20]*B[74] + W[29]*B[75] + W[38]*B[76] + W[47]*B[77] + W[56]*B[78] + W[65]*B[79] + W[74]*B[80];
  A[75] = W[3]*B[72] + W[12]*B[73] + W[21]*B[74] + W[30]*B[75] + W[39]*B[76] + W[48]*B[77] + W[57]*B[78] + W[66]*B[79] + W[75]*B[80];
  A[76] = W[4]*B[72] + W[13]*B[73] + W[22]*B[74] + W[31]*B[75] + W[40]*B[76] + W[49]*B[77] + W[58]*B[78] + W[67]*B[79] + W[76]*B[80];
  A[77] = W[5]*B[72] + W[14]*B[73] + W[23]*B[74] + W[32]*B[75] + W[41]*B[76] + W[50]*B[77] + W[59]*B[78] + W[68]*B[79] + W[77]*B[80];
  A[78] = W[6]*B[72] + W[15]*B[73] + W[24]*B[74] + W[33]*B[75] + W[42]*B[76] + W[51]*B[77] + W[60]*B[78] + W[69]*B[79] + W[78]*B[80];
  A[79] = W[7]*B[72] + W[16]*B[73] + W[25]*B[74] + W[34]*B[75] + W[43]*B[76] + W[52]*B[77] + W[61]*B[78] + W[70]*B[79] + W[79]*B[80];
  A[80] = W[8]*B[72] + W[17]*B[73] + W[26]*B[74] + W[35]*B[75] + W[44]*B[76] + W[53]*B[77] + W[62]*B[78] + W[71]*B[79] + W[80]*B[80];
  return 0;
}
#endif

#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES) && !defined(PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND)
PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_9(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  PetscInt i;
  __m256d  A0,A1,A2,A3,A4,A5,A6,A7,A8,B0,B1,B2,B3,B4,B5,B6,B7,B8,C0,C1,C2,C3,C4,C5,C6,C7,C8;

  for (i=0; i<3; i++) {
    A0 = _mm256_loadu_pd  (A+ 0); A1 = _mm256_loadu_pd  (A+ 4); A2 = _mm256_loadu_pd(A+ 8);
    A3 = _mm256_loadu_pd  (A+ 9); A4 = _mm256_loadu_pd  (A+13); A5 = _mm256_loadu_pd(A+17);
    A6 = _mm256_loadu_pd  (A+18); A7 = _mm256_loadu_pd  (A+22); A8 = _mm256_loadu_pd(A+26);

    B0 = _mm256_loadu_pd   (B+ 0); B1 = _mm256_loadu_pd   (B+ 4); B2 = _mm256_loadu_pd   (B+ 8);
    C0 = _mm256_broadcast_sd (C+ 0); C1 = _mm256_broadcast_sd (C+ 9); C2 = _mm256_broadcast_sd (C+18);
    A0 = _mm256_fnmadd_pd(B0,C0,A0); A1 = _mm256_fnmadd_pd(B1,C0,A1); A2 = _mm256_fnmadd_pd(B2,C0,A2);
    A3 = _mm256_fnmadd_pd(B0,C1,A3); A4 = _mm256_fnmadd_pd(B1,C1,A4); A5 = _mm256_fnmadd_pd(B2,C1,A5);
    A6 = _mm256_fnmadd_pd(B0,C2,A6); A7 = _mm256_fnmadd_pd(B1,C2,A7); A8 = _mm256_fnmadd_pd(B2,C2,A8);

    B3 = _mm256_loadu_pd   (B+ 9); B4 = _mm256_loadu_pd   (B+13); B5 = _mm256_loadu_pd   (B+17);
    C3 = _mm256_broadcast_sd (C+ 1); C4 = _mm256_broadcast_sd (C+10); C5 = _mm256_broadcast_sd (C+19);
    A0 = _mm256_fnmadd_pd(B3,C3,A0); A1 = _mm256_fnmadd_pd(B4,C3,A1); A2 = _mm256_fnmadd_pd(B5,C3,A2);
    A3 = _mm256_fnmadd_pd(B3,C4,A3); A4 = _mm256_fnmadd_pd(B4,C4,A4); A5 = _mm256_fnmadd_pd(B5,C4,A5);
    A6 = _mm256_fnmadd_pd(B3,C5,A6); A7 = _mm256_fnmadd_pd(B4,C5,A7); A8 = _mm256_fnmadd_pd(B5,C5,A8);

    B6 = _mm256_loadu_pd   (B+18); B7 = _mm256_loadu_pd   (B+22); B8 = _mm256_loadu_pd   (B+26);
    C6 = _mm256_broadcast_sd (C+ 2); C7 = _mm256_broadcast_sd (C+11); C8 = _mm256_broadcast_sd (C+20);
    A0 = _mm256_fnmadd_pd(B6,C6,A0); A1 = _mm256_fnmadd_pd(B7,C6,A1); A2 = _mm256_fnmadd_pd(B8,C6,A2);
    A3 = _mm256_fnmadd_pd(B6,C7,A3); A4 = _mm256_fnmadd_pd(B7,C7,A4); A5 = _mm256_fnmadd_pd(B8,C7,A5);
    A6 = _mm256_fnmadd_pd(B6,C8,A6); A7 = _mm256_fnmadd_pd(B7,C8,A7); A8 = _mm256_fnmadd_pd(B8,C8,A8);

    B0 = _mm256_loadu_pd   (B+27); B1 = _mm256_loadu_pd   (B+31); B2 = _mm256_loadu_pd   (B+35);
    C0 = _mm256_broadcast_sd (C+ 3); C1 = _mm256_broadcast_sd (C+12); C2 = _mm256_broadcast_sd (C+21);
    A0 = _mm256_fnmadd_pd(B0,C0,A0); A1 = _mm256_fnmadd_pd(B1,C0,A1); A2 = _mm256_fnmadd_pd(B2,C0,A2);
    A3 = _mm256_fnmadd_pd(B0,C1,A3); A4 = _mm256_fnmadd_pd(B1,C1,A4); A5 = _mm256_fnmadd_pd(B2,C1,A5);
    A6 = _mm256_fnmadd_pd(B0,C2,A6); A7 = _mm256_fnmadd_pd(B1,C2,A7); A8 = _mm256_fnmadd_pd(B2,C2,A8);

    B3 = _mm256_loadu_pd   (B+36); B4 = _mm256_loadu_pd   (B+40); B5 = _mm256_loadu_pd   (B+44);
    C3 = _mm256_broadcast_sd (C+ 4); C4 = _mm256_broadcast_sd (C+13); C5 = _mm256_broadcast_sd (C+22);
    A0 = _mm256_fnmadd_pd(B3,C3,A0); A1 = _mm256_fnmadd_pd(B4,C3,A1); A2 = _mm256_fnmadd_pd(B5,C3,A2);
    A3 = _mm256_fnmadd_pd(B3,C4,A3); A4 = _mm256_fnmadd_pd(B4,C4,A4); A5 = _mm256_fnmadd_pd(B5,C4,A5);
    A6 = _mm256_fnmadd_pd(B3,C5,A6); A7 = _mm256_fnmadd_pd(B4,C5,A7); A8 = _mm256_fnmadd_pd(B5,C5,A8);

    B6 = _mm256_loadu_pd   (B+45); B7 = _mm256_loadu_pd   (B+49); B8 = _mm256_loadu_pd   (B+53);
    C6 = _mm256_broadcast_sd (C+ 5); C7 = _mm256_broadcast_sd (C+14); C8 = _mm256_broadcast_sd (C+23);
    A0 = _mm256_fnmadd_pd(B6,C6,A0); A1 = _mm256_fnmadd_pd(B7,C6,A1); A2 = _mm256_fnmadd_pd(B8,C6,A2);
    A3 = _mm256_fnmadd_pd(B6,C7,A3); A4 = _mm256_fnmadd_pd(B7,C7,A4); A5 = _mm256_fnmadd_pd(B8,C7,A5);
    A6 = _mm256_fnmadd_pd(B6,C8,A6); A7 = _mm256_fnmadd_pd(B7,C8,A7); A8 = _mm256_fnmadd_pd(B8,C8,A8);

    B0 = _mm256_loadu_pd   (B+54); B1 = _mm256_loadu_pd   (B+58); B2 = _mm256_loadu_pd   (B+62);
    C0 = _mm256_broadcast_sd (C+ 6); C1 = _mm256_broadcast_sd (C+15); C2 = _mm256_broadcast_sd (C+24);
    A0 = _mm256_fnmadd_pd(B0,C0,A0); A1 = _mm256_fnmadd_pd(B1,C0,A1); A2 = _mm256_fnmadd_pd(B2,C0,A2);
    A3 = _mm256_fnmadd_pd(B0,C1,A3); A4 = _mm256_fnmadd_pd(B1,C1,A4); A5 = _mm256_fnmadd_pd(B2,C1,A5);
    A6 = _mm256_fnmadd_pd(B0,C2,A6); A7 = _mm256_fnmadd_pd(B1,C2,A7); A8 = _mm256_fnmadd_pd(B2,C2,A8);

    B3 = _mm256_loadu_pd   (B+63); B4 = _mm256_loadu_pd   (B+67); B5 = _mm256_loadu_pd   (B+71);
    C3 = _mm256_broadcast_sd (C+ 7); C4 = _mm256_broadcast_sd( C+16); C5 = _mm256_broadcast_sd (C+25);
    A0 = _mm256_fnmadd_pd(B3,C3,A0); A1 = _mm256_fnmadd_pd(B4,C3,A1); A2 = _mm256_fnmadd_pd(B5,C3,A2);
    A3 = _mm256_fnmadd_pd(B3,C4,A3); A4 = _mm256_fnmadd_pd(B4,C4,A4); A5 = _mm256_fnmadd_pd(B5,C4,A5);
    A6 = _mm256_fnmadd_pd(B3,C5,A6); A7 = _mm256_fnmadd_pd(B4,C5,A7); A8 = _mm256_fnmadd_pd(B5,C5,A8);

    B6 = _mm256_loadu_pd   (B+72); B7 = _mm256_loadu_pd   (B+76); B8 = _mm256_broadcast_sd (B+80);
    C6 = _mm256_broadcast_sd (C+ 8); C7 = _mm256_broadcast_sd (C+17); C8 = _mm256_broadcast_sd (C+26);
    A0 = _mm256_fnmadd_pd(B6,C6,A0); A1 = _mm256_fnmadd_pd(B7,C6,A1); A2 = _mm256_fnmadd_pd(B8,C6,A2);
    A3 = _mm256_fnmadd_pd(B6,C7,A3); A4 = _mm256_fnmadd_pd(B7,C7,A4); A5 = _mm256_fnmadd_pd(B8,C7,A5);
    A6 = _mm256_fnmadd_pd(B6,C8,A6); A7 = _mm256_fnmadd_pd(B7,C8,A7); A8 = _mm256_fnmadd_pd(B8,C8,A8);

    _mm256_storeu_pd(&A[ 0], A0); _mm256_storeu_pd(&A[ 4], A1); _mm256_maskstore_pd(&A[ 8], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), A2);
    _mm256_storeu_pd(&A[ 9], A3); _mm256_storeu_pd(&A[13], A4); _mm256_maskstore_pd(&A[17], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), A5);
    _mm256_storeu_pd(&A[18], A6); _mm256_storeu_pd(&A[22], A7); _mm256_maskstore_pd(&A[26], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), A8);

    A += 27; C += 27;
  }
  return 0;
}
#else
PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_9(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[ 0] -= B[0]*C[ 0] + B[ 9]*C[ 1] + B[18]*C[ 2] + B[27]*C[ 3] + B[36]*C[ 4] + B[45]*C[ 5] + B[54]*C[ 6] + B[63]*C[ 7] + B[72]*C[ 8];
  A[ 1] -= B[1]*C[ 0] + B[10]*C[ 1] + B[19]*C[ 2] + B[28]*C[ 3] + B[37]*C[ 4] + B[46]*C[ 5] + B[55]*C[ 6] + B[64]*C[ 7] + B[73]*C[ 8];
  A[ 2] -= B[2]*C[ 0] + B[11]*C[ 1] + B[20]*C[ 2] + B[29]*C[ 3] + B[38]*C[ 4] + B[47]*C[ 5] + B[56]*C[ 6] + B[65]*C[ 7] + B[74]*C[ 8];
  A[ 3] -= B[3]*C[ 0] + B[12]*C[ 1] + B[21]*C[ 2] + B[30]*C[ 3] + B[39]*C[ 4] + B[48]*C[ 5] + B[57]*C[ 6] + B[66]*C[ 7] + B[75]*C[ 8];
  A[ 4] -= B[4]*C[ 0] + B[13]*C[ 1] + B[22]*C[ 2] + B[31]*C[ 3] + B[40]*C[ 4] + B[49]*C[ 5] + B[58]*C[ 6] + B[67]*C[ 7] + B[76]*C[ 8];
  A[ 5] -= B[5]*C[ 0] + B[14]*C[ 1] + B[23]*C[ 2] + B[32]*C[ 3] + B[41]*C[ 4] + B[50]*C[ 5] + B[59]*C[ 6] + B[68]*C[ 7] + B[77]*C[ 8];
  A[ 6] -= B[6]*C[ 0] + B[15]*C[ 1] + B[24]*C[ 2] + B[33]*C[ 3] + B[42]*C[ 4] + B[51]*C[ 5] + B[60]*C[ 6] + B[69]*C[ 7] + B[78]*C[ 8];
  A[ 7] -= B[7]*C[ 0] + B[16]*C[ 1] + B[25]*C[ 2] + B[34]*C[ 3] + B[43]*C[ 4] + B[52]*C[ 5] + B[61]*C[ 6] + B[70]*C[ 7] + B[79]*C[ 8];
  A[ 8] -= B[8]*C[ 0] + B[17]*C[ 1] + B[26]*C[ 2] + B[35]*C[ 3] + B[44]*C[ 4] + B[53]*C[ 5] + B[62]*C[ 6] + B[71]*C[ 7] + B[80]*C[ 8];

  A[ 9] -= B[0]*C[ 9] + B[ 9]*C[10] + B[18]*C[11] + B[27]*C[12] + B[36]*C[13] + B[45]*C[14] + B[54]*C[15] + B[63]*C[16] + B[72]*C[17];
  A[10] -= B[1]*C[ 9] + B[10]*C[10] + B[19]*C[11] + B[28]*C[12] + B[37]*C[13] + B[46]*C[14] + B[55]*C[15] + B[64]*C[16] + B[73]*C[17];
  A[11] -= B[2]*C[ 9] + B[11]*C[10] + B[20]*C[11] + B[29]*C[12] + B[38]*C[13] + B[47]*C[14] + B[56]*C[15] + B[65]*C[16] + B[74]*C[17];
  A[12] -= B[3]*C[ 9] + B[12]*C[10] + B[21]*C[11] + B[30]*C[12] + B[39]*C[13] + B[48]*C[14] + B[57]*C[15] + B[66]*C[16] + B[75]*C[17];
  A[13] -= B[4]*C[ 9] + B[13]*C[10] + B[22]*C[11] + B[31]*C[12] + B[40]*C[13] + B[49]*C[14] + B[58]*C[15] + B[67]*C[16] + B[76]*C[17];
  A[14] -= B[5]*C[ 9] + B[14]*C[10] + B[23]*C[11] + B[32]*C[12] + B[41]*C[13] + B[50]*C[14] + B[59]*C[15] + B[68]*C[16] + B[77]*C[17];
  A[15] -= B[6]*C[ 9] + B[15]*C[10] + B[24]*C[11] + B[33]*C[12] + B[42]*C[13] + B[51]*C[14] + B[60]*C[15] + B[69]*C[16] + B[78]*C[17];
  A[16] -= B[7]*C[ 9] + B[16]*C[10] + B[25]*C[11] + B[34]*C[12] + B[43]*C[13] + B[52]*C[14] + B[61]*C[15] + B[70]*C[16] + B[79]*C[17];
  A[17] -= B[8]*C[ 9] + B[17]*C[10] + B[26]*C[11] + B[35]*C[12] + B[44]*C[13] + B[53]*C[14] + B[62]*C[15] + B[71]*C[16] + B[80]*C[17];

  A[18] -= B[0]*C[18] + B[ 9]*C[19] + B[18]*C[20] + B[27]*C[21] + B[36]*C[22] + B[45]*C[23] + B[54]*C[24] + B[63]*C[25] + B[72]*C[26];
  A[19] -= B[1]*C[18] + B[10]*C[19] + B[19]*C[20] + B[28]*C[21] + B[37]*C[22] + B[46]*C[23] + B[55]*C[24] + B[64]*C[25] + B[73]*C[26];
  A[20] -= B[2]*C[18] + B[11]*C[19] + B[20]*C[20] + B[29]*C[21] + B[38]*C[22] + B[47]*C[23] + B[56]*C[24] + B[65]*C[25] + B[74]*C[26];
  A[21] -= B[3]*C[18] + B[12]*C[19] + B[21]*C[20] + B[30]*C[21] + B[39]*C[22] + B[48]*C[23] + B[57]*C[24] + B[66]*C[25] + B[75]*C[26];
  A[22] -= B[4]*C[18] + B[13]*C[19] + B[22]*C[20] + B[31]*C[21] + B[40]*C[22] + B[49]*C[23] + B[58]*C[24] + B[67]*C[25] + B[76]*C[26];
  A[23] -= B[5]*C[18] + B[14]*C[19] + B[23]*C[20] + B[32]*C[21] + B[41]*C[22] + B[50]*C[23] + B[59]*C[24] + B[68]*C[25] + B[77]*C[26];
  A[24] -= B[6]*C[18] + B[15]*C[19] + B[24]*C[20] + B[33]*C[21] + B[42]*C[22] + B[51]*C[23] + B[60]*C[24] + B[69]*C[25] + B[78]*C[26];
  A[25] -= B[7]*C[18] + B[16]*C[19] + B[25]*C[20] + B[34]*C[21] + B[43]*C[22] + B[52]*C[23] + B[61]*C[24] + B[70]*C[25] + B[79]*C[26];
  A[26] -= B[8]*C[18] + B[17]*C[19] + B[26]*C[20] + B[35]*C[21] + B[44]*C[22] + B[53]*C[23] + B[62]*C[24] + B[71]*C[25] + B[80]*C[26];

  A[27] -= B[0]*C[27] + B[ 9]*C[28] + B[18]*C[29] + B[27]*C[30] + B[36]*C[31] + B[45]*C[32] + B[54]*C[33] + B[63]*C[34] + B[72]*C[35];
  A[28] -= B[1]*C[27] + B[10]*C[28] + B[19]*C[29] + B[28]*C[30] + B[37]*C[31] + B[46]*C[32] + B[55]*C[33] + B[64]*C[34] + B[73]*C[35];
  A[29] -= B[2]*C[27] + B[11]*C[28] + B[20]*C[29] + B[29]*C[30] + B[38]*C[31] + B[47]*C[32] + B[56]*C[33] + B[65]*C[34] + B[74]*C[35];
  A[30] -= B[3]*C[27] + B[12]*C[28] + B[21]*C[29] + B[30]*C[30] + B[39]*C[31] + B[48]*C[32] + B[57]*C[33] + B[66]*C[34] + B[75]*C[35];
  A[31] -= B[4]*C[27] + B[13]*C[28] + B[22]*C[29] + B[31]*C[30] + B[40]*C[31] + B[49]*C[32] + B[58]*C[33] + B[67]*C[34] + B[76]*C[35];
  A[32] -= B[5]*C[27] + B[14]*C[28] + B[23]*C[29] + B[32]*C[30] + B[41]*C[31] + B[50]*C[32] + B[59]*C[33] + B[68]*C[34] + B[77]*C[35];
  A[33] -= B[6]*C[27] + B[15]*C[28] + B[24]*C[29] + B[33]*C[30] + B[42]*C[31] + B[51]*C[32] + B[60]*C[33] + B[69]*C[34] + B[78]*C[35];
  A[34] -= B[7]*C[27] + B[16]*C[28] + B[25]*C[29] + B[34]*C[30] + B[43]*C[31] + B[52]*C[32] + B[61]*C[33] + B[70]*C[34] + B[79]*C[35];
  A[35] -= B[8]*C[27] + B[17]*C[28] + B[26]*C[29] + B[35]*C[30] + B[44]*C[31] + B[53]*C[32] + B[62]*C[33] + B[71]*C[34] + B[80]*C[35];

  A[36] -= B[0]*C[36] + B[ 9]*C[37] + B[18]*C[38] + B[27]*C[39] + B[36]*C[40] + B[45]*C[41] + B[54]*C[42] + B[63]*C[43] + B[72]*C[44];
  A[37] -= B[1]*C[36] + B[10]*C[37] + B[19]*C[38] + B[28]*C[39] + B[37]*C[40] + B[46]*C[41] + B[55]*C[42] + B[64]*C[43] + B[73]*C[44];
  A[38] -= B[2]*C[36] + B[11]*C[37] + B[20]*C[38] + B[29]*C[39] + B[38]*C[40] + B[47]*C[41] + B[56]*C[42] + B[65]*C[43] + B[74]*C[44];
  A[39] -= B[3]*C[36] + B[12]*C[37] + B[21]*C[38] + B[30]*C[39] + B[39]*C[40] + B[48]*C[41] + B[57]*C[42] + B[66]*C[43] + B[75]*C[44];
  A[40] -= B[4]*C[36] + B[13]*C[37] + B[22]*C[38] + B[31]*C[39] + B[40]*C[40] + B[49]*C[41] + B[58]*C[42] + B[67]*C[43] + B[76]*C[44];
  A[41] -= B[5]*C[36] + B[14]*C[37] + B[23]*C[38] + B[32]*C[39] + B[41]*C[40] + B[50]*C[41] + B[59]*C[42] + B[68]*C[43] + B[77]*C[44];
  A[42] -= B[6]*C[36] + B[15]*C[37] + B[24]*C[38] + B[33]*C[39] + B[42]*C[40] + B[51]*C[41] + B[60]*C[42] + B[69]*C[43] + B[78]*C[44];
  A[43] -= B[7]*C[36] + B[16]*C[37] + B[25]*C[38] + B[34]*C[39] + B[43]*C[40] + B[52]*C[41] + B[61]*C[42] + B[70]*C[43] + B[79]*C[44];
  A[44] -= B[8]*C[36] + B[17]*C[37] + B[26]*C[38] + B[35]*C[39] + B[44]*C[40] + B[53]*C[41] + B[62]*C[42] + B[71]*C[43] + B[80]*C[44];

  A[45] -= B[0]*C[45] + B[ 9]*C[46] + B[18]*C[47] + B[27]*C[48] + B[36]*C[49] + B[45]*C[50] + B[54]*C[51] + B[63]*C[52] + B[72]*C[53];
  A[46] -= B[1]*C[45] + B[10]*C[46] + B[19]*C[47] + B[28]*C[48] + B[37]*C[49] + B[46]*C[50] + B[55]*C[51] + B[64]*C[52] + B[73]*C[53];
  A[47] -= B[2]*C[45] + B[11]*C[46] + B[20]*C[47] + B[29]*C[48] + B[38]*C[49] + B[47]*C[50] + B[56]*C[51] + B[65]*C[52] + B[74]*C[53];
  A[48] -= B[3]*C[45] + B[12]*C[46] + B[21]*C[47] + B[30]*C[48] + B[39]*C[49] + B[48]*C[50] + B[57]*C[51] + B[66]*C[52] + B[75]*C[53];
  A[49] -= B[4]*C[45] + B[13]*C[46] + B[22]*C[47] + B[31]*C[48] + B[40]*C[49] + B[49]*C[50] + B[58]*C[51] + B[67]*C[52] + B[76]*C[53];
  A[50] -= B[5]*C[45] + B[14]*C[46] + B[23]*C[47] + B[32]*C[48] + B[41]*C[49] + B[50]*C[50] + B[59]*C[51] + B[68]*C[52] + B[77]*C[53];
  A[51] -= B[6]*C[45] + B[15]*C[46] + B[24]*C[47] + B[33]*C[48] + B[42]*C[49] + B[51]*C[50] + B[60]*C[51] + B[69]*C[52] + B[78]*C[53];
  A[52] -= B[7]*C[45] + B[16]*C[46] + B[25]*C[47] + B[34]*C[48] + B[43]*C[49] + B[52]*C[50] + B[61]*C[51] + B[70]*C[52] + B[79]*C[53];
  A[53] -= B[8]*C[45] + B[17]*C[46] + B[26]*C[47] + B[35]*C[48] + B[44]*C[49] + B[53]*C[50] + B[62]*C[51] + B[71]*C[52] + B[80]*C[53];

  A[54] -= B[0]*C[54] + B[ 9]*C[55] + B[18]*C[56] + B[27]*C[57] + B[36]*C[58] + B[45]*C[59] + B[54]*C[60] + B[63]*C[61] + B[72]*C[62];
  A[55] -= B[1]*C[54] + B[10]*C[55] + B[19]*C[56] + B[28]*C[57] + B[37]*C[58] + B[46]*C[59] + B[55]*C[60] + B[64]*C[61] + B[73]*C[62];
  A[56] -= B[2]*C[54] + B[11]*C[55] + B[20]*C[56] + B[29]*C[57] + B[38]*C[58] + B[47]*C[59] + B[56]*C[60] + B[65]*C[61] + B[74]*C[62];
  A[57] -= B[3]*C[54] + B[12]*C[55] + B[21]*C[56] + B[30]*C[57] + B[39]*C[58] + B[48]*C[59] + B[57]*C[60] + B[66]*C[61] + B[75]*C[62];
  A[58] -= B[4]*C[54] + B[13]*C[55] + B[22]*C[56] + B[31]*C[57] + B[40]*C[58] + B[49]*C[59] + B[58]*C[60] + B[67]*C[61] + B[76]*C[62];
  A[59] -= B[5]*C[54] + B[14]*C[55] + B[23]*C[56] + B[32]*C[57] + B[41]*C[58] + B[50]*C[59] + B[59]*C[60] + B[68]*C[61] + B[77]*C[62];
  A[60] -= B[6]*C[54] + B[15]*C[55] + B[24]*C[56] + B[33]*C[57] + B[42]*C[58] + B[51]*C[59] + B[60]*C[60] + B[69]*C[61] + B[78]*C[62];
  A[61] -= B[7]*C[54] + B[16]*C[55] + B[25]*C[56] + B[34]*C[57] + B[43]*C[58] + B[52]*C[59] + B[61]*C[60] + B[70]*C[61] + B[79]*C[62];
  A[62] -= B[8]*C[54] + B[17]*C[55] + B[26]*C[56] + B[35]*C[57] + B[44]*C[58] + B[53]*C[59] + B[62]*C[60] + B[71]*C[61] + B[80]*C[62];

  A[63] -= B[0]*C[63] + B[ 9]*C[64] + B[18]*C[65] + B[27]*C[66] + B[36]*C[67] + B[45]*C[68] + B[54]*C[69] + B[63]*C[70] + B[72]*C[71];
  A[64] -= B[1]*C[63] + B[10]*C[64] + B[19]*C[65] + B[28]*C[66] + B[37]*C[67] + B[46]*C[68] + B[55]*C[69] + B[64]*C[70] + B[73]*C[71];
  A[65] -= B[2]*C[63] + B[11]*C[64] + B[20]*C[65] + B[29]*C[66] + B[38]*C[67] + B[47]*C[68] + B[56]*C[69] + B[65]*C[70] + B[74]*C[71];
  A[66] -= B[3]*C[63] + B[12]*C[64] + B[21]*C[65] + B[30]*C[66] + B[39]*C[67] + B[48]*C[68] + B[57]*C[69] + B[66]*C[70] + B[75]*C[71];
  A[67] -= B[4]*C[63] + B[13]*C[64] + B[22]*C[65] + B[31]*C[66] + B[40]*C[67] + B[49]*C[68] + B[58]*C[69] + B[67]*C[70] + B[76]*C[71];
  A[68] -= B[5]*C[63] + B[14]*C[64] + B[23]*C[65] + B[32]*C[66] + B[41]*C[67] + B[50]*C[68] + B[59]*C[69] + B[68]*C[70] + B[77]*C[71];
  A[69] -= B[6]*C[63] + B[15]*C[64] + B[24]*C[65] + B[33]*C[66] + B[42]*C[67] + B[51]*C[68] + B[60]*C[69] + B[69]*C[70] + B[78]*C[71];
  A[70] -= B[7]*C[63] + B[16]*C[64] + B[25]*C[65] + B[34]*C[66] + B[43]*C[67] + B[52]*C[68] + B[61]*C[69] + B[70]*C[70] + B[79]*C[71];
  A[71] -= B[8]*C[63] + B[17]*C[64] + B[26]*C[65] + B[35]*C[66] + B[44]*C[67] + B[53]*C[68] + B[62]*C[69] + B[71]*C[70] + B[80]*C[71];

  A[72] -= B[0]*C[72] + B[ 9]*C[73] + B[18]*C[74] + B[27]*C[75] + B[36]*C[76] + B[45]*C[77] + B[54]*C[78] + B[63]*C[79] + B[72]*C[80];
  A[73] -= B[1]*C[72] + B[10]*C[73] + B[19]*C[74] + B[28]*C[75] + B[37]*C[76] + B[46]*C[77] + B[55]*C[78] + B[64]*C[79] + B[73]*C[80];
  A[74] -= B[2]*C[72] + B[11]*C[73] + B[20]*C[74] + B[29]*C[75] + B[38]*C[76] + B[47]*C[77] + B[56]*C[78] + B[65]*C[79] + B[74]*C[80];
  A[75] -= B[3]*C[72] + B[12]*C[73] + B[21]*C[74] + B[30]*C[75] + B[39]*C[76] + B[48]*C[77] + B[57]*C[78] + B[66]*C[79] + B[75]*C[80];
  A[76] -= B[4]*C[72] + B[13]*C[73] + B[22]*C[74] + B[31]*C[75] + B[40]*C[76] + B[49]*C[77] + B[58]*C[78] + B[67]*C[79] + B[76]*C[80];
  A[77] -= B[5]*C[72] + B[14]*C[73] + B[23]*C[74] + B[32]*C[75] + B[41]*C[76] + B[50]*C[77] + B[59]*C[78] + B[68]*C[79] + B[77]*C[80];
  A[78] -= B[6]*C[72] + B[15]*C[73] + B[24]*C[74] + B[33]*C[75] + B[42]*C[76] + B[51]*C[77] + B[60]*C[78] + B[69]*C[79] + B[78]*C[80];
  A[79] -= B[7]*C[72] + B[16]*C[73] + B[25]*C[74] + B[34]*C[75] + B[43]*C[76] + B[52]*C[77] + B[61]*C[78] + B[70]*C[79] + B[79]*C[80];
  A[80] -= B[8]*C[72] + B[17]*C[73] + B[26]*C[74] + B[35]*C[75] + B[44]*C[76] + B[53]*C[77] + B[62]*C[78] + B[71]*C[79] + B[80]*C[80];
  return 0;
}
#endif

/*
  PetscKernel_A_gets_A_times_B_11: A = A * B with size bs=11

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/
PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_11(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,121);CHKERRQ(ierr);
  A[0]  =  W[0]*B[0]   + W[11]*B[1]   + W[22]*B[2]   + W[33]*B[3]  + W[44]*B[4]  + W[55]*B[5]  + W[66]*B[6] + W[77]*B[7] + W[88]*B[8] + W[99]*B[9] + W[110]*B[10];
  A[1]  =  W[1]*B[0]   + W[12]*B[1]   + W[23]*B[2]   + W[34]*B[3]  + W[45]*B[4]  + W[56]*B[5]  + W[67]*B[6] + W[78]*B[7] + W[89]*B[8] + W[100]*B[9]+ W[111]*B[10];
  A[2]  =  W[2]*B[0]   + W[13]*B[1]   + W[24]*B[2]   + W[35]*B[3]  + W[46]*B[4]  + W[57]*B[5]  + W[68]*B[6] + W[79]*B[7] + W[90]*B[8] + W[101]*B[9]+ W[112]*B[10];
  A[3]  =  W[3]*B[0]   + W[14]*B[1]   + W[25]*B[2]   + W[36]*B[3]  + W[47]*B[4]  + W[58]*B[5]  + W[69]*B[6] + W[80]*B[7] + W[91]*B[8] + W[102]*B[9]+ W[113]*B[10];
  A[4]  =  W[4]*B[0]   + W[15]*B[1]   + W[26]*B[2]   + W[37]*B[3]  + W[48]*B[4]  + W[59]*B[5]  + W[70]*B[6] + W[81]*B[7] + W[92]*B[8] + W[103]*B[9]+ W[114]*B[10];
  A[5]  =  W[5]*B[0]   + W[16]*B[1]   + W[27]*B[2]   + W[38]*B[3]  + W[49]*B[4]  + W[60]*B[5]  + W[71]*B[6] + W[82]*B[7] + W[93]*B[8] + W[104]*B[9]+ W[115]*B[10];
  A[6]  =  W[6]*B[0]   + W[17]*B[1]   + W[28]*B[2]   + W[39]*B[3]  + W[50]*B[4]  + W[61]*B[5]  + W[72]*B[6] + W[83]*B[7] + W[94]*B[8] + W[105]*B[9]+ W[116]*B[10];
  A[7]  =  W[7]*B[0]   + W[18]*B[1]   + W[29]*B[2]   + W[40]*B[3]  + W[51]*B[4]  + W[62]*B[5]  + W[73]*B[6] + W[84]*B[7] + W[95]*B[8] + W[106]*B[9]+ W[117]*B[10];
  A[8]  =  W[8]*B[0]   + W[19]*B[1]   + W[30]*B[2]   + W[41]*B[3]  + W[52]*B[4]  + W[63]*B[5]  + W[74]*B[6] + W[85]*B[7] + W[96]*B[8] + W[107]*B[9]+ W[118]*B[10];
  A[9]  =  W[9]*B[0]   + W[20]*B[1]   + W[31]*B[2]   + W[42]*B[3]  + W[53]*B[4]  + W[64]*B[5]  + W[75]*B[6] + W[86]*B[7] + W[97]*B[8] + W[108]*B[9]+ W[119]*B[10];
  A[10] =  W[10]*B[0]  + W[21]*B[1]   + W[32]*B[2]   + W[43]*B[3]  + W[54]*B[4]  + W[65]*B[5]  + W[76]*B[6] + W[87]*B[7] + W[98]*B[8] + W[109]*B[9]+ W[120]*B[10];

  A[11]  =  W[0]*B[11]  + W[11]*B[12]   + W[22]*B[13]   + W[33]*B[14]  + W[44]*B[15]  + W[55]*B[16]  + W[66]*B[17] + W[77]*B[18] + W[88]*B[19] + W[99]*B[20] + W[110]*B[21];
  A[12]  =  W[1]*B[11]  + W[12]*B[12]   + W[23]*B[13]   + W[34]*B[14]  + W[45]*B[15]  + W[56]*B[16]  + W[67]*B[17] + W[78]*B[18] + W[89]*B[19] + W[100]*B[20]+ W[111]*B[21];
  A[13]  =  W[2]*B[11]  + W[13]*B[12]   + W[24]*B[13]   + W[35]*B[14]  + W[46]*B[15]  + W[57]*B[16]  + W[68]*B[17] + W[79]*B[18] + W[90]*B[19] + W[101]*B[20]+ W[112]*B[21];
  A[14]  =  W[3]*B[11]  + W[14]*B[12]   + W[25]*B[13]   + W[36]*B[14]  + W[47]*B[15]  + W[58]*B[16]  + W[69]*B[17] + W[80]*B[18] + W[91]*B[19] + W[102]*B[20]+ W[113]*B[21];
  A[15]  =  W[4]*B[11]  + W[15]*B[12]   + W[26]*B[13]   + W[37]*B[14]  + W[48]*B[15]  + W[59]*B[16]  + W[70]*B[17] + W[81]*B[18] + W[92]*B[19] + W[103]*B[20]+ W[114]*B[21];
  A[16]  =  W[5]*B[11]  + W[16]*B[12]   + W[27]*B[13]   + W[38]*B[14]  + W[49]*B[15]  + W[60]*B[16]  + W[71]*B[17] + W[82]*B[18] + W[93]*B[19] + W[104]*B[20]+ W[115]*B[21];
  A[17]  =  W[6]*B[11]  + W[17]*B[12]   + W[28]*B[13]   + W[39]*B[14]  + W[50]*B[15]  + W[61]*B[16]  + W[72]*B[17] + W[83]*B[18] + W[94]*B[19] + W[105]*B[20]+ W[116]*B[21];
  A[18]  =  W[7]*B[11]  + W[18]*B[12]   + W[29]*B[13]   + W[40]*B[14]  + W[51]*B[15]  + W[62]*B[16]  + W[73]*B[17] + W[84]*B[18] + W[95]*B[19] + W[106]*B[20]+ W[117]*B[21];
  A[19]  =  W[8]*B[11]  + W[19]*B[12]   + W[30]*B[13]   + W[41]*B[14]  + W[52]*B[15]  + W[63]*B[16]  + W[74]*B[17] + W[85]*B[18] + W[96]*B[19] + W[107]*B[20]+ W[118]*B[21];
  A[20]  =  W[9]*B[11]  + W[20]*B[12]   + W[31]*B[13]   + W[42]*B[14]  + W[53]*B[15]  + W[64]*B[16]  + W[75]*B[17] + W[86]*B[18] + W[97]*B[19] + W[108]*B[20]+ W[119]*B[21];
  A[21]  =  W[10]*B[11] + W[21]*B[12]   + W[32]*B[13]   + W[43]*B[14]  + W[54]*B[15]  + W[65]*B[16]  + W[76]*B[17] + W[87]*B[18] + W[98]*B[19] + W[109]*B[20]+ W[120]*B[21];

  A[22]  =  W[0]*B[22]  + W[11]*B[23]   + W[22]*B[24]   + W[33]*B[25]  + W[44]*B[26]  + W[55]*B[27]  + W[66]*B[28] + W[77]*B[29] + W[88]*B[30] + W[99]*B[31] + W[110]*B[32];
  A[23]  =  W[1]*B[22]  + W[12]*B[23]   + W[23]*B[24]   + W[34]*B[25]  + W[45]*B[26]  + W[56]*B[27]  + W[67]*B[28] + W[78]*B[29] + W[89]*B[30] + W[100]*B[31]+ W[111]*B[32];
  A[24]  =  W[2]*B[22]  + W[13]*B[23]   + W[24]*B[24]   + W[35]*B[25]  + W[46]*B[26]  + W[57]*B[27]  + W[68]*B[28] + W[79]*B[29] + W[90]*B[30] + W[101]*B[31]+ W[112]*B[32];
  A[25]  =  W[3]*B[22]  + W[14]*B[23]   + W[25]*B[24]   + W[36]*B[25]  + W[47]*B[26]  + W[58]*B[27]  + W[69]*B[28] + W[80]*B[29] + W[91]*B[30] + W[102]*B[31]+ W[113]*B[32];
  A[26]  =  W[4]*B[22]  + W[15]*B[23]   + W[26]*B[24]   + W[37]*B[25]  + W[48]*B[26]  + W[59]*B[27]  + W[70]*B[28] + W[81]*B[29] + W[92]*B[30] + W[103]*B[31]+ W[114]*B[32];
  A[27]  =  W[5]*B[22]  + W[16]*B[23]   + W[27]*B[24]   + W[38]*B[25]  + W[49]*B[26]  + W[60]*B[27]  + W[71]*B[28] + W[82]*B[29] + W[93]*B[30] + W[104]*B[31]+ W[115]*B[32];
  A[28]  =  W[6]*B[22]  + W[17]*B[23]   + W[28]*B[24]   + W[39]*B[25]  + W[50]*B[26]  + W[61]*B[27]  + W[72]*B[28] + W[83]*B[29] + W[94]*B[30] + W[105]*B[31]+ W[116]*B[32];
  A[29]  =  W[7]*B[22]  + W[18]*B[23]   + W[29]*B[24]   + W[40]*B[25]  + W[51]*B[26]  + W[62]*B[27]  + W[73]*B[28] + W[84]*B[29] + W[95]*B[30] + W[106]*B[31]+ W[117]*B[32];
  A[30]  =  W[8]*B[22]  + W[19]*B[23]   + W[30]*B[24]   + W[41]*B[25]  + W[52]*B[26]  + W[63]*B[27]  + W[74]*B[28] + W[85]*B[29] + W[96]*B[30] + W[107]*B[31]+ W[118]*B[32];
  A[31]  =  W[9]*B[22]  + W[20]*B[23]   + W[31]*B[24]   + W[42]*B[25]  + W[53]*B[26]  + W[64]*B[27]  + W[75]*B[28] + W[86]*B[29] + W[97]*B[30] + W[108]*B[31]+ W[119]*B[32];
  A[32]  =  W[10]*B[22] + W[21]*B[23]   + W[32]*B[24]   + W[43]*B[25]  + W[54]*B[26]  + W[65]*B[27]  + W[76]*B[28] + W[87]*B[29] + W[98]*B[30] + W[109]*B[31]+ W[120]*B[32];

  A[33]  =  W[0]*B[33]  + W[11]*B[34]   + W[22]*B[35]   + W[33]*B[36]  + W[44]*B[37]  + W[55]*B[38]  + W[66]*B[39] + W[77]*B[40] + W[88]*B[41] + W[99]*B[42] + W[110]*B[43];
  A[34]  =  W[1]*B[33]  + W[12]*B[34]   + W[23]*B[35]   + W[34]*B[36]  + W[45]*B[37]  + W[56]*B[38]  + W[67]*B[39] + W[78]*B[40] + W[89]*B[41] + W[100]*B[42]+ W[111]*B[43];
  A[35]  =  W[2]*B[33]  + W[13]*B[34]   + W[24]*B[35]   + W[35]*B[36]  + W[46]*B[37]  + W[57]*B[38]  + W[68]*B[39] + W[79]*B[40] + W[90]*B[41] + W[101]*B[42]+ W[112]*B[43];
  A[36]  =  W[3]*B[33]  + W[14]*B[34]   + W[25]*B[35]   + W[36]*B[36]  + W[47]*B[37]  + W[58]*B[38]  + W[69]*B[39] + W[80]*B[40] + W[91]*B[41] + W[102]*B[42]+ W[113]*B[43];
  A[37]  =  W[4]*B[33]  + W[15]*B[34]   + W[26]*B[35]   + W[37]*B[36]  + W[48]*B[37]  + W[59]*B[38]  + W[70]*B[39] + W[81]*B[40] + W[92]*B[41] + W[103]*B[42]+ W[114]*B[43];
  A[38]  =  W[5]*B[33]  + W[16]*B[34]   + W[27]*B[35]   + W[38]*B[36]  + W[49]*B[37]  + W[60]*B[38]  + W[71]*B[39] + W[82]*B[40] + W[93]*B[41] + W[104]*B[42]+ W[115]*B[43];
  A[39]  =  W[6]*B[33]  + W[17]*B[34]   + W[28]*B[35]   + W[39]*B[36]  + W[50]*B[37]  + W[61]*B[38]  + W[72]*B[39] + W[83]*B[40] + W[94]*B[41] + W[105]*B[42]+ W[116]*B[43];
  A[40]  =  W[7]*B[33]  + W[18]*B[34]   + W[29]*B[35]   + W[40]*B[36]  + W[51]*B[37]  + W[62]*B[38]  + W[73]*B[39] + W[84]*B[40] + W[95]*B[41] + W[106]*B[42]+ W[117]*B[43];
  A[41]  =  W[8]*B[33]  + W[19]*B[34]   + W[30]*B[35]   + W[41]*B[36]  + W[52]*B[37]  + W[63]*B[38]  + W[74]*B[39] + W[85]*B[40] + W[96]*B[41] + W[107]*B[42]+ W[118]*B[43];
  A[42]  =  W[9]*B[33]  + W[20]*B[34]   + W[31]*B[35]   + W[42]*B[36]  + W[53]*B[37]  + W[64]*B[38]  + W[75]*B[39] + W[86]*B[40] + W[97]*B[41] + W[108]*B[42]+ W[119]*B[43];
  A[43]  =  W[10]*B[33] + W[21]*B[34]   + W[32]*B[35]   + W[43]*B[36]  + W[54]*B[37]  + W[65]*B[38]  + W[76]*B[39] + W[87]*B[40] + W[98]*B[41] + W[109]*B[42]+ W[120]*B[43];

  A[44]  =  W[0]*B[44]  + W[11]*B[45]   + W[22]*B[46]   + W[33]*B[47]  + W[44]*B[48]  + W[55]*B[49]  + W[66]*B[50] + W[77]*B[51] + W[88]*B[52] + W[99]*B[53] + W[110]*B[54];
  A[45]  =  W[1]*B[44]  + W[12]*B[45]   + W[23]*B[46]   + W[34]*B[47]  + W[45]*B[48]  + W[56]*B[49]  + W[67]*B[50] + W[78]*B[51] + W[89]*B[52] + W[100]*B[53]+ W[111]*B[54];
  A[46]  =  W[2]*B[44]  + W[13]*B[45]   + W[24]*B[46]   + W[35]*B[47]  + W[46]*B[48]  + W[57]*B[49]  + W[68]*B[50] + W[79]*B[51] + W[90]*B[52] + W[101]*B[53]+ W[112]*B[54];
  A[47]  =  W[3]*B[44]  + W[14]*B[45]   + W[25]*B[46]   + W[36]*B[47]  + W[47]*B[48]  + W[58]*B[49]  + W[69]*B[50] + W[80]*B[51] + W[91]*B[52] + W[102]*B[53]+ W[113]*B[54];
  A[48]  =  W[4]*B[44]  + W[15]*B[45]   + W[26]*B[46]   + W[37]*B[47]  + W[48]*B[48]  + W[59]*B[49]  + W[70]*B[50] + W[81]*B[51] + W[92]*B[52] + W[103]*B[53]+ W[114]*B[54];
  A[49]  =  W[5]*B[44]  + W[16]*B[45]   + W[27]*B[46]   + W[38]*B[47]  + W[49]*B[48]  + W[60]*B[49]  + W[71]*B[50] + W[82]*B[51] + W[93]*B[52] + W[104]*B[53]+ W[115]*B[54];
  A[50]  =  W[6]*B[44]  + W[17]*B[45]   + W[28]*B[46]   + W[39]*B[47]  + W[50]*B[48]  + W[61]*B[49]  + W[72]*B[50] + W[83]*B[51] + W[94]*B[52] + W[105]*B[53]+ W[116]*B[54];
  A[51]  =  W[7]*B[44]  + W[18]*B[45]   + W[29]*B[46]   + W[40]*B[47]  + W[51]*B[48]  + W[62]*B[49]  + W[73]*B[50] + W[84]*B[51] + W[95]*B[52] + W[106]*B[53]+ W[117]*B[54];
  A[52]  =  W[8]*B[44]  + W[19]*B[45]   + W[30]*B[46]   + W[41]*B[47]  + W[52]*B[48]  + W[63]*B[49]  + W[74]*B[50] + W[85]*B[51] + W[96]*B[52] + W[107]*B[53]+ W[118]*B[54];
  A[53]  =  W[9]*B[44]  + W[20]*B[45]   + W[31]*B[46]   + W[42]*B[47]  + W[53]*B[48]  + W[64]*B[49]  + W[75]*B[50] + W[86]*B[51] + W[97]*B[52] + W[108]*B[53]+ W[119]*B[54];
  A[54]  =  W[10]*B[44] + W[21]*B[45]   + W[32]*B[46]   + W[43]*B[47]  + W[54]*B[48]  + W[65]*B[49]  + W[76]*B[50] + W[87]*B[51] + W[98]*B[52] + W[109]*B[53]+ W[120]*B[54];

  A[55]  =  W[0]*B[55]  + W[11]*B[56]   + W[22]*B[57]   + W[33]*B[58]  + W[44]*B[59]  + W[55]*B[60]  + W[66]*B[61] + W[77]*B[62] + W[88]*B[63] + W[99]*B[64] + W[110]*B[65];
  A[56]  =  W[1]*B[55]  + W[12]*B[56]   + W[23]*B[57]   + W[34]*B[58]  + W[45]*B[59]  + W[56]*B[60]  + W[67]*B[61] + W[78]*B[62] + W[89]*B[63] + W[100]*B[64]+ W[111]*B[65];
  A[57]  =  W[2]*B[55]  + W[13]*B[56]   + W[24]*B[57]   + W[35]*B[58]  + W[46]*B[59]  + W[57]*B[60]  + W[68]*B[61] + W[79]*B[62] + W[90]*B[63] + W[101]*B[64]+ W[112]*B[65];
  A[58]  =  W[3]*B[55]  + W[14]*B[56]   + W[25]*B[57]   + W[36]*B[58]  + W[47]*B[59]  + W[58]*B[60]  + W[69]*B[61] + W[80]*B[62] + W[91]*B[63] + W[102]*B[64]+ W[113]*B[65];
  A[59]  =  W[4]*B[55]  + W[15]*B[56]   + W[26]*B[57]   + W[37]*B[58]  + W[48]*B[59]  + W[59]*B[60]  + W[70]*B[61] + W[81]*B[62] + W[92]*B[63] + W[103]*B[64]+ W[114]*B[65];
  A[60]  =  W[5]*B[55]  + W[16]*B[56]   + W[27]*B[57]   + W[38]*B[58]  + W[49]*B[59]  + W[60]*B[60]  + W[71]*B[61] + W[82]*B[62] + W[93]*B[63] + W[104]*B[64]+ W[115]*B[65];
  A[61]  =  W[6]*B[55]  + W[17]*B[56]   + W[28]*B[57]   + W[39]*B[58]  + W[50]*B[59]  + W[61]*B[60]  + W[72]*B[61] + W[83]*B[62] + W[94]*B[63] + W[105]*B[64]+ W[116]*B[65];
  A[62]  =  W[7]*B[55]  + W[18]*B[56]   + W[29]*B[57]   + W[40]*B[58]  + W[51]*B[59]  + W[62]*B[60]  + W[73]*B[61] + W[84]*B[62] + W[95]*B[63] + W[106]*B[64]+ W[117]*B[65];
  A[63]  =  W[8]*B[55]  + W[19]*B[56]   + W[30]*B[57]   + W[41]*B[58]  + W[52]*B[59]  + W[63]*B[60]  + W[74]*B[61] + W[85]*B[62] + W[96]*B[63] + W[107]*B[64]+ W[118]*B[65];
  A[64]  =  W[9]*B[55]  + W[20]*B[56]   + W[31]*B[57]   + W[42]*B[58]  + W[53]*B[59]  + W[64]*B[60]  + W[75]*B[61] + W[86]*B[62] + W[97]*B[63] + W[108]*B[64]+ W[119]*B[65];
  A[65]  =  W[10]*B[55] + W[21]*B[56]   + W[32]*B[57]   + W[43]*B[58]  + W[54]*B[59]  + W[65]*B[60]  + W[76]*B[61] + W[87]*B[62] + W[98]*B[63] + W[109]*B[64]+ W[120]*B[65];

  A[66]  =  W[0]*B[66]  + W[11]*B[67]   + W[22]*B[68]   + W[33]*B[69]  + W[44]*B[70]  + W[55]*B[71]  + W[66]*B[72] + W[77]*B[73] + W[88]*B[74] + W[99]*B[75] + W[110]*B[76];
  A[67]  =  W[1]*B[66]  + W[12]*B[67]   + W[23]*B[68]   + W[34]*B[69]  + W[45]*B[70]  + W[56]*B[71]  + W[67]*B[72] + W[78]*B[73] + W[89]*B[74] + W[100]*B[75]+ W[111]*B[76];
  A[68]  =  W[2]*B[66]  + W[13]*B[67]   + W[24]*B[68]   + W[35]*B[69]  + W[46]*B[70]  + W[57]*B[71]  + W[68]*B[72] + W[79]*B[73] + W[90]*B[74] + W[101]*B[75]+ W[112]*B[76];
  A[69]  =  W[3]*B[66]  + W[14]*B[67]   + W[25]*B[68]   + W[36]*B[69]  + W[47]*B[70]  + W[58]*B[71]  + W[69]*B[72] + W[80]*B[73] + W[91]*B[74] + W[102]*B[75]+ W[113]*B[76];
  A[70]  =  W[4]*B[66]  + W[15]*B[67]   + W[26]*B[68]   + W[37]*B[69]  + W[48]*B[70]  + W[59]*B[71]  + W[70]*B[72] + W[81]*B[73] + W[92]*B[74] + W[103]*B[75]+ W[114]*B[76];
  A[71]  =  W[5]*B[66]  + W[16]*B[67]   + W[27]*B[68]   + W[38]*B[69]  + W[49]*B[70]  + W[60]*B[71]  + W[71]*B[72] + W[82]*B[73] + W[93]*B[74] + W[104]*B[75]+ W[115]*B[76];
  A[72]  =  W[6]*B[66]  + W[17]*B[67]   + W[28]*B[68]   + W[39]*B[69]  + W[50]*B[70]  + W[61]*B[71]  + W[72]*B[72] + W[83]*B[73] + W[94]*B[74] + W[105]*B[75]+ W[116]*B[76];
  A[73]  =  W[7]*B[66]  + W[18]*B[67]   + W[29]*B[68]   + W[40]*B[69]  + W[51]*B[70]  + W[62]*B[71]  + W[73]*B[72] + W[84]*B[73] + W[95]*B[74] + W[106]*B[75]+ W[117]*B[76];
  A[74]  =  W[8]*B[66]  + W[19]*B[67]   + W[30]*B[68]   + W[41]*B[69]  + W[52]*B[70]  + W[63]*B[71]  + W[74]*B[72] + W[85]*B[73] + W[96]*B[74] + W[107]*B[75]+ W[118]*B[76];
  A[75]  =  W[9]*B[66]  + W[20]*B[67]   + W[31]*B[68]   + W[42]*B[69]  + W[53]*B[70]  + W[64]*B[71]  + W[75]*B[72] + W[86]*B[73] + W[97]*B[74] + W[108]*B[75]+ W[119]*B[76];
  A[76]  =  W[10]*B[66] + W[21]*B[67]   + W[32]*B[68]   + W[43]*B[69]  + W[54]*B[70]  + W[65]*B[71]  + W[76]*B[72] + W[87]*B[73] + W[98]*B[74] + W[109]*B[75]+ W[120]*B[76];

  A[77]  =  W[0]*B[77]  + W[11]*B[78]   + W[22]*B[79]   + W[33]*B[80]  + W[44]*B[81]  + W[55]*B[82]  + W[66]*B[83] + W[77]*B[84] + W[88]*B[85] + W[99]*B[86] + W[110]*B[87];
  A[78]  =  W[1]*B[77]  + W[12]*B[78]   + W[23]*B[79]   + W[34]*B[80]  + W[45]*B[81]  + W[56]*B[82]  + W[67]*B[83] + W[78]*B[84] + W[89]*B[85] + W[100]*B[86]+ W[111]*B[87];
  A[79]  =  W[2]*B[77]  + W[13]*B[78]   + W[24]*B[79]   + W[35]*B[80]  + W[46]*B[81]  + W[57]*B[82]  + W[68]*B[83] + W[79]*B[84] + W[90]*B[85] + W[101]*B[86]+ W[112]*B[87];
  A[80]  =  W[3]*B[77]  + W[14]*B[78]   + W[25]*B[79]   + W[36]*B[80]  + W[47]*B[81]  + W[58]*B[82]  + W[69]*B[83] + W[80]*B[84] + W[91]*B[85] + W[102]*B[86]+ W[113]*B[87];
  A[81]  =  W[4]*B[77]  + W[15]*B[78]   + W[26]*B[79]   + W[37]*B[80]  + W[48]*B[81]  + W[59]*B[82]  + W[70]*B[83] + W[81]*B[84] + W[92]*B[85] + W[103]*B[86]+ W[114]*B[87];
  A[82]  =  W[5]*B[77]  + W[16]*B[78]   + W[27]*B[79]   + W[38]*B[80]  + W[49]*B[81]  + W[60]*B[82]  + W[71]*B[83] + W[82]*B[84] + W[93]*B[85] + W[104]*B[86]+ W[115]*B[87];
  A[83]  =  W[6]*B[77]  + W[17]*B[78]   + W[28]*B[79]   + W[39]*B[80]  + W[50]*B[81]  + W[61]*B[82]  + W[72]*B[83] + W[83]*B[84] + W[94]*B[85] + W[105]*B[86]+ W[116]*B[87];
  A[84]  =  W[7]*B[77]  + W[18]*B[78]   + W[29]*B[79]   + W[40]*B[80]  + W[51]*B[81]  + W[62]*B[82]  + W[73]*B[83] + W[84]*B[84] + W[95]*B[85] + W[106]*B[86]+ W[117]*B[87];
  A[85]  =  W[8]*B[77]  + W[19]*B[78]   + W[30]*B[79]   + W[41]*B[80]  + W[52]*B[81]  + W[63]*B[82]  + W[74]*B[83] + W[85]*B[84] + W[96]*B[85] + W[107]*B[86]+ W[118]*B[87];
  A[86]  =  W[9]*B[77]  + W[20]*B[78]   + W[31]*B[79]   + W[42]*B[80]  + W[53]*B[81]  + W[64]*B[82]  + W[75]*B[83] + W[86]*B[84] + W[97]*B[85] + W[108]*B[86]+ W[119]*B[87];
  A[87]  =  W[10]*B[77] + W[21]*B[78]   + W[32]*B[79]   + W[43]*B[80]  + W[54]*B[81]  + W[65]*B[82]  + W[76]*B[83] + W[87]*B[84] + W[98]*B[85] + W[109]*B[86]+ W[120]*B[87];

  A[88]  =  W[0]*B[88]  + W[11]*B[89]   + W[22]*B[90]   + W[33]*B[91]  + W[44]*B[92]  + W[55]*B[93]  + W[66]*B[94] + W[77]*B[95] + W[88]*B[96] + W[99]*B[97] + W[110]*B[98];
  A[89]  =  W[1]*B[88]  + W[12]*B[89]   + W[23]*B[90]   + W[34]*B[91]  + W[45]*B[92]  + W[56]*B[93]  + W[67]*B[94] + W[78]*B[95] + W[89]*B[96] + W[100]*B[97]+ W[111]*B[98];
  A[90]  =  W[2]*B[88]  + W[13]*B[89]   + W[24]*B[90]   + W[35]*B[91]  + W[46]*B[92]  + W[57]*B[93]  + W[68]*B[94] + W[79]*B[95] + W[90]*B[96] + W[101]*B[97]+ W[112]*B[98];
  A[91]  =  W[3]*B[88]  + W[14]*B[89]   + W[25]*B[90]   + W[36]*B[91]  + W[47]*B[92]  + W[58]*B[93]  + W[69]*B[94] + W[80]*B[95] + W[91]*B[96] + W[102]*B[97]+ W[113]*B[98];
  A[92]  =  W[4]*B[88]  + W[15]*B[89]   + W[26]*B[90]   + W[37]*B[91]  + W[48]*B[92]  + W[59]*B[93]  + W[70]*B[94] + W[81]*B[95] + W[92]*B[96] + W[103]*B[97]+ W[114]*B[98];
  A[93]  =  W[5]*B[88]  + W[16]*B[89]   + W[27]*B[90]   + W[38]*B[91]  + W[49]*B[92]  + W[60]*B[93]  + W[71]*B[94] + W[82]*B[95] + W[93]*B[96] + W[104]*B[97]+ W[115]*B[98];
  A[94]  =  W[6]*B[88]  + W[17]*B[89]   + W[28]*B[90]   + W[39]*B[91]  + W[50]*B[92]  + W[61]*B[93]  + W[72]*B[94] + W[83]*B[95] + W[94]*B[96] + W[105]*B[97]+ W[116]*B[98];
  A[95]  =  W[7]*B[88]  + W[18]*B[89]   + W[29]*B[90]   + W[40]*B[91]  + W[51]*B[92]  + W[62]*B[93]  + W[73]*B[94] + W[84]*B[95] + W[95]*B[96] + W[106]*B[97]+ W[117]*B[98];
  A[96]  =  W[8]*B[88]  + W[19]*B[89]   + W[30]*B[90]   + W[41]*B[91]  + W[52]*B[92]  + W[63]*B[93]  + W[74]*B[94] + W[85]*B[95] + W[96]*B[96] + W[107]*B[97]+ W[118]*B[98];
  A[97]  =  W[9]*B[88]  + W[20]*B[89]   + W[31]*B[90]   + W[42]*B[91]  + W[53]*B[92]  + W[64]*B[93]  + W[75]*B[94] + W[86]*B[95] + W[97]*B[96] + W[108]*B[97]+ W[119]*B[98];
  A[98]  =  W[10]*B[88] + W[21]*B[89]   + W[32]*B[90]   + W[43]*B[91]  + W[54]*B[92]  + W[65]*B[93]  + W[76]*B[94] + W[87]*B[95] + W[98]*B[96] + W[109]*B[97]+ W[120]*B[98];

  A[99]  = W[0]*B[99]  + W[11]*B[100] + W[22]*B[101] + W[33]*B[102]  + W[44]*B[103] + W[55]*B[104] + W[66]*B[105] + W[77]*B[106] + W[88]*B[107] + W[99]*B[108] + W[110]*B[109];
  A[100] = W[1]*B[99]  + W[12]*B[100] + W[23]*B[101] + W[34]*B[102]  + W[45]*B[103] + W[56]*B[104] + W[67]*B[105] + W[78]*B[106] + W[89]*B[107] + W[100]*B[108]+ W[111]*B[109];
  A[101] = W[2]*B[99]  + W[13]*B[100] + W[24]*B[101] + W[35]*B[102]  + W[46]*B[103] + W[57]*B[104] + W[68]*B[105] + W[79]*B[106] + W[90]*B[107] + W[101]*B[108]+ W[112]*B[109];
  A[102] = W[3]*B[99]  + W[14]*B[100] + W[25]*B[101] + W[36]*B[102]  + W[47]*B[103] + W[58]*B[104] + W[69]*B[105] + W[80]*B[106] + W[91]*B[107] + W[102]*B[108]+ W[113]*B[109];
  A[103] = W[4]*B[99]  + W[15]*B[100] + W[26]*B[101] + W[37]*B[102]  + W[48]*B[103] + W[59]*B[104] + W[70]*B[105] + W[81]*B[106] + W[92]*B[107] + W[103]*B[108]+ W[114]*B[109];
  A[104] = W[5]*B[99]  + W[16]*B[100] + W[27]*B[101] + W[38]*B[102]  + W[49]*B[103] + W[60]*B[104] + W[71]*B[105] + W[82]*B[106] + W[93]*B[107] + W[104]*B[108]+ W[115]*B[109];
  A[105] = W[6]*B[99]  + W[17]*B[100] + W[28]*B[101] + W[39]*B[102]  + W[50]*B[103] + W[61]*B[104] + W[72]*B[105] + W[83]*B[106] + W[94]*B[107] + W[105]*B[108]+ W[116]*B[109];
  A[106] = W[7]*B[99]  + W[18]*B[100] + W[29]*B[101] + W[40]*B[102]  + W[51]*B[103] + W[62]*B[104] + W[73]*B[105] + W[84]*B[106] + W[95]*B[107] + W[106]*B[108]+ W[117]*B[109];
  A[107] = W[8]*B[99]  + W[19]*B[100] + W[30]*B[101] + W[41]*B[102]  + W[52]*B[103] + W[63]*B[104] + W[74]*B[105] + W[85]*B[106] + W[96]*B[107] + W[107]*B[108]+ W[118]*B[109];
  A[108] = W[9]*B[99]  + W[20]*B[100] + W[31]*B[101] + W[42]*B[102]  + W[53]*B[103] + W[64]*B[104] + W[75]*B[105] + W[86]*B[106] + W[97]*B[107] + W[108]*B[108]+ W[119]*B[109];
  A[109] = W[10]*B[99] + W[21]*B[100] + W[32]*B[101] + W[43]*B[102]  + W[54]*B[103] + W[65]*B[104] + W[76]*B[105] + W[87]*B[106] + W[98]*B[107] + W[109]*B[108]+ W[120]*B[109];

  A[110] = W[0]*B[110]  + W[11]*B[111] + W[22]*B[112] + W[33]*B[113]  + W[44]*B[114] + W[55]*B[115] + W[66]*B[116] + W[77]*B[117] + W[88]*B[118] + W[99]*B[119] + W[110]*B[120];
  A[111] = W[1]*B[110]  + W[12]*B[111] + W[23]*B[112] + W[34]*B[113]  + W[45]*B[114] + W[56]*B[115] + W[67]*B[116] + W[78]*B[117] + W[89]*B[118] + W[100]*B[119]+ W[111]*B[120];
  A[112] = W[2]*B[110]  + W[13]*B[111] + W[24]*B[112] + W[35]*B[113]  + W[46]*B[114] + W[57]*B[115] + W[68]*B[116] + W[79]*B[117] + W[90]*B[118] + W[101]*B[119]+ W[112]*B[120];
  A[113] = W[3]*B[110]  + W[14]*B[111] + W[25]*B[112] + W[36]*B[113]  + W[47]*B[114] + W[58]*B[115] + W[69]*B[116] + W[80]*B[117] + W[91]*B[118] + W[102]*B[119]+ W[113]*B[120];
  A[114] = W[4]*B[110]  + W[15]*B[111] + W[26]*B[112] + W[37]*B[113]  + W[48]*B[114] + W[59]*B[115] + W[70]*B[116] + W[81]*B[117] + W[92]*B[118] + W[103]*B[119]+ W[114]*B[120];
  A[115] = W[5]*B[110]  + W[16]*B[111] + W[27]*B[112] + W[38]*B[113]  + W[49]*B[114] + W[60]*B[115] + W[71]*B[116] + W[82]*B[117] + W[93]*B[118] + W[104]*B[119]+ W[115]*B[120];
  A[116] = W[6]*B[110]  + W[17]*B[111] + W[28]*B[112] + W[39]*B[113]  + W[50]*B[114] + W[61]*B[115] + W[72]*B[116] + W[83]*B[117] + W[94]*B[118] + W[105]*B[119]+ W[116]*B[120];
  A[117] = W[7]*B[110]  + W[18]*B[111] + W[29]*B[112] + W[40]*B[113]  + W[51]*B[114] + W[62]*B[115] + W[73]*B[116] + W[84]*B[117] + W[95]*B[118] + W[106]*B[119]+ W[117]*B[120];
  A[118] = W[8]*B[110]  + W[19]*B[111] + W[30]*B[112] + W[41]*B[113]  + W[52]*B[114] + W[63]*B[115] + W[74]*B[116] + W[85]*B[117] + W[96]*B[118] + W[107]*B[119]+ W[118]*B[120];
  A[119] = W[9]*B[110]  + W[20]*B[111] + W[31]*B[112] + W[42]*B[113]  + W[53]*B[114] + W[64]*B[115] + W[75]*B[116] + W[86]*B[117] + W[97]*B[118] + W[108]*B[119]+ W[119]*B[120];
  A[120] = W[10]*B[110] + W[21]*B[111] + W[32]*B[112] + W[43]*B[113]  + W[54]*B[114] + W[65]*B[115] + W[76]*B[116] + W[87]*B[117] + W[98]*B[118] + W[109]*B[119]+ W[120]*B[120];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_11: A = A - W * B with size bs=11

  Input Parameters:
+  A,W,B - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - W * B
*/
PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_11(PetscScalar *A,const PetscScalar *W,const PetscScalar *B)
{
  A[0]  -=  W[0]*B[0]   + W[11]*B[1]   + W[22]*B[2]   + W[33]*B[3]  + W[44]*B[4]  + W[55]*B[5]  + W[66]*B[6] + W[77]*B[7] + W[88]*B[8] + W[99]*B[9] + W[110]*B[10];
  A[1]  -=  W[1]*B[0]   + W[12]*B[1]   + W[23]*B[2]   + W[34]*B[3]  + W[45]*B[4]  + W[56]*B[5]  + W[67]*B[6] + W[78]*B[7] + W[89]*B[8] + W[100]*B[9]+ W[111]*B[10];
  A[2]  -=  W[2]*B[0]   + W[13]*B[1]   + W[24]*B[2]   + W[35]*B[3]  + W[46]*B[4]  + W[57]*B[5]  + W[68]*B[6] + W[79]*B[7] + W[90]*B[8] + W[101]*B[9]+ W[112]*B[10];
  A[3]  -=  W[3]*B[0]   + W[14]*B[1]   + W[25]*B[2]   + W[36]*B[3]  + W[47]*B[4]  + W[58]*B[5]  + W[69]*B[6] + W[80]*B[7] + W[91]*B[8] + W[102]*B[9]+ W[113]*B[10];
  A[4]  -=  W[4]*B[0]   + W[15]*B[1]   + W[26]*B[2]   + W[37]*B[3]  + W[48]*B[4]  + W[59]*B[5]  + W[70]*B[6] + W[81]*B[7] + W[92]*B[8] + W[103]*B[9]+ W[114]*B[10];
  A[5]  -=  W[5]*B[0]   + W[16]*B[1]   + W[27]*B[2]   + W[38]*B[3]  + W[49]*B[4]  + W[60]*B[5]  + W[71]*B[6] + W[82]*B[7] + W[93]*B[8] + W[104]*B[9]+ W[115]*B[10];
  A[6]  -=  W[6]*B[0]   + W[17]*B[1]   + W[28]*B[2]   + W[39]*B[3]  + W[50]*B[4]  + W[61]*B[5]  + W[72]*B[6] + W[83]*B[7] + W[94]*B[8] + W[105]*B[9]+ W[116]*B[10];
  A[7]  -=  W[7]*B[0]   + W[18]*B[1]   + W[29]*B[2]   + W[40]*B[3]  + W[51]*B[4]  + W[62]*B[5]  + W[73]*B[6] + W[84]*B[7] + W[95]*B[8] + W[106]*B[9]+ W[117]*B[10];
  A[8]  -=  W[8]*B[0]   + W[19]*B[1]   + W[30]*B[2]   + W[41]*B[3]  + W[52]*B[4]  + W[63]*B[5]  + W[74]*B[6] + W[85]*B[7] + W[96]*B[8] + W[107]*B[9]+ W[118]*B[10];
  A[9]  -=  W[9]*B[0]   + W[20]*B[1]   + W[31]*B[2]   + W[42]*B[3]  + W[53]*B[4]  + W[64]*B[5]  + W[75]*B[6] + W[86]*B[7] + W[97]*B[8] + W[108]*B[9]+ W[119]*B[10];
  A[10] -=  W[10]*B[0]  + W[21]*B[1]   + W[32]*B[2]   + W[43]*B[3]  + W[54]*B[4]  + W[65]*B[5]  + W[76]*B[6] + W[87]*B[7] + W[98]*B[8] + W[109]*B[9]+ W[120]*B[10];

  A[11]  -=  W[0]*B[11]  + W[11]*B[12]   + W[22]*B[13]   + W[33]*B[14]  + W[44]*B[15]  + W[55]*B[16]  + W[66]*B[17] + W[77]*B[18] + W[88]*B[19] + W[99]*B[20] + W[110]*B[21];
  A[12]  -=  W[1]*B[11]  + W[12]*B[12]   + W[23]*B[13]   + W[34]*B[14]  + W[45]*B[15]  + W[56]*B[16]  + W[67]*B[17] + W[78]*B[18] + W[89]*B[19] + W[100]*B[20]+ W[111]*B[21];
  A[13]  -=  W[2]*B[11]  + W[13]*B[12]   + W[24]*B[13]   + W[35]*B[14]  + W[46]*B[15]  + W[57]*B[16]  + W[68]*B[17] + W[79]*B[18] + W[90]*B[19] + W[101]*B[20]+ W[112]*B[21];
  A[14]  -=  W[3]*B[11]  + W[14]*B[12]   + W[25]*B[13]   + W[36]*B[14]  + W[47]*B[15]  + W[58]*B[16]  + W[69]*B[17] + W[80]*B[18] + W[91]*B[19] + W[102]*B[20]+ W[113]*B[21];
  A[15]  -=  W[4]*B[11]  + W[15]*B[12]   + W[26]*B[13]   + W[37]*B[14]  + W[48]*B[15]  + W[59]*B[16]  + W[70]*B[17] + W[81]*B[18] + W[92]*B[19] + W[103]*B[20]+ W[114]*B[21];
  A[16]  -=  W[5]*B[11]  + W[16]*B[12]   + W[27]*B[13]   + W[38]*B[14]  + W[49]*B[15]  + W[60]*B[16]  + W[71]*B[17] + W[82]*B[18] + W[93]*B[19] + W[104]*B[20]+ W[115]*B[21];
  A[17]  -=  W[6]*B[11]  + W[17]*B[12]   + W[28]*B[13]   + W[39]*B[14]  + W[50]*B[15]  + W[61]*B[16]  + W[72]*B[17] + W[83]*B[18] + W[94]*B[19] + W[105]*B[20]+ W[116]*B[21];
  A[18]  -=  W[7]*B[11]  + W[18]*B[12]   + W[29]*B[13]   + W[40]*B[14]  + W[51]*B[15]  + W[62]*B[16]  + W[73]*B[17] + W[84]*B[18] + W[95]*B[19] + W[106]*B[20]+ W[117]*B[21];
  A[19]  -=  W[8]*B[11]  + W[19]*B[12]   + W[30]*B[13]   + W[41]*B[14]  + W[52]*B[15]  + W[63]*B[16]  + W[74]*B[17] + W[85]*B[18] + W[96]*B[19] + W[107]*B[20]+ W[118]*B[21];
  A[20]  -=  W[9]*B[11]  + W[20]*B[12]   + W[31]*B[13]   + W[42]*B[14]  + W[53]*B[15]  + W[64]*B[16]  + W[75]*B[17] + W[86]*B[18] + W[97]*B[19] + W[108]*B[20]+ W[119]*B[21];
  A[21]  -=  W[10]*B[11] + W[21]*B[12]   + W[32]*B[13]   + W[43]*B[14]  + W[54]*B[15]  + W[65]*B[16]  + W[76]*B[17] + W[87]*B[18] + W[98]*B[19] + W[109]*B[20]+ W[120]*B[21];

  A[22]  -=  W[0]*B[22]  + W[11]*B[23]   + W[22]*B[24]   + W[33]*B[25]  + W[44]*B[26]  + W[55]*B[27]  + W[66]*B[28] + W[77]*B[29] + W[88]*B[30] + W[99]*B[31] + W[110]*B[32];
  A[23]  -=  W[1]*B[22]  + W[12]*B[23]   + W[23]*B[24]   + W[34]*B[25]  + W[45]*B[26]  + W[56]*B[27]  + W[67]*B[28] + W[78]*B[29] + W[89]*B[30] + W[100]*B[31]+ W[111]*B[32];
  A[24]  -=  W[2]*B[22]  + W[13]*B[23]   + W[24]*B[24]   + W[35]*B[25]  + W[46]*B[26]  + W[57]*B[27]  + W[68]*B[28] + W[79]*B[29] + W[90]*B[30] + W[101]*B[31]+ W[112]*B[32];
  A[25]  -=  W[3]*B[22]  + W[14]*B[23]   + W[25]*B[24]   + W[36]*B[25]  + W[47]*B[26]  + W[58]*B[27]  + W[69]*B[28] + W[80]*B[29] + W[91]*B[30] + W[102]*B[31]+ W[113]*B[32];
  A[26]  -=  W[4]*B[22]  + W[15]*B[23]   + W[26]*B[24]   + W[37]*B[25]  + W[48]*B[26]  + W[59]*B[27]  + W[70]*B[28] + W[81]*B[29] + W[92]*B[30] + W[103]*B[31]+ W[114]*B[32];
  A[27]  -=  W[5]*B[22]  + W[16]*B[23]   + W[27]*B[24]   + W[38]*B[25]  + W[49]*B[26]  + W[60]*B[27]  + W[71]*B[28] + W[82]*B[29] + W[93]*B[30] + W[104]*B[31]+ W[115]*B[32];
  A[28]  -=  W[6]*B[22]  + W[17]*B[23]   + W[28]*B[24]   + W[39]*B[25]  + W[50]*B[26]  + W[61]*B[27]  + W[72]*B[28] + W[83]*B[29] + W[94]*B[30] + W[105]*B[31]+ W[116]*B[32];
  A[29]  -=  W[7]*B[22]  + W[18]*B[23]   + W[29]*B[24]   + W[40]*B[25]  + W[51]*B[26]  + W[62]*B[27]  + W[73]*B[28] + W[84]*B[29] + W[95]*B[30] + W[106]*B[31]+ W[117]*B[32];
  A[30]  -=  W[8]*B[22]  + W[19]*B[23]   + W[30]*B[24]   + W[41]*B[25]  + W[52]*B[26]  + W[63]*B[27]  + W[74]*B[28] + W[85]*B[29] + W[96]*B[30] + W[107]*B[31]+ W[118]*B[32];
  A[31]  -=  W[9]*B[22]  + W[20]*B[23]   + W[31]*B[24]   + W[42]*B[25]  + W[53]*B[26]  + W[64]*B[27]  + W[75]*B[28] + W[86]*B[29] + W[97]*B[30] + W[108]*B[31]+ W[119]*B[32];
  A[32]  -=  W[10]*B[22] + W[21]*B[23]   + W[32]*B[24]   + W[43]*B[25]  + W[54]*B[26]  + W[65]*B[27]  + W[76]*B[28] + W[87]*B[29] + W[98]*B[30] + W[109]*B[31]+ W[120]*B[32];

  A[33]  -=  W[0]*B[33]  + W[11]*B[34]   + W[22]*B[35]   + W[33]*B[36]  + W[44]*B[37]  + W[55]*B[38]  + W[66]*B[39] + W[77]*B[40] + W[88]*B[41] + W[99]*B[42] + W[110]*B[43];
  A[34]  -=  W[1]*B[33]  + W[12]*B[34]   + W[23]*B[35]   + W[34]*B[36]  + W[45]*B[37]  + W[56]*B[38]  + W[67]*B[39] + W[78]*B[40] + W[89]*B[41] + W[100]*B[42]+ W[111]*B[43];
  A[35]  -=  W[2]*B[33]  + W[13]*B[34]   + W[24]*B[35]   + W[35]*B[36]  + W[46]*B[37]  + W[57]*B[38]  + W[68]*B[39] + W[79]*B[40] + W[90]*B[41] + W[101]*B[42]+ W[112]*B[43];
  A[36]  -=  W[3]*B[33]  + W[14]*B[34]   + W[25]*B[35]   + W[36]*B[36]  + W[47]*B[37]  + W[58]*B[38]  + W[69]*B[39] + W[80]*B[40] + W[91]*B[41] + W[102]*B[42]+ W[113]*B[43];
  A[37]  -=  W[4]*B[33]  + W[15]*B[34]   + W[26]*B[35]   + W[37]*B[36]  + W[48]*B[37]  + W[59]*B[38]  + W[70]*B[39] + W[81]*B[40] + W[92]*B[41] + W[103]*B[42]+ W[114]*B[43];
  A[38]  -=  W[5]*B[33]  + W[16]*B[34]   + W[27]*B[35]   + W[38]*B[36]  + W[49]*B[37]  + W[60]*B[38]  + W[71]*B[39] + W[82]*B[40] + W[93]*B[41] + W[104]*B[42]+ W[115]*B[43];
  A[39]  -=  W[6]*B[33]  + W[17]*B[34]   + W[28]*B[35]   + W[39]*B[36]  + W[50]*B[37]  + W[61]*B[38]  + W[72]*B[39] + W[83]*B[40] + W[94]*B[41] + W[105]*B[42]+ W[116]*B[43];
  A[40]  -=  W[7]*B[33]  + W[18]*B[34]   + W[29]*B[35]   + W[40]*B[36]  + W[51]*B[37]  + W[62]*B[38]  + W[73]*B[39] + W[84]*B[40] + W[95]*B[41] + W[106]*B[42]+ W[117]*B[43];
  A[41]  -=  W[8]*B[33]  + W[19]*B[34]   + W[30]*B[35]   + W[41]*B[36]  + W[52]*B[37]  + W[63]*B[38]  + W[74]*B[39] + W[85]*B[40] + W[96]*B[41] + W[107]*B[42]+ W[118]*B[43];
  A[42]  -=  W[9]*B[33]  + W[20]*B[34]   + W[31]*B[35]   + W[42]*B[36]  + W[53]*B[37]  + W[64]*B[38]  + W[75]*B[39] + W[86]*B[40] + W[97]*B[41] + W[108]*B[42]+ W[119]*B[43];
  A[43]  -=  W[10]*B[33] + W[21]*B[34]   + W[32]*B[35]   + W[43]*B[36]  + W[54]*B[37]  + W[65]*B[38]  + W[76]*B[39] + W[87]*B[40] + W[98]*B[41] + W[109]*B[42]+ W[120]*B[43];

  A[44]  -=  W[0]*B[44]  + W[11]*B[45]   + W[22]*B[46]   + W[33]*B[47]  + W[44]*B[48]  + W[55]*B[49]  + W[66]*B[50] + W[77]*B[51] + W[88]*B[52] + W[99]*B[53] + W[110]*B[54];
  A[45]  -=  W[1]*B[44]  + W[12]*B[45]   + W[23]*B[46]   + W[34]*B[47]  + W[45]*B[48]  + W[56]*B[49]  + W[67]*B[50] + W[78]*B[51] + W[89]*B[52] + W[100]*B[53]+ W[111]*B[54];
  A[46]  -=  W[2]*B[44]  + W[13]*B[45]   + W[24]*B[46]   + W[35]*B[47]  + W[46]*B[48]  + W[57]*B[49]  + W[68]*B[50] + W[79]*B[51] + W[90]*B[52] + W[101]*B[53]+ W[112]*B[54];
  A[47]  -=  W[3]*B[44]  + W[14]*B[45]   + W[25]*B[46]   + W[36]*B[47]  + W[47]*B[48]  + W[58]*B[49]  + W[69]*B[50] + W[80]*B[51] + W[91]*B[52] + W[102]*B[53]+ W[113]*B[54];
  A[48]  -=  W[4]*B[44]  + W[15]*B[45]   + W[26]*B[46]   + W[37]*B[47]  + W[48]*B[48]  + W[59]*B[49]  + W[70]*B[50] + W[81]*B[51] + W[92]*B[52] + W[103]*B[53]+ W[114]*B[54];
  A[49]  -=  W[5]*B[44]  + W[16]*B[45]   + W[27]*B[46]   + W[38]*B[47]  + W[49]*B[48]  + W[60]*B[49]  + W[71]*B[50] + W[82]*B[51] + W[93]*B[52] + W[104]*B[53]+ W[115]*B[54];
  A[50]  -=  W[6]*B[44]  + W[17]*B[45]   + W[28]*B[46]   + W[39]*B[47]  + W[50]*B[48]  + W[61]*B[49]  + W[72]*B[50] + W[83]*B[51] + W[94]*B[52] + W[105]*B[53]+ W[116]*B[54];
  A[51]  -=  W[7]*B[44]  + W[18]*B[45]   + W[29]*B[46]   + W[40]*B[47]  + W[51]*B[48]  + W[62]*B[49]  + W[73]*B[50] + W[84]*B[51] + W[95]*B[52] + W[106]*B[53]+ W[117]*B[54];
  A[52]  -=  W[8]*B[44]  + W[19]*B[45]   + W[30]*B[46]   + W[41]*B[47]  + W[52]*B[48]  + W[63]*B[49]  + W[74]*B[50] + W[85]*B[51] + W[96]*B[52] + W[107]*B[53]+ W[118]*B[54];
  A[53]  -=  W[9]*B[44]  + W[20]*B[45]   + W[31]*B[46]   + W[42]*B[47]  + W[53]*B[48]  + W[64]*B[49]  + W[75]*B[50] + W[86]*B[51] + W[97]*B[52] + W[108]*B[53]+ W[119]*B[54];
  A[54]  -=  W[10]*B[44] + W[21]*B[45]   + W[32]*B[46]   + W[43]*B[47]  + W[54]*B[48]  + W[65]*B[49]  + W[76]*B[50] + W[87]*B[51] + W[98]*B[52] + W[109]*B[53]+ W[120]*B[54];

  A[55]  -=  W[0]*B[55]  + W[11]*B[56]   + W[22]*B[57]   + W[33]*B[58]  + W[44]*B[59]  + W[55]*B[60]  + W[66]*B[61] + W[77]*B[62] + W[88]*B[63] + W[99]*B[64] + W[110]*B[65];
  A[56]  -=  W[1]*B[55]  + W[12]*B[56]   + W[23]*B[57]   + W[34]*B[58]  + W[45]*B[59]  + W[56]*B[60]  + W[67]*B[61] + W[78]*B[62] + W[89]*B[63] + W[100]*B[64]+ W[111]*B[65];
  A[57]  -=  W[2]*B[55]  + W[13]*B[56]   + W[24]*B[57]   + W[35]*B[58]  + W[46]*B[59]  + W[57]*B[60]  + W[68]*B[61] + W[79]*B[62] + W[90]*B[63] + W[101]*B[64]+ W[112]*B[65];
  A[58]  -=  W[3]*B[55]  + W[14]*B[56]   + W[25]*B[57]   + W[36]*B[58]  + W[47]*B[59]  + W[58]*B[60]  + W[69]*B[61] + W[80]*B[62] + W[91]*B[63] + W[102]*B[64]+ W[113]*B[65];
  A[59]  -=  W[4]*B[55]  + W[15]*B[56]   + W[26]*B[57]   + W[37]*B[58]  + W[48]*B[59]  + W[59]*B[60]  + W[70]*B[61] + W[81]*B[62] + W[92]*B[63] + W[103]*B[64]+ W[114]*B[65];
  A[60]  -=  W[5]*B[55]  + W[16]*B[56]   + W[27]*B[57]   + W[38]*B[58]  + W[49]*B[59]  + W[60]*B[60]  + W[71]*B[61] + W[82]*B[62] + W[93]*B[63] + W[104]*B[64]+ W[115]*B[65];
  A[61]  -=  W[6]*B[55]  + W[17]*B[56]   + W[28]*B[57]   + W[39]*B[58]  + W[50]*B[59]  + W[61]*B[60]  + W[72]*B[61] + W[83]*B[62] + W[94]*B[63] + W[105]*B[64]+ W[116]*B[65];
  A[62]  -=  W[7]*B[55]  + W[18]*B[56]   + W[29]*B[57]   + W[40]*B[58]  + W[51]*B[59]  + W[62]*B[60]  + W[73]*B[61] + W[84]*B[62] + W[95]*B[63] + W[106]*B[64]+ W[117]*B[65];
  A[63]  -=  W[8]*B[55]  + W[19]*B[56]   + W[30]*B[57]   + W[41]*B[58]  + W[52]*B[59]  + W[63]*B[60]  + W[74]*B[61] + W[85]*B[62] + W[96]*B[63] + W[107]*B[64]+ W[118]*B[65];
  A[64]  -=  W[9]*B[55]  + W[20]*B[56]   + W[31]*B[57]   + W[42]*B[58]  + W[53]*B[59]  + W[64]*B[60]  + W[75]*B[61] + W[86]*B[62] + W[97]*B[63] + W[108]*B[64]+ W[119]*B[65];
  A[65]  -=  W[10]*B[55] + W[21]*B[56]   + W[32]*B[57]   + W[43]*B[58]  + W[54]*B[59]  + W[65]*B[60]  + W[76]*B[61] + W[87]*B[62] + W[98]*B[63] + W[109]*B[64]+ W[120]*B[65];

  A[66]  -=  W[0]*B[66]  + W[11]*B[67]   + W[22]*B[68]   + W[33]*B[69]  + W[44]*B[70]  + W[55]*B[71]  + W[66]*B[72] + W[77]*B[73] + W[88]*B[74] + W[99]*B[75] + W[110]*B[76];
  A[67]  -=  W[1]*B[66]  + W[12]*B[67]   + W[23]*B[68]   + W[34]*B[69]  + W[45]*B[70]  + W[56]*B[71]  + W[67]*B[72] + W[78]*B[73] + W[89]*B[74] + W[100]*B[75]+ W[111]*B[76];
  A[68]  -=  W[2]*B[66]  + W[13]*B[67]   + W[24]*B[68]   + W[35]*B[69]  + W[46]*B[70]  + W[57]*B[71]  + W[68]*B[72] + W[79]*B[73] + W[90]*B[74] + W[101]*B[75]+ W[112]*B[76];
  A[69]  -=  W[3]*B[66]  + W[14]*B[67]   + W[25]*B[68]   + W[36]*B[69]  + W[47]*B[70]  + W[58]*B[71]  + W[69]*B[72] + W[80]*B[73] + W[91]*B[74] + W[102]*B[75]+ W[113]*B[76];
  A[70]  -=  W[4]*B[66]  + W[15]*B[67]   + W[26]*B[68]   + W[37]*B[69]  + W[48]*B[70]  + W[59]*B[71]  + W[70]*B[72] + W[81]*B[73] + W[92]*B[74] + W[103]*B[75]+ W[114]*B[76];
  A[71]  -=  W[5]*B[66]  + W[16]*B[67]   + W[27]*B[68]   + W[38]*B[69]  + W[49]*B[70]  + W[60]*B[71]  + W[71]*B[72] + W[82]*B[73] + W[93]*B[74] + W[104]*B[75]+ W[115]*B[76];
  A[72]  -=  W[6]*B[66]  + W[17]*B[67]   + W[28]*B[68]   + W[39]*B[69]  + W[50]*B[70]  + W[61]*B[71]  + W[72]*B[72] + W[83]*B[73] + W[94]*B[74] + W[105]*B[75]+ W[116]*B[76];
  A[73]  -=  W[7]*B[66]  + W[18]*B[67]   + W[29]*B[68]   + W[40]*B[69]  + W[51]*B[70]  + W[62]*B[71]  + W[73]*B[72] + W[84]*B[73] + W[95]*B[74] + W[106]*B[75]+ W[117]*B[76];
  A[74]  -=  W[8]*B[66]  + W[19]*B[67]   + W[30]*B[68]   + W[41]*B[69]  + W[52]*B[70]  + W[63]*B[71]  + W[74]*B[72] + W[85]*B[73] + W[96]*B[74] + W[107]*B[75]+ W[118]*B[76];
  A[75]  -=  W[9]*B[66]  + W[20]*B[67]   + W[31]*B[68]   + W[42]*B[69]  + W[53]*B[70]  + W[64]*B[71]  + W[75]*B[72] + W[86]*B[73] + W[97]*B[74] + W[108]*B[75]+ W[119]*B[76];
  A[76]  -=  W[10]*B[66] + W[21]*B[67]   + W[32]*B[68]   + W[43]*B[69]  + W[54]*B[70]  + W[65]*B[71]  + W[76]*B[72] + W[87]*B[73] + W[98]*B[74] + W[109]*B[75]+ W[120]*B[76];

  A[77]  -=  W[0]*B[77]  + W[11]*B[78]   + W[22]*B[79]   + W[33]*B[80]  + W[44]*B[81]  + W[55]*B[82]  + W[66]*B[83] + W[77]*B[84] + W[88]*B[85] + W[99]*B[86] + W[110]*B[87];
  A[78]  -=  W[1]*B[77]  + W[12]*B[78]   + W[23]*B[79]   + W[34]*B[80]  + W[45]*B[81]  + W[56]*B[82]  + W[67]*B[83] + W[78]*B[84] + W[89]*B[85] + W[100]*B[86]+ W[111]*B[87];
  A[79]  -=  W[2]*B[77]  + W[13]*B[78]   + W[24]*B[79]   + W[35]*B[80]  + W[46]*B[81]  + W[57]*B[82]  + W[68]*B[83] + W[79]*B[84] + W[90]*B[85] + W[101]*B[86]+ W[112]*B[87];
  A[80]  -=  W[3]*B[77]  + W[14]*B[78]   + W[25]*B[79]   + W[36]*B[80]  + W[47]*B[81]  + W[58]*B[82]  + W[69]*B[83] + W[80]*B[84] + W[91]*B[85] + W[102]*B[86]+ W[113]*B[87];
  A[81]  -=  W[4]*B[77]  + W[15]*B[78]   + W[26]*B[79]   + W[37]*B[80]  + W[48]*B[81]  + W[59]*B[82]  + W[70]*B[83] + W[81]*B[84] + W[92]*B[85] + W[103]*B[86]+ W[114]*B[87];
  A[82]  -=  W[5]*B[77]  + W[16]*B[78]   + W[27]*B[79]   + W[38]*B[80]  + W[49]*B[81]  + W[60]*B[82]  + W[71]*B[83] + W[82]*B[84] + W[93]*B[85] + W[104]*B[86]+ W[115]*B[87];
  A[83]  -=  W[6]*B[77]  + W[17]*B[78]   + W[28]*B[79]   + W[39]*B[80]  + W[50]*B[81]  + W[61]*B[82]  + W[72]*B[83] + W[83]*B[84] + W[94]*B[85] + W[105]*B[86]+ W[116]*B[87];
  A[84]  -=  W[7]*B[77]  + W[18]*B[78]   + W[29]*B[79]   + W[40]*B[80]  + W[51]*B[81]  + W[62]*B[82]  + W[73]*B[83] + W[84]*B[84] + W[95]*B[85] + W[106]*B[86]+ W[117]*B[87];
  A[85]  -=  W[8]*B[77]  + W[19]*B[78]   + W[30]*B[79]   + W[41]*B[80]  + W[52]*B[81]  + W[63]*B[82]  + W[74]*B[83] + W[85]*B[84] + W[96]*B[85] + W[107]*B[86]+ W[118]*B[87];
  A[86]  -=  W[9]*B[77]  + W[20]*B[78]   + W[31]*B[79]   + W[42]*B[80]  + W[53]*B[81]  + W[64]*B[82]  + W[75]*B[83] + W[86]*B[84] + W[97]*B[85] + W[108]*B[86]+ W[119]*B[87];
  A[87]  -=  W[10]*B[77] + W[21]*B[78]   + W[32]*B[79]   + W[43]*B[80]  + W[54]*B[81]  + W[65]*B[82]  + W[76]*B[83] + W[87]*B[84] + W[98]*B[85] + W[109]*B[86]+ W[120]*B[87];

  A[88]  -=  W[0]*B[88]  + W[11]*B[89]   + W[22]*B[90]   + W[33]*B[91]  + W[44]*B[92]  + W[55]*B[93]  + W[66]*B[94] + W[77]*B[95] + W[88]*B[96] + W[99]*B[97] + W[110]*B[98];
  A[89]  -=  W[1]*B[88]  + W[12]*B[89]   + W[23]*B[90]   + W[34]*B[91]  + W[45]*B[92]  + W[56]*B[93]  + W[67]*B[94] + W[78]*B[95] + W[89]*B[96] + W[100]*B[97]+ W[111]*B[98];
  A[90]  -=  W[2]*B[88]  + W[13]*B[89]   + W[24]*B[90]   + W[35]*B[91]  + W[46]*B[92]  + W[57]*B[93]  + W[68]*B[94] + W[79]*B[95] + W[90]*B[96] + W[101]*B[97]+ W[112]*B[98];
  A[91]  -=  W[3]*B[88]  + W[14]*B[89]   + W[25]*B[90]   + W[36]*B[91]  + W[47]*B[92]  + W[58]*B[93]  + W[69]*B[94] + W[80]*B[95] + W[91]*B[96] + W[102]*B[97]+ W[113]*B[98];
  A[92]  -=  W[4]*B[88]  + W[15]*B[89]   + W[26]*B[90]   + W[37]*B[91]  + W[48]*B[92]  + W[59]*B[93]  + W[70]*B[94] + W[81]*B[95] + W[92]*B[96] + W[103]*B[97]+ W[114]*B[98];
  A[93]  -=  W[5]*B[88]  + W[16]*B[89]   + W[27]*B[90]   + W[38]*B[91]  + W[49]*B[92]  + W[60]*B[93]  + W[71]*B[94] + W[82]*B[95] + W[93]*B[96] + W[104]*B[97]+ W[115]*B[98];
  A[94]  -=  W[6]*B[88]  + W[17]*B[89]   + W[28]*B[90]   + W[39]*B[91]  + W[50]*B[92]  + W[61]*B[93]  + W[72]*B[94] + W[83]*B[95] + W[94]*B[96] + W[105]*B[97]+ W[116]*B[98];
  A[95]  -=  W[7]*B[88]  + W[18]*B[89]   + W[29]*B[90]   + W[40]*B[91]  + W[51]*B[92]  + W[62]*B[93]  + W[73]*B[94] + W[84]*B[95] + W[95]*B[96] + W[106]*B[97]+ W[117]*B[98];
  A[96]  -=  W[8]*B[88]  + W[19]*B[89]   + W[30]*B[90]   + W[41]*B[91]  + W[52]*B[92]  + W[63]*B[93]  + W[74]*B[94] + W[85]*B[95] + W[96]*B[96] + W[107]*B[97]+ W[118]*B[98];
  A[97]  -=  W[9]*B[88]  + W[20]*B[89]   + W[31]*B[90]   + W[42]*B[91]  + W[53]*B[92]  + W[64]*B[93]  + W[75]*B[94] + W[86]*B[95] + W[97]*B[96] + W[108]*B[97]+ W[119]*B[98];
  A[98]  -=  W[10]*B[88] + W[21]*B[89]   + W[32]*B[90]   + W[43]*B[91]  + W[54]*B[92]  + W[65]*B[93]  + W[76]*B[94] + W[87]*B[95] + W[98]*B[96] + W[109]*B[97]+ W[120]*B[98];

  A[99]  -= W[0]*B[99]  + W[11]*B[100] + W[22]*B[101] + W[33]*B[102]  + W[44]*B[103] + W[55]*B[104] + W[66]*B[105] + W[77]*B[106] + W[88]*B[107] + W[99]*B[108] + W[110]*B[109];
  A[100] -= W[1]*B[99]  + W[12]*B[100] + W[23]*B[101] + W[34]*B[102]  + W[45]*B[103] + W[56]*B[104] + W[67]*B[105] + W[78]*B[106] + W[89]*B[107] + W[100]*B[108]+ W[111]*B[109];
  A[101] -= W[2]*B[99]  + W[13]*B[100] + W[24]*B[101] + W[35]*B[102]  + W[46]*B[103] + W[57]*B[104] + W[68]*B[105] + W[79]*B[106] + W[90]*B[107] + W[101]*B[108]+ W[112]*B[109];
  A[102] -= W[3]*B[99]  + W[14]*B[100] + W[25]*B[101] + W[36]*B[102]  + W[47]*B[103] + W[58]*B[104] + W[69]*B[105] + W[80]*B[106] + W[91]*B[107] + W[102]*B[108]+ W[113]*B[109];
  A[103] -= W[4]*B[99]  + W[15]*B[100] + W[26]*B[101] + W[37]*B[102]  + W[48]*B[103] + W[59]*B[104] + W[70]*B[105] + W[81]*B[106] + W[92]*B[107] + W[103]*B[108]+ W[114]*B[109];
  A[104] -= W[5]*B[99]  + W[16]*B[100] + W[27]*B[101] + W[38]*B[102]  + W[49]*B[103] + W[60]*B[104] + W[71]*B[105] + W[82]*B[106] + W[93]*B[107] + W[104]*B[108]+ W[115]*B[109];
  A[105] -= W[6]*B[99]  + W[17]*B[100] + W[28]*B[101] + W[39]*B[102]  + W[50]*B[103] + W[61]*B[104] + W[72]*B[105] + W[83]*B[106] + W[94]*B[107] + W[105]*B[108]+ W[116]*B[109];
  A[106] -= W[7]*B[99]  + W[18]*B[100] + W[29]*B[101] + W[40]*B[102]  + W[51]*B[103] + W[62]*B[104] + W[73]*B[105] + W[84]*B[106] + W[95]*B[107] + W[106]*B[108]+ W[117]*B[109];
  A[107] -= W[8]*B[99]  + W[19]*B[100] + W[30]*B[101] + W[41]*B[102]  + W[52]*B[103] + W[63]*B[104] + W[74]*B[105] + W[85]*B[106] + W[96]*B[107] + W[107]*B[108]+ W[118]*B[109];
  A[108] -= W[9]*B[99]  + W[20]*B[100] + W[31]*B[101] + W[42]*B[102]  + W[53]*B[103] + W[64]*B[104] + W[75]*B[105] + W[86]*B[106] + W[97]*B[107] + W[108]*B[108]+ W[119]*B[109];
  A[109] -= W[10]*B[99] + W[21]*B[100] + W[32]*B[101] + W[43]*B[102]  + W[54]*B[103] + W[65]*B[104] + W[76]*B[105] + W[87]*B[106] + W[98]*B[107] + W[109]*B[108]+ W[120]*B[109];

  A[110] -= W[0]*B[110]  + W[11]*B[111] + W[22]*B[112] + W[33]*B[113]  + W[44]*B[114] + W[55]*B[115] + W[66]*B[116] + W[77]*B[117] + W[88]*B[118] + W[99]*B[119] + W[110]*B[120];
  A[111] -= W[1]*B[110]  + W[12]*B[111] + W[23]*B[112] + W[34]*B[113]  + W[45]*B[114] + W[56]*B[115] + W[67]*B[116] + W[78]*B[117] + W[89]*B[118] + W[100]*B[119]+ W[111]*B[120];
  A[112] -= W[2]*B[110]  + W[13]*B[111] + W[24]*B[112] + W[35]*B[113]  + W[46]*B[114] + W[57]*B[115] + W[68]*B[116] + W[79]*B[117] + W[90]*B[118] + W[101]*B[119]+ W[112]*B[120];
  A[113] -= W[3]*B[110]  + W[14]*B[111] + W[25]*B[112] + W[36]*B[113]  + W[47]*B[114] + W[58]*B[115] + W[69]*B[116] + W[80]*B[117] + W[91]*B[118] + W[102]*B[119]+ W[113]*B[120];
  A[114] -= W[4]*B[110]  + W[15]*B[111] + W[26]*B[112] + W[37]*B[113]  + W[48]*B[114] + W[59]*B[115] + W[70]*B[116] + W[81]*B[117] + W[92]*B[118] + W[103]*B[119]+ W[114]*B[120];
  A[115] -= W[5]*B[110]  + W[16]*B[111] + W[27]*B[112] + W[38]*B[113]  + W[49]*B[114] + W[60]*B[115] + W[71]*B[116] + W[82]*B[117] + W[93]*B[118] + W[104]*B[119]+ W[115]*B[120];
  A[116] -= W[6]*B[110]  + W[17]*B[111] + W[28]*B[112] + W[39]*B[113]  + W[50]*B[114] + W[61]*B[115] + W[72]*B[116] + W[83]*B[117] + W[94]*B[118] + W[105]*B[119]+ W[116]*B[120];
  A[117] -= W[7]*B[110]  + W[18]*B[111] + W[29]*B[112] + W[40]*B[113]  + W[51]*B[114] + W[62]*B[115] + W[73]*B[116] + W[84]*B[117] + W[95]*B[118] + W[106]*B[119]+ W[117]*B[120];
  A[118] -= W[8]*B[110]  + W[19]*B[111] + W[30]*B[112] + W[41]*B[113]  + W[52]*B[114] + W[63]*B[115] + W[74]*B[116] + W[85]*B[117] + W[96]*B[118] + W[107]*B[119]+ W[118]*B[120];
  A[119] -= W[9]*B[110]  + W[20]*B[111] + W[31]*B[112] + W[42]*B[113]  + W[53]*B[114] + W[64]*B[115] + W[75]*B[116] + W[86]*B[117] + W[97]*B[118] + W[108]*B[119]+ W[119]*B[120];
  A[120] -= W[10]*B[110] + W[21]*B[111] + W[32]*B[112] + W[43]*B[113]  + W[54]*B[114] + W[65]*B[115] + W[76]*B[116] + W[87]*B[117] + W[98]*B[118] + W[109]*B[119]+ W[120]*B[120];
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_times_B_15(PetscScalar *A,const PetscScalar *B,PetscScalar *W)
{
  PetscErrorCode ierr;

  ierr = PetscArraycpy(W,A,225);CHKERRQ(ierr);
  A[0]   = W[0]*B[0] + W[15]*B[1] + W[30]*B[2] + W[45]*B[3] + W[60]*B[4] + W[75]*B[5] + W[90]*B[6] + W[105]*B[7] + W[120]*B[8] + W[135]*B[9] + W[150]*B[10] + W[165]*B[11] + W[180]*B[12] + W[195]*B[13] + W[210]*B[14];
  A[1]   = W[1]*B[0] + W[16]*B[1] + W[31]*B[2] + W[46]*B[3] + W[61]*B[4] + W[76]*B[5] + W[91]*B[6] + W[106]*B[7] + W[121]*B[8] + W[136]*B[9] + W[151]*B[10] + W[166]*B[11] + W[181]*B[12] + W[196]*B[13] + W[211]*B[14];
  A[2]   = W[2]*B[0] + W[17]*B[1] + W[32]*B[2] + W[47]*B[3] + W[62]*B[4] + W[77]*B[5] + W[92]*B[6] + W[107]*B[7] + W[122]*B[8] + W[137]*B[9] + W[152]*B[10] + W[167]*B[11] + W[182]*B[12] + W[197]*B[13] + W[212]*B[14];
  A[3]   = W[3]*B[0] + W[18]*B[1] + W[33]*B[2] + W[48]*B[3] + W[63]*B[4] + W[78]*B[5] + W[93]*B[6] + W[108]*B[7] + W[123]*B[8] + W[138]*B[9] + W[153]*B[10] + W[168]*B[11] + W[183]*B[12] + W[198]*B[13] + W[213]*B[14];
  A[4]   = W[4]*B[0] + W[19]*B[1] + W[34]*B[2] + W[49]*B[3] + W[64]*B[4] + W[79]*B[5] + W[94]*B[6] + W[109]*B[7] + W[124]*B[8] + W[139]*B[9] + W[154]*B[10] + W[169]*B[11] + W[184]*B[12] + W[199]*B[13] + W[214]*B[14];
  A[5]   = W[5]*B[0] + W[20]*B[1] + W[35]*B[2] + W[50]*B[3] + W[65]*B[4] + W[80]*B[5] + W[95]*B[6] + W[110]*B[7] + W[125]*B[8] + W[140]*B[9] + W[155]*B[10] + W[170]*B[11] + W[185]*B[12] + W[200]*B[13] + W[215]*B[14];
  A[6]   = W[6]*B[0] + W[21]*B[1] + W[36]*B[2] + W[51]*B[3] + W[66]*B[4] + W[81]*B[5] + W[96]*B[6] + W[111]*B[7] + W[126]*B[8] + W[141]*B[9] + W[156]*B[10] + W[171]*B[11] + W[186]*B[12] + W[201]*B[13] + W[216]*B[14];
  A[7]   = W[7]*B[0] + W[22]*B[1] + W[37]*B[2] + W[52]*B[3] + W[67]*B[4] + W[82]*B[5] + W[97]*B[6] + W[112]*B[7] + W[127]*B[8] + W[142]*B[9] + W[157]*B[10] + W[172]*B[11] + W[187]*B[12] + W[202]*B[13] + W[217]*B[14];
  A[8]   = W[8]*B[0] + W[23]*B[1] + W[38]*B[2] + W[53]*B[3] + W[68]*B[4] + W[83]*B[5] + W[98]*B[6] + W[113]*B[7] + W[128]*B[8] + W[143]*B[9] + W[158]*B[10] + W[173]*B[11] + W[188]*B[12] + W[203]*B[13] + W[218]*B[14];
  A[9]   = W[9]*B[0] + W[24]*B[1] + W[39]*B[2] + W[54]*B[3] + W[69]*B[4] + W[84]*B[5] + W[99]*B[6] + W[114]*B[7] + W[129]*B[8] + W[144]*B[9] + W[159]*B[10] + W[174]*B[11] + W[189]*B[12] + W[204]*B[13] + W[219]*B[14];
  A[10]  = W[10]*B[0] + W[25]*B[1] + W[40]*B[2] + W[55]*B[3] + W[70]*B[4] + W[85]*B[5] + W[100]*B[6] + W[115]*B[7] + W[130]*B[8] + W[145]*B[9] + W[160]*B[10] + W[175]*B[11] + W[190]*B[12] + W[205]*B[13] + W[220]*B[14];
  A[11]  = W[11]*B[0] + W[26]*B[1] + W[41]*B[2] + W[56]*B[3] + W[71]*B[4] + W[86]*B[5] + W[101]*B[6] + W[116]*B[7] + W[131]*B[8] + W[146]*B[9] + W[161]*B[10] + W[176]*B[11] + W[191]*B[12] + W[206]*B[13] + W[221]*B[14];
  A[12]  = W[12]*B[0] + W[27]*B[1] + W[42]*B[2] + W[57]*B[3] + W[72]*B[4] + W[87]*B[5] + W[102]*B[6] + W[117]*B[7] + W[132]*B[8] + W[147]*B[9] + W[162]*B[10] + W[177]*B[11] + W[192]*B[12] + W[207]*B[13] + W[222]*B[14];
  A[13]  = W[13]*B[0] + W[28]*B[1] + W[43]*B[2] + W[58]*B[3] + W[73]*B[4] + W[88]*B[5] + W[103]*B[6] + W[118]*B[7] + W[133]*B[8] + W[148]*B[9] + W[163]*B[10] + W[178]*B[11] + W[193]*B[12] + W[208]*B[13] + W[223]*B[14];
  A[14]  = W[14]*B[0] + W[29]*B[1] + W[44]*B[2] + W[59]*B[3] + W[74]*B[4] + W[89]*B[5] + W[104]*B[6] + W[119]*B[7] + W[134]*B[8] + W[149]*B[9] + W[164]*B[10] + W[179]*B[11] + W[194]*B[12] + W[209]*B[13] + W[224]*B[14];
  A[15]  = W[0]*B[15] + W[15]*B[16] + W[30]*B[17] + W[45]*B[18] + W[60]*B[19] + W[75]*B[20] + W[90]*B[21] + W[105]*B[22] + W[120]*B[23] + W[135]*B[24] + W[150]*B[25] + W[165]*B[26] + W[180]*B[27] + W[195]*B[28] + W[210]*B[29];
  A[16]  = W[1]*B[15] + W[16]*B[16] + W[31]*B[17] + W[46]*B[18] + W[61]*B[19] + W[76]*B[20] + W[91]*B[21] + W[106]*B[22] + W[121]*B[23] + W[136]*B[24] + W[151]*B[25] + W[166]*B[26] + W[181]*B[27] + W[196]*B[28] + W[211]*B[29];
  A[17]  = W[2]*B[15] + W[17]*B[16] + W[32]*B[17] + W[47]*B[18] + W[62]*B[19] + W[77]*B[20] + W[92]*B[21] + W[107]*B[22] + W[122]*B[23] + W[137]*B[24] + W[152]*B[25] + W[167]*B[26] + W[182]*B[27] + W[197]*B[28] + W[212]*B[29];
  A[18]  = W[3]*B[15] + W[18]*B[16] + W[33]*B[17] + W[48]*B[18] + W[63]*B[19] + W[78]*B[20] + W[93]*B[21] + W[108]*B[22] + W[123]*B[23] + W[138]*B[24] + W[153]*B[25] + W[168]*B[26] + W[183]*B[27] + W[198]*B[28] + W[213]*B[29];
  A[19]  = W[4]*B[15] + W[19]*B[16] + W[34]*B[17] + W[49]*B[18] + W[64]*B[19] + W[79]*B[20] + W[94]*B[21] + W[109]*B[22] + W[124]*B[23] + W[139]*B[24] + W[154]*B[25] + W[169]*B[26] + W[184]*B[27] + W[199]*B[28] + W[214]*B[29];
  A[20]  = W[5]*B[15] + W[20]*B[16] + W[35]*B[17] + W[50]*B[18] + W[65]*B[19] + W[80]*B[20] + W[95]*B[21] + W[110]*B[22] + W[125]*B[23] + W[140]*B[24] + W[155]*B[25] + W[170]*B[26] + W[185]*B[27] + W[200]*B[28] + W[215]*B[29];
  A[21]  = W[6]*B[15] + W[21]*B[16] + W[36]*B[17] + W[51]*B[18] + W[66]*B[19] + W[81]*B[20] + W[96]*B[21] + W[111]*B[22] + W[126]*B[23] + W[141]*B[24] + W[156]*B[25] + W[171]*B[26] + W[186]*B[27] + W[201]*B[28] + W[216]*B[29];
  A[22]  = W[7]*B[15] + W[22]*B[16] + W[37]*B[17] + W[52]*B[18] + W[67]*B[19] + W[82]*B[20] + W[97]*B[21] + W[112]*B[22] + W[127]*B[23] + W[142]*B[24] + W[157]*B[25] + W[172]*B[26] + W[187]*B[27] + W[202]*B[28] + W[217]*B[29];
  A[23]  = W[8]*B[15] + W[23]*B[16] + W[38]*B[17] + W[53]*B[18] + W[68]*B[19] + W[83]*B[20] + W[98]*B[21] + W[113]*B[22] + W[128]*B[23] + W[143]*B[24] + W[158]*B[25] + W[173]*B[26] + W[188]*B[27] + W[203]*B[28] + W[218]*B[29];
  A[24]  = W[9]*B[15] + W[24]*B[16] + W[39]*B[17] + W[54]*B[18] + W[69]*B[19] + W[84]*B[20] + W[99]*B[21] + W[114]*B[22] + W[129]*B[23] + W[144]*B[24] + W[159]*B[25] + W[174]*B[26] + W[189]*B[27] + W[204]*B[28] + W[219]*B[29];
  A[25]  = W[10]*B[15] + W[25]*B[16] + W[40]*B[17] + W[55]*B[18] + W[70]*B[19] + W[85]*B[20] + W[100]*B[21] + W[115]*B[22] + W[130]*B[23] + W[145]*B[24] + W[160]*B[25] + W[175]*B[26] + W[190]*B[27] + W[205]*B[28] + W[220]*B[29];
  A[26]  = W[11]*B[15] + W[26]*B[16] + W[41]*B[17] + W[56]*B[18] + W[71]*B[19] + W[86]*B[20] + W[101]*B[21] + W[116]*B[22] + W[131]*B[23] + W[146]*B[24] + W[161]*B[25] + W[176]*B[26] + W[191]*B[27] + W[206]*B[28] + W[221]*B[29];
  A[27]  = W[12]*B[15] + W[27]*B[16] + W[42]*B[17] + W[57]*B[18] + W[72]*B[19] + W[87]*B[20] + W[102]*B[21] + W[117]*B[22] + W[132]*B[23] + W[147]*B[24] + W[162]*B[25] + W[177]*B[26] + W[192]*B[27] + W[207]*B[28] + W[222]*B[29];
  A[28]  = W[13]*B[15] + W[28]*B[16] + W[43]*B[17] + W[58]*B[18] + W[73]*B[19] + W[88]*B[20] + W[103]*B[21] + W[118]*B[22] + W[133]*B[23] + W[148]*B[24] + W[163]*B[25] + W[178]*B[26] + W[193]*B[27] + W[208]*B[28] + W[223]*B[29];
  A[29]  = W[14]*B[15] + W[29]*B[16] + W[44]*B[17] + W[59]*B[18] + W[74]*B[19] + W[89]*B[20] + W[104]*B[21] + W[119]*B[22] + W[134]*B[23] + W[149]*B[24] + W[164]*B[25] + W[179]*B[26] + W[194]*B[27] + W[209]*B[28] + W[224]*B[29];
  A[30]  = W[0]*B[30] + W[15]*B[31] + W[30]*B[32] + W[45]*B[33] + W[60]*B[34] + W[75]*B[35] + W[90]*B[36] + W[105]*B[37] + W[120]*B[38] + W[135]*B[39] + W[150]*B[40] + W[165]*B[41] + W[180]*B[42] + W[195]*B[43] + W[210]*B[44];
  A[31]  = W[1]*B[30] + W[16]*B[31] + W[31]*B[32] + W[46]*B[33] + W[61]*B[34] + W[76]*B[35] + W[91]*B[36] + W[106]*B[37] + W[121]*B[38] + W[136]*B[39] + W[151]*B[40] + W[166]*B[41] + W[181]*B[42] + W[196]*B[43] + W[211]*B[44];
  A[32]  = W[2]*B[30] + W[17]*B[31] + W[32]*B[32] + W[47]*B[33] + W[62]*B[34] + W[77]*B[35] + W[92]*B[36] + W[107]*B[37] + W[122]*B[38] + W[137]*B[39] + W[152]*B[40] + W[167]*B[41] + W[182]*B[42] + W[197]*B[43] + W[212]*B[44];
  A[33]  = W[3]*B[30] + W[18]*B[31] + W[33]*B[32] + W[48]*B[33] + W[63]*B[34] + W[78]*B[35] + W[93]*B[36] + W[108]*B[37] + W[123]*B[38] + W[138]*B[39] + W[153]*B[40] + W[168]*B[41] + W[183]*B[42] + W[198]*B[43] + W[213]*B[44];
  A[34]  = W[4]*B[30] + W[19]*B[31] + W[34]*B[32] + W[49]*B[33] + W[64]*B[34] + W[79]*B[35] + W[94]*B[36] + W[109]*B[37] + W[124]*B[38] + W[139]*B[39] + W[154]*B[40] + W[169]*B[41] + W[184]*B[42] + W[199]*B[43] + W[214]*B[44];
  A[35]  = W[5]*B[30] + W[20]*B[31] + W[35]*B[32] + W[50]*B[33] + W[65]*B[34] + W[80]*B[35] + W[95]*B[36] + W[110]*B[37] + W[125]*B[38] + W[140]*B[39] + W[155]*B[40] + W[170]*B[41] + W[185]*B[42] + W[200]*B[43] + W[215]*B[44];
  A[36]  = W[6]*B[30] + W[21]*B[31] + W[36]*B[32] + W[51]*B[33] + W[66]*B[34] + W[81]*B[35] + W[96]*B[36] + W[111]*B[37] + W[126]*B[38] + W[141]*B[39] + W[156]*B[40] + W[171]*B[41] + W[186]*B[42] + W[201]*B[43] + W[216]*B[44];
  A[37]  = W[7]*B[30] + W[22]*B[31] + W[37]*B[32] + W[52]*B[33] + W[67]*B[34] + W[82]*B[35] + W[97]*B[36] + W[112]*B[37] + W[127]*B[38] + W[142]*B[39] + W[157]*B[40] + W[172]*B[41] + W[187]*B[42] + W[202]*B[43] + W[217]*B[44];
  A[38]  = W[8]*B[30] + W[23]*B[31] + W[38]*B[32] + W[53]*B[33] + W[68]*B[34] + W[83]*B[35] + W[98]*B[36] + W[113]*B[37] + W[128]*B[38] + W[143]*B[39] + W[158]*B[40] + W[173]*B[41] + W[188]*B[42] + W[203]*B[43] + W[218]*B[44];
  A[39]  = W[9]*B[30] + W[24]*B[31] + W[39]*B[32] + W[54]*B[33] + W[69]*B[34] + W[84]*B[35] + W[99]*B[36] + W[114]*B[37] + W[129]*B[38] + W[144]*B[39] + W[159]*B[40] + W[174]*B[41] + W[189]*B[42] + W[204]*B[43] + W[219]*B[44];
  A[40]  = W[10]*B[30] + W[25]*B[31] + W[40]*B[32] + W[55]*B[33] + W[70]*B[34] + W[85]*B[35] + W[100]*B[36] + W[115]*B[37] + W[130]*B[38] + W[145]*B[39] + W[160]*B[40] + W[175]*B[41] + W[190]*B[42] + W[205]*B[43] + W[220]*B[44];
  A[41]  = W[11]*B[30] + W[26]*B[31] + W[41]*B[32] + W[56]*B[33] + W[71]*B[34] + W[86]*B[35] + W[101]*B[36] + W[116]*B[37] + W[131]*B[38] + W[146]*B[39] + W[161]*B[40] + W[176]*B[41] + W[191]*B[42] + W[206]*B[43] + W[221]*B[44];
  A[42]  = W[12]*B[30] + W[27]*B[31] + W[42]*B[32] + W[57]*B[33] + W[72]*B[34] + W[87]*B[35] + W[102]*B[36] + W[117]*B[37] + W[132]*B[38] + W[147]*B[39] + W[162]*B[40] + W[177]*B[41] + W[192]*B[42] + W[207]*B[43] + W[222]*B[44];
  A[43]  = W[13]*B[30] + W[28]*B[31] + W[43]*B[32] + W[58]*B[33] + W[73]*B[34] + W[88]*B[35] + W[103]*B[36] + W[118]*B[37] + W[133]*B[38] + W[148]*B[39] + W[163]*B[40] + W[178]*B[41] + W[193]*B[42] + W[208]*B[43] + W[223]*B[44];
  A[44]  = W[14]*B[30] + W[29]*B[31] + W[44]*B[32] + W[59]*B[33] + W[74]*B[34] + W[89]*B[35] + W[104]*B[36] + W[119]*B[37] + W[134]*B[38] + W[149]*B[39] + W[164]*B[40] + W[179]*B[41] + W[194]*B[42] + W[209]*B[43] + W[224]*B[44];
  A[45]  = W[0]*B[45] + W[15]*B[46] + W[30]*B[47] + W[45]*B[48] + W[60]*B[49] + W[75]*B[50] + W[90]*B[51] + W[105]*B[52] + W[120]*B[53] + W[135]*B[54] + W[150]*B[55] + W[165]*B[56] + W[180]*B[57] + W[195]*B[58] + W[210]*B[59];
  A[46]  = W[1]*B[45] + W[16]*B[46] + W[31]*B[47] + W[46]*B[48] + W[61]*B[49] + W[76]*B[50] + W[91]*B[51] + W[106]*B[52] + W[121]*B[53] + W[136]*B[54] + W[151]*B[55] + W[166]*B[56] + W[181]*B[57] + W[196]*B[58] + W[211]*B[59];
  A[47]  = W[2]*B[45] + W[17]*B[46] + W[32]*B[47] + W[47]*B[48] + W[62]*B[49] + W[77]*B[50] + W[92]*B[51] + W[107]*B[52] + W[122]*B[53] + W[137]*B[54] + W[152]*B[55] + W[167]*B[56] + W[182]*B[57] + W[197]*B[58] + W[212]*B[59];
  A[48]  = W[3]*B[45] + W[18]*B[46] + W[33]*B[47] + W[48]*B[48] + W[63]*B[49] + W[78]*B[50] + W[93]*B[51] + W[108]*B[52] + W[123]*B[53] + W[138]*B[54] + W[153]*B[55] + W[168]*B[56] + W[183]*B[57] + W[198]*B[58] + W[213]*B[59];
  A[49]  = W[4]*B[45] + W[19]*B[46] + W[34]*B[47] + W[49]*B[48] + W[64]*B[49] + W[79]*B[50] + W[94]*B[51] + W[109]*B[52] + W[124]*B[53] + W[139]*B[54] + W[154]*B[55] + W[169]*B[56] + W[184]*B[57] + W[199]*B[58] + W[214]*B[59];
  A[50]  = W[5]*B[45] + W[20]*B[46] + W[35]*B[47] + W[50]*B[48] + W[65]*B[49] + W[80]*B[50] + W[95]*B[51] + W[110]*B[52] + W[125]*B[53] + W[140]*B[54] + W[155]*B[55] + W[170]*B[56] + W[185]*B[57] + W[200]*B[58] + W[215]*B[59];
  A[51]  = W[6]*B[45] + W[21]*B[46] + W[36]*B[47] + W[51]*B[48] + W[66]*B[49] + W[81]*B[50] + W[96]*B[51] + W[111]*B[52] + W[126]*B[53] + W[141]*B[54] + W[156]*B[55] + W[171]*B[56] + W[186]*B[57] + W[201]*B[58] + W[216]*B[59];
  A[52]  = W[7]*B[45] + W[22]*B[46] + W[37]*B[47] + W[52]*B[48] + W[67]*B[49] + W[82]*B[50] + W[97]*B[51] + W[112]*B[52] + W[127]*B[53] + W[142]*B[54] + W[157]*B[55] + W[172]*B[56] + W[187]*B[57] + W[202]*B[58] + W[217]*B[59];
  A[53]  = W[8]*B[45] + W[23]*B[46] + W[38]*B[47] + W[53]*B[48] + W[68]*B[49] + W[83]*B[50] + W[98]*B[51] + W[113]*B[52] + W[128]*B[53] + W[143]*B[54] + W[158]*B[55] + W[173]*B[56] + W[188]*B[57] + W[203]*B[58] + W[218]*B[59];
  A[54]  = W[9]*B[45] + W[24]*B[46] + W[39]*B[47] + W[54]*B[48] + W[69]*B[49] + W[84]*B[50] + W[99]*B[51] + W[114]*B[52] + W[129]*B[53] + W[144]*B[54] + W[159]*B[55] + W[174]*B[56] + W[189]*B[57] + W[204]*B[58] + W[219]*B[59];
  A[55]  = W[10]*B[45] + W[25]*B[46] + W[40]*B[47] + W[55]*B[48] + W[70]*B[49] + W[85]*B[50] + W[100]*B[51] + W[115]*B[52] + W[130]*B[53] + W[145]*B[54] + W[160]*B[55] + W[175]*B[56] + W[190]*B[57] + W[205]*B[58] + W[220]*B[59];
  A[56]  = W[11]*B[45] + W[26]*B[46] + W[41]*B[47] + W[56]*B[48] + W[71]*B[49] + W[86]*B[50] + W[101]*B[51] + W[116]*B[52] + W[131]*B[53] + W[146]*B[54] + W[161]*B[55] + W[176]*B[56] + W[191]*B[57] + W[206]*B[58] + W[221]*B[59];
  A[57]  = W[12]*B[45] + W[27]*B[46] + W[42]*B[47] + W[57]*B[48] + W[72]*B[49] + W[87]*B[50] + W[102]*B[51] + W[117]*B[52] + W[132]*B[53] + W[147]*B[54] + W[162]*B[55] + W[177]*B[56] + W[192]*B[57] + W[207]*B[58] + W[222]*B[59];
  A[58]  = W[13]*B[45] + W[28]*B[46] + W[43]*B[47] + W[58]*B[48] + W[73]*B[49] + W[88]*B[50] + W[103]*B[51] + W[118]*B[52] + W[133]*B[53] + W[148]*B[54] + W[163]*B[55] + W[178]*B[56] + W[193]*B[57] + W[208]*B[58] + W[223]*B[59];
  A[59]  = W[14]*B[45] + W[29]*B[46] + W[44]*B[47] + W[59]*B[48] + W[74]*B[49] + W[89]*B[50] + W[104]*B[51] + W[119]*B[52] + W[134]*B[53] + W[149]*B[54] + W[164]*B[55] + W[179]*B[56] + W[194]*B[57] + W[209]*B[58] + W[224]*B[59];
  A[60]  = W[0]*B[60] + W[15]*B[61] + W[30]*B[62] + W[45]*B[63] + W[60]*B[64] + W[75]*B[65] + W[90]*B[66] + W[105]*B[67] + W[120]*B[68] + W[135]*B[69] + W[150]*B[70] + W[165]*B[71] + W[180]*B[72] + W[195]*B[73] + W[210]*B[74];
  A[61]  = W[1]*B[60] + W[16]*B[61] + W[31]*B[62] + W[46]*B[63] + W[61]*B[64] + W[76]*B[65] + W[91]*B[66] + W[106]*B[67] + W[121]*B[68] + W[136]*B[69] + W[151]*B[70] + W[166]*B[71] + W[181]*B[72] + W[196]*B[73] + W[211]*B[74];
  A[62]  = W[2]*B[60] + W[17]*B[61] + W[32]*B[62] + W[47]*B[63] + W[62]*B[64] + W[77]*B[65] + W[92]*B[66] + W[107]*B[67] + W[122]*B[68] + W[137]*B[69] + W[152]*B[70] + W[167]*B[71] + W[182]*B[72] + W[197]*B[73] + W[212]*B[74];
  A[63]  = W[3]*B[60] + W[18]*B[61] + W[33]*B[62] + W[48]*B[63] + W[63]*B[64] + W[78]*B[65] + W[93]*B[66] + W[108]*B[67] + W[123]*B[68] + W[138]*B[69] + W[153]*B[70] + W[168]*B[71] + W[183]*B[72] + W[198]*B[73] + W[213]*B[74];
  A[64]  = W[4]*B[60] + W[19]*B[61] + W[34]*B[62] + W[49]*B[63] + W[64]*B[64] + W[79]*B[65] + W[94]*B[66] + W[109]*B[67] + W[124]*B[68] + W[139]*B[69] + W[154]*B[70] + W[169]*B[71] + W[184]*B[72] + W[199]*B[73] + W[214]*B[74];
  A[65]  = W[5]*B[60] + W[20]*B[61] + W[35]*B[62] + W[50]*B[63] + W[65]*B[64] + W[80]*B[65] + W[95]*B[66] + W[110]*B[67] + W[125]*B[68] + W[140]*B[69] + W[155]*B[70] + W[170]*B[71] + W[185]*B[72] + W[200]*B[73] + W[215]*B[74];
  A[66]  = W[6]*B[60] + W[21]*B[61] + W[36]*B[62] + W[51]*B[63] + W[66]*B[64] + W[81]*B[65] + W[96]*B[66] + W[111]*B[67] + W[126]*B[68] + W[141]*B[69] + W[156]*B[70] + W[171]*B[71] + W[186]*B[72] + W[201]*B[73] + W[216]*B[74];
  A[67]  = W[7]*B[60] + W[22]*B[61] + W[37]*B[62] + W[52]*B[63] + W[67]*B[64] + W[82]*B[65] + W[97]*B[66] + W[112]*B[67] + W[127]*B[68] + W[142]*B[69] + W[157]*B[70] + W[172]*B[71] + W[187]*B[72] + W[202]*B[73] + W[217]*B[74];
  A[68]  = W[8]*B[60] + W[23]*B[61] + W[38]*B[62] + W[53]*B[63] + W[68]*B[64] + W[83]*B[65] + W[98]*B[66] + W[113]*B[67] + W[128]*B[68] + W[143]*B[69] + W[158]*B[70] + W[173]*B[71] + W[188]*B[72] + W[203]*B[73] + W[218]*B[74];
  A[69]  = W[9]*B[60] + W[24]*B[61] + W[39]*B[62] + W[54]*B[63] + W[69]*B[64] + W[84]*B[65] + W[99]*B[66] + W[114]*B[67] + W[129]*B[68] + W[144]*B[69] + W[159]*B[70] + W[174]*B[71] + W[189]*B[72] + W[204]*B[73] + W[219]*B[74];
  A[70]  = W[10]*B[60] + W[25]*B[61] + W[40]*B[62] + W[55]*B[63] + W[70]*B[64] + W[85]*B[65] + W[100]*B[66] + W[115]*B[67] + W[130]*B[68] + W[145]*B[69] + W[160]*B[70] + W[175]*B[71] + W[190]*B[72] + W[205]*B[73] + W[220]*B[74];
  A[71]  = W[11]*B[60] + W[26]*B[61] + W[41]*B[62] + W[56]*B[63] + W[71]*B[64] + W[86]*B[65] + W[101]*B[66] + W[116]*B[67] + W[131]*B[68] + W[146]*B[69] + W[161]*B[70] + W[176]*B[71] + W[191]*B[72] + W[206]*B[73] + W[221]*B[74];
  A[72]  = W[12]*B[60] + W[27]*B[61] + W[42]*B[62] + W[57]*B[63] + W[72]*B[64] + W[87]*B[65] + W[102]*B[66] + W[117]*B[67] + W[132]*B[68] + W[147]*B[69] + W[162]*B[70] + W[177]*B[71] + W[192]*B[72] + W[207]*B[73] + W[222]*B[74];
  A[73]  = W[13]*B[60] + W[28]*B[61] + W[43]*B[62] + W[58]*B[63] + W[73]*B[64] + W[88]*B[65] + W[103]*B[66] + W[118]*B[67] + W[133]*B[68] + W[148]*B[69] + W[163]*B[70] + W[178]*B[71] + W[193]*B[72] + W[208]*B[73] + W[223]*B[74];
  A[74]  = W[14]*B[60] + W[29]*B[61] + W[44]*B[62] + W[59]*B[63] + W[74]*B[64] + W[89]*B[65] + W[104]*B[66] + W[119]*B[67] + W[134]*B[68] + W[149]*B[69] + W[164]*B[70] + W[179]*B[71] + W[194]*B[72] + W[209]*B[73] + W[224]*B[74];
  A[75]  = W[0]*B[75] + W[15]*B[76] + W[30]*B[77] + W[45]*B[78] + W[60]*B[79] + W[75]*B[80] + W[90]*B[81] + W[105]*B[82] + W[120]*B[83] + W[135]*B[84] + W[150]*B[85] + W[165]*B[86] + W[180]*B[87] + W[195]*B[88] + W[210]*B[89];
  A[76]  = W[1]*B[75] + W[16]*B[76] + W[31]*B[77] + W[46]*B[78] + W[61]*B[79] + W[76]*B[80] + W[91]*B[81] + W[106]*B[82] + W[121]*B[83] + W[136]*B[84] + W[151]*B[85] + W[166]*B[86] + W[181]*B[87] + W[196]*B[88] + W[211]*B[89];
  A[77]  = W[2]*B[75] + W[17]*B[76] + W[32]*B[77] + W[47]*B[78] + W[62]*B[79] + W[77]*B[80] + W[92]*B[81] + W[107]*B[82] + W[122]*B[83] + W[137]*B[84] + W[152]*B[85] + W[167]*B[86] + W[182]*B[87] + W[197]*B[88] + W[212]*B[89];
  A[78]  = W[3]*B[75] + W[18]*B[76] + W[33]*B[77] + W[48]*B[78] + W[63]*B[79] + W[78]*B[80] + W[93]*B[81] + W[108]*B[82] + W[123]*B[83] + W[138]*B[84] + W[153]*B[85] + W[168]*B[86] + W[183]*B[87] + W[198]*B[88] + W[213]*B[89];
  A[79]  = W[4]*B[75] + W[19]*B[76] + W[34]*B[77] + W[49]*B[78] + W[64]*B[79] + W[79]*B[80] + W[94]*B[81] + W[109]*B[82] + W[124]*B[83] + W[139]*B[84] + W[154]*B[85] + W[169]*B[86] + W[184]*B[87] + W[199]*B[88] + W[214]*B[89];
  A[80]  = W[5]*B[75] + W[20]*B[76] + W[35]*B[77] + W[50]*B[78] + W[65]*B[79] + W[80]*B[80] + W[95]*B[81] + W[110]*B[82] + W[125]*B[83] + W[140]*B[84] + W[155]*B[85] + W[170]*B[86] + W[185]*B[87] + W[200]*B[88] + W[215]*B[89];
  A[81]  = W[6]*B[75] + W[21]*B[76] + W[36]*B[77] + W[51]*B[78] + W[66]*B[79] + W[81]*B[80] + W[96]*B[81] + W[111]*B[82] + W[126]*B[83] + W[141]*B[84] + W[156]*B[85] + W[171]*B[86] + W[186]*B[87] + W[201]*B[88] + W[216]*B[89];
  A[82]  = W[7]*B[75] + W[22]*B[76] + W[37]*B[77] + W[52]*B[78] + W[67]*B[79] + W[82]*B[80] + W[97]*B[81] + W[112]*B[82] + W[127]*B[83] + W[142]*B[84] + W[157]*B[85] + W[172]*B[86] + W[187]*B[87] + W[202]*B[88] + W[217]*B[89];
  A[83]  = W[8]*B[75] + W[23]*B[76] + W[38]*B[77] + W[53]*B[78] + W[68]*B[79] + W[83]*B[80] + W[98]*B[81] + W[113]*B[82] + W[128]*B[83] + W[143]*B[84] + W[158]*B[85] + W[173]*B[86] + W[188]*B[87] + W[203]*B[88] + W[218]*B[89];
  A[84]  = W[9]*B[75] + W[24]*B[76] + W[39]*B[77] + W[54]*B[78] + W[69]*B[79] + W[84]*B[80] + W[99]*B[81] + W[114]*B[82] + W[129]*B[83] + W[144]*B[84] + W[159]*B[85] + W[174]*B[86] + W[189]*B[87] + W[204]*B[88] + W[219]*B[89];
  A[85]  = W[10]*B[75] + W[25]*B[76] + W[40]*B[77] + W[55]*B[78] + W[70]*B[79] + W[85]*B[80] + W[100]*B[81] + W[115]*B[82] + W[130]*B[83] + W[145]*B[84] + W[160]*B[85] + W[175]*B[86] + W[190]*B[87] + W[205]*B[88] + W[220]*B[89];
  A[86]  = W[11]*B[75] + W[26]*B[76] + W[41]*B[77] + W[56]*B[78] + W[71]*B[79] + W[86]*B[80] + W[101]*B[81] + W[116]*B[82] + W[131]*B[83] + W[146]*B[84] + W[161]*B[85] + W[176]*B[86] + W[191]*B[87] + W[206]*B[88] + W[221]*B[89];
  A[87]  = W[12]*B[75] + W[27]*B[76] + W[42]*B[77] + W[57]*B[78] + W[72]*B[79] + W[87]*B[80] + W[102]*B[81] + W[117]*B[82] + W[132]*B[83] + W[147]*B[84] + W[162]*B[85] + W[177]*B[86] + W[192]*B[87] + W[207]*B[88] + W[222]*B[89];
  A[88]  = W[13]*B[75] + W[28]*B[76] + W[43]*B[77] + W[58]*B[78] + W[73]*B[79] + W[88]*B[80] + W[103]*B[81] + W[118]*B[82] + W[133]*B[83] + W[148]*B[84] + W[163]*B[85] + W[178]*B[86] + W[193]*B[87] + W[208]*B[88] + W[223]*B[89];
  A[89]  = W[14]*B[75] + W[29]*B[76] + W[44]*B[77] + W[59]*B[78] + W[74]*B[79] + W[89]*B[80] + W[104]*B[81] + W[119]*B[82] + W[134]*B[83] + W[149]*B[84] + W[164]*B[85] + W[179]*B[86] + W[194]*B[87] + W[209]*B[88] + W[224]*B[89];
  A[90]  = W[0]*B[90] + W[15]*B[91] + W[30]*B[92] + W[45]*B[93] + W[60]*B[94] + W[75]*B[95] + W[90]*B[96] + W[105]*B[97] + W[120]*B[98] + W[135]*B[99] + W[150]*B[100] + W[165]*B[101] + W[180]*B[102] + W[195]*B[103] + W[210]*B[104];
  A[91]  = W[1]*B[90] + W[16]*B[91] + W[31]*B[92] + W[46]*B[93] + W[61]*B[94] + W[76]*B[95] + W[91]*B[96] + W[106]*B[97] + W[121]*B[98] + W[136]*B[99] + W[151]*B[100] + W[166]*B[101] + W[181]*B[102] + W[196]*B[103] + W[211]*B[104];
  A[92]  = W[2]*B[90] + W[17]*B[91] + W[32]*B[92] + W[47]*B[93] + W[62]*B[94] + W[77]*B[95] + W[92]*B[96] + W[107]*B[97] + W[122]*B[98] + W[137]*B[99] + W[152]*B[100] + W[167]*B[101] + W[182]*B[102] + W[197]*B[103] + W[212]*B[104];
  A[93]  = W[3]*B[90] + W[18]*B[91] + W[33]*B[92] + W[48]*B[93] + W[63]*B[94] + W[78]*B[95] + W[93]*B[96] + W[108]*B[97] + W[123]*B[98] + W[138]*B[99] + W[153]*B[100] + W[168]*B[101] + W[183]*B[102] + W[198]*B[103] + W[213]*B[104];
  A[94]  = W[4]*B[90] + W[19]*B[91] + W[34]*B[92] + W[49]*B[93] + W[64]*B[94] + W[79]*B[95] + W[94]*B[96] + W[109]*B[97] + W[124]*B[98] + W[139]*B[99] + W[154]*B[100] + W[169]*B[101] + W[184]*B[102] + W[199]*B[103] + W[214]*B[104];
  A[95]  = W[5]*B[90] + W[20]*B[91] + W[35]*B[92] + W[50]*B[93] + W[65]*B[94] + W[80]*B[95] + W[95]*B[96] + W[110]*B[97] + W[125]*B[98] + W[140]*B[99] + W[155]*B[100] + W[170]*B[101] + W[185]*B[102] + W[200]*B[103] + W[215]*B[104];
  A[96]  = W[6]*B[90] + W[21]*B[91] + W[36]*B[92] + W[51]*B[93] + W[66]*B[94] + W[81]*B[95] + W[96]*B[96] + W[111]*B[97] + W[126]*B[98] + W[141]*B[99] + W[156]*B[100] + W[171]*B[101] + W[186]*B[102] + W[201]*B[103] + W[216]*B[104];
  A[97]  = W[7]*B[90] + W[22]*B[91] + W[37]*B[92] + W[52]*B[93] + W[67]*B[94] + W[82]*B[95] + W[97]*B[96] + W[112]*B[97] + W[127]*B[98] + W[142]*B[99] + W[157]*B[100] + W[172]*B[101] + W[187]*B[102] + W[202]*B[103] + W[217]*B[104];
  A[98]  = W[8]*B[90] + W[23]*B[91] + W[38]*B[92] + W[53]*B[93] + W[68]*B[94] + W[83]*B[95] + W[98]*B[96] + W[113]*B[97] + W[128]*B[98] + W[143]*B[99] + W[158]*B[100] + W[173]*B[101] + W[188]*B[102] + W[203]*B[103] + W[218]*B[104];
  A[99]  = W[9]*B[90] + W[24]*B[91] + W[39]*B[92] + W[54]*B[93] + W[69]*B[94] + W[84]*B[95] + W[99]*B[96] + W[114]*B[97] + W[129]*B[98] + W[144]*B[99] + W[159]*B[100] + W[174]*B[101] + W[189]*B[102] + W[204]*B[103] + W[219]*B[104];
  A[100] = W[10]*B[90] + W[25]*B[91] + W[40]*B[92] + W[55]*B[93] + W[70]*B[94] + W[85]*B[95] + W[100]*B[96] + W[115]*B[97] + W[130]*B[98] + W[145]*B[99] + W[160]*B[100] + W[175]*B[101] + W[190]*B[102] + W[205]*B[103] + W[220]*B[104];
  A[101] = W[11]*B[90] + W[26]*B[91] + W[41]*B[92] + W[56]*B[93] + W[71]*B[94] + W[86]*B[95] + W[101]*B[96] + W[116]*B[97] + W[131]*B[98] + W[146]*B[99] + W[161]*B[100] + W[176]*B[101] + W[191]*B[102] + W[206]*B[103] + W[221]*B[104];
  A[102] = W[12]*B[90] + W[27]*B[91] + W[42]*B[92] + W[57]*B[93] + W[72]*B[94] + W[87]*B[95] + W[102]*B[96] + W[117]*B[97] + W[132]*B[98] + W[147]*B[99] + W[162]*B[100] + W[177]*B[101] + W[192]*B[102] + W[207]*B[103] + W[222]*B[104];
  A[103] = W[13]*B[90] + W[28]*B[91] + W[43]*B[92] + W[58]*B[93] + W[73]*B[94] + W[88]*B[95] + W[103]*B[96] + W[118]*B[97] + W[133]*B[98] + W[148]*B[99] + W[163]*B[100] + W[178]*B[101] + W[193]*B[102] + W[208]*B[103] + W[223]*B[104];
  A[104] = W[14]*B[90] + W[29]*B[91] + W[44]*B[92] + W[59]*B[93] + W[74]*B[94] + W[89]*B[95] + W[104]*B[96] + W[119]*B[97] + W[134]*B[98] + W[149]*B[99] + W[164]*B[100] + W[179]*B[101] + W[194]*B[102] + W[209]*B[103] + W[224]*B[104];
  A[105] = W[0]*B[105] + W[15]*B[106] + W[30]*B[107] + W[45]*B[108] + W[60]*B[109] + W[75]*B[110] + W[90]*B[111] + W[105]*B[112] + W[120]*B[113] + W[135]*B[114] + W[150]*B[115] + W[165]*B[116] + W[180]*B[117] + W[195]*B[118] + W[210]*B[119];
  A[106] = W[1]*B[105] + W[16]*B[106] + W[31]*B[107] + W[46]*B[108] + W[61]*B[109] + W[76]*B[110] + W[91]*B[111] + W[106]*B[112] + W[121]*B[113] + W[136]*B[114] + W[151]*B[115] + W[166]*B[116] + W[181]*B[117] + W[196]*B[118] + W[211]*B[119];
  A[107] = W[2]*B[105] + W[17]*B[106] + W[32]*B[107] + W[47]*B[108] + W[62]*B[109] + W[77]*B[110] + W[92]*B[111] + W[107]*B[112] + W[122]*B[113] + W[137]*B[114] + W[152]*B[115] + W[167]*B[116] + W[182]*B[117] + W[197]*B[118] + W[212]*B[119];
  A[108] = W[3]*B[105] + W[18]*B[106] + W[33]*B[107] + W[48]*B[108] + W[63]*B[109] + W[78]*B[110] + W[93]*B[111] + W[108]*B[112] + W[123]*B[113] + W[138]*B[114] + W[153]*B[115] + W[168]*B[116] + W[183]*B[117] + W[198]*B[118] + W[213]*B[119];
  A[109] = W[4]*B[105] + W[19]*B[106] + W[34]*B[107] + W[49]*B[108] + W[64]*B[109] + W[79]*B[110] + W[94]*B[111] + W[109]*B[112] + W[124]*B[113] + W[139]*B[114] + W[154]*B[115] + W[169]*B[116] + W[184]*B[117] + W[199]*B[118] + W[214]*B[119];
  A[110] = W[5]*B[105] + W[20]*B[106] + W[35]*B[107] + W[50]*B[108] + W[65]*B[109] + W[80]*B[110] + W[95]*B[111] + W[110]*B[112] + W[125]*B[113] + W[140]*B[114] + W[155]*B[115] + W[170]*B[116] + W[185]*B[117] + W[200]*B[118] + W[215]*B[119];
  A[111] = W[6]*B[105] + W[21]*B[106] + W[36]*B[107] + W[51]*B[108] + W[66]*B[109] + W[81]*B[110] + W[96]*B[111] + W[111]*B[112] + W[126]*B[113] + W[141]*B[114] + W[156]*B[115] + W[171]*B[116] + W[186]*B[117] + W[201]*B[118] + W[216]*B[119];
  A[112] = W[7]*B[105] + W[22]*B[106] + W[37]*B[107] + W[52]*B[108] + W[67]*B[109] + W[82]*B[110] + W[97]*B[111] + W[112]*B[112] + W[127]*B[113] + W[142]*B[114] + W[157]*B[115] + W[172]*B[116] + W[187]*B[117] + W[202]*B[118] + W[217]*B[119];
  A[113] = W[8]*B[105] + W[23]*B[106] + W[38]*B[107] + W[53]*B[108] + W[68]*B[109] + W[83]*B[110] + W[98]*B[111] + W[113]*B[112] + W[128]*B[113] + W[143]*B[114] + W[158]*B[115] + W[173]*B[116] + W[188]*B[117] + W[203]*B[118] + W[218]*B[119];
  A[114] = W[9]*B[105] + W[24]*B[106] + W[39]*B[107] + W[54]*B[108] + W[69]*B[109] + W[84]*B[110] + W[99]*B[111] + W[114]*B[112] + W[129]*B[113] + W[144]*B[114] + W[159]*B[115] + W[174]*B[116] + W[189]*B[117] + W[204]*B[118] + W[219]*B[119];
  A[115] = W[10]*B[105] + W[25]*B[106] + W[40]*B[107] + W[55]*B[108] + W[70]*B[109] + W[85]*B[110] + W[100]*B[111] + W[115]*B[112] + W[130]*B[113] + W[145]*B[114] + W[160]*B[115] + W[175]*B[116] + W[190]*B[117] + W[205]*B[118] + W[220]*B[119];
  A[116] = W[11]*B[105] + W[26]*B[106] + W[41]*B[107] + W[56]*B[108] + W[71]*B[109] + W[86]*B[110] + W[101]*B[111] + W[116]*B[112] + W[131]*B[113] + W[146]*B[114] + W[161]*B[115] + W[176]*B[116] + W[191]*B[117] + W[206]*B[118] + W[221]*B[119];
  A[117] = W[12]*B[105] + W[27]*B[106] + W[42]*B[107] + W[57]*B[108] + W[72]*B[109] + W[87]*B[110] + W[102]*B[111] + W[117]*B[112] + W[132]*B[113] + W[147]*B[114] + W[162]*B[115] + W[177]*B[116] + W[192]*B[117] + W[207]*B[118] + W[222]*B[119];
  A[118] = W[13]*B[105] + W[28]*B[106] + W[43]*B[107] + W[58]*B[108] + W[73]*B[109] + W[88]*B[110] + W[103]*B[111] + W[118]*B[112] + W[133]*B[113] + W[148]*B[114] + W[163]*B[115] + W[178]*B[116] + W[193]*B[117] + W[208]*B[118] + W[223]*B[119];
  A[119] = W[14]*B[105] + W[29]*B[106] + W[44]*B[107] + W[59]*B[108] + W[74]*B[109] + W[89]*B[110] + W[104]*B[111] + W[119]*B[112] + W[134]*B[113] + W[149]*B[114] + W[164]*B[115] + W[179]*B[116] + W[194]*B[117] + W[209]*B[118] + W[224]*B[119];
  A[120] = W[0]*B[120] + W[15]*B[121] + W[30]*B[122] + W[45]*B[123] + W[60]*B[124] + W[75]*B[125] + W[90]*B[126] + W[105]*B[127] + W[120]*B[128] + W[135]*B[129] + W[150]*B[130] + W[165]*B[131] + W[180]*B[132] + W[195]*B[133] + W[210]*B[134];
  A[121] = W[1]*B[120] + W[16]*B[121] + W[31]*B[122] + W[46]*B[123] + W[61]*B[124] + W[76]*B[125] + W[91]*B[126] + W[106]*B[127] + W[121]*B[128] + W[136]*B[129] + W[151]*B[130] + W[166]*B[131] + W[181]*B[132] + W[196]*B[133] + W[211]*B[134];
  A[122] = W[2]*B[120] + W[17]*B[121] + W[32]*B[122] + W[47]*B[123] + W[62]*B[124] + W[77]*B[125] + W[92]*B[126] + W[107]*B[127] + W[122]*B[128] + W[137]*B[129] + W[152]*B[130] + W[167]*B[131] + W[182]*B[132] + W[197]*B[133] + W[212]*B[134];
  A[123] = W[3]*B[120] + W[18]*B[121] + W[33]*B[122] + W[48]*B[123] + W[63]*B[124] + W[78]*B[125] + W[93]*B[126] + W[108]*B[127] + W[123]*B[128] + W[138]*B[129] + W[153]*B[130] + W[168]*B[131] + W[183]*B[132] + W[198]*B[133] + W[213]*B[134];
  A[124] = W[4]*B[120] + W[19]*B[121] + W[34]*B[122] + W[49]*B[123] + W[64]*B[124] + W[79]*B[125] + W[94]*B[126] + W[109]*B[127] + W[124]*B[128] + W[139]*B[129] + W[154]*B[130] + W[169]*B[131] + W[184]*B[132] + W[199]*B[133] + W[214]*B[134];
  A[125] = W[5]*B[120] + W[20]*B[121] + W[35]*B[122] + W[50]*B[123] + W[65]*B[124] + W[80]*B[125] + W[95]*B[126] + W[110]*B[127] + W[125]*B[128] + W[140]*B[129] + W[155]*B[130] + W[170]*B[131] + W[185]*B[132] + W[200]*B[133] + W[215]*B[134];
  A[126] = W[6]*B[120] + W[21]*B[121] + W[36]*B[122] + W[51]*B[123] + W[66]*B[124] + W[81]*B[125] + W[96]*B[126] + W[111]*B[127] + W[126]*B[128] + W[141]*B[129] + W[156]*B[130] + W[171]*B[131] + W[186]*B[132] + W[201]*B[133] + W[216]*B[134];
  A[127] = W[7]*B[120] + W[22]*B[121] + W[37]*B[122] + W[52]*B[123] + W[67]*B[124] + W[82]*B[125] + W[97]*B[126] + W[112]*B[127] + W[127]*B[128] + W[142]*B[129] + W[157]*B[130] + W[172]*B[131] + W[187]*B[132] + W[202]*B[133] + W[217]*B[134];
  A[128] = W[8]*B[120] + W[23]*B[121] + W[38]*B[122] + W[53]*B[123] + W[68]*B[124] + W[83]*B[125] + W[98]*B[126] + W[113]*B[127] + W[128]*B[128] + W[143]*B[129] + W[158]*B[130] + W[173]*B[131] + W[188]*B[132] + W[203]*B[133] + W[218]*B[134];
  A[129] = W[9]*B[120] + W[24]*B[121] + W[39]*B[122] + W[54]*B[123] + W[69]*B[124] + W[84]*B[125] + W[99]*B[126] + W[114]*B[127] + W[129]*B[128] + W[144]*B[129] + W[159]*B[130] + W[174]*B[131] + W[189]*B[132] + W[204]*B[133] + W[219]*B[134];
  A[130] = W[10]*B[120] + W[25]*B[121] + W[40]*B[122] + W[55]*B[123] + W[70]*B[124] + W[85]*B[125] + W[100]*B[126] + W[115]*B[127] + W[130]*B[128] + W[145]*B[129] + W[160]*B[130] + W[175]*B[131] + W[190]*B[132] + W[205]*B[133] + W[220]*B[134];
  A[131] = W[11]*B[120] + W[26]*B[121] + W[41]*B[122] + W[56]*B[123] + W[71]*B[124] + W[86]*B[125] + W[101]*B[126] + W[116]*B[127] + W[131]*B[128] + W[146]*B[129] + W[161]*B[130] + W[176]*B[131] + W[191]*B[132] + W[206]*B[133] + W[221]*B[134];
  A[132] = W[12]*B[120] + W[27]*B[121] + W[42]*B[122] + W[57]*B[123] + W[72]*B[124] + W[87]*B[125] + W[102]*B[126] + W[117]*B[127] + W[132]*B[128] + W[147]*B[129] + W[162]*B[130] + W[177]*B[131] + W[192]*B[132] + W[207]*B[133] + W[222]*B[134];
  A[133] = W[13]*B[120] + W[28]*B[121] + W[43]*B[122] + W[58]*B[123] + W[73]*B[124] + W[88]*B[125] + W[103]*B[126] + W[118]*B[127] + W[133]*B[128] + W[148]*B[129] + W[163]*B[130] + W[178]*B[131] + W[193]*B[132] + W[208]*B[133] + W[223]*B[134];
  A[134] = W[14]*B[120] + W[29]*B[121] + W[44]*B[122] + W[59]*B[123] + W[74]*B[124] + W[89]*B[125] + W[104]*B[126] + W[119]*B[127] + W[134]*B[128] + W[149]*B[129] + W[164]*B[130] + W[179]*B[131] + W[194]*B[132] + W[209]*B[133] + W[224]*B[134];
  A[135] = W[0]*B[135] + W[15]*B[136] + W[30]*B[137] + W[45]*B[138] + W[60]*B[139] + W[75]*B[140] + W[90]*B[141] + W[105]*B[142] + W[120]*B[143] + W[135]*B[144] + W[150]*B[145] + W[165]*B[146] + W[180]*B[147] + W[195]*B[148] + W[210]*B[149];
  A[136] = W[1]*B[135] + W[16]*B[136] + W[31]*B[137] + W[46]*B[138] + W[61]*B[139] + W[76]*B[140] + W[91]*B[141] + W[106]*B[142] + W[121]*B[143] + W[136]*B[144] + W[151]*B[145] + W[166]*B[146] + W[181]*B[147] + W[196]*B[148] + W[211]*B[149];
  A[137] = W[2]*B[135] + W[17]*B[136] + W[32]*B[137] + W[47]*B[138] + W[62]*B[139] + W[77]*B[140] + W[92]*B[141] + W[107]*B[142] + W[122]*B[143] + W[137]*B[144] + W[152]*B[145] + W[167]*B[146] + W[182]*B[147] + W[197]*B[148] + W[212]*B[149];
  A[138] = W[3]*B[135] + W[18]*B[136] + W[33]*B[137] + W[48]*B[138] + W[63]*B[139] + W[78]*B[140] + W[93]*B[141] + W[108]*B[142] + W[123]*B[143] + W[138]*B[144] + W[153]*B[145] + W[168]*B[146] + W[183]*B[147] + W[198]*B[148] + W[213]*B[149];
  A[139] = W[4]*B[135] + W[19]*B[136] + W[34]*B[137] + W[49]*B[138] + W[64]*B[139] + W[79]*B[140] + W[94]*B[141] + W[109]*B[142] + W[124]*B[143] + W[139]*B[144] + W[154]*B[145] + W[169]*B[146] + W[184]*B[147] + W[199]*B[148] + W[214]*B[149];
  A[140] = W[5]*B[135] + W[20]*B[136] + W[35]*B[137] + W[50]*B[138] + W[65]*B[139] + W[80]*B[140] + W[95]*B[141] + W[110]*B[142] + W[125]*B[143] + W[140]*B[144] + W[155]*B[145] + W[170]*B[146] + W[185]*B[147] + W[200]*B[148] + W[215]*B[149];
  A[141] = W[6]*B[135] + W[21]*B[136] + W[36]*B[137] + W[51]*B[138] + W[66]*B[139] + W[81]*B[140] + W[96]*B[141] + W[111]*B[142] + W[126]*B[143] + W[141]*B[144] + W[156]*B[145] + W[171]*B[146] + W[186]*B[147] + W[201]*B[148] + W[216]*B[149];
  A[142] = W[7]*B[135] + W[22]*B[136] + W[37]*B[137] + W[52]*B[138] + W[67]*B[139] + W[82]*B[140] + W[97]*B[141] + W[112]*B[142] + W[127]*B[143] + W[142]*B[144] + W[157]*B[145] + W[172]*B[146] + W[187]*B[147] + W[202]*B[148] + W[217]*B[149];
  A[143] = W[8]*B[135] + W[23]*B[136] + W[38]*B[137] + W[53]*B[138] + W[68]*B[139] + W[83]*B[140] + W[98]*B[141] + W[113]*B[142] + W[128]*B[143] + W[143]*B[144] + W[158]*B[145] + W[173]*B[146] + W[188]*B[147] + W[203]*B[148] + W[218]*B[149];
  A[144] = W[9]*B[135] + W[24]*B[136] + W[39]*B[137] + W[54]*B[138] + W[69]*B[139] + W[84]*B[140] + W[99]*B[141] + W[114]*B[142] + W[129]*B[143] + W[144]*B[144] + W[159]*B[145] + W[174]*B[146] + W[189]*B[147] + W[204]*B[148] + W[219]*B[149];
  A[145] = W[10]*B[135] + W[25]*B[136] + W[40]*B[137] + W[55]*B[138] + W[70]*B[139] + W[85]*B[140] + W[100]*B[141] + W[115]*B[142] + W[130]*B[143] + W[145]*B[144] + W[160]*B[145] + W[175]*B[146] + W[190]*B[147] + W[205]*B[148] + W[220]*B[149];
  A[146] = W[11]*B[135] + W[26]*B[136] + W[41]*B[137] + W[56]*B[138] + W[71]*B[139] + W[86]*B[140] + W[101]*B[141] + W[116]*B[142] + W[131]*B[143] + W[146]*B[144] + W[161]*B[145] + W[176]*B[146] + W[191]*B[147] + W[206]*B[148] + W[221]*B[149];
  A[147] = W[12]*B[135] + W[27]*B[136] + W[42]*B[137] + W[57]*B[138] + W[72]*B[139] + W[87]*B[140] + W[102]*B[141] + W[117]*B[142] + W[132]*B[143] + W[147]*B[144] + W[162]*B[145] + W[177]*B[146] + W[192]*B[147] + W[207]*B[148] + W[222]*B[149];
  A[148] = W[13]*B[135] + W[28]*B[136] + W[43]*B[137] + W[58]*B[138] + W[73]*B[139] + W[88]*B[140] + W[103]*B[141] + W[118]*B[142] + W[133]*B[143] + W[148]*B[144] + W[163]*B[145] + W[178]*B[146] + W[193]*B[147] + W[208]*B[148] + W[223]*B[149];
  A[149] = W[14]*B[135] + W[29]*B[136] + W[44]*B[137] + W[59]*B[138] + W[74]*B[139] + W[89]*B[140] + W[104]*B[141] + W[119]*B[142] + W[134]*B[143] + W[149]*B[144] + W[164]*B[145] + W[179]*B[146] + W[194]*B[147] + W[209]*B[148] + W[224]*B[149];
  A[150] = W[0]*B[150] + W[15]*B[151] + W[30]*B[152] + W[45]*B[153] + W[60]*B[154] + W[75]*B[155] + W[90]*B[156] + W[105]*B[157] + W[120]*B[158] + W[135]*B[159] + W[150]*B[160] + W[165]*B[161] + W[180]*B[162] + W[195]*B[163] + W[210]*B[164];
  A[151] = W[1]*B[150] + W[16]*B[151] + W[31]*B[152] + W[46]*B[153] + W[61]*B[154] + W[76]*B[155] + W[91]*B[156] + W[106]*B[157] + W[121]*B[158] + W[136]*B[159] + W[151]*B[160] + W[166]*B[161] + W[181]*B[162] + W[196]*B[163] + W[211]*B[164];
  A[152] = W[2]*B[150] + W[17]*B[151] + W[32]*B[152] + W[47]*B[153] + W[62]*B[154] + W[77]*B[155] + W[92]*B[156] + W[107]*B[157] + W[122]*B[158] + W[137]*B[159] + W[152]*B[160] + W[167]*B[161] + W[182]*B[162] + W[197]*B[163] + W[212]*B[164];
  A[153] = W[3]*B[150] + W[18]*B[151] + W[33]*B[152] + W[48]*B[153] + W[63]*B[154] + W[78]*B[155] + W[93]*B[156] + W[108]*B[157] + W[123]*B[158] + W[138]*B[159] + W[153]*B[160] + W[168]*B[161] + W[183]*B[162] + W[198]*B[163] + W[213]*B[164];
  A[154] = W[4]*B[150] + W[19]*B[151] + W[34]*B[152] + W[49]*B[153] + W[64]*B[154] + W[79]*B[155] + W[94]*B[156] + W[109]*B[157] + W[124]*B[158] + W[139]*B[159] + W[154]*B[160] + W[169]*B[161] + W[184]*B[162] + W[199]*B[163] + W[214]*B[164];
  A[155] = W[5]*B[150] + W[20]*B[151] + W[35]*B[152] + W[50]*B[153] + W[65]*B[154] + W[80]*B[155] + W[95]*B[156] + W[110]*B[157] + W[125]*B[158] + W[140]*B[159] + W[155]*B[160] + W[170]*B[161] + W[185]*B[162] + W[200]*B[163] + W[215]*B[164];
  A[156] = W[6]*B[150] + W[21]*B[151] + W[36]*B[152] + W[51]*B[153] + W[66]*B[154] + W[81]*B[155] + W[96]*B[156] + W[111]*B[157] + W[126]*B[158] + W[141]*B[159] + W[156]*B[160] + W[171]*B[161] + W[186]*B[162] + W[201]*B[163] + W[216]*B[164];
  A[157] = W[7]*B[150] + W[22]*B[151] + W[37]*B[152] + W[52]*B[153] + W[67]*B[154] + W[82]*B[155] + W[97]*B[156] + W[112]*B[157] + W[127]*B[158] + W[142]*B[159] + W[157]*B[160] + W[172]*B[161] + W[187]*B[162] + W[202]*B[163] + W[217]*B[164];
  A[158] = W[8]*B[150] + W[23]*B[151] + W[38]*B[152] + W[53]*B[153] + W[68]*B[154] + W[83]*B[155] + W[98]*B[156] + W[113]*B[157] + W[128]*B[158] + W[143]*B[159] + W[158]*B[160] + W[173]*B[161] + W[188]*B[162] + W[203]*B[163] + W[218]*B[164];
  A[159] = W[9]*B[150] + W[24]*B[151] + W[39]*B[152] + W[54]*B[153] + W[69]*B[154] + W[84]*B[155] + W[99]*B[156] + W[114]*B[157] + W[129]*B[158] + W[144]*B[159] + W[159]*B[160] + W[174]*B[161] + W[189]*B[162] + W[204]*B[163] + W[219]*B[164];
  A[160] = W[10]*B[150] + W[25]*B[151] + W[40]*B[152] + W[55]*B[153] + W[70]*B[154] + W[85]*B[155] + W[100]*B[156] + W[115]*B[157] + W[130]*B[158] + W[145]*B[159] + W[160]*B[160] + W[175]*B[161] + W[190]*B[162] + W[205]*B[163] + W[220]*B[164];
  A[161] = W[11]*B[150] + W[26]*B[151] + W[41]*B[152] + W[56]*B[153] + W[71]*B[154] + W[86]*B[155] + W[101]*B[156] + W[116]*B[157] + W[131]*B[158] + W[146]*B[159] + W[161]*B[160] + W[176]*B[161] + W[191]*B[162] + W[206]*B[163] + W[221]*B[164];
  A[162] = W[12]*B[150] + W[27]*B[151] + W[42]*B[152] + W[57]*B[153] + W[72]*B[154] + W[87]*B[155] + W[102]*B[156] + W[117]*B[157] + W[132]*B[158] + W[147]*B[159] + W[162]*B[160] + W[177]*B[161] + W[192]*B[162] + W[207]*B[163] + W[222]*B[164];
  A[163] = W[13]*B[150] + W[28]*B[151] + W[43]*B[152] + W[58]*B[153] + W[73]*B[154] + W[88]*B[155] + W[103]*B[156] + W[118]*B[157] + W[133]*B[158] + W[148]*B[159] + W[163]*B[160] + W[178]*B[161] + W[193]*B[162] + W[208]*B[163] + W[223]*B[164];
  A[164] = W[14]*B[150] + W[29]*B[151] + W[44]*B[152] + W[59]*B[153] + W[74]*B[154] + W[89]*B[155] + W[104]*B[156] + W[119]*B[157] + W[134]*B[158] + W[149]*B[159] + W[164]*B[160] + W[179]*B[161] + W[194]*B[162] + W[209]*B[163] + W[224]*B[164];
  A[165] = W[0]*B[165] + W[15]*B[166] + W[30]*B[167] + W[45]*B[168] + W[60]*B[169] + W[75]*B[170] + W[90]*B[171] + W[105]*B[172] + W[120]*B[173] + W[135]*B[174] + W[150]*B[175] + W[165]*B[176] + W[180]*B[177] + W[195]*B[178] + W[210]*B[179];
  A[166] = W[1]*B[165] + W[16]*B[166] + W[31]*B[167] + W[46]*B[168] + W[61]*B[169] + W[76]*B[170] + W[91]*B[171] + W[106]*B[172] + W[121]*B[173] + W[136]*B[174] + W[151]*B[175] + W[166]*B[176] + W[181]*B[177] + W[196]*B[178] + W[211]*B[179];
  A[167] = W[2]*B[165] + W[17]*B[166] + W[32]*B[167] + W[47]*B[168] + W[62]*B[169] + W[77]*B[170] + W[92]*B[171] + W[107]*B[172] + W[122]*B[173] + W[137]*B[174] + W[152]*B[175] + W[167]*B[176] + W[182]*B[177] + W[197]*B[178] + W[212]*B[179];
  A[168] = W[3]*B[165] + W[18]*B[166] + W[33]*B[167] + W[48]*B[168] + W[63]*B[169] + W[78]*B[170] + W[93]*B[171] + W[108]*B[172] + W[123]*B[173] + W[138]*B[174] + W[153]*B[175] + W[168]*B[176] + W[183]*B[177] + W[198]*B[178] + W[213]*B[179];
  A[169] = W[4]*B[165] + W[19]*B[166] + W[34]*B[167] + W[49]*B[168] + W[64]*B[169] + W[79]*B[170] + W[94]*B[171] + W[109]*B[172] + W[124]*B[173] + W[139]*B[174] + W[154]*B[175] + W[169]*B[176] + W[184]*B[177] + W[199]*B[178] + W[214]*B[179];
  A[170] = W[5]*B[165] + W[20]*B[166] + W[35]*B[167] + W[50]*B[168] + W[65]*B[169] + W[80]*B[170] + W[95]*B[171] + W[110]*B[172] + W[125]*B[173] + W[140]*B[174] + W[155]*B[175] + W[170]*B[176] + W[185]*B[177] + W[200]*B[178] + W[215]*B[179];
  A[171] = W[6]*B[165] + W[21]*B[166] + W[36]*B[167] + W[51]*B[168] + W[66]*B[169] + W[81]*B[170] + W[96]*B[171] + W[111]*B[172] + W[126]*B[173] + W[141]*B[174] + W[156]*B[175] + W[171]*B[176] + W[186]*B[177] + W[201]*B[178] + W[216]*B[179];
  A[172] = W[7]*B[165] + W[22]*B[166] + W[37]*B[167] + W[52]*B[168] + W[67]*B[169] + W[82]*B[170] + W[97]*B[171] + W[112]*B[172] + W[127]*B[173] + W[142]*B[174] + W[157]*B[175] + W[172]*B[176] + W[187]*B[177] + W[202]*B[178] + W[217]*B[179];
  A[173] = W[8]*B[165] + W[23]*B[166] + W[38]*B[167] + W[53]*B[168] + W[68]*B[169] + W[83]*B[170] + W[98]*B[171] + W[113]*B[172] + W[128]*B[173] + W[143]*B[174] + W[158]*B[175] + W[173]*B[176] + W[188]*B[177] + W[203]*B[178] + W[218]*B[179];
  A[174] = W[9]*B[165] + W[24]*B[166] + W[39]*B[167] + W[54]*B[168] + W[69]*B[169] + W[84]*B[170] + W[99]*B[171] + W[114]*B[172] + W[129]*B[173] + W[144]*B[174] + W[159]*B[175] + W[174]*B[176] + W[189]*B[177] + W[204]*B[178] + W[219]*B[179];
  A[175] = W[10]*B[165] + W[25]*B[166] + W[40]*B[167] + W[55]*B[168] + W[70]*B[169] + W[85]*B[170] + W[100]*B[171] + W[115]*B[172] + W[130]*B[173] + W[145]*B[174] + W[160]*B[175] + W[175]*B[176] + W[190]*B[177] + W[205]*B[178] + W[220]*B[179];
  A[176] = W[11]*B[165] + W[26]*B[166] + W[41]*B[167] + W[56]*B[168] + W[71]*B[169] + W[86]*B[170] + W[101]*B[171] + W[116]*B[172] + W[131]*B[173] + W[146]*B[174] + W[161]*B[175] + W[176]*B[176] + W[191]*B[177] + W[206]*B[178] + W[221]*B[179];
  A[177] = W[12]*B[165] + W[27]*B[166] + W[42]*B[167] + W[57]*B[168] + W[72]*B[169] + W[87]*B[170] + W[102]*B[171] + W[117]*B[172] + W[132]*B[173] + W[147]*B[174] + W[162]*B[175] + W[177]*B[176] + W[192]*B[177] + W[207]*B[178] + W[222]*B[179];
  A[178] = W[13]*B[165] + W[28]*B[166] + W[43]*B[167] + W[58]*B[168] + W[73]*B[169] + W[88]*B[170] + W[103]*B[171] + W[118]*B[172] + W[133]*B[173] + W[148]*B[174] + W[163]*B[175] + W[178]*B[176] + W[193]*B[177] + W[208]*B[178] + W[223]*B[179];
  A[179] = W[14]*B[165] + W[29]*B[166] + W[44]*B[167] + W[59]*B[168] + W[74]*B[169] + W[89]*B[170] + W[104]*B[171] + W[119]*B[172] + W[134]*B[173] + W[149]*B[174] + W[164]*B[175] + W[179]*B[176] + W[194]*B[177] + W[209]*B[178] + W[224]*B[179];
  A[180] = W[0]*B[180] + W[15]*B[181] + W[30]*B[182] + W[45]*B[183] + W[60]*B[184] + W[75]*B[185] + W[90]*B[186] + W[105]*B[187] + W[120]*B[188] + W[135]*B[189] + W[150]*B[190] + W[165]*B[191] + W[180]*B[192] + W[195]*B[193] + W[210]*B[194];
  A[181] = W[1]*B[180] + W[16]*B[181] + W[31]*B[182] + W[46]*B[183] + W[61]*B[184] + W[76]*B[185] + W[91]*B[186] + W[106]*B[187] + W[121]*B[188] + W[136]*B[189] + W[151]*B[190] + W[166]*B[191] + W[181]*B[192] + W[196]*B[193] + W[211]*B[194];
  A[182] = W[2]*B[180] + W[17]*B[181] + W[32]*B[182] + W[47]*B[183] + W[62]*B[184] + W[77]*B[185] + W[92]*B[186] + W[107]*B[187] + W[122]*B[188] + W[137]*B[189] + W[152]*B[190] + W[167]*B[191] + W[182]*B[192] + W[197]*B[193] + W[212]*B[194];
  A[183] = W[3]*B[180] + W[18]*B[181] + W[33]*B[182] + W[48]*B[183] + W[63]*B[184] + W[78]*B[185] + W[93]*B[186] + W[108]*B[187] + W[123]*B[188] + W[138]*B[189] + W[153]*B[190] + W[168]*B[191] + W[183]*B[192] + W[198]*B[193] + W[213]*B[194];
  A[184] = W[4]*B[180] + W[19]*B[181] + W[34]*B[182] + W[49]*B[183] + W[64]*B[184] + W[79]*B[185] + W[94]*B[186] + W[109]*B[187] + W[124]*B[188] + W[139]*B[189] + W[154]*B[190] + W[169]*B[191] + W[184]*B[192] + W[199]*B[193] + W[214]*B[194];
  A[185] = W[5]*B[180] + W[20]*B[181] + W[35]*B[182] + W[50]*B[183] + W[65]*B[184] + W[80]*B[185] + W[95]*B[186] + W[110]*B[187] + W[125]*B[188] + W[140]*B[189] + W[155]*B[190] + W[170]*B[191] + W[185]*B[192] + W[200]*B[193] + W[215]*B[194];
  A[186] = W[6]*B[180] + W[21]*B[181] + W[36]*B[182] + W[51]*B[183] + W[66]*B[184] + W[81]*B[185] + W[96]*B[186] + W[111]*B[187] + W[126]*B[188] + W[141]*B[189] + W[156]*B[190] + W[171]*B[191] + W[186]*B[192] + W[201]*B[193] + W[216]*B[194];
  A[187] = W[7]*B[180] + W[22]*B[181] + W[37]*B[182] + W[52]*B[183] + W[67]*B[184] + W[82]*B[185] + W[97]*B[186] + W[112]*B[187] + W[127]*B[188] + W[142]*B[189] + W[157]*B[190] + W[172]*B[191] + W[187]*B[192] + W[202]*B[193] + W[217]*B[194];
  A[188] = W[8]*B[180] + W[23]*B[181] + W[38]*B[182] + W[53]*B[183] + W[68]*B[184] + W[83]*B[185] + W[98]*B[186] + W[113]*B[187] + W[128]*B[188] + W[143]*B[189] + W[158]*B[190] + W[173]*B[191] + W[188]*B[192] + W[203]*B[193] + W[218]*B[194];
  A[189] = W[9]*B[180] + W[24]*B[181] + W[39]*B[182] + W[54]*B[183] + W[69]*B[184] + W[84]*B[185] + W[99]*B[186] + W[114]*B[187] + W[129]*B[188] + W[144]*B[189] + W[159]*B[190] + W[174]*B[191] + W[189]*B[192] + W[204]*B[193] + W[219]*B[194];
  A[190] = W[10]*B[180] + W[25]*B[181] + W[40]*B[182] + W[55]*B[183] + W[70]*B[184] + W[85]*B[185] + W[100]*B[186] + W[115]*B[187] + W[130]*B[188] + W[145]*B[189] + W[160]*B[190] + W[175]*B[191] + W[190]*B[192] + W[205]*B[193] + W[220]*B[194];
  A[191] = W[11]*B[180] + W[26]*B[181] + W[41]*B[182] + W[56]*B[183] + W[71]*B[184] + W[86]*B[185] + W[101]*B[186] + W[116]*B[187] + W[131]*B[188] + W[146]*B[189] + W[161]*B[190] + W[176]*B[191] + W[191]*B[192] + W[206]*B[193] + W[221]*B[194];
  A[192] = W[12]*B[180] + W[27]*B[181] + W[42]*B[182] + W[57]*B[183] + W[72]*B[184] + W[87]*B[185] + W[102]*B[186] + W[117]*B[187] + W[132]*B[188] + W[147]*B[189] + W[162]*B[190] + W[177]*B[191] + W[192]*B[192] + W[207]*B[193] + W[222]*B[194];
  A[193] = W[13]*B[180] + W[28]*B[181] + W[43]*B[182] + W[58]*B[183] + W[73]*B[184] + W[88]*B[185] + W[103]*B[186] + W[118]*B[187] + W[133]*B[188] + W[148]*B[189] + W[163]*B[190] + W[178]*B[191] + W[193]*B[192] + W[208]*B[193] + W[223]*B[194];
  A[194] = W[14]*B[180] + W[29]*B[181] + W[44]*B[182] + W[59]*B[183] + W[74]*B[184] + W[89]*B[185] + W[104]*B[186] + W[119]*B[187] + W[134]*B[188] + W[149]*B[189] + W[164]*B[190] + W[179]*B[191] + W[194]*B[192] + W[209]*B[193] + W[224]*B[194];
  A[195] = W[0]*B[195] + W[15]*B[196] + W[30]*B[197] + W[45]*B[198] + W[60]*B[199] + W[75]*B[200] + W[90]*B[201] + W[105]*B[202] + W[120]*B[203] + W[135]*B[204] + W[150]*B[205] + W[165]*B[206] + W[180]*B[207] + W[195]*B[208] + W[210]*B[209];
  A[196] = W[1]*B[195] + W[16]*B[196] + W[31]*B[197] + W[46]*B[198] + W[61]*B[199] + W[76]*B[200] + W[91]*B[201] + W[106]*B[202] + W[121]*B[203] + W[136]*B[204] + W[151]*B[205] + W[166]*B[206] + W[181]*B[207] + W[196]*B[208] + W[211]*B[209];
  A[197] = W[2]*B[195] + W[17]*B[196] + W[32]*B[197] + W[47]*B[198] + W[62]*B[199] + W[77]*B[200] + W[92]*B[201] + W[107]*B[202] + W[122]*B[203] + W[137]*B[204] + W[152]*B[205] + W[167]*B[206] + W[182]*B[207] + W[197]*B[208] + W[212]*B[209];
  A[198] = W[3]*B[195] + W[18]*B[196] + W[33]*B[197] + W[48]*B[198] + W[63]*B[199] + W[78]*B[200] + W[93]*B[201] + W[108]*B[202] + W[123]*B[203] + W[138]*B[204] + W[153]*B[205] + W[168]*B[206] + W[183]*B[207] + W[198]*B[208] + W[213]*B[209];
  A[199] = W[4]*B[195] + W[19]*B[196] + W[34]*B[197] + W[49]*B[198] + W[64]*B[199] + W[79]*B[200] + W[94]*B[201] + W[109]*B[202] + W[124]*B[203] + W[139]*B[204] + W[154]*B[205] + W[169]*B[206] + W[184]*B[207] + W[199]*B[208] + W[214]*B[209];
  A[200] = W[5]*B[195] + W[20]*B[196] + W[35]*B[197] + W[50]*B[198] + W[65]*B[199] + W[80]*B[200] + W[95]*B[201] + W[110]*B[202] + W[125]*B[203] + W[140]*B[204] + W[155]*B[205] + W[170]*B[206] + W[185]*B[207] + W[200]*B[208] + W[215]*B[209];
  A[201] = W[6]*B[195] + W[21]*B[196] + W[36]*B[197] + W[51]*B[198] + W[66]*B[199] + W[81]*B[200] + W[96]*B[201] + W[111]*B[202] + W[126]*B[203] + W[141]*B[204] + W[156]*B[205] + W[171]*B[206] + W[186]*B[207] + W[201]*B[208] + W[216]*B[209];
  A[202] = W[7]*B[195] + W[22]*B[196] + W[37]*B[197] + W[52]*B[198] + W[67]*B[199] + W[82]*B[200] + W[97]*B[201] + W[112]*B[202] + W[127]*B[203] + W[142]*B[204] + W[157]*B[205] + W[172]*B[206] + W[187]*B[207] + W[202]*B[208] + W[217]*B[209];
  A[203] = W[8]*B[195] + W[23]*B[196] + W[38]*B[197] + W[53]*B[198] + W[68]*B[199] + W[83]*B[200] + W[98]*B[201] + W[113]*B[202] + W[128]*B[203] + W[143]*B[204] + W[158]*B[205] + W[173]*B[206] + W[188]*B[207] + W[203]*B[208] + W[218]*B[209];
  A[204] = W[9]*B[195] + W[24]*B[196] + W[39]*B[197] + W[54]*B[198] + W[69]*B[199] + W[84]*B[200] + W[99]*B[201] + W[114]*B[202] + W[129]*B[203] + W[144]*B[204] + W[159]*B[205] + W[174]*B[206] + W[189]*B[207] + W[204]*B[208] + W[219]*B[209];
  A[205] = W[10]*B[195] + W[25]*B[196] + W[40]*B[197] + W[55]*B[198] + W[70]*B[199] + W[85]*B[200] + W[100]*B[201] + W[115]*B[202] + W[130]*B[203] + W[145]*B[204] + W[160]*B[205] + W[175]*B[206] + W[190]*B[207] + W[205]*B[208] + W[220]*B[209];
  A[206] = W[11]*B[195] + W[26]*B[196] + W[41]*B[197] + W[56]*B[198] + W[71]*B[199] + W[86]*B[200] + W[101]*B[201] + W[116]*B[202] + W[131]*B[203] + W[146]*B[204] + W[161]*B[205] + W[176]*B[206] + W[191]*B[207] + W[206]*B[208] + W[221]*B[209];
  A[207] = W[12]*B[195] + W[27]*B[196] + W[42]*B[197] + W[57]*B[198] + W[72]*B[199] + W[87]*B[200] + W[102]*B[201] + W[117]*B[202] + W[132]*B[203] + W[147]*B[204] + W[162]*B[205] + W[177]*B[206] + W[192]*B[207] + W[207]*B[208] + W[222]*B[209];
  A[208] = W[13]*B[195] + W[28]*B[196] + W[43]*B[197] + W[58]*B[198] + W[73]*B[199] + W[88]*B[200] + W[103]*B[201] + W[118]*B[202] + W[133]*B[203] + W[148]*B[204] + W[163]*B[205] + W[178]*B[206] + W[193]*B[207] + W[208]*B[208] + W[223]*B[209];
  A[209] = W[14]*B[195] + W[29]*B[196] + W[44]*B[197] + W[59]*B[198] + W[74]*B[199] + W[89]*B[200] + W[104]*B[201] + W[119]*B[202] + W[134]*B[203] + W[149]*B[204] + W[164]*B[205] + W[179]*B[206] + W[194]*B[207] + W[209]*B[208] + W[224]*B[209];
  A[210] = W[0]*B[210] + W[15]*B[211] + W[30]*B[212] + W[45]*B[213] + W[60]*B[214] + W[75]*B[215] + W[90]*B[216] + W[105]*B[217] + W[120]*B[218] + W[135]*B[219] + W[150]*B[220] + W[165]*B[221] + W[180]*B[222] + W[195]*B[223] + W[210]*B[224];
  A[211] = W[1]*B[210] + W[16]*B[211] + W[31]*B[212] + W[46]*B[213] + W[61]*B[214] + W[76]*B[215] + W[91]*B[216] + W[106]*B[217] + W[121]*B[218] + W[136]*B[219] + W[151]*B[220] + W[166]*B[221] + W[181]*B[222] + W[196]*B[223] + W[211]*B[224];
  A[212] = W[2]*B[210] + W[17]*B[211] + W[32]*B[212] + W[47]*B[213] + W[62]*B[214] + W[77]*B[215] + W[92]*B[216] + W[107]*B[217] + W[122]*B[218] + W[137]*B[219] + W[152]*B[220] + W[167]*B[221] + W[182]*B[222] + W[197]*B[223] + W[212]*B[224];
  A[213] = W[3]*B[210] + W[18]*B[211] + W[33]*B[212] + W[48]*B[213] + W[63]*B[214] + W[78]*B[215] + W[93]*B[216] + W[108]*B[217] + W[123]*B[218] + W[138]*B[219] + W[153]*B[220] + W[168]*B[221] + W[183]*B[222] + W[198]*B[223] + W[213]*B[224];
  A[214] = W[4]*B[210] + W[19]*B[211] + W[34]*B[212] + W[49]*B[213] + W[64]*B[214] + W[79]*B[215] + W[94]*B[216] + W[109]*B[217] + W[124]*B[218] + W[139]*B[219] + W[154]*B[220] + W[169]*B[221] + W[184]*B[222] + W[199]*B[223] + W[214]*B[224];
  A[215] = W[5]*B[210] + W[20]*B[211] + W[35]*B[212] + W[50]*B[213] + W[65]*B[214] + W[80]*B[215] + W[95]*B[216] + W[110]*B[217] + W[125]*B[218] + W[140]*B[219] + W[155]*B[220] + W[170]*B[221] + W[185]*B[222] + W[200]*B[223] + W[215]*B[224];
  A[216] = W[6]*B[210] + W[21]*B[211] + W[36]*B[212] + W[51]*B[213] + W[66]*B[214] + W[81]*B[215] + W[96]*B[216] + W[111]*B[217] + W[126]*B[218] + W[141]*B[219] + W[156]*B[220] + W[171]*B[221] + W[186]*B[222] + W[201]*B[223] + W[216]*B[224];
  A[217] = W[7]*B[210] + W[22]*B[211] + W[37]*B[212] + W[52]*B[213] + W[67]*B[214] + W[82]*B[215] + W[97]*B[216] + W[112]*B[217] + W[127]*B[218] + W[142]*B[219] + W[157]*B[220] + W[172]*B[221] + W[187]*B[222] + W[202]*B[223] + W[217]*B[224];
  A[218] = W[8]*B[210] + W[23]*B[211] + W[38]*B[212] + W[53]*B[213] + W[68]*B[214] + W[83]*B[215] + W[98]*B[216] + W[113]*B[217] + W[128]*B[218] + W[143]*B[219] + W[158]*B[220] + W[173]*B[221] + W[188]*B[222] + W[203]*B[223] + W[218]*B[224];
  A[219] = W[9]*B[210] + W[24]*B[211] + W[39]*B[212] + W[54]*B[213] + W[69]*B[214] + W[84]*B[215] + W[99]*B[216] + W[114]*B[217] + W[129]*B[218] + W[144]*B[219] + W[159]*B[220] + W[174]*B[221] + W[189]*B[222] + W[204]*B[223] + W[219]*B[224];
  A[220] = W[10]*B[210] + W[25]*B[211] + W[40]*B[212] + W[55]*B[213] + W[70]*B[214] + W[85]*B[215] + W[100]*B[216] + W[115]*B[217] + W[130]*B[218] + W[145]*B[219] + W[160]*B[220] + W[175]*B[221] + W[190]*B[222] + W[205]*B[223] + W[220]*B[224];
  A[221] = W[11]*B[210] + W[26]*B[211] + W[41]*B[212] + W[56]*B[213] + W[71]*B[214] + W[86]*B[215] + W[101]*B[216] + W[116]*B[217] + W[131]*B[218] + W[146]*B[219] + W[161]*B[220] + W[176]*B[221] + W[191]*B[222] + W[206]*B[223] + W[221]*B[224];
  A[222] = W[12]*B[210] + W[27]*B[211] + W[42]*B[212] + W[57]*B[213] + W[72]*B[214] + W[87]*B[215] + W[102]*B[216] + W[117]*B[217] + W[132]*B[218] + W[147]*B[219] + W[162]*B[220] + W[177]*B[221] + W[192]*B[222] + W[207]*B[223] + W[222]*B[224];
  A[223] = W[13]*B[210] + W[28]*B[211] + W[43]*B[212] + W[58]*B[213] + W[73]*B[214] + W[88]*B[215] + W[103]*B[216] + W[118]*B[217] + W[133]*B[218] + W[148]*B[219] + W[163]*B[220] + W[178]*B[221] + W[193]*B[222] + W[208]*B[223] + W[223]*B[224];
  A[224] = W[14]*B[210] + W[29]*B[211] + W[44]*B[212] + W[59]*B[213] + W[74]*B[214] + W[89]*B[215] + W[104]*B[216] + W[119]*B[217] + W[134]*B[218] + W[149]*B[219] + W[164]*B[220] + W[179]*B[221] + W[194]*B[222] + W[209]*B[223] + W[224]*B[224];
  return 0;
}

/*
  PetscKernel_A_gets_A_minus_B_times_C_15: A = A - B * C with size bs=15

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

PETSC_STATIC_INLINE PetscErrorCode PetscKernel_A_gets_A_minus_B_times_C_15(PetscScalar *A,const PetscScalar *B,const PetscScalar *C)
{
  A[0]   -= B[0]*C[0] + B[15]*C[1] + B[30]*C[2] + B[45]*C[3] + B[60]*C[4] + B[75]*C[5] + B[90]*C[6] + B[105]*C[7] + B[120]*C[8] + B[135]*C[9] + B[150]*C[10] + B[165]*C[11] + B[180]*C[12] + B[195]*C[13] + B[210]*C[14];
  A[1]   -= B[1]*C[0] + B[16]*C[1] + B[31]*C[2] + B[46]*C[3] + B[61]*C[4] + B[76]*C[5] + B[91]*C[6] + B[106]*C[7] + B[121]*C[8] + B[136]*C[9] + B[151]*C[10] + B[166]*C[11] + B[181]*C[12] + B[196]*C[13] + B[211]*C[14];
  A[2]   -= B[2]*C[0] + B[17]*C[1] + B[32]*C[2] + B[47]*C[3] + B[62]*C[4] + B[77]*C[5] + B[92]*C[6] + B[107]*C[7] + B[122]*C[8] + B[137]*C[9] + B[152]*C[10] + B[167]*C[11] + B[182]*C[12] + B[197]*C[13] + B[212]*C[14];
  A[3]   -= B[3]*C[0] + B[18]*C[1] + B[33]*C[2] + B[48]*C[3] + B[63]*C[4] + B[78]*C[5] + B[93]*C[6] + B[108]*C[7] + B[123]*C[8] + B[138]*C[9] + B[153]*C[10] + B[168]*C[11] + B[183]*C[12] + B[198]*C[13] + B[213]*C[14];
  A[4]   -= B[4]*C[0] + B[19]*C[1] + B[34]*C[2] + B[49]*C[3] + B[64]*C[4] + B[79]*C[5] + B[94]*C[6] + B[109]*C[7] + B[124]*C[8] + B[139]*C[9] + B[154]*C[10] + B[169]*C[11] + B[184]*C[12] + B[199]*C[13] + B[214]*C[14];
  A[5]   -= B[5]*C[0] + B[20]*C[1] + B[35]*C[2] + B[50]*C[3] + B[65]*C[4] + B[80]*C[5] + B[95]*C[6] + B[110]*C[7] + B[125]*C[8] + B[140]*C[9] + B[155]*C[10] + B[170]*C[11] + B[185]*C[12] + B[200]*C[13] + B[215]*C[14];
  A[6]   -= B[6]*C[0] + B[21]*C[1] + B[36]*C[2] + B[51]*C[3] + B[66]*C[4] + B[81]*C[5] + B[96]*C[6] + B[111]*C[7] + B[126]*C[8] + B[141]*C[9] + B[156]*C[10] + B[171]*C[11] + B[186]*C[12] + B[201]*C[13] + B[216]*C[14];
  A[7]   -= B[7]*C[0] + B[22]*C[1] + B[37]*C[2] + B[52]*C[3] + B[67]*C[4] + B[82]*C[5] + B[97]*C[6] + B[112]*C[7] + B[127]*C[8] + B[142]*C[9] + B[157]*C[10] + B[172]*C[11] + B[187]*C[12] + B[202]*C[13] + B[217]*C[14];
  A[8]   -= B[8]*C[0] + B[23]*C[1] + B[38]*C[2] + B[53]*C[3] + B[68]*C[4] + B[83]*C[5] + B[98]*C[6] + B[113]*C[7] + B[128]*C[8] + B[143]*C[9] + B[158]*C[10] + B[173]*C[11] + B[188]*C[12] + B[203]*C[13] + B[218]*C[14];
  A[9]   -= B[9]*C[0] + B[24]*C[1] + B[39]*C[2] + B[54]*C[3] + B[69]*C[4] + B[84]*C[5] + B[99]*C[6] + B[114]*C[7] + B[129]*C[8] + B[144]*C[9] + B[159]*C[10] + B[174]*C[11] + B[189]*C[12] + B[204]*C[13] + B[219]*C[14];
  A[10]  -= B[10]*C[0] + B[25]*C[1] + B[40]*C[2] + B[55]*C[3] + B[70]*C[4] + B[85]*C[5] + B[100]*C[6] + B[115]*C[7] + B[130]*C[8] + B[145]*C[9] + B[160]*C[10] + B[175]*C[11] + B[190]*C[12] + B[205]*C[13] + B[220]*C[14];
  A[11]  -= B[11]*C[0] + B[26]*C[1] + B[41]*C[2] + B[56]*C[3] + B[71]*C[4] + B[86]*C[5] + B[101]*C[6] + B[116]*C[7] + B[131]*C[8] + B[146]*C[9] + B[161]*C[10] + B[176]*C[11] + B[191]*C[12] + B[206]*C[13] + B[221]*C[14];
  A[12]  -= B[12]*C[0] + B[27]*C[1] + B[42]*C[2] + B[57]*C[3] + B[72]*C[4] + B[87]*C[5] + B[102]*C[6] + B[117]*C[7] + B[132]*C[8] + B[147]*C[9] + B[162]*C[10] + B[177]*C[11] + B[192]*C[12] + B[207]*C[13] + B[222]*C[14];
  A[13]  -= B[13]*C[0] + B[28]*C[1] + B[43]*C[2] + B[58]*C[3] + B[73]*C[4] + B[88]*C[5] + B[103]*C[6] + B[118]*C[7] + B[133]*C[8] + B[148]*C[9] + B[163]*C[10] + B[178]*C[11] + B[193]*C[12] + B[208]*C[13] + B[223]*C[14];
  A[14]  -= B[14]*C[0] + B[29]*C[1] + B[44]*C[2] + B[59]*C[3] + B[74]*C[4] + B[89]*C[5] + B[104]*C[6] + B[119]*C[7] + B[134]*C[8] + B[149]*C[9] + B[164]*C[10] + B[179]*C[11] + B[194]*C[12] + B[209]*C[13] + B[224]*C[14];
  A[15]  -= B[0]*C[15] + B[15]*C[16] + B[30]*C[17] + B[45]*C[18] + B[60]*C[19] + B[75]*C[20] + B[90]*C[21] + B[105]*C[22] + B[120]*C[23] + B[135]*C[24] + B[150]*C[25] + B[165]*C[26] + B[180]*C[27] + B[195]*C[28] + B[210]*C[29];
  A[16]  -= B[1]*C[15] + B[16]*C[16] + B[31]*C[17] + B[46]*C[18] + B[61]*C[19] + B[76]*C[20] + B[91]*C[21] + B[106]*C[22] + B[121]*C[23] + B[136]*C[24] + B[151]*C[25] + B[166]*C[26] + B[181]*C[27] + B[196]*C[28] + B[211]*C[29];
  A[17]  -= B[2]*C[15] + B[17]*C[16] + B[32]*C[17] + B[47]*C[18] + B[62]*C[19] + B[77]*C[20] + B[92]*C[21] + B[107]*C[22] + B[122]*C[23] + B[137]*C[24] + B[152]*C[25] + B[167]*C[26] + B[182]*C[27] + B[197]*C[28] + B[212]*C[29];
  A[18]  -= B[3]*C[15] + B[18]*C[16] + B[33]*C[17] + B[48]*C[18] + B[63]*C[19] + B[78]*C[20] + B[93]*C[21] + B[108]*C[22] + B[123]*C[23] + B[138]*C[24] + B[153]*C[25] + B[168]*C[26] + B[183]*C[27] + B[198]*C[28] + B[213]*C[29];
  A[19]  -= B[4]*C[15] + B[19]*C[16] + B[34]*C[17] + B[49]*C[18] + B[64]*C[19] + B[79]*C[20] + B[94]*C[21] + B[109]*C[22] + B[124]*C[23] + B[139]*C[24] + B[154]*C[25] + B[169]*C[26] + B[184]*C[27] + B[199]*C[28] + B[214]*C[29];
  A[20]  -= B[5]*C[15] + B[20]*C[16] + B[35]*C[17] + B[50]*C[18] + B[65]*C[19] + B[80]*C[20] + B[95]*C[21] + B[110]*C[22] + B[125]*C[23] + B[140]*C[24] + B[155]*C[25] + B[170]*C[26] + B[185]*C[27] + B[200]*C[28] + B[215]*C[29];
  A[21]  -= B[6]*C[15] + B[21]*C[16] + B[36]*C[17] + B[51]*C[18] + B[66]*C[19] + B[81]*C[20] + B[96]*C[21] + B[111]*C[22] + B[126]*C[23] + B[141]*C[24] + B[156]*C[25] + B[171]*C[26] + B[186]*C[27] + B[201]*C[28] + B[216]*C[29];
  A[22]  -= B[7]*C[15] + B[22]*C[16] + B[37]*C[17] + B[52]*C[18] + B[67]*C[19] + B[82]*C[20] + B[97]*C[21] + B[112]*C[22] + B[127]*C[23] + B[142]*C[24] + B[157]*C[25] + B[172]*C[26] + B[187]*C[27] + B[202]*C[28] + B[217]*C[29];
  A[23]  -= B[8]*C[15] + B[23]*C[16] + B[38]*C[17] + B[53]*C[18] + B[68]*C[19] + B[83]*C[20] + B[98]*C[21] + B[113]*C[22] + B[128]*C[23] + B[143]*C[24] + B[158]*C[25] + B[173]*C[26] + B[188]*C[27] + B[203]*C[28] + B[218]*C[29];
  A[24]  -= B[9]*C[15] + B[24]*C[16] + B[39]*C[17] + B[54]*C[18] + B[69]*C[19] + B[84]*C[20] + B[99]*C[21] + B[114]*C[22] + B[129]*C[23] + B[144]*C[24] + B[159]*C[25] + B[174]*C[26] + B[189]*C[27] + B[204]*C[28] + B[219]*C[29];
  A[25]  -= B[10]*C[15] + B[25]*C[16] + B[40]*C[17] + B[55]*C[18] + B[70]*C[19] + B[85]*C[20] + B[100]*C[21] + B[115]*C[22] + B[130]*C[23] + B[145]*C[24] + B[160]*C[25] + B[175]*C[26] + B[190]*C[27] + B[205]*C[28] + B[220]*C[29];
  A[26]  -= B[11]*C[15] + B[26]*C[16] + B[41]*C[17] + B[56]*C[18] + B[71]*C[19] + B[86]*C[20] + B[101]*C[21] + B[116]*C[22] + B[131]*C[23] + B[146]*C[24] + B[161]*C[25] + B[176]*C[26] + B[191]*C[27] + B[206]*C[28] + B[221]*C[29];
  A[27]  -= B[12]*C[15] + B[27]*C[16] + B[42]*C[17] + B[57]*C[18] + B[72]*C[19] + B[87]*C[20] + B[102]*C[21] + B[117]*C[22] + B[132]*C[23] + B[147]*C[24] + B[162]*C[25] + B[177]*C[26] + B[192]*C[27] + B[207]*C[28] + B[222]*C[29];
  A[28]  -= B[13]*C[15] + B[28]*C[16] + B[43]*C[17] + B[58]*C[18] + B[73]*C[19] + B[88]*C[20] + B[103]*C[21] + B[118]*C[22] + B[133]*C[23] + B[148]*C[24] + B[163]*C[25] + B[178]*C[26] + B[193]*C[27] + B[208]*C[28] + B[223]*C[29];
  A[29]  -= B[14]*C[15] + B[29]*C[16] + B[44]*C[17] + B[59]*C[18] + B[74]*C[19] + B[89]*C[20] + B[104]*C[21] + B[119]*C[22] + B[134]*C[23] + B[149]*C[24] + B[164]*C[25] + B[179]*C[26] + B[194]*C[27] + B[209]*C[28] + B[224]*C[29];
  A[30]  -= B[0]*C[30] + B[15]*C[31] + B[30]*C[32] + B[45]*C[33] + B[60]*C[34] + B[75]*C[35] + B[90]*C[36] + B[105]*C[37] + B[120]*C[38] + B[135]*C[39] + B[150]*C[40] + B[165]*C[41] + B[180]*C[42] + B[195]*C[43] + B[210]*C[44];
  A[31]  -= B[1]*C[30] + B[16]*C[31] + B[31]*C[32] + B[46]*C[33] + B[61]*C[34] + B[76]*C[35] + B[91]*C[36] + B[106]*C[37] + B[121]*C[38] + B[136]*C[39] + B[151]*C[40] + B[166]*C[41] + B[181]*C[42] + B[196]*C[43] + B[211]*C[44];
  A[32]  -= B[2]*C[30] + B[17]*C[31] + B[32]*C[32] + B[47]*C[33] + B[62]*C[34] + B[77]*C[35] + B[92]*C[36] + B[107]*C[37] + B[122]*C[38] + B[137]*C[39] + B[152]*C[40] + B[167]*C[41] + B[182]*C[42] + B[197]*C[43] + B[212]*C[44];
  A[33]  -= B[3]*C[30] + B[18]*C[31] + B[33]*C[32] + B[48]*C[33] + B[63]*C[34] + B[78]*C[35] + B[93]*C[36] + B[108]*C[37] + B[123]*C[38] + B[138]*C[39] + B[153]*C[40] + B[168]*C[41] + B[183]*C[42] + B[198]*C[43] + B[213]*C[44];
  A[34]  -= B[4]*C[30] + B[19]*C[31] + B[34]*C[32] + B[49]*C[33] + B[64]*C[34] + B[79]*C[35] + B[94]*C[36] + B[109]*C[37] + B[124]*C[38] + B[139]*C[39] + B[154]*C[40] + B[169]*C[41] + B[184]*C[42] + B[199]*C[43] + B[214]*C[44];
  A[35]  -= B[5]*C[30] + B[20]*C[31] + B[35]*C[32] + B[50]*C[33] + B[65]*C[34] + B[80]*C[35] + B[95]*C[36] + B[110]*C[37] + B[125]*C[38] + B[140]*C[39] + B[155]*C[40] + B[170]*C[41] + B[185]*C[42] + B[200]*C[43] + B[215]*C[44];
  A[36]  -= B[6]*C[30] + B[21]*C[31] + B[36]*C[32] + B[51]*C[33] + B[66]*C[34] + B[81]*C[35] + B[96]*C[36] + B[111]*C[37] + B[126]*C[38] + B[141]*C[39] + B[156]*C[40] + B[171]*C[41] + B[186]*C[42] + B[201]*C[43] + B[216]*C[44];
  A[37]  -= B[7]*C[30] + B[22]*C[31] + B[37]*C[32] + B[52]*C[33] + B[67]*C[34] + B[82]*C[35] + B[97]*C[36] + B[112]*C[37] + B[127]*C[38] + B[142]*C[39] + B[157]*C[40] + B[172]*C[41] + B[187]*C[42] + B[202]*C[43] + B[217]*C[44];
  A[38]  -= B[8]*C[30] + B[23]*C[31] + B[38]*C[32] + B[53]*C[33] + B[68]*C[34] + B[83]*C[35] + B[98]*C[36] + B[113]*C[37] + B[128]*C[38] + B[143]*C[39] + B[158]*C[40] + B[173]*C[41] + B[188]*C[42] + B[203]*C[43] + B[218]*C[44];
  A[39]  -= B[9]*C[30] + B[24]*C[31] + B[39]*C[32] + B[54]*C[33] + B[69]*C[34] + B[84]*C[35] + B[99]*C[36] + B[114]*C[37] + B[129]*C[38] + B[144]*C[39] + B[159]*C[40] + B[174]*C[41] + B[189]*C[42] + B[204]*C[43] + B[219]*C[44];
  A[40]  -= B[10]*C[30] + B[25]*C[31] + B[40]*C[32] + B[55]*C[33] + B[70]*C[34] + B[85]*C[35] + B[100]*C[36] + B[115]*C[37] + B[130]*C[38] + B[145]*C[39] + B[160]*C[40] + B[175]*C[41] + B[190]*C[42] + B[205]*C[43] + B[220]*C[44];
  A[41]  -= B[11]*C[30] + B[26]*C[31] + B[41]*C[32] + B[56]*C[33] + B[71]*C[34] + B[86]*C[35] + B[101]*C[36] + B[116]*C[37] + B[131]*C[38] + B[146]*C[39] + B[161]*C[40] + B[176]*C[41] + B[191]*C[42] + B[206]*C[43] + B[221]*C[44];
  A[42]  -= B[12]*C[30] + B[27]*C[31] + B[42]*C[32] + B[57]*C[33] + B[72]*C[34] + B[87]*C[35] + B[102]*C[36] + B[117]*C[37] + B[132]*C[38] + B[147]*C[39] + B[162]*C[40] + B[177]*C[41] + B[192]*C[42] + B[207]*C[43] + B[222]*C[44];
  A[43]  -= B[13]*C[30] + B[28]*C[31] + B[43]*C[32] + B[58]*C[33] + B[73]*C[34] + B[88]*C[35] + B[103]*C[36] + B[118]*C[37] + B[133]*C[38] + B[148]*C[39] + B[163]*C[40] + B[178]*C[41] + B[193]*C[42] + B[208]*C[43] + B[223]*C[44];
  A[44]  -= B[14]*C[30] + B[29]*C[31] + B[44]*C[32] + B[59]*C[33] + B[74]*C[34] + B[89]*C[35] + B[104]*C[36] + B[119]*C[37] + B[134]*C[38] + B[149]*C[39] + B[164]*C[40] + B[179]*C[41] + B[194]*C[42] + B[209]*C[43] + B[224]*C[44];
  A[45]  -= B[0]*C[45] + B[15]*C[46] + B[30]*C[47] + B[45]*C[48] + B[60]*C[49] + B[75]*C[50] + B[90]*C[51] + B[105]*C[52] + B[120]*C[53] + B[135]*C[54] + B[150]*C[55] + B[165]*C[56] + B[180]*C[57] + B[195]*C[58] + B[210]*C[59];
  A[46]  -= B[1]*C[45] + B[16]*C[46] + B[31]*C[47] + B[46]*C[48] + B[61]*C[49] + B[76]*C[50] + B[91]*C[51] + B[106]*C[52] + B[121]*C[53] + B[136]*C[54] + B[151]*C[55] + B[166]*C[56] + B[181]*C[57] + B[196]*C[58] + B[211]*C[59];
  A[47]  -= B[2]*C[45] + B[17]*C[46] + B[32]*C[47] + B[47]*C[48] + B[62]*C[49] + B[77]*C[50] + B[92]*C[51] + B[107]*C[52] + B[122]*C[53] + B[137]*C[54] + B[152]*C[55] + B[167]*C[56] + B[182]*C[57] + B[197]*C[58] + B[212]*C[59];
  A[48]  -= B[3]*C[45] + B[18]*C[46] + B[33]*C[47] + B[48]*C[48] + B[63]*C[49] + B[78]*C[50] + B[93]*C[51] + B[108]*C[52] + B[123]*C[53] + B[138]*C[54] + B[153]*C[55] + B[168]*C[56] + B[183]*C[57] + B[198]*C[58] + B[213]*C[59];
  A[49]  -= B[4]*C[45] + B[19]*C[46] + B[34]*C[47] + B[49]*C[48] + B[64]*C[49] + B[79]*C[50] + B[94]*C[51] + B[109]*C[52] + B[124]*C[53] + B[139]*C[54] + B[154]*C[55] + B[169]*C[56] + B[184]*C[57] + B[199]*C[58] + B[214]*C[59];
  A[50]  -= B[5]*C[45] + B[20]*C[46] + B[35]*C[47] + B[50]*C[48] + B[65]*C[49] + B[80]*C[50] + B[95]*C[51] + B[110]*C[52] + B[125]*C[53] + B[140]*C[54] + B[155]*C[55] + B[170]*C[56] + B[185]*C[57] + B[200]*C[58] + B[215]*C[59];
  A[51]  -= B[6]*C[45] + B[21]*C[46] + B[36]*C[47] + B[51]*C[48] + B[66]*C[49] + B[81]*C[50] + B[96]*C[51] + B[111]*C[52] + B[126]*C[53] + B[141]*C[54] + B[156]*C[55] + B[171]*C[56] + B[186]*C[57] + B[201]*C[58] + B[216]*C[59];
  A[52]  -= B[7]*C[45] + B[22]*C[46] + B[37]*C[47] + B[52]*C[48] + B[67]*C[49] + B[82]*C[50] + B[97]*C[51] + B[112]*C[52] + B[127]*C[53] + B[142]*C[54] + B[157]*C[55] + B[172]*C[56] + B[187]*C[57] + B[202]*C[58] + B[217]*C[59];
  A[53]  -= B[8]*C[45] + B[23]*C[46] + B[38]*C[47] + B[53]*C[48] + B[68]*C[49] + B[83]*C[50] + B[98]*C[51] + B[113]*C[52] + B[128]*C[53] + B[143]*C[54] + B[158]*C[55] + B[173]*C[56] + B[188]*C[57] + B[203]*C[58] + B[218]*C[59];
  A[54]  -= B[9]*C[45] + B[24]*C[46] + B[39]*C[47] + B[54]*C[48] + B[69]*C[49] + B[84]*C[50] + B[99]*C[51] + B[114]*C[52] + B[129]*C[53] + B[144]*C[54] + B[159]*C[55] + B[174]*C[56] + B[189]*C[57] + B[204]*C[58] + B[219]*C[59];
  A[55]  -= B[10]*C[45] + B[25]*C[46] + B[40]*C[47] + B[55]*C[48] + B[70]*C[49] + B[85]*C[50] + B[100]*C[51] + B[115]*C[52] + B[130]*C[53] + B[145]*C[54] + B[160]*C[55] + B[175]*C[56] + B[190]*C[57] + B[205]*C[58] + B[220]*C[59];
  A[56]  -= B[11]*C[45] + B[26]*C[46] + B[41]*C[47] + B[56]*C[48] + B[71]*C[49] + B[86]*C[50] + B[101]*C[51] + B[116]*C[52] + B[131]*C[53] + B[146]*C[54] + B[161]*C[55] + B[176]*C[56] + B[191]*C[57] + B[206]*C[58] + B[221]*C[59];
  A[57]  -= B[12]*C[45] + B[27]*C[46] + B[42]*C[47] + B[57]*C[48] + B[72]*C[49] + B[87]*C[50] + B[102]*C[51] + B[117]*C[52] + B[132]*C[53] + B[147]*C[54] + B[162]*C[55] + B[177]*C[56] + B[192]*C[57] + B[207]*C[58] + B[222]*C[59];
  A[58]  -= B[13]*C[45] + B[28]*C[46] + B[43]*C[47] + B[58]*C[48] + B[73]*C[49] + B[88]*C[50] + B[103]*C[51] + B[118]*C[52] + B[133]*C[53] + B[148]*C[54] + B[163]*C[55] + B[178]*C[56] + B[193]*C[57] + B[208]*C[58] + B[223]*C[59];
  A[59]  -= B[14]*C[45] + B[29]*C[46] + B[44]*C[47] + B[59]*C[48] + B[74]*C[49] + B[89]*C[50] + B[104]*C[51] + B[119]*C[52] + B[134]*C[53] + B[149]*C[54] + B[164]*C[55] + B[179]*C[56] + B[194]*C[57] + B[209]*C[58] + B[224]*C[59];
  A[60]  -= B[0]*C[60] + B[15]*C[61] + B[30]*C[62] + B[45]*C[63] + B[60]*C[64] + B[75]*C[65] + B[90]*C[66] + B[105]*C[67] + B[120]*C[68] + B[135]*C[69] + B[150]*C[70] + B[165]*C[71] + B[180]*C[72] + B[195]*C[73] + B[210]*C[74];
  A[61]  -= B[1]*C[60] + B[16]*C[61] + B[31]*C[62] + B[46]*C[63] + B[61]*C[64] + B[76]*C[65] + B[91]*C[66] + B[106]*C[67] + B[121]*C[68] + B[136]*C[69] + B[151]*C[70] + B[166]*C[71] + B[181]*C[72] + B[196]*C[73] + B[211]*C[74];
  A[62]  -= B[2]*C[60] + B[17]*C[61] + B[32]*C[62] + B[47]*C[63] + B[62]*C[64] + B[77]*C[65] + B[92]*C[66] + B[107]*C[67] + B[122]*C[68] + B[137]*C[69] + B[152]*C[70] + B[167]*C[71] + B[182]*C[72] + B[197]*C[73] + B[212]*C[74];
  A[63]  -= B[3]*C[60] + B[18]*C[61] + B[33]*C[62] + B[48]*C[63] + B[63]*C[64] + B[78]*C[65] + B[93]*C[66] + B[108]*C[67] + B[123]*C[68] + B[138]*C[69] + B[153]*C[70] + B[168]*C[71] + B[183]*C[72] + B[198]*C[73] + B[213]*C[74];
  A[64]  -= B[4]*C[60] + B[19]*C[61] + B[34]*C[62] + B[49]*C[63] + B[64]*C[64] + B[79]*C[65] + B[94]*C[66] + B[109]*C[67] + B[124]*C[68] + B[139]*C[69] + B[154]*C[70] + B[169]*C[71] + B[184]*C[72] + B[199]*C[73] + B[214]*C[74];
  A[65]  -= B[5]*C[60] + B[20]*C[61] + B[35]*C[62] + B[50]*C[63] + B[65]*C[64] + B[80]*C[65] + B[95]*C[66] + B[110]*C[67] + B[125]*C[68] + B[140]*C[69] + B[155]*C[70] + B[170]*C[71] + B[185]*C[72] + B[200]*C[73] + B[215]*C[74];
  A[66]  -= B[6]*C[60] + B[21]*C[61] + B[36]*C[62] + B[51]*C[63] + B[66]*C[64] + B[81]*C[65] + B[96]*C[66] + B[111]*C[67] + B[126]*C[68] + B[141]*C[69] + B[156]*C[70] + B[171]*C[71] + B[186]*C[72] + B[201]*C[73] + B[216]*C[74];
  A[67]  -= B[7]*C[60] + B[22]*C[61] + B[37]*C[62] + B[52]*C[63] + B[67]*C[64] + B[82]*C[65] + B[97]*C[66] + B[112]*C[67] + B[127]*C[68] + B[142]*C[69] + B[157]*C[70] + B[172]*C[71] + B[187]*C[72] + B[202]*C[73] + B[217]*C[74];
  A[68]  -= B[8]*C[60] + B[23]*C[61] + B[38]*C[62] + B[53]*C[63] + B[68]*C[64] + B[83]*C[65] + B[98]*C[66] + B[113]*C[67] + B[128]*C[68] + B[143]*C[69] + B[158]*C[70] + B[173]*C[71] + B[188]*C[72] + B[203]*C[73] + B[218]*C[74];
  A[69]  -= B[9]*C[60] + B[24]*C[61] + B[39]*C[62] + B[54]*C[63] + B[69]*C[64] + B[84]*C[65] + B[99]*C[66] + B[114]*C[67] + B[129]*C[68] + B[144]*C[69] + B[159]*C[70] + B[174]*C[71] + B[189]*C[72] + B[204]*C[73] + B[219]*C[74];
  A[70]  -= B[10]*C[60] + B[25]*C[61] + B[40]*C[62] + B[55]*C[63] + B[70]*C[64] + B[85]*C[65] + B[100]*C[66] + B[115]*C[67] + B[130]*C[68] + B[145]*C[69] + B[160]*C[70] + B[175]*C[71] + B[190]*C[72] + B[205]*C[73] + B[220]*C[74];
  A[71]  -= B[11]*C[60] + B[26]*C[61] + B[41]*C[62] + B[56]*C[63] + B[71]*C[64] + B[86]*C[65] + B[101]*C[66] + B[116]*C[67] + B[131]*C[68] + B[146]*C[69] + B[161]*C[70] + B[176]*C[71] + B[191]*C[72] + B[206]*C[73] + B[221]*C[74];
  A[72]  -= B[12]*C[60] + B[27]*C[61] + B[42]*C[62] + B[57]*C[63] + B[72]*C[64] + B[87]*C[65] + B[102]*C[66] + B[117]*C[67] + B[132]*C[68] + B[147]*C[69] + B[162]*C[70] + B[177]*C[71] + B[192]*C[72] + B[207]*C[73] + B[222]*C[74];
  A[73]  -= B[13]*C[60] + B[28]*C[61] + B[43]*C[62] + B[58]*C[63] + B[73]*C[64] + B[88]*C[65] + B[103]*C[66] + B[118]*C[67] + B[133]*C[68] + B[148]*C[69] + B[163]*C[70] + B[178]*C[71] + B[193]*C[72] + B[208]*C[73] + B[223]*C[74];
  A[74]  -= B[14]*C[60] + B[29]*C[61] + B[44]*C[62] + B[59]*C[63] + B[74]*C[64] + B[89]*C[65] + B[104]*C[66] + B[119]*C[67] + B[134]*C[68] + B[149]*C[69] + B[164]*C[70] + B[179]*C[71] + B[194]*C[72] + B[209]*C[73] + B[224]*C[74];
  A[75]  -= B[0]*C[75] + B[15]*C[76] + B[30]*C[77] + B[45]*C[78] + B[60]*C[79] + B[75]*C[80] + B[90]*C[81] + B[105]*C[82] + B[120]*C[83] + B[135]*C[84] + B[150]*C[85] + B[165]*C[86] + B[180]*C[87] + B[195]*C[88] + B[210]*C[89];
  A[76]  -= B[1]*C[75] + B[16]*C[76] + B[31]*C[77] + B[46]*C[78] + B[61]*C[79] + B[76]*C[80] + B[91]*C[81] + B[106]*C[82] + B[121]*C[83] + B[136]*C[84] + B[151]*C[85] + B[166]*C[86] + B[181]*C[87] + B[196]*C[88] + B[211]*C[89];
  A[77]  -= B[2]*C[75] + B[17]*C[76] + B[32]*C[77] + B[47]*C[78] + B[62]*C[79] + B[77]*C[80] + B[92]*C[81] + B[107]*C[82] + B[122]*C[83] + B[137]*C[84] + B[152]*C[85] + B[167]*C[86] + B[182]*C[87] + B[197]*C[88] + B[212]*C[89];
  A[78]  -= B[3]*C[75] + B[18]*C[76] + B[33]*C[77] + B[48]*C[78] + B[63]*C[79] + B[78]*C[80] + B[93]*C[81] + B[108]*C[82] + B[123]*C[83] + B[138]*C[84] + B[153]*C[85] + B[168]*C[86] + B[183]*C[87] + B[198]*C[88] + B[213]*C[89];
  A[79]  -= B[4]*C[75] + B[19]*C[76] + B[34]*C[77] + B[49]*C[78] + B[64]*C[79] + B[79]*C[80] + B[94]*C[81] + B[109]*C[82] + B[124]*C[83] + B[139]*C[84] + B[154]*C[85] + B[169]*C[86] + B[184]*C[87] + B[199]*C[88] + B[214]*C[89];
  A[80]  -= B[5]*C[75] + B[20]*C[76] + B[35]*C[77] + B[50]*C[78] + B[65]*C[79] + B[80]*C[80] + B[95]*C[81] + B[110]*C[82] + B[125]*C[83] + B[140]*C[84] + B[155]*C[85] + B[170]*C[86] + B[185]*C[87] + B[200]*C[88] + B[215]*C[89];
  A[81]  -= B[6]*C[75] + B[21]*C[76] + B[36]*C[77] + B[51]*C[78] + B[66]*C[79] + B[81]*C[80] + B[96]*C[81] + B[111]*C[82] + B[126]*C[83] + B[141]*C[84] + B[156]*C[85] + B[171]*C[86] + B[186]*C[87] + B[201]*C[88] + B[216]*C[89];
  A[82]  -= B[7]*C[75] + B[22]*C[76] + B[37]*C[77] + B[52]*C[78] + B[67]*C[79] + B[82]*C[80] + B[97]*C[81] + B[112]*C[82] + B[127]*C[83] + B[142]*C[84] + B[157]*C[85] + B[172]*C[86] + B[187]*C[87] + B[202]*C[88] + B[217]*C[89];
  A[83]  -= B[8]*C[75] + B[23]*C[76] + B[38]*C[77] + B[53]*C[78] + B[68]*C[79] + B[83]*C[80] + B[98]*C[81] + B[113]*C[82] + B[128]*C[83] + B[143]*C[84] + B[158]*C[85] + B[173]*C[86] + B[188]*C[87] + B[203]*C[88] + B[218]*C[89];
  A[84]  -= B[9]*C[75] + B[24]*C[76] + B[39]*C[77] + B[54]*C[78] + B[69]*C[79] + B[84]*C[80] + B[99]*C[81] + B[114]*C[82] + B[129]*C[83] + B[144]*C[84] + B[159]*C[85] + B[174]*C[86] + B[189]*C[87] + B[204]*C[88] + B[219]*C[89];
  A[85]  -= B[10]*C[75] + B[25]*C[76] + B[40]*C[77] + B[55]*C[78] + B[70]*C[79] + B[85]*C[80] + B[100]*C[81] + B[115]*C[82] + B[130]*C[83] + B[145]*C[84] + B[160]*C[85] + B[175]*C[86] + B[190]*C[87] + B[205]*C[88] + B[220]*C[89];
  A[86]  -= B[11]*C[75] + B[26]*C[76] + B[41]*C[77] + B[56]*C[78] + B[71]*C[79] + B[86]*C[80] + B[101]*C[81] + B[116]*C[82] + B[131]*C[83] + B[146]*C[84] + B[161]*C[85] + B[176]*C[86] + B[191]*C[87] + B[206]*C[88] + B[221]*C[89];
  A[87]  -= B[12]*C[75] + B[27]*C[76] + B[42]*C[77] + B[57]*C[78] + B[72]*C[79] + B[87]*C[80] + B[102]*C[81] + B[117]*C[82] + B[132]*C[83] + B[147]*C[84] + B[162]*C[85] + B[177]*C[86] + B[192]*C[87] + B[207]*C[88] + B[222]*C[89];
  A[88]  -= B[13]*C[75] + B[28]*C[76] + B[43]*C[77] + B[58]*C[78] + B[73]*C[79] + B[88]*C[80] + B[103]*C[81] + B[118]*C[82] + B[133]*C[83] + B[148]*C[84] + B[163]*C[85] + B[178]*C[86] + B[193]*C[87] + B[208]*C[88] + B[223]*C[89];
  A[89]  -= B[14]*C[75] + B[29]*C[76] + B[44]*C[77] + B[59]*C[78] + B[74]*C[79] + B[89]*C[80] + B[104]*C[81] + B[119]*C[82] + B[134]*C[83] + B[149]*C[84] + B[164]*C[85] + B[179]*C[86] + B[194]*C[87] + B[209]*C[88] + B[224]*C[89];
  A[90]  -= B[0]*C[90] + B[15]*C[91] + B[30]*C[92] + B[45]*C[93] + B[60]*C[94] + B[75]*C[95] + B[90]*C[96] + B[105]*C[97] + B[120]*C[98] + B[135]*C[99] + B[150]*C[100] + B[165]*C[101] + B[180]*C[102] + B[195]*C[103] + B[210]*C[104];
  A[91]  -= B[1]*C[90] + B[16]*C[91] + B[31]*C[92] + B[46]*C[93] + B[61]*C[94] + B[76]*C[95] + B[91]*C[96] + B[106]*C[97] + B[121]*C[98] + B[136]*C[99] + B[151]*C[100] + B[166]*C[101] + B[181]*C[102] + B[196]*C[103] + B[211]*C[104];
  A[92]  -= B[2]*C[90] + B[17]*C[91] + B[32]*C[92] + B[47]*C[93] + B[62]*C[94] + B[77]*C[95] + B[92]*C[96] + B[107]*C[97] + B[122]*C[98] + B[137]*C[99] + B[152]*C[100] + B[167]*C[101] + B[182]*C[102] + B[197]*C[103] + B[212]*C[104];
  A[93]  -= B[3]*C[90] + B[18]*C[91] + B[33]*C[92] + B[48]*C[93] + B[63]*C[94] + B[78]*C[95] + B[93]*C[96] + B[108]*C[97] + B[123]*C[98] + B[138]*C[99] + B[153]*C[100] + B[168]*C[101] + B[183]*C[102] + B[198]*C[103] + B[213]*C[104];
  A[94]  -= B[4]*C[90] + B[19]*C[91] + B[34]*C[92] + B[49]*C[93] + B[64]*C[94] + B[79]*C[95] + B[94]*C[96] + B[109]*C[97] + B[124]*C[98] + B[139]*C[99] + B[154]*C[100] + B[169]*C[101] + B[184]*C[102] + B[199]*C[103] + B[214]*C[104];
  A[95]  -= B[5]*C[90] + B[20]*C[91] + B[35]*C[92] + B[50]*C[93] + B[65]*C[94] + B[80]*C[95] + B[95]*C[96] + B[110]*C[97] + B[125]*C[98] + B[140]*C[99] + B[155]*C[100] + B[170]*C[101] + B[185]*C[102] + B[200]*C[103] + B[215]*C[104];
  A[96]  -= B[6]*C[90] + B[21]*C[91] + B[36]*C[92] + B[51]*C[93] + B[66]*C[94] + B[81]*C[95] + B[96]*C[96] + B[111]*C[97] + B[126]*C[98] + B[141]*C[99] + B[156]*C[100] + B[171]*C[101] + B[186]*C[102] + B[201]*C[103] + B[216]*C[104];
  A[97]  -= B[7]*C[90] + B[22]*C[91] + B[37]*C[92] + B[52]*C[93] + B[67]*C[94] + B[82]*C[95] + B[97]*C[96] + B[112]*C[97] + B[127]*C[98] + B[142]*C[99] + B[157]*C[100] + B[172]*C[101] + B[187]*C[102] + B[202]*C[103] + B[217]*C[104];
  A[98]  -= B[8]*C[90] + B[23]*C[91] + B[38]*C[92] + B[53]*C[93] + B[68]*C[94] + B[83]*C[95] + B[98]*C[96] + B[113]*C[97] + B[128]*C[98] + B[143]*C[99] + B[158]*C[100] + B[173]*C[101] + B[188]*C[102] + B[203]*C[103] + B[218]*C[104];
  A[99]  -= B[9]*C[90] + B[24]*C[91] + B[39]*C[92] + B[54]*C[93] + B[69]*C[94] + B[84]*C[95] + B[99]*C[96] + B[114]*C[97] + B[129]*C[98] + B[144]*C[99] + B[159]*C[100] + B[174]*C[101] + B[189]*C[102] + B[204]*C[103] + B[219]*C[104];
  A[100] -= B[10]*C[90] + B[25]*C[91] + B[40]*C[92] + B[55]*C[93] + B[70]*C[94] + B[85]*C[95] + B[100]*C[96] + B[115]*C[97] + B[130]*C[98] + B[145]*C[99] + B[160]*C[100] + B[175]*C[101] + B[190]*C[102] + B[205]*C[103] + B[220]*C[104];
  A[101] -= B[11]*C[90] + B[26]*C[91] + B[41]*C[92] + B[56]*C[93] + B[71]*C[94] + B[86]*C[95] + B[101]*C[96] + B[116]*C[97] + B[131]*C[98] + B[146]*C[99] + B[161]*C[100] + B[176]*C[101] + B[191]*C[102] + B[206]*C[103] + B[221]*C[104];
  A[102] -= B[12]*C[90] + B[27]*C[91] + B[42]*C[92] + B[57]*C[93] + B[72]*C[94] + B[87]*C[95] + B[102]*C[96] + B[117]*C[97] + B[132]*C[98] + B[147]*C[99] + B[162]*C[100] + B[177]*C[101] + B[192]*C[102] + B[207]*C[103] + B[222]*C[104];
  A[103] -= B[13]*C[90] + B[28]*C[91] + B[43]*C[92] + B[58]*C[93] + B[73]*C[94] + B[88]*C[95] + B[103]*C[96] + B[118]*C[97] + B[133]*C[98] + B[148]*C[99] + B[163]*C[100] + B[178]*C[101] + B[193]*C[102] + B[208]*C[103] + B[223]*C[104];
  A[104] -= B[14]*C[90] + B[29]*C[91] + B[44]*C[92] + B[59]*C[93] + B[74]*C[94] + B[89]*C[95] + B[104]*C[96] + B[119]*C[97] + B[134]*C[98] + B[149]*C[99] + B[164]*C[100] + B[179]*C[101] + B[194]*C[102] + B[209]*C[103] + B[224]*C[104];
  A[105] -= B[0]*C[105] + B[15]*C[106] + B[30]*C[107] + B[45]*C[108] + B[60]*C[109] + B[75]*C[110] + B[90]*C[111] + B[105]*C[112] + B[120]*C[113] + B[135]*C[114] + B[150]*C[115] + B[165]*C[116] + B[180]*C[117] + B[195]*C[118] + B[210]*C[119];
  A[106] -= B[1]*C[105] + B[16]*C[106] + B[31]*C[107] + B[46]*C[108] + B[61]*C[109] + B[76]*C[110] + B[91]*C[111] + B[106]*C[112] + B[121]*C[113] + B[136]*C[114] + B[151]*C[115] + B[166]*C[116] + B[181]*C[117] + B[196]*C[118] + B[211]*C[119];
  A[107] -= B[2]*C[105] + B[17]*C[106] + B[32]*C[107] + B[47]*C[108] + B[62]*C[109] + B[77]*C[110] + B[92]*C[111] + B[107]*C[112] + B[122]*C[113] + B[137]*C[114] + B[152]*C[115] + B[167]*C[116] + B[182]*C[117] + B[197]*C[118] + B[212]*C[119];
  A[108] -= B[3]*C[105] + B[18]*C[106] + B[33]*C[107] + B[48]*C[108] + B[63]*C[109] + B[78]*C[110] + B[93]*C[111] + B[108]*C[112] + B[123]*C[113] + B[138]*C[114] + B[153]*C[115] + B[168]*C[116] + B[183]*C[117] + B[198]*C[118] + B[213]*C[119];
  A[109] -= B[4]*C[105] + B[19]*C[106] + B[34]*C[107] + B[49]*C[108] + B[64]*C[109] + B[79]*C[110] + B[94]*C[111] + B[109]*C[112] + B[124]*C[113] + B[139]*C[114] + B[154]*C[115] + B[169]*C[116] + B[184]*C[117] + B[199]*C[118] + B[214]*C[119];
  A[110] -= B[5]*C[105] + B[20]*C[106] + B[35]*C[107] + B[50]*C[108] + B[65]*C[109] + B[80]*C[110] + B[95]*C[111] + B[110]*C[112] + B[125]*C[113] + B[140]*C[114] + B[155]*C[115] + B[170]*C[116] + B[185]*C[117] + B[200]*C[118] + B[215]*C[119];
  A[111] -= B[6]*C[105] + B[21]*C[106] + B[36]*C[107] + B[51]*C[108] + B[66]*C[109] + B[81]*C[110] + B[96]*C[111] + B[111]*C[112] + B[126]*C[113] + B[141]*C[114] + B[156]*C[115] + B[171]*C[116] + B[186]*C[117] + B[201]*C[118] + B[216]*C[119];
  A[112] -= B[7]*C[105] + B[22]*C[106] + B[37]*C[107] + B[52]*C[108] + B[67]*C[109] + B[82]*C[110] + B[97]*C[111] + B[112]*C[112] + B[127]*C[113] + B[142]*C[114] + B[157]*C[115] + B[172]*C[116] + B[187]*C[117] + B[202]*C[118] + B[217]*C[119];
  A[113] -= B[8]*C[105] + B[23]*C[106] + B[38]*C[107] + B[53]*C[108] + B[68]*C[109] + B[83]*C[110] + B[98]*C[111] + B[113]*C[112] + B[128]*C[113] + B[143]*C[114] + B[158]*C[115] + B[173]*C[116] + B[188]*C[117] + B[203]*C[118] + B[218]*C[119];
  A[114] -= B[9]*C[105] + B[24]*C[106] + B[39]*C[107] + B[54]*C[108] + B[69]*C[109] + B[84]*C[110] + B[99]*C[111] + B[114]*C[112] + B[129]*C[113] + B[144]*C[114] + B[159]*C[115] + B[174]*C[116] + B[189]*C[117] + B[204]*C[118] + B[219]*C[119];
  A[115] -= B[10]*C[105] + B[25]*C[106] + B[40]*C[107] + B[55]*C[108] + B[70]*C[109] + B[85]*C[110] + B[100]*C[111] + B[115]*C[112] + B[130]*C[113] + B[145]*C[114] + B[160]*C[115] + B[175]*C[116] + B[190]*C[117] + B[205]*C[118] + B[220]*C[119];
  A[116] -= B[11]*C[105] + B[26]*C[106] + B[41]*C[107] + B[56]*C[108] + B[71]*C[109] + B[86]*C[110] + B[101]*C[111] + B[116]*C[112] + B[131]*C[113] + B[146]*C[114] + B[161]*C[115] + B[176]*C[116] + B[191]*C[117] + B[206]*C[118] + B[221]*C[119];
  A[117] -= B[12]*C[105] + B[27]*C[106] + B[42]*C[107] + B[57]*C[108] + B[72]*C[109] + B[87]*C[110] + B[102]*C[111] + B[117]*C[112] + B[132]*C[113] + B[147]*C[114] + B[162]*C[115] + B[177]*C[116] + B[192]*C[117] + B[207]*C[118] + B[222]*C[119];
  A[118] -= B[13]*C[105] + B[28]*C[106] + B[43]*C[107] + B[58]*C[108] + B[73]*C[109] + B[88]*C[110] + B[103]*C[111] + B[118]*C[112] + B[133]*C[113] + B[148]*C[114] + B[163]*C[115] + B[178]*C[116] + B[193]*C[117] + B[208]*C[118] + B[223]*C[119];
  A[119] -= B[14]*C[105] + B[29]*C[106] + B[44]*C[107] + B[59]*C[108] + B[74]*C[109] + B[89]*C[110] + B[104]*C[111] + B[119]*C[112] + B[134]*C[113] + B[149]*C[114] + B[164]*C[115] + B[179]*C[116] + B[194]*C[117] + B[209]*C[118] + B[224]*C[119];
  A[120] -= B[0]*C[120] + B[15]*C[121] + B[30]*C[122] + B[45]*C[123] + B[60]*C[124] + B[75]*C[125] + B[90]*C[126] + B[105]*C[127] + B[120]*C[128] + B[135]*C[129] + B[150]*C[130] + B[165]*C[131] + B[180]*C[132] + B[195]*C[133] + B[210]*C[134];
  A[121] -= B[1]*C[120] + B[16]*C[121] + B[31]*C[122] + B[46]*C[123] + B[61]*C[124] + B[76]*C[125] + B[91]*C[126] + B[106]*C[127] + B[121]*C[128] + B[136]*C[129] + B[151]*C[130] + B[166]*C[131] + B[181]*C[132] + B[196]*C[133] + B[211]*C[134];
  A[122] -= B[2]*C[120] + B[17]*C[121] + B[32]*C[122] + B[47]*C[123] + B[62]*C[124] + B[77]*C[125] + B[92]*C[126] + B[107]*C[127] + B[122]*C[128] + B[137]*C[129] + B[152]*C[130] + B[167]*C[131] + B[182]*C[132] + B[197]*C[133] + B[212]*C[134];
  A[123] -= B[3]*C[120] + B[18]*C[121] + B[33]*C[122] + B[48]*C[123] + B[63]*C[124] + B[78]*C[125] + B[93]*C[126] + B[108]*C[127] + B[123]*C[128] + B[138]*C[129] + B[153]*C[130] + B[168]*C[131] + B[183]*C[132] + B[198]*C[133] + B[213]*C[134];
  A[124] -= B[4]*C[120] + B[19]*C[121] + B[34]*C[122] + B[49]*C[123] + B[64]*C[124] + B[79]*C[125] + B[94]*C[126] + B[109]*C[127] + B[124]*C[128] + B[139]*C[129] + B[154]*C[130] + B[169]*C[131] + B[184]*C[132] + B[199]*C[133] + B[214]*C[134];
  A[125] -= B[5]*C[120] + B[20]*C[121] + B[35]*C[122] + B[50]*C[123] + B[65]*C[124] + B[80]*C[125] + B[95]*C[126] + B[110]*C[127] + B[125]*C[128] + B[140]*C[129] + B[155]*C[130] + B[170]*C[131] + B[185]*C[132] + B[200]*C[133] + B[215]*C[134];
  A[126] -= B[6]*C[120] + B[21]*C[121] + B[36]*C[122] + B[51]*C[123] + B[66]*C[124] + B[81]*C[125] + B[96]*C[126] + B[111]*C[127] + B[126]*C[128] + B[141]*C[129] + B[156]*C[130] + B[171]*C[131] + B[186]*C[132] + B[201]*C[133] + B[216]*C[134];
  A[127] -= B[7]*C[120] + B[22]*C[121] + B[37]*C[122] + B[52]*C[123] + B[67]*C[124] + B[82]*C[125] + B[97]*C[126] + B[112]*C[127] + B[127]*C[128] + B[142]*C[129] + B[157]*C[130] + B[172]*C[131] + B[187]*C[132] + B[202]*C[133] + B[217]*C[134];
  A[128] -= B[8]*C[120] + B[23]*C[121] + B[38]*C[122] + B[53]*C[123] + B[68]*C[124] + B[83]*C[125] + B[98]*C[126] + B[113]*C[127] + B[128]*C[128] + B[143]*C[129] + B[158]*C[130] + B[173]*C[131] + B[188]*C[132] + B[203]*C[133] + B[218]*C[134];
  A[129] -= B[9]*C[120] + B[24]*C[121] + B[39]*C[122] + B[54]*C[123] + B[69]*C[124] + B[84]*C[125] + B[99]*C[126] + B[114]*C[127] + B[129]*C[128] + B[144]*C[129] + B[159]*C[130] + B[174]*C[131] + B[189]*C[132] + B[204]*C[133] + B[219]*C[134];
  A[130] -= B[10]*C[120] + B[25]*C[121] + B[40]*C[122] + B[55]*C[123] + B[70]*C[124] + B[85]*C[125] + B[100]*C[126] + B[115]*C[127] + B[130]*C[128] + B[145]*C[129] + B[160]*C[130] + B[175]*C[131] + B[190]*C[132] + B[205]*C[133] + B[220]*C[134];
  A[131] -= B[11]*C[120] + B[26]*C[121] + B[41]*C[122] + B[56]*C[123] + B[71]*C[124] + B[86]*C[125] + B[101]*C[126] + B[116]*C[127] + B[131]*C[128] + B[146]*C[129] + B[161]*C[130] + B[176]*C[131] + B[191]*C[132] + B[206]*C[133] + B[221]*C[134];
  A[132] -= B[12]*C[120] + B[27]*C[121] + B[42]*C[122] + B[57]*C[123] + B[72]*C[124] + B[87]*C[125] + B[102]*C[126] + B[117]*C[127] + B[132]*C[128] + B[147]*C[129] + B[162]*C[130] + B[177]*C[131] + B[192]*C[132] + B[207]*C[133] + B[222]*C[134];
  A[133] -= B[13]*C[120] + B[28]*C[121] + B[43]*C[122] + B[58]*C[123] + B[73]*C[124] + B[88]*C[125] + B[103]*C[126] + B[118]*C[127] + B[133]*C[128] + B[148]*C[129] + B[163]*C[130] + B[178]*C[131] + B[193]*C[132] + B[208]*C[133] + B[223]*C[134];
  A[134] -= B[14]*C[120] + B[29]*C[121] + B[44]*C[122] + B[59]*C[123] + B[74]*C[124] + B[89]*C[125] + B[104]*C[126] + B[119]*C[127] + B[134]*C[128] + B[149]*C[129] + B[164]*C[130] + B[179]*C[131] + B[194]*C[132] + B[209]*C[133] + B[224]*C[134];
  A[135] -= B[0]*C[135] + B[15]*C[136] + B[30]*C[137] + B[45]*C[138] + B[60]*C[139] + B[75]*C[140] + B[90]*C[141] + B[105]*C[142] + B[120]*C[143] + B[135]*C[144] + B[150]*C[145] + B[165]*C[146] + B[180]*C[147] + B[195]*C[148] + B[210]*C[149];
  A[136] -= B[1]*C[135] + B[16]*C[136] + B[31]*C[137] + B[46]*C[138] + B[61]*C[139] + B[76]*C[140] + B[91]*C[141] + B[106]*C[142] + B[121]*C[143] + B[136]*C[144] + B[151]*C[145] + B[166]*C[146] + B[181]*C[147] + B[196]*C[148] + B[211]*C[149];
  A[137] -= B[2]*C[135] + B[17]*C[136] + B[32]*C[137] + B[47]*C[138] + B[62]*C[139] + B[77]*C[140] + B[92]*C[141] + B[107]*C[142] + B[122]*C[143] + B[137]*C[144] + B[152]*C[145] + B[167]*C[146] + B[182]*C[147] + B[197]*C[148] + B[212]*C[149];
  A[138] -= B[3]*C[135] + B[18]*C[136] + B[33]*C[137] + B[48]*C[138] + B[63]*C[139] + B[78]*C[140] + B[93]*C[141] + B[108]*C[142] + B[123]*C[143] + B[138]*C[144] + B[153]*C[145] + B[168]*C[146] + B[183]*C[147] + B[198]*C[148] + B[213]*C[149];
  A[139] -= B[4]*C[135] + B[19]*C[136] + B[34]*C[137] + B[49]*C[138] + B[64]*C[139] + B[79]*C[140] + B[94]*C[141] + B[109]*C[142] + B[124]*C[143] + B[139]*C[144] + B[154]*C[145] + B[169]*C[146] + B[184]*C[147] + B[199]*C[148] + B[214]*C[149];
  A[140] -= B[5]*C[135] + B[20]*C[136] + B[35]*C[137] + B[50]*C[138] + B[65]*C[139] + B[80]*C[140] + B[95]*C[141] + B[110]*C[142] + B[125]*C[143] + B[140]*C[144] + B[155]*C[145] + B[170]*C[146] + B[185]*C[147] + B[200]*C[148] + B[215]*C[149];
  A[141] -= B[6]*C[135] + B[21]*C[136] + B[36]*C[137] + B[51]*C[138] + B[66]*C[139] + B[81]*C[140] + B[96]*C[141] + B[111]*C[142] + B[126]*C[143] + B[141]*C[144] + B[156]*C[145] + B[171]*C[146] + B[186]*C[147] + B[201]*C[148] + B[216]*C[149];
  A[142] -= B[7]*C[135] + B[22]*C[136] + B[37]*C[137] + B[52]*C[138] + B[67]*C[139] + B[82]*C[140] + B[97]*C[141] + B[112]*C[142] + B[127]*C[143] + B[142]*C[144] + B[157]*C[145] + B[172]*C[146] + B[187]*C[147] + B[202]*C[148] + B[217]*C[149];
  A[143] -= B[8]*C[135] + B[23]*C[136] + B[38]*C[137] + B[53]*C[138] + B[68]*C[139] + B[83]*C[140] + B[98]*C[141] + B[113]*C[142] + B[128]*C[143] + B[143]*C[144] + B[158]*C[145] + B[173]*C[146] + B[188]*C[147] + B[203]*C[148] + B[218]*C[149];
  A[144] -= B[9]*C[135] + B[24]*C[136] + B[39]*C[137] + B[54]*C[138] + B[69]*C[139] + B[84]*C[140] + B[99]*C[141] + B[114]*C[142] + B[129]*C[143] + B[144]*C[144] + B[159]*C[145] + B[174]*C[146] + B[189]*C[147] + B[204]*C[148] + B[219]*C[149];
  A[145] -= B[10]*C[135] + B[25]*C[136] + B[40]*C[137] + B[55]*C[138] + B[70]*C[139] + B[85]*C[140] + B[100]*C[141] + B[115]*C[142] + B[130]*C[143] + B[145]*C[144] + B[160]*C[145] + B[175]*C[146] + B[190]*C[147] + B[205]*C[148] + B[220]*C[149];
  A[146] -= B[11]*C[135] + B[26]*C[136] + B[41]*C[137] + B[56]*C[138] + B[71]*C[139] + B[86]*C[140] + B[101]*C[141] + B[116]*C[142] + B[131]*C[143] + B[146]*C[144] + B[161]*C[145] + B[176]*C[146] + B[191]*C[147] + B[206]*C[148] + B[221]*C[149];
  A[147] -= B[12]*C[135] + B[27]*C[136] + B[42]*C[137] + B[57]*C[138] + B[72]*C[139] + B[87]*C[140] + B[102]*C[141] + B[117]*C[142] + B[132]*C[143] + B[147]*C[144] + B[162]*C[145] + B[177]*C[146] + B[192]*C[147] + B[207]*C[148] + B[222]*C[149];
  A[148] -= B[13]*C[135] + B[28]*C[136] + B[43]*C[137] + B[58]*C[138] + B[73]*C[139] + B[88]*C[140] + B[103]*C[141] + B[118]*C[142] + B[133]*C[143] + B[148]*C[144] + B[163]*C[145] + B[178]*C[146] + B[193]*C[147] + B[208]*C[148] + B[223]*C[149];
  A[149] -= B[14]*C[135] + B[29]*C[136] + B[44]*C[137] + B[59]*C[138] + B[74]*C[139] + B[89]*C[140] + B[104]*C[141] + B[119]*C[142] + B[134]*C[143] + B[149]*C[144] + B[164]*C[145] + B[179]*C[146] + B[194]*C[147] + B[209]*C[148] + B[224]*C[149];
  A[150] -= B[0]*C[150] + B[15]*C[151] + B[30]*C[152] + B[45]*C[153] + B[60]*C[154] + B[75]*C[155] + B[90]*C[156] + B[105]*C[157] + B[120]*C[158] + B[135]*C[159] + B[150]*C[160] + B[165]*C[161] + B[180]*C[162] + B[195]*C[163] + B[210]*C[164];
  A[151] -= B[1]*C[150] + B[16]*C[151] + B[31]*C[152] + B[46]*C[153] + B[61]*C[154] + B[76]*C[155] + B[91]*C[156] + B[106]*C[157] + B[121]*C[158] + B[136]*C[159] + B[151]*C[160] + B[166]*C[161] + B[181]*C[162] + B[196]*C[163] + B[211]*C[164];
  A[152] -= B[2]*C[150] + B[17]*C[151] + B[32]*C[152] + B[47]*C[153] + B[62]*C[154] + B[77]*C[155] + B[92]*C[156] + B[107]*C[157] + B[122]*C[158] + B[137]*C[159] + B[152]*C[160] + B[167]*C[161] + B[182]*C[162] + B[197]*C[163] + B[212]*C[164];
  A[153] -= B[3]*C[150] + B[18]*C[151] + B[33]*C[152] + B[48]*C[153] + B[63]*C[154] + B[78]*C[155] + B[93]*C[156] + B[108]*C[157] + B[123]*C[158] + B[138]*C[159] + B[153]*C[160] + B[168]*C[161] + B[183]*C[162] + B[198]*C[163] + B[213]*C[164];
  A[154] -= B[4]*C[150] + B[19]*C[151] + B[34]*C[152] + B[49]*C[153] + B[64]*C[154] + B[79]*C[155] + B[94]*C[156] + B[109]*C[157] + B[124]*C[158] + B[139]*C[159] + B[154]*C[160] + B[169]*C[161] + B[184]*C[162] + B[199]*C[163] + B[214]*C[164];
  A[155] -= B[5]*C[150] + B[20]*C[151] + B[35]*C[152] + B[50]*C[153] + B[65]*C[154] + B[80]*C[155] + B[95]*C[156] + B[110]*C[157] + B[125]*C[158] + B[140]*C[159] + B[155]*C[160] + B[170]*C[161] + B[185]*C[162] + B[200]*C[163] + B[215]*C[164];
  A[156] -= B[6]*C[150] + B[21]*C[151] + B[36]*C[152] + B[51]*C[153] + B[66]*C[154] + B[81]*C[155] + B[96]*C[156] + B[111]*C[157] + B[126]*C[158] + B[141]*C[159] + B[156]*C[160] + B[171]*C[161] + B[186]*C[162] + B[201]*C[163] + B[216]*C[164];
  A[157] -= B[7]*C[150] + B[22]*C[151] + B[37]*C[152] + B[52]*C[153] + B[67]*C[154] + B[82]*C[155] + B[97]*C[156] + B[112]*C[157] + B[127]*C[158] + B[142]*C[159] + B[157]*C[160] + B[172]*C[161] + B[187]*C[162] + B[202]*C[163] + B[217]*C[164];
  A[158] -= B[8]*C[150] + B[23]*C[151] + B[38]*C[152] + B[53]*C[153] + B[68]*C[154] + B[83]*C[155] + B[98]*C[156] + B[113]*C[157] + B[128]*C[158] + B[143]*C[159] + B[158]*C[160] + B[173]*C[161] + B[188]*C[162] + B[203]*C[163] + B[218]*C[164];
  A[159] -= B[9]*C[150] + B[24]*C[151] + B[39]*C[152] + B[54]*C[153] + B[69]*C[154] + B[84]*C[155] + B[99]*C[156] + B[114]*C[157] + B[129]*C[158] + B[144]*C[159] + B[159]*C[160] + B[174]*C[161] + B[189]*C[162] + B[204]*C[163] + B[219]*C[164];
  A[160] -= B[10]*C[150] + B[25]*C[151] + B[40]*C[152] + B[55]*C[153] + B[70]*C[154] + B[85]*C[155] + B[100]*C[156] + B[115]*C[157] + B[130]*C[158] + B[145]*C[159] + B[160]*C[160] + B[175]*C[161] + B[190]*C[162] + B[205]*C[163] + B[220]*C[164];
  A[161] -= B[11]*C[150] + B[26]*C[151] + B[41]*C[152] + B[56]*C[153] + B[71]*C[154] + B[86]*C[155] + B[101]*C[156] + B[116]*C[157] + B[131]*C[158] + B[146]*C[159] + B[161]*C[160] + B[176]*C[161] + B[191]*C[162] + B[206]*C[163] + B[221]*C[164];
  A[162] -= B[12]*C[150] + B[27]*C[151] + B[42]*C[152] + B[57]*C[153] + B[72]*C[154] + B[87]*C[155] + B[102]*C[156] + B[117]*C[157] + B[132]*C[158] + B[147]*C[159] + B[162]*C[160] + B[177]*C[161] + B[192]*C[162] + B[207]*C[163] + B[222]*C[164];
  A[163] -= B[13]*C[150] + B[28]*C[151] + B[43]*C[152] + B[58]*C[153] + B[73]*C[154] + B[88]*C[155] + B[103]*C[156] + B[118]*C[157] + B[133]*C[158] + B[148]*C[159] + B[163]*C[160] + B[178]*C[161] + B[193]*C[162] + B[208]*C[163] + B[223]*C[164];
  A[164] -= B[14]*C[150] + B[29]*C[151] + B[44]*C[152] + B[59]*C[153] + B[74]*C[154] + B[89]*C[155] + B[104]*C[156] + B[119]*C[157] + B[134]*C[158] + B[149]*C[159] + B[164]*C[160] + B[179]*C[161] + B[194]*C[162] + B[209]*C[163] + B[224]*C[164];
  A[165] -= B[0]*C[165] + B[15]*C[166] + B[30]*C[167] + B[45]*C[168] + B[60]*C[169] + B[75]*C[170] + B[90]*C[171] + B[105]*C[172] + B[120]*C[173] + B[135]*C[174] + B[150]*C[175] + B[165]*C[176] + B[180]*C[177] + B[195]*C[178] + B[210]*C[179];
  A[166] -= B[1]*C[165] + B[16]*C[166] + B[31]*C[167] + B[46]*C[168] + B[61]*C[169] + B[76]*C[170] + B[91]*C[171] + B[106]*C[172] + B[121]*C[173] + B[136]*C[174] + B[151]*C[175] + B[166]*C[176] + B[181]*C[177] + B[196]*C[178] + B[211]*C[179];
  A[167] -= B[2]*C[165] + B[17]*C[166] + B[32]*C[167] + B[47]*C[168] + B[62]*C[169] + B[77]*C[170] + B[92]*C[171] + B[107]*C[172] + B[122]*C[173] + B[137]*C[174] + B[152]*C[175] + B[167]*C[176] + B[182]*C[177] + B[197]*C[178] + B[212]*C[179];
  A[168] -= B[3]*C[165] + B[18]*C[166] + B[33]*C[167] + B[48]*C[168] + B[63]*C[169] + B[78]*C[170] + B[93]*C[171] + B[108]*C[172] + B[123]*C[173] + B[138]*C[174] + B[153]*C[175] + B[168]*C[176] + B[183]*C[177] + B[198]*C[178] + B[213]*C[179];
  A[169] -= B[4]*C[165] + B[19]*C[166] + B[34]*C[167] + B[49]*C[168] + B[64]*C[169] + B[79]*C[170] + B[94]*C[171] + B[109]*C[172] + B[124]*C[173] + B[139]*C[174] + B[154]*C[175] + B[169]*C[176] + B[184]*C[177] + B[199]*C[178] + B[214]*C[179];
  A[170] -= B[5]*C[165] + B[20]*C[166] + B[35]*C[167] + B[50]*C[168] + B[65]*C[169] + B[80]*C[170] + B[95]*C[171] + B[110]*C[172] + B[125]*C[173] + B[140]*C[174] + B[155]*C[175] + B[170]*C[176] + B[185]*C[177] + B[200]*C[178] + B[215]*C[179];
  A[171] -= B[6]*C[165] + B[21]*C[166] + B[36]*C[167] + B[51]*C[168] + B[66]*C[169] + B[81]*C[170] + B[96]*C[171] + B[111]*C[172] + B[126]*C[173] + B[141]*C[174] + B[156]*C[175] + B[171]*C[176] + B[186]*C[177] + B[201]*C[178] + B[216]*C[179];
  A[172] -= B[7]*C[165] + B[22]*C[166] + B[37]*C[167] + B[52]*C[168] + B[67]*C[169] + B[82]*C[170] + B[97]*C[171] + B[112]*C[172] + B[127]*C[173] + B[142]*C[174] + B[157]*C[175] + B[172]*C[176] + B[187]*C[177] + B[202]*C[178] + B[217]*C[179];
  A[173] -= B[8]*C[165] + B[23]*C[166] + B[38]*C[167] + B[53]*C[168] + B[68]*C[169] + B[83]*C[170] + B[98]*C[171] + B[113]*C[172] + B[128]*C[173] + B[143]*C[174] + B[158]*C[175] + B[173]*C[176] + B[188]*C[177] + B[203]*C[178] + B[218]*C[179];
  A[174] -= B[9]*C[165] + B[24]*C[166] + B[39]*C[167] + B[54]*C[168] + B[69]*C[169] + B[84]*C[170] + B[99]*C[171] + B[114]*C[172] + B[129]*C[173] + B[144]*C[174] + B[159]*C[175] + B[174]*C[176] + B[189]*C[177] + B[204]*C[178] + B[219]*C[179];
  A[175] -= B[10]*C[165] + B[25]*C[166] + B[40]*C[167] + B[55]*C[168] + B[70]*C[169] + B[85]*C[170] + B[100]*C[171] + B[115]*C[172] + B[130]*C[173] + B[145]*C[174] + B[160]*C[175] + B[175]*C[176] + B[190]*C[177] + B[205]*C[178] + B[220]*C[179];
  A[176] -= B[11]*C[165] + B[26]*C[166] + B[41]*C[167] + B[56]*C[168] + B[71]*C[169] + B[86]*C[170] + B[101]*C[171] + B[116]*C[172] + B[131]*C[173] + B[146]*C[174] + B[161]*C[175] + B[176]*C[176] + B[191]*C[177] + B[206]*C[178] + B[221]*C[179];
  A[177] -= B[12]*C[165] + B[27]*C[166] + B[42]*C[167] + B[57]*C[168] + B[72]*C[169] + B[87]*C[170] + B[102]*C[171] + B[117]*C[172] + B[132]*C[173] + B[147]*C[174] + B[162]*C[175] + B[177]*C[176] + B[192]*C[177] + B[207]*C[178] + B[222]*C[179];
  A[178] -= B[13]*C[165] + B[28]*C[166] + B[43]*C[167] + B[58]*C[168] + B[73]*C[169] + B[88]*C[170] + B[103]*C[171] + B[118]*C[172] + B[133]*C[173] + B[148]*C[174] + B[163]*C[175] + B[178]*C[176] + B[193]*C[177] + B[208]*C[178] + B[223]*C[179];
  A[179] -= B[14]*C[165] + B[29]*C[166] + B[44]*C[167] + B[59]*C[168] + B[74]*C[169] + B[89]*C[170] + B[104]*C[171] + B[119]*C[172] + B[134]*C[173] + B[149]*C[174] + B[164]*C[175] + B[179]*C[176] + B[194]*C[177] + B[209]*C[178] + B[224]*C[179];
  A[180] -= B[0]*C[180] + B[15]*C[181] + B[30]*C[182] + B[45]*C[183] + B[60]*C[184] + B[75]*C[185] + B[90]*C[186] + B[105]*C[187] + B[120]*C[188] + B[135]*C[189] + B[150]*C[190] + B[165]*C[191] + B[180]*C[192] + B[195]*C[193] + B[210]*C[194];
  A[181] -= B[1]*C[180] + B[16]*C[181] + B[31]*C[182] + B[46]*C[183] + B[61]*C[184] + B[76]*C[185] + B[91]*C[186] + B[106]*C[187] + B[121]*C[188] + B[136]*C[189] + B[151]*C[190] + B[166]*C[191] + B[181]*C[192] + B[196]*C[193] + B[211]*C[194];
  A[182] -= B[2]*C[180] + B[17]*C[181] + B[32]*C[182] + B[47]*C[183] + B[62]*C[184] + B[77]*C[185] + B[92]*C[186] + B[107]*C[187] + B[122]*C[188] + B[137]*C[189] + B[152]*C[190] + B[167]*C[191] + B[182]*C[192] + B[197]*C[193] + B[212]*C[194];
  A[183] -= B[3]*C[180] + B[18]*C[181] + B[33]*C[182] + B[48]*C[183] + B[63]*C[184] + B[78]*C[185] + B[93]*C[186] + B[108]*C[187] + B[123]*C[188] + B[138]*C[189] + B[153]*C[190] + B[168]*C[191] + B[183]*C[192] + B[198]*C[193] + B[213]*C[194];
  A[184] -= B[4]*C[180] + B[19]*C[181] + B[34]*C[182] + B[49]*C[183] + B[64]*C[184] + B[79]*C[185] + B[94]*C[186] + B[109]*C[187] + B[124]*C[188] + B[139]*C[189] + B[154]*C[190] + B[169]*C[191] + B[184]*C[192] + B[199]*C[193] + B[214]*C[194];
  A[185] -= B[5]*C[180] + B[20]*C[181] + B[35]*C[182] + B[50]*C[183] + B[65]*C[184] + B[80]*C[185] + B[95]*C[186] + B[110]*C[187] + B[125]*C[188] + B[140]*C[189] + B[155]*C[190] + B[170]*C[191] + B[185]*C[192] + B[200]*C[193] + B[215]*C[194];
  A[186] -= B[6]*C[180] + B[21]*C[181] + B[36]*C[182] + B[51]*C[183] + B[66]*C[184] + B[81]*C[185] + B[96]*C[186] + B[111]*C[187] + B[126]*C[188] + B[141]*C[189] + B[156]*C[190] + B[171]*C[191] + B[186]*C[192] + B[201]*C[193] + B[216]*C[194];
  A[187] -= B[7]*C[180] + B[22]*C[181] + B[37]*C[182] + B[52]*C[183] + B[67]*C[184] + B[82]*C[185] + B[97]*C[186] + B[112]*C[187] + B[127]*C[188] + B[142]*C[189] + B[157]*C[190] + B[172]*C[191] + B[187]*C[192] + B[202]*C[193] + B[217]*C[194];
  A[188] -= B[8]*C[180] + B[23]*C[181] + B[38]*C[182] + B[53]*C[183] + B[68]*C[184] + B[83]*C[185] + B[98]*C[186] + B[113]*C[187] + B[128]*C[188] + B[143]*C[189] + B[158]*C[190] + B[173]*C[191] + B[188]*C[192] + B[203]*C[193] + B[218]*C[194];
  A[189] -= B[9]*C[180] + B[24]*C[181] + B[39]*C[182] + B[54]*C[183] + B[69]*C[184] + B[84]*C[185] + B[99]*C[186] + B[114]*C[187] + B[129]*C[188] + B[144]*C[189] + B[159]*C[190] + B[174]*C[191] + B[189]*C[192] + B[204]*C[193] + B[219]*C[194];
  A[190] -= B[10]*C[180] + B[25]*C[181] + B[40]*C[182] + B[55]*C[183] + B[70]*C[184] + B[85]*C[185] + B[100]*C[186] + B[115]*C[187] + B[130]*C[188] + B[145]*C[189] + B[160]*C[190] + B[175]*C[191] + B[190]*C[192] + B[205]*C[193] + B[220]*C[194];
  A[191] -= B[11]*C[180] + B[26]*C[181] + B[41]*C[182] + B[56]*C[183] + B[71]*C[184] + B[86]*C[185] + B[101]*C[186] + B[116]*C[187] + B[131]*C[188] + B[146]*C[189] + B[161]*C[190] + B[176]*C[191] + B[191]*C[192] + B[206]*C[193] + B[221]*C[194];
  A[192] -= B[12]*C[180] + B[27]*C[181] + B[42]*C[182] + B[57]*C[183] + B[72]*C[184] + B[87]*C[185] + B[102]*C[186] + B[117]*C[187] + B[132]*C[188] + B[147]*C[189] + B[162]*C[190] + B[177]*C[191] + B[192]*C[192] + B[207]*C[193] + B[222]*C[194];
  A[193] -= B[13]*C[180] + B[28]*C[181] + B[43]*C[182] + B[58]*C[183] + B[73]*C[184] + B[88]*C[185] + B[103]*C[186] + B[118]*C[187] + B[133]*C[188] + B[148]*C[189] + B[163]*C[190] + B[178]*C[191] + B[193]*C[192] + B[208]*C[193] + B[223]*C[194];
  A[194] -= B[14]*C[180] + B[29]*C[181] + B[44]*C[182] + B[59]*C[183] + B[74]*C[184] + B[89]*C[185] + B[104]*C[186] + B[119]*C[187] + B[134]*C[188] + B[149]*C[189] + B[164]*C[190] + B[179]*C[191] + B[194]*C[192] + B[209]*C[193] + B[224]*C[194];
  A[195] -= B[0]*C[195] + B[15]*C[196] + B[30]*C[197] + B[45]*C[198] + B[60]*C[199] + B[75]*C[200] + B[90]*C[201] + B[105]*C[202] + B[120]*C[203] + B[135]*C[204] + B[150]*C[205] + B[165]*C[206] + B[180]*C[207] + B[195]*C[208] + B[210]*C[209];
  A[196] -= B[1]*C[195] + B[16]*C[196] + B[31]*C[197] + B[46]*C[198] + B[61]*C[199] + B[76]*C[200] + B[91]*C[201] + B[106]*C[202] + B[121]*C[203] + B[136]*C[204] + B[151]*C[205] + B[166]*C[206] + B[181]*C[207] + B[196]*C[208] + B[211]*C[209];
  A[197] -= B[2]*C[195] + B[17]*C[196] + B[32]*C[197] + B[47]*C[198] + B[62]*C[199] + B[77]*C[200] + B[92]*C[201] + B[107]*C[202] + B[122]*C[203] + B[137]*C[204] + B[152]*C[205] + B[167]*C[206] + B[182]*C[207] + B[197]*C[208] + B[212]*C[209];
  A[198] -= B[3]*C[195] + B[18]*C[196] + B[33]*C[197] + B[48]*C[198] + B[63]*C[199] + B[78]*C[200] + B[93]*C[201] + B[108]*C[202] + B[123]*C[203] + B[138]*C[204] + B[153]*C[205] + B[168]*C[206] + B[183]*C[207] + B[198]*C[208] + B[213]*C[209];
  A[199] -= B[4]*C[195] + B[19]*C[196] + B[34]*C[197] + B[49]*C[198] + B[64]*C[199] + B[79]*C[200] + B[94]*C[201] + B[109]*C[202] + B[124]*C[203] + B[139]*C[204] + B[154]*C[205] + B[169]*C[206] + B[184]*C[207] + B[199]*C[208] + B[214]*C[209];
  A[200] -= B[5]*C[195] + B[20]*C[196] + B[35]*C[197] + B[50]*C[198] + B[65]*C[199] + B[80]*C[200] + B[95]*C[201] + B[110]*C[202] + B[125]*C[203] + B[140]*C[204] + B[155]*C[205] + B[170]*C[206] + B[185]*C[207] + B[200]*C[208] + B[215]*C[209];
  A[201] -= B[6]*C[195] + B[21]*C[196] + B[36]*C[197] + B[51]*C[198] + B[66]*C[199] + B[81]*C[200] + B[96]*C[201] + B[111]*C[202] + B[126]*C[203] + B[141]*C[204] + B[156]*C[205] + B[171]*C[206] + B[186]*C[207] + B[201]*C[208] + B[216]*C[209];
  A[202] -= B[7]*C[195] + B[22]*C[196] + B[37]*C[197] + B[52]*C[198] + B[67]*C[199] + B[82]*C[200] + B[97]*C[201] + B[112]*C[202] + B[127]*C[203] + B[142]*C[204] + B[157]*C[205] + B[172]*C[206] + B[187]*C[207] + B[202]*C[208] + B[217]*C[209];
  A[203] -= B[8]*C[195] + B[23]*C[196] + B[38]*C[197] + B[53]*C[198] + B[68]*C[199] + B[83]*C[200] + B[98]*C[201] + B[113]*C[202] + B[128]*C[203] + B[143]*C[204] + B[158]*C[205] + B[173]*C[206] + B[188]*C[207] + B[203]*C[208] + B[218]*C[209];
  A[204] -= B[9]*C[195] + B[24]*C[196] + B[39]*C[197] + B[54]*C[198] + B[69]*C[199] + B[84]*C[200] + B[99]*C[201] + B[114]*C[202] + B[129]*C[203] + B[144]*C[204] + B[159]*C[205] + B[174]*C[206] + B[189]*C[207] + B[204]*C[208] + B[219]*C[209];
  A[205] -= B[10]*C[195] + B[25]*C[196] + B[40]*C[197] + B[55]*C[198] + B[70]*C[199] + B[85]*C[200] + B[100]*C[201] + B[115]*C[202] + B[130]*C[203] + B[145]*C[204] + B[160]*C[205] + B[175]*C[206] + B[190]*C[207] + B[205]*C[208] + B[220]*C[209];
  A[206] -= B[11]*C[195] + B[26]*C[196] + B[41]*C[197] + B[56]*C[198] + B[71]*C[199] + B[86]*C[200] + B[101]*C[201] + B[116]*C[202] + B[131]*C[203] + B[146]*C[204] + B[161]*C[205] + B[176]*C[206] + B[191]*C[207] + B[206]*C[208] + B[221]*C[209];
  A[207] -= B[12]*C[195] + B[27]*C[196] + B[42]*C[197] + B[57]*C[198] + B[72]*C[199] + B[87]*C[200] + B[102]*C[201] + B[117]*C[202] + B[132]*C[203] + B[147]*C[204] + B[162]*C[205] + B[177]*C[206] + B[192]*C[207] + B[207]*C[208] + B[222]*C[209];
  A[208] -= B[13]*C[195] + B[28]*C[196] + B[43]*C[197] + B[58]*C[198] + B[73]*C[199] + B[88]*C[200] + B[103]*C[201] + B[118]*C[202] + B[133]*C[203] + B[148]*C[204] + B[163]*C[205] + B[178]*C[206] + B[193]*C[207] + B[208]*C[208] + B[223]*C[209];
  A[209] -= B[14]*C[195] + B[29]*C[196] + B[44]*C[197] + B[59]*C[198] + B[74]*C[199] + B[89]*C[200] + B[104]*C[201] + B[119]*C[202] + B[134]*C[203] + B[149]*C[204] + B[164]*C[205] + B[179]*C[206] + B[194]*C[207] + B[209]*C[208] + B[224]*C[209];
  A[210] -= B[0]*C[210] + B[15]*C[211] + B[30]*C[212] + B[45]*C[213] + B[60]*C[214] + B[75]*C[215] + B[90]*C[216] + B[105]*C[217] + B[120]*C[218] + B[135]*C[219] + B[150]*C[220] + B[165]*C[221] + B[180]*C[222] + B[195]*C[223] + B[210]*C[224];
  A[211] -= B[1]*C[210] + B[16]*C[211] + B[31]*C[212] + B[46]*C[213] + B[61]*C[214] + B[76]*C[215] + B[91]*C[216] + B[106]*C[217] + B[121]*C[218] + B[136]*C[219] + B[151]*C[220] + B[166]*C[221] + B[181]*C[222] + B[196]*C[223] + B[211]*C[224];
  A[212] -= B[2]*C[210] + B[17]*C[211] + B[32]*C[212] + B[47]*C[213] + B[62]*C[214] + B[77]*C[215] + B[92]*C[216] + B[107]*C[217] + B[122]*C[218] + B[137]*C[219] + B[152]*C[220] + B[167]*C[221] + B[182]*C[222] + B[197]*C[223] + B[212]*C[224];
  A[213] -= B[3]*C[210] + B[18]*C[211] + B[33]*C[212] + B[48]*C[213] + B[63]*C[214] + B[78]*C[215] + B[93]*C[216] + B[108]*C[217] + B[123]*C[218] + B[138]*C[219] + B[153]*C[220] + B[168]*C[221] + B[183]*C[222] + B[198]*C[223] + B[213]*C[224];
  A[214] -= B[4]*C[210] + B[19]*C[211] + B[34]*C[212] + B[49]*C[213] + B[64]*C[214] + B[79]*C[215] + B[94]*C[216] + B[109]*C[217] + B[124]*C[218] + B[139]*C[219] + B[154]*C[220] + B[169]*C[221] + B[184]*C[222] + B[199]*C[223] + B[214]*C[224];
  A[215] -= B[5]*C[210] + B[20]*C[211] + B[35]*C[212] + B[50]*C[213] + B[65]*C[214] + B[80]*C[215] + B[95]*C[216] + B[110]*C[217] + B[125]*C[218] + B[140]*C[219] + B[155]*C[220] + B[170]*C[221] + B[185]*C[222] + B[200]*C[223] + B[215]*C[224];
  A[216] -= B[6]*C[210] + B[21]*C[211] + B[36]*C[212] + B[51]*C[213] + B[66]*C[214] + B[81]*C[215] + B[96]*C[216] + B[111]*C[217] + B[126]*C[218] + B[141]*C[219] + B[156]*C[220] + B[171]*C[221] + B[186]*C[222] + B[201]*C[223] + B[216]*C[224];
  A[217] -= B[7]*C[210] + B[22]*C[211] + B[37]*C[212] + B[52]*C[213] + B[67]*C[214] + B[82]*C[215] + B[97]*C[216] + B[112]*C[217] + B[127]*C[218] + B[142]*C[219] + B[157]*C[220] + B[172]*C[221] + B[187]*C[222] + B[202]*C[223] + B[217]*C[224];
  A[218] -= B[8]*C[210] + B[23]*C[211] + B[38]*C[212] + B[53]*C[213] + B[68]*C[214] + B[83]*C[215] + B[98]*C[216] + B[113]*C[217] + B[128]*C[218] + B[143]*C[219] + B[158]*C[220] + B[173]*C[221] + B[188]*C[222] + B[203]*C[223] + B[218]*C[224];
  A[219] -= B[9]*C[210] + B[24]*C[211] + B[39]*C[212] + B[54]*C[213] + B[69]*C[214] + B[84]*C[215] + B[99]*C[216] + B[114]*C[217] + B[129]*C[218] + B[144]*C[219] + B[159]*C[220] + B[174]*C[221] + B[189]*C[222] + B[204]*C[223] + B[219]*C[224];
  A[220] -= B[10]*C[210] + B[25]*C[211] + B[40]*C[212] + B[55]*C[213] + B[70]*C[214] + B[85]*C[215] + B[100]*C[216] + B[115]*C[217] + B[130]*C[218] + B[145]*C[219] + B[160]*C[220] + B[175]*C[221] + B[190]*C[222] + B[205]*C[223] + B[220]*C[224];
  A[221] -= B[11]*C[210] + B[26]*C[211] + B[41]*C[212] + B[56]*C[213] + B[71]*C[214] + B[86]*C[215] + B[101]*C[216] + B[116]*C[217] + B[131]*C[218] + B[146]*C[219] + B[161]*C[220] + B[176]*C[221] + B[191]*C[222] + B[206]*C[223] + B[221]*C[224];
  A[222] -= B[12]*C[210] + B[27]*C[211] + B[42]*C[212] + B[57]*C[213] + B[72]*C[214] + B[87]*C[215] + B[102]*C[216] + B[117]*C[217] + B[132]*C[218] + B[147]*C[219] + B[162]*C[220] + B[177]*C[221] + B[192]*C[222] + B[207]*C[223] + B[222]*C[224];
  A[223] -= B[13]*C[210] + B[28]*C[211] + B[43]*C[212] + B[58]*C[213] + B[73]*C[214] + B[88]*C[215] + B[103]*C[216] + B[118]*C[217] + B[133]*C[218] + B[148]*C[219] + B[163]*C[220] + B[178]*C[221] + B[193]*C[222] + B[208]*C[223] + B[223]*C[224];
  A[224] -= B[14]*C[210] + B[29]*C[211] + B[44]*C[212] + B[59]*C[213] + B[74]*C[214] + B[89]*C[215] + B[104]*C[216] + B[119]*C[217] + B[134]*C[218] + B[149]*C[219] + B[164]*C[220] + B[179]*C[221] + B[194]*C[222] + B[209]*C[223] + B[224]*C[224];
  return 0;
}

#endif
