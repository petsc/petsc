
#if !defined(__BAIJ_H)
#define __BAIJ_H
#include "private/matimpl.h"
#include "../src/mat/impls/aij/seq/aij.h"
#include "../src/mat/impls/baij/seq/ftn-kernels/fsolvebaij.h"

/*  
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

/* This header is shared by the SeqSBAIJ matrix */
#define SEQBAIJHEADER \
  PetscInt         bs2;              /*  square of block size */                                     \
  PetscInt         mbs,nbs;          /* rows/bs, columns/bs */                                       \
  PetscScalar      *mult_work;       /* work array for matrix vector product*/                       \
  MatScalar        *saved_values;                                                                    \
                                                                                                     \
  Mat              sbaijMat;         /* mat in sbaij format */                                       \
                                                                                                     \
  PetscTruth       pivotinblocks;    /* pivot inside factorization of each diagonal block */         \
                                                                                                     \
  MatScalar        *idiag;           /* inverse of block diagonal  */                                \
  PetscTruth       idiagvalid       /* if above has correct/current values */


typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQBAIJHEADER;
} Mat_SeqBAIJ;

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatSeqBAIJSetPreallocation_SeqBAIJ(Mat,PetscInt,PetscInt,PetscInt*);
EXTERN_C_END
EXTERN PetscErrorCode MatILUFactorSymbolic_SeqBAIJ(Mat,Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatICCFactorSymbolic_SeqBAIJ(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqBAIJ(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatDuplicate_SeqBAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN PetscErrorCode MatMissingDiagonal_SeqBAIJ(Mat,PetscTruth*,PetscInt*);
EXTERN PetscErrorCode MatMarkDiagonal_SeqBAIJ(Mat);
EXTERN PetscErrorCode MatILUDTFactor_SeqBAIJ(Mat,IS,IS,const MatFactorInfo*,Mat*);

EXTERN PetscErrorCode MatLUFactorSymbolic_SeqBAIJ(Mat,Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactor_SeqBAIJ(Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatIncreaseOverlap_SeqBAIJ(Mat,PetscInt,IS*,PetscInt);
EXTERN PetscErrorCode MatGetSubMatrix_SeqBAIJ(Mat,IS,IS,MatReuse,Mat*);
EXTERN PetscErrorCode MatGetSubMatrices_SeqBAIJ(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
EXTERN PetscErrorCode MatMultTranspose_SeqBAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMultTransposeAdd_SeqBAIJ(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatScale_SeqBAIJ(Mat,PetscScalar);
EXTERN PetscErrorCode MatNorm_SeqBAIJ(Mat,NormType,PetscReal *);
EXTERN PetscErrorCode MatEqual_SeqBAIJ(Mat,Mat,PetscTruth*);
EXTERN PetscErrorCode MatGetDiagonal_SeqBAIJ(Mat,Vec);
EXTERN PetscErrorCode MatDiagonalScale_SeqBAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetInfo_SeqBAIJ(Mat,MatInfoType,MatInfo *);
EXTERN PetscErrorCode MatZeroEntries_SeqBAIJ(Mat);

EXTERN PetscErrorCode MatSeqBAIJ_UpdateFactorNumeric_NaturalOrdering(Mat);

EXTERN PetscErrorCode MatSolve_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_2_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
#if defined(PETSC_HAVE_SSE)
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_SSE_Demotion(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion_usj(Mat,Vec,Vec);
#endif
EXTERN PetscErrorCode MatSolve_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_5_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_6_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_7_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_N(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_N_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_N_NaturalOrdering_newdatastruct(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_Update(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N_newdatastruct(Mat,Mat,const MatFactorInfo*);

EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_newdatastruct(Mat,Mat,const MatFactorInfo*);

EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_newdatastruct(Mat,Mat,const MatFactorInfo*);
#if defined(PETSC_HAVE_SSE)
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj(Mat,Mat,const MatFactorInfo*);
#else
#endif
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat,const MatFactorInfo*);

EXTERN PetscErrorCode MatMult_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_N(Mat,Vec,Vec);

EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_1(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_2(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_3(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_4(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_5(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_6(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_7(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_N(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatLoad_SeqBAIJ(PetscViewer, const MatType,Mat*);

/*
  Kernel_A_gets_A_times_B_2: A = A * B with size bs=2

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

#define Kernel_A_gets_A_times_B_2(A,B,W) 0;\
{\
  PetscMemcpy(W,A,4*sizeof(MatScalar));\
  A[0] = W[0]*B[0] + W[2]*B[1];\
  A[1] = W[1]*B[0] + W[3]*B[1];\
  A[2] = W[0]*B[2] + W[2]*B[3];\
  A[3] = W[1]*B[2] + W[3]*B[3];\
}

/*
  Kernel_A_gets_A_minus_B_times_C_2: A = A - B * C with size bs=2

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define Kernel_A_gets_A_minus_B_times_C_2(A,B,C) 0;\
{\
  A[0] -= B[0]*C[0] + B[2]*C[1];\
  A[1] -= B[1]*C[0] + B[3]*C[1];\
  A[2] -= B[0]*C[2] + B[2]*C[3];\
  A[3] -= B[1]*C[2] + B[3]*C[3];\
}

/*
  Kernel_A_gets_A_times_B_3: A = A * B with size bs=3

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

#define Kernel_A_gets_A_times_B_3(A,B,W) 0;\
{\
  PetscMemcpy(W,A,9*sizeof(MatScalar));\
  A[0] = W[0]*B[0] + W[3]*B[1] + W[6]*B[2];\
  A[1] = W[1]*B[0] + W[4]*B[1] + W[7]*B[2];\
  A[2] = W[2]*B[0] + W[5]*B[1] + W[8]*B[2];\
  A[3] = W[0]*B[3] + W[3]*B[4] + W[6]*B[5];\
  A[4] = W[1]*B[3] + W[4]*B[4] + W[7]*B[5];\
  A[5] = W[2]*B[3] + W[5]*B[4] + W[8]*B[5];\
  A[6] = W[0]*B[6] + W[3]*B[7] + W[6]*B[8];\
  A[7] = W[1]*B[6] + W[4]*B[7] + W[7]*B[8];\
  A[8] = W[2]*B[6] + W[5]*B[7] + W[8]*B[8];\
}

/*
  Kernel_A_gets_A_minus_B_times_C_3: A = A - B * C with size bs=3

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define Kernel_A_gets_A_minus_B_times_C_3(A,B,C) 0;\
{\
  A[0] -= B[0]*C[0] + B[3]*C[1] + B[6]*C[2];\
  A[1] -= B[1]*C[0] + B[4]*C[1] + B[7]*C[2];\
  A[2] -= B[2]*C[0] + B[5]*C[1] + B[8]*C[2];\
  A[3] -= B[0]*C[3] + B[3]*C[4] + B[6]*C[5];\
  A[4] -= B[1]*C[3] + B[4]*C[4] + B[7]*C[5];\
  A[5] -= B[2]*C[3] + B[5]*C[4] + B[8]*C[5];\
  A[6] -= B[0]*C[6] + B[3]*C[7] + B[6]*C[8];\
  A[7] -= B[1]*C[6] + B[4]*C[7] + B[7]*C[8];\
  A[8] -= B[2]*C[6] + B[5]*C[7] + B[8]*C[8];\
}

/*
  Kernel_A_gets_A_times_B_4: A = A * B with size bs=4

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

#define Kernel_A_gets_A_times_B_4(A,B,W) 0;\
{\
  PetscMemcpy(W,A,16*sizeof(MatScalar));\
  A[0] =  W[0]*B[0]  + W[4]*B[1]  + W[8]*B[2]   + W[12]*B[3];\
  A[1] =  W[1]*B[0]  + W[5]*B[1]  + W[9]*B[2]   + W[13]*B[3];\
  A[2] =  W[2]*B[0]  + W[6]*B[1]  + W[10]*B[2]  + W[14]*B[3];\
  A[3] =  W[3]*B[0]  + W[7]*B[1]  + W[11]*B[2]  + W[15]*B[3];\
  A[4] =  W[0]*B[4]  + W[4]*B[5]  + W[8]*B[6]   + W[12]*B[7];\
  A[5] =  W[1]*B[4]  + W[5]*B[5]  + W[9]*B[6]   + W[13]*B[7];\
  A[6] =  W[2]*B[4]  + W[6]*B[5]  + W[10]*B[6]  + W[14]*B[7];\
  A[7] =  W[3]*B[4]  + W[7]*B[5]  + W[11]*B[6]  + W[15]*B[7];\
  A[8] =  W[0]*B[8]  + W[4]*B[9]  + W[8]*B[10]  + W[12]*B[11];\
  A[9] =  W[1]*B[8]  + W[5]*B[9]  + W[9]*B[10]  + W[13]*B[11];\
  A[10] = W[2]*B[8]  + W[6]*B[9]  + W[10]*B[10] + W[14]*B[11];\
  A[11] = W[3]*B[8]  + W[7]*B[9]  + W[11]*B[10] + W[15]*B[11];\
  A[12] = W[0]*B[12] + W[4]*B[13] + W[8]*B[14]  + W[12]*B[15];\
  A[13] = W[1]*B[12] + W[5]*B[13] + W[9]*B[14]  + W[13]*B[15];\
  A[14] = W[2]*B[12] + W[6]*B[13] + W[10]*B[14] + W[14]*B[15];\
  A[15] = W[3]*B[12] + W[7]*B[13] + W[11]*B[14] + W[15]*B[15];\
}

/*
  Kernel_A_gets_A_minus_B_times_C_4: A = A - B * C with size bs=4

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define Kernel_A_gets_A_minus_B_times_C_4(A,B,C) 0;\
{\
  A[0] -=  B[0]*C[0]  + B[4]*C[1]  + B[8]*C[2]   + B[12]*C[3];\
  A[1] -=  B[1]*C[0]  + B[5]*C[1]  + B[9]*C[2]   + B[13]*C[3];\
  A[2] -=  B[2]*C[0]  + B[6]*C[1]  + B[10]*C[2]  + B[14]*C[3];\
  A[3] -=  B[3]*C[0]  + B[7]*C[1]  + B[11]*C[2]  + B[15]*C[3];\
  A[4] -=  B[0]*C[4]  + B[4]*C[5]  + B[8]*C[6]   + B[12]*C[7];\
  A[5] -=  B[1]*C[4]  + B[5]*C[5]  + B[9]*C[6]   + B[13]*C[7];\
  A[6] -=  B[2]*C[4]  + B[6]*C[5]  + B[10]*C[6]  + B[14]*C[7];\
  A[7] -=  B[3]*C[4]  + B[7]*C[5]  + B[11]*C[6]  + B[15]*C[7];\
  A[8] -=  B[0]*C[8]  + B[4]*C[9]  + B[8]*C[10]  + B[12]*C[11];\
  A[9] -=  B[1]*C[8]  + B[5]*C[9]  + B[9]*C[10]  + B[13]*C[11];\
  A[10] -= B[2]*C[8]  + B[6]*C[9]  + B[10]*C[10] + B[14]*C[11];\
  A[11] -= B[3]*C[8]  + B[7]*C[9]  + B[11]*C[10] + B[15]*C[11];\
  A[12] -= B[0]*C[12] + B[4]*C[13] + B[8]*C[14]  + B[12]*C[15];\
  A[13] -= B[1]*C[12] + B[5]*C[13] + B[9]*C[14]  + B[13]*C[15];\
  A[14] -= B[2]*C[12] + B[6]*C[13] + B[10]*C[14] + B[14]*C[15];\
  A[15] -= B[3]*C[12] + B[7]*C[13] + B[11]*C[14] + B[15]*C[15];\
}

#endif
