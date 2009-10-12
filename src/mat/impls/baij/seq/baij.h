
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
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_newdatastruct(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering_newdatastruct(Mat,Mat,const MatFactorInfo*);
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

#define Kernel_A_gets_A_times_B_5(A,B,W) 0;\
{\
  PetscMemcpy(W,A,25*sizeof(MatScalar));\
  A[0] =  W[0]*B[0]  + W[5]*B[1]  + W[10]*B[2]   + W[15]*B[3] + W[20]*B[4];\
  A[1] =  W[1]*B[0]  + W[6]*B[1]  + W[11]*B[2]   + W[16]*B[3] + W[21]*B[4];\
  A[2] =  W[2]*B[0]  + W[7]*B[1]  + W[12]*B[2]  + W[17]*B[3]  + W[22]*B[4];\
  A[3] =  W[3]*B[0]  + W[8]*B[1]  + W[13]*B[2]  + W[18]*B[3]  + W[23]*B[4];\
  A[4] =  W[4]*B[0]  + W[9]*B[1]  + W[14]*B[2]   + W[19]*B[3] + W[24]*B[4];\
  A[5] =  W[0]*B[5]  + W[5]*B[6]  + W[10]*B[7]   + W[15]*B[8] + W[20]*B[9];\
  A[6] =  W[1]*B[5]  + W[6]*B[6]  + W[11]*B[7]   + W[16]*B[8] + W[21]*B[9];\
  A[7] =  W[2]*B[5]  + W[7]*B[6]  + W[12]*B[7]  + W[17]*B[8]  + W[22]*B[9];\
  A[8] =  W[3]*B[5]  + W[8]*B[6]  + W[13]*B[7]  + W[18]*B[8]  + W[23]*B[9];\
  A[9] =  W[4]*B[5]  + W[9]*B[6]  + W[14]*B[7]   + W[19]*B[8] + W[24]*B[9];\
  A[10] =  W[0]*B[10]  + W[5]*B[11]  + W[10]*B[12]   + W[15]*B[13] + W[20]*B[14];\
  A[11] =  W[1]*B[10]  + W[6]*B[11]  + W[11]*B[12]   + W[16]*B[13] + W[21]*B[14];\
  A[12] =  W[2]*B[10]  + W[7]*B[11]  + W[12]*B[12]  + W[17]*B[13]  + W[22]*B[14];\
  A[13] =  W[3]*B[10]  + W[8]*B[11]  + W[13]*B[12]  + W[18]*B[13]  + W[23]*B[14];\
  A[14] =  W[4]*B[10]  + W[9]*B[11]  + W[14]*B[12]   + W[19]*B[13] + W[24]*B[14];\
  A[15] =  W[0]*B[15]  + W[5]*B[16]  + W[10]*B[17]   + W[15]*B[18] + W[20]*B[19];\
  A[16] =  W[1]*B[15]  + W[6]*B[16]  + W[11]*B[17]   + W[16]*B[18] + W[21]*B[19];\
  A[17] =  W[2]*B[15]  + W[7]*B[16]  + W[12]*B[17]  + W[17]*B[18]  + W[22]*B[19];\
  A[18] =  W[3]*B[15]  + W[8]*B[16]  + W[13]*B[17]  + W[18]*B[18]  + W[23]*B[19];\
  A[19] =  W[4]*B[15]  + W[9]*B[16]  + W[14]*B[17]   + W[19]*B[18] + W[24]*B[19];\
  A[20] =  W[0]*B[20]  + W[5]*B[21]  + W[10]*B[22]   + W[15]*B[23] + W[20]*B[24];\
  A[21] =  W[1]*B[20]  + W[6]*B[21]  + W[11]*B[22]   + W[16]*B[23] + W[21]*B[24];\
  A[22] =  W[2]*B[20]  + W[7]*B[21]  + W[12]*B[22]  + W[17]*B[23]  + W[22]*B[24];\
  A[23] =  W[3]*B[20]  + W[8]*B[21]  + W[13]*B[22]  + W[18]*B[23]  + W[23]*B[24];\
  A[24] =  W[4]*B[20]  + W[9]*B[21]  + W[14]*B[22]   + W[19]*B[23] + W[24]*B[24];\
}

/*
  Kernel_A_gets_A_minus_B_times_C_5: A = A - B * C with size bs=5

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define Kernel_A_gets_A_minus_B_times_C_5(A,B,C) 0;\
{\
  A[0] -=  B[0]*C[0]  + B[5]*C[1]  + B[10]*C[2]   + B[15]*C[3] + B[20]*C[4];\
  A[1] -=  B[1]*C[0]  + B[6]*C[1]  + B[11]*C[2]   + B[16]*C[3] + B[21]*C[4];\
  A[2] -=  B[2]*C[0]  + B[7]*C[1]  + B[12]*C[2]  + B[17]*C[3]  + B[22]*C[4];\
  A[3] -=  B[3]*C[0]  + B[8]*C[1]  + B[13]*C[2]  + B[18]*C[3]  + B[23]*C[4];\
  A[4] -=  B[4]*C[0]  + B[9]*C[1]  + B[14]*C[2]   + B[19]*C[3] + B[24]*C[4];\
  A[5] -=  B[0]*C[5]  + B[5]*C[6]  + B[10]*C[7]   + B[15]*C[8] + B[20]*C[9];\
  A[6] -=  B[1]*C[5]  + B[6]*C[6]  + B[11]*C[7]   + B[16]*C[8] + B[21]*C[9];\
  A[7] -=  B[2]*C[5]  + B[7]*C[6]  + B[12]*C[7]  + B[17]*C[8]  + B[22]*C[9];\
  A[8] -=  B[3]*C[5]  + B[8]*C[6]  + B[13]*C[7]  + B[18]*C[8]  + B[23]*C[9];\
  A[9] -=  B[4]*C[5]  + B[9]*C[6]  + B[14]*C[7]   + B[19]*C[8] + B[24]*C[9];\
  A[10] -=  B[0]*C[10]  + B[5]*C[11]  + B[10]*C[12]   + B[15]*C[13] + B[20]*C[14];\
  A[11] -=  B[1]*C[10]  + B[6]*C[11]  + B[11]*C[12]   + B[16]*C[13] + B[21]*C[14];\
  A[12] -=  B[2]*C[10]  + B[7]*C[11]  + B[12]*C[12]  + B[17]*C[13]  + B[22]*C[14];\
  A[13] -=  B[3]*C[10]  + B[8]*C[11]  + B[13]*C[12]  + B[18]*C[13]  + B[23]*C[14];\
  A[14] -=  B[4]*C[10]  + B[9]*C[11]  + B[14]*C[12]   + B[19]*C[13] + B[24]*C[14];\
  A[15] -=  B[0]*C[15]  + B[5]*C[16]  + B[10]*C[17]   + B[15]*C[18] + B[20]*C[19];\
  A[16] -=  B[1]*C[15]  + B[6]*C[16]  + B[11]*C[17]   + B[16]*C[18] + B[21]*C[19];\
  A[17] -=  B[2]*C[15]  + B[7]*C[16]  + B[12]*C[17]  + B[17]*C[18]  + B[22]*C[19];\
  A[18] -=  B[3]*C[15]  + B[8]*C[16]  + B[13]*C[17]  + B[18]*C[18]  + B[23]*C[19];\
  A[19] -=  B[4]*C[15]  + B[9]*C[16]  + B[14]*C[17]   + B[19]*C[18] + B[24]*C[19];\
  A[20] -=  B[0]*C[20]  + B[5]*C[21]  + B[10]*C[22]   + B[15]*C[23] + B[20]*C[24];\
  A[21] -=  B[1]*C[20]  + B[6]*C[21]  + B[11]*C[22]   + B[16]*C[23] + B[21]*C[24];\
  A[22] -=  B[2]*C[20]  + B[7]*C[21]  + B[12]*C[22]  + B[17]*C[23]  + B[22]*C[24];\
  A[23] -=  B[3]*C[20]  + B[8]*C[21]  + B[13]*C[22]  + B[18]*C[23]  + B[23]*C[24];\
  A[24] -=  B[4]*C[20]  + B[9]*C[21]  + B[14]*C[22]   + B[19]*C[23] + B[24]*C[24];\
}

#define Kernel_A_gets_A_times_B_6(A,B,W) 0;\
{\
  PetscMemcpy(W,A,36*sizeof(MatScalar));\
  A[0]  =  W[0]*B[0]   + W[6]*B[1]   + W[12]*B[2]   + W[18]*B[3]  + W[24]*B[4]  + W[30]*B[5];\
  A[1]  =  W[1]*B[0]   + W[7]*B[1]   + W[13]*B[2]   + W[19]*B[3]  + W[25]*B[4]  + W[31]*B[5];\
  A[2]  =  W[2]*B[0]   + W[8]*B[1]   + W[14]*B[2]   + W[20]*B[3]  + W[26]*B[4]  + W[32]*B[5];\
  A[3]  =  W[3]*B[0]   + W[9]*B[1]   + W[15]*B[2]   + W[21]*B[3]  + W[27]*B[4]  + W[33]*B[5];\
  A[4]  =  W[4]*B[0]   + W[10]*B[1]  + W[16]*B[2]   + W[22]*B[3]  + W[28]*B[4]  + W[34]*B[5];\
  A[5]  =  W[5]*B[0]   + W[11]*B[1]  + W[17]*B[2]   + W[23]*B[3]  + W[29]*B[4]  + W[35]*B[5];\
  A[6]  =  W[0]*B[6]   + W[6]*B[7]   + W[12]*B[8]   + W[18]*B[9]  + W[24]*B[10] + W[30]*B[11];\
  A[7]  =  W[1]*B[6]   + W[7]*B[7]   + W[13]*B[8]   + W[19]*B[9]  + W[25]*B[10] + W[31]*B[11];\
  A[8]  =  W[2]*B[6]   + W[8]*B[7]   + W[14]*B[8]   + W[20]*B[9]  + W[26]*B[10] + W[32]*B[11];\
  A[9]  =  W[3]*B[6]   + W[9]*B[7]   + W[15]*B[8]   + W[21]*B[9]  + W[27]*B[10] + W[33]*B[11];\
  A[10] =  W[4]*B[6]   + W[10]*B[7]  + W[16]*B[8]   + W[22]*B[9]  + W[28]*B[10] + W[34]*B[11];\
  A[11] =  W[5]*B[6]   + W[11]*B[7]  + W[17]*B[8]   + W[23]*B[9]  + W[29]*B[10] + W[35]*B[11];\
  A[12] =  W[0]*B[12]  + W[6]*B[13]  + W[12]*B[14]  + W[18]*B[15] + W[24]*B[16] + W[30]*B[17];\
  A[13] =  W[1]*B[12]  + W[7]*B[13]  + W[13]*B[14]  + W[19]*B[15] + W[25]*B[16] + W[31]*B[17];\
  A[14] =  W[2]*B[12]  + W[8]*B[13]  + W[14]*B[14]  + W[20]*B[15] + W[26]*B[16] + W[32]*B[17];\
  A[15] =  W[3]*B[12]  + W[9]*B[13]  + W[15]*B[14]  + W[21]*B[15] + W[27]*B[16] + W[33]*B[17];\
  A[16] =  W[4]*B[12]  + W[10]*B[13] + W[16]*B[14]  + W[22]*B[15] + W[28]*B[16] + W[34]*B[17];\
  A[17] =  W[5]*B[12]  + W[11]*B[13] + W[17]*B[14]  + W[23]*B[15] + W[29]*B[16] + W[35]*B[17];\
  A[18] =  W[0]*B[18]  + W[6]*B[19]  + W[12]*B[20]  + W[18]*B[21] + W[24]*B[22] + W[30]*B[23];\
  A[19] =  W[1]*B[18]  + W[7]*B[19]  + W[13]*B[20]  + W[19]*B[21] + W[25]*B[22] + W[31]*B[23];\
  A[20] =  W[2]*B[18]  + W[8]*B[19]  + W[14]*B[20]  + W[20]*B[21] + W[26]*B[22] + W[32]*B[23];\
  A[21] =  W[3]*B[18]  + W[9]*B[19]  + W[15]*B[20]  + W[21]*B[21] + W[27]*B[22] + W[33]*B[23];\
  A[22] =  W[4]*B[18]  + W[10]*B[19] + W[16]*B[20]  + W[22]*B[21] + W[28]*B[22] + W[34]*B[23];\
  A[23] =  W[5]*B[18]  + W[11]*B[19] + W[17]*B[20]  + W[23]*B[21] + W[29]*B[22] + W[35]*B[23];\
  A[24] =  W[0]*B[24]  + W[6]*B[25]  + W[12]*B[26]  + W[18]*B[27] + W[24]*B[28] + W[30]*B[29];\
  A[25] =  W[1]*B[24]  + W[7]*B[25]  + W[13]*B[26]  + W[19]*B[27] + W[25]*B[28] + W[31]*B[29];\
  A[26] =  W[2]*B[24]  + W[8]*B[25]  + W[14]*B[26]  + W[20]*B[27] + W[26]*B[28] + W[32]*B[29];\
  A[27] =  W[3]*B[24]  + W[9]*B[25]  + W[15]*B[26]  + W[21]*B[27] + W[27]*B[28] + W[33]*B[29];\
  A[28] =  W[4]*B[24]  + W[10]*B[25] + W[16]*B[26]  + W[22]*B[27] + W[28]*B[28] + W[34]*B[29];\
  A[29] =  W[5]*B[24]  + W[11]*B[25] + W[17]*B[26]  + W[23]*B[27] + W[29]*B[28] + W[35]*B[29];\
  A[30] =  W[0]*B[30]  + W[6]*B[31]  + W[12]*B[32]  + W[18]*B[33] + W[24]*B[34] + W[30]*B[35];\
  A[31] =  W[1]*B[30]  + W[7]*B[31]  + W[13]*B[32]  + W[19]*B[33] + W[25]*B[34] + W[31]*B[35];\
  A[32] =  W[2]*B[30]  + W[8]*B[31]  + W[14]*B[32]  + W[20]*B[33] + W[26]*B[34] + W[32]*B[35];\
  A[33] =  W[3]*B[30]  + W[9]*B[31]  + W[15]*B[32]  + W[21]*B[33] + W[27]*B[34] + W[33]*B[35];\
  A[34] =  W[4]*B[30]  + W[10]*B[31] + W[16]*B[32]  + W[22]*B[33] + W[28]*B[34] + W[34]*B[35];\
  A[35] =  W[5]*B[30]  + W[11]*B[31] + W[17]*B[32]  + W[23]*B[33] + W[29]*B[34] + W[35]*B[35];\
}

/*
  Kernel_A_gets_A_minus_B_times_C_6: A = A - B * C with size bs=6

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define Kernel_A_gets_A_minus_B_times_C_6(A,B,C) 0;\
{\
  A[0]  -=  B[0]*C[0]   + B[6]*C[1]   + B[12]*C[2]   + B[18]*C[3]  + B[24]*C[4]  + B[30]*C[5];\
  A[1]  -=  B[1]*C[0]   + B[7]*C[1]   + B[13]*C[2]   + B[19]*C[3]  + B[25]*C[4]  + B[31]*C[5];\
  A[2]  -=  B[2]*C[0]   + B[8]*C[1]   + B[14]*C[2]   + B[20]*C[3]  + B[26]*C[4]  + B[32]*C[5];\
  A[3]  -=  B[3]*C[0]   + B[9]*C[1]   + B[15]*C[2]   + B[21]*C[3]  + B[27]*C[4]  + B[33]*C[5];\
  A[4]  -=  B[4]*C[0]   + B[10]*C[1]  + B[16]*C[2]   + B[22]*C[3]  + B[28]*C[4]  + B[34]*C[5];\
  A[5]  -=  B[5]*C[0]   + B[11]*C[1]  + B[17]*C[2]   + B[23]*C[3]  + B[29]*C[4]  + B[35]*C[5];\
  A[6]  -=  B[0]*C[6]   + B[6]*C[7]   + B[12]*C[8]   + B[18]*C[9]  + B[24]*C[10] + B[30]*C[11];\
  A[7]  -=  B[1]*C[6]   + B[7]*C[7]   + B[13]*C[8]   + B[19]*C[9]  + B[25]*C[10] + B[31]*C[11];\
  A[8]  -=  B[2]*C[6]   + B[8]*C[7]   + B[14]*C[8]   + B[20]*C[9]  + B[26]*C[10] + B[32]*C[11];\
  A[9]  -=  B[3]*C[6]   + B[9]*C[7]   + B[15]*C[8]   + B[21]*C[9]  + B[27]*C[10] + B[33]*C[11];\
  A[10] -=  B[4]*C[6]   + B[10]*C[7]  + B[16]*C[8]   + B[22]*C[9]  + B[28]*C[10] + B[34]*C[11];\
  A[11] -=  B[5]*C[6]   + B[11]*C[7]  + B[17]*C[8]   + B[23]*C[9]  + B[29]*C[10] + B[35]*C[11];\
  A[12] -=  B[0]*C[12]  + B[6]*C[13]  + B[12]*C[14]  + B[18]*C[15] + B[24]*C[16] + B[30]*C[17];\
  A[13] -=  B[1]*C[12]  + B[7]*C[13]  + B[13]*C[14]  + B[19]*C[15] + B[25]*C[16] + B[31]*C[17];\
  A[14] -=  B[2]*C[12]  + B[8]*C[13]  + B[14]*C[14]  + B[20]*C[15] + B[26]*C[16] + B[32]*C[17];\
  A[15] -=  B[3]*C[12]  + B[9]*C[13]  + B[15]*C[14]  + B[21]*C[15] + B[27]*C[16] + B[33]*C[17];\
  A[16] -=  B[4]*C[12]  + B[10]*C[13] + B[16]*C[14]  + B[22]*C[15] + B[28]*C[16] + B[34]*C[17];\
  A[17] -=  B[5]*C[12]  + B[11]*C[13] + B[17]*C[14]  + B[23]*C[15] + B[29]*C[16] + B[35]*C[17];\
  A[18] -=  B[0]*C[18]  + B[6]*C[19]  + B[12]*C[20]  + B[18]*C[21] + B[24]*C[22] + B[30]*C[23];\
  A[19] -=  B[1]*C[18]  + B[7]*C[19]  + B[13]*C[20]  + B[19]*C[21] + B[25]*C[22] + B[31]*C[23];\
  A[20] -=  B[2]*C[18]  + B[8]*C[19]  + B[14]*C[20]  + B[20]*C[21] + B[26]*C[22] + B[32]*C[23];\
  A[21] -=  B[3]*C[18]  + B[9]*C[19]  + B[15]*C[20]  + B[21]*C[21] + B[27]*C[22] + B[33]*C[23];\
  A[22] -=  B[4]*C[18]  + B[10]*C[19] + B[16]*C[20]  + B[22]*C[21] + B[28]*C[22] + B[34]*C[23];\
  A[23] -=  B[5]*C[18]  + B[11]*C[19] + B[17]*C[20]  + B[23]*C[21] + B[29]*C[22] + B[35]*C[23];\
  A[24] -=  B[0]*C[24]  + B[6]*C[25]  + B[12]*C[26]  + B[18]*C[27] + B[24]*C[28] + B[30]*C[29];\
  A[25] -=  B[1]*C[24]  + B[7]*C[25]  + B[13]*C[26]  + B[19]*C[27] + B[25]*C[28] + B[31]*C[29];\
  A[26] -=  B[2]*C[24]  + B[8]*C[25]  + B[14]*C[26]  + B[20]*C[27] + B[26]*C[28] + B[32]*C[29];\
  A[27] -=  B[3]*C[24]  + B[9]*C[25]  + B[15]*C[26]  + B[21]*C[27] + B[27]*C[28] + B[33]*C[29];\
  A[28] -=  B[4]*C[24]  + B[10]*C[25] + B[16]*C[26]  + B[22]*C[27] + B[28]*C[28] + B[34]*C[29];\
  A[29] -=  B[5]*C[24]  + B[11]*C[25] + B[17]*C[26]  + B[23]*C[27] + B[29]*C[28] + B[35]*C[29];\
  A[30] -=  B[0]*C[30]  + B[6]*C[31]  + B[12]*C[32]  + B[18]*C[33] + B[24]*C[34] + B[30]*C[35];\
  A[31] -=  B[1]*C[30]  + B[7]*C[31]  + B[13]*C[32]  + B[19]*C[33] + B[25]*C[34] + B[31]*C[35];\
  A[32] -=  B[2]*C[30]  + B[8]*C[31]  + B[14]*C[32]  + B[20]*C[33] + B[26]*C[34] + B[32]*C[35];\
  A[33] -=  B[3]*C[30]  + B[9]*C[31]  + B[15]*C[32]  + B[21]*C[33] + B[27]*C[34] + B[33]*C[35];\
  A[34] -=  B[4]*C[30]  + B[10]*C[31] + B[16]*C[32]  + B[22]*C[33] + B[28]*C[34] + B[34]*C[35];\
  A[35] -=  B[5]*C[30]  + B[11]*C[31] + B[17]*C[32]  + B[23]*C[33] + B[29]*C[34] + B[35]*C[35];\
}

#define Kernel_A_gets_A_times_B_7(A,B,W) 0;\
{\
  PetscMemcpy(W,A,49*sizeof(MatScalar));\
  A[0]  =  W[0]*B[0]   + W[7]*B[1]   + W[14]*B[2]   + W[21]*B[3]  + W[28]*B[4]  + W[35]*B[5]  + W[42]*B[6];\
  A[1]  =  W[1]*B[0]   + W[8]*B[1]   + W[15]*B[2]   + W[22]*B[3]  + W[29]*B[4]  + W[36]*B[5]  + W[43]*B[6];\
  A[2]  =  W[2]*B[0]   + W[9]*B[1]   + W[16]*B[2]   + W[23]*B[3]  + W[30]*B[4]  + W[37]*B[5]  + W[44]*B[6];\
  A[3]  =  W[3]*B[0]   + W[10]*B[1]   + W[17]*B[2]   + W[24]*B[3]  + W[31]*B[4]  + W[38]*B[5]  + W[45]*B[6];\
  A[4]  =  W[4]*B[0]   + W[11]*B[1]  + W[18]*B[2]   + W[25]*B[3]  + W[32]*B[4]  + W[39]*B[5]  + W[46]*B[6];\
  A[5]  =  W[5]*B[0]   + W[12]*B[1]  + W[19]*B[2]   + W[26]*B[3]  + W[33]*B[4]  + W[40]*B[5]  + W[47]*B[6];\
  A[6]  =  W[6]*B[0]   + W[13]*B[1]   + W[20]*B[2]   + W[27]*B[3]  + W[34]*B[4] + W[41]*B[5]  + W[48]*B[6];\
  A[7]  =  W[0]*B[7]   + W[7]*B[8]   + W[14]*B[9]   + W[21]*B[10]  + W[28]*B[11]  + W[35]*B[12]  + W[42]*B[13];\
  A[8]  =  W[1]*B[7]   + W[8]*B[8]   + W[15]*B[9]   + W[22]*B[10]  + W[29]*B[11]  + W[36]*B[12]  + W[43]*B[13];\
  A[9]  =  W[2]*B[7]   + W[9]*B[8]   + W[16]*B[9]   + W[23]*B[10]  + W[30]*B[11]  + W[37]*B[12]  + W[44]*B[13];\
  A[10]  =  W[3]*B[7]   + W[10]*B[8]   + W[17]*B[9]   + W[24]*B[10]  + W[31]*B[11]  + W[38]*B[12]  + W[45]*B[13];\
  A[11]  =  W[4]*B[7]   + W[11]*B[8]  + W[18]*B[9]   + W[25]*B[10]  + W[32]*B[11]  + W[39]*B[12]  + W[46]*B[13];\
  A[12]  =  W[5]*B[7]   + W[12]*B[8]  + W[19]*B[9]   + W[26]*B[10]  + W[33]*B[11]  + W[40]*B[12]  + W[47]*B[13];\
  A[13]  =  W[6]*B[7]   + W[13]*B[8]   + W[20]*B[9]   + W[27]*B[10]  + W[34]*B[11] + W[41]*B[12]  + W[48]*B[13];\
  A[14]  =  W[0]*B[14]   + W[7]*B[15]   + W[14]*B[16]   + W[21]*B[17]  + W[28]*B[18]  + W[35]*B[19]  + W[42]*B[20];\
  A[15]  =  W[1]*B[14]   + W[8]*B[15]   + W[15]*B[16]   + W[22]*B[17]  + W[29]*B[18]  + W[36]*B[19]  + W[43]*B[20];\
  A[16]  =  W[2]*B[14]   + W[9]*B[15]   + W[16]*B[16]   + W[23]*B[17]  + W[30]*B[18]  + W[37]*B[19]  + W[44]*B[20];\
  A[17]  =  W[3]*B[14]   + W[10]*B[15]   + W[17]*B[16]   + W[24]*B[17]  + W[31]*B[18]  + W[38]*B[19]  + W[45]*B[20];\
  A[18]  =  W[4]*B[14]   + W[11]*B[15]  + W[18]*B[16]   + W[25]*B[17]  + W[32]*B[18]  + W[39]*B[19]  + W[46]*B[20];\
  A[19]  =  W[5]*B[14]   + W[12]*B[15]  + W[19]*B[16]   + W[26]*B[17]  + W[33]*B[18]  + W[40]*B[19]  + W[47]*B[20];\
  A[20]  =  W[6]*B[14]   + W[13]*B[15]   + W[20]*B[16]   + W[27]*B[17]  + W[34]*B[18] + W[41]*B[19]  + W[48]*B[20];\
  A[21]  =  W[0]*B[21]   + W[7]*B[22]   + W[14]*B[23]   + W[21]*B[24]  + W[28]*B[25]  + W[35]*B[26]  + W[42]*B[27];\
  A[22]  =  W[1]*B[21]   + W[8]*B[22]   + W[15]*B[23]   + W[22]*B[24]  + W[29]*B[25]  + W[36]*B[26]  + W[43]*B[27];\
  A[23]  =  W[2]*B[21]   + W[9]*B[22]   + W[16]*B[23]   + W[23]*B[24]  + W[30]*B[25]  + W[37]*B[26]  + W[44]*B[27];\
  A[24]  =  W[3]*B[21]   + W[10]*B[22]   + W[17]*B[23]   + W[24]*B[24]  + W[31]*B[25]  + W[38]*B[26]  + W[45]*B[27];\
  A[25]  =  W[4]*B[21]   + W[11]*B[22]  + W[18]*B[23]   + W[25]*B[24]  + W[32]*B[25]  + W[39]*B[26]  + W[46]*B[27];\
  A[26]  =  W[5]*B[21]   + W[12]*B[22]  + W[19]*B[23]   + W[26]*B[24]  + W[33]*B[25]  + W[40]*B[26]  + W[47]*B[27];\
  A[27]  =  W[6]*B[21]   + W[13]*B[22]   + W[20]*B[23]   + W[27]*B[24]  + W[34]*B[25] + W[41]*B[26]  + W[48]*B[27];\
  A[28]  =  W[0]*B[28]   + W[7]*B[29]   + W[14]*B[30]   + W[21]*B[31]  + W[28]*B[32]  + W[35]*B[33]  + W[42]*B[34];\
  A[29]  =  W[1]*B[28]   + W[8]*B[29]   + W[15]*B[30]   + W[22]*B[31]  + W[29]*B[32]  + W[36]*B[33]  + W[43]*B[34];\
  A[30]  =  W[2]*B[28]   + W[9]*B[29]   + W[16]*B[30]   + W[23]*B[31]  + W[30]*B[32]  + W[37]*B[33]  + W[44]*B[34];\
  A[31]  =  W[3]*B[28]   + W[10]*B[29]   + W[17]*B[30]   + W[24]*B[31]  + W[31]*B[32]  + W[38]*B[33]  + W[45]*B[34];\
  A[32]  =  W[4]*B[28]   + W[11]*B[29]  + W[18]*B[30]   + W[25]*B[31]  + W[32]*B[32]  + W[39]*B[33]  + W[46]*B[34];\
  A[33]  =  W[5]*B[28]   + W[12]*B[29]  + W[19]*B[30]   + W[26]*B[31]  + W[33]*B[32]  + W[40]*B[33]  + W[47]*B[34];\
  A[34]  =  W[6]*B[28]   + W[13]*B[29]   + W[20]*B[30]   + W[27]*B[31]  + W[34]*B[32] + W[41]*B[33]  + W[48]*B[34];\
  A[35]  =  W[0]*B[35]   + W[7]*B[36]   + W[14]*B[37]   + W[21]*B[38]  + W[28]*B[39]  + W[35]*B[40]  + W[42]*B[41];\
  A[36]  =  W[1]*B[35]   + W[8]*B[36]   + W[15]*B[37]   + W[22]*B[38]  + W[29]*B[39]  + W[36]*B[40]  + W[43]*B[41];\
  A[37]  =  W[2]*B[35]   + W[9]*B[36]   + W[16]*B[37]   + W[23]*B[38]  + W[30]*B[39]  + W[37]*B[40]  + W[44]*B[41];\
  A[38]  =  W[3]*B[35]   + W[10]*B[36]   + W[17]*B[37]   + W[24]*B[38]  + W[31]*B[39]  + W[38]*B[40]  + W[45]*B[41];\
  A[39]  =  W[4]*B[35]   + W[11]*B[36]  + W[18]*B[37]   + W[25]*B[38]  + W[32]*B[39]  + W[39]*B[40]  + W[46]*B[41];\
  A[40]  =  W[5]*B[35]   + W[12]*B[36]  + W[19]*B[37]   + W[26]*B[38]  + W[33]*B[39]  + W[40]*B[40]  + W[47]*B[41];\
  A[41]  =  W[6]*B[35]   + W[13]*B[36]   + W[20]*B[37]   + W[27]*B[38]  + W[34]*B[39] + W[41]*B[40]  + W[48]*B[41];\
  A[42]  =  W[0]*B[42]   + W[7]*B[43]   + W[14]*B[44]   + W[21]*B[45]  + W[28]*B[46]  + W[35]*B[47]  + W[42]*B[48];\
  A[43]  =  W[1]*B[42]   + W[8]*B[43]   + W[15]*B[44]   + W[22]*B[45]  + W[29]*B[46]  + W[36]*B[47]  + W[43]*B[48];\
  A[44]  =  W[2]*B[42]   + W[9]*B[43]   + W[16]*B[44]   + W[23]*B[45]  + W[30]*B[46]  + W[37]*B[47]  + W[44]*B[48];\
  A[45]  =  W[3]*B[42]   + W[10]*B[43]   + W[17]*B[44]   + W[24]*B[45]  + W[31]*B[46]  + W[38]*B[47]  + W[45]*B[48];\
  A[46]  =  W[4]*B[42]   + W[11]*B[43]  + W[18]*B[44]   + W[25]*B[45]  + W[32]*B[46]  + W[39]*B[47]  + W[46]*B[48];\
  A[47]  =  W[5]*B[42]   + W[12]*B[43]  + W[19]*B[44]   + W[26]*B[45]  + W[33]*B[46]  + W[40]*B[47]  + W[47]*B[48];\
  A[48]  =  W[6]*B[42]   + W[13]*B[43]   + W[20]*B[44]   + W[27]*B[45]  + W[34]*B[46] + W[41]*B[47]  + W[48]*B[48];\
}

/*
  Kernel_A_gets_A_minus_B_times_C_7: A = A - B * C with size bs=7

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define Kernel_A_gets_A_minus_B_times_C_7(A,B,C) 0;\
{\
  A[0]  -=  B[0]*C[0]   + B[7]*C[1]   + B[14]*C[2]   + B[21]*C[3]  + B[28]*C[4]  + B[35]*C[5]  + B[42]*C[6];\
  A[1]  -=  B[1]*C[0]   + B[8]*C[1]   + B[15]*C[2]   + B[22]*C[3]  + B[29]*C[4]  + B[36]*C[5]  + B[43]*C[6];\
  A[2]  -=  B[2]*C[0]   + B[9]*C[1]   + B[16]*C[2]   + B[23]*C[3]  + B[30]*C[4]  + B[37]*C[5]  + B[44]*C[6];\
  A[3]  -=  B[3]*C[0]   + B[10]*C[1]   + B[17]*C[2]   + B[24]*C[3]  + B[31]*C[4]  + B[38]*C[5]  + B[45]*C[6];\
  A[4]  -=  B[4]*C[0]   + B[11]*C[1]  + B[18]*C[2]   + B[25]*C[3]  + B[32]*C[4]  + B[39]*C[5]  + B[46]*C[6];\
  A[5]  -=  B[5]*C[0]   + B[12]*C[1]  + B[19]*C[2]   + B[26]*C[3]  + B[33]*C[4]  + B[40]*C[5]  + B[47]*C[6];\
  A[6]  -=  B[6]*C[0]   + B[13]*C[1]   + B[20]*C[2]   + B[27]*C[3]  + B[34]*C[4] + B[41]*C[5]  + B[48]*C[6];\
  A[7]  -=  B[0]*C[7]   + B[7]*C[8]   + B[14]*C[9]   + B[21]*C[10]  + B[28]*C[11]  + B[35]*C[12]  + B[42]*C[13];\
  A[8]  -=  B[1]*C[7]   + B[8]*C[8]   + B[15]*C[9]   + B[22]*C[10]  + B[29]*C[11]  + B[36]*C[12]  + B[43]*C[13];\
  A[9]  -=  B[2]*C[7]   + B[9]*C[8]   + B[16]*C[9]   + B[23]*C[10]  + B[30]*C[11]  + B[37]*C[12]  + B[44]*C[13];\
  A[10]  -=  B[3]*C[7]   + B[10]*C[8]   + B[17]*C[9]   + B[24]*C[10]  + B[31]*C[11]  + B[38]*C[12]  + B[45]*C[13];\
  A[11]  -=  B[4]*C[7]   + B[11]*C[8]  + B[18]*C[9]   + B[25]*C[10]  + B[32]*C[11]  + B[39]*C[12]  + B[46]*C[13];\
  A[12]  -=  B[5]*C[7]   + B[12]*C[8]  + B[19]*C[9]   + B[26]*C[10]  + B[33]*C[11]  + B[40]*C[12]  + B[47]*C[13];\
  A[13]  -=  B[6]*C[7]   + B[13]*C[8]   + B[20]*C[9]   + B[27]*C[10]  + B[34]*C[11] + B[41]*C[12]  + B[48]*C[13];\
  A[14]  -=  B[0]*C[14]   + B[7]*C[15]   + B[14]*C[16]   + B[21]*C[17]  + B[28]*C[18]  + B[35]*C[19]  + B[42]*C[20];\
  A[15]  -=  B[1]*C[14]   + B[8]*C[15]   + B[15]*C[16]   + B[22]*C[17]  + B[29]*C[18]  + B[36]*C[19]  + B[43]*C[20];\
  A[16]  -=  B[2]*C[14]   + B[9]*C[15]   + B[16]*C[16]   + B[23]*C[17]  + B[30]*C[18]  + B[37]*C[19]  + B[44]*C[20];\
  A[17]  -=  B[3]*C[14]   + B[10]*C[15]   + B[17]*C[16]   + B[24]*C[17]  + B[31]*C[18]  + B[38]*C[19]  + B[45]*C[20];\
  A[18]  -=  B[4]*C[14]   + B[11]*C[15]  + B[18]*C[16]   + B[25]*C[17]  + B[32]*C[18]  + B[39]*C[19]  + B[46]*C[20];\
  A[19]  -=  B[5]*C[14]   + B[12]*C[15]  + B[19]*C[16]   + B[26]*C[17]  + B[33]*C[18]  + B[40]*C[19]  + B[47]*C[20];\
  A[20]  -=  B[6]*C[14]   + B[13]*C[15]   + B[20]*C[16]   + B[27]*C[17]  + B[34]*C[18] + B[41]*C[19]  + B[48]*C[20];\
  A[21]  -=  B[0]*C[21]   + B[7]*C[22]   + B[14]*C[23]   + B[21]*C[24]  + B[28]*C[25]  + B[35]*C[26]  + B[42]*C[27];\
  A[22]  -=  B[1]*C[21]   + B[8]*C[22]   + B[15]*C[23]   + B[22]*C[24]  + B[29]*C[25]  + B[36]*C[26]  + B[43]*C[27];\
  A[23]  -=  B[2]*C[21]   + B[9]*C[22]   + B[16]*C[23]   + B[23]*C[24]  + B[30]*C[25]  + B[37]*C[26]  + B[44]*C[27];\
  A[24]  -=  B[3]*C[21]   + B[10]*C[22]   + B[17]*C[23]   + B[24]*C[24]  + B[31]*C[25]  + B[38]*C[26]  + B[45]*C[27];\
  A[25]  -=  B[4]*C[21]   + B[11]*C[22]  + B[18]*C[23]   + B[25]*C[24]  + B[32]*C[25]  + B[39]*C[26]  + B[46]*C[27];\
  A[26]  -=  B[5]*C[21]   + B[12]*C[22]  + B[19]*C[23]   + B[26]*C[24]  + B[33]*C[25]  + B[40]*C[26]  + B[47]*C[27];\
  A[27]  -=  B[6]*C[21]   + B[13]*C[22]   + B[20]*C[23]   + B[27]*C[24]  + B[34]*C[25] + B[41]*C[26]  + B[48]*C[27];\
  A[28]  -=  B[0]*C[28]   + B[7]*C[29]   + B[14]*C[30]   + B[21]*C[31]  + B[28]*C[32]  + B[35]*C[33]  + B[42]*C[34];\
  A[29]  -=  B[1]*C[28]   + B[8]*C[29]   + B[15]*C[30]   + B[22]*C[31]  + B[29]*C[32]  + B[36]*C[33]  + B[43]*C[34];\
  A[30]  -=  B[2]*C[28]   + B[9]*C[29]   + B[16]*C[30]   + B[23]*C[31]  + B[30]*C[32]  + B[37]*C[33]  + B[44]*C[34];\
  A[31]  -=  B[3]*C[28]   + B[10]*C[29]   + B[17]*C[30]   + B[24]*C[31]  + B[31]*C[32]  + B[38]*C[33]  + B[45]*C[34];\
  A[32]  -=  B[4]*C[28]   + B[11]*C[29]  + B[18]*C[30]   + B[25]*C[31]  + B[32]*C[32]  + B[39]*C[33]  + B[46]*C[34];\
  A[33]  -=  B[5]*C[28]   + B[12]*C[29]  + B[19]*C[30]   + B[26]*C[31]  + B[33]*C[32]  + B[40]*C[33]  + B[47]*C[34];\
  A[34]  -=  B[6]*C[28]   + B[13]*C[29]   + B[20]*C[30]   + B[27]*C[31]  + B[34]*C[32] + B[41]*C[33]  + B[48]*C[34];\
  A[35]  -=  B[0]*C[35]   + B[7]*C[36]   + B[14]*C[37]   + B[21]*C[38]  + B[28]*C[39]  + B[35]*C[40]  + B[42]*C[41];\
  A[36]  -=  B[1]*C[35]   + B[8]*C[36]   + B[15]*C[37]   + B[22]*C[38]  + B[29]*C[39]  + B[36]*C[40]  + B[43]*C[41];\
  A[37]  -=  B[2]*C[35]   + B[9]*C[36]   + B[16]*C[37]   + B[23]*C[38]  + B[30]*C[39]  + B[37]*C[40]  + B[44]*C[41];\
  A[38]  -=  B[3]*C[35]   + B[10]*C[36]   + B[17]*C[37]   + B[24]*C[38]  + B[31]*C[39]  + B[38]*C[40]  + B[45]*C[41];\
  A[39]  -=  B[4]*C[35]   + B[11]*C[36]  + B[18]*C[37]   + B[25]*C[38]  + B[32]*C[39]  + B[39]*C[40]  + B[46]*C[41];\
  A[40]  -=  B[5]*C[35]   + B[12]*C[36]  + B[19]*C[37]   + B[26]*C[38]  + B[33]*C[39]  + B[40]*C[40]  + B[47]*C[41];\
  A[41]  -=  B[6]*C[35]   + B[13]*C[36]   + B[20]*C[37]   + B[27]*C[38]  + B[34]*C[39] + B[41]*C[40]  + B[48]*C[41];\
  A[42]  -=  B[0]*C[42]   + B[7]*C[43]   + B[14]*C[44]   + B[21]*C[45]  + B[28]*C[46]  + B[35]*C[47]  + B[42]*C[48];\
  A[43]  -=  B[1]*C[42]   + B[8]*C[43]   + B[15]*C[44]   + B[22]*C[45]  + B[29]*C[46]  + B[36]*C[47]  + B[43]*C[48];\
  A[44]  -=  B[2]*C[42]   + B[9]*C[43]   + B[16]*C[44]   + B[23]*C[45]  + B[30]*C[46]  + B[37]*C[47]  + B[44]*C[48];\
  A[45]  -=  B[3]*C[42]   + B[10]*C[43]   + B[17]*C[44]   + B[24]*C[45]  + B[31]*C[46]  + B[38]*C[47]  + B[45]*C[48];\
  A[46]  -=  B[4]*C[42]   + B[11]*C[43]  + B[18]*C[44]   + B[25]*C[45]  + B[32]*C[46]  + B[39]*C[47]  + B[46]*C[48];\
  A[47]  -=  B[5]*C[42]   + B[12]*C[43]  + B[19]*C[44]   + B[26]*C[45]  + B[33]*C[46]  + B[40]*C[47]  + B[47]*C[48];\
  A[48]  -=  B[6]*C[42]   + B[13]*C[43]   + B[20]*C[44]   + B[27]*C[45]  + B[34]*C[46] + B[41]*C[47]  + B[48]*C[48];\
}


#endif
