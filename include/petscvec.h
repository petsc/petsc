/*
    Defines the vector component of PETSc. Vectors generally represent
  degrees of freedom for finite element/finite difference functions
  on a grid. They have more mathematical structure then simple arrays.
*/
#ifndef PETSCVEC_H
#define PETSCVEC_H

#include <petscsys.h>
#include <petscsftypes.h> /* for VecScatter, VecScatterType */
#include <petscis.h>
#include <petscdevicetypes.h>
#include <petscviewer.h>

/* SUBMANSEC = Vec */

/*S
     Vec - Abstract PETSc vector object

   Level: beginner

.seealso: [](doc_vector), [](chapter_vectors), `VecCreate()`, `VecType`, `VecSetType()`
S*/
typedef struct _p_Vec *Vec;

/*E
  ScatterMode - Determines the direction of a scatter

  Values:
+  `SCATTER_FORWARD` - Scatters the values as dictated by the `VecScatterCreate()` call
.  `SCATTER_REVERSE` - Moves the values in the opposite direction than the directions indicated in the `VecScatterCreate()` call
.  `SCATTER_FORWARD_LOCAL` - Scatters the values as dictated by the `VecScatterCreate()` call except NO MPI communication is done
-  `SCATTER_REVERSE_LOCAL` - Moves the values in the opposite direction than the directions indicated in the `VecScatterCreate()` call
                             except NO MPI communication is done

  Level: beginner

.seealso: [](chapter_vectors), `VecScatter`, `VecScatterBegin()`, `VecScatterEnd()`, `SCATTER_FORWARD`, `SCATTER_REVERSE`, `SCATTER_FORWARD_LOCAL`, `SCATTER_REVERSE_LOCAL`
E*/
typedef enum {
  SCATTER_FORWARD       = 0,
  SCATTER_REVERSE       = 1,
  SCATTER_FORWARD_LOCAL = 2,
  SCATTER_REVERSE_LOCAL = 3
} ScatterMode;

/*MC
    SCATTER_FORWARD - Scatters the values as dictated by the `VecScatterCreate()` call

    Level: beginner

.seealso: [](chapter_vectors), `VecScatter`, `ScatterMode`, `VecScatterCreate()`, `VecScatterBegin()`, `VecScatterEnd()`, `SCATTER_REVERSE`, `SCATTER_FORWARD_LOCAL`,
          `SCATTER_REVERSE_LOCAL`
M*/

/*MC
    SCATTER_REVERSE - Moves the values in the opposite direction then the directions indicated in
         in the `VecScatterCreate()`

    Level: beginner

.seealso: [](chapter_vectors), `VecScatter`, `ScatterMode`, `VecScatterCreate()`, `VecScatterBegin()`, `VecScatterEnd()`, `SCATTER_FORWARD`, `SCATTER_FORWARD_LOCAL`,
          `SCATTER_REVERSE_LOCAL`
M*/

/*MC
    SCATTER_FORWARD_LOCAL - Scatters the values as dictated by the `VecScatterCreate()` call except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: [](chapter_vectors), `VecScatter`, `ScatterMode`, `VecScatterCreate()`, `VecScatterBegin()`, `VecScatterEnd()`, `SCATTER_REVERSE`, `SCATTER_FORWARD`,
          `SCATTER_REVERSE_LOCAL`
M*/

/*MC
    SCATTER_REVERSE_LOCAL - Moves the values in the opposite direction then the directions indicated in
         in the `VecScatterCreate()`  except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: [](chapter_vectors), `VecScatter`, `ScatterMode`, `VecScatterCreate()`, `VecScatterBegin()`, `VecScatterEnd()`, `SCATTER_FORWARD`, `SCATTER_FORWARD_LOCAL`,
          `SCATTER_REVERSE`
M*/

/*J
    VecType - String with the name of a PETSc vector

   Level: beginner

.seealso: [](doc_vector), [](chapter_vectors), `VecSetType()`, `Vec`, `VecCreate()`, `VecDestroy()`
J*/
typedef const char *VecType;
#define VECSEQ         "seq"
#define VECMPI         "mpi"
#define VECSTANDARD    "standard" /* seq on one process and mpi on several */
#define VECSHARED      "shared"
#define VECSEQVIENNACL "seqviennacl"
#define VECMPIVIENNACL "mpiviennacl"
#define VECVIENNACL    "viennacl" /* seqviennacl on one process and mpiviennacl on several */
#define VECSEQCUDA     "seqcuda"
#define VECMPICUDA     "mpicuda"
#define VECCUDA        "cuda" /* seqcuda on one process and mpicuda on several */
#define VECSEQHIP      "seqhip"
#define VECMPIHIP      "mpihip"
#define VECHIP         "hip" /* seqcuda on one process and mpicuda on several */
#define VECNEST        "nest"
#define VECSEQKOKKOS   "seqkokkos"
#define VECMPIKOKKOS   "mpikokkos"
#define VECKOKKOS      "kokkos" /* seqkokkos on one process and mpikokkos on several */

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscErrorCode VecScatterSetType(VecScatter, VecScatterType);
PETSC_EXTERN PetscErrorCode VecScatterGetType(VecScatter, VecScatterType *);
PETSC_EXTERN PetscErrorCode VecScatterSetFromOptions(VecScatter);
PETSC_EXTERN PetscErrorCode VecScatterRegister(const char[], PetscErrorCode (*)(VecScatter));
PETSC_EXTERN PetscErrorCode VecScatterCreate(Vec, IS, Vec, IS, VecScatter *);

/* Logging support */
#define REAL_FILE_CLASSID 1211213
#define VEC_FILE_CLASSID  1211214
PETSC_EXTERN PetscClassId VEC_CLASSID;
PETSC_EXTERN PetscClassId PETSCSF_CLASSID;

PETSC_EXTERN PetscErrorCode VecInitializePackage(void);
PETSC_EXTERN PetscErrorCode VecFinalizePackage(void);

PETSC_EXTERN PetscErrorCode VecCreate(MPI_Comm, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateSeq(MPI_Comm, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateMPI(MPI_Comm, PetscInt, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateSeqWithArray(MPI_Comm, PetscInt, PetscInt, const PetscScalar[], Vec *);
PETSC_EXTERN PetscErrorCode VecCreateMPIWithArray(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscScalar[], Vec *);
PETSC_EXTERN PetscErrorCode VecCreateShared(MPI_Comm, PetscInt, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateNode(MPI_Comm, PetscInt, PetscInt, Vec *);

PETSC_EXTERN PetscErrorCode VecSetFromOptions(Vec);
PETSC_EXTERN PetscErrorCode VecViewFromOptions(Vec, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode VecSetUp(Vec);
PETSC_EXTERN PetscErrorCode VecDestroy(Vec *);
PETSC_EXTERN PetscErrorCode VecZeroEntries(Vec);
PETSC_EXTERN PetscErrorCode VecSetOptionsPrefix(Vec, const char[]);
PETSC_EXTERN PetscErrorCode VecAppendOptionsPrefix(Vec, const char[]);
PETSC_EXTERN PetscErrorCode VecGetOptionsPrefix(Vec, const char *[]);

PETSC_EXTERN PetscErrorCode VecSetSizes(Vec, PetscInt, PetscInt);

PETSC_EXTERN PetscErrorCode VecDotNorm2(Vec, Vec, PetscScalar *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecDot(Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecDotRealPart(Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode VecTDot(Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecMDot(Vec, PetscInt, const Vec[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecMTDot(Vec, PetscInt, const Vec[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecGetSubVector(Vec, IS, Vec *);
PETSC_EXTERN PetscErrorCode VecRestoreSubVector(Vec, IS, Vec *);
PETSC_EXTERN PetscErrorCode VecConcatenate(PetscInt, const Vec[], Vec *, IS *[]);

/*E
    NormType - determines what type of norm to compute

    Values:
+    `NORM_1` - the one norm, ||v|| = sum_i | v_i |. ||A|| = max_j || v_*j ||, maximum column sum
.    `NORM_2` - the two norm, ||v|| = sqrt(sum_i |v_i|^2) (vectors only)
.    `NORM_FROBENIUS` - ||A|| = sqrt(sum_ij |A_ij|^2), same as `NORM_2` for vectors
.    `NORM_INFINITY` - ||v|| = max_i |v_i|. ||A|| = max_i || v_i* ||, maximum row sum
-    `NORM_1_AND_2` - computes both the 1 and 2 norm of a vector

    Level: beginner

  Note:
  The `v` above represents a `Vec` while the `A` represents a `Mat`

.seealso: [](chapter_vectors), `Vec`, `Mat`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `MatNorm()`, `NORM_1`,
          `NORM_2`, `NORM_FROBENIUS`, `NORM_INFINITY`, `NORM_1_AND_2`
E*/
typedef enum {
  NORM_1         = 0,
  NORM_2         = 1,
  NORM_FROBENIUS = 2,
  NORM_INFINITY  = 3,
  NORM_1_AND_2   = 4
} NormType;
PETSC_EXTERN const char *const NormTypes[];
#define NORM_MAX NORM_INFINITY

/*MC
     NORM_1 - the one norm, ||v|| = sum_i | v_i |. ||A|| = max_j || v_*j ||, maximum column sum

   Level: beginner

.seealso: [](chapter_vectors), `NormType`, `MatNorm()`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `NORM_2`, `NORM_FROBENIUS`,
          `NORM_INFINITY`, `NORM_1_AND_2`
M*/

/*MC
     NORM_2 - the two norm, ||v|| = sqrt(sum_i |v_i|^2) (vectors only)

   Level: beginner

.seealso: [](chapter_vectors), `NormType`, `MatNorm()`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `NORM_1`, `NORM_FROBENIUS`,
          `NORM_INFINITY`, `NORM_1_AND_2`
M*/

/*MC
     NORM_FROBENIUS - ||A|| = sqrt(sum_ij |A_ij|^2), same as `NORM_2` for vectors

   Level: beginner

.seealso: [](chapter_vectors), `NormType`, `MatNorm()`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `NORM_1`, `NORM_2`,
          `NORM_INFINITY`, `NORM_1_AND_2`
M*/

/*MC
     NORM_INFINITY - ||v|| = max_i |v_i|. ||A|| = max_i || v_i* ||, maximum row sum

   Level: beginner

.seealso: [](chapter_vectors), `NormType`, `MatNorm()`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `NORM_1`, `NORM_2`,
          `NORM_FROBENIUS`, `NORM_1_AND_2`
M*/

/*MC
     NORM_1_AND_2 - computes both the 1 and 2 norm of a vector. The values are stored in two adjacent `PetscReal` memory locations

   Level: beginner

.seealso: [](chapter_vectors), `NormType`, `MatNorm()`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `NORM_1`, `NORM_2`,
          `NORM_FROBENIUS`, `NORM_INFINITY`
M*/

/*MC
     NORM_MAX - see `NORM_INFINITY`

   Level: beginner
M*/

/*E
    ReductionType - determines what type of column reduction (one that is not a type of norm defined in `NormType`) to compute

    Values:
+  `REDUCTION_SUM_REALPART` - sum of real part of each matrix column
.  `REDUCTION_SUM_IMAGINARYPART` - sum of imaginary part of each matrix column
.  `REDUCTION_MEAN_REALPART` - arithmetic mean of real part of each matrix column
-  `REDUCTION_MEAN_IMAGINARYPART` - arithmetic mean of imaginary part of each matrix column

    Level: beginner

    Developer Note:
  The constants defined in `ReductionType` MUST BE DISTINCT from those defined in `NormType`.
  This is because `MatGetColumnReductions()` is used to compute both norms and other types of reductions,
  and the constants defined in both `NormType` and `ReductionType` are used to designate the desired operation.

.seealso: [](chapter_vectors), `MatGetColumnReductions()`, `MatGetColumnNorms()`, `NormType`, `REDUCTION_SUM_REALPART`,
          `REDUCTION_SUM_IMAGINARYPART`, `REDUCTION_MEAN_REALPART`, `REDUCTION_NORM_1`, `REDUCTION_NORM_2`, `REDUCTION_NORM_FROBENIUS`, `REDUCTION_NORM_INFINITY`
E*/
typedef enum {
  REDUCTION_SUM_REALPART       = 10,
  REDUCTION_MEAN_REALPART      = 11,
  REDUCTION_SUM_IMAGINARYPART  = 12,
  REDUCTION_MEAN_IMAGINARYPART = 13
} ReductionType;

/*MC
     REDUCTION_SUM_REALPART - sum of real part of matrix column

   Level: beginner

.seealso: [](chapter_vectors), `ReductionType`, `MatGetColumnReductions()`, `REDUCTION_SUM_IMAGINARYPART`, `REDUCTION_MEAN_REALPART`, `REDUCTION_NORM_1`,
          `REDUCTION_NORM_2`, `REDUCTION_NORM_FROBENIUS`, `REDUCTION_NORM_INFINITY`
M*/

/*MC
     REDUCTION_SUM_IMAGINARYPART - sum of imaginary part of matrix column

   Level: beginner

.seealso: [](chapter_vectors), `ReductionType`, `MatGetColumnReductions()`, `REDUCTION_SUM_REALPART`, `REDUCTION_MEAN_IMAGINARYPART`, `REDUCTION_NORM_1`,
          `REDUCTION_NORM_2`, `REDUCTION_NORM_FROBENIUS`, `REDUCTION_NORM_INFINITY`
M*/

/*MC
     REDUCTION_MEAN_REALPART - arithmetic mean of real part of matrix column

   Level: beginner

.seealso: [](chapter_vectors), `ReductionType`, `MatGetColumnReductions()`, `REDUCTION_MEAN_IMAGINARYPART`, `REDUCTION_SUM_REALPART`, `REDUCTION_NORM_1`,
          `REDUCTION_NORM_2`, `REDUCTION_NORM_FROBENIUS`, `REDUCTION_NORM_INFINITY`
M*/

/*MC
     REDUCTION_MEAN_IMAGINARYPART - arithmetic mean of imaginary part of matrix column

   Level: beginner

.seealso: [](chapter_vectors), `ReductionType`, `MatGetColumnReductions()`, `REDUCTION_MEAN_REALPART`, `REDUCTION_SUM_IMAGINARYPART`, `REDUCTION_NORM_1`,
          `REDUCTION_NORM_2`, `REDUCTION_NORM_FROBENIUS`, `REDUCTION_NORM_INFINITY`
M*/

PETSC_EXTERN PetscErrorCode VecNorm(Vec, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode VecNormAvailable(Vec, NormType, PetscBool *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecNormalize(Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode VecSum(Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecMean(Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecMax(Vec, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecMin(Vec, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecScale(Vec, PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecSetRandom(Vec, PetscRandom);
PETSC_EXTERN PetscErrorCode VecSet(Vec, PetscScalar);
PETSC_EXTERN PetscErrorCode VecSetInf(Vec);
PETSC_EXTERN PetscErrorCode VecSwap(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecAXPY(Vec, PetscScalar, Vec);
PETSC_EXTERN PetscErrorCode VecAXPBY(Vec, PetscScalar, PetscScalar, Vec);
PETSC_EXTERN PetscErrorCode VecMAXPY(Vec, PetscInt, const PetscScalar[], Vec[]);
PETSC_EXTERN PetscErrorCode VecAYPX(Vec, PetscScalar, Vec);
PETSC_EXTERN PetscErrorCode VecWAXPY(Vec, PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecAXPBYPCZ(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecPointwiseMax(Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecPointwiseMaxAbs(Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecPointwiseMin(Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecPointwiseMult(Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecPointwiseDivide(Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecMaxPointwiseDivide(Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode VecShift(Vec, PetscScalar);
PETSC_EXTERN PetscErrorCode VecReciprocal(Vec);
PETSC_EXTERN PetscErrorCode VecPermute(Vec, IS, PetscBool);
PETSC_EXTERN PetscErrorCode VecSqrtAbs(Vec);
PETSC_EXTERN PetscErrorCode VecLog(Vec);
PETSC_EXTERN PetscErrorCode VecExp(Vec);
PETSC_EXTERN PetscErrorCode VecAbs(Vec);
PETSC_EXTERN PetscErrorCode VecDuplicate(Vec, Vec *);
PETSC_EXTERN PetscErrorCode VecDuplicateVecs(Vec, PetscInt, Vec *[]);
PETSC_EXTERN PetscErrorCode VecDestroyVecs(PetscInt, Vec *[]);
PETSC_EXTERN PetscErrorCode VecStrideNormAll(Vec, NormType, PetscReal[]);
PETSC_EXTERN PetscErrorCode VecStrideMaxAll(Vec, PetscInt[], PetscReal[]);
PETSC_EXTERN PetscErrorCode VecStrideMinAll(Vec, PetscInt[], PetscReal[]);
PETSC_EXTERN PetscErrorCode VecStrideScaleAll(Vec, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecStrideSumAll(Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecUniqueEntries(Vec, PetscInt *, PetscScalar **);

PETSC_EXTERN PetscErrorCode VecStrideNorm(Vec, PetscInt, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode VecStrideMax(Vec, PetscInt, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecStrideMin(Vec, PetscInt, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecStrideScale(Vec, PetscInt, PetscScalar);
PETSC_EXTERN PetscErrorCode VecStrideSum(Vec, PetscInt, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecStrideSet(Vec, PetscInt, PetscScalar);

PETSC_EXTERN PetscErrorCode VecStrideGather(Vec, PetscInt, Vec, InsertMode);
PETSC_EXTERN PetscErrorCode VecStrideScatter(Vec, PetscInt, Vec, InsertMode);
PETSC_EXTERN PetscErrorCode VecStrideGatherAll(Vec, Vec[], InsertMode);
PETSC_EXTERN PetscErrorCode VecStrideScatterAll(Vec[], Vec, InsertMode);

PETSC_EXTERN PetscErrorCode VecStrideSubSetScatter(Vec, PetscInt, const PetscInt[], const PetscInt[], Vec, InsertMode);
PETSC_EXTERN PetscErrorCode VecStrideSubSetGather(Vec, PetscInt, const PetscInt[], const PetscInt[], Vec, InsertMode);

PETSC_EXTERN PetscErrorCode VecSetValues(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode VecGetValues(Vec, PetscInt, const PetscInt[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecAssemblyBegin(Vec);
PETSC_EXTERN PetscErrorCode VecAssemblyEnd(Vec);
PETSC_EXTERN PetscErrorCode VecStashSetInitialSize(Vec, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode VecStashView(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecStashViewFromOptions(Vec, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode VecStashGetInfo(Vec, PetscInt *, PetscInt *, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode VecSetPreallocationCOO(Vec, PetscCount, const PetscInt[]);
PETSC_EXTERN PetscErrorCode VecSetPreallocationCOOLocal(Vec, PetscCount, PetscInt[]);
PETSC_EXTERN PetscErrorCode VecSetValuesCOO(Vec, const PetscScalar[], InsertMode);

/*MC
   VecSetValue - Set a single entry into a vector.

   Synopsis:
   #include <petscvec.h>
   PetscErrorCode VecSetValue(Vec v,PetscInt row,PetscScalar value, InsertMode mode);

   Not Collective

   Input Parameters:
+  v - the vector
.  row - the row location of the entry
.  value - the value to insert
-  mode - either `INSERT_VALUES` or `ADD_VALUES`

   Level: beginner

   Notes:
   For efficiency one should use `VecSetValues()` and set several or
   many values simultaneously if possible.

   These values may be cached, so `VecAssemblyBegin()` and `VecAssemblyEnd()`
   MUST be called after all calls to `VecSetValue()` have been completed.

   `VecSetValue()` uses 0-based indices in Fortran as well as in C.

.seealso: [](chapter_vectors), `VecSetValues()`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValuesBlockedLocal()`, `VecSetValueLocal()`
M*/
static inline PetscErrorCode VecSetValue(Vec v, PetscInt i, PetscScalar va, InsertMode mode)
{
  return VecSetValues(v, 1, &i, &va, mode);
}

PETSC_EXTERN PetscErrorCode VecSetBlockSize(Vec, PetscInt);
PETSC_EXTERN PetscErrorCode VecGetBlockSize(Vec, PetscInt *);
PETSC_EXTERN PetscErrorCode VecSetValuesBlocked(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFunctionList VecList;
PETSC_EXTERN PetscErrorCode    VecSetType(Vec, VecType);
PETSC_EXTERN PetscErrorCode    VecGetType(Vec, VecType *);
PETSC_EXTERN PetscErrorCode    VecRegister(const char[], PetscErrorCode (*)(Vec));

PETSC_EXTERN PetscErrorCode VecScatterBegin(VecScatter, Vec, Vec, InsertMode, ScatterMode);
PETSC_EXTERN PetscErrorCode VecScatterEnd(VecScatter, Vec, Vec, InsertMode, ScatterMode);
PETSC_EXTERN PetscErrorCode VecScatterDestroy(VecScatter *);
PETSC_EXTERN PetscErrorCode VecScatterSetUp(VecScatter);
PETSC_EXTERN PetscErrorCode VecScatterCopy(VecScatter, VecScatter *);
PETSC_EXTERN PetscErrorCode VecScatterView(VecScatter, PetscViewer);
PETSC_EXTERN PetscErrorCode VecScatterViewFromOptions(VecScatter, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode VecScatterRemap(VecScatter, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode VecScatterGetMerged(VecScatter, PetscBool *);

PETSC_EXTERN PetscErrorCode VecGetArray4d(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray4d(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecGetArray3d(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ***[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray3d(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ***[]);
PETSC_EXTERN PetscErrorCode VecGetArray2d(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray2d(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecGetArray1d(Vec, PetscInt, PetscInt, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray1d(Vec, PetscInt, PetscInt, PetscScalar *[]);

PETSC_EXTERN PetscErrorCode VecGetArray4dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecGetArray4dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray4dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecGetArray3dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ***[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray3dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ***[]);
PETSC_EXTERN PetscErrorCode VecGetArray2dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray2dWrite(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecGetArray1dWrite(Vec, PetscInt, PetscInt, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray1dWrite(Vec, PetscInt, PetscInt, PetscScalar *[]);

PETSC_EXTERN PetscErrorCode VecGetArray4dRead(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray4dRead(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ****[]);
PETSC_EXTERN PetscErrorCode VecGetArray3dRead(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ***[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray3dRead(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar ***[]);
PETSC_EXTERN PetscErrorCode VecGetArray2dRead(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray2dRead(Vec, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecGetArray1dRead(Vec, PetscInt, PetscInt, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode VecRestoreArray1dRead(Vec, PetscInt, PetscInt, PetscScalar *[]);

PETSC_EXTERN PetscErrorCode VecPlaceArray(Vec, const PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecResetArray(Vec);
PETSC_EXTERN PetscErrorCode VecReplaceArray(Vec, const PetscScalar[]);

PETSC_EXTERN PetscErrorCode VecGetArrays(const Vec[], PetscInt, PetscScalar **[]);
PETSC_EXTERN PetscErrorCode VecRestoreArrays(const Vec[], PetscInt, PetscScalar **[]);

PETSC_EXTERN PetscErrorCode VecView(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecViewNative(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecEqual(Vec, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode VecLoad(Vec, PetscViewer);

PETSC_EXTERN PetscErrorCode VecGetSize(Vec, PetscInt *);
PETSC_EXTERN PetscErrorCode VecGetLocalSize(Vec, PetscInt *);
PETSC_EXTERN PetscErrorCode VecGetOwnershipRange(Vec, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode VecGetOwnershipRanges(Vec, const PetscInt *[]);

PETSC_EXTERN PetscErrorCode VecSetLocalToGlobalMapping(Vec, ISLocalToGlobalMapping);
PETSC_EXTERN PetscErrorCode VecSetValuesLocal(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);

PETSC_EXTERN PetscErrorCode VecViennaCLGetCLContext(Vec, PETSC_UINTPTR_T *);
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLQueue(Vec, PETSC_UINTPTR_T *);
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemRead(Vec, PETSC_UINTPTR_T *);
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemWrite(Vec, PETSC_UINTPTR_T *);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMemWrite(Vec);
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMem(Vec, PETSC_UINTPTR_T *);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMem(Vec);

/*MC
   VecSetValueLocal - Set a single entry into a vector using the local numbering, see `VecSetValuesLocal()`

   Synopsis:
   #include <petscvec.h>
   PetscErrorCode VecSetValueLocal(Vec v,PetscInt row,PetscScalar value, InsertMode mode);

   Not Collective

   Input Parameters:
+  v - the vector
.  row - the row location of the entry
.  value - the value to insert
-  mode - either `INSERT_VALUES` or `ADD_VALUES`

   Level: beginner

   Notes:
   For efficiency one should use `VecSetValuesLocal()` and set several or
   many values simultaneously if possible.

   These values may be cached, so `VecAssemblyBegin()` and `VecAssemblyEnd()`
   MUST be called after all calls to `VecSetValueLocal()` have been completed.

   `VecSetValues()` uses 0-based indices in Fortran as well as in C.

.seealso: [](chapter_vectors), `VecSetValuesLocal()`, `VecSetValues()`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValuesBlockedLocal()`, `VecSetValue()`
M*/
static inline PetscErrorCode VecSetValueLocal(Vec v, PetscInt i, PetscScalar va, InsertMode mode)
{
  return VecSetValuesLocal(v, 1, &i, &va, mode);
}

/*MC
   VecCheckAssembled - checks if values have been changed in the vector, by `VecSetValues()` or related routines,  but it has not been assembled

   Synopsis:
   #include <petscvec.h>
   VecCheckAssembled(Vec v);

   Not Collective

   Input Parameter:
.  v - the vector to check

   Level: developer

   Note:
   After calls to `VecSetValues()` and related routines on must call ``VecAssemblyBegin()` and `VecAssemblyEnd()` before using the vector

.seealso: [](chapter_vectors), `Vec`, `VecSetValues()`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`
M*/
#define VecCheckAssembled(v) PetscCheck(v->stash.insertmode == NOT_SET_VALUES, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled vector, did you call VecAssemblyBegin()/VecAssemblyEnd()?");

PETSC_EXTERN PetscErrorCode VecSetValuesBlockedLocal(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode VecGetLocalToGlobalMapping(Vec, ISLocalToGlobalMapping *);

PETSC_EXTERN PetscErrorCode VecDotBegin(Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecDotEnd(Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecTDotBegin(Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecTDotEnd(Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecNormBegin(Vec, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode VecNormEnd(Vec, NormType, PetscReal *);

PETSC_EXTERN PetscErrorCode VecMDotBegin(Vec, PetscInt, const Vec[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecMDotEnd(Vec, PetscInt, const Vec[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecMTDotBegin(Vec, PetscInt, const Vec[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode VecMTDotEnd(Vec, PetscInt, const Vec[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscCommSplitReductionBegin(MPI_Comm);

PETSC_EXTERN PetscErrorCode VecBindToCPU(Vec, PetscBool);
PETSC_DEPRECATED_FUNCTION("Use VecBindToCPU (since v3.13)") static inline PetscErrorCode VecPinToCPU(Vec v, PetscBool flg)
{
  return VecBindToCPU(v, flg);
}
PETSC_EXTERN PetscErrorCode VecBoundToCPU(Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode VecSetBindingPropagates(Vec, PetscBool);
PETSC_EXTERN PetscErrorCode VecGetBindingPropagates(Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode VecSetPinnedMemoryMin(Vec, size_t);
PETSC_EXTERN PetscErrorCode VecGetPinnedMemoryMin(Vec, size_t *);

PETSC_EXTERN PetscErrorCode VecGetOffloadMask(Vec, PetscOffloadMask *);

typedef enum {
  VEC_IGNORE_OFF_PROC_ENTRIES,
  VEC_IGNORE_NEGATIVE_INDICES,
  VEC_SUBSET_OFF_PROC_ENTRIES
} VecOption;
PETSC_EXTERN PetscErrorCode VecSetOption(Vec, VecOption, PetscBool);

PETSC_EXTERN PetscErrorCode VecGetArray(Vec, PetscScalar **);
PETSC_EXTERN PetscErrorCode VecGetArrayWrite(Vec, PetscScalar **);
PETSC_EXTERN PetscErrorCode VecGetArrayRead(Vec, const PetscScalar **);
PETSC_EXTERN PetscErrorCode VecRestoreArray(Vec, PetscScalar **);
PETSC_EXTERN PetscErrorCode VecRestoreArrayWrite(Vec, PetscScalar **);
PETSC_EXTERN PetscErrorCode VecRestoreArrayRead(Vec, const PetscScalar **);
PETSC_EXTERN PetscErrorCode VecCreateLocalVector(Vec, Vec *);
PETSC_EXTERN PetscErrorCode VecGetLocalVector(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecRestoreLocalVector(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecGetLocalVectorRead(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecRestoreLocalVectorRead(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecGetArrayAndMemType(Vec, PetscScalar **, PetscMemType *);
PETSC_EXTERN PetscErrorCode VecRestoreArrayAndMemType(Vec, PetscScalar **);
PETSC_EXTERN PetscErrorCode VecGetArrayReadAndMemType(Vec, const PetscScalar **, PetscMemType *);
PETSC_EXTERN PetscErrorCode VecRestoreArrayReadAndMemType(Vec, const PetscScalar **);
PETSC_EXTERN PetscErrorCode VecGetArrayWriteAndMemType(Vec, PetscScalar **, PetscMemType *);
PETSC_EXTERN PetscErrorCode VecRestoreArrayWriteAndMemType(Vec, PetscScalar **);

/*@C
   VecGetArrayPair - Accesses a pair of pointers for two vectors that may be common. When not common the first pointer is read only

   Logically Collective; No Fortran Support

   Input Parameters:
+  x - the vector
-  y - the second vector

   Output Parameters:
+  xv - location to put pointer to the first array
-  yv - location to put pointer to the second array

   Level: developer

.seealso: [](chapter_vectors), `VecGetArray()`, `VecGetArrayRead()`, `VecRestoreArrayPair()`
@*/
static inline PetscErrorCode VecGetArrayPair(Vec x, Vec y, PetscScalar **xv, PetscScalar **yv)
{
  PetscFunctionBegin;
  PetscCall(VecGetArray(y, yv));
  if (x == y) *xv = *yv;
  else PetscCall(VecGetArrayRead(x, (const PetscScalar **)xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrayPair - Returns a pair of pointers for two vectors that may be common obtained with `VecGetArrayPair()`

   Logically Collective; No Fortran Support

   Input Parameters:
+  x - the vector
.  y - the second vector
.  xv - location to put pointer to the first array
-  yv - location to put pointer to the second array

   Level: developer

.seealso: [](chapter_vectors), `VecGetArray()`, `VecGetArrayRead()`, `VecGetArrayPair()`
@*/
static inline PetscErrorCode VecRestoreArrayPair(Vec x, Vec y, PetscScalar **xv, PetscScalar **yv)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreArray(y, yv));
  if (x != y) PetscCall(VecRestoreArrayRead(x, (const PetscScalar **)xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_DEBUG)
PETSC_EXTERN PetscErrorCode  VecLockReadPush(Vec);
PETSC_EXTERN PetscErrorCode  VecLockReadPop(Vec);
PETSC_EXTERN PetscErrorCode  VecLockWriteSet(Vec, PetscBool);
PETSC_EXTERN PetscErrorCode  VecLockGet(Vec, PetscInt *);
PETSC_EXTERN PetscErrorCode  VecLockGetLocation(Vec, const char *[], const char *[], int *);
static inline PetscErrorCode VecSetErrorIfLocked(Vec x, PetscInt arg)
{
  PetscInt state;

  PetscFunctionBegin;
  PetscCall(VecLockGet(x, &state));
  if (PetscUnlikely(state != 0)) {
    const char *file, *func, *name;
    int         line;

    PetscCall(VecLockGetLocation(x, &file, &func, &line));
    PetscCall(PetscObjectGetName((PetscObject)x, &name));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector '%s' (argument #%" PetscInt_FMT ") was locked for %s access in %s() at %s:%d (line numbers only accurate to function begin)", name, arg, state > 0 ? "read-only" : "write-only", func ? func : "unknown_function", file ? file : "unknown file", line);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* The three are deprecated */
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use VecLockReadPush() (since version 3.11)") PetscErrorCode VecLockPush(Vec);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use VecLockReadPop() (since version 3.11)") PetscErrorCode VecLockPop(Vec);
  #define VecLocked(x, arg) VecSetErrorIfLocked(x, arg) PETSC_DEPRECATED_MACRO("GCC warning \"Use VecSetErrorIfLocked() (since version 3.11)\"")
#else
  #define VecLockReadPush(x)          PETSC_SUCCESS
  #define VecLockReadPop(x)           PETSC_SUCCESS
  #define VecLockGet(x, s)            (*(s) = 0, PETSC_SUCCESS)
  #define VecSetErrorIfLocked(x, arg) PETSC_SUCCESS
  #define VecLockWriteSet(x, flg)     PETSC_SUCCESS
  /* The three are deprecated */
  #define VecLockPush(x)              PETSC_SUCCESS
  #define VecLockPop(x)               PETSC_SUCCESS
  #define VecLocked(x, arg)           PETSC_SUCCESS
#endif

/*E
  VecOperation - Enumeration of overide-able methods in the `Vec` implementation function-table.

  Values:
+ `VECOP_DUPLICATE`  - `VecDuplicate()`
. `VECOP_SET`        - `VecSet()`
. `VECOP_VIEW`       - `VecView()`
. `VECOP_LOAD`       - `VecLoad()`
. `VECOP_VIEWNATIVE` - `VecViewNative()`
- `VECOP_LOADNATIVE` - `VecLoadNative()`

  Level: advanced

  Notes:
  Some operations may serve as the implementation for other routines not listed above. For
  example `VECOP_SET` can be used to simultaneously overriding the implementation used in
  `VecSet()`, `VecSetInf()`, and `VecZeroEntries()`.

  Entries to `VecOperation` are added as needed so if you do not see the operation listed which
  you'd like to replace, please send mail to `petsc-maint@mcs.anl.gov`!

.seealso: [](chapter_vectors), `Vec`, `VecSetOperation()`
E*/
typedef enum {
  VECOP_DUPLICATE  = 0,
  VECOP_SET        = 10,
  VECOP_VIEW       = 33,
  VECOP_LOAD       = 41,
  VECOP_VIEWNATIVE = 68,
  VECOP_LOADNATIVE = 69
} VecOperation;
PETSC_EXTERN PetscErrorCode VecSetOperation(Vec, VecOperation, void (*)(void));

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
PETSC_EXTERN PetscErrorCode VecMPISetGhost(Vec, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode VecCreateGhost(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], Vec *);
PETSC_EXTERN PetscErrorCode VecCreateGhostWithArray(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscScalar[], Vec *);
PETSC_EXTERN PetscErrorCode VecCreateGhostBlock(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], Vec *);
PETSC_EXTERN PetscErrorCode VecCreateGhostBlockWithArray(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscScalar[], Vec *);
PETSC_EXTERN PetscErrorCode VecGhostGetLocalForm(Vec, Vec *);
PETSC_EXTERN PetscErrorCode VecGhostRestoreLocalForm(Vec, Vec *);
PETSC_EXTERN PetscErrorCode VecGhostIsLocalForm(Vec, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode VecGhostUpdateBegin(Vec, InsertMode, ScatterMode);
PETSC_EXTERN PetscErrorCode VecGhostUpdateEnd(Vec, InsertMode, ScatterMode);

PETSC_EXTERN PetscErrorCode VecConjugate(Vec);
PETSC_EXTERN PetscErrorCode VecImaginaryPart(Vec);
PETSC_EXTERN PetscErrorCode VecRealPart(Vec);

PETSC_EXTERN PetscErrorCode VecScatterCreateToAll(Vec, VecScatter *, Vec *);
PETSC_EXTERN PetscErrorCode VecScatterCreateToZero(Vec, VecScatter *, Vec *);

PETSC_EXTERN PetscErrorCode ISComplementVec(IS, Vec, IS *);
PETSC_EXTERN PetscErrorCode VecPow(Vec, PetscScalar);
PETSC_EXTERN PetscErrorCode VecMedian(Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecWhichInactive(Vec, Vec, Vec, Vec, PetscBool, IS *);
PETSC_EXTERN PetscErrorCode VecWhichBetween(Vec, Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode VecWhichBetweenOrEqual(Vec, Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode VecWhichGreaterThan(Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode VecWhichLessThan(Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode VecWhichEqual(Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode VecISAXPY(Vec, IS, PetscScalar, Vec);
PETSC_EXTERN PetscErrorCode VecISCopy(Vec, IS, ScatterMode, Vec);
PETSC_EXTERN PetscErrorCode VecISSet(Vec, IS, PetscScalar);
PETSC_EXTERN PetscErrorCode VecBoundGradientProjection(Vec, Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode VecStepBoundInfo(Vec, Vec, Vec, Vec, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode VecStepMax(Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode VecStepMaxBounded(Vec, Vec, Vec, Vec, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscViewerMathematicaGetVector(PetscViewer, Vec);
PETSC_EXTERN PetscErrorCode PetscViewerMathematicaPutVector(PetscViewer, Vec);

/*S
     Vecs - Collection of vectors where the data for the vectors is stored in
            one contiguous memory

   Level: advanced

   Notes:
    Temporary construct for handling multiply right hand side solves

    This is faked by storing a single vector that has enough array space for
    n vectors

S*/
struct _n_Vecs {
  PetscInt n;
  Vec      v;
};
typedef struct _n_Vecs     *Vecs;
PETSC_EXTERN PetscErrorCode VecsDestroy(Vecs);
PETSC_EXTERN PetscErrorCode VecsCreateSeq(MPI_Comm, PetscInt, PetscInt, Vecs *);
PETSC_EXTERN PetscErrorCode VecsCreateSeqWithArray(MPI_Comm, PetscInt, PetscInt, PetscScalar *, Vecs *);
PETSC_EXTERN PetscErrorCode VecsDuplicate(Vecs, Vecs *);

#if defined(PETSC_HAVE_VIENNACL)
typedef struct _p_PetscViennaCLIndices *PetscViennaCLIndices;
PETSC_EXTERN PetscErrorCode             PetscViennaCLIndicesCreate(PetscInt, PetscInt *, PetscInt, PetscInt *, PetscViennaCLIndices *);
PETSC_EXTERN PetscErrorCode             PetscViennaCLIndicesDestroy(PetscViennaCLIndices *);
PETSC_EXTERN PetscErrorCode             VecViennaCLCopyToGPUSome_Public(Vec, PetscViennaCLIndices);
PETSC_EXTERN PetscErrorCode             VecViennaCLCopyFromGPUSome_Public(Vec, PetscViennaCLIndices);
PETSC_EXTERN PetscErrorCode             VecCreateSeqViennaCL(MPI_Comm, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode             VecCreateMPIViennaCL(MPI_Comm, PetscInt, PetscInt, Vec *);
#endif
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
PETSC_EXTERN PetscErrorCode VecScatterInitializeForGPU(VecScatter, Vec);
PETSC_EXTERN PetscErrorCode VecScatterFinalizeForGPU(VecScatter);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode VecCreateSeqKokkos(MPI_Comm, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateSeqKokkosWithArray(MPI_Comm, PetscInt, PetscInt, const PetscScalar *, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateMPIKokkos(MPI_Comm, PetscInt, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateMPIKokkosWithArray(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscScalar *, Vec *);
#endif

PETSC_EXTERN PetscErrorCode VecNestGetSubVecs(Vec, PetscInt *, Vec **);
PETSC_EXTERN PetscErrorCode VecNestGetSubVec(Vec, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode VecNestSetSubVecs(Vec, PetscInt, PetscInt *, Vec *);
PETSC_EXTERN PetscErrorCode VecNestSetSubVec(Vec, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode VecCreateNest(MPI_Comm, PetscInt, IS *, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode VecNestGetSize(Vec, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscOptionsGetVec(PetscOptions, const char[], const char[], Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode VecChop(Vec, PetscReal);

PETSC_EXTERN PetscErrorCode VecGetLayout(Vec, PetscLayout *);
PETSC_EXTERN PetscErrorCode VecSetLayout(Vec, PetscLayout);

PETSC_EXTERN PetscErrorCode PetscSectionVecView(PetscSection, Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecGetValuesSection(Vec, PetscSection, PetscInt, PetscScalar **);
PETSC_EXTERN PetscErrorCode VecSetValuesSection(Vec, PetscSection, PetscInt, PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode PetscSectionVecNorm(PetscSection, PetscSection, Vec, NormType, PetscReal[]);

/*S
  VecTagger - Object used to manage the tagging of a subset of indices based on the values of a vector.  The
              motivating application is the selection of cells for refinement or coarsening based on vector containing
              the values in an error indicator metric.

  Values:
+  `VECTAGGERABSOLUTE` - "absolute" values are in a interval (box for complex values) of explicitly defined values
.  `VECTAGGERRELATIVE` - "relative" values are in a interval (box for complex values) of values relative to the set of all values in the vector
.  `VECTAGGERCDF` - "cdf" values are in a relative range of the *cumulative distribution* of values in the vector
.  `VECTAGGEROR` - "or" values are in the union of other tags
-  `VECTAGGERAND` - "and" values are in the intersection of other tags

  Level: advanced

  Developer Note:
  Why not use a `DMLabel` or similar object

.seealso: [](chapter_vectors), `Vec`, `VecTaggerType`, `VecTaggerCreate()`
S*/
typedef struct _p_VecTagger *VecTagger;

/*J
  VecTaggerType - String with the name of a `VecTagger` type

  Level: advanced

.seealso: [](chapter_vectors), `Vec`, `VecTagger`, `VecTaggerCreate()`
J*/
typedef const char *VecTaggerType;
#define VECTAGGERABSOLUTE "absolute"
#define VECTAGGERRELATIVE "relative"
#define VECTAGGERCDF      "cdf"
#define VECTAGGEROR       "or"
#define VECTAGGERAND      "and"

PETSC_EXTERN PetscClassId      VEC_TAGGER_CLASSID;
PETSC_EXTERN PetscFunctionList VecTaggerList;
PETSC_EXTERN PetscErrorCode    VecTaggerRegister(const char[], PetscErrorCode (*)(VecTagger));

PETSC_EXTERN PetscErrorCode VecTaggerCreate(MPI_Comm, VecTagger *);
PETSC_EXTERN PetscErrorCode VecTaggerSetBlockSize(VecTagger, PetscInt);
PETSC_EXTERN PetscErrorCode VecTaggerGetBlockSize(VecTagger, PetscInt *);
PETSC_EXTERN PetscErrorCode VecTaggerSetType(VecTagger, VecTaggerType);
PETSC_EXTERN PetscErrorCode VecTaggerGetType(VecTagger, VecTaggerType *);
PETSC_EXTERN PetscErrorCode VecTaggerSetInvert(VecTagger, PetscBool);
PETSC_EXTERN PetscErrorCode VecTaggerGetInvert(VecTagger, PetscBool *);
PETSC_EXTERN PetscErrorCode VecTaggerSetFromOptions(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerSetUp(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerView(VecTagger, PetscViewer);
PETSC_EXTERN PetscErrorCode VecTaggerComputeIS(VecTagger, Vec, IS *, PetscBool *);
PETSC_EXTERN PetscErrorCode VecTaggerDestroy(VecTagger *);

/*S
   VecTaggerBox - A interval (box for complex numbers) range used to tag values.  For real scalars, this is just a closed interval; for complex scalars,
   the box is the closed region in the complex plane such that real(min) <= real(z) <= real(max) and imag(min) <= imag(z) <= imag(max).  `INF` is an acceptable endpoint.

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecTagger`, `VecTaggerType`, `VecTaggerCreate()`, `VecTaggerComputeIntervals()`
S*/
typedef struct {
  PetscScalar min;
  PetscScalar max;
} VecTaggerBox;
PETSC_EXTERN PetscErrorCode VecTaggerComputeBoxes(VecTagger, Vec, PetscInt *, VecTaggerBox **, PetscBool *);

PETSC_EXTERN PetscErrorCode VecTaggerAbsoluteSetBox(VecTagger, VecTaggerBox *);
PETSC_EXTERN PetscErrorCode VecTaggerAbsoluteGetBox(VecTagger, const VecTaggerBox **);

PETSC_EXTERN PetscErrorCode VecTaggerRelativeSetBox(VecTagger, VecTaggerBox *);
PETSC_EXTERN PetscErrorCode VecTaggerRelativeGetBox(VecTagger, const VecTaggerBox **);

PETSC_EXTERN PetscErrorCode VecTaggerCDFSetBox(VecTagger, VecTaggerBox *);
PETSC_EXTERN PetscErrorCode VecTaggerCDFGetBox(VecTagger, const VecTaggerBox **);

/*E
  VecTaggerCDFMethod - Determines what method is used to compute absolute values from cumulative distribution values (e.g., what value is the preimage of .95 in the cdf).

   Values:
+  `VECTAGGER_CDF_GATHER` - gather results to rank 0, perform the computation and broadcast the result
-  `VECTAGGER_CDF_ITERATIVE` - compute the results on all ranks iteratively using `MPI_Allreduce()`

  Level: advanced

  Note:
  Relevant only in parallel: in serial it is directly computed.

.seealso: [](chapter_vectors), `Vec`, `VecTagger`, `VecTaggerType`, `VecTaggerCreate()`, `VecTaggerCDFSetMethod()`, `VecTaggerCDFMethods`
E*/
typedef enum {
  VECTAGGER_CDF_GATHER,
  VECTAGGER_CDF_ITERATIVE,
  VECTAGGER_CDF_NUM_METHODS
} VecTaggerCDFMethod;
PETSC_EXTERN const char *const VecTaggerCDFMethods[];

PETSC_EXTERN PetscErrorCode VecTaggerCDFSetMethod(VecTagger, VecTaggerCDFMethod);
PETSC_EXTERN PetscErrorCode VecTaggerCDFGetMethod(VecTagger, VecTaggerCDFMethod *);
PETSC_EXTERN PetscErrorCode VecTaggerCDFIterativeSetTolerances(VecTagger, PetscInt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode VecTaggerCDFIterativeGetTolerances(VecTagger, PetscInt *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode VecTaggerOrSetSubs(VecTagger, PetscInt, VecTagger *, PetscCopyMode);
PETSC_EXTERN PetscErrorCode VecTaggerOrGetSubs(VecTagger, PetscInt *, VecTagger **);

PETSC_EXTERN PetscErrorCode VecTaggerAndSetSubs(VecTagger, PetscInt, VecTagger *, PetscCopyMode);
PETSC_EXTERN PetscErrorCode VecTaggerAndGetSubs(VecTagger, PetscInt *, VecTagger **);

PETSC_EXTERN PetscErrorCode VecTaggerInitializePackage(void);
PETSC_EXTERN PetscErrorCode VecTaggerFinalizePackage(void);

#if PetscDefined(USE_DEBUG)
/* This is an internal debug-only routine that should not be used by users */
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode VecValidValues_Internal(Vec, PetscInt, PetscBool);
#else
  #define VecValidValues_Internal(...) PETSC_SUCCESS
#endif /* PETSC_USE_DEBUG */

#define VEC_CUPM_NOT_CONFIGURED(impl) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Must configure PETSc with " PetscStringize(impl) " support to use %s", PETSC_FUNCTION_NAME)

#if PetscDefined(HAVE_CUDA)
  #define VEC_CUDA_DECL_OR_STUB(__decl__, ...) PETSC_EXTERN __decl__;
#else
  #define VEC_CUDA_DECL_OR_STUB(__decl__, ...) \
    static inline __decl__ \
    { \
      __VA_ARGS__; \
      VEC_CUPM_NOT_CONFIGURED(cuda); \
    }
#endif /* PETSC_HAVE_CUDA */

/* extra underscore here to make it line up with the cuda versions */
#if PetscDefined(HAVE_HIP)
  #define VEC_HIP__DECL_OR_STUB(__decl__, ...) PETSC_EXTERN __decl__;
#else
  #define VEC_HIP__DECL_OR_STUB(__decl__, ...) \
    static inline __decl__ \
    { \
      __VA_ARGS__; \
      VEC_CUPM_NOT_CONFIGURED(hip); \
    }
#endif /* PETSC_HAVE_HIP */

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCreateSeqCUDA(MPI_Comm a, PetscInt b, Vec *c), (void)a, (void)b, (void)c)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecCreateSeqHIP(MPI_Comm a, PetscInt b, Vec *c), (void)a, (void)b, (void)c)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCreateSeqCUDAWithArray(MPI_Comm a, PetscInt b, PetscInt c, const PetscScalar *d, Vec *e), (void)a, (void)b, (void)c, (void)d, (void)e)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecCreateSeqHIPWithArray(MPI_Comm a, PetscInt b, PetscInt c, const PetscScalar *d, Vec *e), (void)a, (void)b, (void)c, (void)d, (void)e)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCreateSeqCUDAWithArrays(MPI_Comm a, PetscInt b, PetscInt c, const PetscScalar *d, const PetscScalar *e, Vec *f), (void)a, (void)b, (void)c, (void)d, (void)e, (void)f)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecCreateSeqHIPWithArrays(MPI_Comm a, PetscInt b, PetscInt c, const PetscScalar *d, const PetscScalar *e, Vec *f), (void)a, (void)b, (void)c, (void)d, (void)e, (void)f)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCreateMPICUDA(MPI_Comm a, PetscInt b, PetscInt c, Vec *d), (void)a, (void)b, (void)c, (void)d)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecCreateMPIHIP(MPI_Comm a, PetscInt b, PetscInt c, Vec *d), (void)a, (void)b, (void)c, (void)d)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCreateMPICUDAWithArray(MPI_Comm a, PetscInt b, PetscInt c, PetscInt d, const PetscScalar *e, Vec *f), (void)a, (void)b, (void)c, (void)d, (void)e, (void)f)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecCreateMPIHIPWithArray(MPI_Comm a, PetscInt b, PetscInt c, PetscInt d, const PetscScalar *e, Vec *f), (void)a, (void)b, (void)c, (void)d, (void)e, (void)f)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCreateMPICUDAWithArrays(MPI_Comm a, PetscInt b, PetscInt c, PetscInt d, const PetscScalar *e, const PetscScalar *f, Vec *g), (void)a, (void)b, (void)c, (void)d, (void)e, (void)f, (void)g)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecCreateMPIHIPWithArrays(MPI_Comm a, PetscInt b, PetscInt c, PetscInt d, const PetscScalar *e, const PetscScalar *f, Vec *g), (void)a, (void)b, (void)c, (void)d, (void)e, (void)f, (void)g)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDAGetArray(Vec a, PetscScalar **b), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPGetArray(Vec a, PetscScalar **b), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDARestoreArray(Vec a, PetscScalar **b), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPRestoreArray(Vec a, PetscScalar **b), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDAGetArrayRead(Vec a, const PetscScalar **b), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPGetArrayRead(Vec a, const PetscScalar **b), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDARestoreArrayRead(Vec a, const PetscScalar **b), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPRestoreArrayRead(Vec a, const PetscScalar **b), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDAGetArrayWrite(Vec a, PetscScalar **b), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPGetArrayWrite(Vec a, PetscScalar **b), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDARestoreArrayWrite(Vec a, PetscScalar **b), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPRestoreArrayWrite(Vec a, PetscScalar **b), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDAPlaceArray(Vec a, const PetscScalar b[]), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPPlaceArray(Vec a, const PetscScalar b[]), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDAReplaceArray(Vec a, const PetscScalar b[]), (void)a, (void)b)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPReplaceArray(Vec a, const PetscScalar b[]), (void)a, (void)b)

VEC_CUDA_DECL_OR_STUB(PetscErrorCode VecCUDAResetArray(Vec a), (void)a)
VEC_HIP__DECL_OR_STUB(PetscErrorCode VecHIPResetArray(Vec a), (void)a)

#undef VEC_CUPM_NOT_CONFIGURED
#undef VEC_CUDA_DECL_OR_STUB
#undef VEC_HIP__DECL_OR_STUB

#endif
