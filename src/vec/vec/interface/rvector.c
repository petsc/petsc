/*
     Provides the interface functions for vector operations that have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include "petsc/private/sfimpl.h"
#include "petscsystypes.h"
#include <petsc/private/vecimpl.h> /*I  "petscvec.h"   I*/

PetscInt VecGetSubVectorSavedStateId = -1;

#if PetscDefined(USE_DEBUG)
// this is a no-op '0' macro in optimized builds
PetscErrorCode VecValidValues_Internal(Vec vec, PetscInt argnum, PetscBool begin)
{
  PetscFunctionBegin;
  if (vec->petscnative || vec->ops->getarray) {
    PetscInt           n;
    const PetscScalar *x;
    PetscOffloadMask   mask;

    PetscCall(VecGetOffloadMask(vec, &mask));
    if (!PetscOffloadHost(mask)) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(VecGetLocalSize(vec, &n));
    PetscCall(VecGetArrayRead(vec, &x));
    for (PetscInt i = 0; i < n; i++) {
      if (begin) {
        PetscCheck(!PetscIsInfOrNanScalar(x[i]), PETSC_COMM_SELF, PETSC_ERR_FP, "Vec entry at local location %" PetscInt_FMT " is not-a-number or infinite at beginning of function: Parameter number %" PetscInt_FMT, i, argnum);
      } else {
        PetscCheck(!PetscIsInfOrNanScalar(x[i]), PETSC_COMM_SELF, PETSC_ERR_FP, "Vec entry at local location %" PetscInt_FMT " is not-a-number or infinite at end of function: Parameter number %" PetscInt_FMT, i, argnum);
      }
    }
    PetscCall(VecRestoreArrayRead(vec, &x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
   VecMaxPointwiseDivide - Computes the maximum of the componentwise division `max = max_i abs(x[i]/y[i])`.

   Logically Collective

   Input Parameters:
+  x - the numerators
-  y - the denominators

   Output Parameter:
.  max - the result

   Level: advanced

   Notes:
   `x` and `y` may be the same vector

  if a particular `y[i]` is zero, it is treated as 1 in the above formula

.seealso: [](chapter_vectors), `Vec`, `VecPointwiseDivide()`, `VecPointwiseMult()`, `VecPointwiseMax()`, `VecPointwiseMin()`, `VecPointwiseMaxAbs()`
@*/
PetscErrorCode VecMaxPointwiseDivide(Vec x, Vec y, PetscReal *max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidRealPointer(max, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscUseTypeMethod(x, maxpointwisedivide, y, max);
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecDot - Computes the vector dot product.

   Collective

   Input Parameters:
+  x - first vector
-  y - second vector

   Output Parameter:
.  val - the dot product

   Performance Issues:
.vb
    per-processor memory bandwidth
    interprocessor latency
    work load imbalance that causes certain processes to arrive much earlier than others
.ve

   Level: intermediate

   Notes for Users of Complex Numbers:
   For complex vectors, `VecDot()` computes
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y. Note that this corresponds to the usual "mathematicians" complex
   inner product where the SECOND argument gets the complex conjugate. Since the `BLASdot()` complex conjugates the first
   first argument we call the `BLASdot()` with the arguments reversed.

   Use `VecTDot()` for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

.seealso: [](chapter_vectors), `Vec`, `VecMDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`, `VecDotRealPart()`
@*/
PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidScalarPointer(val, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_Dot, x, y, 0, 0));
  PetscUseTypeMethod(x, dot, y, val);
  PetscCall(PetscLogEventEnd(VEC_Dot, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecDotRealPart - Computes the real part of the vector dot product.

   Collective

   Input Parameters:
+  x - first vector
-  y - second vector

   Output Parameter:
.  val - the real part of the dot product;

   Level: intermediate

   Performance Issues:
.vb
    per-processor memory bandwidth
    interprocessor latency
    work load imbalance that causes certain processes to arrive much earlier than others
.ve

   Notes for Users of Complex Numbers:
     See `VecDot()` for more details on the definition of the dot product for complex numbers

     For real numbers this returns the same value as `VecDot()`

     For complex numbers in C^n (that is a vector of n components with a complex number for each component) this is equal to the usual real dot product on the
     the space R^{2n} (that is a vector of 2n components with the real or imaginary part of the complex numbers for components)

   Developer Note:
    This is not currently optimized to compute only the real part of the dot product.

.seealso: [](chapter_vectors), `Vec`, `VecMDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`, `VecDot()`, `VecDotNorm2()`
@*/
PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val)
{
  PetscScalar fdot;

  PetscFunctionBegin;
  PetscCall(VecDot(x, y, &fdot));
  *val = PetscRealPart(fdot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecNorm  - Computes the vector norm.

   Collective

   Input Parameters:
+  x - the vector
-  type - the type of the norm requested

   Output Parameter:
.  val - the norm

   Values of NormType:
+     `NORM_1` - sum_i |x[i]|
.     `NORM_2` - sqrt(sum_i |x[i]|^2)
.     `NORM_INFINITY` - max_i |x[i]|
-     `NORM_1_AND_2` - computes efficiently both  `NORM_1` and `NORM_2` and stores them each in an output array

    Level: intermediate

   Notes:
      For complex numbers `NORM_1` will return the traditional 1 norm of the 2 norm of the complex numbers; that is the 1
      norm of the absolute values of the complex entries. In PETSc 3.6 and earlier releases it returned the 1 norm of
      the 1 norm of the complex entries (what is returned by the BLAS routine asum()). Both are valid norms but most
      people expect the former.

      This routine stashes the computed norm value, repeated calls before the vector entries are changed are then rapid since the
      precomputed value is immediately available. Certain vector operations such as `VecSet()` store the norms so the value is
      immediately available and does not need to be explicitly computed. `VecScale()` updates any stashed norm values, thus calls after `VecScale()`
      do not need to explicitly recompute the norm.

   Performance Issues:
+    per-processor memory bandwidth - limits the speed of the computation of local portion of the norm
.    interprocessor latency - limits the accumulation of the result across ranks, .i.e. MPI_Allreduce() time
.    number of ranks - the time for the result will grow with the log base 2 of the number of ranks sharing the vector
-    work load imbalance - the rank with the largest number of vector entries will limit the speed up

.seealso: [](chapter_vectors), `Vec`, `NormType`, `VecDot()`, `VecTDot()`, `VecDotBegin()`, `VecDotEnd()`, `VecNormAvailable()`,
          `VecNormBegin()`, `VecNormEnd()`, `NormType()`
@*/
PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val)
{
  PetscBool flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveEnum(x, type, 2);
  PetscValidRealPointer(val, 3);

  /* Cached data? */
  PetscCall(VecNormAvailable(x, type, &flg, val));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Norm, x, 0, 0, 0));
  PetscUseTypeMethod(x, norm, type, val);
  PetscCall(PetscLogEventEnd(VEC_Norm, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));

  if (type != NORM_1_AND_2) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[type], *val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecNormAvailable  - Returns the vector norm if it is already known. That is, it has been previously computed and cached in the vector

   Not Collective

   Input Parameters:
+  x - the vector
-  type - one of `NORM_1` (sum_i |x[i]|), `NORM_2` sqrt(sum_i (x[i])^2), `NORM_INFINITY` max_i |x[i]|.  Also available
          `NORM_1_AND_2`, which computes both norms and stores them
          in a two element array.

   Output Parameters:
+  available - `PETSC_TRUE` if the val returned is valid
-  val - the norm

   Level: intermediate

   Performance Issues:
.vb
    per-processor memory bandwidth
    interprocessor latency
    work load imbalance that causes certain processes to arrive much earlier than others
.ve

   Developer Note:
   `PETSC_HAVE_SLOW_BLAS_NORM2` will cause a C (loop unrolled) version of the norm to be used, rather
   than the BLAS. This should probably only be used when one is using the FORTRAN BLAS routines
   (as opposed to vendor provided) because the FORTRAN BLAS `NRM2()` routine is very slow.

.seealso: [](chapter_vectors), `Vec`, `VecDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`,
          `VecNormBegin()`, `VecNormEnd()`
@*/
PetscErrorCode VecNormAvailable(Vec x, NormType type, PetscBool *available, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidBoolPointer(available, 3);
  PetscValidRealPointer(val, 4);

  if (type == NORM_1_AND_2) {
    *available = PETSC_FALSE;
  } else {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[type], *val, *available));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecNormalize - Normalizes a vector by its 2-norm.

   Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  val - the vector norm before normalization. May be `NULL` if the value is not needed.

   Level: intermediate

.seealso: [](chapter_vectors), `Vec`, `VecNorm()`, `NORM_2`, `NormType`
@*/
PetscErrorCode VecNormalize(Vec x, PetscReal *val)
{
  PetscReal norm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (val) PetscValidRealPointer(val, 2);
  PetscCall(PetscLogEventBegin(VEC_Normalize, x, 0, 0, 0));
  PetscCall(VecNorm(x, NORM_2, &norm));
  if (norm == 0.0) {
    PetscCall(PetscInfo(x, "Vector of zero norm can not be normalized; Returning only the zero norm\n"));
  } else if (norm != 1.0) {
    PetscCall(VecScale(x, 1.0 / norm));
  }
  PetscCall(PetscLogEventEnd(VEC_Normalize, x, 0, 0, 0));
  if (val) *val = norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecMax - Determines the vector component with maximum real part and its location.

   Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
+  p - the index of `val` (pass `NULL` if you don't want this) in the vector
-  val - the maximum component

   Level: intermediate

 Notes:
   Returns the value `PETSC_MIN_REAL` and negative `p` if the vector is of length 0.

   Returns the smallest index with the maximum value

.seealso: [](chapter_vectors), `Vec`, `VecNorm()`, `VecMin()`
@*/
PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  if (p) PetscValidIntPointer(p, 2);
  PetscValidRealPointer(val, 3);
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Max, x, 0, 0, 0));
  PetscUseTypeMethod(x, max, p, val);
  PetscCall(PetscLogEventEnd(VEC_Max, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecMin - Determines the vector component with minimum real part and its location.

   Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
+  p - the index of `val` (pass `NULL` if you don't want this location) in the vector
-  val - the minimum component

   Level: intermediate

   Notes:
   Returns the value `PETSC_MAX_REAL` and negative `p` if the vector is of length 0.

   This returns the smallest index with the minimum value

.seealso: [](chapter_vectors), `Vec`, `VecMax()`
@*/
PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  if (p) PetscValidIntPointer(p, 2);
  PetscValidRealPointer(val, 3);
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Min, x, 0, 0, 0));
  PetscUseTypeMethod(x, min, p, val);
  PetscCall(PetscLogEventEnd(VEC_Min, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecTDot - Computes an indefinite vector dot product. That is, this
   routine does NOT use the complex conjugate.

   Collective

   Input Parameters:
+  x - first vector
-  y - second vector

   Output Parameter:
.  val - the dot product

   Level: intermediate

   Notes for Users of Complex Numbers:
   For complex vectors, VecTDot() computes the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecDot() for the inner product
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

.seealso: [](chapter_vectors), `Vec`, `VecDot()`, `VecMTDot()`
@*/
PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidScalarPointer(val, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_TDot, x, y, 0, 0));
  PetscUseTypeMethod(x, tdot, y, val);
  PetscCall(PetscLogEventEnd(VEC_TDot, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecScale - Scales a vector.

   Not Collective

   Input Parameters:
+  x - the vector
-  alpha - the scalar

   Level: intermediate

 Note:
   For a vector with n components, `VecScale()` computes  x[i] = alpha * x[i], for i=1,...,n.

.seealso: [](chapter_vectors), `Vec`, `VecSet()`
@*/
PetscErrorCode VecScale(Vec x, PetscScalar alpha)
{
  PetscReal norms[4];
  PetscBool flgs[4];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (alpha == (PetscScalar)1.0) PetscFunctionReturn(PETSC_SUCCESS);

  /* get current stashed norms */
  for (PetscInt i = 0; i < 4; i++) PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[i], norms[i], flgs[i]));

  PetscCall(PetscLogEventBegin(VEC_Scale, x, 0, 0, 0));
  PetscUseTypeMethod(x, scale, alpha);
  PetscCall(PetscLogEventEnd(VEC_Scale, x, 0, 0, 0));

  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  /* put the scaled stashed norms back into the Vec */
  for (PetscInt i = 0; i < 4; i++) {
    if (flgs[i]) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[i], PetscAbsScalar(alpha) * norms[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecSet - Sets all components of a vector to a single scalar value.

   Logically Collective

   Input Parameters:
+  x  - the vector
-  alpha - the scalar

   Level: beginner

   Notes:
   For a vector of dimension n, `VecSet()` sets x[i] = alpha, for i=1,...,n,
   so that all vector entries then equal the identical
   scalar value, `alpha`.  Use the more general routine
   `VecSetValues()` to set different vector entries.

   You CANNOT call this after you have called `VecSetValues()` but before you call
   `VecAssemblyBegin()`

.seealso: [](chapter_vectors), `Vec`, `VecSetValues()`, `VecSetValuesBlocked()`, `VecSetRandom()`
@*/
PetscErrorCode VecSet(Vec x, PetscScalar alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveScalar(x, alpha, 2);
  PetscCall(VecSetErrorIfLocked(x, 1));

  PetscCall(PetscLogEventBegin(VEC_Set, x, 0, 0, 0));
  PetscUseTypeMethod(x, set, alpha);
  PetscCall(PetscLogEventEnd(VEC_Set, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));

  /*  norms can be simply set (if |alpha|*N not too large) */

  {
    PetscReal      val = PetscAbsScalar(alpha);
    const PetscInt N   = x->map->N;

    if (N == 0) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_1], 0.0l));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_INFINITY], 0.0));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_2], 0.0));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_FROBENIUS], 0.0));
    } else if (val > PETSC_MAX_REAL / N) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_INFINITY], val));
    } else {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_1], N * val));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_INFINITY], val));
      val *= PetscSqrtReal((PetscReal)N);
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_2], val));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_FROBENIUS], val));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecAXPY - Computes `y = alpha x + y`.

   Logically Collective

   Input Parameters:
+  alpha - the scalar
.  x - vector scale by `alpha`
-  y - vector accumulated into

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    This routine is optimized for alpha of 0.0, otherwise it calls the BLAS routine
.vb
    VecAXPY(y,alpha,x)                   y = alpha x           +      y
    VecAYPX(y,beta,x)                    y =       x           + beta y
    VecAXPBY(y,alpha,beta,x)             y = alpha x           + beta y
    VecWAXPY(w,alpha,x,y)                w = alpha x           +      y
    VecAXPBYPCZ(w,alpha,beta,gamma,x,y)  z = alpha x           + beta y + gamma z
    VecMAXPY(y,nv,alpha[],x[])           y = sum alpha[i] x[i] +      y
.ve

.seealso: [](chapter_vectors), `Vec`, `VecAYPX()`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPBYPCZ()`, `VecAXPBY()`
@*/
PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 3, y, 1);
  VecCheckSameSize(x, 3, y, 1);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecSetErrorIfLocked(y, 1));
  if (x == y) {
    PetscCall(VecScale(y, alpha + 1.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_AXPY, x, y, 0, 0));
  PetscUseTypeMethod(y, axpy, alpha, x);
  PetscCall(PetscLogEventEnd(VEC_AXPY, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecAYPX - Computes `y = x + beta y`.

   Logically Collective

   Input Parameters:
+  beta - the scalar
.  x - the unscaled vector
-  y - the vector to be scaled

   Output Parameter:
.  y - output vector

   Level: intermediate

   Developer Note:
    The implementation is optimized for `beta` of -1.0, 0.0, and 1.0

.seealso: [](chapter_vectors), `Vec`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBYPCZ()`, `VecAXPBY()`
@*/
PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 3, y, 1);
  VecCheckSameSize(x, 1, y, 3);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, beta, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  if (x == y) {
    PetscCall(VecScale(y, beta + 1.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  if (beta == (PetscScalar)0.0) {
    PetscCall(VecCopy(x, y));
  } else {
    PetscCall(PetscLogEventBegin(VEC_AYPX, x, y, 0, 0));
    PetscUseTypeMethod(y, aypx, beta, x);
    PetscCall(PetscLogEventEnd(VEC_AYPX, x, y, 0, 0));
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecAXPBY - Computes `y = alpha x + beta y`.

   Logically Collective

   Input Parameters:
+  alpha - first scalar
.  beta - second scalar
.  x - the first scaled vector
-  y - the second scaled vector

   Output Parameter:
.  y - output vector

   Level: intermediate

   Developer Note:
   The implementation is optimized for `alpha` and/or `beta` values of 0.0 and 1.0

.seealso: [](chapter_vectors), `Vec`, `VecAYPX()`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBYPCZ()`
@*/
PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 4);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 4, y, 1);
  VecCheckSameSize(y, 1, x, 4);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscValidLogicalCollectiveScalar(y, beta, 3);
  if (alpha == (PetscScalar)0.0 && beta == (PetscScalar)1.0) PetscFunctionReturn(PETSC_SUCCESS);
  if (x == y) {
    PetscCall(VecScale(y, alpha + beta));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_AXPY, y, x, 0, 0));
  PetscUseTypeMethod(y, axpby, alpha, beta, x);
  PetscCall(PetscLogEventEnd(VEC_AXPY, y, x, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecAXPBYPCZ - Computes `z = alpha x + beta y + gamma z`

   Logically Collective

   Input Parameters:
+  alpha - first scalar
.  beta - second scalar
.  gamma - third scalar
.  x  - first vector
.  y  - second vector
-  z  - third vector

   Output Parameter:
.  z - output vector

   Level: intermediate

   Note:
   `x`, `y` and `z` must be different vectors

   Developer Note:
    The implementation is optimized for `alpha` of 1.0 and `gamma` of 1.0 or 0.0

.seealso: [](chapter_vectors), `Vec`, `VecAYPX()`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBY()`
@*/
PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(z, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscValidType(z, 1);
  PetscValidType(x, 5);
  PetscValidType(y, 6);
  PetscCheckSameTypeAndComm(x, 5, y, 6);
  PetscCheckSameTypeAndComm(x, 5, z, 1);
  VecCheckSameSize(x, 1, y, 5);
  VecCheckSameSize(x, 1, z, 6);
  PetscCheck(x != y && x != z, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_IDN, "x, y, and z must be different vectors");
  PetscCheck(y != z, PetscObjectComm((PetscObject)y), PETSC_ERR_ARG_IDN, "x, y, and z must be different vectors");
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  VecCheckAssembled(z);
  PetscValidLogicalCollectiveScalar(z, alpha, 2);
  PetscValidLogicalCollectiveScalar(z, beta, 3);
  PetscValidLogicalCollectiveScalar(z, gamma, 4);
  if (alpha == (PetscScalar)0.0 && beta == (PetscScalar)0.0 && gamma == (PetscScalar)1.0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecSetErrorIfLocked(z, 1));
  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_AXPBYPCZ, x, y, z, 0));
  PetscUseTypeMethod(z, axpbypcz, alpha, beta, gamma, x, y);
  PetscCall(PetscLogEventEnd(VEC_AXPBYPCZ, x, y, z, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)z));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecWAXPY - Computes `w = alpha x + y`.

   Logically Collective

   Input Parameters:
+  alpha - the scalar
.  x  - first vector, multiplied by `alpha`
-  y  - second vector

   Output Parameter:
.  w - the result

   Level: intermediate

   Note:
    `w` cannot be either `x` or `y`, but `x` and `y` can be the same

   Developer Note:
    The implementation is optimized for alpha of -1.0, 0.0, and 1.0

.seealso: [](chapter_vectors), `Vec`, `VecAXPY()`, `VecAYPX()`, `VecAXPBY()`, `VecMAXPY()`, `VecAXPBYPCZ()`
@*/
PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidType(w, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 4);
  PetscCheckSameTypeAndComm(x, 3, y, 4);
  PetscCheckSameTypeAndComm(y, 4, w, 1);
  VecCheckSameSize(x, 3, y, 4);
  VecCheckSameSize(x, 3, w, 1);
  PetscCheck(w != y, PETSC_COMM_SELF, PETSC_ERR_SUP, "Result vector w cannot be same as input vector y, suggest VecAXPY()");
  PetscCheck(w != x, PETSC_COMM_SELF, PETSC_ERR_SUP, "Result vector w cannot be same as input vector x, suggest VecAYPX()");
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscCall(VecSetErrorIfLocked(w, 1));

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecCopy(y, w));
  } else {
    PetscCall(PetscLogEventBegin(VEC_WAXPY, x, y, w, 0));
    PetscUseTypeMethod(w, waxpy, alpha, x, y);
    PetscCall(PetscLogEventEnd(VEC_WAXPY, x, y, w, 0));
    PetscCall(PetscObjectStateIncrease((PetscObject)w));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecSetValues - Inserts or adds values into certain locations of a vector.

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either `INSERT_VALUES` to replace the current values or `ADD_VALUES` to add values to any existing entries

   Level: beginner

   Notes:
.vb
   `VecSetValues()` sets x[ix[i]] = y[i], for i=0,...,ni-1.
.ve

   Calls to `VecSetValues()` with the `INSERT_VALUES` and `ADD_VALUES`
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so `VecAssemblyBegin()` and `VecAssemblyEnd()`
   MUST be called after all calls to `VecSetValues()` have been completed.

   VecSetValues() uses 0-based indices in Fortran as well as in C.

   If you call `VecSetOption`(x, `VEC_IGNORE_NEGATIVE_INDICES`,`PETSC_TRUE`),
   negative indices may be passed in ix. These rows are
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

.seealso: [](chapter_vectors), `Vec`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValuesLocal()`,
          `VecSetValue()`, `VecSetValuesBlocked()`, `InsertMode`, `INSERT_VALUES`, `ADD_VALUES`, `VecGetValues()`
@*/
PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  PetscUseTypeMethod(x, setvalues, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetValues - Gets values from certain locations of a vector. Currently
          can only get values on the same processor on which they are owned

    Not Collective

   Input Parameters:
+  x - vector to get values from
.  ni - number of elements to get
-  ix - indices where to get them from (in global 1d numbering)

   Output Parameter:
.   y - array of values

   Level: beginner

   Notes:
   The user provides the allocated array y; it is NOT allocated in this routine

   `VecGetValues()` gets y[i] = x[ix[i]], for i=0,...,ni-1.

   `VecAssemblyBegin()` and `VecAssemblyEnd()`  MUST be called before calling this if `VecSetValues()` or related routine has been called

   VecGetValues() uses 0-based indices in Fortran as well as in C.

   If you call `VecSetOption`(x, `VEC_IGNORE_NEGATIVE_INDICES`,`PETSC_TRUE`),
   negative indices may be passed in ix. These rows are
   simply ignored.

.seealso: [](chapter_vectors), `Vec`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValues()`
@*/
PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[], PetscScalar y[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscUseTypeMethod(x, getvalues, ni, ix, y);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecSetValuesBlocked - Inserts or adds blocks of values into certain locations of a vector.

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, rather than element count
.  y - array of values
-  iora - either `INSERT_VALUES` replaces existing entries with new values, `ADD_VALUES`, adds values to any existing entries

   Level: intermediate

   Notes:
   `VecSetValuesBlocked()` sets x[bs*ix[i]+j] = y[bs*i+j],
   for j=0,...,bs-1, for i=0,...,ni-1. where bs was set with VecSetBlockSize().

   Calls to `VecSetValuesBlocked()` with the `INSERT_VALUES` and `ADD_VALUES`
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so `VecAssemblyBegin()` and `VecAssemblyEnd()`
   MUST be called after all calls to `VecSetValuesBlocked()` have been completed.

   `VecSetValuesBlocked()` uses 0-based indices in Fortran as well as in C.

   Negative indices may be passed in ix, these rows are
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

.seealso: [](chapter_vectors), `Vec`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValuesBlockedLocal()`,
          `VecSetValues()`
@*/
PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  PetscUseTypeMethod(x, setvaluesblocked, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecSetValuesLocal - Inserts or adds values into certain locations of a vector,
   using a local ordering of the nodes.

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either `INSERT_VALUES` replaces existing entries with new values, `ADD_VALUES` adds values to any existing entries

   Level: intermediate

   Notes:
   `VecSetValuesLocal()` sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to `VecSetValues()` with the `INSERT_VALUES` and `ADD_VALUES`
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so `VecAssemblyBegin()` and `VecAssemblyEnd()`
   MUST be called after all calls to `VecSetValuesLocal()` have been completed.

   `VecSetValuesLocal()` uses 0-based indices in Fortran as well as in C.

.seealso: [](chapter_vectors), `Vec`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValues()`, `VecSetLocalToGlobalMapping()`,
          `VecSetValuesBlockedLocal()`
@*/
PetscErrorCode VecSetValuesLocal(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora)
{
  PetscInt lixp[128], *lix = lixp;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  if (!x->ops->setvalueslocal) {
    if (x->map->mapping) {
      if (ni > 128) PetscCall(PetscMalloc1(ni, &lix));
      PetscCall(ISLocalToGlobalMappingApply(x->map->mapping, ni, (PetscInt *)ix, lix));
      PetscUseTypeMethod(x, setvalues, ni, lix, y, iora);
      if (ni > 128) PetscCall(PetscFree(lix));
    } else PetscUseTypeMethod(x, setvalues, ni, ix, y, iora);
  } else PetscUseTypeMethod(x, setvalueslocal, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecSetValuesBlockedLocal - Inserts or adds values into certain locations of a vector,
   using a local ordering of the nodes.

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, not element count
.  y - array of values
-  iora - either `INSERT_VALUES` replaces existing entries with new values, `ADD_VALUES` adds values to any existing entries

   Level: intermediate

   Notes:
   `VecSetValuesBlockedLocal()` sets x[bs*ix[i]+j] = y[bs*i+j],
   for j=0,..bs-1, for i=0,...,ni-1, where bs has been set with `VecSetBlockSize()`.

   Calls to `VecSetValuesBlockedLocal()` with the `INSERT_VALUES` and `ADD_VALUES`
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so `VecAssemblyBegin()` and `VecAssemblyEnd()`
   MUST be called after all calls to `VecSetValuesBlockedLocal()` have been completed.

   `VecSetValuesBlockedLocal()` uses 0-based indices in Fortran as well as in C.

.seealso: [](chapter_vectors), `Vec`, `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValues()`, `VecSetValuesBlocked()`,
          `VecSetLocalToGlobalMapping()`
@*/
PetscErrorCode VecSetValuesBlockedLocal(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora)
{
  PetscInt lixp[128], *lix = lixp;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);
  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  if (x->map->mapping) {
    if (ni > 128) PetscCall(PetscMalloc1(ni, &lix));
    PetscCall(ISLocalToGlobalMappingApplyBlock(x->map->mapping, ni, (PetscInt *)ix, lix));
    PetscUseTypeMethod(x, setvaluesblocked, ni, lix, y, iora);
    if (ni > 128) PetscCall(PetscFree(lix));
  } else {
    PetscUseTypeMethod(x, setvaluesblocked, ni, ix, y, iora);
  }
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMXDot_Private(Vec x, PetscInt nv, const Vec y[], PetscScalar result[], PetscErrorCode (*mxdot)(Vec, PetscInt, const Vec[], PetscScalar[]), PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveInt(x, nv, 2);
  if (!nv) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidPointer(y, 3);
  for (PetscInt i = 0; i < nv; ++i) {
    PetscValidHeaderSpecific(y[i], VEC_CLASSID, 3);
    PetscValidType(y[i], 3);
    PetscCheckSameTypeAndComm(x, 1, y[i], 3);
    VecCheckSameSize(x, 1, y[i], 3);
    VecCheckAssembled(y[i]);
    PetscCall(VecLockReadPush(y[i]));
  }
  PetscValidScalarPointer(result, 4);
  PetscValidFunction(mxdot, 5);

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(event, x, *y, 0, 0));
  PetscCall((*mxdot)(x, nv, y, result));
  PetscCall(PetscLogEventEnd(event, x, *y, 0, 0));
  PetscCall(VecLockReadPop(x));
  for (PetscInt i = 0; i < nv; ++i) PetscCall(VecLockReadPop(y[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecMTDot - Computes indefinite vector multiple dot products.
   That is, it does NOT use the complex conjugate.

   Collective

   Input Parameters:
+  x - one vector
.  nv - number of vectors
-  y - array of vectors.  Note that vectors are pointers

   Output Parameter:
.  val - array of the dot products

   Level: intermediate

   Notes for Users of Complex Numbers:
   For complex vectors, `VecMTDot()` computes the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use `VecMDot()` for the inner product
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

.seealso: [](chapter_vectors), `Vec`, `VecMDot()`, `VecTDot()`
@*/
PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDot_Private(x, nv, y, val, x->ops->mtdot, VEC_MTDot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecMDot - Computes multiple vector dot products.

   Collective

   Input Parameters:
+  x - one vector
.  nv - number of vectors
-  y - array of vectors.

   Output Parameter:
.  val - array of the dot products (does not allocate the array)

   Level: intermediate

   Notes for Users of Complex Numbers:
   For complex vectors, `VecMDot()` computes
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use `VecMTDot()` for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

.seealso: [](chapter_vectors), `Vec`, `VecMTDot()`, `VecDot()`
@*/
PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDot_Private(x, nv, y, val, x->ops->mdot, VEC_MDot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecMAXPY - Computes `y = y + sum alpha[i] x[i]`

   Logically Collective

   Input Parameters:
+  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  y - one vector
-  x - array of vectors

   Level: intermediate

   Note:
    `y` cannot be any of the `x` vectors

.seealso: [](chapter_vectors), `Vec`, `VecAYPX()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBYPCZ()`, `VecAXPBY()`
@*/
PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[], Vec x[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveInt(y, nv, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCheck(nv >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vectors (given %" PetscInt_FMT ") cannot be negative", nv);
  if (nv) {
    PetscInt zeros = 0;

    PetscValidScalarPointer(alpha, 3);
    PetscValidPointer(x, 4);
    for (PetscInt i = 0; i < nv; ++i) {
      PetscValidLogicalCollectiveScalar(y, alpha[i], 3);
      PetscValidHeaderSpecific(x[i], VEC_CLASSID, 4);
      PetscValidType(x[i], 4);
      PetscCheckSameTypeAndComm(y, 1, x[i], 4);
      VecCheckSameSize(y, 1, x[i], 4);
      PetscCheck(y != x[i], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array of vectors 'x' cannot contain y, found x[%" PetscInt_FMT "] == y", i);
      VecCheckAssembled(x[i]);
      PetscCall(VecLockReadPush(x[i]));
      zeros += alpha[i] == (PetscScalar)0.0;
    }

    if (zeros < nv) {
      PetscCall(PetscLogEventBegin(VEC_MAXPY, y, *x, 0, 0));
      PetscUseTypeMethod(y, maxpy, nv, alpha, x);
      PetscCall(PetscLogEventEnd(VEC_MAXPY, y, *x, 0, 0));
      PetscCall(PetscObjectStateIncrease((PetscObject)y));
    }

    for (PetscInt i = 0; i < nv; ++i) PetscCall(VecLockReadPop(x[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecConcatenate - Creates a new vector that is a vertical concatenation of all the given array of vectors
                    in the order they appear in the array. The concatenated vector resides on the same
                    communicator and is the same type as the source vectors.

   Collective

   Input Parameters:
+  nx   - number of vectors to be concatenated
-  X    - array containing the vectors to be concatenated in the order of concatenation

   Output Parameters:
+  Y    - concatenated vector
-  x_is - array of index sets corresponding to the concatenated components of `Y` (pass `NULL` if not needed)

   Level: advanced

   Notes:
   Concatenation is similar to the functionality of a `VECNEST` object; they both represent combination of
   different vector spaces. However, concatenated vectors do not store any information about their
   sub-vectors and own their own data. Consequently, this function provides index sets to enable the
   manipulation of data in the concatenated vector that corresponds to the original components at creation.

   This is a useful tool for outer loop algorithms, particularly constrained optimizers, where the solver
   has to operate on combined vector spaces and cannot utilize `VECNEST` objects due to incompatibility with
   bound projections.

.seealso: [](chapter_vectors), `Vec`, `VECNEST`, `VECSCATTER`, `VecScatterCreate()`
@*/
PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[])
{
  MPI_Comm comm;
  VecType  vec_type;
  Vec      Ytmp, Xtmp;
  IS      *is_tmp;
  PetscInt i, shift = 0, Xnl, Xng, Xbegin;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(*X, nx, 1);
  PetscValidHeaderSpecific(*X, VEC_CLASSID, 2);
  PetscValidType(*X, 2);
  PetscValidPointer(Y, 3);

  if ((*X)->ops->concatenate) {
    /* use the dedicated concatenation function if available */
    PetscCall((*(*X)->ops->concatenate)(nx, X, Y, x_is));
  } else {
    /* loop over vectors and start creating IS */
    comm = PetscObjectComm((PetscObject)(*X));
    PetscCall(VecGetType(*X, &vec_type));
    PetscCall(PetscMalloc1(nx, &is_tmp));
    for (i = 0; i < nx; i++) {
      PetscCall(VecGetSize(X[i], &Xng));
      PetscCall(VecGetLocalSize(X[i], &Xnl));
      PetscCall(VecGetOwnershipRange(X[i], &Xbegin, NULL));
      PetscCall(ISCreateStride(comm, Xnl, shift + Xbegin, 1, &is_tmp[i]));
      shift += Xng;
    }
    /* create the concatenated vector */
    PetscCall(VecCreate(comm, &Ytmp));
    PetscCall(VecSetType(Ytmp, vec_type));
    PetscCall(VecSetSizes(Ytmp, PETSC_DECIDE, shift));
    PetscCall(VecSetUp(Ytmp));
    /* copy data from X array to Y and return */
    for (i = 0; i < nx; i++) {
      PetscCall(VecGetSubVector(Ytmp, is_tmp[i], &Xtmp));
      PetscCall(VecCopy(X[i], Xtmp));
      PetscCall(VecRestoreSubVector(Ytmp, is_tmp[i], &Xtmp));
    }
    *Y = Ytmp;
    if (x_is) {
      *x_is = is_tmp;
    } else {
      for (i = 0; i < nx; i++) PetscCall(ISDestroy(&is_tmp[i]));
      PetscCall(PetscFree(is_tmp));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A helper function for VecGetSubVector to check if we can implement it with no-copy (i.e. the subvector shares
   memory with the original vector), and the block size of the subvector.

    Input Parameters:
+   X - the original vector
-   is - the index set of the subvector

    Output Parameters:
+   contig - PETSC_TRUE if the index set refers to contiguous entries on this process, else PETSC_FALSE
.   start  - start of contiguous block, as an offset from the start of the ownership range of the original vector
-   blocksize - the block size of the subvector

*/
PetscErrorCode VecGetSubVectorContiguityAndBS_Private(Vec X, IS is, PetscBool *contig, PetscInt *start, PetscInt *blocksize)
{
  PetscInt  gstart, gend, lstart;
  PetscBool red[2] = {PETSC_TRUE /*contiguous*/, PETSC_TRUE /*validVBS*/};
  PetscInt  n, N, ibs, vbs, bs = -1;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetSize(is, &N));
  PetscCall(ISGetBlockSize(is, &ibs));
  PetscCall(VecGetBlockSize(X, &vbs));
  PetscCall(VecGetOwnershipRange(X, &gstart, &gend));
  PetscCall(ISContiguousLocal(is, gstart, gend, &lstart, &red[0]));
  /* block size is given by IS if ibs > 1; otherwise, check the vector */
  if (ibs > 1) {
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, red, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)is)));
    bs = ibs;
  } else {
    if (n % vbs || vbs == 1) red[1] = PETSC_FALSE; /* this process invalidate the collectiveness of block size */
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, red, 2, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)is)));
    if (red[0] && red[1]) bs = vbs; /* all processes have a valid block size and the access will be contiguous */
  }

  *contig    = red[0];
  *start     = lstart;
  *blocksize = bs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A helper function for VecGetSubVector, to be used when we have to build a standalone subvector through VecScatter

    Input Parameters:
+   X - the original vector
.   is - the index set of the subvector
-   bs - the block size of the subvector, gotten from VecGetSubVectorContiguityAndBS_Private()

    Output Parameter:
.   Z  - the subvector, which will compose the VecScatter context on output
*/
PetscErrorCode VecGetSubVectorThroughVecScatter_Private(Vec X, IS is, PetscInt bs, Vec *Z)
{
  PetscInt   n, N;
  VecScatter vscat;
  Vec        Y;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetSize(is, &N));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)is), &Y));
  PetscCall(VecSetSizes(Y, n, N));
  PetscCall(VecSetBlockSize(Y, bs));
  PetscCall(VecSetType(Y, ((PetscObject)X)->type_name));
  PetscCall(VecScatterCreate(X, is, Y, NULL, &vscat));
  PetscCall(VecScatterBegin(vscat, X, Y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat, X, Y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(PetscObjectCompose((PetscObject)Y, "VecGetSubVector_Scatter", (PetscObject)vscat));
  PetscCall(VecScatterDestroy(&vscat));
  *Z = Y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecGetSubVector - Gets a vector representing part of another vector

   Collective

   Input Parameters:
+  X - vector from which to extract a subvector
-  is - index set representing portion of `X` to extract

   Output Parameter:
.  Y - subvector corresponding to `is`

   Level: advanced

   Notes:
   The subvector `Y` should be returned with `VecRestoreSubVector()`.
   `X` and must be defined on the same communicator

   This function may return a subvector without making a copy, therefore it is not safe to use the original vector while
   modifying the subvector.  Other non-overlapping subvectors can still be obtained from X using this function.

   The resulting subvector inherits the block size from `is` if greater than one. Otherwise, the block size is guessed from the block size of the original `X`.

.seealso: [](chapter_vectors), `Vec`, `IS`, `VECNEST`, `MatCreateSubMatrix()`
@*/
PetscErrorCode VecGetSubVector(Vec X, IS is, Vec *Y)
{
  Vec Z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscCheckSameComm(X, 1, is, 2);
  PetscValidPointer(Y, 3);
  if (X->ops->getsubvector) {
    PetscUseTypeMethod(X, getsubvector, is, &Z);
  } else { /* Default implementation currently does no caching */
    PetscBool contig;
    PetscInt  n, N, start, bs;

    PetscCall(ISGetLocalSize(is, &n));
    PetscCall(ISGetSize(is, &N));
    PetscCall(VecGetSubVectorContiguityAndBS_Private(X, is, &contig, &start, &bs));
    if (contig) { /* We can do a no-copy implementation */
      const PetscScalar *x;
      PetscInt           state = 0;
      PetscBool          isstd, iscuda, iship;

      PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &isstd, VECSEQ, VECMPI, VECSTANDARD, ""));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iscuda, VECSEQCUDA, VECMPICUDA, ""));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iship, VECSEQHIP, VECMPIHIP, ""));
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        const PetscScalar *x_d;
        PetscMPIInt        size;
        PetscOffloadMask   flg;

        PetscCall(VecCUDAGetArrays_Private(X, &x, &x_d, &flg));
        PetscCheck(flg != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for PETSC_OFFLOAD_UNALLOCATED");
        PetscCheck(!n || x || x_d, PETSC_COMM_SELF, PETSC_ERR_SUP, "Missing vector data");
        if (x) x += start;
        if (x_d) x_d += start;
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)X), &size));
        if (size == 1) {
          PetscCall(VecCreateSeqCUDAWithArrays(PetscObjectComm((PetscObject)X), bs, n, x, x_d, &Z));
        } else {
          PetscCall(VecCreateMPICUDAWithArrays(PetscObjectComm((PetscObject)X), bs, n, N, x, x_d, &Z));
        }
        Z->offloadmask = flg;
#endif
      } else if (iship) {
#if defined(PETSC_HAVE_HIP)
        const PetscScalar *x_d;
        PetscMPIInt        size;
        PetscOffloadMask   flg;

        PetscCall(VecHIPGetArrays_Private(X, &x, &x_d, &flg));
        PetscCheck(flg != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for PETSC_OFFLOAD_UNALLOCATED");
        PetscCheck(!n || x || x_d, PETSC_COMM_SELF, PETSC_ERR_SUP, "Missing vector data");
        if (x) x += start;
        if (x_d) x_d += start;
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)X), &size));
        if (size == 1) {
          PetscCall(VecCreateSeqHIPWithArrays(PetscObjectComm((PetscObject)X), bs, n, x, x_d, &Z));
        } else {
          PetscCall(VecCreateMPIHIPWithArrays(PetscObjectComm((PetscObject)X), bs, n, N, x, x_d, &Z));
        }
        Z->offloadmask = flg;
#endif
      } else if (isstd) {
        PetscMPIInt size;

        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)X), &size));
        PetscCall(VecGetArrayRead(X, &x));
        if (x) x += start;
        if (size == 1) {
          PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)X), bs, n, x, &Z));
        } else {
          PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), bs, n, N, x, &Z));
        }
        PetscCall(VecRestoreArrayRead(X, &x));
      } else { /* default implementation: use place array */
        PetscCall(VecGetArrayRead(X, &x));
        PetscCall(VecCreate(PetscObjectComm((PetscObject)X), &Z));
        PetscCall(VecSetType(Z, ((PetscObject)X)->type_name));
        PetscCall(VecSetSizes(Z, n, N));
        PetscCall(VecSetBlockSize(Z, bs));
        PetscCall(VecPlaceArray(Z, x ? x + start : NULL));
        PetscCall(VecRestoreArrayRead(X, &x));
      }

      /* this is relevant only in debug mode */
      PetscCall(VecLockGet(X, &state));
      if (state) PetscCall(VecLockReadPush(Z));
      Z->ops->placearray   = NULL;
      Z->ops->replacearray = NULL;
    } else { /* Have to create a scatter and do a copy */
      PetscCall(VecGetSubVectorThroughVecScatter_Private(X, is, bs, &Z));
    }
  }
  /* Record the state when the subvector was gotten so we know whether its values need to be put back */
  if (VecGetSubVectorSavedStateId < 0) PetscCall(PetscObjectComposedDataRegister(&VecGetSubVectorSavedStateId));
  PetscCall(PetscObjectComposedDataSetInt((PetscObject)Z, VecGetSubVectorSavedStateId, 1));
  *Y = Z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecRestoreSubVector - Restores a subvector extracted using `VecGetSubVector()`

   Collective

   Input Parameters:
+ X - vector from which subvector was obtained
. is - index set representing the subset of `X`
- Y - subvector being restored

   Level: advanced

.seealso: [](chapter_vectors), `Vec`, `IS`, `VecGetSubVector()`
@*/
PetscErrorCode VecRestoreSubVector(Vec X, IS is, Vec *Y)
{
  PETSC_UNUSED PetscObjectState dummystate = 0;
  PetscBool                     unchanged;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscCheckSameComm(X, 1, is, 2);
  PetscValidPointer(Y, 3);
  PetscValidHeaderSpecific(*Y, VEC_CLASSID, 3);

  if (X->ops->restoresubvector) PetscUseTypeMethod(X, restoresubvector, is, Y);
  else {
    PetscCall(PetscObjectComposedDataGetInt((PetscObject)*Y, VecGetSubVectorSavedStateId, dummystate, unchanged));
    if (!unchanged) { /* If Y's state has not changed since VecGetSubVector(), we only need to destroy Y */
      VecScatter scatter;
      PetscInt   state;

      PetscCall(VecLockGet(X, &state));
      PetscCheck(state == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vec X is locked for read-only or read/write access");

      PetscCall(PetscObjectQuery((PetscObject)*Y, "VecGetSubVector_Scatter", (PetscObject *)&scatter));
      if (scatter) {
        PetscCall(VecScatterBegin(scatter, *Y, X, INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter, *Y, X, INSERT_VALUES, SCATTER_REVERSE));
      } else {
        PetscBool iscuda, iship;
        PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iscuda, VECSEQCUDA, VECMPICUDA, ""));
        PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iship, VECSEQHIP, VECMPIHIP, ""));

        if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
          PetscOffloadMask ymask = (*Y)->offloadmask;

          /* The offloadmask of X dictates where to move memory
              If X GPU data is valid, then move Y data on GPU if needed
              Otherwise, move back to the CPU */
          switch (X->offloadmask) {
          case PETSC_OFFLOAD_BOTH:
            if (ymask == PETSC_OFFLOAD_CPU) {
              PetscCall(VecCUDAResetArray(*Y));
            } else if (ymask == PETSC_OFFLOAD_GPU) {
              X->offloadmask = PETSC_OFFLOAD_GPU;
            }
            break;
          case PETSC_OFFLOAD_GPU:
            if (ymask == PETSC_OFFLOAD_CPU) PetscCall(VecCUDAResetArray(*Y));
            break;
          case PETSC_OFFLOAD_CPU:
            if (ymask == PETSC_OFFLOAD_GPU) PetscCall(VecResetArray(*Y));
            break;
          case PETSC_OFFLOAD_UNALLOCATED:
          case PETSC_OFFLOAD_KOKKOS:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This should not happen");
          }
#endif
        } else if (iship) {
#if defined(PETSC_HAVE_HIP)
          PetscOffloadMask ymask = (*Y)->offloadmask;

          /* The offloadmask of X dictates where to move memory
              If X GPU data is valid, then move Y data on GPU if needed
              Otherwise, move back to the CPU */
          switch (X->offloadmask) {
          case PETSC_OFFLOAD_BOTH:
            if (ymask == PETSC_OFFLOAD_CPU) {
              PetscCall(VecHIPResetArray(*Y));
            } else if (ymask == PETSC_OFFLOAD_GPU) {
              X->offloadmask = PETSC_OFFLOAD_GPU;
            }
            break;
          case PETSC_OFFLOAD_GPU:
            if (ymask == PETSC_OFFLOAD_CPU) PetscCall(VecHIPResetArray(*Y));
            break;
          case PETSC_OFFLOAD_CPU:
            if (ymask == PETSC_OFFLOAD_GPU) PetscCall(VecResetArray(*Y));
            break;
          case PETSC_OFFLOAD_UNALLOCATED:
          case PETSC_OFFLOAD_KOKKOS:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This should not happen");
          }
#endif
        } else {
          /* If OpenCL vecs updated the device memory, this triggers a copy on the CPU */
          PetscCall(VecResetArray(*Y));
        }
        PetscCall(PetscObjectStateIncrease((PetscObject)X));
      }
    }
  }
  PetscCall(VecDestroy(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecCreateLocalVector - Creates a vector object suitable for use with `VecGetLocalVector()` and friends. You must call `VecDestroy()` when the
   vector is no longer needed.

   Not Collective.

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecGetLocalVectorRead()`, `VecRestoreLocalVectorRead()`, `VecGetLocalVector()`, `VecRestoreLocalVector()`
@*/
PetscErrorCode VecCreateLocalVector(Vec v, Vec *w)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidPointer(w, 2);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v), &size));
  if (size == 1) PetscCall(VecDuplicate(v, w));
  else if (v->ops->createlocalvector) PetscUseTypeMethod(v, createlocalvector, w);
  else {
    VecType  type;
    PetscInt n;

    PetscCall(VecCreate(PETSC_COMM_SELF, w));
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(VecSetSizes(*w, n, n));
    PetscCall(VecGetBlockSize(v, &n));
    PetscCall(VecSetBlockSize(*w, n));
    PetscCall(VecGetType(v, &type));
    PetscCall(VecSetType(*w, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecGetLocalVectorRead - Maps the local portion of a vector into a
   vector.

   Not Collective.

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

   Notes:
   You must call `VecRestoreLocalVectorRead()` when the local
   vector is no longer needed.

   This function is similar to `VecGetArrayRead()` which maps the local
   portion into a raw pointer.  `VecGetLocalVectorRead()` is usually
   almost as efficient as `VecGetArrayRead()` but in certain circumstances
   `VecGetLocalVectorRead()` can be much more efficient than
   `VecGetArrayRead()`.  This is because the construction of a contiguous
   array representing the vector data required by `VecGetArrayRead()` can
   be an expensive operation for certain vector types.  For example, for
   GPU vectors `VecGetArrayRead()` requires that the data between device
   and host is synchronized.

   Unlike `VecGetLocalVector()`, this routine is not collective and
   preserves cached information.

.seealso: [](chapter_vectors), `Vec`, `VecCreateLocalVector()`, `VecRestoreLocalVectorRead()`, `VecGetLocalVector()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecGetLocalVectorRead(Vec v, Vec w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  VecCheckSameLocalSize(v, 1, w, 2);
  if (v->ops->getlocalvectorread) {
    PetscUseTypeMethod(v, getlocalvectorread, w);
  } else {
    PetscScalar *a;

    PetscCall(VecGetArrayRead(v, (const PetscScalar **)&a));
    PetscCall(VecPlaceArray(w, a));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscCall(VecLockReadPush(v));
  PetscCall(VecLockReadPush(w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecRestoreLocalVectorRead - Unmaps the local portion of a vector
   previously mapped into a vector using `VecGetLocalVectorRead()`.

   Not Collective.

   Input parameter:
+  v - The local portion of this vector was previously mapped into `w` using `VecGetLocalVectorRead()`.
-  w - The vector into which the local portion of `v` was mapped.

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecCreateLocalVector()`, `VecGetLocalVectorRead()`, `VecGetLocalVector()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecRestoreLocalVectorRead(Vec v, Vec w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  if (v->ops->restorelocalvectorread) {
    PetscUseTypeMethod(v, restorelocalvectorread, w);
  } else {
    const PetscScalar *a;

    PetscCall(VecGetArrayRead(w, &a));
    PetscCall(VecRestoreArrayRead(v, &a));
    PetscCall(VecResetArray(w));
  }
  PetscCall(VecLockReadPop(v));
  PetscCall(VecLockReadPop(w));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecGetLocalVector - Maps the local portion of a vector into a
   vector.

   Collective

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

   Notes:
   You must call `VecRestoreLocalVector()` when the local
   vector is no longer needed.

   This function is similar to `VecGetArray()` which maps the local
   portion into a raw pointer.  `VecGetLocalVector()` is usually about as
   efficient as `VecGetArray()` but in certain circumstances
   `VecGetLocalVector()` can be much more efficient than `VecGetArray()`.
   This is because the construction of a contiguous array representing
   the vector data required by `VecGetArray()` can be an expensive
   operation for certain vector types.  For example, for GPU vectors
   `VecGetArray()` requires that the data between device and host is
   synchronized.

.seealso: [](chapter_vectors), `Vec`, `VecCreateLocalVector()`, `VecRestoreLocalVector()`, `VecGetLocalVectorRead()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecGetLocalVector(Vec v, Vec w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  VecCheckSameLocalSize(v, 1, w, 2);
  if (v->ops->getlocalvector) {
    PetscUseTypeMethod(v, getlocalvector, w);
  } else {
    PetscScalar *a;

    PetscCall(VecGetArray(v, &a));
    PetscCall(VecPlaceArray(w, a));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecRestoreLocalVector - Unmaps the local portion of a vector
   previously mapped into a vector using `VecGetLocalVector()`.

   Logically Collective.

   Input parameter:
+  v - The local portion of this vector was previously mapped into `w` using `VecGetLocalVector()`.
-  w - The vector into which the local portion of `v` was mapped.

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecCreateLocalVector()`, `VecGetLocalVector()`, `VecGetLocalVectorRead()`, `VecRestoreLocalVectorRead()`, `LocalVectorRead()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecRestoreLocalVector(Vec v, Vec w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  if (v->ops->restorelocalvector) {
    PetscUseTypeMethod(v, restorelocalvector, w);
  } else {
    PetscScalar *a;
    PetscCall(VecGetArray(w, &a));
    PetscCall(VecRestoreArray(v, &a));
    PetscCall(VecResetArray(w));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray - Returns a pointer to a contiguous array that contains this
   processor's portion of the vector data. For the standard PETSc
   vectors, `VecGetArray()` returns a pointer to the local data array and
   does not use any copies. If the underlying vector data is not stored
   in a contiguous array this routine will copy the data to a contiguous
   array and return a pointer to that. You MUST call `VecRestoreArray()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

   Fortran Note:
   `VecGetArray()` Fortran binding is deprecated (since PETSc 3.19), use `VecGetArrayF90()`

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
PetscErrorCode VecGetArray(Vec x, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (x->ops->getarray) { /* The if-else order matters! VECNEST, VECCUDA etc should have ops->getarray while VECCUDA etc are petscnative */
    PetscUseTypeMethod(x, getarray, a);
  } else if (x->petscnative) { /* VECSTANDARD */
    *a = *((PetscScalar **)x->data);
  } else SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot get array for vector type \"%s\"", ((PetscObject)x)->type_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray - Restores a vector after `VecGetArray()` has been called and the array is no longer needed

   Logically Collective

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from `VecGetArray()`

   Level: beginner

   Fortran Note:
   `VecRestoreArray()` Fortran binding is deprecated (since PETSc 3.19), use `VecRestoreArrayF90()`

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArrayRead()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecRestoreArrayReadF90()`, `VecPlaceArray()`, `VecRestoreArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->ops->restorearray) {
    PetscUseTypeMethod(x, restorearray, a);
  } else PetscCheck(x->petscnative, PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot restore array for vector type \"%s\"", ((PetscObject)x)->type_name);
  if (a) *a = NULL;
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@C
   VecGetArrayRead - Get read-only pointer to contiguous array containing this processor's portion of the vector data.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - the array

   Level: beginner

   Notes:
   The array must be returned using a matching call to `VecRestoreArrayRead()`.

   Unlike `VecGetArray()`, preserves cached information like vector norms.

   Standard PETSc vectors use contiguous storage so that this routine does not perform a copy.  Other vector
   implementations may require a copy, but such implementations should cache the contiguous representation so that
   only one copy is performed when this routine is called multiple times in sequence.

   Fortran Note:
   `VecGetArrayRead()` Fortran binding is deprecated (since PETSc 3.19), use `VecGetArrayReadF90()`

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayReadF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 2);
  if (x->ops->getarrayread) {
    PetscUseTypeMethod(x, getarrayread, a);
  } else if (x->ops->getarray) {
    PetscObjectState state;

    /* VECNEST, VECCUDA, VECKOKKOS etc */
    // x->ops->getarray may bump the object state, but since we know this is a read-only get
    // we can just undo that
    PetscCall(PetscObjectStateGet((PetscObject)x, &state));
    PetscUseTypeMethod(x, getarray, (PetscScalar **)a);
    PetscCall(PetscObjectStateSet((PetscObject)x, state));
  } else if (x->petscnative) {
    /* VECSTANDARD */
    *a = *((PetscScalar **)x->data);
  } else SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot get array read for vector type \"%s\"", ((PetscObject)x)->type_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrayRead - Restore array obtained with `VecGetArrayRead()`

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

   Fortran Note:
   `VecRestoreArrayRead()` Fortran binding is deprecated (since PETSc 3.19), use `VecRestoreArrayReadF90()`

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayReadF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->petscnative) { /* VECSTANDARD, VECCUDA, VECKOKKOS etc */
    /* nothing */
  } else if (x->ops->restorearrayread) { /* VECNEST */
    PetscUseTypeMethod(x, restorearrayread, a);
  } else { /* No one? */
    PetscObjectState state;

    // x->ops->restorearray may bump the object state, but since we know this is a read-restore
    // we can just undo that
    PetscCall(PetscObjectStateGet((PetscObject)x, &state));
    PetscUseTypeMethod(x, restorearray, (PetscScalar **)a);
    PetscCall(PetscObjectStateSet((PetscObject)x, state));
  }
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArrayWrite - Returns a pointer to a contiguous array that WILL contain this
   processor's portion of the vector data. The values in this array are NOT valid, the caller of this
   routine is responsible for putting values into the array; any values it does not set will be invalid

   Logically Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Level: intermediate

   Note:
   The array must be returned using a matching call to `VecRestoreArrayRead()`.

   For vectors associated with GPUs, the host and device vectors are not synchronized before giving access. If you need correct
   values in the array use `VecGetArray()`

   Fortran Note:
   `VecGetArrayWrite()` Fortran binding is deprecated (since PETSc 3.19), use `VecGetArrayWriteF90()`

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayWriteF90()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArray()`, `VecRestoreArrayWrite()`
@*/
PetscErrorCode VecGetArrayWrite(Vec x, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (x->ops->getarraywrite) {
    PetscUseTypeMethod(x, getarraywrite, a);
  } else {
    PetscCall(VecGetArray(x, a));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrayWrite - Restores a vector after `VecGetArrayWrite()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from `VecGetArray()`

   Level: beginner

   Fortran Note:
   `VecRestoreArrayWrite()` Fortran binding is deprecated (since PETSc 3.19), use `VecRestoreArrayWriteF90()`

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayWriteF90()`, `VecGetArray()`, `VecRestoreArrayRead()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecRestoreArrayReadF90()`, `VecPlaceArray()`, `VecRestoreArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`
@*/
PetscErrorCode VecRestoreArrayWrite(Vec x, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->ops->restorearraywrite) {
    PetscUseTypeMethod(x, restorearraywrite, a);
  } else if (x->ops->restorearray) {
    PetscUseTypeMethod(x, restorearray, a);
  }
  if (a) *a = NULL;
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArrays - Returns a pointer to the arrays in a set of vectors
   that were created by a call to `VecDuplicateVecs()`.

   Logically Collective; No Fortran Support

   Input Parameters:
+  x - the vectors
-  n - the number of vectors

   Output Parameter:
.  a - location to put pointer to the array

   Level: intermediate

   Note:
   You MUST call `VecRestoreArrays()` when you no longer need access to the arrays.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArrays()`
@*/
PetscErrorCode VecGetArrays(const Vec x[], PetscInt n, PetscScalar **a[])
{
  PetscInt      i;
  PetscScalar **q;

  PetscFunctionBegin;
  PetscValidPointer(x, 1);
  PetscValidHeaderSpecific(*x, VEC_CLASSID, 1);
  PetscValidPointer(a, 3);
  PetscCheck(n > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must get at least one array n = %" PetscInt_FMT, n);
  PetscCall(PetscMalloc1(n, &q));
  for (i = 0; i < n; ++i) PetscCall(VecGetArray(x[i], &q[i]));
  *a = q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrays - Restores a group of vectors after `VecGetArrays()`
   has been called.

   Logically Collective; No Fortran Support

   Input Parameters:
+  x - the vector
.  n - the number of vectors
-  a - location of pointer to arrays obtained from `VecGetArrays()`

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the arrays obtained with `VecGetArrays()`.

   Level: intermediate

.seealso: [](chapter_vectors), `Vec`, `VecGetArrays()`, `VecRestoreArray()`
@*/
PetscErrorCode VecRestoreArrays(const Vec x[], PetscInt n, PetscScalar **a[])
{
  PetscInt      i;
  PetscScalar **q = *a;

  PetscFunctionBegin;
  PetscValidPointer(x, 1);
  PetscValidHeaderSpecific(*x, VEC_CLASSID, 1);
  PetscValidPointer(a, 3);

  for (i = 0; i < n; ++i) PetscCall(VecRestoreArray(x[i], &q[i]));
  PetscCall(PetscFree(q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArrayAndMemType - Like `VecGetArray()`, but if this is a standard device vector (e.g., `VECCUDA`), the returned pointer will be a device
   pointer to the device memory that contains this processor's portion of the vector data. Device data is guaranteed to have the latest value.
   Otherwise, when this is a host vector (e.g., `VECMPI`), this routine functions the same as `VecGetArray()` and returns a host pointer.

   For `VECKOKKOS`, if Kokkos is configured without device (e.g., use serial or openmp), per this function, the vector works like `VECSEQ`/`VECMPI`;
   otherwise, it works like `VECCUDA` or `VECHIP` etc.

   Logically Collective; No Fortran Support

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - location to put pointer to the array
-  mtype - memory type of the array

   Level: beginner

   Note:
   Use `VecRestoreArrayAndMemType()` when the array access is no longer needed

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayAndMemType()`, `VecGetArrayReadAndMemType()`, `VecGetArrayWriteAndMemType()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`,
          `VecPlaceArray()`, `VecGetArray2d()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
PetscErrorCode VecGetArrayAndMemType(Vec x, PetscScalar **a, PetscMemType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(a, 2);
  if (mtype) PetscValidPointer(mtype, 3);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (x->ops->getarrayandmemtype) {
    /* VECCUDA, VECKOKKOS etc */
    PetscUseTypeMethod(x, getarrayandmemtype, a, mtype);
  } else {
    /* VECSTANDARD, VECNEST, VECVIENNACL */
    PetscCall(VecGetArray(x, a));
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrayAndMemType - Restores a vector after `VecGetArrayAndMemType()` has been called.

   Logically Collective; No Fortran Support

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from `VecGetArrayAndMemType()`

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayAndMemType()`, `VecGetArray()`, `VecRestoreArrayRead()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecRestoreArrayReadF90()`,
          `VecPlaceArray()`, `VecRestoreArray2d()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayAndMemType(Vec x, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->ops->restorearrayandmemtype) {
    /* VECCUDA, VECKOKKOS etc */
    PetscUseTypeMethod(x, restorearrayandmemtype, a);
  } else {
    /* VECNEST, VECVIENNACL */
    PetscCall(VecRestoreArray(x, a));
  } /* VECSTANDARD does nothing */
  if (a) *a = NULL;
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArrayReadAndMemType - Like `VecGetArrayRead()`, but if the input vector is a device vector, it will return a read-only device pointer.
   The returned pointer is guaranteed to point to up-to-date data. For host vectors, it functions as `VecGetArrayRead()`.

   Not Collective; No Fortran Support

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - the array
-  mtype - memory type of the array

   Level: beginner

   Notes:
   The array must be returned using a matching call to `VecRestoreArrayReadAndMemType()`.

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayReadAndMemType()`, `VecGetArrayAndMemType()`, `VecGetArrayWriteAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayAndMemType()`
@*/
PetscErrorCode VecGetArrayReadAndMemType(Vec x, const PetscScalar **a, PetscMemType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(a, 2);
  if (mtype) PetscValidPointer(mtype, 3);
  if (x->ops->getarrayreadandmemtype) {
    /* VECCUDA/VECHIP though they are also petscnative */
    PetscUseTypeMethod(x, getarrayreadandmemtype, a, mtype);
  } else if (x->ops->getarrayandmemtype) {
    /* VECKOKKOS */
    PetscObjectState state;

    // see VecGetArrayRead() for why
    PetscCall(PetscObjectStateGet((PetscObject)x, &state));
    PetscUseTypeMethod(x, getarrayandmemtype, (PetscScalar **)a, mtype);
    PetscCall(PetscObjectStateSet((PetscObject)x, state));
  } else {
    PetscCall(VecGetArrayRead(x, a));
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrayReadAndMemType - Restore array obtained with `VecGetArrayReadAndMemType()`

   Not Collective; No Fortran Support

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayReadAndMemType()`, `VecRestoreArrayAndMemType()`, `VecRestoreArrayWriteAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayReadAndMemType(Vec x, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->ops->restorearrayreadandmemtype) {
    /* VECCUDA/VECHIP */
    PetscUseTypeMethod(x, restorearrayreadandmemtype, a);
  } else if (!x->petscnative) {
    /* VECNEST */
    PetscCall(VecRestoreArrayRead(x, a));
  }
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArrayWriteAndMemType - Like `VecGetArrayWrite()`, but if this is a device vector it will always return
    a device pointer to the device memory that contains this processor's portion of the vector data.

   Not Collective; No Fortran Support

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - the array
-  mtype - memory type of the array

   Level: beginner

   Note:
   The array must be returned using a matching call to `VecRestoreArrayWriteAndMemType()`, where it will label the device memory as most recent.

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayWriteAndMemType()`, `VecGetArrayReadAndMemType()`, `VecGetArrayAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`,
@*/
PetscErrorCode VecGetArrayWriteAndMemType(Vec x, PetscScalar **a, PetscMemType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscValidPointer(a, 2);
  if (mtype) PetscValidPointer(mtype, 3);
  if (x->ops->getarraywriteandmemtype) {
    /* VECCUDA, VECHIP, VECKOKKOS etc, though they are also petscnative */
    PetscUseTypeMethod(x, getarraywriteandmemtype, a, mtype);
  } else if (x->ops->getarrayandmemtype) {
    PetscCall(VecGetArrayAndMemType(x, a, mtype));
  } else {
    /* VECNEST, VECVIENNACL */
    PetscCall(VecGetArrayWrite(x, a));
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArrayWriteAndMemType - Restore array obtained with `VecGetArrayWriteAndMemType()`

   Not Collective; No Fortran Support

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayWriteAndMemType()`, `VecRestoreArrayAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayWriteAndMemType(Vec x, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (a) PetscValidPointer(a, 2);
  if (x->ops->restorearraywriteandmemtype) {
    /* VECCUDA/VECHIP */
    PetscMemType PETSC_UNUSED mtype; // since this function doesn't accept a memtype?
    PetscUseTypeMethod(x, restorearraywriteandmemtype, a, &mtype);
  } else if (x->ops->restorearrayandmemtype) {
    PetscCall(VecRestoreArrayAndMemType(x, a));
  } else {
    PetscCall(VecRestoreArray(x, a));
  }
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecPlaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.

   Not Collective; No Fortran Support

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: developer

   Notes:
   Use `VecReplaceArray()` instead to permanently replace the array

   You can return to the original array with a call to `VecResetArray()`. `vec` does not take
   ownership of `array` in any way.

   The user must free `array` themselves but be careful not to
   do so before the vector has either been destroyed, had its original array restored with
   `VecResetArray()` or permanently replaced with `VecReplaceArray()`.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecReplaceArray()`, `VecResetArray()`
@*/
PetscErrorCode VecPlaceArray(Vec vec, const PetscScalar array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidType(vec, 1);
  if (array) PetscValidScalarPointer(array, 2);
  PetscUseTypeMethod(vec, placearray, array);
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecReplaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.

   Not Collective; No Fortran Support

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: developer

   Notes:
   This permanently replaces the array and frees the memory associated
   with the old array. Use `VecPlaceArray()` to temporarily replace the array.

   The memory passed in MUST be obtained with `PetscMalloc()` and CANNOT be
   freed by the user. It will be freed when the vector is destroyed.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecPlaceArray()`, `VecResetArray()`
@*/
PetscErrorCode VecReplaceArray(Vec vec, const PetscScalar array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidType(vec, 1);
  PetscUseTypeMethod(vec, replacearray, array);
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
    VecDuplicateVecsF90 - Creates several vectors of the same type as an existing vector
    and makes them accessible via a Fortran pointer.

    Synopsis:
    VecDuplicateVecsF90(Vec x,PetscInt n,{Vec, pointer :: y(:)},integer ierr)

    Collective

    Input Parameters:
+   x - a vector to mimic
-   n - the number of vectors to obtain

    Output Parameters:
+   y - Fortran pointer to the array of vectors
-   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    Vec x
    Vec, pointer :: y(:)
    ....
    call VecDuplicateVecsF90(x,2,y,ierr)
    call VecSet(y(2),alpha,ierr)
    call VecSet(y(2),alpha,ierr)
    ....
    call VecDestroyVecsF90(2,y,ierr)
.ve

    Level: beginner

    Note:
    Use `VecDestroyVecsF90()` to free the space.

.seealso: [](chapter_vectors), `Vec`, `VecDestroyVecsF90()`, `VecDuplicateVecs()`
M*/

/*MC
    VecRestoreArrayF90 - Restores a vector to a usable state after a call to
    `VecGetArrayF90()`.

    Synopsis:
    VecRestoreArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective

    Input Parameters:
+   x - vector
-   xx_v - the Fortran pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    xx_v(3) = a
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrayReadF90()`
M*/

/*MC
    VecDestroyVecsF90 - Frees a block of vectors obtained with `VecDuplicateVecsF90()`.

    Synopsis:
    VecDestroyVecsF90(PetscInt n,{Vec, pointer :: x(:)},PetscErrorCode ierr)

    Collective

    Input Parameters:
+   n - the number of vectors previously obtained
-   x - pointer to array of vector pointers

    Output Parameter:
.   ierr - error code

    Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecDestroyVecs()`, `VecDuplicateVecsF90()`
M*/

/*MC
    VecGetArrayF90 - Accesses a vector array from Fortran. For default PETSc
    vectors, `VecGetArrayF90()` returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call `VecRestoreArrayF90()`
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran pointer to the array
-   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    xx_v(3) = a
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve

     Level: beginner

    Note:
    If you ONLY intend to read entries from the array and not change any entries you should use `VecGetArrayReadF90()`.

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayReadF90()`
M*/

/*MC
    VecGetArrayReadF90 - Accesses a read only array from Fortran. For default PETSc
    vectors, `VecGetArrayF90()` returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call `VecRestoreArrayReadF90()`
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayReadF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran pointer to the array
-   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayReadF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayReadF90(x,xx_v,ierr)
.ve

    Level: beginner

    Note:
    If you intend to write entries into the array you must use `VecGetArrayF90()`.

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArrayReadF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecRestoreArrayRead()`, `VecGetArrayF90()`
M*/

/*MC
    VecRestoreArrayReadF90 - Restores a readonly vector to a usable state after a call to
    `VecGetArrayReadF90()`.

    Synopsis:
    VecRestoreArrayReadF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective

    Input Parameters:
+   x - vector
-   xx_v - the Fortran pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayReadF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayReadF90(x,xx_v,ierr)
.ve

    Level: beginner

.seealso: [](chapter_vectors), `Vec`, `VecGetArrayReadF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecRestoreArrayRead()`, `VecRestoreArrayF90()`
M*/

/*@C
   VecGetArray2d - Returns a pointer to a 2d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray2d()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart and nstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray2d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray2d(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[])
{
  PetscInt     i, N;
  PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 2d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n);
  PetscCall(VecGetArray(x, &aa));

  PetscCall(PetscMalloc1(m, a));
  for (i = 0; i < m; i++) (*a)[i] = aa + i * n - nstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray2dWrite - Returns a pointer to a 2d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call `VecRestoreArray2dWrite()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart and nstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray2d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray2dWrite(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[])
{
  PetscInt     i, N;
  PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 2d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n);
  PetscCall(VecGetArrayWrite(x, &aa));

  PetscCall(PetscMalloc1(m, a));
  for (i = 0; i < m; i++) (*a)[i] = aa + i * n - nstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray2d - Restores a vector after `VecGetArray2d()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of the two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray2d()`

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray2d(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray2dWrite - Restores a vector after VecGetArray2dWrite`()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of the two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray2d()`

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray2dWrite(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray1d - Returns a pointer to a 1d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray1d()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray2d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray1d(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[])
{
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 4);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m == N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local array size %" PetscInt_FMT " does not match 1d array dimensions %" PetscInt_FMT, N, m);
  PetscCall(VecGetArray(x, a));
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray1dWrite - Returns a pointer to a 1d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call `VecRestoreArray1dWrite()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray2d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray1dWrite(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[])
{
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 4);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m == N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local array size %" PetscInt_FMT " does not match 1d array dimensions %" PetscInt_FMT, N, m);
  PetscCall(VecGetArrayWrite(x, a));
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray1d - Restores a vector after `VecGetArray1d()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray1d()`

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray1d()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray2d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray1d(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray1dWrite - Restores a vector after `VecGetArray1dWrite()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray1d()`

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray1d()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray2d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray1dWrite(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray3d - Returns a pointer to a 3d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray3d()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of three dimensional array
.  p - third dimension of three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  pstart - first index in the third coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray3d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray3d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[])
{
  PetscInt     i, N, j;
  PetscScalar *aa, **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 3d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p);
  PetscCall(VecGetArray(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar **) + m * n * sizeof(PetscScalar *), a));
  b = (PetscScalar **)((*a) + m);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = aa + i * n * p + j * p - pstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray3dWrite - Returns a pointer to a 3d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call `VecRestoreArray3dWrite()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of three dimensional array
.  p - third dimension of three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  pstart - first index in the third coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray3d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray3dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[])
{
  PetscInt     i, N, j;
  PetscScalar *aa, **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 3d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p);
  PetscCall(VecGetArrayWrite(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar **) + m * n * sizeof(PetscScalar *), a));
  b = (PetscScalar **)((*a) + m);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = aa + i * n * p + j * p - pstart;

  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray3d - Restores a vector after `VecGetArray3d()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray3d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray3d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray3dWrite - Restores a vector after `VecGetArray3dWrite()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray3d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray3dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray4d - Returns a pointer to a 4d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray4d()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of four dimensional array
.  p - third dimension of four dimensional array
.  q - fourth dimension of four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  qstart - first index in the fourth coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray3d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray4d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[])
{
  PetscInt     i, N, j, k;
  PetscScalar *aa, ***b, **c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p * q == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 4d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p, q);
  PetscCall(VecGetArray(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar ***) + m * n * sizeof(PetscScalar **) + m * n * p * sizeof(PetscScalar *), a));
  b = (PetscScalar ***)((*a) + m);
  c = (PetscScalar **)(b + m * n);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = c + i * n * p + j * p - pstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < p; k++) c[i * n * p + j * p + k] = aa + i * n * p * q + j * p * q + k * q - qstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray4dWrite - Returns a pointer to a 4d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call `VecRestoreArray4dWrite()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of four dimensional array
.  p - third dimension of four dimensional array
.  q - fourth dimension of four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  qstart - first index in the fourth coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray3d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray4dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[])
{
  PetscInt     i, N, j, k;
  PetscScalar *aa, ***b, **c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p * q == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 4d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p, q);
  PetscCall(VecGetArrayWrite(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar ***) + m * n * sizeof(PetscScalar **) + m * n * p * sizeof(PetscScalar *), a));
  b = (PetscScalar ***)((*a) + m);
  c = (PetscScalar **)(b + m * n);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = c + i * n * p + j * p - pstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < p; k++) c[i * n * p + j * p + k] = aa + i * n * p * q + j * p * q + k * q - qstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray4d - Restores a vector after `VecGetArray4d()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of the four dimensional array
.  p - third dimension of the four dimensional array
.  q - fourth dimension of the four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
.  qstart - first index in the fourth coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray4d()

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray4d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray4dWrite - Restores a vector after `VecGetArray4dWrite()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of the four dimensional array
.  p - third dimension of the four dimensional array
.  q - fourth dimension of the four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
.  qstart - first index in the fourth coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray4d()`

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray4dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray2dRead - Returns a pointer to a 2d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray2dRead()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart and nstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray2d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray2dRead(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[])
{
  PetscInt           i, N;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 2d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n);
  PetscCall(VecGetArrayRead(x, &aa));

  PetscCall(PetscMalloc1(m, a));
  for (i = 0; i < m; i++) (*a)[i] = (PetscScalar *)aa + i * n - nstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray2dRead - Restores a vector after `VecGetArray2dRead()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of the two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray2d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray2dRead(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray1dRead - Returns a pointer to a 1d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray1dRead()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray2d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray1dRead(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[])
{
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 4);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m == N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local array size %" PetscInt_FMT " does not match 1d array dimensions %" PetscInt_FMT, N, m);
  PetscCall(VecGetArrayRead(x, (const PetscScalar **)a));
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray1dRead - Restores a vector after `VecGetArray1dRead()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray1dRead()`

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray1dRead()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray2d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray1dRead(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray3dRead - Returns a pointer to a 3d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray3dRead()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of three dimensional array
.  p - third dimension of three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  pstart - first index in the third coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray3dRead()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray3dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[])
{
  PetscInt           i, N, j;
  const PetscScalar *aa;
  PetscScalar      **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 3d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p);
  PetscCall(VecGetArrayRead(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar **) + m * n * sizeof(PetscScalar *), a));
  b = (PetscScalar **)((*a) + m);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = (PetscScalar *)aa + i * n * p + j * p - pstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray3dRead - Restores a vector after `VecGetArray3dRead()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray3dRead()`

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray3dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecGetArray4dRead - Returns a pointer to a 4d contiguous array that contains this
   processor's portion of the vector data.  You MUST call `VecRestoreArray4dRead()`
   when you no longer need access to the array.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of four dimensional array
.  p - third dimension of four dimensional array
.  q - fourth dimension of four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  qstart - first index in the fourth coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

  Notes:
   For a vector obtained from `DMCreateLocalVector()` mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from `DMDAGetGhostCorners()` while for
   `DMCreateGlobalVector()` they are the corner indices from `DMDAGetCorners()`. In both cases
   the arguments from `DMDAGet[Ghost]Corners()` are reversed in the call to `VecGetArray3d()`.

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray4dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[])
{
  PetscInt           i, N, j, k;
  const PetscScalar *aa;
  PetscScalar     ***b, **c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p * q == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 4d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p, q);
  PetscCall(VecGetArrayRead(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar ***) + m * n * sizeof(PetscScalar **) + m * n * p * sizeof(PetscScalar *), a));
  b = (PetscScalar ***)((*a) + m);
  c = (PetscScalar **)(b + m * n);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = c + i * n * p + j * p - pstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < p; k++) c[i * n * p + j * p + k] = (PetscScalar *)aa + i * n * p * q + j * p * q + k * q - qstart;
  *a -= mstart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecRestoreArray4dRead - Restores a vector after `VecGetArray4d()` has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of the four dimensional array
.  p - third dimension of the four dimensional array
.  q - fourth dimension of the four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
.  qstart - first index in the fourth coordinate direction (often 0)
-  a - location of pointer to array obtained from `VecGetArray4dRead()`

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with `VecGetArray()`.

   This routine actually zeros out the a pointer.

.seealso: [](chapter_vectors), `Vec`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray4dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[])
{
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_DEBUG)

/*@
   VecLockGet  - Gets the current lock status of a vector

   Logically Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  state - greater than zero indicates the vector is locked for read; less then zero indicates the vector is
           locked for write; equal to zero means the vector is unlocked, that is, it is free to read or write.

   Level: advanced

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPush()`, `VecLockReadPop()`
@*/
PetscErrorCode VecLockGet(Vec x, PetscInt *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  *state = x->lock;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecLockGetLocation(Vec x, const char *file[], const char *func[], int *line)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(file, 2);
  PetscValidPointer(func, 3);
  PetscValidIntPointer(line, 4);
  #if !PetscDefined(HAVE_THREADSAFETY)
  {
    const int index = x->lockstack.currentsize - 1;

    PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Corrupted vec lock stack, have negative index %d", index);
    *file = x->lockstack.file[index];
    *func = x->lockstack.function[index];
    *line = x->lockstack.line[index];
  }
  #else
  *file = NULL;
  *func = NULL;
  *line = 0;
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecLockReadPush  - Pushes a read-only lock on a vector to prevent it from being written to

   Logically Collective

   Input Parameter:
.  x - the vector

   Level: intermediate

   Notes:
    If this is set then calls to `VecGetArray()` or `VecSetValues()` or any other routines that change the vectors values will generate an error.

    The call can be nested, i.e., called multiple times on the same vector, but each `VecLockReadPush()` has to have one matching
    `VecLockReadPop()`, which removes the latest read-only lock.

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPop()`, `VecLockGet()`
@*/
PetscErrorCode VecLockReadPush(Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCheck(x->lock++ >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is already locked for exclusive write access but you want to read it");
  #if !PetscDefined(HAVE_THREADSAFETY)
  {
    const char *file, *func;
    int         index, line;

    if ((index = petscstack.currentsize - 2) == -1) {
      // vec was locked "outside" of petsc, either in user-land or main. the error message will
      // now show this function as the culprit, but it will include the stacktrace
      file = "unknown user-file";
      func = "unknown_user_function";
      line = 0;
    } else {
      PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected petscstack, have negative index %d", index);
      file = petscstack.file[index];
      func = petscstack.function[index];
      line = petscstack.line[index];
    }
    PetscStackPush_Private(x->lockstack, file, func, line, petscstack.petscroutine[index], PETSC_FALSE);
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecLockReadPop  - Pops a read-only lock from a vector

   Logically Collective

   Input Parameter:
.  x - the vector

   Level: intermediate

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPush()`, `VecLockGet()`
@*/
PetscErrorCode VecLockReadPop(Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCheck(--x->lock >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector has been unlocked from read-only access too many times");
  #if !PetscDefined(HAVE_THREADSAFETY)
  {
    const char *previous = x->lockstack.function[x->lockstack.currentsize - 1];

    PetscStackPop_Private(x->lockstack, previous);
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecLockWriteSet  - Lock or unlock a vector for exclusive read/write access

   Logically Collective

   Input Parameters:
+  x   - the vector
-  flg - `PETSC_TRUE` to lock the vector for exclusive read/write access; `PETSC_FALSE` to unlock it.

   Level: intermediate

   Notes:
    The function is useful in split-phase computations, which usually have a begin phase and an end phase.
    One can call `VecLockWriteSet`(x,`PETSC_TRUE`) in the begin phase to lock a vector for exclusive
    access, and call `VecLockWriteSet`(x,`PETSC_FALSE`) in the end phase to unlock the vector from exclusive
    access. In this way, one is ensured no other operations can access the vector in between. The code may like

.vb
       VecGetArray(x,&xdata); // begin phase
       VecLockWriteSet(v,PETSC_TRUE);

       Other operations, which can not access x anymore (they can access xdata, of course)

       VecRestoreArray(x,&vdata); // end phase
       VecLockWriteSet(v,PETSC_FALSE);
.ve

    The call can not be nested on the same vector, in other words, one can not call `VecLockWriteSet`(x,`PETSC_TRUE`)
    again before calling `VecLockWriteSet`(v,`PETSC_FALSE`).

.seealso: [](chapter_vectors), `Vec`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPush()`, `VecLockReadPop()`, `VecLockGet()`
@*/
PetscErrorCode VecLockWriteSet(Vec x, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (flg) {
    PetscCheck(x->lock <= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is already locked for read-only access but you want to write it");
    PetscCheck(x->lock >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is already locked for exclusive write access but you want to write it");
    x->lock = -1;
  } else {
    PetscCheck(x->lock == -1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is not locked for exclusive write access but you want to unlock it from that");
    x->lock = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecLockPush  - Pushes a read-only lock on a vector to prevent it from being written to

   Level: deprecated

.seealso: [](chapter_vectors), `Vec`, `VecLockReadPush()`
@*/
PetscErrorCode VecLockPush(Vec x)
{
  PetscFunctionBegin;
  PetscCall(VecLockReadPush(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecLockPop  - Pops a read-only lock from a vector

   Level: deprecated

.seealso: [](chapter_vectors), `Vec`, `VecLockReadPop()`
@*/
PetscErrorCode VecLockPop(Vec x)
{
  PetscFunctionBegin;
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
