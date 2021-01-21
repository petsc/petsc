/*
     Provides the interface functions for vector operations that have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include "petsc/private/sfimpl.h"
#include "petscsystypes.h"
#include <petsc/private/vecimpl.h>       /*I  "petscvec.h"   I*/
#if defined(PETSC_HAVE_CUDA)
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/cudavecimpl.h>
#endif
#if defined(PETSC_HAVE_HIP)
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/hipvecimpl.h>
#endif
static PetscInt VecGetSubVectorSavedStateId = -1;

PETSC_EXTERN PetscErrorCode VecValidValues(Vec vec,PetscInt argnum,PetscBool begin)
{
#if defined(PETSC_USE_DEBUG)
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscScalar *x;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_DEVICE)
  if ((vec->petscnative || vec->ops->getarray) && (vec->offloadmask & PETSC_OFFLOAD_CPU)) {
#else
  if (vec->petscnative || vec->ops->getarray) {
#endif
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (begin) {
        if (PetscIsInfOrNanScalar(x[i])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FP,"Vec entry at local location %D is not-a-number or infinite at beginning of function: Parameter number %D",i,argnum);
      } else {
        if (PetscIsInfOrNanScalar(x[i])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FP,"Vec entry at local location %D is not-a-number or infinite at end of function: Parameter number %D",i,argnum);
      }
    }
    ierr = VecRestoreArrayRead(vec,&x);CHKERRQ(ierr);
  }
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

/*@
   VecMaxPointwiseDivide - Computes the maximum of the componentwise division max = max_i abs(x_i/y_i).

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  max - the result

   Level: advanced

   Notes:
    x and y may be the same vector
          if a particular y_i is zero, it is treated as 1 in the above formula

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMax(), VecPointwiseMin(), VecPointwiseMaxAbs()
@*/
PetscErrorCode  VecMaxPointwiseDivide(Vec x,Vec y,PetscReal *max)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidRealPointer(max,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  VecCheckSameSize(x,1,y,2);
  ierr = (*x->ops->maxpointwisedivide)(x,y,max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecDot - Computes the vector dot product.

   Collective on Vec

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the dot product

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load inbalance that causes certain processes to arrive much earlier than others

   Notes for Users of Complex Numbers:
   For complex vectors, VecDot() computes
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y. Note that this corresponds to the usual "mathematicians" complex
   inner product where the SECOND argument gets the complex conjugate. Since the BLASdot() complex conjugates the first
   first argument we call the BLASdot() with the arguments reversed.

   Use VecTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Level: intermediate


.seealso: VecMDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), VecDotRealPart()
@*/
PetscErrorCode  VecDot(Vec x,Vec y,PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  VecCheckSameSize(x,1,y,2);

  ierr = PetscLogEventBegin(VEC_Dot,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->dot)(x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Dot,x,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecDotRealPart - Computes the real part of the vector dot product.

   Collective on Vec

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the real part of the dot product;

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load inbalance that causes certain processes to arrive much earlier than others

   Notes for Users of Complex Numbers:
     See VecDot() for more details on the definition of the dot product for complex numbers

     For real numbers this returns the same value as VecDot()

     For complex numbers in C^n (that is a vector of n components with a complex number for each component) this is equal to the usual real dot product on the
     the space R^{2n} (that is a vector of 2n components with the real or imaginary part of the complex numbers for components)

   Developer Note: This is not currently optimized to compute only the real part of the dot product.

   Level: intermediate


.seealso: VecMDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), VecDot(), VecDotNorm2()
@*/
PetscErrorCode  VecDotRealPart(Vec x,Vec y,PetscReal *val)
{
  PetscErrorCode ierr;
  PetscScalar    fdot;

  PetscFunctionBegin;
  ierr = VecDot(x,y,&fdot);CHKERRQ(ierr);
  *val = PetscRealPart(fdot);
  PetscFunctionReturn(0);
}

/*@
   VecNorm  - Computes the vector norm.

   Collective on Vec

   Input Parameters:
+  x - the vector
-  type - one of NORM_1, NORM_2, NORM_INFINITY.  Also available
          NORM_1_AND_2, which computes both norms and stores them
          in a two element array.

   Output Parameter:
.  val - the norm

   Notes:
$     NORM_1 denotes sum_i |x_i|
$     NORM_2 denotes sqrt(sum_i |x_i|^2)
$     NORM_INFINITY denotes max_i |x_i|

      For complex numbers NORM_1 will return the traditional 1 norm of the 2 norm of the complex numbers; that is the 1
      norm of the absolute values of the complex entries. In PETSc 3.6 and earlier releases it returned the 1 norm of
      the 1 norm of the complex entries (what is returned by the BLAS routine asum()). Both are valid norms but most
      people expect the former.

   Level: intermediate

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load inbalance that causes certain processes to arrive much earlier than others


.seealso: VecDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), VecNormAvailable(),
          VecNormBegin(), VecNormEnd()

@*/

PetscErrorCode  VecNorm(Vec x,NormType type,PetscReal *val)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidRealPointer(val,3);
  PetscValidType(x,1);

  /*
   * Cached data?
   */
  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[type],*val,flg);CHKERRQ(ierr);
    if (flg) PetscFunctionReturn(0);
  }
  ierr = PetscLogEventBegin(VEC_Norm,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->norm)(x,type,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Norm,x,0,0,0);CHKERRQ(ierr);
  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[type],*val);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   VecNormAvailable  - Returns the vector norm if it is already known.

   Not Collective

   Input Parameters:
+  x - the vector
-  type - one of NORM_1, NORM_2, NORM_INFINITY.  Also available
          NORM_1_AND_2, which computes both norms and stores them
          in a two element array.

   Output Parameter:
+  available - PETSC_TRUE if the val returned is valid
-  val - the norm

   Notes:
$     NORM_1 denotes sum_i |x_i|
$     NORM_2 denotes sqrt(sum_i (x_i)^2)
$     NORM_INFINITY denotes max_i |x_i|

   Level: intermediate

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load inbalance that causes certain processes to arrive much earlier than others

   Compile Option:
   PETSC_HAVE_SLOW_BLAS_NORM2 will cause a C (loop unrolled) version of the norm to be used, rather
 than the BLAS. This should probably only be used when one is using the FORTRAN BLAS routines
 (as opposed to vendor provided) because the FORTRAN BLAS NRM2() routine is very slow.


.seealso: VecDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), VecNorm()
          VecNormBegin(), VecNormEnd()

@*/
PetscErrorCode  VecNormAvailable(Vec x,NormType type,PetscBool  *available,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidRealPointer(val,3);
  PetscValidType(x,1);

  *available = PETSC_FALSE;
  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[type],*val,*available);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   VecNormalize - Normalizes a vector by 2-norm.

   Collective on Vec

   Input Parameters:
+  x - the vector

   Output Parameter:
.  x - the normalized vector
-  val - the vector norm before normalization

   Level: intermediate


@*/
PetscErrorCode  VecNormalize(Vec x,PetscReal *val)
{
  PetscErrorCode ierr;
  PetscReal      norm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_Normalize,x,0,0,0);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  if (norm == 0.0) {
    ierr = PetscInfo(x,"Vector of zero norm can not be normalized; Returning only the zero norm\n");CHKERRQ(ierr);
  } else if (norm != 1.0) {
    PetscScalar tmp = 1.0/norm;
    ierr = VecScale(x,tmp);CHKERRQ(ierr);
  }
  if (val) *val = norm;
  ierr = PetscLogEventEnd(VEC_Normalize,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecMax - Determines the vector component with maximum real part and its location.

   Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameters:
+  p - the location of val (pass NULL if you don't want this)
-  val - the maximum component

   Notes:
   Returns the value PETSC_MIN_REAL and p = -1 if the vector is of length 0.

   Returns the smallest index with the maximum value
   Level: intermediate


.seealso: VecNorm(), VecMin()
@*/
PetscErrorCode  VecMax(Vec x,PetscInt *p,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_Max,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->max)(x,p,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Max,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecMin - Determines the vector component with minimum real part and its location.

   Collective on Vec

   Input Parameters:
.  x - the vector

   Output Parameter:
+  p - the location of val (pass NULL if you don't want this location)
-  val - the minimum component

   Level: intermediate

   Notes:
   Returns the value PETSC_MAX_REAL and p = -1 if the vector is of length 0.

   This returns the smallest index with the minumum value


.seealso: VecMax()
@*/
PetscErrorCode  VecMin(Vec x,PetscInt *p,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_Min,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->min)(x,p,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Min,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecTDot - Computes an indefinite vector dot product. That is, this
   routine does NOT use the complex conjugate.

   Collective on Vec

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the dot product

   Notes for Users of Complex Numbers:
   For complex vectors, VecTDot() computes the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecDot() for the inner product
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Level: intermediate

.seealso: VecDot(), VecMTDot()
@*/
PetscErrorCode  VecTDot(Vec x,Vec y,PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  VecCheckSameSize(x,1,y,2);

  ierr = PetscLogEventBegin(VEC_TDot,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->tdot)(x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_TDot,x,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecScale - Scales a vector.

   Not collective on Vec

   Input Parameters:
+  x - the vector
-  alpha - the scalar

   Output Parameter:
.  x - the scaled vector

   Note:
   For a vector with n components, VecScale() computes
$      x[i] = alpha * x[i], for i=1,...,n.

   Level: intermediate


@*/
PetscErrorCode  VecScale(Vec x, PetscScalar alpha)
{
  PetscReal      norms[4] = {0.0,0.0,0.0, 0.0};
  PetscBool      flgs[4];
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = PetscLogEventBegin(VEC_Scale,x,0,0,0);CHKERRQ(ierr);
  if (alpha != (PetscScalar)1.0) {
    ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);
    /* get current stashed norms */
    for (i=0; i<4; i++) {
      ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],norms[i],flgs[i]);CHKERRQ(ierr);
    }
    ierr = (*x->ops->scale)(x,alpha);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
    /* put the scaled stashed norms back into the Vec */
    for (i=0; i<4; i++) {
      if (flgs[i]) {
        ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[i],PetscAbsScalar(alpha)*norms[i]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(VEC_Scale,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecSet - Sets all components of a vector to a single scalar value.

   Logically Collective on Vec

   Input Parameters:
+  x  - the vector
-  alpha - the scalar

   Output Parameter:
.  x  - the vector

   Note:
   For a vector of dimension n, VecSet() computes
$     x[i] = alpha, for i=1,...,n,
   so that all vector entries then equal the identical
   scalar value, alpha.  Use the more general routine
   VecSetValues() to set different vector entries.

   You CANNOT call this after you have called VecSetValues() but before you call
   VecAssemblyBegin/End().

   Level: beginner

.seealso VecSetValues(), VecSetValuesBlocked(), VecSetRandom()

@*/
PetscErrorCode  VecSet(Vec x,PetscScalar alpha)
{
  PetscReal      val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"You cannot call this after you have called VecSetValues() but\n before you have called VecAssemblyBegin/End()");
  PetscValidLogicalCollectiveScalar(x,alpha,2);
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(VEC_Set,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->set)(x,alpha);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Set,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);

  /*  norms can be simply set (if |alpha|*N not too large) */
  val  = PetscAbsScalar(alpha);
  if (x->map->N == 0) {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_1],0.0l);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_INFINITY],0.0);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_2],0.0);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_FROBENIUS],0.0);CHKERRQ(ierr);
  } else if (val > PETSC_MAX_REAL/x->map->N) {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_INFINITY],val);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_1],x->map->N * val);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_INFINITY],val);CHKERRQ(ierr);
    val  = PetscSqrtReal((PetscReal)x->map->N) * val;
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_2],val);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_FROBENIUS],val);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@
   VecAXPY - Computes y = alpha x + y.

   Logically Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    x and y MUST be different vectors
    This routine is optimized for alpha of 0.0, otherwise it calls the BLAS routine

$    VecAXPY(y,alpha,x)                   y = alpha x           +      y
$    VecAYPX(y,beta,x)                    y =       x           + beta y
$    VecAXPBY(y,alpha,beta,x)             y = alpha x           + beta y
$    VecWAXPY(w,alpha,x,y)                w = alpha x           +      y
$    VecAXPBYPCZ(w,alpha,beta,gamma,x,y)  z = alpha x           + beta y + gamma z
$    VecMAXPY(y,nv,alpha[],x[])           y = sum alpha[i] x[i] +      y


.seealso:  VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPBYPCZ(), VecAXPBY()
@*/
PetscErrorCode  VecAXPY(Vec y,PetscScalar alpha,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,1);
  PetscValidType(x,3);
  PetscValidType(y,1);
  PetscCheckSameTypeAndComm(x,3,y,1);
  VecCheckSameSize(x,1,y,3);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_IDN,"x and y cannot be the same vector");
  PetscValidLogicalCollectiveScalar(y,alpha,2);
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(0);
  ierr = VecSetErrorIfLocked(y,1);CHKERRQ(ierr);

  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->axpy)(y,alpha,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecAXPBY - Computes y = alpha x + beta y.

   Logically Collective on Vec

   Input Parameters:
+  alpha,beta - the scalars
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    x and y MUST be different vectors
    The implementation is optimized for alpha and/or beta values of 0.0 and 1.0


.seealso: VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY(), VecAXPBYPCZ()
@*/
PetscErrorCode  VecAXPBY(Vec y,PetscScalar alpha,PetscScalar beta,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,4);
  PetscValidHeaderSpecific(y,VEC_CLASSID,1);
  PetscValidType(x,4);
  PetscValidType(y,1);
  PetscCheckSameTypeAndComm(x,4,y,1);
  VecCheckSameSize(y,1,x,4);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_IDN,"x and y cannot be the same vector");
  PetscValidLogicalCollectiveScalar(y,alpha,2);
  PetscValidLogicalCollectiveScalar(y,beta,3);
  if (alpha == (PetscScalar)0.0 && beta == (PetscScalar)1.0) PetscFunctionReturn(0);
  ierr = VecSetErrorIfLocked(y,1);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->axpby)(y,alpha,beta,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecAXPBYPCZ - Computes z = alpha x + beta y + gamma z

   Logically Collective on Vec

   Input Parameters:
+  alpha,beta, gamma - the scalars
-  x, y, z  - the vectors

   Output Parameter:
.  z - output vector

   Level: intermediate

   Notes:
    x, y and z must be different vectors
    The implementation is optimized for alpha of 1.0 and gamma of 1.0 or 0.0


.seealso:  VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY(), VecAXPBY()
@*/
PetscErrorCode  VecAXPBYPCZ(Vec z,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,5);
  PetscValidHeaderSpecific(y,VEC_CLASSID,6);
  PetscValidHeaderSpecific(z,VEC_CLASSID,1);
  PetscValidType(x,5);
  PetscValidType(y,6);
  PetscValidType(z,1);
  PetscCheckSameTypeAndComm(x,5,y,6);
  PetscCheckSameTypeAndComm(x,5,z,1);
  VecCheckSameSize(x,1,y,5);
  VecCheckSameSize(x,1,z,6);
  if (x == y || x == z) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_IDN,"x, y, and z must be different vectors");
  if (y == z) SETERRQ(PetscObjectComm((PetscObject)y),PETSC_ERR_ARG_IDN,"x, y, and z must be different vectors");
  PetscValidLogicalCollectiveScalar(z,alpha,2);
  PetscValidLogicalCollectiveScalar(z,beta,3);
  PetscValidLogicalCollectiveScalar(z,gamma,4);
  if (alpha == (PetscScalar)0.0 && beta == (PetscScalar)0.0 && gamma == (PetscScalar)1.0) PetscFunctionReturn(0);
  ierr = VecSetErrorIfLocked(z,1);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(VEC_AXPBYPCZ,x,y,z,0);CHKERRQ(ierr);
  ierr = (*y->ops->axpbypcz)(z,alpha,beta,gamma,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPBYPCZ,x,y,z,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecAYPX - Computes y = x + beta y.

   Logically Collective on Vec

   Input Parameters:
+  beta - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    x and y MUST be different vectors
    The implementation is optimized for beta of -1.0, 0.0, and 1.0


.seealso:  VecMAXPY(), VecWAXPY(), VecAXPY(), VecAXPBYPCZ(), VecAXPBY()
@*/
PetscErrorCode  VecAYPX(Vec y,PetscScalar beta,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,1);
  PetscValidType(x,3);
  PetscValidType(y,1);
  PetscCheckSameTypeAndComm(x,3,y,1);
  VecCheckSameSize(x,1,y,3);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  PetscValidLogicalCollectiveScalar(y,beta,2);
  ierr = VecSetErrorIfLocked(y,1);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(VEC_AYPX,x,y,0,0);CHKERRQ(ierr);
  ierr =  (*y->ops->aypx)(y,beta,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AYPX,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
   VecWAXPY - Computes w = alpha x + y.

   Logically Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: intermediate

   Notes:
    w cannot be either x or y, but x and y can be the same
    The implementation is optimzed for alpha of -1.0, 0.0, and 1.0


.seealso: VecAXPY(), VecAYPX(), VecAXPBY(), VecMAXPY(), VecAXPBYPCZ()
@*/
PetscErrorCode  VecWAXPY(Vec w,PetscScalar alpha,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidType(w,1);
  PetscValidType(x,3);
  PetscValidType(y,4);
  PetscCheckSameTypeAndComm(x,3,y,4);
  PetscCheckSameTypeAndComm(y,4,w,1);
  VecCheckSameSize(x,3,y,4);
  VecCheckSameSize(x,3,w,1);
  if (w == y) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Result vector w cannot be same as input vector y, suggest VecAXPY()");
  if (w == x) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Result vector w cannot be same as input vector x, suggest VecAYPX()");
  PetscValidLogicalCollectiveScalar(y,alpha,2);
  ierr = VecSetErrorIfLocked(w,1);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(VEC_WAXPY,x,y,w,0);CHKERRQ(ierr);
  ierr =  (*w->ops->waxpy)(w,alpha,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_WAXPY,x,y,w,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
   VecSetValues - Inserts or adds values into certain locations of a vector.

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   VecSetValues() sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValues() have been completed.

   VecSetValues() uses 0-based indices in Fortran as well as in C.

   If you call VecSetOption(x, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE),
   negative indices may be passed in ix. These rows are
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

   Level: beginner

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesLocal(),
           VecSetValue(), VecSetValuesBlocked(), InsertMode, INSERT_VALUES, ADD_VALUES, VecGetValues()
@*/
PetscErrorCode  VecSetValues(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setvalues)(x,ni,ix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetValues - Gets values from certain locations of a vector. Currently
          can only get values on the same processor

    Not Collective

   Input Parameters:
+  x - vector to get values from
.  ni - number of elements to get
-  ix - indices where to get them from (in global 1d numbering)

   Output Parameter:
.   y - array of values

   Notes:
   The user provides the allocated array y; it is NOT allocated in this routine

   VecGetValues() gets y[i] = x[ix[i]], for i=0,...,ni-1.

   VecAssemblyBegin() and VecAssemblyEnd()  MUST be called before calling this

   VecGetValues() uses 0-based indices in Fortran as well as in C.

   If you call VecSetOption(x, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE),
   negative indices may be passed in ix. These rows are
   simply ignored.

   Level: beginner

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues()
@*/
PetscErrorCode  VecGetValues(Vec x,PetscInt ni,const PetscInt ix[],PetscScalar y[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  ierr = (*x->ops->getvalues)(x,ni,ix,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecSetValuesBlocked - Inserts or adds blocks of values into certain locations of a vector.

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, rather than element count
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   VecSetValuesBlocked() sets x[bs*ix[i]+j] = y[bs*i+j],
   for j=0,...,bs-1, for i=0,...,ni-1. where bs was set with VecSetBlockSize().

   Calls to VecSetValuesBlocked() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValuesBlocked() have been completed.

   VecSetValuesBlocked() uses 0-based indices in Fortran as well as in C.

   Negative indices may be passed in ix, these rows are
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

   Level: intermediate

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesBlockedLocal(),
           VecSetValues()
@*/
PetscErrorCode  VecSetValuesBlocked(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)(x,ni,ix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: intermediate

   Notes:
   VecSetValuesLocal() sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValuesLocal() have been completed.

   VecSetValuesLocal() uses 0-based indices in Fortran as well as in C.

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetLocalToGlobalMapping(),
           VecSetValuesBlockedLocal()
@*/
PetscErrorCode  VecSetValuesLocal(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora)
{
  PetscErrorCode ierr;
  PetscInt       lixp[128],*lix = lixp;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  if (!x->ops->setvalueslocal) {
    if (!x->map->mapping) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with VecSetLocalToGlobalMapping()");
    if (ni > 128) {
      ierr = PetscMalloc1(ni,&lix);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingApply(x->map->mapping,ni,(PetscInt*)ix,lix);CHKERRQ(ierr);
    ierr = (*x->ops->setvalues)(x,ni,lix,y,iora);CHKERRQ(ierr);
    if (ni > 128) {
      ierr = PetscFree(lix);CHKERRQ(ierr);
    }
  } else {
    ierr = (*x->ops->setvalueslocal)(x,ni,ix,y,iora);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: intermediate

   Notes:
   VecSetValuesBlockedLocal() sets x[bs*ix[i]+j] = y[bs*i+j],
   for j=0,..bs-1, for i=0,...,ni-1, where bs has been set with VecSetBlockSize().

   Calls to VecSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValuesBlockedLocal() have been completed.

   VecSetValuesBlockedLocal() uses 0-based indices in Fortran as well as in C.


.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesBlocked(),
           VecSetLocalToGlobalMapping()
@*/
PetscErrorCode  VecSetValuesBlockedLocal(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora)
{
  PetscErrorCode ierr;
  PetscInt       lixp[128],*lix = lixp;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  if (ni > 128) {
    ierr = PetscMalloc1(ni,&lix);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(x->map->mapping,ni,(PetscInt*)ix,lix);CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)(x,ni,lix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  if (ni > 128) {
    ierr = PetscFree(lix);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecMTDot - Computes indefinite vector multiple dot products.
   That is, it does NOT use the complex conjugate.

   Collective on Vec

   Input Parameters:
+  x - one vector
.  nv - number of vectors
-  y - array of vectors.  Note that vectors are pointers

   Output Parameter:
.  val - array of the dot products

   Notes for Users of Complex Numbers:
   For complex vectors, VecMTDot() computes the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecMDot() for the inner product
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Level: intermediate


.seealso: VecMDot(), VecTDot()
@*/
PetscErrorCode  VecMTDot(Vec x,PetscInt nv,const Vec y[],PetscScalar val[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(x,nv,2);
  if (!nv) PetscFunctionReturn(0);
  PetscValidPointer(y,3);
  PetscValidHeaderSpecific(*y,VEC_CLASSID,3);
  PetscValidScalarPointer(val,4);
  PetscValidType(x,2);
  PetscValidType(*y,3);
  PetscCheckSameTypeAndComm(x,2,*y,3);
  VecCheckSameSize(x,1,*y,3);

  ierr = PetscLogEventBegin(VEC_MTDot,x,*y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mtdot)(x,nv,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MTDot,x,*y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecMDot - Computes vector multiple dot products.

   Collective on Vec

   Input Parameters:
+  x - one vector
.  nv - number of vectors
-  y - array of vectors.

   Output Parameter:
.  val - array of the dot products (does not allocate the array)

   Notes for Users of Complex Numbers:
   For complex vectors, VecMDot() computes
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use VecMTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Level: intermediate


.seealso: VecMTDot(), VecDot()
@*/
PetscErrorCode  VecMDot(Vec x,PetscInt nv,const Vec y[],PetscScalar val[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(x,nv,2);
  if (!nv) PetscFunctionReturn(0);
  if (nv < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",nv);
  PetscValidPointer(y,3);
  PetscValidHeaderSpecific(*y,VEC_CLASSID,3);
  PetscValidScalarPointer(val,4);
  PetscValidType(x,2);
  PetscValidType(*y,3);
  PetscCheckSameTypeAndComm(x,2,*y,3);
  VecCheckSameSize(x,1,*y,3);

  ierr = PetscLogEventBegin(VEC_MDot,x,*y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mdot)(x,nv,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MDot,x,*y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecMAXPY - Computes y = y + sum alpha[i] x[i]

   Logically Collective on Vec

   Input Parameters:
+  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  y - one vector
-  x - array of vectors

   Level: intermediate

   Notes:
    y cannot be any of the x vectors

.seealso:  VecAYPX(), VecWAXPY(), VecAXPY(), VecAXPBYPCZ(), VecAXPBY()
@*/
PetscErrorCode  VecMAXPY(Vec y,PetscInt nv,const PetscScalar alpha[],Vec x[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      nonzero;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(y,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(y,nv,2);
  if (!nv) PetscFunctionReturn(0);
  if (nv < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",nv);
  PetscValidScalarPointer(alpha,3);
  PetscValidPointer(x,4);
  PetscValidHeaderSpecific(*x,VEC_CLASSID,4);
  PetscValidType(y,1);
  PetscValidType(*x,4);
  PetscCheckSameTypeAndComm(y,1,*x,4);
  VecCheckSameSize(y,1,*x,4);
  for (i=0; i<nv; i++) PetscValidLogicalCollectiveScalar(y,alpha[i],3);
  for (i=0, nonzero = PETSC_FALSE; i<nv && !nonzero; i++) nonzero = (PetscBool)(nonzero || alpha[i] != (PetscScalar)0.0);
  if (!nonzero) PetscFunctionReturn(0);
  ierr = VecSetErrorIfLocked(y,1);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VEC_MAXPY,*x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->maxpy)(y,nv,alpha,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MAXPY,*x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecConcatenate - Creates a new vector that is a vertical concatenation of all the given array of vectors
                    in the order they appear in the array. The concatenated vector resides on the same
                    communicator and is the same type as the source vectors.

   Collective on X

   Input Arguments:
+  nx   - number of vectors to be concatenated
-  X    - array containing the vectors to be concatenated in the order of concatenation

   Output Arguments:
+  Y    - concatenated vector
-  x_is - array of index sets corresponding to the concatenated components of Y (NULL if not needed)

   Notes:
   Concatenation is similar to the functionality of a VecNest object; they both represent combination of
   different vector spaces. However, concatenated vectors do not store any information about their
   sub-vectors and own their own data. Consequently, this function provides index sets to enable the
   manipulation of data in the concatenated vector that corresponds to the original components at creation.

   This is a useful tool for outer loop algorithms, particularly constrained optimizers, where the solver
   has to operate on combined vector spaces and cannot utilize VecNest objects due to incompatibility with
   bound projections.

   Level: advanced

.seealso: VECNEST, VECSCATTER, VecScatterCreate()
@*/
PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[])
{
  MPI_Comm       comm;
  VecType        vec_type;
  Vec            Ytmp, Xtmp;
  IS             *is_tmp;
  PetscInt       i, shift=0, Xnl, Xng, Xbegin;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(*X,nx,1);
  PetscValidHeaderSpecific(*X,VEC_CLASSID,2);
  PetscValidType(*X,2);
  PetscValidPointer(Y, 3);

  if ((*X)->ops->concatenate) {
    /* use the dedicated concatenation function if available */
    ierr = (*(*X)->ops->concatenate)(nx,X,Y,x_is);
  } else {
    /* loop over vectors and start creating IS */
    comm = PetscObjectComm((PetscObject)(*X));
    ierr = VecGetType(*X, &vec_type);CHKERRQ(ierr);
    ierr = PetscMalloc1(nx, &is_tmp);
    for (i=0; i<nx; i++) {
      ierr = VecGetSize(X[i], &Xng);CHKERRQ(ierr);
      ierr = VecGetLocalSize(X[i], &Xnl);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(X[i], &Xbegin, NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm, Xnl, shift + Xbegin, 1, &is_tmp[i]);
      shift += Xng;
    }
    /* create the concatenated vector */
    ierr = VecCreate(comm, &Ytmp);CHKERRQ(ierr);
    ierr = VecSetType(Ytmp, vec_type);CHKERRQ(ierr);
    ierr = VecSetSizes(Ytmp, PETSC_DECIDE, shift);CHKERRQ(ierr);
    ierr = VecSetUp(Ytmp);CHKERRQ(ierr);
    /* copy data from X array to Y and return */
    for (i=0; i<nx; i++) {
      ierr = VecGetSubVector(Ytmp, is_tmp[i], &Xtmp);CHKERRQ(ierr);
      ierr = VecCopy(X[i], Xtmp);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(Ytmp, is_tmp[i], &Xtmp);CHKERRQ(ierr);
    }
    *Y = Ytmp;
    if (x_is) {
      *x_is = is_tmp;
    } else {
      for (i=0; i<nx; i++) {
        ierr = ISDestroy(&is_tmp[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(is_tmp);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   VecGetSubVector - Gets a vector representing part of another vector

   Collective on X and IS

   Input Arguments:
+ X - vector from which to extract a subvector
- is - index set representing portion of X to extract

   Output Arguments:
. Y - subvector corresponding to is

   Level: advanced

   Notes:
   The subvector Y should be returned with VecRestoreSubVector().
   X and is must be defined on the same communicator

   This function may return a subvector without making a copy, therefore it is not safe to use the original vector while
   modifying the subvector.  Other non-overlapping subvectors can still be obtained from X using this function.
   The resulting subvector inherits the block size from the IS if greater than one. Otherwise, the block size is guessed from the block size of the original vec.

.seealso: MatCreateSubMatrix()
@*/
PetscErrorCode  VecGetSubVector(Vec X,IS is,Vec *Y)
{
  PetscErrorCode   ierr;
  Vec              Z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscCheckSameComm(X,1,is,2);
  PetscValidPointer(Y,3);
  if (X->ops->getsubvector) {
    ierr = (*X->ops->getsubvector)(X,is,&Z);CHKERRQ(ierr);
  } else { /* Default implementation currently does no caching */
    PetscInt  gstart,gend,start;
    PetscBool red[2] = { PETSC_TRUE, PETSC_TRUE };
    PetscInt  n,N,ibs,vbs,bs = -1;

    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetSize(is,&N);CHKERRQ(ierr);
    ierr = ISGetBlockSize(is,&ibs);CHKERRQ(ierr);
    ierr = VecGetBlockSize(X,&vbs);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(X,&gstart,&gend);CHKERRQ(ierr);
    ierr = ISContiguousLocal(is,gstart,gend,&start,&red[0]);CHKERRQ(ierr);
    /* block size is given by IS if ibs > 1; otherwise, check the vector */
    if (ibs > 1) {
      ierr = MPIU_Allreduce(MPI_IN_PLACE,red,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)is));CHKERRQ(ierr);
      bs   = ibs;
    } else {
      if (n%vbs || vbs == 1) red[1] = PETSC_FALSE; /* this process invalidate the collectiveness of block size */
      ierr = MPIU_Allreduce(MPI_IN_PLACE,red,2,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)is));CHKERRQ(ierr);
      if (red[0] && red[1]) bs = vbs; /* all processes have a valid block size and the access will be contiguous */
    }
    if (red[0]) { /* We can do a no-copy implementation */
      const PetscScalar *x;
      PetscInt          state = 0;
      PetscBool         isstd,iscuda,iship;

      ierr = PetscObjectTypeCompareAny((PetscObject)X,&isstd,VECSEQ,VECMPI,VECSTANDARD,"");CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)X,&iscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)X,&iship,VECSEQHIP,VECMPIHIP,"");CHKERRQ(ierr);
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        const PetscScalar *x_d;
        PetscMPIInt       size;
        PetscOffloadMask  flg;

        ierr = VecCUDAGetArrays_Private(X,&x,&x_d,&flg);CHKERRQ(ierr);
        if (flg == PETSC_OFFLOAD_UNALLOCATED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for PETSC_OFFLOAD_UNALLOCATED");
        if (n && !x && !x_d) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Missing vector data");
        if (x) x += start;
        if (x_d) x_d += start;
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)X),&size);CHKERRMPI(ierr);
        if (size == 1) {
          ierr = VecCreateSeqCUDAWithArrays(PetscObjectComm((PetscObject)X),bs,n,x,x_d,&Z);CHKERRQ(ierr);
        } else {
          ierr = VecCreateMPICUDAWithArrays(PetscObjectComm((PetscObject)X),bs,n,N,x,x_d,&Z);CHKERRQ(ierr);
        }
        Z->offloadmask = flg;
#endif
      } else if (iship) {
#if defined(PETSC_HAVE_HIP)
        const PetscScalar *x_d;
        PetscMPIInt       size;
        PetscOffloadMask  flg;

        ierr = VecHIPGetArrays_Private(X,&x,&x_d,&flg);CHKERRQ(ierr);
        if (flg == PETSC_OFFLOAD_UNALLOCATED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for PETSC_OFFLOAD_UNALLOCATED");
        if (n && !x && !x_d) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Missing vector data");
        if (x) x += start;
        if (x_d) x_d += start;
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)X),&size);CHKERRQ(ierr);
        if (size == 1) {
          ierr = VecCreateSeqHIPWithArrays(PetscObjectComm((PetscObject)X),bs,n,x,x_d,&Z);CHKERRQ(ierr);
        } else {
          ierr = VecCreateMPIHIPWithArrays(PetscObjectComm((PetscObject)X),bs,n,N,x,x_d,&Z);CHKERRQ(ierr);
        }
        Z->offloadmask = flg;
#endif
      } else if (isstd) {
        PetscMPIInt size;

        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)X),&size);CHKERRMPI(ierr);
        ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
        if (x) x += start;
        if (size == 1) {
          ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)X),bs,n,x,&Z);CHKERRQ(ierr);
        } else {
          ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)X),bs,n,N,x,&Z);CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
      } else { /* default implementation: use place array */
        ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
        ierr = VecCreate(PetscObjectComm((PetscObject)X),&Z);CHKERRQ(ierr);
        ierr = VecSetType(Z,((PetscObject)X)->type_name);CHKERRQ(ierr);
        ierr = VecSetSizes(Z,n,N);CHKERRQ(ierr);
        ierr = VecSetBlockSize(Z,bs);CHKERRQ(ierr);
        ierr = VecPlaceArray(Z,x ? x+start : NULL);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
      }

      /* this is relevant only in debug mode */
      ierr = VecLockGet(X,&state);CHKERRQ(ierr);
      if (state) {
        ierr = VecLockReadPush(Z);CHKERRQ(ierr);
      }
      Z->ops->placearray = NULL;
      Z->ops->replacearray = NULL;
    } else { /* Have to create a scatter and do a copy */
      VecScatter scatter;

      ierr = VecCreate(PetscObjectComm((PetscObject)is),&Z);CHKERRQ(ierr);
      ierr = VecSetSizes(Z,n,N);CHKERRQ(ierr);
      ierr = VecSetBlockSize(Z,bs);CHKERRQ(ierr);
      ierr = VecSetType(Z,((PetscObject)X)->type_name);CHKERRQ(ierr);
      ierr = VecScatterCreate(X,is,Z,NULL,&scatter);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter,X,Z,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(scatter,X,Z,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)Z,"VecGetSubVector_Scatter",(PetscObject)scatter);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
    }
  }
  /* Record the state when the subvector was gotten so we know whether its values need to be put back */
  if (VecGetSubVectorSavedStateId < 0) {ierr = PetscObjectComposedDataRegister(&VecGetSubVectorSavedStateId);CHKERRQ(ierr);}
  ierr = PetscObjectComposedDataSetInt((PetscObject)Z,VecGetSubVectorSavedStateId,1);CHKERRQ(ierr);
  *Y   = Z;
  PetscFunctionReturn(0);
}

/*@
   VecRestoreSubVector - Restores a subvector extracted using VecGetSubVector()

   Collective on IS

   Input Arguments:
+ X - vector from which subvector was obtained
. is - index set representing the subset of X
- Y - subvector being restored

   Level: advanced

.seealso: VecGetSubVector()
@*/
PetscErrorCode  VecRestoreSubVector(Vec X,IS is,Vec *Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscCheckSameComm(X,1,is,2);
  PetscValidPointer(Y,3);
  PetscValidHeaderSpecific(*Y,VEC_CLASSID,3);
  if (X->ops->restoresubvector) {
    ierr = (*X->ops->restoresubvector)(X,is,Y);CHKERRQ(ierr);
  } else {
    PETSC_UNUSED PetscObjectState dummystate = 0;
    PetscBool valid;

    ierr = PetscObjectComposedDataGetInt((PetscObject)*Y,VecGetSubVectorSavedStateId,dummystate,valid);CHKERRQ(ierr);
    if (!valid) {
      VecScatter scatter;
      PetscInt   state;

      ierr = VecLockGet(X,&state);CHKERRQ(ierr);
      if (state != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Vec X is locked for read-only or read/write access");

      ierr = PetscObjectQuery((PetscObject)*Y,"VecGetSubVector_Scatter",(PetscObject*)&scatter);CHKERRQ(ierr);
      if (scatter) {
        ierr = VecScatterBegin(scatter,*Y,X,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(scatter,*Y,X,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      } else {
        PetscBool         iscuda,iship;
        ierr = PetscObjectTypeCompareAny((PetscObject)X,&iscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
        ierr = PetscObjectTypeCompareAny((PetscObject)X,&iship,VECSEQHIP,VECMPIHIP,"");CHKERRQ(ierr);

        if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
          PetscOffloadMask ymask = (*Y)->offloadmask;

          /* The offloadmask of X dictates where to move memory
             If X GPU data is valid, then move Y data on GPU if needed
             Otherwise, move back to the CPU */
          switch (X->offloadmask) {
          case PETSC_OFFLOAD_BOTH:
            if (ymask == PETSC_OFFLOAD_CPU) {
              ierr = VecCUDAResetArray(*Y);CHKERRQ(ierr);
            } else if (ymask == PETSC_OFFLOAD_GPU) {
              X->offloadmask = PETSC_OFFLOAD_GPU;
            }
            break;
          case PETSC_OFFLOAD_GPU:
            if (ymask == PETSC_OFFLOAD_CPU) {
              ierr = VecCUDAResetArray(*Y);CHKERRQ(ierr);
            }
            break;
          case PETSC_OFFLOAD_CPU:
            if (ymask == PETSC_OFFLOAD_GPU) {
              ierr = VecResetArray(*Y);CHKERRQ(ierr);
            }
            break;
          case PETSC_OFFLOAD_UNALLOCATED:
          case PETSC_OFFLOAD_VECKOKKOS:
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
            break;
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
              ierr = VecHIPResetArray(*Y);CHKERRQ(ierr);
            } else if (ymask == PETSC_OFFLOAD_GPU) {
              X->offloadmask = PETSC_OFFLOAD_GPU;
            }
            break;
          case PETSC_OFFLOAD_GPU:
            if (ymask == PETSC_OFFLOAD_CPU) {
              ierr = VecHIPResetArray(*Y);CHKERRQ(ierr);
            }
            break;
          case PETSC_OFFLOAD_CPU:
            if (ymask == PETSC_OFFLOAD_GPU) {
              ierr = VecResetArray(*Y);CHKERRQ(ierr);
            }
            break;
          case PETSC_OFFLOAD_UNALLOCATED:
          case PETSC_OFFLOAD_VECKOKKOS:
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
            break;
          }
#endif
        } else {
          /* If OpenCL vecs updated the device memory, this triggers a copy on the CPU */
          ierr = VecResetArray(*Y);CHKERRQ(ierr);
        }
        ierr = PetscObjectStateIncrease((PetscObject)X);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   VecGetLocalVectorRead - Maps the local portion of a vector into a
   vector.  You must call VecRestoreLocalVectorRead() when the local
   vector is no longer needed.

   Not collective.

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

   Notes:
   This function is similar to VecGetArrayRead() which maps the local
   portion into a raw pointer.  VecGetLocalVectorRead() is usually
   almost as efficient as VecGetArrayRead() but in certain circumstances
   VecGetLocalVectorRead() can be much more efficient than
   VecGetArrayRead().  This is because the construction of a contiguous
   array representing the vector data required by VecGetArrayRead() can
   be an expensive operation for certain vector types.  For example, for
   GPU vectors VecGetArrayRead() requires that the data between device
   and host is synchronized.

   Unlike VecGetLocalVector(), this routine is not collective and
   preserves cached information.

.seealso: VecRestoreLocalVectorRead(), VecGetLocalVector(), VecGetArrayRead(), VecGetArray()
@*/
PetscErrorCode VecGetLocalVectorRead(Vec v,Vec w)
{
  PetscErrorCode ierr;
  PetscScalar    *a;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  VecCheckSameLocalSize(v,1,w,2);
  if (v->ops->getlocalvectorread) {
    ierr = (*v->ops->getlocalvectorread)(v,w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(v,(const PetscScalar**)&a);CHKERRQ(ierr);
    ierr = VecPlaceArray(w,a);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  ierr = VecLockReadPush(v);CHKERRQ(ierr);
  ierr = VecLockReadPush(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecRestoreLocalVectorRead - Unmaps the local portion of a vector
   previously mapped into a vector using VecGetLocalVectorRead().

   Not collective.

   Input parameter:
+  v - The local portion of this vector was previously mapped into w using VecGetLocalVectorRead().
-  w - The vector into which the local portion of v was mapped.

   Level: beginner

.seealso: VecGetLocalVectorRead(), VecGetLocalVector(), VecGetArrayRead(), VecGetArray()
@*/
PetscErrorCode VecRestoreLocalVectorRead(Vec v,Vec w)
{
  PetscErrorCode ierr;
  PetscScalar    *a;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  if (v->ops->restorelocalvectorread) {
    ierr = (*v->ops->restorelocalvectorread)(v,w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(w,(const PetscScalar**)&a);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v,(const PetscScalar**)&a);CHKERRQ(ierr);
    ierr = VecResetArray(w);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(v);CHKERRQ(ierr);
  ierr = VecLockReadPop(w);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecGetLocalVector - Maps the local portion of a vector into a
   vector.

   Collective on v, not collective on w.

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

   Notes:
   This function is similar to VecGetArray() which maps the local
   portion into a raw pointer.  VecGetLocalVector() is usually about as
   efficient as VecGetArray() but in certain circumstances
   VecGetLocalVector() can be much more efficient than VecGetArray().
   This is because the construction of a contiguous array representing
   the vector data required by VecGetArray() can be an expensive
   operation for certain vector types.  For example, for GPU vectors
   VecGetArray() requires that the data between device and host is
   synchronized.

.seealso: VecRestoreLocalVector(), VecGetLocalVectorRead(), VecGetArrayRead(), VecGetArray()
@*/
PetscErrorCode VecGetLocalVector(Vec v,Vec w)
{
  PetscErrorCode ierr;
  PetscScalar    *a;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  VecCheckSameLocalSize(v,1,w,2);
  if (v->ops->getlocalvector) {
    ierr = (*v->ops->getlocalvector)(v,w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&a);CHKERRQ(ierr);
    ierr = VecPlaceArray(w,a);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecRestoreLocalVector - Unmaps the local portion of a vector
   previously mapped into a vector using VecGetLocalVector().

   Logically collective.

   Input parameter:
+  v - The local portion of this vector was previously mapped into w using VecGetLocalVector().
-  w - The vector into which the local portion of v was mapped.

   Level: beginner

.seealso: VecGetLocalVector(), VecGetLocalVectorRead(), VecRestoreLocalVectorRead(), LocalVectorRead(), VecGetArrayRead(), VecGetArray()
@*/
PetscErrorCode VecRestoreLocalVector(Vec v,Vec w)
{
  PetscErrorCode ierr;
  PetscScalar    *a;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  if (v->ops->restorelocalvector) {
    ierr = (*v->ops->restorelocalvector)(v,w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(w,&a);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&a);CHKERRQ(ierr);
    ierr = VecResetArray(w);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray - Returns a pointer to a contiguous array that contains this
   processor's portion of the vector data. For the standard PETSc
   vectors, VecGetArray() returns a pointer to the local data array and
   does not use any copies. If the underlying vector data is not stored
   in a contiguous array this routine will copy the data to a contiguous
   array and return a pointer to that. You MUST call VecRestoreArray()
   when you no longer need access to the array.

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is used differently from Fortran 77
$    Vec         x
$    PetscScalar x_array(1)
$    PetscOffset i_x
$    PetscErrorCode ierr
$       call VecGetArray(x,x_array,i_x,ierr)
$
$   Access first local entry in vector with
$      value = x_array(i_x + 1)
$
$      ...... other code
$       call VecRestoreArray(x,x_array,i_x,ierr)
   For Fortran 90 see VecGetArrayF90()

   See the Fortran chapter of the users manual and
   petsc/src/snes/tutorials/ex5f.F for details.

   Level: beginner

.seealso: VecRestoreArray(), VecGetArrayRead(), VecGetArrays(), VecGetArrayF90(), VecGetArrayReadF90(), VecPlaceArray(), VecGetArray2d(),
          VecGetArrayPair(), VecRestoreArrayPair(), VecGetArrayWrite(), VecRestoreArrayWrite()
@*/
PetscErrorCode VecGetArray(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);
  if (x->ops->getarray) { /* The if-else order matters! VECNEST, VECCUDA etc should have ops->getarray while VECCUDA etc are petscnative */
    ierr = (*x->ops->getarray)(x,a);CHKERRQ(ierr);
  } else if (x->petscnative) { /* VECSTANDARD */
    *a = *((PetscScalar**)x->data);
  } else SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot get array for vector type \"%s\"",((PetscObject)x)->type_name);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray - Restores a vector after VecGetArray() has been called.

   Logically Collective on Vec

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArray()

   Level: beginner

.seealso: VecGetArray(), VecRestoreArrayRead(), VecRestoreArrays(), VecRestoreArrayF90(), VecRestoreArrayReadF90(), VecPlaceArray(), VecRestoreArray2d(),
          VecGetArrayPair(), VecRestoreArrayPair()
@*/
PetscErrorCode VecRestoreArray(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (x->ops->restorearray) { /* VECNEST, VECCUDA etc */
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  } else if (x->petscnative) { /* VECSTANDARD */
    /* nothing */
  } else SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot restore array for vector type \"%s\"",((PetscObject)x)->type_name);
  if (a) *a = NULL;
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
   The array must be returned using a matching call to VecRestoreArrayRead().

   Unlike VecGetArray(), this routine is not collective and preserves cached information like vector norms.

   Standard PETSc vectors use contiguous storage so that this routine does not perform a copy.  Other vector
   implementations may require a copy, but must such implementations should cache the contiguous representation so that
   only one copy is performed when this routine is called multiple times in sequence.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrayPair(), VecRestoreArrayPair()
@*/
PetscErrorCode VecGetArrayRead(Vec x,const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (x->ops->getarray) { /* VECNEST, VECCUDA, VECKOKKOS etc */
    ierr = (*x->ops->getarray)(x,(PetscScalar**)a);CHKERRQ(ierr);
  } else if (x->petscnative) { /* VECSTANDARD */
    *a = *((PetscScalar**)x->data);
  } else SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot get array read for vector type \"%s\"",((PetscObject)x)->type_name);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayRead - Restore array obtained with VecGetArrayRead()

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrayPair(), VecRestoreArrayPair()
@*/
PetscErrorCode VecRestoreArrayRead(Vec x,const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (x->petscnative) { /* VECSTANDARD, VECCUDA, VECKOKKOS etc */
    /* nothing */
  } else if (x->ops->restorearrayread) { /* VECNEST */
    ierr = (*x->ops->restorearrayread)(x,a);CHKERRQ(ierr);
  } else { /* No one? */
    ierr = (*x->ops->restorearray)(x,(PetscScalar**)a);CHKERRQ(ierr);
  }
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayWrite - Returns a pointer to a contiguous array that WILL contains this
   processor's portion of the vector data. The values in this array are NOT valid, the routine calling this
   routine is responsible for putting values into the array; any values it does not set will be invalid

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Level: intermediate

   This is for vectors associate with GPUs, the vector is not copied up before giving access. If you need correct
   values in the array use VecGetArray()

   Concepts: vector^accessing local values

.seealso: VecRestoreArray(), VecGetArrayRead(), VecGetArrays(), VecGetArrayF90(), VecGetArrayReadF90(), VecPlaceArray(), VecGetArray2d(),
          VecGetArrayPair(), VecRestoreArrayPair(), VecGetArray(), VecRestoreArrayWrite()
@*/
PetscErrorCode VecGetArrayWrite(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);
  if (x->ops->getarraywrite) {
    ierr = (*x->ops->getarraywrite)(x,a);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(x,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayWrite - Restores a vector after VecGetArrayWrite() has been called.

   Logically Collective on Vec

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArray()

   Level: beginner

.seealso: VecGetArray(), VecRestoreArrayRead(), VecRestoreArrays(), VecRestoreArrayF90(), VecRestoreArrayReadF90(), VecPlaceArray(), VecRestoreArray2d(),
          VecGetArrayPair(), VecRestoreArrayPair(), VecGetArrayWrite()
@*/
PetscErrorCode VecRestoreArrayWrite(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (x->ops->restorearraywrite) {
    ierr = (*x->ops->restorearraywrite)(x,a);CHKERRQ(ierr);
  } else if (x->ops->restorearray) {
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  }
  if (a) *a = NULL;
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrays - Returns a pointer to the arrays in a set of vectors
   that were created by a call to VecDuplicateVecs().  You MUST call
   VecRestoreArrays() when you no longer need access to the array.

   Logically Collective on Vec

   Input Parameters:
+  x - the vectors
-  n - the number of vectors

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is not supported in Fortran.

   Level: intermediate

.seealso: VecGetArray(), VecRestoreArrays()
@*/
PetscErrorCode  VecGetArrays(const Vec x[],PetscInt n,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    **q;

  PetscFunctionBegin;
  PetscValidPointer(x,1);
  PetscValidHeaderSpecific(*x,VEC_CLASSID,1);
  PetscValidPointer(a,3);
  if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must get at least one array n = %D",n);
  ierr = PetscMalloc1(n,&q);CHKERRQ(ierr);
  for (i=0; i<n; ++i) {
    ierr = VecGetArray(x[i],&q[i]);CHKERRQ(ierr);
  }
  *a = q;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrays - Restores a group of vectors after VecGetArrays()
   has been called.

   Logically Collective on Vec

   Input Parameters:
+  x - the vector
.  n - the number of vectors
-  a - location of pointer to arrays obtained from VecGetArrays()

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the arrays obtained with VecGetArrays().

   Fortran Note:
   This routine is not supported in Fortran.

   Level: intermediate

.seealso: VecGetArrays(), VecRestoreArray()
@*/
PetscErrorCode  VecRestoreArrays(const Vec x[],PetscInt n,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    **q = *a;

  PetscFunctionBegin;
  PetscValidPointer(x,1);
  PetscValidHeaderSpecific(*x,VEC_CLASSID,1);
  PetscValidPointer(a,3);

  for (i=0; i<n; ++i) {
    ierr = VecRestoreArray(x[i],&q[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayAndMemType - Like VecGetArray(), but if this is a GPU vector and it is currently offloaded to GPU,
   the returned pointer will be a GPU pointer to the GPU memory that contains this processor's portion of the
   vector data. Otherwise, it functions as VecGetArray().

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - location to put pointer to the array
-  mtype - memory type of the array

   Level: beginner

.seealso: VecRestoreArrayAndMemType(), VecRestoreArrayAndMemType(), VecRestoreArray(), VecGetArrayRead(), VecGetArrays(), VecGetArrayF90(), VecGetArrayReadF90(),
          VecPlaceArray(), VecGetArray2d(), VecGetArrayPair(), VecRestoreArrayPair(), VecGetArrayWrite(), VecRestoreArrayWrite()
@*/
PetscErrorCode VecGetArrayAndMemType(Vec x,PetscScalar **a,PetscMemType *mtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);
  if (x->ops->getarrayandmemtype) { /* VECCUDA, VECKOKKOS etc */
    ierr = (*x->ops->getarrayandmemtype)(x,a,mtype);CHKERRQ(ierr);
  } else { /* VECSTANDARD, VECNEST, VECVIENNACL */
    ierr = VecGetArray(x,a);CHKERRQ(ierr);
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayAndMemType - Restores a vector after VecGetArrayAndMemType() has been called.

   Logically Collective on Vec

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArrayAndMemType()

   Level: beginner

.seealso: VecGetArrayAndMemType(), VecGetArray(), VecRestoreArrayRead(), VecRestoreArrays(), VecRestoreArrayF90(), VecRestoreArrayReadF90(),
          VecPlaceArray(), VecRestoreArray2d(), VecGetArrayPair(), VecRestoreArrayPair()
@*/
PetscErrorCode VecRestoreArrayAndMemType(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->ops->restorearrayandmemtype) { /* VECCUDA, VECKOKKOS etc */
    ierr = (*x->ops->restorearrayandmemtype)(x,a);CHKERRQ(ierr);
  } else if (x->ops->restorearray) { /* VECNEST, VECVIENNACL */
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  } /* VECSTANDARD does nothing */
  if (a) *a = NULL;
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayReadAndMemType - Like VecGetArrayRead(), but if this is a CUDA vector and it is currently offloaded to GPU,
   the returned pointer will be a GPU pointer to the GPU memory that contains this processor's portion of the
   vector data. Otherwise, it functions as VecGetArrayRead().

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - the array
-  mtype - memory type of the array

   Level: beginner

   Notes:
   The array must be returned using a matching call to VecRestoreArrayReadAndMemType().


.seealso: VecRestoreArrayReadAndMemType(), VecGetArray(), VecRestoreArray(), VecGetArrayPair(), VecRestoreArrayPair(), VecGetArrayAndMemType()
@*/
PetscErrorCode VecGetArrayReadAndMemType(Vec x,const PetscScalar **a,PetscMemType *mtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
 #if defined(PETSC_USE_DEBUG)
  if (x->ops->getarrayreadandmemtype) SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Not expected vector type \"%s\" has ops->getarrayreadandmemtype",((PetscObject)x)->type_name);
 #endif

  if (x->ops->getarrayandmemtype) { /* VECCUDA, VECKOKKOS etc, though they are also petscnative */
    ierr = (*x->ops->getarrayandmemtype)(x,(PetscScalar**)a,mtype);CHKERRQ(ierr);
  } else if (x->ops->getarray) { /* VECNEST, VECVIENNACL */
    ierr = (*x->ops->getarray)(x,(PetscScalar**)a);CHKERRQ(ierr);
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  } else if (x->petscnative) { /* VECSTANDARD */
    *a = *((PetscScalar**)x->data);
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  } else SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot get array read in place for vector type \"%s\"",((PetscObject)x)->type_name);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayReadAndMemType - Restore array obtained with VecGetArrayReadAndMemType()

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: VecGetArrayReadAndMemType(), VecRestoreArrayAndMemType(), VecGetArray(), VecRestoreArray(), VecGetArrayPair(), VecRestoreArrayPair()
@*/
PetscErrorCode VecRestoreArrayReadAndMemType(Vec x,const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->petscnative) { /* VECSTANDARD, VECCUDA, VECKOKKOS, VECVIENNACL etc */
    /* nothing */
  } else if (x->ops->restorearrayread) { /* VECNEST */
    ierr = (*x->ops->restorearrayread)(x,a);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot restore array read in place for vector type \"%s\"",((PetscObject)x)->type_name);
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

/*@
   VecPlaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the array

   Notes:
   You can return to the original array with a call to VecResetArray()

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray()

@*/
PetscErrorCode  VecPlaceArray(Vec vec,const PetscScalar array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (array) PetscValidScalarPointer(array,2);
  if (vec->ops->placearray) {
    ierr = (*vec->ops->placearray)(vec,array);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)vec),PETSC_ERR_SUP,"Cannot place array in this type of vector");
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecReplaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the array

   Notes:
   This permanently replaces the array and frees the memory associated
   with the old array.

   The memory passed in MUST be obtained with PetscMalloc() and CANNOT be
   freed by the user. It will be freed when the vector is destroyed.

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray()

@*/
PetscErrorCode  VecReplaceArray(Vec vec,const PetscScalar array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (vec->ops->replacearray) {
    ierr = (*vec->ops->replacearray)(vec,array);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)vec),PETSC_ERR_SUP,"Cannot replace array in this type of vector");
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
   VecCUDAGetArray - Provides access to the CUDA buffer inside a vector.

   This function has semantics similar to VecGetArray():  the pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUDAGetArray() assumes that
   the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The CUDA device pointer has to be released by calling
   VecCUDARestoreArray().  Upon restoring the vector data
   the data on the host will be marked as out of date.  A subsequent
   access of the host data will thus incur a data transfer from the
   device to the host.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA device pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDARestoreArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDAGetArray(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
 #if defined(PETSC_HAVE_CUDA)
  {
    PetscErrorCode ierr;
    ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
    *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  }
 #endif
  PetscFunctionReturn(0);
}

/*@C
   VecCUDARestoreArray - Restore a CUDA device pointer previously acquired with VecCUDAGetArray().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUDARestoreArray() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDARestoreArray(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
#if defined(PETSC_HAVE_CUDA)
  v->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAGetArrayRead - Provides read access to the CUDA buffer inside a vector.

   This function is analogous to VecGetArrayRead():  The pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUDAGetArrayRead() assumes that the
   user will not modify the vector data.  This is analgogous to
   intent(in) in Fortran.

   The CUDA device pointer has to be released by calling
   VecCUDARestoreArrayRead().  If the data on the host side was
   previously up to date it will remain so, i.e. data on both the device
   and the host is up to date.  Accessing data on the host side does not
   incur a device to host data transfer.

   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA pointer.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDARestoreArrayRead(), VecCUDAGetArray(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDAGetArrayRead(Vec v,const PetscScalar** a)
{
   PetscErrorCode ierr;
   PetscFunctionBegin;
   ierr = VecCUDAGetArray(v,(PetscScalar**)a);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

/*@C
   VecCUDARestoreArrayRead - Restore a CUDA device pointer previously acquired with VecCUDAGetArrayRead().

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUDARestoreArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecCUDAGetArray(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayRead(Vec v, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAGetArrayWrite - Provides write access to the CUDA buffer inside a vector.

   The data pointed to by the device pointer is uninitialized.  The user
   may not read from this data.  Furthermore, the entire array needs to
   be filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The device pointer needs to be released with
   VecCUDARestoreArrayWrite().  When the pointer is released the
   host data of the vector is marked as out of data.  Subsequent access
   of the host data with e.g. VecGetArray() incurs a device to host data
   transfer.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecCUDARestoreArrayWrite(), VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDAGetArrayWrite(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
 #if defined(PETSC_HAVE_CUDA)
  {
    PetscErrorCode ierr;
    ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
    *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  }
 #endif
  PetscFunctionReturn(0);
}

/*@C
   VecCUDARestoreArrayWrite - Restore a CUDA device pointer previously acquired with VecCUDAGetArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUDARestoreArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDAGetArrayWrite(), VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
 #if defined(PETSC_HAVE_CUDA)
  v->offloadmask = PETSC_OFFLOAD_GPU;
  a              = NULL;
 #endif
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAPlaceArray - Allows one to replace the GPU array in a vector with a
   GPU array provided by the user. This is useful to avoid copying an
   array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   You can return to the original GPU array with a call to VecCUDAResetArray()
   It is not possible to use VecCUDAPlaceArray() and VecPlaceArray() at the
   same time on the same vector.

   Level: developer

.seealso: VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray(), VecCUDAResetArray(), VecCUDAReplaceArray()

@*/
PetscErrorCode VecCUDAPlaceArray(Vec vin,const PetscScalar a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUDA,VECMPICUDA);
#if defined(PETSC_HAVE_CUDA)
  ierr = VecCUDACopyToGPU(vin);CHKERRQ(ierr);
  if (((Vec_Seq*)vin->data)->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecCUDAPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecCUDAResetArray()/VecResetArray()");
  ((Vec_Seq*)vin->data)->unplacedarray  = (PetscScalar *) ((Vec_CUDA*)vin->spptr)->GPUarray; /* save previous GPU array so reset can bring it back */
  ((Vec_CUDA*)vin->spptr)->GPUarray = (PetscScalar*)a;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAReplaceArray - Allows one to replace the GPU array in a vector
   with a GPU array provided by the user. This is useful to avoid copying
   a GPU array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   This permanently replaces the GPU array and frees the memory associated
   with the old GPU array.

   The memory passed in CANNOT be freed by the user. It will be freed
   when the vector is destroyed.

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray(), VecCUDAResetArray(), VecCUDAPlaceArray(), VecReplaceArray()

@*/
PetscErrorCode VecCUDAReplaceArray(Vec vin,const PetscScalar a[])
{
#if defined(PETSC_HAVE_CUDA)
  cudaError_t err;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUDA,VECMPICUDA);
#if defined(PETSC_HAVE_CUDA)
  if (((Vec_CUDA*)vin->spptr)->GPUarray_allocated) {
    err = cudaFree(((Vec_CUDA*)vin->spptr)->GPUarray_allocated);CHKERRCUDA(err);
  }
  ((Vec_CUDA*)vin->spptr)->GPUarray_allocated = ((Vec_CUDA*)vin->spptr)->GPUarray = (PetscScalar*)a;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAResetArray - Resets a vector to use its default memory. Call this
   after the use of VecCUDAPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray(), VecResetArray(), VecCUDAPlaceArray(), VecCUDAReplaceArray()

@*/
PetscErrorCode VecCUDAResetArray(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUDA,VECMPICUDA);
#if defined(PETSC_HAVE_CUDA)
  ierr = VecCUDACopyToGPU(vin);CHKERRQ(ierr);
  ((Vec_CUDA*)vin->spptr)->GPUarray = (PetscScalar *) ((Vec_Seq*)vin->data)->unplacedarray;
  ((Vec_Seq*)vin->data)->unplacedarray = 0;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecHIPGetArray - Provides access to the HIP buffer inside a vector.

   This function has semantics similar to VecGetArray():  the pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecHIPGetArray() assumes that
   the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The HIP device pointer has to be released by calling
   VecHIPRestoreArray().  Upon restoring the vector data
   the data on the host will be marked as out of date.  A subsequent
   access of the host data will thus incur a data transfer from the
   device to the host.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the HIP device pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHIPRestoreArray(), VecHIPGetArrayRead(), VecHIPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHIPGetArray(Vec v, PetscScalar **a)
{
#if defined(PETSC_HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  *a   = 0;
  ierr = VecHIPCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_HIP*)v->spptr)->GPUarray;
#endif
  PetscFunctionReturn(0);
}

/*@C
   VecHIPRestoreArray - Restore a HIP device pointer previously acquired with VecHIPGetArray().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the HIP device pointer.  This pointer is invalid after
       VecHIPRestoreArray() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHIPGetArray(), VecHIPGetArrayRead(), VecHIPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHIPRestoreArray(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  v->offloadmask = PETSC_OFFLOAD_GPU;
#endif

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecHIPGetArrayRead - Provides read access to the HIP buffer inside a vector.

   This function is analogous to VecGetArrayRead():  The pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecHIPGetArrayRead() assumes that the
   user will not modify the vector data.  This is analgogous to
   intent(in) in Fortran.

   The HIP device pointer has to be released by calling
   VecHIPRestoreArrayRead().  If the data on the host side was
   previously up to date it will remain so, i.e. data on both the device
   and the host is up to date.  Accessing data on the host side does not
   incur a device to host data transfer.

   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the HIP pointer.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHIPRestoreArrayRead(), VecHIPGetArray(), VecHIPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHIPGetArrayRead(Vec v, const PetscScalar **a)
{
#if defined(PETSC_HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  *a   = 0;
  ierr = VecHIPCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_HIP*)v->spptr)->GPUarray;
#endif
  PetscFunctionReturn(0);
}

/*@C
   VecHIPRestoreArrayRead - Restore a HIP device pointer previously acquired with VecHIPGetArrayRead().

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the HIP device pointer.  This pointer is invalid after
       VecHIPRestoreArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHIPGetArrayRead(), VecHIPGetArrayWrite(), VecHIPGetArray(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHIPRestoreArrayRead(Vec v, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHIP,VECMPIHIP);
  *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecHIPGetArrayWrite - Provides write access to the HIP buffer inside a vector.

   The data pointed to by the device pointer is uninitialized.  The user
   may not read from this data.  Furthermore, the entire array needs to
   be filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The device pointer needs to be released with
   VecHIPRestoreArrayWrite().  When the pointer is released the
   host data of the vector is marked as out of data.  Subsequent access
   of the host data with e.g. VecGetArray() incurs a device to host data
   transfer.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the HIP pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecHIPRestoreArrayWrite(), VecHIPGetArray(), VecHIPGetArrayRead(), VecHIPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHIPGetArrayWrite(Vec v, PetscScalar **a)
{
#if defined(PETSC_HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  *a   = 0;
  ierr = VecHIPAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_HIP*)v->spptr)->GPUarray;
#endif
  PetscFunctionReturn(0);
}

/*@C
   VecHIPRestoreArrayWrite - Restore a HIP device pointer previously acquired with VecHIPGetArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the HIP device pointer.  This pointer is invalid after
       VecHIPRestoreArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHIPGetArrayWrite(), VecHIPGetArray(), VecHIPGetArrayRead(), VecHIPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHIPRestoreArrayWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  v->offloadmask = PETSC_OFFLOAD_GPU;
#endif

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecHIPPlaceArray - Allows one to replace the GPU array in a vector with a
   GPU array provided by the user. This is useful to avoid copying an
   array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   You can return to the original GPU array with a call to VecHIPResetArray()
   It is not possible to use VecHIPPlaceArray() and VecPlaceArray() at the
   same time on the same vector.

   Level: developer

.seealso: VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray(), VecHIPResetArray(), VecHIPReplaceArray()

@*/
PetscErrorCode VecHIPPlaceArray(Vec vin,const PetscScalar a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  ierr = VecHIPCopyToGPU(vin);CHKERRQ(ierr);
  if (((Vec_Seq*)vin->data)->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecHIPPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecHIPResetArray()/VecResetArray()");
  ((Vec_Seq*)vin->data)->unplacedarray  = (PetscScalar *) ((Vec_HIP*)vin->spptr)->GPUarray; /* save previous GPU array so reset can bring it back */
  ((Vec_HIP*)vin->spptr)->GPUarray = (PetscScalar*)a;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecHIPReplaceArray - Allows one to replace the GPU array in a vector
   with a GPU array provided by the user. This is useful to avoid copying
   a GPU array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   This permanently replaces the GPU array and frees the memory associated
   with the old GPU array.

   The memory passed in CANNOT be freed by the user. It will be freed
   when the vector is destroyed.

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray(), VecHIPResetArray(), VecHIPPlaceArray(), VecReplaceArray()

@*/
PetscErrorCode VecHIPReplaceArray(Vec vin,const PetscScalar a[])
{
#if defined(PETSC_HAVE_HIP)
  hipError_t err;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  err = hipFree(((Vec_HIP*)vin->spptr)->GPUarray);CHKERRHIP(err);
  ((Vec_HIP*)vin->spptr)->GPUarray = (PetscScalar*)a;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecHIPResetArray - Resets a vector to use its default memory. Call this
   after the use of VecHIPPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray(), VecResetArray(), VecHIPPlaceArray(), VecHIPReplaceArray()

@*/
PetscErrorCode VecHIPResetArray(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQHIP,VECMPIHIP);
#if defined(PETSC_HAVE_HIP)
  ierr = VecHIPCopyToGPU(vin);CHKERRQ(ierr);
  ((Vec_HIP*)vin->spptr)->GPUarray = (PetscScalar *) ((Vec_Seq*)vin->data)->unplacedarray;
  ((Vec_Seq*)vin->data)->unplacedarray = 0;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
    VecDuplicateVecsF90 - Creates several vectors of the same type as an existing vector
    and makes them accessible via a Fortran90 pointer.

    Synopsis:
    VecDuplicateVecsF90(Vec x,PetscInt n,{Vec, pointer :: y(:)},integer ierr)

    Collective on Vec

    Input Parameters:
+   x - a vector to mimic
-   n - the number of vectors to obtain

    Output Parameters:
+   y - Fortran90 pointer to the array of vectors
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

    Notes:
    Not yet supported for all F90 compilers

    Use VecDestroyVecsF90() to free the space.

    Level: beginner

.seealso:  VecDestroyVecsF90(), VecDuplicateVecs()

M*/

/*MC
    VecRestoreArrayF90 - Restores a vector to a usable state after a call to
    VecGetArrayF90().

    Synopsis:
    VecRestoreArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameters:
+   x - vector
-   xx_v - the Fortran90 pointer to the array

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

.seealso:  VecGetArrayF90(), VecGetArray(), VecRestoreArray(), UsingFortran, VecRestoreArrayReadF90()

M*/

/*MC
    VecDestroyVecsF90 - Frees a block of vectors obtained with VecDuplicateVecsF90().

    Synopsis:
    VecDestroyVecsF90(PetscInt n,{Vec, pointer :: x(:)},PetscErrorCode ierr)

    Collective on Vec

    Input Parameters:
+   n - the number of vectors previously obtained
-   x - pointer to array of vector pointers

    Output Parameter:
.   ierr - error code

    Notes:
    Not yet supported for all F90 compilers

    Level: beginner

.seealso:  VecDestroyVecs(), VecDuplicateVecsF90()

M*/

/*MC
    VecGetArrayF90 - Accesses a vector array from Fortran90. For default PETSc
    vectors, VecGetArrayF90() returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call VecRestoreArrayF90()
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
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

    If you ONLY intend to read entries from the array and not change any entries you should use VecGetArrayReadF90().

    Level: beginner

.seealso:  VecRestoreArrayF90(), VecGetArray(), VecRestoreArray(), VecGetArrayReadF90(), UsingFortran

M*/

 /*MC
    VecGetArrayReadF90 - Accesses a read only array from Fortran90. For default PETSc
    vectors, VecGetArrayF90() returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call VecRestoreArrayReadF90()
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayReadF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
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

    If you intend to write entries into the array you must use VecGetArrayF90().

    Level: beginner

.seealso:  VecRestoreArrayReadF90(), VecGetArray(), VecRestoreArray(), VecGetArrayRead(), VecRestoreArrayRead(), VecGetArrayF90(), UsingFortran

M*/

/*MC
    VecRestoreArrayReadF90 - Restores a readonly vector to a usable state after a call to
    VecGetArrayReadF90().

    Synopsis:
    VecRestoreArrayReadF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameters:
+   x - vector
-   xx_v - the Fortran90 pointer to the array

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

.seealso:  VecGetArrayReadF90(), VecGetArray(), VecRestoreArray(), VecGetArrayRead(), VecRestoreArrayRead(),UsingFortran, VecRestoreArrayF90()

M*/

/*@C
   VecGetArray2d - Returns a pointer to a 2d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray2d()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetArray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray2d(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  PetscScalar    *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n != N) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 2d array dimensions %D by %D",N,m,n);
  ierr = VecGetArray(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m,a);CHKERRQ(ierr);
  for (i=0; i<m; i++) (*a)[i] = aa + i*n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray2dWrite - Returns a pointer to a 2d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray2dWrite()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 2d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetArray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray2dWrite(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  PetscScalar    *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n != N) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 2d array dimensions %D by %D",N,m,n);
  ierr = VecGetArrayWrite(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m,a);CHKERRQ(ierr);
  for (i=0; i<m; i++) (*a)[i] = aa + i*n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray2d - Restores a vector after VecGetArray2d() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecRestoreArray2d(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArray(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray2dWrite - Restores a vector after VecGetArray2dWrite() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecRestoreArray2dWrite(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArrayWrite(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray1d - Returns a pointer to a 1d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray1d()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetArray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray2d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray1d(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,4);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local array size %D does not match 1d array dimensions %D",N,m);
  ierr = VecGetArray(x,a);CHKERRQ(ierr);
  *a  -= mstart;
  PetscFunctionReturn(0);
}

 /*@C
   VecGetArray1dWrite - Returns a pointer to a 1d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray1dWrite()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetArray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray2d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray1dWrite(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,4);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local array size %D does not match 1d array dimensions %D",N,m);
  ierr = VecGetArrayWrite(x,a);CHKERRQ(ierr);
  *a  -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray1d - Restores a vector after VecGetArray1d() has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray21()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray1d().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray2d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecRestoreArray1d(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  ierr = VecRestoreArray(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray1dWrite - Restores a vector after VecGetArray1dWrite() has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray21()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray1d().

   This routine actually zeros out the a pointer.

   Concepts: vector^accessing local values as 1d array

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray2d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecRestoreArray1dWrite(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  ierr = VecRestoreArrayWrite(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray3d - Returns a pointer to a 3d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray3d()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
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
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetarray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray3d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N,j;
  PetscScalar    *aa,**b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p != N) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 3d array dimensions %D by %D by %D",N,m,n,p);
  ierr = VecGetArray(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m*sizeof(PetscScalar**)+m*n,a);CHKERRQ(ierr);
  b    = (PetscScalar**)((*a) + m);
  for (i=0; i<m; i++) (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i*n+j] = aa + i*n*p + j*p - pstart;

  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray3dWrite - Returns a pointer to a 3d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray3dWrite()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
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
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 3d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetarray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray3dWrite(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N,j;
  PetscScalar    *aa,**b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p != N) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 3d array dimensions %D by %D by %D",N,m,n,p);
  ierr = VecGetArrayWrite(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m*sizeof(PetscScalar**)+m*n,a);CHKERRQ(ierr);
  b    = (PetscScalar**)((*a) + m);
  for (i=0; i<m; i++) (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i*n+j] = aa + i*n*p + j*p - pstart;

  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray3d - Restores a vector after VecGetArray3d() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode  VecRestoreArray3d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArray(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray3dWrite - Restores a vector after VecGetArray3dWrite() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode  VecRestoreArray3dWrite(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArrayWrite(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray4d - Returns a pointer to a 4d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray4d()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
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
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetarray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray4d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N,j,k;
  PetscScalar    *aa,***b,**c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,10);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p*q != N) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 4d array dimensions %D by %D by %D by %D",N,m,n,p,q);
  ierr = VecGetArray(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m*sizeof(PetscScalar***)+m*n*sizeof(PetscScalar**)+m*n*p,a);CHKERRQ(ierr);
  b    = (PetscScalar***)((*a) + m);
  c    = (PetscScalar**)(b + m*n);
  for (i=0; i<m; i++) (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i*n+j] = c + i*n*p + j*p - pstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      for (k=0; k<p; k++)
        c[i*n*p+j*p+k] = aa + i*n*p*q + j*p*q + k*q - qstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray4dWrite - Returns a pointer to a 4d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray4dWrite()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
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
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 3d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetarray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray4dWrite(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N,j,k;
  PetscScalar    *aa,***b,**c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,10);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p*q != N) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 4d array dimensions %D by %D by %D by %D",N,m,n,p,q);
  ierr = VecGetArrayWrite(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m*sizeof(PetscScalar***)+m*n*sizeof(PetscScalar**)+m*n*p,a);CHKERRQ(ierr);
  b    = (PetscScalar***)((*a) + m);
  c    = (PetscScalar**)(b + m*n);
  for (i=0; i<m; i++) (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i*n+j] = c + i*n*p + j*p - pstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      for (k=0; k<p; k++)
        c[i*n*p+j*p+k] = aa + i*n*p*q + j*p*q + k*q - qstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray4d - Restores a vector after VecGetArray3d() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode  VecRestoreArray4d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArray(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray4dWrite - Restores a vector after VecGetArray3dWrite() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode  VecRestoreArray4dWrite(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArrayWrite(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray2dRead - Returns a pointer to a 2d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray2dRead()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetArray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray2dRead(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode    ierr;
  PetscInt          i,N;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n != N) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 2d array dimensions %D by %D",N,m,n);
  ierr = VecGetArrayRead(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m,a);CHKERRQ(ierr);
  for (i=0; i<m; i++) (*a)[i] = (PetscScalar*) aa + i*n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray2dRead - Restores a vector after VecGetArray2dRead() has been called.

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
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecRestoreArray2dRead(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArrayRead(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray1dRead - Returns a pointer to a 1d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray1dRead()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetArray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray2d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray1dRead(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,4);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local array size %D does not match 1d array dimensions %D",N,m);
  ierr = VecGetArrayRead(x,(const PetscScalar**)a);CHKERRQ(ierr);
  *a  -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray1dRead - Restores a vector after VecGetArray1dRead() has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray21()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray1dRead().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray2d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecRestoreArray1dRead(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  ierr = VecRestoreArrayRead(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
   VecGetArray3dRead - Returns a pointer to a 3d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray3dRead()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
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
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3dRead().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetarray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray3dRead(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode    ierr;
  PetscInt          i,N,j;
  const PetscScalar *aa;
  PetscScalar       **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p != N) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 3d array dimensions %D by %D by %D",N,m,n,p);
  ierr = VecGetArrayRead(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m*sizeof(PetscScalar**)+m*n,a);CHKERRQ(ierr);
  b    = (PetscScalar**)((*a) + m);
  for (i=0; i<m; i++) (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i*n+j] = (PetscScalar *)aa + i*n*p + j*p - pstart;

  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray3dRead - Restores a vector after VecGetArray3dRead() has been called.

   Logically Collective

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray3dRead()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode  VecRestoreArray3dRead(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArrayRead(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray4dRead - Returns a pointer to a 4d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray4dRead()
   when you no longer need access to the array.

   Logically Collective

   Input Parameter:
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
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DMDAVecGetarray(), DMDAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode  VecGetArray4dRead(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode    ierr;
  PetscInt          i,N,j,k;
  const PetscScalar *aa;
  PetscScalar       ***b,**c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,10);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p*q != N) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 4d array dimensions %D by %D by %D by %D",N,m,n,p,q);
  ierr = VecGetArrayRead(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc1(m*sizeof(PetscScalar***)+m*n*sizeof(PetscScalar**)+m*n*p,a);CHKERRQ(ierr);
  b    = (PetscScalar***)((*a) + m);
  c    = (PetscScalar**)(b + m*n);
  for (i=0; i<m; i++) (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i*n+j] = c + i*n*p + j*p - pstart;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      for (k=0; k<p; k++)
        c[i*n*p+j*p+k] = (PetscScalar*) aa + i*n*p*q + j*p*q + k*q - qstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray4dRead - Restores a vector after VecGetArray3d() has been called.

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
-  a - location of pointer to array obtained from VecGetArray4dRead()

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: VecGetArray(), VecRestoreArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(),
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DMDAVecGetArray(), DMDAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode  VecRestoreArray4dRead(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr  = PetscFree(dummy);CHKERRQ(ierr);
  ierr  = VecRestoreArrayRead(x,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)

/*@
   VecLockGet  - Gets the current lock status of a vector

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  state - greater than zero indicates the vector is locked for read; less then zero indicates the vector is
           locked for write; equal to zero means the vector is unlocked, that is, it is free to read or write.

   Level: beginner

.seealso: VecRestoreArray(), VecGetArrayRead(), VecLockReadPush(), VecLockReadPop()
@*/
PetscErrorCode VecLockGet(Vec x,PetscInt *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  *state = x->lock;
  PetscFunctionReturn(0);
}

/*@
   VecLockReadPush  - Pushes a read-only lock on a vector to prevent it from writing

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Notes:
    If this is set then calls to VecGetArray() or VecSetValues() or any other routines that change the vectors values will fail.

    The call can be nested, i.e., called multiple times on the same vector, but each VecLockReadPush(x) has to have one matching
    VecLockReadPop(x), which removes the latest read-only lock.

   Level: beginner

.seealso: VecRestoreArray(), VecGetArrayRead(), VecLockReadPop(), VecLockGet()
@*/
PetscErrorCode VecLockReadPush(Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (x->lock < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Vector is already locked for exclusive write access but you want to read it");
  x->lock++;
  PetscFunctionReturn(0);
}

/*@
   VecLockReadPop  - Pops a read-only lock from a vector

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Level: beginner

.seealso: VecRestoreArray(), VecGetArrayRead(), VecLockReadPush(), VecLockGet()
@*/
PetscErrorCode VecLockReadPop(Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  x->lock--;
  if (x->lock < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Vector has been unlocked from read-only access too many times");
  PetscFunctionReturn(0);
}

/*@C
   VecLockWriteSet_Private  - Lock or unlock a vector for exclusive read/write access

   Logically Collective on Vec

   Input Parameter:
+  x   - the vector
-  flg - PETSC_TRUE to lock the vector for writing; PETSC_FALSE to unlock it.

   Notes:
    The function is usefull in split-phase computations, which usually have a begin phase and an end phase.
    One can call VecLockWriteSet_Private(x,PETSC_TRUE) in the begin phase to lock a vector for exclusive
    access, and call VecLockWriteSet_Private(x,PETSC_FALSE) in the end phase to unlock the vector from exclusive
    access. In this way, one is ensured no other operations can access the vector in between. The code may like


       VecGetArray(x,&xdata); // begin phase
       VecLockWriteSet_Private(v,PETSC_TRUE);

       Other operations, which can not acceess x anymore (they can access xdata, of course)

       VecRestoreArray(x,&vdata); // end phase
       VecLockWriteSet_Private(v,PETSC_FALSE);

    The call can not be nested on the same vector, in other words, one can not call VecLockWriteSet_Private(x,PETSC_TRUE)
    again before calling VecLockWriteSet_Private(v,PETSC_FALSE).

   Level: beginner

.seealso: VecRestoreArray(), VecGetArrayRead(), VecLockReadPush(), VecLockReadPop(), VecLockGet()
@*/
PetscErrorCode VecLockWriteSet_Private(Vec x,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (flg) {
    if (x->lock > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Vector is already locked for read-only access but you want to write it");
    else if (x->lock < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Vector is already locked for exclusive write access but you want to write it");
    else x->lock = -1;
  } else {
    if (x->lock != -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Vector is not locked for exclusive write access but you want to unlock it from that");
    x->lock = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   VecLockPush  - Pushes a read-only lock on a vector to prevent it from writing

   Level: deprecated

.seealso: VecLockReadPush()
@*/
PetscErrorCode VecLockPush(Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecLockReadPush(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecLockPop  - Pops a read-only lock from a vector

   Level: deprecated

.seealso: VecLockReadPop()
@*/
PetscErrorCode VecLockPop(Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecLockReadPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
