#define PETSCVEC_DLL

/*
     Provides the interface functions for vector operations that have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include "private/vecimpl.h"    /*I "petscvec.h" I*/

#define PetscCheckSameSizeVec(x,y) \
  if ((x)->map->N != (y)->map->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths"); \
  if ((x)->map->n != (y)->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");


#undef __FUNCT__  
#define __FUNCT__ "VecMaxPointwiseDivide"
/*@
   VecMaxPointwiseDivide - Computes the maximum of the componentwise division max = max_i abs(x_i/y_i).

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  max - the result

   Level: advanced

   Notes: any subset of the x and may be the same vector.
          if a particular y_i is zero, it is treated as 1 in the above formula

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMax(), VecPointwiseMin(), VecPointwiseMaxAbs()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMaxPointwiseDivide(Vec x,Vec y,PetscReal *max)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidDoublePointer(max,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  PetscCheckSameSizeVec(x,y);

  ierr = (*x->ops->maxpointwisedivide)(x,y,max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDot"
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
   where y^H denotes the conjugate transpose of y.

   Use VecTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Level: intermediate

   Concepts: inner product
   Concepts: vector^inner product

.seealso: VecMDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecDot(Vec x,Vec y,PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  PetscCheckSameSizeVec(x,y);

  ierr = PetscLogEventBarrierBegin(VEC_DotBarrier,x,y,0,0,((PetscObject)x)->comm);CHKERRQ(ierr);
  ierr = (*x->ops->dot)(x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventBarrierEnd(VEC_DotBarrier,x,y,0,0,((PetscObject)x)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNorm"
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

   Concepts: norm
   Concepts: vector^norm

.seealso: VecDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), VecNormAvailable(),
          VecNormBegin(), VecNormEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecNorm(Vec x,NormType type,PetscReal *val)  
{
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidDoublePointer(val,3);
  PetscValidType(x,1);

  /*
   * Cached data?
   */
  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[type],*val,flg);CHKERRQ(ierr);
    if (flg) PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBarrierBegin(VEC_NormBarrier,x,0,0,0,((PetscObject)x)->comm);CHKERRQ(ierr);
  ierr = (*x->ops->norm)(x,type,val);CHKERRQ(ierr);
  ierr = PetscLogEventBarrierEnd(VEC_NormBarrier,x,0,0,0,((PetscObject)x)->comm);CHKERRQ(ierr);

  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[type],*val);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNormAvailable"
/*@
   VecNormAvailable  - Returns the vector norm if it is already known.

   Collective on Vec

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

   Concepts: norm
   Concepts: vector^norm

.seealso: VecDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), VecNorm()
          VecNormBegin(), VecNormEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecNormAvailable(Vec x,NormType type,PetscTruth *available,PetscReal *val)  
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidDoublePointer(val,3);
  PetscValidType(x,1);

  *available = PETSC_FALSE;
  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[type],*val,*available);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNormalize"
/*@
   VecNormalize - Normalizes a vector by 2-norm. 

   Collective on Vec

   Input Parameters:
+  x - the vector

   Output Parameter:
.  x - the normalized vector
-  val - the vector norm before normalization

   Level: intermediate

   Concepts: vector^normalizing
   Concepts: normalizing^vector

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecNormalize(Vec x,PetscReal *val)
{
  PetscErrorCode ierr;
  PetscReal      norm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
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

#undef __FUNCT__  
#define __FUNCT__ "VecMax"
/*@C
   VecMax - Determines the maximum vector component and its location.

   Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameters:
+  val - the maximum component
-  p - the location of val

   Notes:
   Returns the value PETSC_MIN and p = -1 if the vector is of length 0.

   Level: intermediate

   Concepts: maximum^of vector
   Concepts: vector^maximum value

.seealso: VecNorm(), VecMin()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMax(Vec x,PetscInt *p,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_Max,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->max)(x,p,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Max,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMin"
/*@
   VecMin - Determines the minimum vector component and its location.

   Collective on Vec

   Input Parameters:
.  x - the vector

   Output Parameter:
+  val - the minimum component
-  p - the location of val

   Level: intermediate

   Notes:
   Returns the value PETSC_MAX and p = -1 if the vector is of length 0.

   Concepts: minimum^of vector
   Concepts: vector^minimum entry

.seealso: VecMax()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMin(Vec x,PetscInt *p,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_Min,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->min)(x,p,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Min,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot"
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

   Concepts: inner product^non-Hermitian
   Concepts: vector^inner product
   Concepts: non-Hermitian inner product

.seealso: VecDot(), VecMTDot()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecTDot(Vec x,Vec y,PetscScalar *val) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  PetscCheckSameSizeVec(x,y);

  ierr = PetscLogEventBegin(VEC_TDot,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->tdot)(x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_TDot,x,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale"
/*@
   VecScale - Scales a vector. 

   Collective on Vec

   Input Parameters:
+  x - the vector
-  alpha - the scalar

   Output Parameter:
.  x - the scaled vector

   Note:
   For a vector with n components, VecScale() computes 
$      x[i] = alpha * x[i], for i=1,...,n.

   Level: intermediate

   Concepts: vector^scaling
   Concepts: scaling^vector

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecScale (Vec x, PetscScalar alpha)
{
  PetscReal      norms[4] = {0.0,0.0,0.0, 0.0};
  PetscTruth     flgs[4];
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = PetscLogEventBegin(VEC_Scale,x,0,0,0);CHKERRQ(ierr);
  if (alpha != 1.0) {
    ierr = (*x->ops->scale)(x,alpha);CHKERRQ(ierr);

    /*
     * Update cached data
     */
    for (i=0; i<4; i++) {
      ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],norms[i],flgs[i]);CHKERRQ(ierr);
    }

    /* in general we consider this object touched */
    ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);

    for (i=0; i<4; i++) {
      if (flgs[i]) {
        ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[i],PetscAbsScalar(alpha)*norms[i]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(VEC_Scale,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecSet"
/*@
   VecSet - Sets all components of a vector to a single scalar value. 

   Collective on Vec

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

   Concepts: vector^setting to constant

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSet(Vec x,PetscScalar alpha) 
{
  PetscReal      val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"You cannot call this after you have called VecSetValues() but\n before you have called VecAssemblyBegin/End()");
#if defined (PETSC_USE_DEBUG)
 {
   PetscReal alpha_local,alpha_max;
   alpha_local = PetscAbsScalar(alpha);
   ierr = MPI_Allreduce(&alpha_local,&alpha_max,1,MPIU_REAL,MPI_MAX,((PetscObject)x)->comm);CHKERRQ(ierr);
   if (alpha_local != alpha_max) SETERRQ(PETSC_ERR_ARG_WRONG,"Same value should be used across all processors");
 }
#endif

  ierr = PetscLogEventBegin(VEC_Set,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->set)(x,alpha);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Set,x,0,0,0);CHKERRQ(ierr);

  /*
   * Update cached data
   */
  /* in general we consider this object touched */
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);

  /* however, norms can be simply set */
  val = PetscAbsScalar(alpha);
  ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_1],x->map->N * val);CHKERRQ(ierr);
  ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_INFINITY],val);CHKERRQ(ierr);
  val = sqrt((double)x->map->N) * val;
  ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_2],val);CHKERRQ(ierr);
  ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[NORM_FROBENIUS],val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "VecAXPY"
/*@
   VecAXPY - Computes y = alpha x + y. 

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes: x and y must be different vectors

   Concepts: vector^BLAS
   Concepts: BLAS

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAXPY(Vec y,PetscScalar alpha,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidHeaderSpecific(y,VEC_COOKIE,1);
  PetscValidType(x,3);
  PetscValidType(y,1);
  PetscCheckSameTypeAndComm(x,3,y,1);
  PetscCheckSameSizeVec(x,y);

  ierr = PetscLogEventBegin(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->axpy)(y,alpha,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY"
/*@
   VecAXPBY - Computes y = alpha x + beta y. 

   Collective on Vec

   Input Parameters:
+  alpha,beta - the scalars
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes: x and y must be different vectors 

   Concepts: BLAS
   Concepts: vector^BLAS

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAXPBY(Vec y,PetscScalar alpha,PetscScalar beta,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,4);
  PetscValidHeaderSpecific(y,VEC_COOKIE,1);
  PetscValidType(x,4);
  PetscValidType(y,1);
  PetscCheckSameTypeAndComm(x,4,y,1);
  PetscCheckSameSizeVec(x,y);

  ierr = PetscLogEventBegin(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->axpby)(y,alpha,beta,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBYPCZ"
/*@
   VecAXPBYPCZ - Computes z = alpha x + beta y + gamma z

   Collective on Vec

   Input Parameters:
+  alpha,beta, gamma - the scalars
-  x, y, z  - the vectors

   Output Parameter:
.  z - output vector

   Level: intermediate

   Notes: x, y and z must be different vectors 

          alpha = 1 or gamma = 1 are handled as special cases

   Concepts: BLAS
   Concepts: vector^BLAS

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAXPBYPCZ(Vec z,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,5);
  PetscValidHeaderSpecific(y,VEC_COOKIE,6);
  PetscValidHeaderSpecific(z,VEC_COOKIE,1);
  PetscValidType(x,5);
  PetscValidType(y,6);
  PetscValidType(z,1);
  PetscCheckSameTypeAndComm(x,5,y,6);
  PetscCheckSameTypeAndComm(x,5,z,1);
  PetscCheckSameSizeVec(x,y);
  PetscCheckSameSizeVec(x,z);

  ierr = PetscLogEventBegin(VEC_AXPBYPCZ,x,y,z,0);CHKERRQ(ierr);
  ierr = (*y->ops->axpbypcz)(z,alpha,beta,gamma,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPBYPCZ,x,y,z,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecAYPX"
/*@
   VecAYPX - Computes y = x + alpha y.

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes: x and y must be different vectors

   Concepts: vector^BLAS
   Concepts: BLAS

.seealso: VecAXPY(), VecWAXPY()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAYPX(Vec y,PetscScalar alpha,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,3); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,1);
  PetscValidType(x,3);
  PetscValidType(y,1);

  ierr = PetscLogEventBegin(VEC_AYPX,x,y,0,0);CHKERRQ(ierr);
  ierr =  (*y->ops->aypx)(y,alpha,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AYPX,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY"
/*@
   VecWAXPY - Computes w = alpha x + y.

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: intermediate

   Notes: Neither the vector x or y can be the same as vector w

   Concepts: vector^BLAS
   Concepts: BLAS

.seealso: VecAXPY(), VecAYPX(), VecAXPBY()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecWAXPY(Vec w,PetscScalar alpha,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,4);
  PetscValidType(w,1);
  PetscValidType(x,3);
  PetscValidType(y,4);
  PetscCheckSameTypeAndComm(x,3,y,4); 
  PetscCheckSameTypeAndComm(y,4,w,1);
  PetscCheckSameSizeVec(x,y);
  PetscCheckSameSizeVec(x,w);
  if (w == y) SETERRQ(PETSC_ERR_SUP,"Result vector w cannot be same as input vector y, suggest VecAXPY()");
  if (w == x) SETERRQ(PETSC_ERR_SUP,"Result vector w cannot be same as input vector x, suggest VecAYPX()");

  ierr = PetscLogEventBegin(VEC_WAXPY,x,y,w,0);CHKERRQ(ierr);
  ierr =  (*w->ops->waxpy)(w,alpha,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_WAXPY,x,y,w,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecSetValues"
/*@
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

   Concepts: vector^setting values

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesLocal(),
           VecSetValue(), VecSetValuesBlocked(), InsertMode, INSERT_VALUES, ADD_VALUES, VecGetValues()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetValues(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setvalues)(x,ni,ix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetValues"
/*@
   VecGetValues - Gets values from certain locations of a vector. Currently 
          can only get values on the same processor

    Collective on Vec
 
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

   Concepts: vector^getting values

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecGetValuesLocal(),
           VecGetValuesBlocked(), InsertMode, INSERT_VALUES, ADD_VALUES, VecSetValues()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetValues(Vec x,PetscInt ni,const PetscInt ix[],PetscScalar y[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  ierr = (*x->ops->getvalues)(x,ni,ix,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesBlocked"
/*@
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
   for j=0,...,bs, for i=0,...,ni-1. where bs was set with VecSetBlockSize().

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

   Concepts: vector^setting values blocked

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesBlockedLocal(),
           VecSetValues()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetValuesBlocked(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)(x,ni,ix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesLocal"
/*@
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

   Concepts: vector^setting values with local numbering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetLocalToGlobalMapping(),
           VecSetValuesBlockedLocal()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetValuesLocal(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora) 
{
  PetscErrorCode ierr;
  PetscInt       lixp[128],*lix = lixp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  if (!x->ops->setvalueslocal) {
    if (!x->mapping) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with VecSetLocalToGlobalMapping()");
    }
    if (ni > 128) {
      ierr = PetscMalloc(ni*sizeof(PetscInt),&lix);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingApply(x->mapping,ni,(PetscInt*)ix,lix);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesBlockedLocal"
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


   Concepts: vector^setting values blocked with local numbering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesBlocked(), 
           VecSetLocalToGlobalMappingBlock()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetValuesBlockedLocal(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora) 
{
  PetscErrorCode ierr;
  PetscInt       lixp[128],*lix = lixp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  if (!x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with VecSetLocalToGlobalMappingBlock()");
  }
  if (ni > 128) {
    ierr = PetscMalloc(ni*sizeof(PetscInt),&lix);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(x->bmapping,ni,(PetscInt*)ix,lix);CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)(x,ni,lix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  if (ni > 128) {
    ierr = PetscFree(lix);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "VecMTDot"
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

   Concepts: inner product^multiple
   Concepts: vector^multiple inner products

.seealso: VecMDot(), VecTDot()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMTDot(Vec x,PetscInt nv,const Vec y[],PetscScalar val[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(y,3);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,3);
  PetscValidScalarPointer(val,4);
  PetscValidType(x,2);
  PetscValidType(*y,3);
  PetscCheckSameTypeAndComm(x,2,*y,3);
  PetscCheckSameSizeVec(x,*y);

  ierr = PetscLogEventBegin(VEC_MTDot,x,*y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mtdot)(x,nv,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MTDot,x,*y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot"
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

   Concepts: inner product^multiple
   Concepts: vector^multiple inner products

.seealso: VecMTDot(), VecDot()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMDot(Vec x,PetscInt nv,const Vec y[],PetscScalar val[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  if (!nv) PetscFunctionReturn(0);
  if (nv < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",nv);
  PetscValidPointer(y,3);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,3);
  PetscValidScalarPointer(val,4);
  PetscValidType(x,2);
  PetscValidType(*y,3);
  PetscCheckSameTypeAndComm(x,2,*y,3);
  PetscCheckSameSizeVec(x,*y);

  ierr = PetscLogEventBarrierBegin(VEC_MDotBarrier,x,*y,0,0,((PetscObject)x)->comm);CHKERRQ(ierr);
  ierr = (*x->ops->mdot)(x,nv,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventBarrierEnd(VEC_MDotBarrier,x,*y,0,0,((PetscObject)x)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY"
/*@
   VecMAXPY - Computes y = y + sum alpha[j] x[j]

   Collective on Vec

   Input Parameters:
+  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  y - one vector
-  x - array of vectors

   Level: intermediate

   Concepts: BLAS

.seealso: VecAXPY(), VecWAXPY(), VecAYPX()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMAXPY(Vec y,PetscInt nv,const PetscScalar alpha[],Vec x[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(y,VEC_COOKIE,1);
  if (!nv) PetscFunctionReturn(0);
  if (nv < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",nv);
  PetscValidScalarPointer(alpha,3);
  PetscValidPointer(x,4);
  PetscValidHeaderSpecific(*x,VEC_COOKIE,4);
  PetscValidType(y,1);
  PetscValidType(*x,4);
  PetscCheckSameTypeAndComm(y,1,*x,4);
  PetscCheckSameSizeVec(y,*x);

  ierr = PetscLogEventBegin(VEC_MAXPY,*x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->maxpy)(y,nv,alpha,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MAXPY,*x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

/*MC
   VecGetArray - Returns a pointer to a contiguous array that contains this 
   processor's portion of the vector data. For the standard PETSc
   vectors, VecGetArray() returns a pointer to the local data array and
   does not use any copies. If the underlying vector data is not stored
   in a contiquous array this routine will copy the data to a contiquous
   array and return a pointer to that. You MUST call VecRestoreArray() 
   when you no longer need access to the array.

   Synopsis:
   PetscErrorCode VecGetArray(Vec x,PetscScalar *a[])

   Not Collective

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
   petsc/src/snes/examples/tutorials/ex5f.F for details.

   Level: beginner

   Concepts: vector^accessing local values

.seealso: VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(), VecGetArray2d()
M*/
#undef __FUNCT__  
#define __FUNCT__ "VecGetArray_Private"
PetscErrorCode VecGetArray_Private(Vec x,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getarray)(x,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecGetArrays" 
/*@C
   VecGetArrays - Returns a pointer to the arrays in a set of vectors
   that were created by a call to VecDuplicateVecs().  You MUST call
   VecRestoreArrays() when you no longer need access to the array.

   Not Collective

   Input Parameter:
+  x - the vectors
-  n - the number of vectors

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is not supported in Fortran.

   Level: intermediate

.seealso: VecGetArray(), VecRestoreArrays()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetArrays(const Vec x[],PetscInt n,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    **q;

  PetscFunctionBegin;
  PetscValidPointer(x,1);
  PetscValidHeaderSpecific(*x,VEC_COOKIE,1);
  PetscValidPointer(a,3);
  if (n <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Must get at least one array n = %D",n);
  ierr = PetscMalloc(n*sizeof(PetscScalar*),&q);CHKERRQ(ierr);
  for (i=0; i<n; ++i) {
    ierr = VecGetArray(x[i],&q[i]);CHKERRQ(ierr);
  }
  *a = q;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArrays"
/*@C
   VecRestoreArrays - Restores a group of vectors after VecGetArrays()
   has been called.

   Not Collective

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
PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArrays(const Vec x[],PetscInt n,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    **q = *a;

  PetscFunctionBegin;
  PetscValidPointer(x,1);
  PetscValidHeaderSpecific(*x,VEC_COOKIE,1);
  PetscValidPointer(a,3);

  for(i=0;i<n;++i) {
    ierr = VecRestoreArray(x[i],&q[i]);CHKERRQ(ierr);
 }
  ierr = PetscFree(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VecRestoreArray - Restores a vector after VecGetArray() has been called.

   Synopsis:
   PetscErrorCode VecRestoreArray(Vec x,PetscScalar *a[])

   Not Collective

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArray()

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying 
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer. This is to prevent accidental
   us of the array after it has been restored. If you pass null for a it will 
   not zero the array pointer a.

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

   See the Fortran chapter of the users manual and 
   petsc/src/snes/examples/tutorials/ex5f.F for details.
   For Fortran 90 see VecRestoreArrayF90()

.seealso: VecGetArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray(), VecRestoreArray2d()
M*/
#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray_Private"
PetscErrorCode VecRestoreArray_Private(Vec x,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  if (a) PetscValidPointer(a,2);
  PetscValidType(x,1);
#if defined(PETSC_USE_DEBUG)
  CHKMEMQ;
#endif
  if (x->ops->restorearray) {
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecPlaceArray"
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
PetscErrorCode PETSCVEC_DLLEXPORT VecPlaceArray(Vec vec,const PetscScalar array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (array) PetscValidScalarPointer(array,2);
  if (vec->ops->placearray) {
    ierr = (*vec->ops->placearray)(vec,array);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Cannot place array in this type of vector");
  }
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecReplaceArray"
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
   freed by the user. It will be freed when the vector is destroy. 

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecReplaceArray(Vec vec,const PetscScalar array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (vec->ops->replacearray) {
    ierr = (*vec->ops->replacearray)(vec,array);CHKERRQ(ierr);
 } else {
    SETERRQ(PETSC_ERR_SUP,"Cannot replace array in this type of vector");
  }
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
    VecDuplicateVecsF90 - Creates several vectors of the same type as an existing vector
    and makes them accessible via a Fortran90 pointer.

    Synopsis:
    VecDuplicateVecsF90(Vec x,int n,{Vec, pointer :: y(:)},integer ierr)

    Collective on Vec

    Input Parameters:
+   x - a vector to mimic
-   n - the number of vectors to obtain

    Output Parameters:
+   y - Fortran90 pointer to the array of vectors
-   ierr - error code

    Example of Usage: 
.vb
    Vec x
    Vec, pointer :: y(:)
    ....
    call VecDuplicateVecsF90(x,2,y,ierr)
    call VecSet(y(2),alpha,ierr)
    call VecSet(y(2),alpha,ierr)
    ....
    call VecDestroyVecsF90(y,2,ierr)
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

    Not collective

    Input Parameters:
+   x - vector
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage: 
.vb
    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve
   
    Level: beginner

.seealso:  VecGetArrayF90(), VecGetArray(), VecRestoreArray(), UsingFortran

M*/

/*MC
    VecDestroyVecsF90 - Frees a block of vectors obtained with VecDuplicateVecsF90().

    Synopsis:
    VecDestroyVecsF90({Vec, pointer :: x(:)},integer n,integer ierr)

    Collective on Vec

    Input Parameters:
+   x - pointer to array of vector pointers
-   n - the number of vectors previously obtained

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

    Not Collective 

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage: 
.vb
    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: beginner

.seealso:  VecRestoreArrayF90(), VecGetArray(), VecRestoreArray(), UsingFortran

M*/


#undef __FUNCT__  
#define __FUNCT__ "VecGetArray2d"
/*@C
   VecGetArray2d - Returns a pointer to a 2d contiguous array that contains this 
   processor's portion of the vector data.  You MUST call VecRestoreArray2d() 
   when you no longer need access to the array.

   Not Collective

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
   For a vector obtained from DACreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DAGetGhostCorners() while for
   DACreateGlobalVector() they are the corner indices from DAGetCorners(). In both cases
   the arguments from DAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().
   
   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 2d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DAVecGetArray(), DAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray2d(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  PetscScalar    *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n != N) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 2d array dimensions %D by %D",N,m,n);
  ierr = VecGetArray(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc(m*sizeof(PetscScalar*),a);CHKERRQ(ierr);
  for (i=0; i<m; i++) (*a)[i] = aa + i*n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray2d"
/*@C
   VecRestoreArray2d - Restores a vector after VecGetArray2d() has been called.

   Not Collective

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
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DAVecGetArray(), DAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray2d(Vec x,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetArray1d"
/*@C
   VecGetArray1d - Returns a pointer to a 1d contiguous array that contains this 
   processor's portion of the vector data.  You MUST call VecRestoreArray1d() 
   when you no longer need access to the array.

   Not Collective

   Input Parameter:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DACreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DAGetGhostCorners() while for
   DACreateGlobalVector() they are the corner indices from DAGetCorners(). 
   
   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DAVecGetArray(), DAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray2d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray1d(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,4);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m != N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local array size %D does not match 1d array dimensions %D",N,m);
  ierr = VecGetArray(x,a);CHKERRQ(ierr);
  *a  -= mstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray1d"
/*@C
   VecRestoreArray1d - Restores a vector after VecGetArray1d() has been called.

   Not Collective

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
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DAVecGetArray(), DAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray2d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray1d(Vec x,PetscInt m,PetscInt mstart,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  ierr = VecRestoreArray(x,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecGetArray3d"
/*@C
   VecGetArray3d - Returns a pointer to a 3d contiguous array that contains this 
   processor's portion of the vector data.  You MUST call VecRestoreArray3d() 
   when you no longer need access to the array.

   Not Collective

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
   For a vector obtained from DACreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DAGetGhostCorners() while for
   DACreateGlobalVector() they are the corner indices from DAGetCorners(). In both cases
   the arguments from DAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().
   
   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 3d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DAVecGetarray(), DAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray3d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N,j;
  PetscScalar    *aa,**b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p != N) SETERRQ4(PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 3d array dimensions %D by %D by %D",N,m,n,p);
  ierr = VecGetArray(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc(m*sizeof(PetscScalar**)+m*n*sizeof(PetscScalar*),a);CHKERRQ(ierr);
  b    = (PetscScalar **)((*a) + m);
  for (i=0; i<m; i++)   (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      b[i*n+j] = aa + i*n*p + j*p - pstart;
    }
  }
  *a -= mstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray3d"
/*@C
   VecRestoreArray3d - Restores a vector after VecGetArray3d() has been called.

   Not Collective

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
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DAVecGetArray(), DAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray3d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscScalar ***a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetArray4d"
/*@C
   VecGetArray4d - Returns a pointer to a 4d contiguous array that contains this 
   processor's portion of the vector data.  You MUST call VecRestoreArray4d() 
   when you no longer need access to the array.

   Not Collective

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
   For a vector obtained from DACreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DAGetGhostCorners() while for
   DACreateGlobalVector() they are the corner indices from DAGetCorners(). In both cases
   the arguments from DAGet[Ghost}Corners() are reversed in the call to VecGetArray3d().
   
   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 3d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DAVecGetarray(), DAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray4d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  PetscInt       i,N,j,k;
  PetscScalar    *aa,***b,**c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,10);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p*q != N) SETERRQ5(PETSC_ERR_ARG_INCOMP,"Local array size %D does not match 4d array dimensions %D by %D by %D by %D",N,m,n,p,q);
  ierr = VecGetArray(x,&aa);CHKERRQ(ierr);

  ierr = PetscMalloc(m*sizeof(PetscScalar***)+m*n*sizeof(PetscScalar**)+m*n*p*sizeof(PetscScalar*),a);CHKERRQ(ierr);
  b    = (PetscScalar ***)((*a) + m);
  c    = (PetscScalar **)(b + m*n);
  for (i=0; i<m; i++)   (*a)[i] = b + i*n - nstart;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      b[i*n+j] = c + i*n*p + j*p - pstart;
    }
  }
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      for (k=0; k<p; k++) {
        c[i*n*p+j*p+k] = aa + i*n*p*q + j*p*q + k*q - qstart;
      }
    }
  }
  *a -= mstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray4d"
/*@C
   VecRestoreArray4d - Restores a vector after VecGetArray3d() has been called.

   Not Collective

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
          VecGetArray2d(), VecGetArray3d(), VecRestoreArray3d(), DAVecGetArray(), DAVecRestoreArray()
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d(), VecGet
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray4d(Vec x,PetscInt m,PetscInt n,PetscInt p,PetscInt q,PetscInt mstart,PetscInt nstart,PetscInt pstart,PetscInt qstart,PetscScalar ****a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  dummy = (void*)(*a + mstart);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

