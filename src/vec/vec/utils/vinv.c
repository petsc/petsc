
/*
     Some useful vector utility functions.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>          /*I "petscvec.h" I*/

/*@
   VecStrideSet - Sets a subvector of a vector defined
   by a starting point and a stride with a given value

   Logically Collective on Vec

   Input Parameters:
+  v - the vector
.  start - starting point of the subvector (defined by a stride)
-  s - value to set for each entry in that subvector

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   This will only work if the desire subvector is a stride subvector

   Level: advanced

.seealso: `VecNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideScale()`
@*/
PetscErrorCode  VecStrideSet(Vec v,PetscInt start,PetscScalar s)
{
  PetscInt       i,n,bs;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n  Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,bs);
  PetscCall(VecGetArray(v,&x));
  for (i=start; i<n; i+=bs) x[i] = s;
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideScale - Scales a subvector of a vector defined
   by a starting point and a stride.

   Logically Collective on Vec

   Input Parameters:
+  v - the vector
.  start - starting point of the subvector (defined by a stride)
-  scale - value to multiply each subvector entry by

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   This will only work if the desire subvector is a stride subvector

   Level: advanced

.seealso: `VecNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideScale()`
@*/
PetscErrorCode  VecStrideScale(Vec v,PetscInt start,PetscScalar scale)
{
  PetscInt       i,n,bs;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n  Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,bs);
  PetscCall(VecGetArray(v,&x));
  for (i=start; i<n; i+=bs) x[i] *= scale;
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideNorm - Computes the norm of subvector of a vector defined
   by a starting point and a stride.

   Collective on Vec

   Input Parameters:
+  v - the vector
.  start - starting point of the subvector (defined by a stride)
-  ntype - type of norm, one of NORM_1, NORM_2, NORM_INFINITY

   Output Parameter:
.  norm - the norm

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this computes the norm
   of the array (x[start],x[start+stride],x[start+2*stride], ....)

   This is useful for computing, say the norm of the pressure variable when
   the pressure is stored (interlaced) with other variables, say density etc.

   This will only work if the desire subvector is a stride subvector

   Level: advanced

.seealso: `VecNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideNorm(Vec v,PetscInt start,NormType ntype,PetscReal *nrm)
{
  PetscInt          i,n,bs;
  const PetscScalar *x;
  PetscReal         tnorm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(v,ntype,3);
  PetscValidRealPointer(nrm,4);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,bs);
  PetscCall(VecGetArrayRead(v,&x));
  if (ntype == NORM_2) {
    PetscScalar sum = 0.0;
    for (i=start; i<n; i+=bs) sum += x[i]*(PetscConj(x[i]));
    tnorm = PetscRealPart(sum);
    PetscCall(MPIU_Allreduce(&tnorm,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)v)));
    *nrm  = PetscSqrtReal(*nrm);
  } else if (ntype == NORM_1) {
    tnorm = 0.0;
    for (i=start; i<n; i+=bs) tnorm += PetscAbsScalar(x[i]);
    PetscCall(MPIU_Allreduce(&tnorm,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)v)));
  } else if (ntype == NORM_INFINITY) {
    tnorm = 0.0;
    for (i=start; i<n; i+=bs) {
      if (PetscAbsScalar(x[i]) > tnorm) tnorm = PetscAbsScalar(x[i]);
    }
    PetscCall(MPIU_Allreduce(&tnorm,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)v)));
  } else SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown norm type");
  PetscCall(VecRestoreArrayRead(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideMax - Computes the maximum of subvector of a vector defined
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameters:
+  v - the vector
-  start - starting point of the subvector (defined by a stride)

   Output Parameters:
+  idex - the location where the maximum occurred  (pass NULL if not required)
-  nrm - the maximum value in the subvector

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If xa is the array representing the vector x, then this computes the max
   of the array (xa[start],xa[start+stride],xa[start+2*stride], ....)

   This is useful for computing, say the maximum of the pressure variable when
   the pressure is stored (interlaced) with other variables, e.g., density, etc.
   This will only work if the desire subvector is a stride subvector.

   Level: advanced

.seealso: `VecMax()`, `VecStrideNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`
@*/
PetscErrorCode  VecStrideMax(Vec v,PetscInt start,PetscInt *idex,PetscReal *nrm)
{
  PetscInt          i,n,bs,id = -1;
  const PetscScalar *x;
  PetscReal         max = PETSC_MIN_REAL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidRealPointer(nrm,4);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,bs);
  PetscCall(VecGetArrayRead(v,&x));
  for (i=start; i<n; i+=bs) {
    if (PetscRealPart(x[i]) > max) { max = PetscRealPart(x[i]); id = i;}
  }
  PetscCall(VecRestoreArrayRead(v,&x));
#if defined(PETSC_HAVE_MPIUNI)
  *nrm = max;
  if (idex) *idex = id;
#else
  if (!idex) {
    PetscCall(MPIU_Allreduce(&max,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)v)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;
    PetscInt rstart;

    PetscCall(VecGetOwnershipRange(v,&rstart,NULL));
    in.v  = max;
    in.i  = rstart+id;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)v)));
    *nrm  = out.v;
    *idex = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

/*@
   VecStrideMin - Computes the minimum of subvector of a vector defined
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameters:
+  v - the vector
-  start - starting point of the subvector (defined by a stride)

   Output Parameters:
+  idex - the location where the minimum occurred. (pass NULL if not required)
-  nrm - the minimum value in the subvector

   Level: advanced

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If xa is the array representing the vector x, then this computes the min
   of the array (xa[start],xa[start+stride],xa[start+2*stride], ....)

   This is useful for computing, say the minimum of the pressure variable when
   the pressure is stored (interlaced) with other variables, e.g., density, etc.
   This will only work if the desire subvector is a stride subvector.

.seealso: `VecMin()`, `VecStrideNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideMin(Vec v,PetscInt start,PetscInt *idex,PetscReal *nrm)
{
  PetscInt          i,n,bs,id = -1;
  const PetscScalar *x;
  PetscReal         min = PETSC_MAX_REAL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidRealPointer(nrm,4);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\nHave you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,bs);
  PetscCall(VecGetArrayRead(v,&x));
  for (i=start; i<n; i+=bs) {
    if (PetscRealPart(x[i]) < min) { min = PetscRealPart(x[i]); id = i;}
  }
  PetscCall(VecRestoreArrayRead(v,&x));
#if defined(PETSC_HAVE_MPIUNI)
  *nrm = min;
  if (idex) *idex = id;
#else
  if (!idex) {
    PetscCall(MPIU_Allreduce(&min,nrm,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)v)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;
    PetscInt rstart;

    PetscCall(VecGetOwnershipRange(v,&rstart,NULL));
    in.v  = min;
    in.i  = rstart+id;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)v)));
    *nrm  = out.v;
    *idex = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

/*@
   VecStrideSum - Computes the sum of subvector of a vector defined
   by a starting point and a stride.

   Collective on Vec

   Input Parameters:
+  v - the vector
.  start - starting point of the subvector (defined by a stride)

   Output Parameter:
.  sum - the sum

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this computes the sum
   of the array (x[start],x[start+stride],x[start+2*stride], ....)

   Level: advanced

.seealso: `VecSum()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideSum(Vec v,PetscInt start,PetscScalar *sum)
{
  PetscInt          i,n,bs;
  const PetscScalar *x;
  PetscScalar       local_sum = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(sum,3);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,bs);
  PetscCall(VecGetArrayRead(v,&x));
  for (i=start; i<n; i+=bs) local_sum += x[i];
  PetscCall(MPIU_Allreduce(&local_sum,sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)v)));
  PetscCall(VecRestoreArrayRead(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideScaleAll - Scales the subvectors of a vector defined
   by a starting point and a stride.

   Logically Collective on Vec

   Input Parameters:
+  v - the vector
-  scales - values to multiply each subvector entry by

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   The dimension of scales must be the same as the vector block size

   Level: advanced

.seealso: `VecNorm()`, `VecStrideScale()`, `VecScale()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideScaleAll(Vec v,const PetscScalar *scales)
{
  PetscInt       i,j,n,bs;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(scales,2);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCall(VecGetArray(v,&x));
  /* need to provide optimized code for each bs */
  for (i=0; i<n; i+=bs) {
    for (j=0; j<bs; j++) x[i+j] *= scales[j];
  }
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideNormAll - Computes the norms of subvectors of a vector defined
   by a starting point and a stride.

   Collective on Vec

   Input Parameters:
+  v - the vector
-  ntype - type of norm, one of NORM_1, NORM_2, NORM_INFINITY

   Output Parameter:
.  nrm - the norms

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this computes the norm
   of the array (x[start],x[start+stride],x[start+2*stride], ....) for each start < stride

   The dimension of nrm must be the same as the vector block size

   This will only work if the desire subvector is a stride subvector

   Level: advanced

.seealso: `VecNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideNormAll(Vec v,NormType ntype,PetscReal nrm[])
{
  PetscInt          i,j,n,bs;
  const PetscScalar *x;
  PetscReal         tnorm[128];
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(v,ntype,2);
  PetscValidRealPointer(nrm,3);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(PetscObjectGetComm((PetscObject)v,&comm));

  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(bs <= 128,comm,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  if (ntype == NORM_2) {
    PetscScalar sum[128];
    for (j=0; j<bs; j++) sum[j] = 0.0;
    for (i=0; i<n; i+=bs) {
      for (j=0; j<bs; j++) sum[j] += x[i+j]*(PetscConj(x[i+j]));
    }
    for (j=0; j<bs; j++) tnorm[j]  = PetscRealPart(sum[j]);

    PetscCall(MPIU_Allreduce(tnorm,nrm,bs,MPIU_REAL,MPIU_SUM,comm));
    for (j=0; j<bs; j++) nrm[j] = PetscSqrtReal(nrm[j]);
  } else if (ntype == NORM_1) {
    for (j=0; j<bs; j++) tnorm[j] = 0.0;

    for (i=0; i<n; i+=bs) {
      for (j=0; j<bs; j++) tnorm[j] += PetscAbsScalar(x[i+j]);
    }

    PetscCall(MPIU_Allreduce(tnorm,nrm,bs,MPIU_REAL,MPIU_SUM,comm));
  } else if (ntype == NORM_INFINITY) {
    PetscReal tmp;
    for (j=0; j<bs; j++) tnorm[j] = 0.0;

    for (i=0; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
        if ((tmp = PetscAbsScalar(x[i+j])) > tnorm[j]) tnorm[j] = tmp;
        /* check special case of tmp == NaN */
        if (tmp != tmp) {tnorm[j] = tmp; break;}
      }
    }
    PetscCall(MPIU_Allreduce(tnorm,nrm,bs,MPIU_REAL,MPIU_MAX,comm));
  } else SETERRQ(comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown norm type");
  PetscCall(VecRestoreArrayRead(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideMaxAll - Computes the maximums of subvectors of a vector defined
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameters:
+  index - the location where the maximum occurred (not supported, pass NULL,
           if you need this, send mail to petsc-maint@mcs.anl.gov to request it)
-  nrm - the maximum values of each subvector

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   The dimension of nrm must be the same as the vector block size

   Level: advanced

.seealso: `VecMax()`, `VecStrideNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`
@*/
PetscErrorCode  VecStrideMaxAll(Vec v,PetscInt idex[],PetscReal nrm[])
{
  PetscInt          i,j,n,bs;
  const PetscScalar *x;
  PetscReal         max[128],tmp;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidRealPointer(nrm,3);
  PetscCheck(!idex,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for returning index; send mail to petsc-maint@mcs.anl.gov asking for it");
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(PetscObjectGetComm((PetscObject)v,&comm));

  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(bs <= 128,comm,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  if (!n) {
    for (j=0; j<bs; j++) max[j] = PETSC_MIN_REAL;
  } else {
    for (j=0; j<bs; j++) max[j] = PetscRealPart(x[j]);

    for (i=bs; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
        if ((tmp = PetscRealPart(x[i+j])) > max[j]) max[j] = tmp;
      }
    }
  }
  PetscCall(MPIU_Allreduce(max,nrm,bs,MPIU_REAL,MPIU_MAX,comm));

  PetscCall(VecRestoreArrayRead(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideMinAll - Computes the minimum of subvector of a vector defined
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameters:
+  idex - the location where the minimum occurred (not supported, pass NULL,
           if you need this, send mail to petsc-maint@mcs.anl.gov to request it)
-  nrm - the minimums of each subvector

   Level: advanced

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   The dimension of nrm must be the same as the vector block size

.seealso: `VecMin()`, `VecStrideNorm()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideMinAll(Vec v,PetscInt idex[],PetscReal nrm[])
{
  PetscInt          i,n,bs,j;
  const PetscScalar *x;
  PetscReal         min[128],tmp;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidRealPointer(nrm,3);
  PetscCheck(!idex,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for returning index; send mail to petsc-maint@mcs.anl.gov asking for it");
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(PetscObjectGetComm((PetscObject)v,&comm));

  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(bs <= 128,comm,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  if (!n) {
    for (j=0; j<bs; j++) min[j] = PETSC_MAX_REAL;
  } else {
    for (j=0; j<bs; j++) min[j] = PetscRealPart(x[j]);

    for (i=bs; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
        if ((tmp = PetscRealPart(x[i+j])) < min[j]) min[j] = tmp;
      }
    }
  }
  PetscCall(MPIU_Allreduce(min,nrm,bs,MPIU_REAL,MPIU_MIN,comm));

  PetscCall(VecRestoreArrayRead(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecStrideSumAll - Computes the sums of subvectors of a vector defined
   by a starting point and a stride.

   Collective on Vec

   Input Parameters:
+  v - the vector

   Output Parameter:
.  sums - the sums

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this computes the sum
   of the array (x[start],x[start+stride],x[start+2*stride], ....) for each start < stride

   Level: advanced

.seealso: `VecSum()`, `VecStrideGather()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`
@*/
PetscErrorCode  VecStrideSumAll(Vec v,PetscScalar sums[])
{
  PetscInt          i,j,n,bs;
  const PetscScalar *x;
  PetscScalar       local_sums[128];
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(sums,2);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(PetscObjectGetComm((PetscObject)v,&comm));

  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(bs <= 128,comm,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  for (j=0; j<bs; j++) local_sums[j] = 0.0;
  for (i=0; i<n; i+=bs) {
    for (j=0; j<bs; j++) {
      local_sums[j] += x[i+j];
    }
  }
  PetscCall(MPIU_Allreduce(local_sums,sums,bs,MPIU_SCALAR,MPIU_SUM,comm));

  PetscCall(VecRestoreArrayRead(v,&x));
  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------------------------------------*/
/*@
   VecStrideGatherAll - Gathers all the single components from a multi-component vector into
   separate vectors.

   Collective on Vec

   Input Parameters:
+  v - the vector
-  addv - one of ADD_VALUES,INSERT_VALUES,MAX_VALUES

   Output Parameter:
.  s - the location where the subvectors are stored

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this gathers
   the arrays (x[start],x[start+stride],x[start+2*stride], ....)
   for start=0,1,2,...bs-1

   The parallel layout of the vector and the subvector must be the same;
   i.e., nlocal of v = stride*(nlocal of s)

   Not optimized; could be easily

   Level: advanced

.seealso: `VecStrideNorm()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideGather()`,
          `VecStrideScatterAll()`
@*/
PetscErrorCode  VecStrideGatherAll(Vec v,Vec s[],InsertMode addv)
{
  PetscInt          i,n,n2,bs,j,k,*bss = NULL,nv,jj,nvc;
  PetscScalar       **y;
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(s,2);
  PetscValidHeaderSpecific(*s,VEC_CLASSID,2);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetLocalSize(s[0],&n2));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(bs > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input vector does not have a valid blocksize set");

  PetscCall(PetscMalloc2(bs,&y,bs,&bss));
  nv   = 0;
  nvc  = 0;
  for (i=0; i<bs; i++) {
    PetscCall(VecGetBlockSize(s[i],&bss[i]));
    if (bss[i] < 1) bss[i] = 1; /* if user never set it then assume 1  Re: [PETSC #8241] VecStrideGatherAll */
    PetscCall(VecGetArray(s[i],&y[i]));
    nvc += bss[i];
    nv++;
    PetscCheck(nvc <= bs,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of subvectors in subvectors > number of vectors in main vector");
    if (nvc == bs) break;
  }

  n =  n/bs;

  jj = 0;
  if (addv == INSERT_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
        for (i=0; i<n; i++) y[j][i*bss[j] + k] = x[bs*i+jj+k];
      }
      jj += bss[j];
    }
  } else if (addv == ADD_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
        for (i=0; i<n; i++) y[j][i*bss[j] + k] += x[bs*i+jj+k];
      }
      jj += bss[j];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
        for (i=0; i<n; i++) y[j][i*bss[j] + k] = PetscMax(y[j][i*bss[j] + k],x[bs*i+jj+k]);
      }
      jj += bss[j];
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  PetscCall(VecRestoreArrayRead(v,&x));
  for (i=0; i<nv; i++) {
    PetscCall(VecRestoreArray(s[i],&y[i]));
  }

  PetscCall(PetscFree2(y,bss));
  PetscFunctionReturn(0);
}

/*@
   VecStrideScatterAll - Scatters all the single components from separate vectors into
     a multi-component vector.

   Collective on Vec

   Input Parameters:
+  s - the location where the subvectors are stored
-  addv - one of ADD_VALUES,INSERT_VALUES,MAX_VALUES

   Output Parameter:
.  v - the multicomponent vector

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   The parallel layout of the vector and the subvector must be the same;
   i.e., nlocal of v = stride*(nlocal of s)

   Not optimized; could be easily

   Level: advanced

.seealso: `VecStrideNorm()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideGather()`,
          `VecStrideScatterAll()`
@*/
PetscErrorCode  VecStrideScatterAll(Vec s[],Vec v,InsertMode addv)
{
  PetscInt          i,n,n2,bs,j,jj,k,*bss = NULL,nv,nvc;
  PetscScalar       *x;
  PetscScalar const **y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidPointer(s,1);
  PetscValidHeaderSpecific(*s,VEC_CLASSID,1);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetLocalSize(s[0],&n2));
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetBlockSize(v,&bs));
  PetscCheck(bs > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input vector does not have a valid blocksize set");

  PetscCall(PetscMalloc2(bs,(PetscScalar***)&y,bs,&bss));
  nv   = 0;
  nvc  = 0;
  for (i=0; i<bs; i++) {
    PetscCall(VecGetBlockSize(s[i],&bss[i]));
    if (bss[i] < 1) bss[i] = 1; /* if user never set it then assume 1  Re: [PETSC #8241] VecStrideGatherAll */
    PetscCall(VecGetArrayRead(s[i],&y[i]));
    nvc += bss[i];
    nv++;
    PetscCheck(nvc <= bs,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of subvectors in subvectors > number of vectors in main vector");
    if (nvc == bs) break;
  }

  n =  n/bs;

  jj = 0;
  if (addv == INSERT_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
        for (i=0; i<n; i++) x[bs*i+jj+k] = y[j][i*bss[j] + k];
      }
      jj += bss[j];
    }
  } else if (addv == ADD_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
        for (i=0; i<n; i++) x[bs*i+jj+k] += y[j][i*bss[j] + k];
      }
      jj += bss[j];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
        for (i=0; i<n; i++) x[bs*i+jj+k] = PetscMax(x[bs*i+jj+k],y[j][i*bss[j] + k]);
      }
      jj += bss[j];
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  PetscCall(VecRestoreArray(v,&x));
  for (i=0; i<nv; i++) {
    PetscCall(VecRestoreArrayRead(s[i],&y[i]));
  }
  PetscCall(PetscFree2(*(PetscScalar***)&y,bss));
  PetscFunctionReturn(0);
}

/*@
   VecStrideGather - Gathers a single component from a multi-component vector into
   another vector.

   Collective on Vec

   Input Parameters:
+  v - the vector
.  start - starting point of the subvector (defined by a stride)
-  addv - one of ADD_VALUES,INSERT_VALUES,MAX_VALUES

   Output Parameter:
.  s - the location where the subvector is stored

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this gathers
   the array (x[start],x[start+stride],x[start+2*stride], ....)

   The parallel layout of the vector and the subvector must be the same;
   i.e., nlocal of v = stride*(nlocal of s)

   Not optimized; could be easily

   Level: advanced

.seealso: `VecStrideNorm()`, `VecStrideScatter()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideGatherAll()`,
          `VecStrideScatterAll()`
@*/
PetscErrorCode  VecStrideGather(Vec v,PetscInt start,Vec s,InsertMode addv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,3);
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < v->map->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,v->map->bs);
  PetscCheck(v->ops->stridegather,PetscObjectComm((PetscObject)s),PETSC_ERR_SUP,"Not implemented for this Vec class");
  PetscCall((*v->ops->stridegather)(v,start,s,addv));
  PetscFunctionReturn(0);
}

/*@
   VecStrideScatter - Scatters a single component from a vector into a multi-component vector.

   Collective on Vec

   Input Parameters:
+  s - the single-component vector
.  start - starting point of the subvector (defined by a stride)
-  addv - one of ADD_VALUES,INSERT_VALUES,MAX_VALUES

   Output Parameter:
.  v - the location where the subvector is scattered (the multi-component vector)

   Notes:
   One must call VecSetBlockSize() on the multi-component vector before this
   routine to set the stride  information, or use a vector created from a multicomponent DMDA.

   The parallel layout of the vector and the subvector must be the same;
   i.e., nlocal of v = stride*(nlocal of s)

   Not optimized; could be easily

   Level: advanced

.seealso: `VecStrideNorm()`, `VecStrideGather()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideGatherAll()`,
          `VecStrideScatterAll()`, `VecStrideSubSetScatter()`, `VecStrideSubSetGather()`
@*/
PetscErrorCode  VecStrideScatter(Vec s,PetscInt start,Vec v,InsertMode addv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  PetscCheck(start >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %" PetscInt_FMT,start);
  PetscCheck(start < v->map->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%" PetscInt_FMT ") is too large for stride\n Have you set the vector blocksize (%" PetscInt_FMT ") correctly with VecSetBlockSize()?",start,v->map->bs);
  PetscCheck(v->ops->stridescatter,PetscObjectComm((PetscObject)s),PETSC_ERR_SUP,"Not implemented for this Vec class");
  PetscCall((*v->ops->stridescatter)(s,start,v,addv));
  PetscFunctionReturn(0);
}

/*@
   VecStrideSubSetGather - Gathers a subset of components from a multi-component vector into
   another vector.

   Collective on Vec

   Input Parameters:
+  v - the vector
.  nidx - the number of indices
.  idxv - the indices of the components 0 <= idxv[0] ...idxv[nidx-1] < bs(v), they need not be sorted
.  idxs - the indices of the components 0 <= idxs[0] ...idxs[nidx-1] < bs(s), they need not be sorted, may be null if nidx == bs(s) or is PETSC_DETERMINE
-  addv - one of ADD_VALUES,INSERT_VALUES,MAX_VALUES

   Output Parameter:
.  s - the location where the subvector is stored

   Notes:
   One must call VecSetBlockSize() on both vectors before this routine to set the stride
   information, or use a vector created from a multicomponent DMDA.

   The parallel layout of the vector and the subvector must be the same;

   Not optimized; could be easily

   Level: advanced

.seealso: `VecStrideNorm()`, `VecStrideScatter()`, `VecStrideGather()`, `VecStrideSubSetScatter()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideGatherAll()`,
          `VecStrideScatterAll()`
@*/
PetscErrorCode  VecStrideSubSetGather(Vec v,PetscInt nidx,const PetscInt idxv[],const PetscInt idxs[],Vec s,InsertMode addv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,5);
  if (nidx == PETSC_DETERMINE) nidx = s->map->bs;
  PetscCheck(v->ops->stridesubsetgather,PetscObjectComm((PetscObject)s),PETSC_ERR_SUP,"Not implemented for this Vec class");
  PetscCall((*v->ops->stridesubsetgather)(v,nidx,idxv,idxs,s,addv));
  PetscFunctionReturn(0);
}

/*@
   VecStrideSubSetScatter - Scatters components from a vector into a subset of components of a multi-component vector.

   Collective on Vec

   Input Parameters:
+  s - the smaller-component vector
.  nidx - the number of indices in idx
.  idxs - the indices of the components in the smaller-component vector, 0 <= idxs[0] ...idxs[nidx-1] < bs(s) they need not be sorted, may be null if nidx == bs(s) or is PETSC_DETERMINE
.  idxv - the indices of the components in the larger-component vector, 0 <= idx[0] ...idx[nidx-1] < bs(v) they need not be sorted
-  addv - one of ADD_VALUES,INSERT_VALUES,MAX_VALUES

   Output Parameter:
.  v - the location where the subvector is into scattered (the multi-component vector)

   Notes:
   One must call VecSetBlockSize() on the vectors before this
   routine to set the stride  information, or use a vector created from a multicomponent DMDA.

   The parallel layout of the vector and the subvector must be the same;

   Not optimized; could be easily

   Level: advanced

.seealso: `VecStrideNorm()`, `VecStrideGather()`, `VecStrideGather()`, `VecStrideSubSetGather()`, `VecStrideMin()`, `VecStrideMax()`, `VecStrideGatherAll()`,
          `VecStrideScatterAll()`
@*/
PetscErrorCode  VecStrideSubSetScatter(Vec s,PetscInt nidx,const PetscInt idxs[],const PetscInt idxv[],Vec v,InsertMode addv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(s,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,5);
  if (nidx == PETSC_DETERMINE) nidx = s->map->bs;
  PetscCheck(v->ops->stridesubsetscatter,PetscObjectComm((PetscObject)s),PETSC_ERR_SUP,"Not implemented for this Vec class");
  PetscCall((*v->ops->stridesubsetscatter)(s,nidx,idxs,idxv,v,addv));
  PetscFunctionReturn(0);
}

PetscErrorCode  VecStrideGather_Default(Vec v,PetscInt start,Vec s,InsertMode addv)
{
  PetscInt       i,n,bs,ns;
  const PetscScalar *x;
  PetscScalar       *y;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetLocalSize(s,&ns));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(VecGetArray(s,&y));

  bs = v->map->bs;
  PetscCheck(n == ns*bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subvector length * blocksize %" PetscInt_FMT " not correct for gather from original vector %" PetscInt_FMT,ns*bs,n);
  x += start;
  n  =  n/bs;

  if (addv == INSERT_VALUES) {
    for (i=0; i<n; i++) y[i] = x[bs*i];
  } else if (addv == ADD_VALUES) {
    for (i=0; i<n; i++) y[i] += x[bs*i];
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (i=0; i<n; i++) y[i] = PetscMax(y[i],x[bs*i]);
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  PetscCall(VecRestoreArrayRead(v,&x));
  PetscCall(VecRestoreArray(s,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode  VecStrideScatter_Default(Vec s,PetscInt start,Vec v,InsertMode addv)
{
  PetscInt          i,n,bs,ns;
  PetscScalar       *x;
  const PetscScalar *y;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetLocalSize(s,&ns));
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetArrayRead(s,&y));

  bs = v->map->bs;
  PetscCheck(n == ns*bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subvector length * blocksize %" PetscInt_FMT " not correct for scatter to multicomponent vector %" PetscInt_FMT,ns*bs,n);
  x += start;
  n  =  n/bs;

  if (addv == INSERT_VALUES) {
    for (i=0; i<n; i++) x[bs*i] = y[i];
  } else if (addv == ADD_VALUES) {
    for (i=0; i<n; i++) x[bs*i] += y[i];
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (i=0; i<n; i++) x[bs*i] = PetscMax(y[i],x[bs*i]);
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  PetscCall(VecRestoreArray(v,&x));
  PetscCall(VecRestoreArrayRead(s,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode  VecStrideSubSetGather_Default(Vec v,PetscInt nidx,const PetscInt idxv[],const PetscInt idxs[],Vec s,InsertMode addv)
{
  PetscInt          i,j,n,bs,bss,ns;
  const PetscScalar *x;
  PetscScalar       *y;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetLocalSize(s,&ns));
  PetscCall(VecGetArrayRead(v,&x));
  PetscCall(VecGetArray(s,&y));

  bs  = v->map->bs;
  bss = s->map->bs;
  n  =  n/bs;

  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(n == ns/bss,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible layout of vectors");
    for (j=0; j<nidx; j++) {
      PetscCheck(idxv[j] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"idx[%" PetscInt_FMT "] %" PetscInt_FMT " is negative",j,idxv[j]);
      PetscCheck(idxv[j] < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"idx[%" PetscInt_FMT "] %" PetscInt_FMT " is greater than or equal to vector blocksize %" PetscInt_FMT,j,idxv[j],bs);
    }
    PetscCheck(idxs || bss == nidx,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must provide idxs when not gathering into all locations");
  }

  if (addv == INSERT_VALUES) {
    if (!idxs) {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) y[bss*i+j] = x[bs*i+idxv[j]];
      }
    } else {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) y[bss*i+idxs[j]] = x[bs*i+idxv[j]];
      }
    }
  } else if (addv == ADD_VALUES) {
    if (!idxs) {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) y[bss*i+j] += x[bs*i+idxv[j]];
      }
    } else {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) y[bss*i+idxs[j]] += x[bs*i+idxv[j]];
      }
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    if (!idxs) {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) y[bss*i+j] = PetscMax(y[bss*i+j],x[bs*i+idxv[j]]);
      }
    } else {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) y[bss*i+idxs[j]] = PetscMax(y[bss*i+idxs[j]],x[bs*i+idxv[j]]);
      }
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  PetscCall(VecRestoreArrayRead(v,&x));
  PetscCall(VecRestoreArray(s,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode  VecStrideSubSetScatter_Default(Vec s,PetscInt nidx,const PetscInt idxs[],const PetscInt idxv[],Vec v,InsertMode addv)
{
  PetscInt          j,i,n,bs,ns,bss;
  PetscScalar       *x;
  const PetscScalar *y;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetLocalSize(s,&ns));
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetArrayRead(s,&y));

  bs  = v->map->bs;
  bss = s->map->bs;
  n  =  n/bs;

  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(n == ns/bss,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible layout of vectors");
    for (j=0; j<bss; j++) {
      if (idxs) {
        PetscCheck(idxs[j] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"idx[%" PetscInt_FMT "] %" PetscInt_FMT " is negative",j,idxs[j]);
        PetscCheck(idxs[j] < bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"idx[%" PetscInt_FMT "] %" PetscInt_FMT " is greater than or equal to vector blocksize %" PetscInt_FMT,j,idxs[j],bs);
      }
    }
    PetscCheck(idxs || bss == nidx,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must provide idxs when not scattering from all locations");
  }

  if (addv == INSERT_VALUES) {
    if (!idxs) {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) x[bs*i + idxv[j]] = y[bss*i+j];
      }
    } else {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) x[bs*i + idxv[j]] = y[bss*i+idxs[j]];
      }
    }
  } else if (addv == ADD_VALUES) {
    if (!idxs) {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) x[bs*i + idxv[j]] += y[bss*i+j];
      }
    } else {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) x[bs*i + idxv[j]] += y[bss*i+idxs[j]];
      }
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    if (!idxs) {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) x[bs*i + idxv[j]] = PetscMax(y[bss*i+j],x[bs*i + idxv[j]]);
      }
    } else {
      for (i=0; i<n; i++) {
        for (j=0; j<bss; j++) x[bs*i + idxv[j]] = PetscMax(y[bss*i+idxs[j]],x[bs*i + idxv[j]]);
      }
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  PetscCall(VecRestoreArray(v,&x));
  PetscCall(VecRestoreArrayRead(s,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode VecReciprocal_Default(Vec v)
{
  PetscInt       i,n;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArray(v,&x));
  for (i=0; i<n; i++) {
    if (x[i] != (PetscScalar)0.0) x[i] = (PetscScalar)1.0/x[i];
  }
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

/*@
  VecExp - Replaces each component of a vector by e^x_i

  Not collective

  Input Parameter:
. v - The vector

  Output Parameter:
. v - The vector of exponents

  Level: beginner

.seealso: `VecLog()`, `VecAbs()`, `VecSqrtAbs()`, `VecReciprocal()`

@*/
PetscErrorCode  VecExp(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (v->ops->exp) {
    PetscCall((*v->ops->exp)(v));
  } else {
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(VecGetArray(v, &x));
    for (i = 0; i < n; i++) x[i] = PetscExpScalar(x[i]);
    PetscCall(VecRestoreArray(v, &x));
  }
  PetscFunctionReturn(0);
}

/*@
  VecLog - Replaces each component of a vector by log(x_i), the natural logarithm

  Not collective

  Input Parameter:
. v - The vector

  Output Parameter:
. v - The vector of logs

  Level: beginner

.seealso: `VecExp()`, `VecAbs()`, `VecSqrtAbs()`, `VecReciprocal()`

@*/
PetscErrorCode  VecLog(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (v->ops->log) {
    PetscCall((*v->ops->log)(v));
  } else {
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(VecGetArray(v, &x));
    for (i = 0; i < n; i++) x[i] = PetscLogScalar(x[i]);
    PetscCall(VecRestoreArray(v, &x));
  }
  PetscFunctionReturn(0);
}

/*@
  VecSqrtAbs - Replaces each component of a vector by the square root of its magnitude.

  Not collective

  Input Parameter:
. v - The vector

  Output Parameter:
. v - The vector square root

  Level: beginner

  Note: The actual function is sqrt(|x_i|)

.seealso: `VecLog()`, `VecExp()`, `VecReciprocal()`, `VecAbs()`

@*/
PetscErrorCode  VecSqrtAbs(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (v->ops->sqrt) {
    PetscCall((*v->ops->sqrt)(v));
  } else {
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(VecGetArray(v, &x));
    for (i = 0; i < n; i++) x[i] = PetscSqrtReal(PetscAbsScalar(x[i]));
    PetscCall(VecRestoreArray(v, &x));
  }
  PetscFunctionReturn(0);
}

/*@
  VecDotNorm2 - computes the inner product of two vectors and the 2-norm squared of the second vector

  Collective on Vec

  Input Parameters:
+ s - first vector
- t - second vector

  Output Parameters:
+ dp - s'conj(t)
- nm - t'conj(t)

  Level: advanced

  Notes:
    conj(x) is the complex conjugate of x when x is complex

.seealso: `VecDot()`, `VecNorm()`, `VecDotBegin()`, `VecNormBegin()`, `VecDotEnd()`, `VecNormEnd()`

@*/
PetscErrorCode  VecDotNorm2(Vec s,Vec t,PetscScalar *dp, PetscReal *nm)
{
  const PetscScalar *sx, *tx;
  PetscScalar       dpx = 0.0, nmx = 0.0,work[2],sum[2];
  PetscInt          i, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, VEC_CLASSID,1);
  PetscValidHeaderSpecific(t, VEC_CLASSID,2);
  PetscValidScalarPointer(dp,3);
  PetscValidRealPointer(nm,4);
  PetscValidType(s,1);
  PetscValidType(t,2);
  PetscCheckSameTypeAndComm(s,1,t,2);
  PetscCheck(s->map->N == t->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  PetscCheck(s->map->n == t->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  PetscCall(PetscLogEventBegin(VEC_DotNorm2,s,t,0,0));
  if (s->ops->dotnorm2) {
    PetscCall((*s->ops->dotnorm2)(s,t,dp,&dpx));
    *nm  = PetscRealPart(dpx);
  } else {
    PetscCall(VecGetLocalSize(s, &n));
    PetscCall(VecGetArrayRead(s, &sx));
    PetscCall(VecGetArrayRead(t, &tx));

    for (i = 0; i<n; i++) {
      dpx += sx[i]*PetscConj(tx[i]);
      nmx += tx[i]*PetscConj(tx[i]);
    }
    work[0] = dpx;
    work[1] = nmx;

    PetscCall(MPIU_Allreduce(work,sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s)));
    *dp  = sum[0];
    *nm  = PetscRealPart(sum[1]);

    PetscCall(VecRestoreArrayRead(t, &tx));
    PetscCall(VecRestoreArrayRead(s, &sx));
    PetscCall(PetscLogFlops(4.0*n));
  }
  PetscCall(PetscLogEventEnd(VEC_DotNorm2,s,t,0,0));
  PetscFunctionReturn(0);
}

/*@
   VecSum - Computes the sum of all the components of a vector.

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameter:
.  sum - the result

   Level: beginner

.seealso: `VecMean()`, `VecNorm()`
@*/
PetscErrorCode  VecSum(Vec v,PetscScalar *sum)
{
  PetscInt          i,n;
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(sum,2);
  *sum = 0.0;
  if (v->ops->sum) {
    PetscCall((*v->ops->sum)(v,sum));
  } else {
    PetscCall(VecGetLocalSize(v,&n));
    PetscCall(VecGetArrayRead(v,&x));
    for (i=0; i<n; i++) *sum += x[i];
    PetscCall(VecRestoreArrayRead(v,&x));
  }
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)v)));
  PetscFunctionReturn(0);
}

/*@
   VecMean - Computes the arithmetic mean of all the components of a vector.

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameter:
.  mean - the result

   Level: beginner

.seealso: `VecSum()`, `VecNorm()`
@*/
PetscErrorCode  VecMean(Vec v,PetscScalar *mean)
{
  PetscInt          n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(mean,2);
  PetscCall(VecGetSize(v,&n));
  PetscCall(VecSum(v,mean));
  *mean /= n;
  PetscFunctionReturn(0);
}

/*@
   VecImaginaryPart - Replaces a complex vector with its imginary part

   Collective on Vec

   Input Parameter:
.  v - the vector

   Level: beginner

.seealso: `VecNorm()`, `VecRealPart()`
@*/
PetscErrorCode  VecImaginaryPart(Vec v)
{
  PetscInt          i,n;
  PetscScalar       *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArray(v,&x));
  for (i=0; i<n; i++) x[i] = PetscImaginaryPart(x[i]);
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecRealPart - Replaces a complex vector with its real part

   Collective on Vec

   Input Parameter:
.  v - the vector

   Level: beginner

.seealso: `VecNorm()`, `VecImaginaryPart()`
@*/
PetscErrorCode  VecRealPart(Vec v)
{
  PetscInt          i,n;
  PetscScalar       *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecGetArray(v,&x));
  for (i=0; i<n; i++) x[i] = PetscRealPart(x[i]);
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

/*@
   VecShift - Shifts all of the components of a vector by computing
   x[i] = x[i] + shift.

   Logically Collective on Vec

   Input Parameters:
+  v - the vector
-  shift - the shift

   Level: intermediate

@*/
PetscErrorCode  VecShift(Vec v,PetscScalar shift)
{
  PetscInt       i,n;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(v,shift,2);
  PetscCall(VecSetErrorIfLocked(v,1));
  if (shift == 0.0) PetscFunctionReturn(0);

  if (v->ops->shift) {
    PetscCall((*v->ops->shift)(v,shift));
  } else {
    PetscCall(VecGetLocalSize(v,&n));
    PetscCall(VecGetArray(v,&x));
    for (i=0; i<n; i++) x[i] += shift;
    PetscCall(VecRestoreArray(v,&x));
  }
  PetscFunctionReturn(0);
}

/*@
   VecAbs - Replaces every element in a vector with its absolute value.

   Logically Collective on Vec

   Input Parameters:
.  v - the vector

   Level: intermediate

@*/
PetscErrorCode  VecAbs(Vec v)
{
  PetscInt       i,n;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(VecSetErrorIfLocked(v,1));

  if (v->ops->abs) {
    PetscCall((*v->ops->abs)(v));
  } else {
    PetscCall(VecGetLocalSize(v,&n));
    PetscCall(VecGetArray(v,&x));
    for (i=0; i<n; i++) x[i] = PetscAbsScalar(x[i]);
    PetscCall(VecRestoreArray(v,&x));
  }
  PetscFunctionReturn(0);
}

/*@
  VecPermute - Permutes a vector in place using the given ordering.

  Input Parameters:
+ vec   - The vector
. order - The ordering
- inv   - The flag for inverting the permutation

  Level: beginner

  Note: This function does not yet support parallel Index Sets with non-local permutations

.seealso: `MatPermute()`
@*/
PetscErrorCode  VecPermute(Vec x, IS row, PetscBool inv)
{
  const PetscScalar *array;
  PetscScalar       *newArray;
  const PetscInt    *idx;
  PetscInt          i,rstart,rend;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(row,IS_CLASSID,2);
  PetscCall(VecSetErrorIfLocked(x,1));
  PetscCall(VecGetOwnershipRange(x,&rstart,&rend));
  PetscCall(ISGetIndices(row, &idx));
  PetscCall(VecGetArrayRead(x, &array));
  PetscCall(PetscMalloc1(x->map->n, &newArray));
  if (PetscDefined(USE_DEBUG)) {
    for (i = 0; i < x->map->n; i++) {
      PetscCheck(!(idx[i] < rstart) && !(idx[i] >= rend),PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT, "Permutation index %" PetscInt_FMT " is out of bounds: %" PetscInt_FMT, i, idx[i]);
    }
  }
  if (!inv) {
    for (i = 0; i < x->map->n; i++) newArray[i] = array[idx[i]-rstart];
  } else {
    for (i = 0; i < x->map->n; i++) newArray[idx[i]-rstart] = array[i];
  }
  PetscCall(VecRestoreArrayRead(x, &array));
  PetscCall(ISRestoreIndices(row, &idx));
  PetscCall(VecReplaceArray(x, newArray));
  PetscFunctionReturn(0);
}

/*@
   VecEqual - Compares two vectors. Returns true if the two vectors are either pointing to the same memory buffer,
   or if the two vectors have the same local and global layout as well as bitwise equality of all entries.
   Does NOT take round-off errors into account.

   Collective on Vec

   Input Parameters:
+  vec1 - the first vector
-  vec2 - the second vector

   Output Parameter:
.  flg - PETSC_TRUE if the vectors are equal; PETSC_FALSE otherwise.

   Level: intermediate
@*/
PetscErrorCode  VecEqual(Vec vec1,Vec vec2,PetscBool  *flg)
{
  const PetscScalar  *v1,*v2;
  PetscInt           n1,n2,N1,N2;
  PetscBool          flg1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(vec2,VEC_CLASSID,2);
  PetscValidBoolPointer(flg,3);
  if (vec1 == vec2) *flg = PETSC_TRUE;
  else {
    PetscCall(VecGetSize(vec1,&N1));
    PetscCall(VecGetSize(vec2,&N2));
    if (N1 != N2) flg1 = PETSC_FALSE;
    else {
      PetscCall(VecGetLocalSize(vec1,&n1));
      PetscCall(VecGetLocalSize(vec2,&n2));
      if (n1 != n2) flg1 = PETSC_FALSE;
      else {
        PetscCall(VecGetArrayRead(vec1,&v1));
        PetscCall(VecGetArrayRead(vec2,&v2));
        PetscCall(PetscArraycmp(v1,v2,n1,&flg1));
        PetscCall(VecRestoreArrayRead(vec1,&v1));
        PetscCall(VecRestoreArrayRead(vec2,&v2));
      }
    }
    /* combine results from all processors */
    PetscCall(MPIU_Allreduce(&flg1,flg,1,MPIU_BOOL,MPI_MIN,PetscObjectComm((PetscObject)vec1)));
  }
  PetscFunctionReturn(0);
}

/*@
   VecUniqueEntries - Compute the number of unique entries, and those entries

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Output Parameters:
+  n - The number of unique entries
-  e - The entries

   Level: intermediate

@*/
PetscErrorCode  VecUniqueEntries(Vec vec, PetscInt *n, PetscScalar **e)
{
  const PetscScalar *v;
  PetscScalar       *tmp, *vals;
  PetscMPIInt       *N, *displs, l;
  PetscInt          ng, m, i, j, p;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidIntPointer(n,2);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) vec), &size));
  PetscCall(VecGetLocalSize(vec, &m));
  PetscCall(VecGetArrayRead(vec, &v));
  PetscCall(PetscMalloc2(m,&tmp,size,&N));
  for (i = 0, j = 0, l = 0; i < m; ++i) {
    /* Can speed this up with sorting */
    for (j = 0; j < l; ++j) {
      if (v[i] == tmp[j]) break;
    }
    if (j == l) {
      tmp[j] = v[i];
      ++l;
    }
  }
  PetscCall(VecRestoreArrayRead(vec, &v));
  /* Gather serial results */
  PetscCallMPI(MPI_Allgather(&l, 1, MPI_INT, N, 1, MPI_INT, PetscObjectComm((PetscObject) vec)));
  for (p = 0, ng = 0; p < size; ++p) {
    ng += N[p];
  }
  PetscCall(PetscMalloc2(ng,&vals,size+1,&displs));
  for (p = 1, displs[0] = 0; p <= size; ++p) {
    displs[p] = displs[p-1] + N[p-1];
  }
  PetscCallMPI(MPI_Allgatherv(tmp, l, MPIU_SCALAR, vals, N, displs, MPIU_SCALAR, PetscObjectComm((PetscObject) vec)));
  /* Find unique entries */
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PetscObjectComm((PetscObject) vec), PETSC_ERR_SUP, "Does not work with complex numbers");
#else
  *n = displs[size];
  PetscCall(PetscSortRemoveDupsReal(n, (PetscReal *) vals));
  if (e) {
    PetscValidPointer(e,3);
    PetscCall(PetscMalloc1(*n, e));
    for (i = 0; i < *n; ++i) {
      (*e)[i] = vals[i];
    }
  }
  PetscCall(PetscFree2(vals,displs));
  PetscCall(PetscFree2(tmp,N));
  PetscFunctionReturn(0);
#endif
}
