
/*
     Some useful vector utility functions.
*/
#include <petsc-private/vecimpl.h>          /*I "petscvec.h" I*/

extern MPI_Op VecMax_Local_Op;
extern MPI_Op VecMin_Local_Op;

#undef __FUNCT__  
#define __FUNCT__ "VecStrideSet"
/*@
   VecStrideSet - Sets a subvector of a vector defined 
   by a starting point and a stride with a given value

   Logically Collective on Vec

   Input Parameter:
+  v - the vector 
.  start - starting point of the subvector (defined by a stride)
-  s - value to multiply each subvector entry by

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   This will only work if the desire subvector is a stride subvector

   Level: advanced

   Concepts: scale^on stride of vector
   Concepts: stride^scale

.seealso: VecNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMin(), VecStrideMax(), VecStrideScale()
@*/
PetscErrorCode  VecStrideSet(Vec v,PetscInt start,PetscScalar s)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(v,start,2);  
  PetscValidLogicalCollectiveScalar(v,s,3);  

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  bs   = v->map->bs;
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  else if (start >= bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%D) is too large for stride\n  Have you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,bs);
  x += start;

  for (i=0; i<n; i+=bs) {
    x[i] = s;
  }
  x -= start;

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideScale"
/*@
   VecStrideScale - Scales a subvector of a vector defined 
   by a starting point and a stride.

   Logically Collective on Vec

   Input Parameter:
+  v - the vector 
.  start - starting point of the subvector (defined by a stride)
-  scale - value to multiply each subvector entry by

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   This will only work if the desire subvector is a stride subvector

   Level: advanced

   Concepts: scale^on stride of vector
   Concepts: stride^scale

.seealso: VecNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMin(), VecStrideMax(), VecStrideScale()
@*/
PetscErrorCode  VecStrideScale(Vec v,PetscInt start,PetscScalar scale)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(v,start,2);  
  PetscValidLogicalCollectiveScalar(v,scale,3);  

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  bs   = v->map->bs;
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  else if (start >= bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%D) is too large for stride\n  Have you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,bs);
  x += start;

  for (i=0; i<n; i+=bs) {
    x[i] *= scale;
  }
  x -= start;

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideNorm"
/*@
   VecStrideNorm - Computes the norm of subvector of a vector defined 
   by a starting point and a stride.

   Collective on Vec

   Input Parameter:
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

   Concepts: norm^on stride of vector
   Concepts: stride^norm

.seealso: VecNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMin(), VecStrideMax()
@*/
PetscErrorCode  VecStrideNorm(Vec v,PetscInt start,NormType ntype,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs;
  PetscScalar    *x;
  PetscReal      tnorm;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidDoublePointer(nrm,3);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  else if (start >= bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%D) is too large for stride\n Have you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,bs);
  x += start;

  if (ntype == NORM_2) {
    PetscScalar sum = 0.0;
    for (i=0; i<n; i+=bs) {
      sum += x[i]*(PetscConj(x[i]));
    }
    tnorm  = PetscRealPart(sum);
    ierr   = MPI_Allreduce(&tnorm,nrm,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
    *nrm = PetscSqrtReal(*nrm);
  } else if (ntype == NORM_1) {
    tnorm = 0.0;
    for (i=0; i<n; i+=bs) {
      tnorm += PetscAbsScalar(x[i]);
    }
    ierr   = MPI_Allreduce(&tnorm,nrm,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
  } else if (ntype == NORM_INFINITY) {
    PetscReal tmp;
    tnorm = 0.0;

    for (i=0; i<n; i+=bs) {
      if ((tmp = PetscAbsScalar(x[i])) > tnorm) tnorm = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {tnorm = tmp; break;}
    } 
    ierr   = MPI_Allreduce(&tnorm,nrm,1,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown norm type");
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideMax"
/*@
   VecStrideMax - Computes the maximum of subvector of a vector defined 
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameter:
+  v - the vector 
-  start - starting point of the subvector (defined by a stride)

   Output Parameter:
+  index - the location where the maximum occurred  (pass PETSC_NULL if not required)
-  nrm - the max

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   If xa is the array representing the vector x, then this computes the max
   of the array (xa[start],xa[start+stride],xa[start+2*stride], ....)

   This is useful for computing, say the maximum of the pressure variable when
   the pressure is stored (interlaced) with other variables, e.g., density, etc.
   This will only work if the desire subvector is a stride subvector.

   Level: advanced

   Concepts: maximum^on stride of vector
   Concepts: stride^maximum

.seealso: VecMax(), VecStrideNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMin()
@*/
PetscErrorCode  VecStrideMax(Vec v,PetscInt start,PetscInt *idex,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs,id;
  PetscScalar    *x;
  PetscReal      max,tmp;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidDoublePointer(nrm,3);

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  else if (start >= bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Start of stride subvector (%D) is too large for stride\n Have you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,bs);
  x += start;

  id = -1;
  if (!n) {
    max = PETSC_MIN_REAL;
  } else {
    id  = 0;
#if defined(PETSC_USE_COMPLEX)
    max = PetscRealPart(x[0]);
#else
    max = x[0];
#endif
    for (i=bs; i<n; i+=bs) {
#if defined(PETSC_USE_COMPLEX)
      if ((tmp = PetscRealPart(x[i])) > max) { max = tmp; id = i;}
#else
      if ((tmp = x[i]) > max) { max = tmp; id = i;} 
#endif
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);

  if (!idex) {
    ierr   = MPI_Allreduce(&max,nrm,1,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
  } else {
    PetscReal in[2],out[2];
    PetscInt  rstart;

    ierr  = VecGetOwnershipRange(v,&rstart,PETSC_NULL);CHKERRQ(ierr);
    in[0] = max;
    in[1] = rstart+id+start;
    ierr  = MPI_Allreduce(in,out,2,MPIU_REAL,VecMax_Local_Op,((PetscObject)v)->comm);CHKERRQ(ierr);
    *nrm  = out[0];
    *idex = (PetscInt)out[1];
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideMin"
/*@
   VecStrideMin - Computes the minimum of subvector of a vector defined 
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameter:
+  v - the vector 
-  start - starting point of the subvector (defined by a stride)

   Output Parameter:
+  idex - the location where the minimum occurred. (pass PETSC_NULL if not required)
-  nrm - the min

   Level: advanced

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   If xa is the array representing the vector x, then this computes the min
   of the array (xa[start],xa[start+stride],xa[start+2*stride], ....)

   This is useful for computing, say the minimum of the pressure variable when
   the pressure is stored (interlaced) with other variables, e.g., density, etc.
   This will only work if the desire subvector is a stride subvector.

   Concepts: minimum^on stride of vector
   Concepts: stride^minimum

.seealso: VecMin(), VecStrideNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMax()
@*/
PetscErrorCode  VecStrideMin(Vec v,PetscInt start,PetscInt *idex,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs,id;
  PetscScalar    *x;
  PetscReal      min,tmp;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidDoublePointer(nrm,4);

  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (start < 0)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  else if (start >= bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Start of stride subvector (%D) is too large for stride\nHave you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,bs);
  x += start;

  id = -1;
  if (!n) {
    min = PETSC_MAX_REAL;
  } else {
    id = 0;
#if defined(PETSC_USE_COMPLEX)
    min = PetscRealPart(x[0]);
#else
    min = x[0];
#endif
    for (i=bs; i<n; i+=bs) {
#if defined(PETSC_USE_COMPLEX)
      if ((tmp = PetscRealPart(x[i])) < min) { min = tmp; id = i;}
#else
      if ((tmp = x[i]) < min) { min = tmp; id = i;} 
#endif
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);

  if (!idex) {
    ierr   = MPI_Allreduce(&min,nrm,1,MPIU_REAL,MPIU_MIN,comm);CHKERRQ(ierr);
  } else {
    PetscReal in[2],out[2];
    PetscInt  rstart;

    ierr  = VecGetOwnershipRange(v,&rstart,PETSC_NULL);CHKERRQ(ierr);
    in[0] = min;
    in[1] = rstart+id;
    ierr  = MPI_Allreduce(in,out,2,MPIU_REAL,VecMin_Local_Op,((PetscObject)v)->comm);CHKERRQ(ierr);
    *nrm  = out[0];
    *idex = (PetscInt)out[1];
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideScaleAll"
/*@
   VecStrideScaleAll - Scales the subvectors of a vector defined 
   by a starting point and a stride.

   Logically Collective on Vec

   Input Parameter:
+  v - the vector 
-  scales - values to multiply each subvector entry by

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.


   Level: advanced

   Concepts: scale^on stride of vector
   Concepts: stride^scale

.seealso: VecNorm(), VecStrideScale(), VecScale(), VecStrideGather(), VecStrideScatter(), VecStrideMin(), VecStrideMax()
@*/
PetscErrorCode  VecStrideScaleAll(Vec v,const PetscScalar *scales)
{
  PetscErrorCode ierr;
  PetscInt       i,j,n,bs;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(scales,2);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);

  bs   = v->map->bs;

  /* need to provide optimized code for each bs */
  for (i=0; i<n; i+=bs) {
    for (j=0; j<bs; j++) {
      x[i+j] *= scales[j];
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideNormAll"
/*@
   VecStrideNormAll - Computes the norms  subvectors of a vector defined 
   by a starting point and a stride.

   Collective on Vec

   Input Parameter:
+  v - the vector 
-  ntype - type of norm, one of NORM_1, NORM_2, NORM_INFINITY

   Output Parameter:
.  nrm - the norms

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   If x is the array representing the vector x then this computes the norm 
   of the array (x[start],x[start+stride],x[start+2*stride], ....)

   This is useful for computing, say the norm of the pressure variable when
   the pressure is stored (interlaced) with other variables, say density etc.

   This will only work if the desire subvector is a stride subvector

   Level: advanced

   Concepts: norm^on stride of vector
   Concepts: stride^norm

.seealso: VecNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMin(), VecStrideMax()
@*/
PetscErrorCode  VecStrideNormAll(Vec v,NormType ntype,PetscReal nrm[])
{
  PetscErrorCode ierr;
  PetscInt       i,j,n,bs;
  PetscScalar    *x;
  PetscReal      tnorm[128];
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidDoublePointer(nrm,3);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (bs > 128) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  if (ntype == NORM_2) {
    PetscScalar sum[128];
    for (j=0; j<bs; j++) sum[j] = 0.0;
    for (i=0; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
	sum[j] += x[i+j]*(PetscConj(x[i+j]));
      }
    }
    for (j=0; j<bs; j++) {
      tnorm[j]  = PetscRealPart(sum[j]);
    }
    ierr   = MPI_Allreduce(tnorm,nrm,bs,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
    for (j=0; j<bs; j++) {
      nrm[j] = PetscSqrtReal(nrm[j]);
    }
  } else if (ntype == NORM_1) {
    for (j=0; j<bs; j++) {
      tnorm[j] = 0.0;
    }
    for (i=0; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
	tnorm[j] += PetscAbsScalar(x[i+j]);
      }
    }
    ierr   = MPI_Allreduce(tnorm,nrm,bs,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
  } else if (ntype == NORM_INFINITY) {
    PetscReal tmp;
    for (j=0; j<bs; j++) {
      tnorm[j] = 0.0;
    }

    for (i=0; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
	if ((tmp = PetscAbsScalar(x[i+j])) > tnorm[j]) tnorm[j] = tmp;
	/* check special case of tmp == NaN */
	if (tmp != tmp) {tnorm[j] = tmp; break;}
      }
    } 
    ierr   = MPI_Allreduce(tnorm,nrm,bs,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown norm type");
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideMaxAll"
/*@
   VecStrideMaxAll - Computes the maximums of subvectors of a vector defined 
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameter:
.  v - the vector 

   Output Parameter:
+  index - the location where the maximum occurred (not supported, pass PETSC_NULL,
           if you need this, send mail to petsc-maint@mcs.anl.gov to request it)
-  nrm - the maximums

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   This is useful for computing, say the maximum of the pressure variable when
   the pressure is stored (interlaced) with other variables, e.g., density, etc.
   This will only work if the desire subvector is a stride subvector.

   Level: advanced

   Concepts: maximum^on stride of vector
   Concepts: stride^maximum

.seealso: VecMax(), VecStrideNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMin()
@*/
PetscErrorCode  VecStrideMaxAll(Vec v,PetscInt idex[],PetscReal nrm[])
{
  PetscErrorCode ierr;
  PetscInt       i,j,n,bs;
  PetscScalar    *x;
  PetscReal      max[128],tmp;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidDoublePointer(nrm,3);
  if (idex) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for returning index; send mail to petsc-maint@mcs.anl.gov asking for it");
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (bs > 128) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  if (!n) {
    for (j=0; j<bs; j++) {
      max[j] = PETSC_MIN_REAL;
    }
  } else {
    for (j=0; j<bs; j++) {
#if defined(PETSC_USE_COMPLEX)
      max[j] = PetscRealPart(x[j]);
#else
      max[j] = x[j];
#endif
    }
    for (i=bs; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
#if defined(PETSC_USE_COMPLEX)
	if ((tmp = PetscRealPart(x[i+j])) > max[j]) { max[j] = tmp;}
#else
	if ((tmp = x[i+j]) > max[j]) { max[j] = tmp; } 
#endif
      }
    }
  }
  ierr   = MPI_Allreduce(max,nrm,bs,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideMinAll"
/*@
   VecStrideMinAll - Computes the minimum of subvector of a vector defined 
   by a starting point and a stride and optionally its location.

   Collective on Vec

   Input Parameter:
.  v - the vector 

   Output Parameter:
+  idex - the location where the minimum occurred (not supported, pass PETSC_NULL,
           if you need this, send mail to petsc-maint@mcs.anl.gov to request it)
-  nrm - the minimums

   Level: advanced

   Notes:
   One must call VecSetBlockSize() before this routine to set the stride 
   information, or use a vector created from a multicomponent DMDA.

   This is useful for computing, say the minimum of the pressure variable when
   the pressure is stored (interlaced) with other variables, e.g., density, etc.
   This will only work if the desire subvector is a stride subvector.

   Concepts: minimum^on stride of vector
   Concepts: stride^minimum

.seealso: VecMin(), VecStrideNorm(), VecStrideGather(), VecStrideScatter(), VecStrideMax()
@*/
PetscErrorCode  VecStrideMinAll(Vec v,PetscInt idex[],PetscReal nrm[])
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs,j;
  PetscScalar    *x;
  PetscReal      min[128],tmp;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidDoublePointer(nrm,3);
  if (idex) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for returning index; send mail to petsc-maint@mcs.anl.gov asking for it");
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (bs > 128) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently supports only blocksize up to 128");

  if (!n) {
    for (j=0; j<bs; j++) {
      min[j] = PETSC_MAX_REAL;
    }
  } else {
    for (j=0; j<bs; j++) {
#if defined(PETSC_USE_COMPLEX)
      min[j] = PetscRealPart(x[j]);
#else
      min[j] = x[j];
#endif
    }
    for (i=bs; i<n; i+=bs) {
      for (j=0; j<bs; j++) {
#if defined(PETSC_USE_COMPLEX)
	if ((tmp = PetscRealPart(x[i+j])) < min[j]) { min[j] = tmp;}
#else
	if ((tmp = x[i+j]) < min[j]) { min[j] = tmp; } 
#endif
      }
    }
  }
  ierr   = MPI_Allreduce(min,nrm,bs,MPIU_REAL,MPIU_MIN,comm);CHKERRQ(ierr);

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecStrideGatherAll"
/*@
   VecStrideGatherAll - Gathers all the single components from a multi-component vector into
   separate vectors.

   Collective on Vec

   Input Parameter:
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

   Concepts: gather^into strided vector

.seealso: VecStrideNorm(), VecStrideScatter(), VecStrideMin(), VecStrideMax(), VecStrideGather(),
          VecStrideScatterAll()
@*/
PetscErrorCode  VecStrideGatherAll(Vec v,Vec s[],InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       i,n,n2,bs,j,k,*bss = PETSC_NULL,nv,jj,nvc;
  PetscScalar    *x,**y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(s,2);
  PetscValidHeaderSpecific(*s,VEC_CLASSID,2);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(s[0],&n2);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  bs   = v->map->bs;
  if (bs < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input vector does not have a valid blocksize set");

  ierr = PetscMalloc2(bs,PetscReal*,&y,bs,PetscInt,&bss);CHKERRQ(ierr);
  nv   = 0;
  nvc  = 0;
  for (i=0; i<bs; i++) {
    ierr = VecGetBlockSize(s[i],&bss[i]);CHKERRQ(ierr);
    if (bss[i] < 1) bss[i] = 1; /* if user never set it then assume 1  Re: [PETSC #8241] VecStrideGatherAll */
    ierr = VecGetArray(s[i],&y[i]);CHKERRQ(ierr);
    nvc  += bss[i];
    nv++;
    if (nvc > bs)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of subvectors in subvectors > number of vectors in main vector");
    if (nvc == bs) break;
  }

  n =  n/bs;

  jj = 0;
  if (addv == INSERT_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
	for (i=0; i<n; i++) {
	  y[j][i*bss[j] + k] = x[bs*i+jj+k];
        }
      }
      jj += bss[j];
    }
  } else if (addv == ADD_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
	for (i=0; i<n; i++) {
	  y[j][i*bss[j] + k] += x[bs*i+jj+k];
        }
      }
      jj += bss[j];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
	for (i=0; i<n; i++) {
	  y[j][i*bss[j] + k] = PetscMax(y[j][i*bss[j] + k],x[bs*i+jj+k]);
        }
      }
      jj += bss[j];
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<nv; i++) {
    ierr = VecRestoreArray(s[i],&y[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree2(y,bss);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideScatterAll"
/*@
   VecStrideScatterAll - Scatters all the single components from separate vectors into 
     a multi-component vector.

   Collective on Vec

   Input Parameter:
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

   Concepts:  scatter^into strided vector

.seealso: VecStrideNorm(), VecStrideScatter(), VecStrideMin(), VecStrideMax(), VecStrideGather(),
          VecStrideScatterAll()
@*/
PetscErrorCode  VecStrideScatterAll(Vec s[],Vec v,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt        i,n,n2,bs,j,jj,k,*bss = PETSC_NULL,nv,nvc;
  PetscScalar     *x,**y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(s,2);
  PetscValidHeaderSpecific(*s,VEC_CLASSID,2);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(s[0],&n2);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  bs   = v->map->bs;
  if (bs < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input vector does not have a valid blocksize set");

  ierr = PetscMalloc2(bs,PetscScalar**,&y,bs,PetscInt,&bss);CHKERRQ(ierr);
  nv  = 0;
  nvc = 0;
  for (i=0; i<bs; i++) {
    ierr = VecGetBlockSize(s[i],&bss[i]);CHKERRQ(ierr);
    if (bss[i] < 1) bss[i] = 1; /* if user never set it then assume 1  Re: [PETSC #8241] VecStrideGatherAll */
    ierr = VecGetArray(s[i],&y[i]);CHKERRQ(ierr);
    nvc  += bss[i];
    nv++;
    if (nvc > bs)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of subvectors in subvectors > number of vectors in main vector");
    if (nvc == bs) break;
  }

  n =  n/bs;

  jj = 0;
  if (addv == INSERT_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
	for (i=0; i<n; i++) {
	  x[bs*i+jj+k] = y[j][i*bss[j] + k];
        }
      }
      jj += bss[j];
    }
  } else if (addv == ADD_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
	for (i=0; i<n; i++) {
	  x[bs*i+jj+k] += y[j][i*bss[j] + k];
        }
      }
      jj += bss[j];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (j=0; j<nv; j++) {
      for (k=0; k<bss[j]; k++) {
	for (i=0; i<n; i++) {
	  x[bs*i+jj+k] = PetscMax(x[bs*i+jj+k],y[j][i*bss[j] + k]);
        }
      }
      jj += bss[j];
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<nv; i++) {
    ierr = VecRestoreArray(s[i],&y[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(y,bss);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideGather"
/*@
   VecStrideGather - Gathers a single component from a multi-component vector into
   another vector.

   Collective on Vec

   Input Parameter:
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

   Concepts: gather^into strided vector

.seealso: VecStrideNorm(), VecStrideScatter(), VecStrideMin(), VecStrideMax(), VecStrideGatherAll(),
          VecStrideScatterAll()
@*/
PetscErrorCode  VecStrideGather(Vec v,PetscInt start,Vec s,InsertMode addv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,3);
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  if (start >= v->map->bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%D) is too large for stride\n Have you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,v->map->bs);
  if (!v->ops->stridegather) SETERRQ(((PetscObject)s)->comm,PETSC_ERR_SUP,"Not implemented for this Vec class");
  ierr = (*v->ops->stridegather)(v,start,s,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideScatter"
/*@
   VecStrideScatter - Scatters a single component from a vector into a multi-component vector.

   Collective on Vec

   Input Parameter:
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

   Concepts: scatter^into strided vector

.seealso: VecStrideNorm(), VecStrideGather(), VecStrideMin(), VecStrideMax(), VecStrideGatherAll(),
          VecStrideScatterAll()
@*/
PetscErrorCode  VecStrideScatter(Vec s,PetscInt start,Vec v,InsertMode addv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative start %D",start);
  if (start >= v->map->bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Start of stride subvector (%D) is too large for stride\n Have you set the vector blocksize (%D) correctly with VecSetBlockSize()?",start,v->map->bs);
  if (!v->ops->stridescatter) SETERRQ(((PetscObject)s)->comm,PETSC_ERR_SUP,"Not implemented for this Vec class");
  ierr = (*v->ops->stridescatter)(s,start,v,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideGather_Default"
PetscErrorCode  VecStrideGather_Default(Vec v,PetscInt start,Vec s,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs,ns;
  PetscScalar    *x,*y;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(s,&ns);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetArray(s,&y);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (n != ns*bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subvector length * blocksize %D not correct for gather from original vector %D",ns*bs,n);
  x += start;
  n =  n/bs;

  if (addv == INSERT_VALUES) {
    for (i=0; i<n; i++) {
      y[i] = x[bs*i];
    }
  } else if (addv == ADD_VALUES) {
    for (i=0; i<n; i++) {
      y[i] += x[bs*i];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (i=0; i<n; i++) {
      y[i] = PetscMax(y[i],x[bs*i]);
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(s,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStrideScatter_Default"
PetscErrorCode  VecStrideScatter_Default(Vec s,PetscInt start,Vec v,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       i,n,bs,ns;
  PetscScalar    *x,*y;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(s,&ns);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetArray(s,&y);CHKERRQ(ierr);

  bs   = v->map->bs;
  if (n != ns*bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subvector length * blocksize %D not correct for scatter to multicomponent vector %D",ns*bs,n);
  x += start;
  n =  n/bs;

  if (addv == INSERT_VALUES) {
    for (i=0; i<n; i++) {
      x[bs*i] = y[i];
    }
  } else if (addv == ADD_VALUES) {
    for (i=0; i<n; i++) {
      x[bs*i] += y[i];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (i=0; i<n; i++) {
      x[bs*i] = PetscMax(y[i],x[bs*i]);
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown insert type");

  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(s,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecReciprocal_Default"
PetscErrorCode VecReciprocal_Default(Vec v)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *x;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (x[i] != (PetscScalar)0.0) x[i] = (PetscScalar)1.0/x[i];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecExp"
/*@
  VecExp - Replaces each component of a vector by e^x_i

  Not collective

  Input Parameter:
. v - The vector

  Output Parameter:
. v - The vector of exponents

  Level: beginner

.seealso:  VecLog(), VecAbs(), VecSqrtAbs(), VecReciprocal()

.keywords: vector, sqrt, square root
@*/
PetscErrorCode  VecExp(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (v->ops->exp) {
    ierr = (*v->ops->exp)(v);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
    ierr = VecGetArray(v, &x);CHKERRQ(ierr);
    for(i = 0; i < n; i++) {
      x[i] = PetscExpScalar(x[i]);
    }
    ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLog"
/*@
  VecLog - Replaces each component of a vector by log(x_i), the natural logarithm

  Not collective

  Input Parameter:
. v - The vector

  Output Parameter:
. v - The vector of logs

  Level: beginner

.seealso:  VecExp(), VecAbs(), VecSqrtAbs(), VecReciprocal()

.keywords: vector, sqrt, square root
@*/
PetscErrorCode  VecLog(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (v->ops->log) {
    ierr = (*v->ops->log)(v);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
    ierr = VecGetArray(v, &x);CHKERRQ(ierr);
    for(i = 0; i < n; i++) {
      x[i] = PetscLogScalar(x[i]);
    }
    ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSqrtAbs"
/*@
  VecSqrtAbs - Replaces each component of a vector by the square root of its magnitude.

  Not collective

  Input Parameter:
. v - The vector

  Output Parameter:
. v - The vector square root

  Level: beginner

  Note: The actual function is sqrt(|x_i|)

.seealso: VecLog(), VecExp(), VecReciprocal(), VecAbs()

.keywords: vector, sqrt, square root
@*/
PetscErrorCode  VecSqrtAbs(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (v->ops->sqrt) {
    ierr = (*v->ops->sqrt)(v);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
    ierr = VecGetArray(v, &x);CHKERRQ(ierr);
    for(i = 0; i < n; i++) {
      x[i] = PetscSqrtReal(PetscAbsScalar(x[i]));
    }
    ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDotNorm2"
/*@
  VecDotNorm2 - computes the inner product of two vectors and the 2-norm squared of the second vector

  Collective on Vec

  Input Parameter:
+ s - first vector
- t - second vector

  Output Parameter:
+ dp - s't
- nm - t't

  Level: advanced

  Developer Notes: Even though the second return argument is a norm and hence could be a PetscReal value it is returned as PetscScalar

.seealso:   VecDot(), VecNorm(), VecDotBegin(), VecNormBegin(), VecDotEnd(), VecNormEnd()

.keywords: vector, sqrt, square root
@*/
PetscErrorCode  VecDotNorm2(Vec s,Vec t,PetscScalar *dp, PetscScalar *nm)
{
  PetscScalar    *sx, *tx, dpx = 0.0, nmx = 0.0,work[2],sum[2];
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(s, VEC_CLASSID,1);
  PetscValidHeaderSpecific(t, VEC_CLASSID,2);
  PetscValidScalarPointer(dp,3);
  PetscValidScalarPointer(nm,4);
  PetscValidType(s,1);
  PetscValidType(t,2);
  PetscCheckSameTypeAndComm(s,1,t,2);
  if (s->map->N != t->map->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (s->map->n != t->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBarrierBegin(VEC_DotNormBarrier,s,t,0,0,((PetscObject)s)->comm);CHKERRQ(ierr);
  if (s->ops->dotnorm2) {
    ierr = (*s->ops->dotnorm2)(s,t,dp,nm);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(s, &n);CHKERRQ(ierr);
    ierr = VecGetArray(s, &sx);CHKERRQ(ierr);
    ierr = VecGetArray(t, &tx);CHKERRQ(ierr);

    for (i = 0; i<n; i++) {
      dpx += sx[i]*PetscConj(tx[i]);
      nmx += tx[i]*PetscConj(tx[i]);
    }
    work[0] = dpx;
    work[1] = nmx;
    ierr = MPI_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,((PetscObject)s)->comm);CHKERRQ(ierr);
    *dp  = sum[0];
    *nm  = sum[1];

    ierr = VecRestoreArray(t, &tx);CHKERRQ(ierr);
    ierr = VecRestoreArray(s, &sx);CHKERRQ(ierr);
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBarrierEnd(VEC_DotNormBarrier,s,t,0,0,((PetscObject)s)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSum"
/*@
   VecSum - Computes the sum of all the components of a vector.

   Collective on Vec

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  sum - the result

   Level: beginner

   Concepts: sum^of vector entries

.seealso: VecNorm()
@*/
PetscErrorCode  VecSum(Vec v,PetscScalar *sum)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *x,lsum = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(sum,2);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    lsum += x[i];
  }
  ierr = MPI_Allreduce(&lsum,sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)v)->comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecShift"
/*@
   VecShift - Shifts all of the components of a vector by computing
   x[i] = x[i] + shift.

   Logically Collective on Vec

   Input Parameters:
+  v - the vector 
-  shift - the shift

   Output Parameter:
.  v - the shifted vector 

   Level: intermediate

   Concepts: vector^adding constant

@*/
PetscErrorCode  VecShift(Vec v,PetscScalar shift)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(v,shift,2);  
  if (v->ops->shift) {
    ierr = (*v->ops->shift)(v);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr); 
    ierr = VecGetArray(v,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      x[i] += shift;
    }
    ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAbs"
/*@
   VecAbs - Replaces every element in a vector with its absolute value.

   Logically Collective on Vec

   Input Parameters:
.  v - the vector 

   Level: intermediate

   Concepts: vector^absolute value

@*/
PetscErrorCode  VecAbs(Vec v)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  if (v->ops->abs) {
    ierr = (*v->ops->abs)(v);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
    ierr = VecGetArray(v,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      x[i] = PetscAbsScalar(x[i]);
    }
    ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPermute"
/*@
  VecPermute - Permutes a vector in place using the given ordering.

  Input Parameters:
+ vec   - The vector
. order - The ordering
- inv   - The flag for inverting the permutation

  Level: beginner

  Note: This function does not yet support parallel Index Sets

.seealso: MatPermute()
.keywords: vec, permute
@*/
PetscErrorCode  VecPermute(Vec x, IS row, PetscBool  inv)
{
  PetscScalar    *array, *newArray;
  const PetscInt *idx;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetIndices(row, &idx);CHKERRQ(ierr);
  ierr = VecGetArray(x, &array);CHKERRQ(ierr);
  ierr = PetscMalloc(x->map->n*sizeof(PetscScalar), &newArray);CHKERRQ(ierr);
#ifdef PETSC_USE_DEBUG
  for(i = 0; i < x->map->n; i++) {
    if ((idx[i] < 0) || (idx[i] >= x->map->n)) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT, "Permutation index %D is out of bounds: %D", i, idx[i]);
    }
  }
#endif
  if (!inv) {
    for(i = 0; i < x->map->n; i++) newArray[i]      = array[idx[i]];
  } else {
    for(i = 0; i < x->map->n; i++) newArray[idx[i]] = array[i];
  }
  ierr = VecRestoreArray(x, &array);CHKERRQ(ierr);
  ierr = ISRestoreIndices(row, &idx);CHKERRQ(ierr);
  ierr = VecReplaceArray(x, newArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecEqual"
/*@
   VecEqual - Compares two vectors.

   Collective on Vec

   Input Parameters:
+  vec1 - the first vector
-  vec2 - the second vector

   Output Parameter:
.  flg - PETSC_TRUE if the vectors are equal; PETSC_FALSE otherwise.

   Level: intermediate

   Concepts: equal^two vectors
   Concepts: vector^equality

@*/
PetscErrorCode  VecEqual(Vec vec1,Vec vec2,PetscBool  *flg)
{
  PetscScalar    *v1,*v2; 
  PetscErrorCode ierr; 
  PetscInt       n1,n2,N1,N2; 
  PetscInt       state1,state2; 
  PetscBool      flg1; 
 
  PetscFunctionBegin; 
  PetscValidHeaderSpecific(vec1,VEC_CLASSID,1); 
  PetscValidHeaderSpecific(vec2,VEC_CLASSID,2); 
  PetscValidPointer(flg,3); 
  if (vec1 == vec2) { 
    *flg = PETSC_TRUE; 
  } else { 
    ierr = VecGetSize(vec1,&N1);CHKERRQ(ierr); 
    ierr = VecGetSize(vec2,&N2);CHKERRQ(ierr); 
    if (N1 != N2) { 
      flg1 = PETSC_FALSE; 
    } else { 
      ierr = VecGetLocalSize(vec1,&n1);CHKERRQ(ierr); 
      ierr = VecGetLocalSize(vec2,&n2);CHKERRQ(ierr); 
      if (n1 != n2) { 
        flg1 = PETSC_FALSE; 
      } else { 
        ierr = PetscObjectStateQuery((PetscObject) vec1,&state1);CHKERRQ(ierr); 
        ierr = PetscObjectStateQuery((PetscObject) vec2,&state2);CHKERRQ(ierr); 
        ierr = VecGetArray(vec1,&v1);CHKERRQ(ierr); 
        ierr = VecGetArray(vec2,&v2);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        {
          PetscInt k;
          flg1 = PETSC_TRUE;
          for (k=0; k<n1; k++){
            if (PetscRealPart(v1[k]) != PetscRealPart(v2[k]) || PetscImaginaryPart(v1[k]) != PetscImaginaryPart(v2[k])){
              flg1 = PETSC_FALSE;
              break;
            }
          }
        }
#else 
        ierr = PetscMemcmp(v1,v2,n1*sizeof(PetscScalar),&flg1);CHKERRQ(ierr); 
#endif
        ierr = VecRestoreArray(vec1,&v1);CHKERRQ(ierr); 
        ierr = VecRestoreArray(vec2,&v2);CHKERRQ(ierr); 
        ierr = PetscObjectSetState((PetscObject) vec1,state1);CHKERRQ(ierr); 
        ierr = PetscObjectSetState((PetscObject) vec2,state2);CHKERRQ(ierr); 
      } 
    } 
    /* combine results from all processors */ 
    ierr = MPI_Allreduce(&flg1,flg,1,MPI_INT,MPI_MIN,((PetscObject)vec1)->comm);CHKERRQ(ierr); 
  } 
  PetscFunctionReturn(0); 
}



