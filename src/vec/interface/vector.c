/*
     Provides the interface functions for all vector operations.
   These are the vector functions the user calls.
*/
#include "vecimpl.h"    /*I "petscvec.h" I*/

/* Logging support */
int VEC_COOKIE = 0;
int VEC_View = 0, VEC_Max = 0, VEC_Min = 0, VEC_DotBarrier = 0, VEC_Dot = 0, VEC_MDotBarrier = 0, VEC_MDot = 0, VEC_TDot = 0, VEC_MTDot = 0, VEC_NormBarrier = 0;
int VEC_Norm = 0, VEC_Normalize = 0, VEC_Scale = 0, VEC_Copy = 0, VEC_Set = 0, VEC_AXPY = 0, VEC_AYPX = 0, VEC_WAXPY = 0, VEC_MAXPY = 0, VEC_Swap = 0, VEC_AssemblyBegin = 0;
int VEC_AssemblyEnd = 0, VEC_PointwiseMult = 0, VEC_SetValues = 0, VEC_Load = 0, VEC_ScatterBarrier = 0, VEC_ScatterBegin = 0, VEC_ScatterEnd = 0;
int VEC_SetRandom = 0, VEC_ReduceArithmetic = 0, VEC_ReduceBarrier = 0, VEC_ReduceCommunication = 0;

/* ugly globals for VecSetValue() and VecSetValueLocal() */
int         VecSetValue_Row = 0;
PetscScalar VecSetValue_Value = 0.0;

#undef __FUNCT__  
#define __FUNCT__ "VecSetTypeFromOptions_Private"
/*
  VecSetTypeFromOptions_Private - Sets the type of vector from user options. Defaults to a PETSc sequential vector on one
  processor and a PETSc MPI vector on more than one processor.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Level: intermediate

.keywords: Vec, set, options, database, type
.seealso: VecSetFromOptions(), VecSetType()
*/
static int VecSetTypeFromOptions_Private(Vec vec)
{
  PetscTruth opt;
  const char *defaultType;
  char       typeName[256];
  int        size;
  int        ierr;

  PetscFunctionBegin;
  if (vec->type_name) {
    defaultType = vec->type_name;
  } else {
    ierr = MPI_Comm_size(vec->comm, &size);                                                           CHKERRQ(ierr);
    if (size > 1) {
      defaultType = VECMPI;
    } else {
      defaultType = VECSEQ;
    }
  }

  if (!VecRegisterAllCalled) {
    ierr = VecRegisterAll(PETSC_NULL);                                                                    CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-vec_type","Vector type","VecSetType",VecList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = VecSetType(vec, typeName);                                                                     CHKERRQ(ierr);
  } else {
    ierr = VecSetType(vec, defaultType);                                                                  CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetFromOptions"
/*@C
  VecSetFromOptions - Configures the vector from the options database.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Notes:  To see all options, run your program with the -help option, or consult the users manual.
          Must be called after VecCreate() but before the vector is used.

  Level: beginner

  Concepts: vectors^setting options
  Concepts: vectors^setting type

.keywords: Vec, set, options, database
.seealso: VecCreate(), VecPrintHelp(), VecSetOptionsPrefix()
@*/
int VecSetFromOptions(Vec vec)
{
  PetscTruth opt;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);

  ierr = PetscOptionsBegin(vec->comm, vec->prefix, "Vector options", "Vec");                              CHKERRQ(ierr);

  /* Handle generic vector options */
  ierr = PetscOptionsHasName(PETSC_NULL, "-help", &opt);                                                  CHKERRQ(ierr);
  if (opt) {
    ierr = VecPrintHelp(vec);                                                                             CHKERRQ(ierr);
  }

  /* Handle vector type options */
  ierr = VecSetTypeFromOptions_Private(vec);                                                              CHKERRQ(ierr);

  /* Handle specific vector options */
  if (vec->ops->setfromoptions) {
    ierr = (*vec->ops->setfromoptions)(vec);                                                              CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();                                                                               CHKERRQ(ierr);

#if defined(__cplusplus) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && defined(PETSC_HAVE_CXX_NAMESPACE)
  ierr = VecESISetFromOptions(vec);                                                                       CHKERRQ(ierr);
#endif

  ierr = VecViewFromOptions(vec, vec->name);                                                              CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPrintHelp"
/*@
  VecPrintHelp - Prints some options for the Vec.

  Input Parameter:
. vec - The vector

  Options Database Keys:
$  -help, -h

  Level: intermediate

.keywords: Vec, help
.seealso: VecSetFromOptions()
@*/
int VecPrintHelp(Vec vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_COOKIE,1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetSizes"
/*@
  VecSetSizes - Sets the local and global sizes, and checks to determine compatibility

  Collective on Vec

  Input Parameters:
+ v - the vector
. n - the local size (or PETSC_DECIDE to have it set)
- N - the global size (or PETSC_DECIDE)

  Notes:
  n and N cannot be both PETSC_DECIDE
  If one processor calls this with N of PETSC_DECIDE then all processors must, otherwise the program will hang.

  Level: intermediate

.seealso: VecGetSize(), PetscSplitOwnership()
@*/
int VecSetSizes(Vec v, int n, int N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_COOKIE,1); 
  v->n = n;
  v->N = N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetBlockSize"
/*@
   VecSetBlockSize - Sets the blocksize for future calls to VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Collective on Vec

   Input Parameter:
+  v - the vector
-  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMappingBlocked(), VecGetBlockSize()

  Concepts: block size^vectors
@*/
int VecSetBlockSize(Vec v,int bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1); 
  if (bs <= 0) bs = 1;
  if (bs == v->bs) PetscFunctionReturn(0);
  if (v->bs != -1) SETERRQ2(PETSC_ERR_ARG_WRONGSTATE,"Cannot reset blocksize. Current size %d new %d",v->bs,bs);
  if (v->N != -1 && v->N % bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Vector length not divisible by blocksize %d %d",v->N,bs);
  if (v->n != -1 && v->n % bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local vector length not divisible by blocksize %d %d\n\
   Try setting blocksize before setting the vector type",v->n,bs);
  
  v->bs        = bs;
  v->bstash.bs = bs; /* use the same blocksize for the vec's block-stash */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetBlockSize"
/*@
   VecGetBlockSize - Gets the blocksize for the vector, i.e. what is used for VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameter:
.  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMappingBlocked(), VecSetBlockSize()

   Concepts: vector^block size
   Concepts: block^vector

@*/
int VecGetBlockSize(Vec v,int *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1); 
  PetscValidIntPointer(bs,2);
  *bs = v->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecValid"
/*@
   VecValid - Checks whether a vector object is valid.

   Not Collective

   Input Parameter:
.  v - the object to check

   Output Parameter:
   flg - flag indicating vector status, either
   PETSC_TRUE if vector is valid, or PETSC_FALSE otherwise.

   Level: developer

@*/
int VecValid(Vec v,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg,2);
  if (!v)                           *flg = PETSC_FALSE;
  else if (v->cookie != VEC_COOKIE) *flg = PETSC_FALSE;
  else                              *flg = PETSC_TRUE;
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
.  alpha - the dot product

   Performance Issues:
+    per-processor memory bandwidth
.    interprocessor latency
-    work load inbalance that causes certain processes to arrive much earlier than
     others

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
int VecDot(Vec x,Vec y,PetscScalar *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBarrierBegin(VEC_DotBarrier,x,y,0,0,x->comm);CHKERRQ(ierr);
  ierr = (*x->ops->dot)(x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventBarrierEnd(VEC_DotBarrier,x,y,0,0,x->comm);CHKERRQ(ierr);
  /*
     The next block is for incremental debugging
  */
  if (PetscCompare) {
    int flag;
    ierr = MPI_Comm_compare(PETSC_COMM_WORLD,x->comm,&flag);CHKERRQ(ierr);
    if (flag != MPI_UNEQUAL) {
      ierr = PetscCompareScalar(*val);CHKERRQ(ierr);
    }
  }
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
+    per-processor memory bandwidth
.    interprocessor latency
-    work load inbalance that causes certain processes to arrive much earlier than
     others

   Compile Option:
   PETSC_HAVE_SLOW_NRM2 will cause a C (loop unrolled) version of the norm to be used, rather
 than the BLAS. This should probably only be used when one is using the FORTRAN BLAS routines 
 (as opposed to vendor provided) because the FORTRAN BLAS NRM2() routine is very slow. 

   Concepts: norm
   Concepts: vector^norm

.seealso: VecDot(), VecTDot(), VecNorm(), VecDotBegin(), VecDotEnd(), 
          VecNormBegin(), VecNormEnd()

@*/
int VecNorm(Vec x,NormType type,PetscReal *val)  
{
  PetscTruth flg;
  int        type_id, ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidDoublePointer(val,3);
  PetscValidType(x,1);

  /*
   * Cached data?
   */
  ierr = VecNormComposedDataID(type,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id,*val,flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);
  

  ierr = PetscLogEventBarrierBegin(VEC_NormBarrier,x,0,0,0,x->comm);CHKERRQ(ierr);
  ierr = (*x->ops->norm)(x,type,val);CHKERRQ(ierr);
  ierr = PetscLogEventBarrierEnd(VEC_NormBarrier,x,0,0,0,x->comm);CHKERRQ(ierr);

  /*
     The next block is for incremental debugging
  */
  if (PetscCompare) {
    int flag;
    ierr = MPI_Comm_compare(PETSC_COMM_WORLD,x->comm,&flag);CHKERRQ(ierr);
    if (flag != MPI_UNEQUAL) {
      ierr = PetscCompareDouble(*val);CHKERRQ(ierr);
    }
  }

  if (type!=NORM_1_AND_2) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id,*val);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNormRegisterComposedDataID"
int VecNormComposedDataID(NormType type,int *type_id)
{
  static int id_norm1=-1,id_norm2=-1,id_normInf=-1,id_normF=-1,id_norm12=-1;
  int ierr;
  PetscFunctionBegin;
  switch (type) {
  case NORM_1 :
    if (id_norm1==-1) {
      ierr = PetscRegisterComposedData(&id_norm1); CHKERRQ(ierr);}
    *type_id = id_norm1; break;
  case NORM_2 :
    if (id_norm2==-1) {
      ierr = PetscRegisterComposedData(&id_norm2); CHKERRQ(ierr);}
    *type_id = id_norm2; break;
  case NORM_1_AND_2 :
    /* we don't handle this one yet */
    if (id_norm1==-1) {
      ierr = PetscRegisterComposedData(&id_norm1); CHKERRQ(ierr);}
    if (id_norm2==-1) {
      ierr = PetscRegisterComposedData(&id_norm2); CHKERRQ(ierr);}
    *type_id = id_norm12; break;
  case NORM_INFINITY :
    if (id_normInf==-1) {
      ierr = PetscRegisterComposedData(&id_normInf); CHKERRQ(ierr);}
    *type_id = id_normInf; break;
  case NORM_FROBENIUS :
    if (id_normF==-1) {
      ierr = PetscRegisterComposedData(&id_normF); CHKERRQ(ierr);}
    *type_id = id_normF; break;
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
int VecNormalize (Vec x,PetscReal *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidScalarPointer(val,2);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_Normalize,x,0,0,0);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,val); CHKERRQ(ierr);
  if (*val == 0.0) {
    PetscLogInfo(x,"Vector of zero norm can not be normalized; Returning only the zero norm");
  } else {
    PetscScalar tmp = 1.0/(*val);
    ierr = VecScale(&tmp,x);CHKERRQ(ierr);
  }

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
int VecMax(Vec x,int *p,PetscReal *val)
{
  int ierr;

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
int VecMin(Vec x,int *p,PetscReal *val)
{
  int ierr;

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
int VecTDot(Vec x,Vec y,PetscScalar *val) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidScalarPointer(val,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

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
int VecScale (const PetscScalar *alpha,Vec x)
{
  PetscReal  scale,norm1=0.0,norm2=0.0,normInf=0.0,normF=0.0;
  PetscTruth flg1,flg2,flgInf,flgF;
  int        type_id1,type_id2,type_idInf,type_idF,ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidType(x,2);
  ierr = PetscLogEventBegin(VEC_Scale,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->scale)(alpha,x);CHKERRQ(ierr);

  /*
   * Update cached data
   */
  /* see if we have cached norms */
  /* 1 */
  ierr = VecNormComposedDataID(NORM_1,&type_id1); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id1,norm1,flg1);CHKERRQ(ierr);
  /* 2 */
  ierr = VecNormComposedDataID(NORM_2,&type_id2); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id2,norm2,flg2);CHKERRQ(ierr);
  /* inf */
  ierr = VecNormComposedDataID(NORM_INFINITY,&type_idInf); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_idInf,normInf,flgInf);CHKERRQ(ierr);
  /* frobenius */
  ierr = VecNormComposedDataID(NORM_FROBENIUS,&type_idF); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_idF,normF,flgF);CHKERRQ(ierr);

  /* in general we consider this object touched */
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);

  /* however, norms can be simply updated */
  scale = PetscAbsScalar(*alpha);
  /* 1 */
  if (flg1) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id1,scale*norm1);CHKERRQ(ierr);
  }
  /* 2 */
  if (flg2) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id2,scale*norm2);CHKERRQ(ierr);
  }
  /* inf */
  if (flgInf) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_idInf,scale*normInf);CHKERRQ(ierr);
  }
  /* frobenius */
  if (flgF) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_idF,scale*normF);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(VEC_Scale,x,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy"
/*@
   VecCopy - Copies a vector. 

   Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  y - the copy

   Notes:
   For default parallel PETSc vectors, both x and y must be distributed in
   the same manner; local copies are done.

   Level: beginner

.seealso: VecDuplicate()
@*/
int VecCopy(Vec x,Vec y)
{
  PetscTruth flg;
  PetscReal  norm=0.0;
  int        type_id,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameComm(x,1,y,2);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->copy)(x,y);CHKERRQ(ierr);

  /*
   * Update cached data
   */
  /* in general we consider this object touched */
  ierr = PetscObjectIncreaseState((PetscObject)y); CHKERRQ(ierr);
  /* however, norms can be simply copied over */
  /* 2 */
  ierr = VecNormComposedDataID(NORM_2,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id,norm,flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_id,norm);CHKERRQ(ierr);
  }
  /* 1 */
  ierr = VecNormComposedDataID(NORM_1,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id,norm,flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_id,norm);CHKERRQ(ierr);
  }
  /* inf */
  ierr = VecNormComposedDataID(NORM_INFINITY,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id,norm,flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_id,norm);CHKERRQ(ierr);
  }
  /* frobenius */
  ierr = VecNormComposedDataID(NORM_FROBENIUS,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id,norm,flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_id,norm);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSet"
/*@
   VecSet - Sets all components of a vector to a single scalar value. 

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x  - the vector

   Output Parameter:
.  x  - the vector

   Note:
   For a vector of dimension n, VecSet() computes
$     x[i] = alpha, for i=1,...,n,
   so that all vector entries then equal the identical
   scalar value, alpha.  Use the more general routine
   VecSetValues() to set different vector entries.

   Level: beginner

.seealso VecSetValues(), VecSetValuesBlocked(), VecSetRandom()

   Concepts: vector^setting to constant

@*/
int VecSet(const PetscScalar *alpha,Vec x) 
{
  PetscReal  val;
  int        type_id,ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidType(x,2);

  ierr = PetscLogEventBegin(VEC_Set,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->set)(alpha,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Set,x,0,0,0);CHKERRQ(ierr);

  /*
   * Update cached data
   */
  /* in general we consider this object touched */
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
  /* however, norms can be simply set */
  /* 1 */
  val = PetscAbsScalar(*alpha);
  ierr = VecNormComposedDataID(NORM_1,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id,x->N * val);CHKERRQ(ierr);
  /* inf */
  ierr = VecNormComposedDataID(NORM_INFINITY,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id,val);CHKERRQ(ierr);
  /* 2 */
  val = sqrt((double)x->N) * val;
  ierr = VecNormComposedDataID(NORM_2,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id,val);CHKERRQ(ierr);
  /* frobenius */
  ierr = VecNormComposedDataID(NORM_FROBENIUS,&type_id); CHKERRQ(ierr);
  ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id,val);CHKERRQ(ierr);

  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecSetRandom"
/*@C
   VecSetRandom - Sets all components of a vector to random numbers.

   Collective on Vec

   Input Parameters:
+  rctx - the random number context, formed by PetscRandomCreate(), or PETSC_NULL and
          it will create one internally.
-  x  - the vector

   Output Parameter:
.  x  - the vector

   Example of Usage:
.vb
     PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);
     VecSetRandom(rctx,x);
     PetscRandomDestroy(rctx);
.ve

   Level: intermediate

   Concepts: vector^setting to random
   Concepts: random^vector

.seealso: VecSet(), VecSetValues(), PetscRandomCreate(), PetscRandomDestroy()
@*/
int VecSetRandom(PetscRandom rctx,Vec x) 
{
  int         ierr;
  PetscRandom randObj = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_COOKIE,1);
  PetscValidType(x,2);

  if (!rctx) {
    MPI_Comm    comm;
    ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
    ierr = PetscRandomCreate(comm,RANDOM_DEFAULT,&randObj);CHKERRQ(ierr);
    rctx = randObj;
  }

  ierr = PetscLogEventBegin(VEC_SetRandom,x,rctx,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setrandom)(rctx,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetRandom,x,rctx,0,0);CHKERRQ(ierr);
  
  if (randObj) {
    ierr = PetscRandomDestroy(randObj);CHKERRQ(ierr);
  }
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
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

   Concepts: vector^BLAS
   Concepts: BLAS

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY()
@*/
int VecAXPY(const PetscScalar *alpha,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->axpy)(alpha,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)y); CHKERRQ(ierr);
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

   Concepts: BLAS
   Concepts: vector^BLAS

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY()
@*/
int VecAXPBY(const PetscScalar *alpha,const PetscScalar *beta,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidScalarPointer(beta,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidHeaderSpecific(y,VEC_COOKIE,4);
  PetscValidType(x,3);
  PetscValidType(y,4);
  PetscCheckSameTypeAndComm(x,3,y,4);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->axpby)(alpha,beta,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AXPY,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)y); CHKERRQ(ierr);
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

   Concepts: vector^BLAS
   Concepts: BLAS

.seealso: VecAXPY(), VecWAXPY()
@*/
int VecAYPX(const PetscScalar *alpha,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_AYPX,x,y,0,0);CHKERRQ(ierr);
  ierr =  (*x->ops->aypx)(alpha,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_AYPX,x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecSwap"
/*@
   VecSwap - Swaps the vectors x and y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Level: advanced

   Concepts: vector^swapping values

@*/
int VecSwap(Vec x,Vec y)
{
  PetscReal  norm1x=0.0,norm2x=0.0,normInfx=0.0,normFx=0.0;
  PetscReal  norm1y=0.0,norm2y=0.0,normInfy=0.0,normFy=0.0;
  PetscTruth flg1x,flg2x,flgInfx,flgFx;
  PetscTruth flg1y,flg2y,flgInfy,flgFy;
  int        type_id1,type_id2,type_idInf,type_idF;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);  
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_Swap,x,y,0,0);CHKERRQ(ierr);

  /* See if we have cached norms */
  /* 1 */
  ierr = VecNormComposedDataID(NORM_1,&type_id1); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id1,norm1x,flg1x);CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)y,type_id1,norm1y,flg1y);CHKERRQ(ierr);
  /* 2 */
  ierr = VecNormComposedDataID(NORM_2,&type_id2); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_id2,norm2x,flg2x);CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)y,type_id2,norm2y,flg2y);CHKERRQ(ierr);
  /* inf */
  ierr = VecNormComposedDataID(NORM_INFINITY,&type_idInf); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_idInf,normInfx,flgInfx);CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)y,type_idInf,normInfy,flgInfy);CHKERRQ(ierr);
  /* frobenius */
  ierr = VecNormComposedDataID(NORM_FROBENIUS,&type_idF); CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)x,type_idF,normFx,flgFx);CHKERRQ(ierr);
  ierr = PetscObjectGetRealComposedData((PetscObject)y,type_idF,normFy,flgFy);CHKERRQ(ierr);

  /* Do the actual swap */
  ierr = (*x->ops->swap)(x,y);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)y); CHKERRQ(ierr);

  /* Swap any cached norms */
  /* 1 */
  if (flg1x) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_id1,norm1x);CHKERRQ(ierr);
  }
  if (flg1y) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id1,norm1y);CHKERRQ(ierr);
  }
  /* 2 */
  if (flg2x) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_id2,norm2x);CHKERRQ(ierr);
  }
  if (flg2y) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_id2,norm2y);CHKERRQ(ierr);
  }
  /* inf */
  if (flgInfx) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_idInf,normInfx);CHKERRQ(ierr);
  }
  if (flgInfy) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_idInf,normInfy);CHKERRQ(ierr);
  }
  /* frobenius */
  if (flgFx) {
    ierr = PetscObjectSetRealComposedData((PetscObject)y,type_idF,normFx);CHKERRQ(ierr);
  }
  if (flgFy) {
    ierr = PetscObjectSetRealComposedData((PetscObject)x,type_idF,normFy);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(VEC_Swap,x,y,0,0);CHKERRQ(ierr);

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

   Concepts: vector^BLAS
   Concepts: BLAS

.seealso: VecAXPY(), VecAYPX()
@*/
int VecWAXPY(const PetscScalar *alpha,Vec x,Vec y,Vec w)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidHeaderSpecific(w,VEC_COOKIE,4);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscValidType(w,4);
  PetscCheckSameTypeAndComm(x,2,y,3); 
  PetscCheckSameTypeAndComm(y,3,w,4);
  if (x->N != y->N || x->N != w->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n || x->n != w->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_WAXPY,x,y,w,0);CHKERRQ(ierr);
  ierr =  (*x->ops->waxpy)(alpha,x,y,w);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_WAXPY,x,y,w,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)w); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult"
/*@
   VecPointwiseMult - Computes the componentwise multiplication w = x*y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise multiply

.seealso: VecPointwiseDivide()
@*/
int VecPointwiseMult(Vec x,Vec y,Vec w)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidHeaderSpecific(w,VEC_COOKIE,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscValidType(w,3);
  PetscCheckSameTypeAndComm(x,1,y,2);
  PetscCheckSameTypeAndComm(y,2,w,3);
  if (x->N != y->N || x->N != w->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n || x->n != w->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_PointwiseMult,x,y,w,0);CHKERRQ(ierr);
  ierr = (*x->ops->pointwisemult)(x,y,w);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_PointwiseMult,x,y,w,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)w); CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide"
/*@
   VecPointwiseDivide - Computes the componentwise division w = x/y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise divide

.seealso: VecPointwiseMult()
@*/
int VecPointwiseDivide(Vec x,Vec y,Vec w)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidHeaderSpecific(w,VEC_COOKIE,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscValidType(w,3);
  PetscCheckSameTypeAndComm(x,1,y,2);
  PetscCheckSameTypeAndComm(y,2,w,3);
  if (x->N != y->N || x->N != w->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n || x->n != w->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*x->ops->pointwisedivide)(x,y,w);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)w); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMaxPointwiseDivide"
/*@
   VecMaxPointwiseDivide - Computes the maximum of the componentwise division w = abs(x/y).

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  max - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

.seealso: VecPointwiseDivide(), VecPointwiseMult()
@*/
int VecMaxPointwiseDivide(Vec x,Vec y,PetscReal *max)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidDoublePointer(max,3);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  if (x->N != y->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != y->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*x->ops->maxpointwisedivide)(x,y,max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate"
/*@C
   VecDuplicate - Creates a new vector of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
.  v - a vector to mimic

   Output Parameter:
.  newv - location to put new vector

   Notes:
   VecDuplicate() does not copy the vector, but rather allocates storage
   for the new vector.  Use VecCopy() to copy a vector.

   Use VecDestroy() to free the space. Use VecDuplicateVecs() to get several
   vectors. 

   Level: beginner

.seealso: VecDestroy(), VecDuplicateVecs(), VecCreate(), VecCopy()
@*/
int VecDuplicate(Vec x,Vec *newv) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(newv,2);
  PetscValidType(x,1);
  ierr = (*x->ops->duplicate)(x,newv);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)*newv); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy"
/*@C
   VecDestroy - Destroys a vector.

   Collective on Vec

   Input Parameters:
.  v  - the vector

   Level: beginner

.seealso: VecDuplicate(), VecDestroyVecs()
@*/
int VecDestroy(Vec v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  if (--v->refct > 0) PetscFunctionReturn(0);
  /* destroy the internal part */
  if (v->ops->destroy) {
    ierr = (*v->ops->destroy)(v);CHKERRQ(ierr);
  }
  /* destroy the external/common part */
  if (v->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(v->mapping);CHKERRQ(ierr);
  }
  if (v->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(v->bmapping);CHKERRQ(ierr);
  }
  if (v->map) {
    ierr = PetscMapDestroy(v->map);CHKERRQ(ierr);
  }
  PetscLogObjectDestroy(v);
  PetscHeaderDestroy(v); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicateVecs"
/*@C
   VecDuplicateVecs - Creates several vectors of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
+  m - the number of vectors to obtain
-  v - a vector to mimic

   Output Parameter:
.  V - location to put pointer to array of vectors

   Notes:
   Use VecDestroyVecs() to free the space. Use VecDuplicate() to form a single
   vector.

   Fortran Note:
   The Fortran interface is slightly different from that given below, it 
   requires one to pass in V a Vec (integer) array of size at least m.
   See the Fortran chapter of the users manual and petsc/src/vec/examples for details.

   Level: intermediate

.seealso:  VecDestroyVecs(), VecDuplicate(), VecCreate(), VecDuplicateVecsF90()
@*/
int VecDuplicateVecs(Vec v,int m,Vec *V[])  
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidPointer(V,3);
  PetscValidType(v,1);
  ierr = (*v->ops->duplicatevecs)(v, m,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroyVecs"
/*@C
   VecDestroyVecs - Frees a block of vectors obtained with VecDuplicateVecs().

   Collective on Vec

   Input Parameters:
+  vv - pointer to array of vector pointers
-  m - the number of vectors previously obtained

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual and 
   petsc/src/vec/examples for details.

   Level: intermediate

.seealso: VecDuplicateVecs(), VecDestroyVecsF90()
@*/
int VecDestroyVecs(const Vec vv[],int m)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidPointer(vv,1);
  PetscValidHeaderSpecific(*vv,VEC_COOKIE,1);
  PetscValidType(*vv,1);
  ierr = (*(*vv)->ops->destroyvecs)(vv,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecSetValues"
/*@
   VecSetValues - Inserts or adds values into certain locations of a vector. 

   Input Parameters:
   Not Collective

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

   Negative indices may be passed in ix, these rows are 
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

   Level: beginner

   Concepts: vector^setting values

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesLocal(),
           VecSetValue(), VecSetValuesBlocked()
@*/
int VecSetValues(Vec x,int ni,const int ix[],const PetscScalar y[],InsertMode iora) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setvalues)(x,ni,ix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
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
int VecSetValuesBlocked(Vec x,int ni,const int ix[],const PetscScalar y[],InsertMode iora) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)(x,ni,ix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetLocalToGlobalMapping"
/*@
   VecSetLocalToGlobalMapping - Sets a local numbering to global numbering used
   by the routine VecSetValuesLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes: 
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

   Level: intermediate

   Concepts: vector^setting values with local numbering

seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMappingBlocked(), VecSetValuesBlockedLocal()
@*/
int VecSetLocalToGlobalMapping(Vec x,ISLocalToGlobalMapping mapping)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,2);

  if (x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for vector");
  }

  if (x->ops->setlocaltoglobalmapping) {
    ierr = (*x->ops->setlocaltoglobalmapping)(x,mapping);CHKERRQ(ierr);
  } else {
    x->mapping = mapping;
    ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetLocalToGlobalMappingBlock"
/*@
   VecSetLocalToGlobalMappingBlock - Sets a local numbering to global numbering used
   by the routine VecSetValuesBlockedLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes: 
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

   Level: intermediate

   Concepts: vector^setting values blocked with local numbering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMapping(), VecSetValuesBlockedLocal()
@*/
int VecSetLocalToGlobalMappingBlock(Vec x,ISLocalToGlobalMapping mapping)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,2);

  if (x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for vector");
  }
  x->bmapping = mapping;
  ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
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
int VecSetValuesLocal(Vec x,int ni,const int ix[],const PetscScalar y[],InsertMode iora) 
{
  int ierr,lixp[128],*lix = lixp;

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
      ierr = PetscMalloc(ni*sizeof(int),&lix);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingApply(x->mapping,ni,(int*)ix,lix);CHKERRQ(ierr);
    ierr = (*x->ops->setvalues)(x,ni,lix,y,iora);CHKERRQ(ierr);
    if (ni > 128) {
      ierr = PetscFree(lix);CHKERRQ(ierr);
    }
  } else {
    ierr = (*x->ops->setvalueslocal)(x,ni,ix,y,iora);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
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
           VecSetLocalToGlobalMappingBlocked()
@*/
int VecSetValuesBlockedLocal(Vec x,int ni,const int ix[],const PetscScalar y[],InsertMode iora) 
{
  int ierr,lixp[128],*lix = lixp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(ix,3);
  PetscValidScalarPointer(y,4);
  PetscValidType(x,1);
  if (!x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with VecSetLocalToGlobalMappingBlocked()");
  }
  if (ni > 128) {
    ierr = PetscMalloc(ni*sizeof(int),&lix);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(x->bmapping,ni,(int*)ix,lix);CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)(x,ni,lix,y,iora);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetValues,x,0,0,0);CHKERRQ(ierr);
  if (ni > 128) {
    ierr = PetscFree(lix);CHKERRQ(ierr);
  }
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAssemblyBegin"
/*@
   VecAssemblyBegin - Begins assembling the vector.  This routine should
   be called after completing all calls to VecSetValues().

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Level: beginner

   Concepts: assembly^vectors

.seealso: VecAssemblyEnd(), VecSetValues()
@*/
int VecAssemblyBegin(Vec vec)
{
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);

  ierr = PetscOptionsHasName(vec->prefix,"-vec_view_stash",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecStashView(vec,PETSC_VIEWER_STDOUT_(vec->comm));CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(VEC_AssemblyBegin,vec,0,0,0);CHKERRQ(ierr);
  if (vec->ops->assemblybegin) {
    ierr = (*vec->ops->assemblybegin)(vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_AssemblyBegin,vec,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)vec); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAssemblyEnd"
/*@
   VecAssemblyEnd - Completes assembling the vector.  This routine should
   be called after VecAssemblyBegin().

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Options Database Keys:
+  -vec_view - Prints vector in ASCII format
.  -vec_view_matlab - Prints vector in ASCII Matlab format to stdout
.  -vec_view_matlab_file - Prints vector in Matlab format to matlaboutput.mat
.  -vec_view_draw - Activates vector viewing using drawing tools
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
.  -vec_view_socket - Activates vector viewing using a socket
-  -vec_view_ams - Activates vector viewing using the ALICE Memory Snooper (AMS)
 
   Level: beginner

.seealso: VecAssemblyBegin(), VecSetValues()
@*/
int VecAssemblyEnd(Vec vec)
{
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  ierr = PetscLogEventBegin(VEC_AssemblyEnd,vec,0,0,0);CHKERRQ(ierr);
  PetscValidType(vec,1);
  if (vec->ops->assemblyend) {
    ierr = (*vec->ops->assemblyend)(vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_AssemblyEnd,vec,0,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(vec->comm,vec->prefix,"Vector Options","Vec");CHKERRQ(ierr);
    ierr = PetscOptionsName("-vec_view","Print vector to stdout","VecView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_STDOUT_(vec->comm));CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-vec_view_matlab","Print vector to stdout in a format Matlab can read","VecView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(vec->comm),PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = VecView(vec,PETSC_VIEWER_STDOUT_(vec->comm));CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(vec->comm));CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
    ierr = PetscOptionsName("-vec_view_matlab_file","Print vector to matlaboutput.mat format Matlab can read","VecView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_MATLAB_(vec->comm));CHKERRQ(ierr);
    }
#endif
    ierr = PetscOptionsName("-vec_view_socket","Send vector to socket (can be read from matlab)","VecView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_SOCKET_(vec->comm));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_SOCKET_(vec->comm));CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-vec_view_binary","Save vector to file in binary format","VecView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_BINARY_(vec->comm));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_BINARY_(vec->comm));CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_AMS)
    ierr = PetscOptionsName("-vec_view_ams","View vector using AMS","VecView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_AMS_(vec->comm));CHKERRQ(ierr);
    }
#endif
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* These invoke PetscDrawGetDraw which invokes PetscOptionsBegin/End, */
  /* hence they should not be inside the above PetscOptionsBegin/End block. */
  ierr = PetscOptionsHasName(vec->prefix,"-vec_view_draw",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(vec,PETSC_VIEWER_DRAW_(vec->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(vec->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(vec->prefix,"-vec_view_draw_lg",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerSetFormat(PETSC_VIEWER_DRAW_(vec->comm),PETSC_VIEWER_DRAW_LG);CHKERRQ(ierr);
    ierr = VecView(vec,PETSC_VIEWER_DRAW_(vec->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(vec->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecMTDot"
/*@C
   VecMTDot - Computes indefinite vector multiple dot products. 
   That is, it does NOT use the complex conjugate.

   Collective on Vec

   Input Parameters:
+  nv - number of vectors
.  x - one vector
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
int VecMTDot(int nv,Vec x,const Vec y[],PetscScalar *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(y,3);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,3);
  PetscValidScalarPointer(val,4);
  PetscValidType(x,2);
  PetscValidType(*y,3);
  PetscCheckSameTypeAndComm(x,2,*y,3);
  if (x->N != (*y)->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != (*y)->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_MTDot,x,*y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mtdot)(nv,x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MTDot,x,*y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMDot"
/*@C
   VecMDot - Computes vector multiple dot products. 

   Collective on Vec

   Input Parameters:
+  nv - number of vectors
.  x - one vector
-  y - array of vectors. 

   Output Parameter:
.  val - array of the dot products

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
int VecMDot(int nv,Vec x,const Vec y[],PetscScalar *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,2); 
  PetscValidPointer(y,3);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,3);
  PetscValidScalarPointer(val,4);
  PetscValidType(x,2);
  PetscValidType(*y,3);
  PetscCheckSameTypeAndComm(x,2,*y,3);
  if (x->N != (*y)->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->n != (*y)->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBarrierBegin(VEC_MDotBarrier,x,*y,0,0,x->comm);CHKERRQ(ierr);
  ierr = (*x->ops->mdot)(nv,x,y,val);CHKERRQ(ierr);
  ierr = PetscLogEventBarrierEnd(VEC_MDotBarrier,x,*y,0,0,x->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY"
/*@C
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
int  VecMAXPY(int nv,const PetscScalar *alpha,Vec y,Vec *x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(alpha,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidPointer(x,3);
  PetscValidHeaderSpecific(*x,VEC_COOKIE,3);
  PetscValidType(y,3);
  PetscValidType(*x,4);
  PetscCheckSameTypeAndComm(y,3,*x,4);
  if (y->N != (*x)->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (y->n != (*x)->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_MAXPY,*x,y,0,0);CHKERRQ(ierr);
  ierr = (*y->ops->maxpy)(nv,alpha,y,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_MAXPY,*x,y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)y); CHKERRQ(ierr);
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
$    int         ierr
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
int VecGetArray_Private(Vec x,PetscScalar *a[])
{
  int ierr;

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
int VecGetArrays(const Vec x[],int n,PetscScalar **a[])
{
  int         i,ierr;
  PetscScalar **q;

  PetscFunctionBegin;
  PetscValidPointer(x,1);
  PetscValidHeaderSpecific(*x,VEC_COOKIE,1);
  PetscValidPointer(a,3);
  if (n <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Must get at least one array n = %d",n);
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
int VecRestoreArrays(const Vec x[],int n,PetscScalar **a[])
{
  int         i,ierr;
  PetscScalar **q = *a;

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
$    int         ierr
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
int VecRestoreArray_Private(Vec x,PetscScalar *a[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  if (a) PetscValidPointer(a,2);
  PetscValidType(x,1);
#if defined(PETSC_USE_BOPT_g)
  CHKMEMQ;
#endif
  if (x->ops->restorearray) {
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  }
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "VecViewFromOptions"
/*@
  VecViewFromOptions - This function visualizes the vector based upon user options.

  Collective on Vec

  Input Parameters:
. vec   - The vector
. title - The title

  Level: intermediate

.keywords: Vec, view, options, database
.seealso: VecSetFromOptions(), VecView()
@*/
int VecViewFromOptions(Vec vec, char *title)
{
  PetscViewer viewer;
  PetscDraw   draw;
  PetscTruth  opt;
  char       *titleStr;
  char        typeName[1024];
  char        fileName[PETSC_MAX_PATH_LEN];
  int         len;
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(vec->prefix, "-vec_view", &opt);                                             CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscOptionsGetString(vec->prefix, "-vec_view", typeName, 1024, &opt);                         CHKERRQ(ierr);
    ierr = PetscStrlen(typeName, &len);                                                                   CHKERRQ(ierr);
    if (len > 0) {
      ierr = PetscViewerCreate(vec->comm, &viewer);                                                       CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, typeName);                                                        CHKERRQ(ierr);
      ierr = PetscOptionsGetString(vec->prefix, "-vec_view_file", fileName, 1024, &opt);                  CHKERRQ(ierr);
      if (opt == PETSC_TRUE) {
        ierr = PetscViewerSetFilename(viewer, fileName);                                                  CHKERRQ(ierr);
      } else {
        ierr = PetscViewerSetFilename(viewer, vec->name);                                                 CHKERRQ(ierr);
      }
      ierr = VecView(vec, viewer);                                                                        CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);                                                                    CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);                                                                  CHKERRQ(ierr);
    } else {
      ierr = VecView(vec, PETSC_VIEWER_STDOUT_(vec->comm));                                               CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsHasName(vec->prefix, "-vec_view_draw", &opt);                                        CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscViewerDrawOpen(vec->comm, 0, 0, 0, 0, 300, 300, &viewer);                                 CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, 0, &draw);                                                      CHKERRQ(ierr);
    if (title != PETSC_NULL) {
      titleStr = title;
    } else {
      ierr = PetscObjectName((PetscObject) vec);                                                          CHKERRQ(ierr) ;
      titleStr = vec->name;
    }
    ierr = PetscDrawSetTitle(draw, titleStr);                                                             CHKERRQ(ierr);
    ierr = VecView(vec, viewer);                                                                          CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);                                                                      CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);                                                                          CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);                                                                    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView"
/*@C
   VecView - Views a vector object. 

   Collective on Vec

   Input Parameters:
+  v - the vector
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the vector is printed using the 
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs vector to a specified file
.    PetscViewerBinaryOpen() - Outputs vector in binary to a
         specified file; corresponding input uses VecLoad()
.    PetscViewerDrawOpen() - Outputs vector to an X window display
-    PetscViewerSocketOpen() - Outputs vector to Socket viewer

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints vector contents
.    PETSC_VIEWER_ASCII_MATLAB - prints vector contents in Matlab format
.    PETSC_VIEWER_ASCII_INDEX - prints vector contents, including indices of vector elements
-    PETSC_VIEWER_ASCII_COMMON - prints vector contents, using a 
         format common among all vector types

   Level: beginner

   Concepts: vector^printing
   Concepts: vector^saving to disk

.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscDrawLGCreate(),
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), VecLoad(), PetscViewerCreate(),
          PetscRealView(), PetscScalarView(), PetscIntView()
@*/
int VecView(Vec vec,PetscViewer viewer)
{
  int               ierr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(vec->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(vec,1,viewer,2);
  if (vec->stash.n || vec->bstash.n) SETERRQ(1,"Must call VecAssemblyBegin/End() before viewing this vector");

  /*
     Check if default viewer has been overridden, but user request it anyways
  */
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (vec->ops->viewnative && format == PETSC_VIEWER_NATIVE) {
    ierr   = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = (*vec->ops->viewnative)(vec,viewer);CHKERRQ(ierr);
    ierr   = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  } else {
    ierr = (*vec->ops->view)(vec,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetSize"
/*@
   VecGetSize - Returns the global number of elements of the vector.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  size - the global length of the vector

   Level: beginner

   Concepts: vector^local size

.seealso: VecGetLocalSize()
@*/
int VecGetSize(Vec x,int *size)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getsize)(x,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetLocalSize"
/*@
   VecGetLocalSize - Returns the number of elements of the vector stored 
   in local memory. This routine may be implementation dependent, so use 
   with care.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  size - the length of the local piece of the vector

   Level: beginner

   Concepts: vector^size

.seealso: VecGetSize()
@*/
int VecGetLocalSize(Vec x,int *size)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getlocalsize)(x,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetOwnershipRange"
/*@C
   VecGetOwnershipRange - Returns the range of indices owned by 
   this processor, assuming that the vectors are laid out with the
   first n1 elements on the first processor, next n2 elements on the
   second, etc.  For certain parallel layouts this range may not be 
   well defined. 

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
+  low - the first local element, pass in PETSC_NULL if not interested
-  high - one more than the last local element, pass in PETSC_NULL if not interested

   Note:
   The high argument is one more than the last element stored locally.

   Fortran: PETSC_NULL_INTEGER should be used instead of PETSC_NULL

   Level: beginner

   Concepts: ownership^of vectors
   Concepts: vector^ownership of elements

@*/
int VecGetOwnershipRange(Vec x,int *low,int *high)
{
  int      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (low) PetscValidIntPointer(low,2);
  if (high) PetscValidIntPointer(high,3);
  ierr = PetscMapGetLocalRange(x->map,low,high);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetPetscMap"
/*@C
   VecGetPetscMap - Returns the map associated with the vector

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  map - the map

   Level: developer

@*/
int VecGetPetscMap(Vec x,PetscMap *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(map,2);
  PetscValidType(x,1);
  *map = x->map;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetOption"
/*@
   VecSetOption - Sets an option for controling a vector's behavior.

   Collective on Vec

   Input Parameter:
+  x - the vector
-  op - the option

   Supported Options:
+     VEC_IGNORE_OFF_PROC_ENTRIES, which causes VecSetValues() to ignore 
      entries destined to be stored on a seperate processor. This can be used
      to eliminate the global reduction in the VecAssemblyXXXX() if you know 
      that you have only used VecSetValues() to set local elements
-   VEC_TREAT_OFF_PROC_ENTRIES restores the treatment of off processor entries.

   Level: intermediate

@*/
int VecSetOption(Vec x,VecOption op)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (x->ops->setoption) {
    ierr = (*x->ops->setoption)(x,op);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicateVecs_Default"
/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
int VecDuplicateVecs_Default(Vec w,int m,Vec *V[])
{
  int  i,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidPointer(V,3);
  if (m <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %d",m);
  ierr = PetscMalloc(m*sizeof(Vec*),V);CHKERRQ(ierr);
  for (i=0; i<m; i++) {ierr = VecDuplicate(w,*V+i);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroyVecs_Default"
int VecDestroyVecs_Default(const Vec v[], int m)
{
  int i,ierr;

  PetscFunctionBegin;
  PetscValidPointer(v,1);
  if (m <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %d",m);
  for (i=0; i<m; i++) {ierr = VecDestroy(v[i]);CHKERRQ(ierr);}
  ierr = PetscFree((Vec*)v);CHKERRQ(ierr);
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
int VecPlaceArray(Vec vec,const PetscScalar array[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  PetscValidScalarPointer(array,2);
  if (vec->ops->placearray) {
    ierr = (*vec->ops->placearray)(vec,array);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Cannot place array in this type of vector");
  }
  ierr = PetscObjectIncreaseState((PetscObject)vec); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecResetArray"
/*@
   VecResetArray - Resets a vector to use its default memory. Call this 
   after the use of VecPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray()

@*/
int VecResetArray(Vec vec)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (vec->ops->resetarray) {
    ierr = (*vec->ops->resetarray)(vec);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Cannot reset array in this type of vector");
  }
  ierr = PetscObjectIncreaseState((PetscObject)vec); CHKERRQ(ierr);
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
int VecReplaceArray(Vec vec,const PetscScalar array[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (vec->ops->replacearray) {
    ierr = (*vec->ops->replacearray)(vec,array);CHKERRQ(ierr);
 } else {
    SETERRQ(1,"Cannot replace array in this type of vector");
  }
  ierr = PetscObjectIncreaseState((PetscObject)vec); CHKERRQ(ierr);
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
    call VecSet(alpha,y(2),ierr)
    call VecSet(alpha,y(2),ierr)
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
   
    Notes:
    Not yet supported for all F90 compilers

    Level: beginner

.seealso:  VecGetArrayF90(), VecGetArray(), VecRestoreArray()

M*/

/*MC
    VecDestroyVecsF90 - Frees a block of vectors obtained with VecDuplicateVecsF90().

    Synopsis:
    VecDestroyVecsF90({Vec, pointer :: x(:)},integer n,integer ierr)

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

    Notes:
    Not yet supported for all F90 compilers

    Level: beginner

.seealso:  VecRestoreArrayF90(), VecGetArray(), VecRestoreArray()

M*/

#undef __FUNCT__  
#define __FUNCT__ "VecLoadIntoVector"
/*@C 
  VecLoadIntoVector - Loads a vector that has been stored in binary format
  with VecView().

  Collective on PetscViewer 

  Input Parameters:
+ viewer - binary file viewer, obtained from PetscViewerBinaryOpen()
- vec - vector to contain files values (must be of correct length)

  Level: intermediate

  Notes:
  The input file must contain the full global vector, as
  written by the routine VecView().

  Use VecLoad() to create the vector as the values are read in

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since VecLoad() and VecView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     int    VEC_FILE_COOKIE
     int    number of rows
     PetscScalar *values of all nonzeros
.ve

   Note for Cray users, the int's stored in the binary file are 32 bit
integers; not 64 as they are represented in the memory, so if you
write your own routines to read/write these binary files from the Cray
you need to adjust the integer sizes that you read in, see
PetscReadBinary() and PetscWriteBinary() to see how this may be
done.

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, nt and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscReadBinary()
and PetscWriteBinary() to see how this may be done.

   Concepts: vector^loading from file

.seealso: PetscViewerBinaryOpen(), VecView(), MatLoad(), VecLoad() 
@*/  
int VecLoadIntoVector(PetscViewer viewer,Vec vec)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscValidType(vec,2);
  if (!vec->ops->loadintovector) {
    SETERRQ(1,"Vector does not support load");
  }
  ierr = (*vec->ops->loadintovector)(viewer,vec);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)vec); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecReciprocal"
/*@
   VecReciprocal - Replaces each component of a vector by its reciprocal.

   Collective on Vec

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  v - the vector reciprocal

   Level: intermediate

   Concepts: vector^reciprocal

@*/
int VecReciprocal(Vec vec)
{
  int    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (!vec->ops->reciprocal) {
    SETERRQ(1,"Vector does not support reciprocal operation");
  }
  ierr = (*vec->ops->reciprocal)(vec);CHKERRQ(ierr);
  ierr = PetscObjectIncreaseState((PetscObject)vec); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetOperation"
int VecSetOperation(Vec vec,VecOperation op, void (*f)(void))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  /* save the native version of the viewer */
  if (op == VECOP_VIEW && !vec->ops->viewnative) {
    vec->ops->viewnative = vec->ops->view;
  }
  (((void(**)(void))vec->ops)[(int)op]) = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStashSetInitialSize"
/*@
   VecStashSetInitialSize - sets the sizes of the vec-stash, that is
   used during the assembly process to store values that belong to 
   other processors.

   Collective on Vec

   Input Parameters:
+  vec   - the vector
.  size  - the initial size of the stash.
-  bsize - the initial size of the block-stash(if used).

   Options Database Keys:
+   -vecstash_initial_size <size> or <size0,size1,...sizep-1>
-   -vecstash_block_initial_size <bsize> or <bsize0,bsize1,...bsizep-1>

   Level: intermediate

   Notes: 
     The block-stash is used for values set with VecSetValuesBlocked() while
     the stash is used for values set with VecSetValues()

     Run with the option -log_info and look for output of the form
     VecAssemblyBegin_MPIXXX:Stash has MM entries, uses nn mallocs.
     to determine the appropriate value, MM, to use for size and 
     VecAssemblyBegin_MPIXXX:Block-Stash has BMM entries, uses nn mallocs.
     to determine the value, BMM to use for bsize

   Concepts: vector^stash
   Concepts: stash^vector

.seealso: VecSetBlockSize(), VecSetValues(), VecSetValuesBlocked(), VecStashView()

@*/
int VecStashSetInitialSize(Vec vec,int size,int bsize)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  ierr = VecStashSetInitialSize_Private(&vec->stash,size);CHKERRQ(ierr);
  ierr = VecStashSetInitialSize_Private(&vec->bstash,bsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecStashView"
/*@
   VecStashView - Prints the entries in the vector stash and block stash.

   Collective on Vec

   Input Parameters:
+  vec   - the vector
-  viewer - the viewer

   Level: advanced

   Concepts: vector^stash
   Concepts: stash^vector

.seealso: VecSetBlockSize(), VecSetValues(), VecSetValuesBlocked()

@*/
int VecStashView(Vec v,PetscViewer viewer)
{
  int          ierr,rank,i,j;
  PetscTruth   match;
  VecStash     *s;
  PetscScalar  val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(v,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&match);CHKERRQ(ierr);
  if (!match) SETERRQ1(1,"Stash viewer only works with ASCII viewer not %s\n",((PetscObject)v)->type_name);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);
  s = &v->bstash;

  /* print block stash */
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Vector Block stash size %d block size %d\n",rank,s->n,s->bs);CHKERRQ(ierr);
  for (i=0; i<s->n; i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %d ",rank,s->idx[i]);CHKERRQ(ierr);
    for (j=0; j<s->bs; j++) {
      val = s->array[i*s->bs+j];
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"(%18.16e %18.16e) ",PetscRealPart(val),PetscImaginaryPart(val));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%18.16e ",val);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

  s = &v->stash;

  /* print basic stash */
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Vector stash size %d\n",rank,s->n);CHKERRQ(ierr);
  for (i=0; i<s->n; i++) {
    val = s->array[i];
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %d (%18.16e %18.16e) ",rank,s->idx[i],PetscRealPart(val),PetscImaginaryPart(val));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %d %18.16e\n",rank,s->idx[i],val);CHKERRQ(ierr);
#endif
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

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

   Level: beginner

  Notes:
   For a vector obtained from DACreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DAGetGhostCorners() while for
   DACreateGlobalVector() they are the corner indices from DAGetCorners(). In both cases
   the arguments from DAGet[Ghost}Corners() are reversed in the call to VecGetArray2d().
   
   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

   Concepts: vector^accessing local values as 2d array

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DAVecGetarray(), DAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray1d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
int VecGetArray2d(Vec x,int m,int n,int mstart,int nstart,PetscScalar **a[])
{
  int         i,ierr,N;
  PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n != N) SETERRQ3(1,"Local array size %d does not match 2d array dimensions %d by %d",N,m,n);
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

   Level: beginner

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
int VecRestoreArray2d(Vec x,int m,int n,int mstart,int nstart,PetscScalar **a[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,6);
  PetscValidType(x,1);
  ierr = PetscFree(*a + mstart);CHKERRQ(ierr);
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

   Level: beginner

  Notes:
   For a vector obtained from DACreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DAGetGhostCorners() while for
   DACreateGlobalVector() they are the corner indices from DAGetCorners(). 
   
   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: VecGetArray(), VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray(),
          VecRestoreArray2d(), DAVecGetArray(), DAVecRestoreArray(), VecGetArray3d(), VecRestoreArray3d(),
          VecGetArray2d(), VecRestoreArray1d(), VecGetArray4d(), VecRestoreArray4d()
@*/
int VecGetArray1d(Vec x,int m,int mstart,PetscScalar *a[])
{
  int ierr,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,4);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m != N) SETERRQ2(1,"Local array size %d does not match 1d array dimensions %d",N,m);
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

   Level: beginner

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
int VecRestoreArray1d(Vec x,int m,int mstart,PetscScalar *a[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  ierr = VecRestoreArray(x,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecConjugate"
/*@C
   VecConjugate - Conjugates a vector.

   Collective on Vec

   Input Parameters:
.  x - the vector

   Level: intermediate

   Concepts: vector^conjugate

@*/
int VecConjugate(Vec x)
{
#ifdef PETSC_USE_COMPLEX
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  ierr = (*x->ops->conjugate)(x);CHKERRQ(ierr);
  /* we need to copy norms here */
  ierr = PetscObjectIncreaseState((PetscObject)x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  return(0);
#endif
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
int VecGetArray3d(Vec x,int m,int n,int p,int mstart,int nstart,int pstart,PetscScalar ***a[])
{
  int         i,ierr,N,j;
  PetscScalar *aa,**b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  ierr = VecGetLocalSize(x,&N);CHKERRQ(ierr);
  if (m*n*p != N) SETERRQ4(1,"Local array size %d does not match 3d array dimensions %d by %d by %d",N,m,n,p);
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
int VecRestoreArray3d(Vec x,int m,int n,int p,int mstart,int nstart,int pstart,PetscScalar ***a[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidPointer(a,8);
  PetscValidType(x,1);
  ierr = PetscFree(*a + mstart);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern int VecStashGetInfo_Private(VecStash*,int*,int*);
#undef __FUNCT__  
#define __FUNCT__ "VecStashGetInfo"
/*@ 
   VecStashGetInfo - Gets how many values are currently in the vector stash, i.e. need
       to be communicated to other processors during the VecAssemblyBegin/End() process

    Not collective

   Input Parameter:
.   vec - the vector

   Output Parameters:
+   nstash   - the size of the stash
.   reallocs - the number of additional mallocs incurred.
.   bnstash   - the size of the block stash
-   breallocs - the number of additional mallocs incurred.in the block stash
 
   Level: advanced

.seealso: VecAssemblyBegin(), VecAssemblyEnd(), Vec, VecStashSetInitialSize(), VecStashView()
  
@*/
int VecStashGetInfo(Vec vec,int *nstash,int *reallocs,int *bnstash,int *brealloc)
{
  int ierr;
  PetscFunctionBegin;
  ierr = VecStashGetInfo_Private(&vec->stash,nstash,reallocs);CHKERRQ(ierr);
  ierr = VecStashGetInfo_Private(&vec->bstash,nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
