#define PETSCVEC_DLL
/*
   This file contains routines for Parallel vector operations.
 */
#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "VecDot_MPI"
PetscErrorCode VecDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_Seq(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot_MPI"
PetscErrorCode VecTDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_Seq(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,((PetscObject)xin)->comm);CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
EXTERN PetscErrorCode VecView_MPI_Draw(Vec,PetscViewer);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "VecPlaceArray_MPI"
PetscErrorCode VecPlaceArray_MPI(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  Vec_MPI        *v = (Vec_MPI *)vin->data;

  PetscFunctionBegin;
  if (v->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array = (PetscScalar *)a;
  if (v->localrep) {
    ierr = VecPlaceArray(v->localrep,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode VecGetValues_MPI(Vec,PetscInt,const PetscInt [],PetscScalar []);

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPI"
static PetscErrorCode VecDuplicate_MPI(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI *)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,v);CHKERRQ(ierr);

  /* use the map that exists aleady in win */
  ierr = PetscLayoutDestroy((*v)->map);CHKERRQ(ierr);
  (*v)->map = win->map;
  win->map->refcnt++;

  ierr = VecCreate_MPI_Private(*v,PETSC_TRUE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArrayPrivate(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArrayPrivate(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(*v,vw->localrep);CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }    

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;
  
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  if (win->mapping) {
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
    (*v)->mapping = win->mapping;
  }
  if (win->bmapping) {
    ierr = PetscObjectReference((PetscObject)win->bmapping);CHKERRQ(ierr);
    (*v)->bmapping = win->bmapping;
  }
  (*v)->map->bs    = win->map->bs;
  (*v)->bstash.bs = win->bstash.bs;

  PetscFunctionReturn(0);
}

extern PetscErrorCode VecSetOption_MPI(Vec,VecOption,PetscBool);
extern PetscErrorCode VecResetArray_MPI(Vec);

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMax_Seq"
static PetscErrorCode VecPointwiseMax_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = PetscMax(PetscRealPart(xx[i]),PetscRealPart(yy[i]));
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMin_Seq"
static PetscErrorCode VecPointwiseMin_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = PetscMin(PetscRealPart(xx[i]),PetscRealPart(yy[i]));
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMaxAbs_Seq"
static PetscErrorCode VecPointwiseMaxAbs_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = PetscMax(PetscAbsScalar(xx[i]),PetscAbsScalar(yy[i]));
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "../src/vec/vec/impls/seq/ftn-kernels/fxtimesy.h"
#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_Seq"
static PetscErrorCode VecPointwiseMult_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  if (ww == xx) {
    for (i=0; i<n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=0; i<n; i++) ww[i] *= xx[i];
  } else {
    /*  This was suppose to help on SGI but didn't really seem to
          PetscReal * PETSC_RESTRICT www = ww;
          PetscReal * PETSC_RESTRICT yyy = yy;
          PetscReal * PETSC_RESTRICT xxx = xx;
          for (i=0; i<n; i++) www[i] = xxx[i] * yyy[i];
    */
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  ierr = VecRestoreArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetRandom_Seq"
static PetscErrorCode VecSetRandom_Seq(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx = *(PetscScalar**)xin->data;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Seq"
static PetscErrorCode VecPointwiseDivide_Seq(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n,i;
  PetscScalar    *ww,*xx,*yy;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ww[i] = xx[i] / yy[i];
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecRestoreArrayPrivate3(win,&ww,xin,&xx,yin,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetSize_Seq"
static PetscErrorCode VecGetSize_Seq(Vec vin,PetscInt *size)
{
  PetscFunctionBegin;
  *size = vin->map->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecConjugate_Seq"
static PetscErrorCode VecConjugate_Seq(Vec xin)
{
  PetscScalar    *x;
  PetscInt       n = xin->map->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate(xin,&x);CHKERRQ(ierr);
  while (n-->0) {
    *x = PetscConj(*x);
    x++;
  }
  ierr = VecRestoreArrayPrivate(xin,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Seq"
static PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
    ierr = PetscMemcpy(ya,xa,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "VecSwap_Seq"
static PetscErrorCode VecSwap_Seq(Vec xin,Vec yin)
{
  PetscScalar    *ya, *xa;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
    BLASswap_(&bn,xa,&one,ya,&one);
    ierr = VecRestoreArrayPrivate2(xin,&xa,yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = { VecDuplicate_MPI, /* 1 */
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_MPI,
            VecMDot_MPI,
            VecNorm_MPI,
            VecTDot_MPI,
            VecMTDot_MPI,
            VecScale_Seq,
            VecCopy_Seq, /* 10 */
            VecSet_Seq,
            VecSwap_Seq,
            VecAXPY_Seq,
            VecAXPBY_Seq,
            VecMAXPY_Seq,
            VecAYPX_Seq,
            VecWAXPY_Seq,
            VecAXPBYPCZ_Seq,
            VecPointwiseMult_Seq,
            VecPointwiseDivide_Seq,
            VecSetValues_MPI, /* 20 */
            VecAssemblyBegin_MPI,
            VecAssemblyEnd_MPI,
            VecGetArray_Seq,
            VecGetSize_MPI,
            VecGetSize_Seq,
            VecRestoreArray_Seq,
            VecMax_MPI,
            VecMin_MPI,
            VecSetRandom_Seq,
            VecSetOption_MPI,
            VecSetValuesBlocked_MPI,
            VecDestroy_MPI,
            VecView_MPI,
            VecPlaceArray_MPI,
            VecReplaceArray_Seq,
            VecDot_Seq,
            VecTDot_Seq,
            VecNorm_Seq,
            VecMDot_Seq,
            VecMTDot_Seq,
	    VecLoad_Default,			
            VecReciprocal_Default,
            VecConjugate_Seq,
            0,
            0,
            VecResetArray_MPI,
            0,
            VecMaxPointwiseDivide_Seq,
            VecPointwiseMax_Seq,
            VecPointwiseMaxAbs_Seq,
            VecPointwiseMin_Seq,
  	    VecGetValues_MPI,
    	    0,
    	    0,
    	    0,
    	    0,
    	    0,
    	    0,
   	    VecStrideGather_Default,
   	    VecStrideScatter_Default
};

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPI_Private"
/*
    VecCreate_MPI_Private - Basic create routine called by VecCreate_MPI() (i.e. VecCreateMPI()),
    VecCreateMPIWithArray(), VecCreate_Shared() (i.e. VecCreateShared()), VecCreateGhost(),
    VecDuplicate_MPI(), VecCreateGhostWithArray(), VecDuplicate_MPI(), and VecDuplicate_Shared()

    If alloc is true and array is PETSC_NULL then this routine allocates the space, otherwise
    no space is allocated.
*/
PetscErrorCode VecCreate_MPI_Private(Vec v,PetscBool  alloc,PetscInt nghost,const PetscScalar array[])
{
  Vec_MPI        *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr           = PetscNewLog(v,Vec_MPI,&s);CHKERRQ(ierr);
  v->data        = (void*)s;
  ierr           = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  s->nghost      = nghost;
  v->mapping     = 0;
  v->bmapping    = 0;
  v->petscnative = PETSC_TRUE;

  if (v->map->bs == -1) v->map->bs = 1;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  s->array           = (PetscScalar *)array;
  s->array_allocated = 0;
  if (alloc && !array) {
    PetscInt n         = v->map->n+nghost;
    ierr               = PetscMalloc(n*sizeof(PetscScalar),&s->array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory(v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscMemzero(s->array,v->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array_allocated = s->array;
  }

  /* By default parallel vectors do not have local representation */
  s->localrep    = 0;
  s->localupdate = 0;

  v->stash.insertmode  = NOT_SET_VALUES;
  /* create the stashes. The block-size for bstash is set later when 
     VecSetValuesBlocked is called.
  */
  ierr = VecStashCreate_Private(((PetscObject)v)->comm,1,&v->stash);CHKERRQ(ierr);
  ierr = VecStashCreate_Private(((PetscObject)v)->comm,v->map->bs,&v->bstash);CHKERRQ(ierr); 
                                                        
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEnginePut_C","VecMatlabEnginePut_Default",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEngineGet_C","VecMatlabEngineGet_Default",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VECMPI - VECMPI = "mpi" - The basic parallel vector

   Options Database Keys:
. -vec_type mpi - sets the vector type to VECMPI during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMpiWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMpi()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPI"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_MPI(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_TRUE,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   VECSTANDARD = "standard" - A VECSEQ on one process and VECMPI on more than one process

   Options Database Keys:
. -vec_type standard - sets a vector type to standard on calls to VecSetFromOptions()

  Level: beginner

.seealso: VecCreateSeq(), VecCreateMPI()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_Standard"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Standard(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)v)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQ);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPI);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "VecCreateMPIWithArray"
/*@C
   VecCreateMPIWithArray - Creates a parallel, array-style vector,
   where the user provides the array space to store the vector values.

   Collective on MPI_Comm

   Input Parameters:
+  comm  - the MPI communicator to use
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  array - the user provided array to store the vector values

   Output Parameter:
.  vv - the vector
 
   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is PETSC_NULL, then VecPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

   Concepts: vectors^creating with array

.seealso: VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateMPIWithArray(MPI_Comm comm,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateGhostWithArray"
/*@C
   VecCreateGhostWithArray - Creates a parallel vector with ghost padding on each processor;
   the caller allocates the array space.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length 
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost points
.  ghosts - global indices of ghost points (or PETSC_NULL if not needed)
-  array - the space to store the vector values (as long as n + nghost)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)
 
   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation 
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

   Concepts: vectors^creating with array
   Concepts: vectors^ghosted

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), 
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhostWithArray(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  PetscErrorCode         ierr;
  Vec_MPI                *w;
  PetscScalar            *larray;
  IS                     from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt               rstart,i,*indices;

  PetscFunctionBegin;
  *vv = 0;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost,array);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArrayPrivate(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArrayPrivate(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   
  ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localupdate);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);

  /* set local to global mapping for ghosted vector */
  ierr = PetscMalloc((n+nghost)*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*vv,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    indices[i] = rstart + i;
  }
  for (i=0; i<nghost; i++) {
    indices[n+i] = ghosts[i];
  }
  ierr = ISLocalToGlobalMappingCreate(comm,n+nghost,indices,PETSC_OWN_POINTER,&ltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*vv,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateGhost"
/*@
   VecCreateGhost - Creates a parallel vector with ghost padding on each processor.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length 
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost points
-  ghosts - global indices of ghost points

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)
 
   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation 
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

   Concepts: vectors^ghosted

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhost(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostWithArray(comm,n,N,nghost,ghosts,0,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecCreateGhostBlockWithArray"
/*@C
   VecCreateGhostBlockWithArray - Creates a parallel vector with ghost padding on each processor;
   the caller allocates the array space. Indices in the ghost region are based on blocks.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator to use
.  bs - block size
.  n - local vector length 
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost blocks
.  ghosts - global indices of ghost blocks (or PETSC_NULL if not needed), counts are by block not by index
-  array - the space to store the vector values (as long as n + nghost*bs)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)
 
   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation 
   of the vector.

   n is the local vector size (total local size not the number of blocks) while nghost
   is the number of blocks in the ghost portion, i.e. the number of elements in the ghost
   portion is bs*nghost

   Level: advanced

   Concepts: vectors^creating ghosted
   Concepts: vectors^creating with array

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), 
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostWithArray(), VecCreateGhostBlocked()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhostBlockWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;
  Vec_MPI        *w;
  PetscScalar    *larray;
  IS             from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt       rstart,i,nb,*indices;

  PetscFunctionBegin;
  *vv = 0;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  if (n % bs)                 SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size must be a multiple of block size");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost*bs,array);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArrayPrivate(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+bs*nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = VecSetBlockSize(w->localrep,bs);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArrayPrivate(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateBlock(comm,bs,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   
  ierr = ISCreateStride(PETSC_COMM_SELF,bs*nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localupdate);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);

  /* set local to global mapping for ghosted vector */
  nb = n/bs;
  ierr = PetscMalloc((nb+nghost)*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*vv,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<nb; i++) {
    indices[i] = rstart + i*bs;
  }
  for (i=0; i<nghost; i++) {
    indices[nb+i] = ghosts[i];
  }
  ierr = ISLocalToGlobalMappingCreate(comm,nb+nghost,indices,PETSC_OWN_POINTER,&ltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*vv,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateGhostBlock"
/*@
   VecCreateGhostBlock - Creates a parallel vector with ghost padding on each processor.
        The indicing of the ghost points is done with blocks.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator to use
.  bs - the block size
.  n - local vector length 
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost blocks
-  ghosts - global indices of ghost blocks, counts are by block, not by individual index

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)
 
   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation 
   of the vector.

   n is the local vector size (total local size not the number of blocks) while nghost
   is the number of blocks in the ghost portion, i.e. the number of elements in the ghost
   portion is bs*nghost

   Level: advanced

   Concepts: vectors^ghosted

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecCreateGhostBlockWithArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhostBlock(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostBlockWithArray(comm,bs,n,N,nghost,ghosts,0,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
