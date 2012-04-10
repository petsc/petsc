
/*
   This file contains routines for Parallel vector operations.
 */
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/

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
extern PetscErrorCode VecView_MPI_Draw(Vec,PetscViewer);
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

extern PetscErrorCode VecGetValues_MPI(Vec,PetscInt,const PetscInt [],PetscScalar []);

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPI"
static PetscErrorCode VecDuplicate_MPI(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI *)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)win)->comm,v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPI_Private(*v,PETSC_TRUE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
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
  (*v)->map->bs    = win->map->bs;
  (*v)->bstash.bs = win->bstash.bs;

  PetscFunctionReturn(0);
}

extern PetscErrorCode VecSetOption_MPI(Vec,VecOption,PetscBool);
extern PetscErrorCode VecResetArray_MPI(Vec);

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
            0,
            VecGetSize_MPI,
            VecGetSize_Seq,
            0,
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
  v->petscnative = PETSC_TRUE;

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
PetscErrorCode  VecCreate_MPI(Vec vv)
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
PetscErrorCode  VecCreate_Standard(Vec v)
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
PetscErrorCode  VecCreateMPIWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
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
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecMPISetGhost()

@*/ 
PetscErrorCode  VecCreateGhostWithArray(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
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
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   
  ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localupdate);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

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
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
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
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecMPISetGhost()

@*/ 
PetscErrorCode  VecCreateGhost(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostWithArray(comm,n,N,nghost,ghosts,0,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMPISetGhost"
/*@
   VecMPISetGhost - Sets the ghost points for an MPI ghost vector

   Collective on Vec

   Input Parameters:
+  vv - the MPI vector
.  nghost - number of local ghost points
-  ghosts - global indices of ghost points

 
   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation 
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   You must call this AFTER you have set the type of the vector (with VecSetType()) and the size (with VecSetSizes()).

   Level: advanced

   Concepts: vectors^ghosted

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/ 
PetscErrorCode  VecMPISetGhost(Vec vv,PetscInt nghost,const PetscInt ghosts[])
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)vv,VECMPI,&flg);CHKERRQ(ierr);
  /* if already fully existant VECMPI then basically destroy it and rebuild with ghosting */
  if (flg) {
    PetscInt               n,N;
    Vec_MPI                *w;
    PetscScalar            *larray;
    IS                     from,to;
    ISLocalToGlobalMapping ltog;
    PetscInt               rstart,i,*indices;
    MPI_Comm               comm = ((PetscObject)vv)->comm;

    n = vv->map->n;
    N = vv->map->N;
    ierr = (*vv->ops->destroy)(vv);CHKERRQ(ierr);
    ierr = VecSetSizes(vv,n,N);CHKERRQ(ierr);
    ierr = VecCreate_MPI_Private(vv,PETSC_TRUE,nghost,PETSC_NULL);CHKERRQ(ierr);
    w    = (Vec_MPI *)(vv)->data;
    /* Create local representation */
    ierr = VecGetArray(vv,&larray);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(vv,w->localrep);CHKERRQ(ierr);
    ierr = VecRestoreArray(vv,&larray);CHKERRQ(ierr);

    /*
     Create scatter context for scattering (updating) ghost values 
     */
    ierr = ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   
    ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to);CHKERRQ(ierr);
    ierr = VecScatterCreate(vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(vv,w->localupdate);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);

    /* set local to global mapping for ghosted vector */
    ierr = PetscMalloc((n+nghost)*sizeof(PetscInt),&indices);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vv,&rstart,PETSC_NULL);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      indices[i] = rstart + i;
    }
    for (i=0; i<nghost; i++) {
      indices[n+i] = ghosts[i];
    }
    ierr = ISLocalToGlobalMappingCreate(comm,n+nghost,indices,PETSC_OWN_POINTER,&ltog);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(vv,ltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr); 
  } else if (vv->ops->create == VecCreate_MPI) SETERRQ(((PetscObject)vv)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set local or global size before setting ghosting");
  else if (!((PetscObject)vv)->type_name) SETERRQ(((PetscObject)vv)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set type to VECMPI before ghosting");
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
          VecCreateGhostWithArray(), VecCreateGhostBlock()

@*/ 
PetscErrorCode  VecCreateGhostBlockWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  PetscErrorCode         ierr;
  Vec_MPI                *w;
  PetscScalar            *larray;
  IS                     from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt               rstart,i,nb,*indices;

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
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost*bs,array);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n+bs*nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateBlock(comm,bs,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   
  ierr = ISCreateStride(PETSC_COMM_SELF,bs*nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localupdate);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

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
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
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
PetscErrorCode  VecCreateGhostBlock(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostBlockWithArray(comm,bs,n,N,nghost,ghosts,0,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
