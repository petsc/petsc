#define PETSCVEC_DLL
/*
   This file contains routines for Parallel vector operations.
 */
#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/

#if 0
#undef __FUNCT__  
#define __FUNCT__ "VecPublish_MPI"
static PetscErrorCode VecPublish_MPI(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

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

#undef __FUNCT__  
#define __FUNCT__ "VecSetOption_MPI"
PetscErrorCode VecSetOption_MPI(Vec v,VecOption op,PetscTruth flag)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_OFF_PROC_ENTRIES) {
    v->stash.donotstash = flag;
  } else if (op == VEC_IGNORE_NEGATIVE_INDICES) {
    v->stash.ignorenegidx = flag;
  }
  PetscFunctionReturn(0);
}
    
EXTERN PetscErrorCode VecDuplicate_MPI(Vec,Vec *);
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
  if (v->unplacedarray) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array = (PetscScalar *)a;
  if (v->localrep) {
    ierr = VecPlaceArray(v->localrep,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecResetArray_MPI"
PetscErrorCode VecResetArray_MPI(Vec vin)
{
  Vec_MPI        *v = (Vec_MPI *)vin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  if (v->localrep) {
    ierr = VecResetArray(v->localrep);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode VecLoad_Binary(PetscViewer, const VecType, Vec*);
EXTERN PetscErrorCode VecGetValues_MPI(Vec,PetscInt,const PetscInt [],PetscScalar []);

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
            VecLoadIntoVector_Default,
            0, /* VecLoadIntoVectorNative */
            VecReciprocal_Default,
            0, /* VecViewNative... */
            VecConjugate_Seq,
            0,
            0,
            VecResetArray_MPI,
            0,
            VecMaxPointwiseDivide_Seq,
            VecLoad_Binary,
            VecPointwiseMax_Seq,
            VecPointwiseMaxAbs_Seq,
            VecPointwiseMin_Seq,
            VecGetValues_MPI};

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPI_Private"
/*
    VecCreate_MPI_Private - Basic create routine called by VecCreate_MPI() (i.e. VecCreateMPI()),
    VecCreateMPIWithArray(), VecCreate_Shared() (i.e. VecCreateShared()), VecCreateGhost(),
    VecDuplicate_MPI(), VecCreateGhostWithArray(), VecDuplicate_MPI(), and VecDuplicate_Shared()

    If alloc is true and array is PETSC_NULL then this routine allocates the space, otherwise
    no space is allocated.
*/
PetscErrorCode VecCreate_MPI_Private(Vec v,PetscTruth alloc,PetscInt nghost,const PetscScalar array[])
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
  ierr = PetscPublishAll(v);CHKERRQ(ierr);
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
  if (n == PETSC_DECIDE) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  }
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostStateSync_Private"
/*
  This is used in VecGhostGetLocalForm and VecGhostRestoreLocalForm to ensure
  that the state is updated if either vector has changed since the last time
  one of these functions was called.  It could apply to any PetscObject, but
  VecGhost is quite different from other objects in that two separate vectors
  look at the same memory.

  In principle, we could only propagate state to the local vector on
  GetLocalForm and to the global vector on RestoreLocalForm, but this version is
  more conservative (i.e. robust against misuse) and simpler.

  Note that this function is correct and changes nothing if both arguments are the
  same, which is the case in serial.
*/
static PetscErrorCode VecGhostStateSync_Private(Vec g,Vec l)
{
  PetscErrorCode ierr;
  PetscInt       gstate,lstate;

  PetscFunctionBegin;
  ierr = PetscObjectStateQuery((PetscObject)g,&gstate);CHKERRQ(ierr);
  ierr = PetscObjectStateQuery((PetscObject)l,&lstate);CHKERRQ(ierr);
  ierr = PetscObjectSetState((PetscObject)g,PetscMax(gstate,lstate));CHKERRQ(ierr);
  ierr = PetscObjectSetState((PetscObject)l,PetscMax(gstate,lstate));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostGetLocalForm"
/*@
    VecGhostGetLocalForm - Obtains the local ghosted representation of 
    a parallel vector created with VecCreateGhost().

    Not Collective

    Input Parameter:
.   g - the global vector. Vector must be have been obtained with either
        VecCreateGhost(), VecCreateGhostWithArray() or VecCreateSeq().

    Output Parameter:
.   l - the local (ghosted) representation

    Notes:
    This routine does not actually update the ghost values, but rather it
    returns a sequential vector that includes the locations for the ghost
    values and their current values. The returned vector and the original
    vector passed in share the same array that contains the actual vector data.

    One should call VecGhostRestoreLocalForm() or VecDestroy() once one is
    finished using the object.

    Level: advanced

   Concepts: vectors^ghost point access

.seealso: VecCreateGhost(), VecGhostRestoreLocalForm(), VecCreateGhostWithArray()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGhostGetLocalForm(Vec g,Vec *l)
{
  PetscErrorCode ierr;
  PetscTruth     isseq,ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE,1);
  PetscValidPointer(l,2);

  ierr = PetscTypeCompare((PetscObject)g,VECSEQ,&isseq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)g,VECMPI,&ismpi);CHKERRQ(ierr);
  if (ismpi) {
    Vec_MPI *v  = (Vec_MPI*)g->data;
    if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
    *l = v->localrep;
  } else if (isseq) {
    *l = g;
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Vector type %s does not have local representation",((PetscObject)g)->type_name);
  }
  ierr = VecGhostStateSync_Private(g,*l);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostRestoreLocalForm"
/*@
    VecGhostRestoreLocalForm - Restores the local ghosted representation of 
    a parallel vector obtained with VecGhostGetLocalForm().

    Not Collective

    Input Parameter:
+   g - the global vector
-   l - the local (ghosted) representation

    Notes:
    This routine does not actually update the ghost values, but rather it
    returns a sequential vector that includes the locations for the ghost values
    and their current values.

    Level: advanced

.seealso: VecCreateGhost(), VecGhostGetLocalForm(), VecCreateGhostWithArray()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGhostRestoreLocalForm(Vec g,Vec *l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGhostStateSync_Private(g,*l);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostUpdateBegin"
/*@
   VecGhostUpdateBegin - Begins the vector scatter to update the vector from
   local representation to global or global representation to local.

   Collective on Vec

   Input Parameters:
+  g - the vector (obtained with VecCreateGhost() or VecDuplicate())
.  insertmode - one of ADD_VALUES or INSERT_VALUES
-  scattermode - one of SCATTER_FORWARD or SCATTER_REVERSE

   Notes:
   Use the following to update the ghost regions with correct values from the owning process
.vb
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Use the following to accumulate the ghost region values onto the owning processors
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
.ve

   To accumulate the ghost region values onto the owning processors and then update
   the ghost regions correctly, call the later followed by the former, i.e.,
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Level: advanced

.seealso: VecCreateGhost(), VecGhostUpdateEnd(), VecGhostGetLocalForm(),
          VecGhostRestoreLocalForm(),VecCreateGhostWithArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecGhostUpdateBegin(Vec g,InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI        *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE,1);

  v  = (Vec_MPI*)g->data;
  if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
  if (!v->localupdate) PetscFunctionReturn(0);
 
  if (scattermode == SCATTER_REVERSE) {
    ierr = VecScatterBegin(v->localupdate,v->localrep,g,insertmode,scattermode);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(v->localupdate,g,v->localrep,insertmode,scattermode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostUpdateEnd"
/*@
   VecGhostUpdateEnd - End the vector scatter to update the vector from
   local representation to global or global representation to local.

   Collective on Vec

   Input Parameters:
+  g - the vector (obtained with VecCreateGhost() or VecDuplicate())
.  insertmode - one of ADD_VALUES or INSERT_VALUES
-  scattermode - one of SCATTER_FORWARD or SCATTER_REVERSE

   Notes:

   Use the following to update the ghost regions with correct values from the owning process
.vb
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Use the following to accumulate the ghost region values onto the owning processors
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
.ve

   To accumulate the ghost region values onto the owning processors and then update
   the ghost regions correctly, call the later followed by the former, i.e.,
.vb
       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
.ve

   Level: advanced

.seealso: VecCreateGhost(), VecGhostUpdateBegin(), VecGhostGetLocalForm(),
          VecGhostRestoreLocalForm(),VecCreateGhostWithArray()

@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecGhostUpdateEnd(Vec g,InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI        *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE,1);

  v  = (Vec_MPI*)g->data;
  if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
  if (!v->localupdate) PetscFunctionReturn(0);

  if (scattermode == SCATTER_REVERSE) {
    ierr = VecScatterEnd(v->localupdate,v->localrep,g,insertmode,scattermode);CHKERRQ(ierr);
  } else {
    ierr = VecScatterEnd(v->localupdate,g,v->localrep,insertmode,scattermode);CHKERRQ(ierr);
  }
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

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost,array);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateGeneral(comm,nghost,ghosts,&from);CHKERRQ(ierr);   
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
  ierr = ISLocalToGlobalMappingCreate(comm,n+nghost,indices,&ltog);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*vv,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(ltog);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPI"
PetscErrorCode VecDuplicate_MPI(Vec win,Vec *v)
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
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
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
.  ghosts - global indices of ghost blocks (or PETSC_NULL if not needed)
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

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  if (n % bs)                 SETERRQ(PETSC_ERR_ARG_INCOMP,"Local size must be a multiple of block size");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost*bs,array);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+bs*nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = VecSetBlockSize(w->localrep,bs);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*vv,w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateBlock(comm,bs,nghost,ghosts,&from);CHKERRQ(ierr);   
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
  ierr = ISLocalToGlobalMappingCreate(comm,nb+nghost,indices,&ltog);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*vv,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(ltog);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

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
-  ghosts - global indices of ghost blocks

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

/*
    These introduce a ghosted vector where the ghosting is determined by the call to 
  VecSetLocalToGlobalMapping()
*/

#undef __FUNCT__  
#define __FUNCT__ "VecSetLocalToGlobalMapping_FETI"
PetscErrorCode VecSetLocalToGlobalMapping_FETI(Vec vv,ISLocalToGlobalMapping map)
{
  PetscErrorCode ierr;
  Vec_MPI        *v = (Vec_MPI *)vv->data;

  PetscFunctionBegin;
  v->nghost = map->n - vv->map->n;

  /* we need to make longer the array space that was allocated when the vector was created */
  ierr     = PetscFree(v->array_allocated);CHKERRQ(ierr);
  ierr     = PetscMalloc(map->n*sizeof(PetscScalar),&v->array_allocated);CHKERRQ(ierr);
  v->array = v->array_allocated;
  
  /* Create local representation */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,map->n,v->array,&v->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(vv,v->localrep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesLocal_FETI"
PetscErrorCode VecSetValuesLocal_FETI(Vec vv,PetscInt n,const PetscInt *ix,const PetscScalar *values,InsertMode mode)
{
  PetscErrorCode ierr;
  Vec_MPI        *v = (Vec_MPI *)vv->data;

  PetscFunctionBegin;
  ierr = VecSetValues(v->localrep,n,ix,values,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_FETI"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_FETI(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetType(vv,VECMPI);CHKERRQ(ierr);
  
  /* overwrite the functions to handle setting values locally */
  vv->ops->setlocaltoglobalmapping = VecSetLocalToGlobalMapping_FETI;
  vv->ops->setvalueslocal          = VecSetValuesLocal_FETI;
  vv->ops->assemblybegin           = 0;
  vv->ops->assemblyend             = 0;
  vv->ops->setvaluesblocked        = 0;
  vv->ops->setvaluesblocked        = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END









