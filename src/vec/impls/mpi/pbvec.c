/*$Id: pbvec.c,v 1.173 2001/09/12 03:26:59 bsmith Exp $*/

/*
   This file contains routines for Parallel vector operations.
 */
#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/

/*
       Note this code is very similar to VecPublish_Seq()
*/
#undef __FUNCT__  
#define __FUNCT__ "VecPublish_MPI"
static int VecPublish_MPI(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  Vec          v = (Vec) obj;
  Vec_MPI      *s = (Vec_MPI*)v->data;
  int          ierr,(*f)(AMS_Memory,char *,Vec);
#endif  

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"values",s->array,v->n,AMS_DOUBLE,AMS_READ,
                                AMS_DISTRIBUTED,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  /*
     If the vector knows its "layout" let it set it, otherwise it defaults
     to correct 1d distribution
  */
  ierr = PetscObjectQueryFunction(obj,"AMSSetFieldBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)((AMS_Memory)v->amem,"values",v);CHKERRQ(ierr);
  }
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDot_MPI"
int VecDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar  sum,work;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecDot_Seq(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,PetscSum_Op,xin->comm);CHKERRQ(ierr);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecTDot_MPI"
int VecTDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar  sum,work;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecTDot_Seq(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,PetscSum_Op,xin->comm);CHKERRQ(ierr);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetOption_MPI"
int VecSetOption_MPI(Vec v,VecOption op)
{
  PetscFunctionBegin;
  if (op == VEC_IGNORE_OFF_PROC_ENTRIES) {
    v->stash.donotstash = PETSC_TRUE;
  } else if (op == VEC_TREAT_OFF_PROC_ENTRIES) {
    v->stash.donotstash = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
    
EXTERN int VecDuplicate_MPI(Vec,Vec *);
EXTERN_C_BEGIN
EXTERN int VecView_MPI_Draw(Vec,PetscViewer);
EXTERN_C_END

static struct _VecOps DvOps = { VecDuplicate_MPI,
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_MPI,
            VecMDot_MPI,
            VecNorm_MPI,
            VecTDot_MPI,
            VecMTDot_MPI,
            VecScale_Seq,
            VecCopy_Seq,
            VecSet_Seq,
            VecSwap_Seq,
            VecAXPY_Seq,
            VecAXPBY_Seq,
            VecMAXPY_Seq,
            VecAYPX_Seq,
            VecWAXPY_Seq,
            VecPointwiseMult_Seq,
            VecPointwiseDivide_Seq,
            VecSetValues_MPI,
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
            VecPlaceArray_Seq,
            VecReplaceArray_Seq,
            VecDot_Seq,
            VecTDot_Seq,
            VecNorm_Seq,
            VecLoadIntoVector_Default,
            VecReciprocal_Default,
            0, /* VecViewNative... */
            VecConjugate_Seq,
            0,
            0,
            VecResetArray_Seq,
            0,
            VecMaxPointwiseDivide_Seq};

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPI_Private"
/*
    VecCreate_MPI_Private - Basic create routine called by VecCreate_MPI() (i.e. VecCreateMPI()),
    VecCreateMPIWithArray(), VecCreate_Shared() (i.e. VecCreateShared()), VecCreateGhost(),
    VecDuplicate_MPI(), VecCreateGhostWithArray(), VecDuplicate_MPI(), and VecDuplicate_Shared()
*/
int VecCreate_MPI_Private(Vec v,int nghost,const PetscScalar array[],PetscMap map)
{
  Vec_MPI *s;
  int     ierr,size,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(v->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);

  v->bops->publish   = VecPublish_MPI;
  PetscLogObjectMemory(v,sizeof(Vec_MPI) + (v->n+nghost+1)*sizeof(PetscScalar));
  ierr           = PetscNew(Vec_MPI,&s);CHKERRQ(ierr);
  ierr           = PetscMemzero(s,sizeof(Vec_MPI));CHKERRQ(ierr);
  ierr           = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  v->data        = (void*)s;
  s->nghost      = nghost;
  v->mapping     = 0;
  v->bmapping    = 0;
  v->petscnative = PETSC_TRUE;

  if (array) {
    s->array           = (PetscScalar *)array;
    s->array_allocated = 0;
  } else {
    int n              = ((v->n+nghost) > 0) ? v->n+nghost : 1;
    ierr               = PetscMalloc(n*sizeof(PetscScalar),&s->array);CHKERRQ(ierr);
    s->array_allocated = s->array;
    ierr               = PetscMemzero(s->array,v->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* By default parallel vectors do not have local representation */
  s->localrep    = 0;
  s->localupdate = 0;

  v->stash.insertmode  = NOT_SET_VALUES;

  if (!v->map) {
    if (!map) {
      ierr = PetscMapCreateMPI(v->comm,v->n,v->N,&v->map);CHKERRQ(ierr);
    } else {
      v->map = map;
      ierr = PetscObjectReference((PetscObject)map);CHKERRQ(ierr);
    }
  }
  /* create the stashes. The block-size for bstash is set later when 
     VecSetValuesBlocked is called.
  */
  ierr = VecStashCreate_Private(v->comm,1,&v->stash);CHKERRQ(ierr);
  ierr = VecStashCreate_Private(v->comm,v->bs,&v->bstash);CHKERRQ(ierr); 
                                                        
#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEnginePut_C","VecMatlabEnginePut_Default",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscMatlabEngineGet_C","VecMatlabEngineGet_Default",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPI);CHKERRQ(ierr);
  ierr = PetscPublishAll(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPI"
int VecCreate_MPI(Vec vv)
{
  int ierr;

  PetscFunctionBegin;
  if (vv->bs > 0) {
    ierr = PetscSplitOwnershipBlock(vv->comm,vv->bs,&vv->n,&vv->N);CHKERRQ(ierr);
  } else {
    ierr = PetscSplitOwnership(vv->comm,&vv->n,&vv->N);CHKERRQ(ierr);
  }
  ierr = VecCreate_MPI_Private(vv,0,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecSetSerializeType(vv,VEC_SER_MPI_BINARY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSerialize_MPI"
int VecSerialize_MPI(MPI_Comm comm, Vec *vec, PetscViewer viewer, PetscTruth store)
{
  Vec          v;
  Vec_MPI     *x;
  PetscScalar *array;
  int          fd;
  int          vars, locVars, ghostVars;
  int          size;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetDescriptor(viewer, &fd);                                                     CHKERRQ(ierr);
  if (store) {
    v    = *vec;
    x    = (Vec_MPI *) v->data;
    ierr = PetscBinaryWrite(fd, &v->N,      1,                PETSC_INT,     0);                          CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &v->n,      1,                PETSC_INT,     0);                          CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &x->nghost, 1,                PETSC_INT,     0);                          CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  x->array,  v->n + x->nghost, PETSC_SCALAR,  0);                          CHKERRQ(ierr);
  } else {
    ierr = PetscBinaryRead(fd, &vars,      1,                   PETSC_INT);                               CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd, &locVars,   1,                   PETSC_INT);                               CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd, &ghostVars, 1,                   PETSC_INT);                               CHKERRQ(ierr);
    ierr = MPI_Allreduce(&locVars, &size, 1, MPI_INT, MPI_SUM, comm);                                     CHKERRQ(ierr);
    if (size != vars) SETERRQ(PETSC_ERR_ARG_CORRUPT, "Invalid row partition");
    ierr = VecCreate(comm, &v);                                                                           CHKERRQ(ierr);
    ierr = VecSetSizes(v, locVars, vars);                                                                 CHKERRQ(ierr);
    if (locVars + ghostVars > 0) {
      ierr = PetscMalloc((locVars + ghostVars) * sizeof(PetscScalar), &array);                            CHKERRQ(ierr);
      ierr = PetscBinaryRead(fd,  array,     locVars + ghostVars, PETSC_SCALAR);                          CHKERRQ(ierr);
      ierr = VecCreate_MPI_Private(v, ghostVars, array, PETSC_NULL);                                      CHKERRQ(ierr);
      ((Vec_MPI *) v->data)->array_allocated = array;
    } else {
      ierr = VecCreate_MPI_Private(v, ghostVars, PETSC_NULL, PETSC_NULL);                                 CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(v);                                                                           CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);                                                                             CHKERRQ(ierr);
    *vec = v;
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
int VecCreateMPIWithArray(MPI_Comm comm,int n,int N,const PetscScalar array[],Vec *vv)
{
  int ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  }
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,0,array,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostGetLocalForm"
/*@C
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
int VecGhostGetLocalForm(Vec g,Vec *l)
{
  int        ierr;
  PetscTruth isseq,ismpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  PetscValidPointer(l);

  ierr = PetscTypeCompare((PetscObject)g,VECSEQ,&isseq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)g,VECMPI,&ismpi);CHKERRQ(ierr);
  if (ismpi) {
    Vec_MPI *v  = (Vec_MPI*)g->data;
    if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
    *l = v->localrep;
  } else if (isseq) {
    *l = g;
  } else {
    SETERRQ1(1,"Vector type %s does not have local representation",g->type_name);
  }
  ierr = PetscObjectReference((PetscObject)*l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGhostRestoreLocalForm"
/*@C
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
int VecGhostRestoreLocalForm(Vec g,Vec *l)
{
  PetscFunctionBegin;
  PetscObjectDereference((PetscObject)*l);
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
int VecGhostUpdateBegin(Vec g,InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI *v;
  int     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);

  v  = (Vec_MPI*)g->data;
  if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
  if (!v->localupdate) PetscFunctionReturn(0);
 
  if (scattermode == SCATTER_REVERSE) {
    ierr = VecScatterBegin(v->localrep,g,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(g,v->localrep,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
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
int VecGhostUpdateEnd(Vec g,InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI *v;
  int     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);

  v  = (Vec_MPI*)g->data;
  if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector is not ghosted");
  if (!v->localupdate) PetscFunctionReturn(0);

  if (scattermode == SCATTER_REVERSE) {
    ierr = VecScatterEnd(v->localrep,g,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
  } else {
    ierr = VecScatterEnd(g,v->localrep,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
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

   Level: advanced

   Concepts: vectors^creating with array
   Concepts: vectors^ghosted

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), 
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/ 
int VecCreateGhostWithArray(MPI_Comm comm,int n,int N,int nghost,const int ghosts[],const PetscScalar array[],Vec *vv)
{
  int          ierr;
  Vec_MPI      *w;
  PetscScalar  *larray;
  IS           from,to;

  PetscFunctionBegin;
  *vv = 0;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,nghost,array,PETSC_NULL);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
  PetscLogObjectParent(*vv,w->localrep);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateGeneral(comm,nghost,ghosts,&from);CHKERRQ(ierr);   
  ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  PetscLogObjectParent(*vv,w->localupdate);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateGhost"
/*@C
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

   Level: advanced

   Concepts: vectors^ghosted

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/ 
int VecCreateGhost(MPI_Comm comm,int n,int N,int nghost,const int ghosts[],Vec *vv)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostWithArray(comm,n,N,nghost,ghosts,0,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPI"
int VecDuplicate_MPI(Vec win,Vec *v)
{
  int          ierr;
  Vec_MPI      *vw,*w = (Vec_MPI *)win->data;
  PetscScalar  *array;
#if defined(PETSC_HAVE_AMS)
  int          (*f)(AMS_Memory,char *,Vec);
#endif

  PetscFunctionBegin;
  ierr = VecCreate(win->comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,win->n,win->N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*v,w->nghost,0,win->map);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,win->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
    PetscLogObjectParent(*v,vw->localrep);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }    

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash = win->stash.donotstash;
  
  ierr = PetscOListDuplicate(win->olist,&(*v)->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(win->qlist,&(*v)->qlist);CHKERRQ(ierr);
  if (win->mapping) {
    (*v)->mapping = win->mapping;
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
  }
  if (win->bmapping) {
    (*v)->bmapping = win->bmapping;
    ierr = PetscObjectReference((PetscObject)win->bmapping);CHKERRQ(ierr);
  }
  (*v)->bs        = win->bs;
  (*v)->bstash.bs = win->bstash.bs;

#if defined(PETSC_HAVE_AMS)
  /*
     If the vector knows its "layout" let it set it, otherwise it defaults
     to correct 1d distribution
  */
  ierr = PetscObjectQueryFunction((PetscObject)(*v),"AMSSetFieldBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)((AMS_Memory)(*v)->amem,"values",*v);CHKERRQ(ierr);
  }
#endif
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
int VecCreateGhostBlockWithArray(MPI_Comm comm,int bs,int n,int N,int nghost,const int ghosts[],const PetscScalar array[],Vec *vv)
{
  int          ierr;
  Vec_MPI      *w;
  PetscScalar  *larray;
  IS           from,to;

  PetscFunctionBegin;
  *vv = 0;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,nghost*bs,array,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+bs*nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = VecSetBlockSize(w->localrep,bs);CHKERRQ(ierr);
  PetscLogObjectParent(*vv,w->localrep);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  ierr = ISCreateBlock(comm,bs,nghost,ghosts,&from);CHKERRQ(ierr);   
  ierr = ISCreateStride(PETSC_COMM_SELF,bs*nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  PetscLogObjectParent(*vv,w->localupdate);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateGhostBlock"
/*@C
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
int VecCreateGhostBlock(MPI_Comm comm,int bs,int n,int N,int nghost,const int ghosts[],Vec *vv)
{
  int ierr;

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
int VecSetLocalToGlobalMapping_FETI(Vec vv,ISLocalToGlobalMapping map)
{
  int     ierr;
  Vec_MPI *v = (Vec_MPI *)vv->data;

  PetscFunctionBegin;
  v->nghost = map->n - vv->n;

  /* we need to make longer the array space that was allocated when the vector was created */
  ierr     = PetscFree(v->array_allocated);CHKERRQ(ierr);
  ierr     = PetscMalloc(map->n*sizeof(PetscScalar),&v->array_allocated);CHKERRQ(ierr);
  v->array = v->array_allocated;
  
  /* Create local representation */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,map->n,v->array,&v->localrep);CHKERRQ(ierr);
  PetscLogObjectParent(vv,v->localrep);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecSetValuesLocal_FETI"
int VecSetValuesLocal_FETI(Vec vv,int n,const int *ix,const PetscScalar *values,InsertMode mode)
{
  int      ierr;
  Vec_MPI *v = (Vec_MPI *)vv->data;

  PetscFunctionBegin;
  ierr = VecSetValues(v->localrep,n,ix,values,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_FETI"
int VecCreate_FETI(Vec vv)
{
  int ierr;

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









