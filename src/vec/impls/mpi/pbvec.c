#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pbvec.c,v 1.127 1999/03/18 02:00:17 balay Exp bsmith $";
#endif

/*
   This file contains routines for Parallel vector operations.
 */

#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "vec.h"   I*/
extern int VecReciprocal_General(Vec);

/*
       Note this code is very similar to VecPublish_Seq()
*/
#undef __FUNC__  
#define __FUNC__ "VecPublish_MPI"
static int VecPublish_MPI(PetscObject object)
{
#if defined(HAVE_AMS)
  Vec          v = (Vec) object;
  Vec_MPI      *s = (Vec_MPI *) v->data;
  int          ierr;
  int          (*f)(AMS_Memory,char *,Vec);
  
  PetscFunctionBegin;

  /* if it is already published then return */
  if (v->amem >=0 ) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(object);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"values",s->array,v->n,AMS_DOUBLE,AMS_READ,
                                AMS_DISTRIBUTED,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  /*
     If the vector knows its "layout" let it set it, otherwise it defaults
     to correct 1d distribution
  */
  ierr = PetscObjectQueryFunction((PetscObject)v,"AMSSetFieldBlock_C",(void**)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)((AMS_Memory)v->amem,"values",v);CHKERRQ(ierr);
  }
  ierr = PetscObjectPublishBaseEnd(object);CHKERRQ(ierr);

#else
  PetscFunctionBegin;
#endif

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDot_MPI"
int VecDot_MPI( Vec xin, Vec yin, Scalar *z )
{
  Scalar    sum, work;
  int       ierr;

  PetscFunctionBegin;
  ierr = VecDot_Seq(  xin, yin, &work ); CHKERRQ(ierr);
/*
   This is a ugly hack. But to do it right is kind of silly.
*/
  PLogEventBarrierBegin(VEC_DotBarrier,0,0,0,0,xin->comm);
#if defined(USE_PETSC_COMPLEX)
  ierr = MPI_Allreduce(&work,&sum,2,MPI_DOUBLE,MPI_SUM,xin->comm);CHKERRQ(ierr);
#else
  ierr = MPI_Allreduce(&work,&sum,1,MPI_DOUBLE,MPI_SUM,xin->comm);CHKERRQ(ierr);
#endif
  PLogEventBarrierEnd(VEC_DotBarrier,0,0,0,0,xin->comm);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecTDot_MPI"
int VecTDot_MPI( Vec xin, Vec yin, Scalar *z )
{
  Scalar    sum, work;
  int       ierr;

  PetscFunctionBegin;
  VecTDot_Seq(  xin, yin, &work );
/*
   This is a ugly hack. But to do it right is kind of silly.
*/
  PLogEventBarrierBegin(VEC_DotBarrier,0,0,0,0,xin->comm);
#if defined(USE_PETSC_COMPLEX)
  ierr = MPI_Allreduce(&work, &sum,2,MPI_DOUBLE,MPI_SUM,xin->comm );CHKERRQ(ierr);
#else
  ierr = MPI_Allreduce(&work, &sum,1,MPI_DOUBLE,MPI_SUM,xin->comm );CHKERRQ(ierr);
#endif
  PLogEventBarrierEnd(VEC_DotBarrier,0,0,0,0,xin->comm);
  *z = sum;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetOption_MPI"
int VecSetOption_MPI(Vec v,VecOption op)
{
  Vec_MPI *w = (Vec_MPI *) v->data;

  PetscFunctionBegin;
  if (op == VEC_IGNORE_OFF_PROC_ENTRIES) {
    w->donotstash = 1;
  }
  PetscFunctionReturn(0);
}
    
extern int VecDuplicate_MPI(Vec,Vec *);
EXTERN_C_BEGIN
extern int VecView_MPI_Draw(Vec, Viewer);
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
            VecGetOwnershipRange_MPI,
            VecRestoreArray_Seq,
            VecMax_MPI,VecMin_MPI,
            VecSetRandom_Seq,
            VecSetOption_MPI,
            VecSetValuesBlocked_MPI,
            VecDestroy_MPI,
            VecView_MPI,
            VecPlaceArray_Seq,
            VecReplaceArray_Seq,
            VecGetMap_Seq,
            VecDot_Seq,
            VecTDot_Seq,
            VecNorm_Seq,
            VecLoadIntoVector_Default,
            VecReciprocal_General};

#undef __FUNC__  
#define __FUNC__ "VecCreate_MPI_Private"
/*
    VecCreate_MPI_Private - Basic create routine called by VecCreate_MPI() (i.e. VecCreateMPI()), 
    VecCreateMPIWithArray(), VecCreate_Shared() (i.e. VecCreateShared()), VecCreateGhost(),
    VecDuplicate_MPI(), VecCreateGhostWithArray(), VecDuplicate_MPI(), and VecDuplicate_Shared()
*/
int VecCreate_MPI_Private(Vec v,int nghost,const Scalar array[],Map map)
{
  Vec_MPI *s;
  int     ierr,size,rank;

  PetscFunctionBegin;
  MPI_Comm_size(v->comm,&size);
  MPI_Comm_rank(v->comm,&rank); 

  v->bops->publish   = VecPublish_MPI;
  PLogObjectMemory(v, sizeof(Vec_MPI) + (v->n+nghost+1)*sizeof(Scalar));
  s              = (Vec_MPI *) PetscMalloc(sizeof(Vec_MPI)); CHKPTRQ(s);
  PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));
  v->data        = (void *) s;
  s->n           = v->n;
  s->nghost      = nghost;
  s->N           = v->N;
  v->mapping     = 0;
  v->bmapping    = 0;
  v->bs          = 1;
  s->size        = size;
  s->rank        = rank;
  s->browners    = 0;
  v->type_name   = (char *) PetscMalloc((1+PetscStrlen(VEC_MPI))*sizeof(char));CHKPTRQ(v->type_name);
  PetscStrcpy(v->type_name,VEC_MPI);
  if (array) {
    s->array           = (Scalar *)array;
    s->array_allocated = 0;
  } else {
    s->array           = (Scalar *) PetscMalloc((v->n+nghost+1)*sizeof(Scalar));CHKPTRQ(s->array);
    s->array_allocated = s->array;
    PetscMemzero(s->array,v->n*sizeof(Scalar));
  }

  /* By default parallel vectors do not have local representation */
  s->localrep    = 0;
  s->localupdate = 0;

  s->insertmode  = NOT_SET_VALUES;

  /* create the stashes. The block-size for bstash is set later when 
     VecSetValuesBlocked is called.
  */
  ierr = VecStashCreate_Private(v->comm,1,&v->stash); CHKERRQ(ierr);
  ierr = VecStashCreate_Private(v->comm,1,&v->bstash); CHKERRQ(ierr); 
                                                        
  if (!v->map) {
    if (!map) {
      ierr = MapCreateMPI(v->comm,v->n,v->N,&v->map); CHKERRQ(ierr);
    } else {
      v->map = map;
      ierr = PetscObjectReference((PetscObject)map);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectComposeFunction((PetscObject)v,"VecView_MPI_Draw_C","VecView_MPI_Draw",
                                     (void *)VecView_MPI_Draw);CHKERRQ(ierr);
  PetscPublishAll(v);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "VecCreate_MPI"
int VecCreate_MPI(Vec vv)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscSplitOwnership(vv->comm,&vv->n,&vv->N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(vv,0,0,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "VecCreateMPIWithArray"
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

.keywords: vector, create, MPI

.seealso: VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/ 
int VecCreateMPIWithArray(MPI_Comm comm,int n,int N,const Scalar array[],Vec *vv)
{
  int ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local size of vector");
  }
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,n,N,vv);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,0,array,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGhostGetLocalForm"
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

.keywords:  ghost points, local representation

.seealso: VecCreateGhost(), VecGhostRestoreLocalForm(), VecCreateGhostWithArray()

@*/
int VecGhostGetLocalForm(Vec g,Vec *l)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);

  if (PetscTypeCompare(g->type_name,VEC_MPI)) {
    Vec_MPI *v  = (Vec_MPI *) g->data;
    if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG ,1,"Vector is not ghosted");
    *l = v->localrep;
  } else if (PetscTypeCompare(g->type_name,VEC_SEQ)) {
    *l = g;
  } else {
    SETERRQ(1,1,"Vector type does not have local representation");
  }
  PetscObjectReference((PetscObject)*l);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGhostRestoreLocalForm"
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

.keywords:  ghost points, local representation

.seealso: VecCreateGhost(), VecGhostGetLocalForm(), VecCreateGhostWithArray()
@*/
int VecGhostRestoreLocalForm(Vec g,Vec *l)
{
  PetscFunctionBegin;
  PetscObjectDereference((PetscObject)*l);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGhostUpdateBegin"
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
int VecGhostUpdateBegin(Vec g, InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI *v;
  int     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);

  v  = (Vec_MPI *) g->data;
  if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG ,1,"Vector is not ghosted");
 
  if (scattermode == SCATTER_REVERSE) {
    ierr = VecScatterBegin(v->localrep,g,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(g,v->localrep,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGhostUpdateEnd"
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
int VecGhostUpdateEnd(Vec g, InsertMode insertmode,ScatterMode scattermode)
{
  Vec_MPI *v;
  int     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);

  v  = (Vec_MPI *) g->data;
  if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG ,1,"Vector is not ghosted");
 
  if (scattermode == SCATTER_REVERSE) {
    ierr = VecScatterEnd(v->localrep,g,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
  } else {
    ierr = VecScatterEnd(g,v->localrep,insertmode,scattermode,v->localupdate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

.keywords: vector, create, MPI, ghost points, ghost padding

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), 
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray()

@*/ 
int VecCreateGhostWithArray(MPI_Comm comm,int n,int N,int nghost,const int ghosts[],
                            const Scalar array[],Vec *vv)
{
  int     ierr;
  Vec_MPI *w;
  Scalar  *larray;

  PetscFunctionBegin;
  *vv = 0;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Ghost length must be >= 0");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,n,N,vv);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,nghost,array,PETSC_NULL); CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray); CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+nghost,larray,&w->localrep); CHKERRQ(ierr);
  PLogObjectParent(*vv,w->localrep);
  ierr = VecRestoreArray(*vv,&larray); CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values 
  */
  if (ghosts) {
    IS from, to;
  
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nghost,ghosts,&from);CHKERRQ(ierr);   
    ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to); CHKERRQ(ierr);
    ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
    PLogObjectParent(*vv,w->localupdate);
    ierr = ISDestroy(to); CHKERRQ(ierr);
    ierr = ISDestroy(from); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

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

.keywords: vector, create, MPI, ghost points, ghost padding

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray()

@*/ 
int VecCreateGhost(MPI_Comm comm,int n,int N,int nghost,const int ghosts[],Vec *vv)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostWithArray(comm,n,N,nghost,ghosts,0,vv); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDuplicate_MPI"
int VecDuplicate_MPI( Vec win, Vec *v)
{
  int     ierr;
  Vec_MPI *vw, *w = (Vec_MPI *)win->data;
  Scalar  *array;
#if defined(HAVE_AMS)
  int     (*f)(AMS_Memory,char *,Vec);
#endif

  PetscFunctionBegin;
  ierr = VecCreate(win->comm,w->n,w->N,v);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*v,w->nghost,0,win->map);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,w->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array); CHKERRQ(ierr);
    PLogObjectParent(*v,vw->localrep);
    vw->localupdate = w->localupdate;
    PetscObjectReference((PetscObject)vw->localupdate);
  }    

  /* New vector should inherit stashing property of parent */
  vw->donotstash = w->donotstash;
  
  ierr = OListDuplicate(win->olist,&(*v)->olist);CHKERRQ(ierr);
  ierr = FListDuplicate(win->qlist,&(*v)->qlist);CHKERRQ(ierr);
  if (win->mapping) {
    (*v)->mapping = win->mapping;
    PetscObjectReference((PetscObject)win->mapping);
  }
  if (win->bmapping) {
    (*v)->bmapping = win->bmapping;
    PetscObjectReference((PetscObject)win->bmapping);
  }
  (*v)->bs = win->bs;

#if defined(HAVE_AMS)
  /*
     If the vector knows its "layout" let it set it, otherwise it defaults
     to correct 1d distribution
  */
  ierr = PetscObjectQueryFunction((PetscObject)(*v),"AMSSetFieldBlock_C",(void**)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)((AMS_Memory)(*v)->amem,"values",*v);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

