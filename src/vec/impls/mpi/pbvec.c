

#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pbvec.c,v 1.107 1998/08/16 22:06:10 curfman Exp bsmith $";
#endif

/*
   This file contains routines for Parallel vector operations.
 */

#include "petsc.h"
#include <math.h>
#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "vec.h"   I*/

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
  static int   counter = 0;
  int          ierr,rank;
  char         name[16];
  AMS_Memory   amem;
  AMS_Comm     acomm;
  int          (*f)(AMS_Memory,char *,Vec);
  
  PetscFunctionBegin;

  /* if it is already published then return */
  if (v->amem >=0 ) PetscFunctionReturn(0);

  ierr = ViewerAMSGetAMSComm(VIEWER_AMS_(v->comm),&acomm);CHKERRQ(ierr);
  if (v->name) {
    PetscStrcpy(name,v->name);
  } else {
    sprintf(name,"MPIVector_%d",counter++);
  }
  ierr = AMS_Memory_create(acomm,name,&amem);CHKERRQ(ierr);
  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr); 
  ierr = AMS_Memory_add_field(amem,"values",s->array,v->n,AMS_DOUBLE,AMS_READ,
                                AMS_DISTRIBUTED,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  /*
     If the vector knows its "layout" let it set it, otherwise it defaults
     to correct 1d distribution
  */
  ierr = PetscObjectQueryFunction((PetscObject)v,"AMSSetFieldBlock_C",(void**)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(amem,"values",v);CHKERRQ(ierr);
  }


  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);
  v->amem = (int) amem;

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
    w->stash.donotstash = 1;
  }
  PetscFunctionReturn(0);
}
    
int VecDuplicate_MPI(Vec,Vec *);

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
            VecGetOwnershipRange_MPI,0,
            VecMax_MPI,VecMin_MPI,
            VecSetRandom_Seq,
            VecSetOption_MPI,
            VecSetValuesBlocked_MPI,
            VecDestroy_MPI,
            VecView_MPI,
            VecPlaceArray_Seq,
            VecGetMap_Seq};

#undef __FUNC__  
#define __FUNC__ "VecCreateMPI_Private"
/*
    VecCreateMPI_Private - Basic create routine called by VecCreateMPI(), VecCreateGhost()
  and VecDuplicate_MPI() to reduce code duplication.
*/
int VecCreateMPI_Private(MPI_Comm comm,int n,int N,int nghost,int size,int rank,Scalar *array,Map map,Vec *vv)
{
  Vec     v;
  Vec_MPI *s;
  int     ierr;

  PetscFunctionBegin;
  *vv = 0;

  PetscHeaderCreate(v,_p_Vec,struct _VecOps,VEC_COOKIE,VECMPI,comm,VecDestroy,VecView);
  v->bops->publish   = VecPublish_MPI;
  PLogObjectCreate(v);
  PLogObjectMemory(v, sizeof(Vec_MPI) + sizeof(struct _p_Vec) + (n+nghost+1)*sizeof(Scalar));
  s              = (Vec_MPI *) PetscMalloc(sizeof(Vec_MPI)); CHKPTRQ(s);
  PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));
  v->data        = (void *) s;
  s->n           = n;
  s->nghost      = nghost;
  s->N           = N;
  v->n           = n;
  v->N           = N;
  v->mapping     = 0;
  v->bmapping    = 0;
  v->bs          = 1;
  s->size        = size;
  s->rank        = rank;
  v->type_name   = (char *) PetscMalloc(3*sizeof(char));CHKPTRQ(v->type_name);
  PetscStrcpy(v->type_name,"MPI");
  if (array) {
    s->array           = array;
    s->array_allocated = 0;
  } else {
    s->array           = (Scalar *) PetscMalloc((n+nghost+1)*sizeof(Scalar));CHKPTRQ(s->array);
    s->array_allocated = s->array;
    PetscMemzero(s->array,n*sizeof(Scalar));
  }

  /* By default parallel vectors do not have local representation */
  s->localrep    = 0;
  s->localupdate = 0;

  s->insertmode  = NOT_SET_VALUES;

  /* initialize the stash */
  s->stash.donotstash = 0;
  s->stash.nmax       = 10;
  s->stash.n          = 0;
  s->stash.array      = (Scalar *) PetscMalloc(10*(sizeof(Scalar)+sizeof(int)));CHKPTRQ(s->stash.array);
  s->stash.idx        = (int *) (s->stash.array + 10);
  PLogObjectMemory(v,10*sizeof(Scalar) + 10 *sizeof(int));

  if (!map) {
    ierr = MapCreateMPI(comm,n,N,&v->map); CHKERRQ(ierr);
  } else {
    v->map = map;
    ierr = PetscObjectReference((PetscObject)map);CHKERRQ(ierr);
  }

  *vv = v;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCreateMPI"
/*@C
   VecCreateMPI - Creates a parallel vector.

   Collective on MPI_Comm
 
   Input Parameters:
+  comm - the MPI communicator to use 
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
-  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, create, MPI

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPIWithArray(), VecCreateGhostWithArray()

@*/ 
int VecCreateMPI(MPI_Comm comm,int n,int N,Vec *vv)
{
  int sum, work = n, size, rank,ierr;

  PetscFunctionBegin;
  *vv = 0;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank); 
  if (N == PETSC_DECIDE) { 
    ierr = MPI_Allreduce( &work, &sum,1,MPI_INT,MPI_SUM,comm );CHKERRQ(ierr);
    N = sum;
  }
  if (n == PETSC_DECIDE) { 
    n = N/size + ((N % size) > rank);
  }
  ierr = VecCreateMPI_Private(comm,n,N,0,size,rank,0,PETSC_NULL,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCreateMPIWithArray"
/*@C
   VecCreateMPIWithArray - Creates a parallel, array-style vector,
   where the user provides the array space to store the vector values.

   Collective on MPI_Comm

   Input Parameters:
+  comm  - the MPI communicator to use
.  n     - local vector length (or PETSC_DECIDE to have calculated if N is given)
.  N     - global vector length (or PETSC_DECIDE to have calculated if n is given)
-  array - the user provided array to store the vector values

   Output Parameter:
.  vv - the vector
 
   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is PETSC_NULL, then VecPlaceArray() can be used
   at a later atage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

.keywords: vector, create, MPI

.seealso: VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/ 
int VecCreateMPIWithArray(MPI_Comm comm,int n,int N,Scalar *array,Vec *vv)
{
  int sum, work = n, size, rank,ierr;

  PetscFunctionBegin;
  *vv = 0;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank); 
  if (N == PETSC_DECIDE) { 
    ierr = MPI_Allreduce( &work, &sum,1,MPI_INT,MPI_SUM,comm );CHKERRQ(ierr);
    N = sum;
  }
  if (n == PETSC_DECIDE) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local size of vector");
  }
  ierr =  VecCreateMPI_Private(comm,n,N,0,size,rank,array,PETSC_NULL,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGhostGetLocalForm"
/*@C
     VecGhostGetLocalForm - Obtain the local ghosted representation of 
         a parallel vector created with VecCreateGhost().

    Not Collective

    Input Parameter:
.    g - the global vector. Vector must be obtained with either VecCreateGhost(),
         VecCreateGhostWithArray() or VecCreateSeq().

    Output Parameter:
.    l - the local (ghosted) representation

     Notes:
       This routine does not actually update the ghost values, it returns a 
     sequential vector that includes the locations for the ghost values and their
     current values. The returned vector and the original vector passed in share
     the same array that contains the actual vector data.

       One should call VecGhostRestoreLocalForm() or VecDestroy() once one is
     finished using the object.

.keywords:  ghost points, local representation

.seealso: VecCreateGhost(), VecGhostRestoreLocalForm(), VecCreateGhostWithArray()

@*/
int VecGhostGetLocalForm(Vec g,Vec *l)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(g,VEC_COOKIE);

  if (g->type == VECMPI) {
    Vec_MPI *v  = (Vec_MPI *) g->data;
    if (!v->localrep) SETERRQ(PETSC_ERR_ARG_WRONG ,1,"Vector is not ghosted");
    *l = v->localrep;
  } else if (g->type == VECSEQ) {
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
     VecGhostRestoreLocalForm - Restore the local ghosted representation of 
         a parallel vector obtained with VecGhostGetLocalForm().

    Not Collective

    Input Parameter:
+   g - the global vector
-   l - the local (ghosted) representation

    Notes:
    This routine does not actually update the ghost values, it allow returns a 
    sequential vector that includes the locations for the ghost values and their
    current values.

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
   VecGhostUpdateBegin - Begin the vector scatter to update the vector from
   local representation to global or global representation to local.

   Collective on Vec

   Input Parameters:
+  g - the vector (obtained with VecCreateGhost() or VecDuplicate())
.  insertmode - one of ADD_VALUES or INSERT_VALUES
-  scattermode - one of SCATTER_FORWARD or SCATTER_REVERSE

   Notes:
   Use the following to update the ghost regions with correct values from the owning process
$       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
$       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
   Use the following to accumulate the ghost region values onto the owning processors
$       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
$       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
   Use the following to accumulate the values onto the owning processors 
   and then set the ghost values correctly call the later followed by the former.

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
$       VecGhostUpdateBegin(v,INSERT_VALUES,SCATTER_FORWARD);
$       VecGhostUpdateEnd(v,INSERT_VALUES,SCATTER_FORWARD);
   Use the following to accumulate the ghost region values onto the owning processors
$       VecGhostUpdateBegin(v,ADD_VALUES,SCATTER_REVERSE);
$       VecGhostUpdateEnd(v,ADD_VALUES,SCATTER_REVERSE);
   Use the following to accumulate the values onto the owning processors 
   and then set the ghost values correctly call the later followed by the former.

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

.keywords: vector, create, MPI, ghost points, ghost padding

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhost(), VecCreateMPIWithArray()

@*/ 
int VecCreateGhostWithArray(MPI_Comm comm,int n,int N,int nghost,int *ghosts,Scalar *array,Vec *vv)
{
  int     sum, work = n, size, rank, ierr;
  Vec_MPI *w;

  PetscFunctionBegin;
  *vv = 0;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Ghost length must be >= 0");

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank); 
  if (N == PETSC_DECIDE) { 
    ierr = MPI_Allreduce( &work, &sum,1,MPI_INT,MPI_SUM,comm );CHKERRQ(ierr);
    N = sum;
  }
  /* Create global representation */
  ierr = VecCreateMPI_Private(comm,n,N,nghost,size,rank,array,PETSC_NULL,vv); CHKERRQ(ierr);
  w    = (Vec_MPI *)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&array); CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,n+nghost,array,&w->localrep); CHKERRQ(ierr);
  PLogObjectParent(*vv,w->localrep);
  ierr = VecRestoreArray(*vv,&array); CHKERRQ(ierr);

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

.keywords: vector, create, MPI, ghost points, ghost padding

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray()

@*/ 
int VecCreateGhost(MPI_Comm comm,int n,int N,int nghost,int *ghosts,Vec *vv)
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

  PetscFunctionBegin;
  ierr = VecCreateMPI_Private(win->comm,w->n,w->N,w->nghost,w->size,w->rank,0,win->map,v);CHKERRQ(ierr);
  vw   = (Vec_MPI *)(*v)->data;

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,w->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    PLogObjectParent(*v,vw->localrep);
    ierr = VecRestoreArray(*v,&array); CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    PetscObjectReference((PetscObject)vw->localupdate);
  }    

  /* New vector should inherit stashing property of parent */
  vw->stash.donotstash = w->stash.donotstash;
  
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

  PetscFunctionReturn(0);
}




