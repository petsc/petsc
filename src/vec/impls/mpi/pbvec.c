
#ifndef lint
static char vcid[] = "$Id: pbvec.c,v 1.74 1997/03/13 16:32:42 curfman Exp balay $";
#endif

/*
   This file contains routines for Parallel vector operations.
 */

#include "petsc.h"
#include <math.h>
#include "pvecimpl.h"   /*I  "vec.h"   I*/

#undef __FUNC__  
#define __FUNC__ "VecDot_MPI"
void AZ_gdot_vec(MPI_Comm , int, double*, double*);
int VecDot_MPI( Vec xin, Vec yin, Scalar *z )
{
  Scalar    work;
  VecDot_Seq(  xin, yin, z );
  AZ_gdot_vec(xin->comm,1,z,&work);
  return 0;
}


void AZ_gdot_vec(MPI_Comm comm, int N, double *dots, double *dots2)
{
  /* local variables */
  
  int	partner;	  /* processor I exchange with */
  int	mask;		  /* bit pattern identifying partner */
  int	hbit;		  /* largest nonzero bit in nprocs */
  int	nprocs_small;	  /* largest power of 2 <= nprocs */
  int	i;		  /* loop counter */
  int	node, nprocs;
  int	tag = 1;
  MPI_Request request;	  /* Message handle */
  MPI_Status  status;
  /*
  node	 = PetscGlobalRank;
  nprocs = PetscGlobalSize;
  */
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&node);
  
  for (hbit = 0; (nprocs >> hbit) != 1; hbit++);
  
  nprocs_small = 1 << hbit;
  if (nprocs_small *2 == nprocs) {
    nprocs_small <<= 1;
    hbit++;
  }
  
  partner  = node ^ nprocs_small;
  if (node+nprocs_small < nprocs) {
    MPI_Irecv(dots2,N,MPIU_SCALAR,partner,tag,comm,&request);
  }
  else if (node & nprocs_small) {
    MPI_Send((void *) dots,N,MPIU_SCALAR,partner,1,comm);
  }
  if (node+nprocs_small < nprocs) {
    MPI_Wait(&request,&status);
    for (i = 0; i < N; i++) dots[i] += dots2[i];
  }
  /* Now do a binary exchange on nprocs_small nodes. */
    
  if (!(node & nprocs_small)) { 
    for (mask = nprocs_small>>1; mask; mask >>= 1) {
      partner = node ^ mask;
      MPI_Irecv((void *) dots2,N,MPIU_SCALAR,partner,1,comm,&request);
      MPI_Send((void *) dots,N,MPIU_SCALAR,partner,1,comm);
      MPI_Wait(&request,&status);
      for (i = 0; i < N; i++) dots[i] += dots2[i];
    }
  }
  
  /* Finally, send message from lower half to upper half. */
  
  partner = node ^ nprocs_small;
  if (node & nprocs_small) {
    MPI_Irecv((void *) dots,N,MPIU_SCALAR,partner,1,comm,&request);
  }
  else if (node+nprocs_small < nprocs ) {
    MPI_Send((void *) dots,N,MPIU_SCALAR,partner,1,comm);
  }
  if (node & nprocs_small) {
    MPI_Wait(&request,&status);
  }
}

#undef __FUNC__  
#define __FUNC__ "VecSetOption_MPI" /* ADIC Ignore */
int VecSetOption_MPI(Vec v,VecOption op)
{
  Vec_MPI *w = (Vec_MPI *) v->data;

  if (op == VEC_IGNORE_OFF_PROCESSOR_ENTRIES) {
    w->stash.donotstash = 1;
  }
  return 0;
}
    
int VecDuplicate_MPI(Vec,Vec *);

static struct _VeOps DvOps = { VecDuplicate_MPI, 
            VecDuplicateVecs_Default, VecDestroyVecs_Default, VecDot_MPI, 
            VecMDot_MPI,
            VecNorm_MPI, VecDot_MPI, 
            VecMDot_MPI,
            VecScale_Seq, VecCopy_Seq,
            VecSet_Seq, VecSwap_Seq, VecAXPY_Seq, VecAXPBY_Seq,
            VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPointwiseMult_Seq,
            VecPointwiseDivide_Seq, 
            VecSetValues_MPI,
            VecAssemblyBegin_MPI,VecAssemblyEnd_MPI,
            VecGetArray_Seq,VecGetSize_MPI,VecGetSize_Seq,
            VecGetOwnershipRange_MPI,0,VecMax_MPI,VecMin_MPI,
            VecSetRandom_Seq,
            VecSetOption_MPI};

#undef __FUNC__  
#define __FUNC__ "VecCreateMPI_Private"
/*
    VecCreateMPI_Private - Basic create routine called by VecCreateMPI(), VecCreateMPIGhost()
  and VecDuplicate_MPI() to reduce code duplication.
*/
static int VecCreateMPI_Private(MPI_Comm comm,int n,int nghost,int N,int size,int rank,int *owners,Vec *vv)
{
  Vec     v;
  Vec_MPI *s;
  int     mem,i;
  *vv = 0;

  mem           = sizeof(Vec_MPI)+(size+1)*sizeof(int);
  PetscHeaderCreate(v,_Vec,VEC_COOKIE,VECMPI,comm);
  PLogObjectCreate(v);
  PLogObjectMemory(v,mem + sizeof(struct _Vec) + (n+1)*sizeof(Scalar));
  s              = (Vec_MPI *) PetscMalloc(mem); CHKPTRQ(s);
  PetscMemcpy(&v->ops,&DvOps,sizeof(DvOps));
  v->data        = (void *) s;
  v->destroy     = VecDestroy_MPI;
  v->view        = VecView_MPI;
  s->n           = n;
  s->nghost      = nghost;
  s->N           = N;
  v->n           = n;
  v->N           = N;
  v->mapping     = 0;
  s->size        = size;
  s->rank        = rank;
  s->array       = (Scalar *) PetscMalloc((nghost+1)*sizeof(Scalar));CHKPTRQ(s->array);
  s->array_allocated = s->array;

  PetscMemzero(s->array,n*sizeof(Scalar));
  s->ownership   = (int *) (s + 1);
  s->insertmode  = NOT_SET_VALUES;
  if (owners) {
    PetscMemcpy(s->ownership,owners,(size+1)*sizeof(int));
  }
  else {
    MPI_Allgather(&n,1,MPI_INT,s->ownership+1,1,MPI_INT,comm);
    s->ownership[0] = 0;
    for (i=2; i<=size; i++ ) {
      s->ownership[i] += s->ownership[i-1];
    }
  }
  s->stash.donotstash = 0;
  s->stash.nmax       = 10;
  s->stash.n          = 0;
  s->stash.array      = (Scalar *) PetscMalloc(10*(sizeof(Scalar)+sizeof(int)));
  CHKPTRQ(s->stash.array);
  PLogObjectMemory(v,10*sizeof(Scalar) + 10 *sizeof(int));
  s->stash.idx = (int *) (s->stash.array + 10);
  *vv = v;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecCreateMPI"
/*@C
   VecCreateMPI - Creates a parallel vector.

   Input Parameters:
.  comm - the MPI communicator to use
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  vv - the vector
 
   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, create, MPI

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPIGhost()
@*/ 
int VecCreateMPI(MPI_Comm comm,int n,int N,Vec *vv)
{
  int sum, work = n, size, rank;
  *vv = 0;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank); 
  if (N == PETSC_DECIDE) { 
    MPI_Allreduce( &work, &sum,1,MPI_INT,MPI_SUM,comm );
    N = sum;
  }
  if (n == PETSC_DECIDE) { 
    n = N/size + ((N % size) > rank);
  }
  return VecCreateMPI_Private(comm,n,n,N,size,rank,0,vv);
}

/*@C
   VecCreateGhost - Creates a parallel vector with ghost padding on each processor.

   Input Parameters:
.  comm - the MPI communicator to use
.  n - local vector length 
.  nghost - local vector length including ghost points
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  lv - the local vector representation (with ghost points as part of vector)
.  vv - the global vector representation (without ghost points as part of vector)
 
   Notes:
   The two vectors returned share the same array storage space.
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector. 

.keywords: vector, create, MPI, ghost points, ghost padding

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI()
@*/ 
int VecCreateGhost(MPI_Comm comm,int n,int nghost,int N,Vec *lv,Vec *vv)
{
  int    sum, work = n, size, rank, ierr;
  Scalar *array;

  *vv = 0;

  if (n == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Must set local ghost size");
  if (nghost < n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Ghost padded length must be no shorter then length");

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank); 
  if (N == PETSC_DECIDE) { 
    MPI_Allreduce( &work, &sum,1,MPI_INT,MPI_SUM,comm );
    N = sum;
  }
  ierr = VecCreateMPI_Private(comm,n,nghost,N,size,rank,0,vv); CHKERRQ(ierr);
  ierr = VecGetArray(*vv,&array); CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,nghost,array,lv); CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&array); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecDuplicate_MPI"
int VecDuplicate_MPI( Vec win, Vec *v)
{
  int     ierr;
  Vec_MPI *vw, *w = (Vec_MPI *)win->data;

  ierr = VecCreateMPI_Private(win->comm,w->n,w->nghost,w->N,w->size,w->rank,w->ownership,v);CHKERRQ(ierr);

  /* New vector should inherit stashing property of parent */
  vw                   = (Vec_MPI *)(*v)->data;
  vw->stash.donotstash = w->stash.donotstash;
  
  (*v)->childcopy    = win->childcopy;
  (*v)->childdestroy = win->childdestroy;
  if (win->mapping) {
    (*v)->mapping = win->mapping;
    ISLocalToGlobalMappingReference(win->mapping);
  }
  if (win->child) return (*win->childcopy)(win->child,&(*v)->child);
  return 0;
}


