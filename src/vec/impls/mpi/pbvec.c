
#ifndef lint
static char vcid[] = "$Id: pbvec.c,v 1.46 1995/11/01 19:08:38 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "inline/dot.h"
#include <math.h>
#include "pvecimpl.h"   /*I  "vec.h"   I*/

#include "../bvec1.c"
#include "../dvec2.c"
#include "pdvec.c"
#include "pvec2.c"

/*
   This file contains routines for Parallel vector operations.
   Basically, these are just the local routine EXCEPT for the
   Dot products and norms.
 */

static int VecDot_MPI( Vec xin, Vec yin, Scalar *z )
{
  Scalar    sum, work;
  VecDot_Seq(  xin, yin, &work );
/*
   This is a ugly hack. But to do it right is kind of silly.
*/
#if defined(PETSC_COMPLEX)
  MPI_Allreduce( &work, &sum,2,MPI_DOUBLE,MPI_SUM,xin->comm );
#else
  MPI_Allreduce( &work, &sum,1,MPI_DOUBLE,MPI_SUM,xin->comm );
#endif
  *z = sum;
  return 0;
}

static int VecDuplicate_MPI( Vec, Vec *);

static struct _VeOps DvOps = { VecDuplicate_MPI, 
            Veiobtain_vectors, Veirelease_vectors, VecDot_MPI, 
            VecMDot_MPI,
            VecNorm_MPI, VecDot_MPI, 
            VecMDot_MPI,
            VecScale_Seq, VecCopy_Seq,
            VecSet_Seq, VecSwap_Seq, VecAXPY_Seq, VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPMult_Seq,
            VecPDiv_Seq, 
            VecSetValues_MPI,
            VecAssemblyBegin_MPI,VecAssemblyEnd_MPI,
            VecGetArray_Seq,VecGetSize_MPI,VecGetSize_Seq,
            VecGetOwnershipRange_MPI,0,VecMax_MPI,VecMin_MPI};

static int VecCreateMPIBase(MPI_Comm comm,int n,int N,int size,
                                int rank,int *owners,Vec *vv)
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
  s->N           = N;
  s->size        = size;
  s->rank        = rank;
  s->array       = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar));CHKPTRQ(s->array);
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
  s->stash.nmax = 10; s->stash.n = 0;
  s->stash.array = (Scalar *) PetscMalloc( 10*sizeof(Scalar) + 10*sizeof(int) );
  CHKPTRQ(s->stash.array);
  PLogObjectMemory(v,10*sizeof(Scalar) + 10 *sizeof(int));
  s->stash.idx = (int *) (s->stash.array + 10);
  *vv = v;
  return 0;
}

/*@C
   VecCreateMPI - Creates a parallel vector.

   Input Parameters:
.  comm - the MPI communicator to use
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  vv - the vector
 
   Notes:
   Use VecDuplicate() or VecGetVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, create, MPI

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecGetVecs()
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
  return VecCreateMPIBase(comm,n,N,size,rank,0,vv);
}

static int VecDuplicate_MPI( Vec win, Vec *v)
{
  Vec_MPI *w = (Vec_MPI *)win->data;
  return VecCreateMPIBase(win->comm,w->n,w->N,w->size,w->rank,w->ownership,v);
}


