

#ifndef lint
static char vcid[] = "$Id: pbvec.c,v 1.37 1995/08/15 20:26:37 bsmith Exp bsmith $";
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
   Dot products and norms.  Note that they use the processor 
   subset to allow the vectors to be distributed across any 
   subset of the processors.
 */

static int VecDot_MPIBlas( Vec xin, Vec yin, Scalar *z )
{
  Scalar    sum, work;
  VecDot_Blas(  xin, yin, &work );
/*
   This is a ugly hack. But to do it right is kind of silly.
*/
#if defined(PETSC_COMPLEX)
  MPI_Allreduce((void *) &work,(void *) &sum,2,MPI_DOUBLE,MPI_SUM,xin->comm );
#else
  MPI_Allreduce((void *) &work,(void *) &sum,1,MPI_DOUBLE,MPI_SUM,xin->comm );
#endif
  *z = sum;
  return 0;
}

static int VecAsum_MPIBlas(  Vec xin, double *z )
{
  double work;
  VecAsum_Blas( xin, &work );
  MPI_Allreduce((void *) &work,(void *) z,1,MPI_DOUBLE,MPI_SUM,xin->comm );
  return 0;
}

static int VecDuplicate_MPIBlas( Vec, Vec *);

static struct _VeOps DvOps = { VecDuplicate_MPIBlas, 
            Veiobtain_vectors, Veirelease_vectors, VecDot_MPIBlas, 
            VecMDot_MPI,
            VecNorm_MPI, VecAMax_MPI, VecAsum_MPIBlas, VecDot_MPIBlas, 
            VecMDot_MPI,
            VecScale_Blas, VecCopy_Blas,
            VecSet_Seq, VecSwap_Blas, VecAXPY_Blas, VecMAXPY_Seq, VecAYPX_Seq,
            VecWAXPY_Seq, VecPMult_Seq,
            VecPDiv_Seq, 
            VecSetValues_MPI,
            VecAssemblyBegin_MPI,VecAssemblyEnd_MPI,
            VecGetArray_Seq,VecGetSize_MPI,VecGetSize_Seq,
            VecGetOwnershipRange_MPI,0,VecMax_MPI,VecMin_MPI};

static int VecCreateMPIBLASBase(MPI_Comm comm,int n,int N,int numtids,
                                int mytid,int *owners,Vec *vv)
{
  Vec     v;
  Vec_MPI *s;
  int     size,i;
  *vv = 0;

  size           = sizeof(Vec_MPI)+(numtids+1)*sizeof(int);
  PETSCHEADERCREATE(v,_Vec,VEC_COOKIE,VECMPI,comm);
  PLogObjectCreate(v);
  PLogObjectMemory(v,size + sizeof(struct _Vec) + (n+1)*sizeof(Scalar));
  s              = (Vec_MPI *) PETSCMALLOC(size); CHKPTRQ(s);
  v->ops         = &DvOps;
  v->data        = (void *) s;
  v->destroy     = VecDestroy_MPI;
  v->view        = VecView_MPI;
  s->n           = n;
  s->N           = N;
  s->numtids     = numtids;
  s->mytid       = mytid;
  s->array       = (Scalar *) PETSCMALLOC((n+1)*sizeof(Scalar));CHKPTRQ(s->array);
  s->ownership   = (int *) (s + 1);
  s->insertmode  = NOTSETVALUES;
  if (owners) {
    PETSCMEMCPY(s->ownership,owners,(numtids+1)*sizeof(int));
  }
  else {
    MPI_Allgather(&n,1,MPI_INT,s->ownership+1,1,MPI_INT,comm);
    s->ownership[0] = 0;
    for (i=2; i<=numtids; i++ ) {
      s->ownership[i] += s->ownership[i-1];
    }
  }
  s->stash.nmax = 10; s->stash.n = 0;
  s->stash.array = (Scalar *) PETSCMALLOC( 10*sizeof(Scalar) + 10 *sizeof(int) );
  PLogObjectMemory(v,10*sizeof(Scalar) + 10 *sizeof(int));
  CHKPTRQ(s->stash.array);
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

.seealso: VecCreateSequential(), VecCreate(), VecDuplicate(), VecGetVecs()
@*/ 
int VecCreateMPI(MPI_Comm comm,int n,int N,Vec *vv)
{
  int       sum, work = n; 
  int       numtids,mytid;
  *vv = 0;

  MPI_Comm_size(comm,&numtids);
  MPI_Comm_rank(comm,&mytid); 
  if (N == PETSC_DECIDE) { 
    MPI_Allreduce((void *) &work,(void *) &sum,1,MPI_INT,MPI_SUM,comm );
    N = sum;
  }
  if (n == PETSC_DECIDE) { 
    n = N/numtids + ((N % numtids) > mytid);
  }
  return VecCreateMPIBLASBase(comm,n,N,numtids,mytid,0,vv);
}

static int VecDuplicate_MPIBlas( Vec win, Vec *v)
{
  Vec_MPI *w = (Vec_MPI *)win->data;
  return VecCreateMPIBLASBase(win->comm,w->n,w->N,w->numtids,w->mytid,
                              w->ownership,v);
}


