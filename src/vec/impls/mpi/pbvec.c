#ifndef lint
static char vcid[] = "$Id: $";
#endif

#include "ptscimpl.h"
#include "inline/dot.h"
#include "sys/flog.h"
#include <math.h>
#include "pvecimpl.h" 

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


static int VeiDVPBdot( Vec xin, Vec yin, Scalar *z )
{
  DvPVector *x = (DvPVector *)xin->data;
  Scalar    sum, work;
  VeiDVBdot(  xin, yin, &work );
  MPI_Allreduce((void *) &work,(void *) &sum,1,MPI_SCALAR,MPI_SUM,x->comm );
  *z = sum;
  return 0;
}

static int VeiDVPBasum(  Vec xin, double *z )
{
  DvPVector *x = (DvPVector *) xin->data;
  double work;
  VeiDVBasum( xin, &work );
  MPI_Allreduce((void *) &work,(void *) z,1,MPI_DOUBLE,MPI_SUM,x->comm );
  return 0;
}

static int VeiDVPCreateVectorMPIBLAS( Vec, Vec *);

static struct _VeOps DvOps = { VeiDVPCreateVectorMPIBLAS, 
            Veiobtain_vectors, Veirelease_vectors, VeiDVPBdot, VeiDVPmdot,
            VeiDVPnorm, VeiDVPmax, VeiDVPBasum, VeiDVPBdot, VeiDVPmdot,
            VeiDVBscal, VeiDVBcopy,
            VeiDVset, VeiDVBswap, VeiDVBaxpy, VeiDVmaxpy, VeiDVaypx,
            VeiDVwaxpy,
            VeiDVpmult,
            VeiDVpdiv,
            VeiPDVinsertvalues,
            VeiDVPBeginAssembly,VeiDVPEndAssembly,
            VeiDVgetarray, VeiPgsize,VeiDVsize,VeiPrange};

static int VecCreateMPIBLASBase(MPI_Comm comm,int n,int N,int numtids,int mytid,
                            int *owners,Vec *vv)
{
  Vec       v;
  DvPVector *s;
  int       size,i;
  *vv = 0;

  size           = sizeof(DvPVector)+n*sizeof(Scalar)+(numtids+1)*sizeof(int);
  CREATEHEADER(v,_Vec);
  s              = (DvPVector *) MALLOC(size); CHKPTR(s);
  v->cookie      = VEC_COOKIE;
  v->type        = MPIVECTOR;
  v->ops         = &DvOps;
  v->data        = (void *) s;
  v->destroy     = VeiPDestroyVector;
  v->view        = VeiDVPview;
  s->n           = n;
  s->N           = N;
  s->comm        = comm;
  s->numtids     = numtids;
  s->mytid       = mytid;
  s->array       = (Scalar *)(s + 1);
  s->ownership   = (int *) (s->array + n);
  s->insertmode  = NotSetValues;
  if (owners) {
    MEMCPY(s->ownership,owners,(numtids+1)*sizeof(int));
  }
  else {
    MPI_Allgather(&n,1,MPI_INT,s->ownership+1,1,MPI_INT,comm);
    s->ownership[0] = 0;
    for (i=2; i<=numtids; i++ ) {
      s->ownership[i] += s->ownership[i-1];
    }
  }
  s->stash.nmax = 10; s->stash.n = 0;
  s->stash.array = (Scalar *) MALLOC( 10*sizeof(Scalar) + 10 *sizeof(int) );
  CHKPTR(s->stash.array);
  s->stash.idx = (int *) (s->stash.array + 10);
  *vv = v;
  return 0;
}

/*@
     VecCreateMPIBLAS - Creates parallel vector.

  Input Parameters:
.   comm - the MPI communicator to use
.   n - length of local piece of vector (or -1 if N is given)
.   N - global lenght of vector (or -1 if n is given)

  Output Parameter:
.  vv - the vector
 
  Keywords: vector, create, MPI
@*/ 
int VecCreateMPIBLAS(MPI_Comm comm,int n,int N,Vec *vv)
{
  int       sum, work = n; 
  int       numtids,mytid;
  *vv = 0;

  MPI_Comm_size(comm,&numtids);
  MPI_Comm_rank(comm,&mytid); 
  if (N == -1) { 
    MPI_Allreduce((void *) &work,(void *) &sum,1,MPI_INT,MPI_SUM,comm );
    N = sum;
  }
  if (n == -1) { 
    n = N/numtids + ((N % numtids) > mytid);
  }
  return VecCreateMPIBLASBase(comm,n,N,numtids,mytid,0,vv);
}

static int VeiDVPCreateVectorMPIBLAS( Vec win, Vec *v)
{
  DvPVector *w = (DvPVector *)win->data;
  return VecCreateMPIBLASBase(w->comm,w->n,w->N,w->numtids,w->mytid,w->ownership,v);
}


