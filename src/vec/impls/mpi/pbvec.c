
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
  DvPVector *x = (DvPVector *)xin->data, *y = (DvPVector *)yin->data;
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

static int VeiDVPBCreateVector( Vec,Vec *);

static struct _VeOps DvOps = { VeiDVPBCreateVector, 
            Veiobtain_vectors, Veirelease_vectors, VeiDVPBdot, VeiDVPmdot,
            VeiDVPnorm, VeiDVPmax, VeiDVPBasum, VeiDVPBdot, VeiDVPmdot,
            VeiDVBscal, VeiDVBcopy,
            VeiDVset, VeiDVswap, VeiDVBaxpy, VeiDVmaxpy, VeiDVaypx,
            VeiDVwaxpy,
            VeiDVpmult,
            VeiDVpdiv, 0,0,0,0,
            0,0,0,0,
            VeiDVgetarray, VeiDVPview,0,0};

int VecCreateMPIBLAS(MPI_Comm comm,int n,int N, Vec *vv)
{
  Vec       v;
  DvPVector *s;
  int       size,sum, work = n; 
  *vv = 0;
  if (N == -1) { 
    MPI_Allreduce((void *) &work,(void *) &sum,1,MPI_INT,MPI_SUM,comm );
    N = sum;
  }
  if (n == -1) { 
    int numtids,mytid;
    MPI_Comm_size(comm,&numtids);
    MPI_Comm_rank(comm,&mytid); 
    n = N/numtids + ((N % numtids) > mytid);
  }
  size           = sizeof(DvPVector)+n*sizeof(Scalar);
  CREATEHEADER(v,_Vec);
  s              = (DvPVector *) MALLOC(size); CHKPTR(s);
  v->cookie      = VEC_COOKIE;
  v->type        = MPIVECTOR;
  v->ops         = &DvOps;
  v->destroy     = VeiDestroyVector;
  v->data        = (void *) s;
  s->n           = n;
  s->N           = N;
  s->comm        = comm;
  s->array       = (Scalar *)(s + 1);
  *vv = v;
  return 0;
}

static int VeiDVPBCreateVector( Vec win,Vec *v)
{
  DvPVector *w = (DvPVector *) win->data;
  return VecCreateMPIBLAS(w->comm,w->n,w->N,v);
}

