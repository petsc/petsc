
/*
   Defines the sequential BLAS based vectors
*/

#include "system/flog.h"
#include "inline/dot.h"
#include "inline/vmult.h"
#include "inline/setval.h"
#include "inline/copy.h"
#include "inline/axpy.h"

#include <math.h>
#include "vecimpl.h" 
#include "dvecimpl.h" 

static struct _VeOps DvOps = { VeiDVBCreateVector, 
              Veiobtain_vectors, Veirelease_vectors, VeiDVBdot,
              VeiDVmdot, VeiDVBnorm, VeiDVmax, VeiDVBasum, VeiDVBdot,
              VeiDVmdot, VeiDVBscal, VeiDVBcopy, VeiDVset, VeiDVswap,
              VeiDVBaxpy, VeiDVmaxpy, VeiDVaypx, VeiDVwaxpy, VeiDVpmult,
              VeiDVpdiv, 0,0,0,0, 
              VeiDVaddvalues,VeiDVinsertvalues,0,0,
              VeiDVgetarray,VeiDVview, VeiDVsize, VeiDVsize };

int VecCreateSequentialBLAS(n,V)
int n;
Vec *V;
{
  int      size = sizeof(DvVector)+n*sizeof(double);
  Vec      v;
  DvVector *s;
  *V             = 0;
  CREATEHEADER(v,_Vec);
  v->destroy     = VeiDestroyVector;
  s              = (DvVector *) MALLOC(size); CHKPTR(s);
  v->cookie      = VEC_COOKIE;
  v->type        = SEQVECTOR;
  v->ops         = &DvOps;
  v->data        = (void *) s;
  s->n           = n;
  s->array       = (double *)(s + 1);
  *V = v; return 0;
}

int VeiDVBCreateVector( w,V)
DvVector *w;
Vec      *V;
{
  return VecCreateSequentialBLAS(w->n,V);
}

