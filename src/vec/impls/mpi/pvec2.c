#ifndef lint
static char vcid[] = "$Id: $";
#endif

#include <math.h>
#include "pvecimpl.h" 
#include "inline/dot.h"
#include "sys/flog.h"


static int VeiDVPmdot( int nv, Vec xin, Vec *y, Scalar *z )
{
  DvPVector *x = (DvPVector *)xin->data;
  Scalar *work;
  work = (Scalar *)MALLOC( nv * sizeof(Scalar) );  CHKPTR(work);
  VeiDVmdot(  nv, xin, y, work );
  MPI_Allreduce((void *) work,(void *)z,nv,MPI_SCALAR,MPI_SUM,x->comm );
  FREE(work);
  return 0;
}

static int VeiDVPnorm(  Vec xin, double *z )
{
  DvPVector *x = (DvPVector *) xin->data;
  double sum, work = 0.0;
  Scalar  *xx = x->array;
  register int n = x->n;
#if defined(PETSC_COMPLEX)
  int i;
  for (i=0; i<n; i++ ) {
    work += real(conj(xx[i])*xx[i]);
  }
#else
  SQR(work,xx,n);
#endif
  MPI_Allreduce((void *) &work,(void *) &sum,1,MPI_DOUBLE,MPI_SUM,x->comm );
  *z = sqrt( sum );
  return 0;
}


static int VeiDVPmax( Vec xin, int *idx, double *z )
{
  DvPVector *x = (DvPVector *) xin->data;
  double work;
  /* Find the local max */
  VeiDVmax( xin, idx, &work );

  /* Find the global max */
  if (!idx) {
    MPI_Allreduce((void *) &work,(void *) z,1,MPI_DOUBLE,MPI_MAX,x->comm );
  }
  else {
    /* Need to use special linked max */
    SETERR( 1, "Parallel max with index not yet supported" );
  }
  return 0;
}

