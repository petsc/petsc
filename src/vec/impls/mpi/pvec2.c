
/* cannot have vcid because included in other files */

#include <math.h>
#include "pvecimpl.h" 
#include "inline/dot.h"

static int VecMDot_MPI( int nv, Vec xin, Vec *y, Scalar *z )
{
  Vec_MPI *x = (Vec_MPI *)xin->data;
  Scalar *work;
  work = (Scalar *)MALLOC( nv * sizeof(Scalar) );  CHKPTR(work);
  VecMDot_Seq(  nv, xin, y, work );
  MPI_Allreduce((void *) work,(void *)z,nv,MPI_SCALAR,MPI_SUM,x->comm );
  FREE(work);
  return 0;
}

static int VecNorm_MPI(  Vec xin, double *z )
{
  Vec_MPI *x = (Vec_MPI *) xin->data;
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


static int VecAMax_MPI( Vec xin, int *idx, double *z )
{
  Vec_MPI *x = (Vec_MPI *) xin->data;
  double work;
  /* Find the local max */
  VecAMax_Seq( xin, idx, &work );

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

static int VecMax_MPI( Vec xin, int *idx, double *z )
{
  Vec_MPI *x = (Vec_MPI *) xin->data;
  double work;
  /* Find the local max */
  VecMax_Seq( xin, idx, &work );

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

static int VecMin_MPI( Vec xin, int *idx, double *z )
{
  Vec_MPI *x = (Vec_MPI *) xin->data;
  double work;
  /* Find the local Min */
  VecMin_Seq( xin, idx, &work );

  /* Find the global Min */
  if (!idx) {
    MPI_Allreduce((void *) &work,(void *) z,1,MPI_DOUBLE,MPI_MIN,x->comm );
  }
  else {
    /* Need to use special linked Min */
    SETERR( 1, "Parallel Min with index not yet supported" );
  }
  return 0;
}
