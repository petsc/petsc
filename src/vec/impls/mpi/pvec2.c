
/* $Id: pvec2.c,v 1.10 1995/07/17 03:53:35 bsmith Exp bsmith $ */

#include <math.h>
#include "pvecimpl.h" 
#include "inline/dot.h"

static int VecMDot_MPI( int nv, Vec xin, Vec *y, Scalar *z )
{
  Scalar *work;
  work = (Scalar *)PETSCMALLOC( nv * sizeof(Scalar) );  CHKPTRQ(work);
  VecMDot_Seq(  nv, xin, y, work );
#if defined(PETSC_COMPLEX)
  MPI_Allreduce( work,z,2*nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#else
  MPI_Allreduce( work,z,nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#endif
  PETSCFREE(work);
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
  MPI_Allreduce( &work, &sum,1,MPI_DOUBLE,MPI_SUM,xin->comm );
  *z = sqrt( sum );
  PLogFlops(2*x->n);
  return 0;
}


static int VecAMax_MPI( Vec xin, int *idx, double *z )
{
  double work;
  /* Find the local max */
  VecAMax_Seq( xin, idx, &work );

  /* Find the global max */
  if (!idx) {
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_MAX,xin->comm );
  }
  else {
    /* Need to use special linked max */
    SETERRQ( 1, "VecAMax_MPI:Parallel max with index not supported" );
  }
  return 0;
}

static int VecMax_MPI( Vec xin, int *idx, double *z )
{
  double work;
  /* Find the local max */
  VecMax_Seq( xin, idx, &work );

  /* Find the global max */
  if (!idx) {
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_MAX,xin->comm );
  }
  else {
    /* Need to use special linked max */
    SETERRQ( 1, "VecMax_MPI:Parallel max with index not supported" );
  }
  return 0;
}

static int VecMin_MPI( Vec xin, int *idx, double *z )
{
  double work;
  /* Find the local Min */
  VecMin_Seq( xin, idx, &work );

  /* Find the global Min */
  if (!idx) {
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_MIN,xin->comm );
  }
  else {
    /* Need to use special linked Min */
    SETERRQ( 1, "VecMin_MPI:Parallel Min with index not supported" );
  }
  return 0;
}
