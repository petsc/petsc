#ifndef lint
static char vcid[] = "$Id: vinv.c,v 1.12 1995/09/06 03:04:14 bsmith Exp bsmith $";
#endif

#include "vec.h"   /*I "vec.h" I*/
#include "vecimpl.h"

/*@
   VecReciprocal - Replaces each component of a vector by its reciprocal.

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  v - the vector reciprocal

.keywords: vector, reciprocal
@*/
int VecReciprocal(Vec v)
{
  int    ierr, i,n;
  Scalar *x;
  PETSCVALIDHEADERSPECIFIC(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n); CHKERRQ(ierr);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/x[i];
  }
  return 0;
}

/*@
   VecSum - Computes the sum of all the components of a vector.

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  sum - the result

.keywords: vector, sum

.seealso: VecASum()
@*/
int VecSum(Vec v,Scalar *sum)
{
  int    ierr, i,n;
  Scalar *x,lsum = 0.0;

  PETSCVALIDHEADERSPECIFIC(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n); CHKERRQ(ierr);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    lsum += x[i];
  }
#if defined(PETSC_COMPLEX)
  MPI_Allreduce((void *) &lsum,(void *)sum,2,MPI_DOUBLE,MPI_SUM,v->comm);
#else
  MPI_Allreduce((void *) &lsum,(void *)sum,1,MPI_DOUBLE,MPI_SUM,v->comm);
#endif
  return 0;
}

/*@
   VecShift - Shifts all of the components of a vector by computing
   x[i] = x[i] + shift.

   Input Parameters:
.  v - the vector 
.  sum - the shift

   Output Parameter:
.  v - the shifted vector 

.keywords: vector, shift
@*/
int VecShift(Scalar *shift,Vec v)
{
  int    ierr, i,n;
  Scalar *x,lsum = *shift;

  PETSCVALIDHEADERSPECIFIC(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n); CHKERRQ(ierr);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    x[i] += lsum;
  }
  return 0;
}
/*@
   VecAbs - Replaces every element in a vector with its absolute value.

   Input Parameters:
.  v - the vector 

.keywords: vector,absolute value
@*/
int VecAbs(Vec v)
{
  int    ierr, i,n;
  Scalar *x;

  PETSCVALIDHEADERSPECIFIC(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n); CHKERRQ(ierr);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
    x[i] = abs(x[i]);
#else
    if (x[i] < 0.0) x[i] = -x[i];
#endif
  }
  return 0;
}
