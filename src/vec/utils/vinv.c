#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vinv.c,v 1.35 1997/08/22 15:10:22 bsmith Exp bsmith $";
#endif
/*
     Some useful vector utility functions.
*/
#include "vec.h"   /*I "vec.h" I*/
#include "src/vec/vecimpl.h"

#undef __FUNC__  
#define __FUNC__ "VecReciprocal"
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
  int    i,n;
  Scalar *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  VecGetLocalSize_Fast(v,n);
  VecGetArray_Fast(v,x);
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/x[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSum"
/*@
   VecSum - Computes the sum of all the components of a vector.

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  sum - the result

.keywords: vector, sum

.seealso: VecNorm()
@*/
int VecSum(Vec v,Scalar *sum)
{
  int    i,n;
  Scalar *x,lsum = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  VecGetLocalSize_Fast(v,n);
  VecGetArray_Fast(v,x);
  for ( i=0; i<n; i++ ) {
    lsum += x[i];
  }
#if defined(USE_PETSC_COMPLEX)
  MPI_Allreduce(&lsum,sum,2,MPI_DOUBLE,MPI_SUM,v->comm);
#else
  MPI_Allreduce(&lsum,sum,1,MPI_DOUBLE,MPI_SUM,v->comm);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecShift"
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
  int    i,n;
  Scalar *x,lsum = *shift;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  VecGetLocalSize_Fast(v,n); 
  VecGetArray_Fast(v,x);
  for ( i=0; i<n; i++ ) {
    x[i] += lsum;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAbs"
/*@
   VecAbs - Replaces every element in a vector with its absolute value.

   Input Parameters:
.  v - the vector 

.keywords: vector,absolute value
@*/
int VecAbs(Vec v)
{
  int    i,n;
  Scalar *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  VecGetLocalSize_Fast(v,n);
  VecGetArray_Fast(v,x);
  for ( i=0; i<n; i++ ) {
    x[i] = PetscAbsScalar(x[i]);
  }
  PetscFunctionReturn(0);
}

#include "src/vec/impls/dvecimpl.h"
#undef __FUNC__  
#define __FUNC__ "VecPlaceArray"
/*@
   VecPlaceArray - Allows one to replace the array in a vector with 
   a user-provided one. This is useful to avoid copying an array
   into a vector.  FOR EXPERTS ONLY!

   Input Parameters:
.  vec - the vector
.  array - the array

  Notes:
  You should back up the original array by calling VecGetArray() and 
  stashing the value somewhere.  Then when finished using the vector,
  call VecPlaceArray() with that stashed value; otherwise, you may
  lose access to the original array.

.seealso: VecGetArray(), VecRestoreArray()

.keywords: vec, place, array
@*/
int VecPlaceArray(Vec vec,Scalar *array)
{
  Vec_Seq *xin = (Vec_Seq *) vec->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  if (vec->type != VECSEQ && vec->type != VECMPI) SETERRQ(PETSC_ERR_SUP,0,"");
  xin->array = array;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecEqual"
/*@
   VecEqual - Compares two vectors.

   Input Parameters:
.  vec1 - the first matrix
.  vec2 - the second matrix

   Output Parameter:
.  flg : PETSC_TRUE if the vectors are equal;
         PETSC_FALSE otherwise.

.keywords: vec, equal, equivalent
@*/
int VecEqual(Vec vec1,Vec vec2,PetscTruth *flg)
{
  Scalar *v1,*v2;
  int    n1,n2,ierr;

  PetscFunctionBegin;
  ierr = VecGetSize(vec1,&n1); CHKERRQ(ierr);
  ierr = VecGetSize(vec2,&n2); CHKERRQ(ierr);
  if (n1 != n2) { *flg = PETSC_FALSE; PetscFunctionReturn(0);}

  ierr = VecGetArray(vec1,&v1); CHKERRQ(ierr);
  ierr = VecGetArray(vec2,&v2); CHKERRQ(ierr);

  if (PetscMemcmp(v1,v2,n1*sizeof(Scalar))) *flg = PETSC_FALSE;
  else  *flg = PETSC_TRUE;
  
  ierr = VecRestoreArray(vec1,&v1); CHKERRQ(ierr);
  ierr = VecRestoreArray(vec2,&v2); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



