#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vinv.c,v 1.40 1998/04/27 14:31:38 curfman Exp bsmith $";
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

   Collective on Vec

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  v - the vector reciprocal

.keywords: vector, reciprocal
@*/
int VecReciprocal(Vec v)
{
  int    i,n,ierr;
  Scalar *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/x[i];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSum"
/*@
   VecSum - Computes the sum of all the components of a vector.

   Collective on Vec

   Input Parameter:
.  v - the vector 

   Output Parameter:
.  sum - the result

.keywords: vector, sum

.seealso: VecNorm()
@*/
int VecSum(Vec v,Scalar *sum)
{
  int    i,n,ierr;
  Scalar *x,lsum = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    lsum += x[i];
  }
#if defined(USE_PETSC_COMPLEX)
  ierr = MPI_Allreduce(&lsum,sum,2,MPI_DOUBLE,MPI_SUM,v->comm);CHKERRQ(ierr);
#else
  ierr = MPI_Allreduce(&lsum,sum,1,MPI_DOUBLE,MPI_SUM,v->comm);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecShift"
/*@
   VecShift - Shifts all of the components of a vector by computing
   x[i] = x[i] + shift.

   Collective on Vec

   Input Parameters:
+  v - the vector 
-  sum - the shift

   Output Parameter:
.  v - the shifted vector 

.keywords: vector, shift
@*/
int VecShift(Scalar *shift,Vec v)
{
  int    i,n,ierr;
  Scalar *x,lsum = *shift;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr); 
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    x[i] += lsum;
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAbs"
/*@
   VecAbs - Replaces every element in a vector with its absolute value.

   Collective on Vec

   Input Parameters:
.  v - the vector 

.keywords: vector,absolute value
@*/
int VecAbs(Vec v)
{
  int    i,n,ierr;
  Scalar *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    x[i] = PetscAbsScalar(x[i]);
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "VecEqual"
/*@
   VecEqual - Compares two vectors.

   Collective on Vec

   Input Parameters:
+  vec1 - the first matrix
-  vec2 - the second matrix

   Output Parameter:
.  flg - PETSC_TRUE if the vectors are equal; PETSC_FALSE otherwise.

.keywords: vec, equal, equivalent
@*/
int VecEqual(Vec vec1,Vec vec2,PetscTruth *flg)
{
  Scalar *v1,*v2;
  int    n1,n2,ierr,flg1;

  PetscFunctionBegin;
  ierr = VecGetSize(vec1,&n1); CHKERRQ(ierr);
  ierr = VecGetSize(vec2,&n2); CHKERRQ(ierr);
  if (n1 != n2) {
    flg1 = PETSC_FALSE;
  } else {
    ierr = VecGetArray(vec1,&v1); CHKERRQ(ierr);
    ierr = VecGetArray(vec2,&v2); CHKERRQ(ierr);

    if (PetscMemcmp(v1,v2,n1*sizeof(Scalar))) flg1 = PETSC_FALSE;
    else  flg1 = PETSC_TRUE;
    ierr = VecRestoreArray(vec1,&v1); CHKERRQ(ierr);
    ierr = VecRestoreArray(vec2,&v2); CHKERRQ(ierr);
  }

  /* combine results from all processors */
  MPI_Allreduce(&flg1,flg,1,MPI_INT,MPI_MIN,vec1->comm);
  

  PetscFunctionReturn(0);
}



