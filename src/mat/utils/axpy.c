/*$Id: axpy.c,v 1.45 2000/05/05 22:16:35 balay Exp bsmith $*/

#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatAXPY"
/*@
   MatAXPY - Computes Y = a*X + Y.

   Collective on Mat

   Input Parameters:
+  X, Y - the matrices
-  a - the scalar multiplier

   Contributed by: Matthew Knepley

   Notes:
   Since the current implementation of MatAXPY() uses MatGetRow() to access
   matrix data, efficiency is somewhat limited.

   Level: intermediate

.keywords: matrix, add

.seealso: MatAYPX()
 @*/
int MatAXPY(Scalar *a,Mat X,Mat Y)
{
  int    m1,m2,n1,n2,i,*row,start,end,j,ncols,ierr;
  Scalar *val,*vals;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE); 
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);

  MatGetSize(X,&m1,&n1);  MatGetSize(Y,&m2,&n2);
  if (m1 != m2 || n1 != n2) SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Non conforming matrix add: %d %d %d %d",m1,m2,n1,n2);

  if (X->ops->axpy) {
    ierr = (*X->ops->axpy)(a,X,Y);CHKERRQ(ierr);
  } else {
    ierr = MatGetOwnershipRange(X,&start,&end);CHKERRQ(ierr);
    if (*a == 1.0) {
      for (i = start; i < end; i++) {
        ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
        ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      }
    } else {
      vals = (Scalar*)PetscMalloc((n1+1)*sizeof(Scalar));CHKPTRQ(vals);
      for (i=start; i<end; i++) {
        ierr = MatGetRow(X,i,&ncols,&row,&val);CHKERRQ(ierr);
        for (j=0; j<ncols; j++) {
          vals[j] = (*a)*val[j];
        }
        ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(X,i,&ncols,&row,&val);CHKERRQ(ierr);
      }
      ierr = PetscFree(vals);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatShift"
/*@
   MatShift - Computes Y =  Y + a I, where a is a scalar and I is the identity matrix.

   Collective on Mat

   Input Parameters:
+  Y - the matrices
-  a - the scalar 

   Level: intermediate

.keywords: matrix, add, shift

.seealso: MatDiagonalShift()
 @*/
int MatShift(Scalar *a,Mat Y)
{
  int    i,start,end,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);
  if (Y->ops->shift) {
    ierr = (*Y->ops->shift)(a,Y);CHKERRQ(ierr);
  } else {
    ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatSetValues(Y,1,&i,1,&i,a,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatDiagonalShift"
/*@
   MatDiagonalShift - Computes Y = Y + D, where D is a diagonal matrix
   that is represented as a vector.

   Input Parameters:
+  Y - the input matrix
-  D - the diagonal matrix, represented as a vector

   Input Parameters:
.  Y - the shifted ouput matrix

   Collective on Mat and Vec

   Level: intermediate

.keywords: matrix, add, shift, diagonal

.seealso: MatShift()
@*/
int MatDiagonalShift(Mat Y,Vec D)
{
  int    i,start,end,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidHeaderSpecific(D,VEC_COOKIE);
  if (Y->ops->diagonalshift) {
    ierr = (*Y->ops->diagonalshift)(D,Y);CHKERRQ(ierr);
  } else {
    int    vstart,vend;
    Scalar *v;
    ierr = VecGetOwnershipRange(D,&vstart,&vend);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
    if (vstart != start || vend != end) {
      SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Vector ownership range not compatible with matrix: %d %d vec %d %d mat",vstart,vend,start,end);
    }

    ierr = VecGetArray(D,&v);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatSetValues(Y,1,&i,1,&i,v+i-start,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(D,&v);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatAYPX"
/*@
   MatAYPX - Computes Y = X + a*Y.

   Collective on Mat

   Input Parameters:
+  X,Y - the matrices
-  a - the scalar multiplier

   Contributed by: Matthew Knepley

   Notes:
   This routine currently uses the MatAXPY() implementation.

   Level: intermediate

.keywords: matrix, add

.seealso: MatAXPY()
 @*/
int MatAYPX(Scalar *a,Mat X,Mat Y)
{
  Scalar one = 1.0;
  int    mX,mY,nX,nY,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE);
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);

  ierr = MatGetSize(X,&mX,&nX);CHKERRQ(ierr);
  ierr = MatGetSize(X,&mY,&nY);CHKERRQ(ierr);
  if (mX != mY || nX != nY) SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Non conforming matrices: %d %d first %d %d second",mX,mY,nX,nY);

  ierr = MatScale(a,Y);CHKERRQ(ierr);
  ierr = MatAXPY(&one,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
