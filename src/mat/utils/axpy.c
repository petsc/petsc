/*$Id: axpy.c,v 1.54 2001/08/06 21:16:10 bsmith Exp bsmith $*/

#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY"
/*@
   MatAXPY - Computes Y = a*X + Y.

   Collective on Mat

   Input Parameters:
+  a - the scalar multiplier
.  X - the first matrix
.  Y - the second matrix
-  str - either SAME_NONZERO_PATTERN or DIFFERENT_NONZERO_PATTERN

   Contributed by: Matthew Knepley

   Notes:
     Will only be efficient if one has the SAME_NONZERO_PATTERN

   Level: intermediate

.keywords: matrix, add

.seealso: MatAYPX()
 @*/
int MatAXPY(PetscScalar *a,Mat X,Mat Y,MatStructure str)
{
  int         m1,m2,n1,n2,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE); 
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);

  ierr = MatGetSize(X,&m1,&n1);CHKERRQ(ierr);
  ierr = MatGetSize(Y,&m2,&n2);CHKERRQ(ierr);
  if (m1 != m2 || n1 != n2) SETERRQ4(PETSC_ERR_ARG_SIZ,"Non conforming matrix add: %d %d %d %d",m1,m2,n1,n2);

  if (X->ops->axpy) {
    ierr = (*X->ops->axpy)(a,X,Y,str);CHKERRQ(ierr);
  } else {
    ierr = MatAXPY_Basic(a,X,Y,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_Basic"
int MatAXPY_Basic(PetscScalar *a,Mat X,Mat Y,MatStructure str)
{
  int         i,*row,start,end,j,ncols,ierr,m,n;
  PetscScalar *val,*vals;

  PetscFunctionBegin;
  ierr = MatGetSize(X,&m,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(X,&start,&end);CHKERRQ(ierr);
  if (*a == 1.0) {
    for (i = start; i < end; i++) {
      ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMalloc((n+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShift"
/*@
   MatShift - Computes Y =  Y + a I, where a is a PetscScalar and I is the identity matrix.

   Collective on Mat

   Input Parameters:
+  Y - the matrices
-  a - the PetscScalar 

   Level: intermediate

.keywords: matrix, add, shift

.seealso: MatDiagonalSet()
 @*/
int MatShift(PetscScalar *a,Mat Y)
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

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalSet"
/*@
   MatDiagonalSet - Computes Y = Y + D, where D is a diagonal matrix
   that is represented as a vector. Or Y[i,i] = D[i] if InsertMode is
   INSERT_VALUES.

   Input Parameters:
+  Y - the input matrix
.  D - the diagonal matrix, represented as a vector
-  i - INSERT_VALUES or ADD_VALUES

   Collective on Mat and Vec

   Level: intermediate

.keywords: matrix, add, shift, diagonal

.seealso: MatShift()
@*/
int MatDiagonalSet(Mat Y,Vec D,InsertMode is)
{
  int    i,start,end,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidHeaderSpecific(D,VEC_COOKIE);
  if (Y->ops->diagonalset) {
    ierr = (*Y->ops->diagonalset)(Y,D,is);CHKERRQ(ierr);
  } else {
    int    vstart,vend;
    PetscScalar *v;
    ierr = VecGetOwnershipRange(D,&vstart,&vend);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
    if (vstart != start || vend != end) {
      SETERRQ4(PETSC_ERR_ARG_SIZ,"Vector ownership range not compatible with matrix: %d %d vec %d %d mat",vstart,vend,start,end);
    }
    ierr = VecGetArray(D,&v);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatSetValues(Y,1,&i,1,&i,v+i-start,is);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(D,&v);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAYPX"
/*@
   MatAYPX - Computes Y = X + a*Y.

   Collective on Mat

   Input Parameters:
+  X,Y - the matrices
-  a - the PetscScalar multiplier

   Contributed by: Matthew Knepley

   Notes:
   This routine currently uses the MatAXPY() implementation.

   This is slow, if you need it fast send email to petsc-maint@mcs.anl.gov

   Level: intermediate

.keywords: matrix, add

.seealso: MatAXPY()
 @*/
int MatAYPX(PetscScalar *a,Mat X,Mat Y)
{
  PetscScalar one = 1.0;
  int         mX,mY,nX,nY,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE);
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);

  ierr = MatGetSize(X,&mX,&nX);CHKERRQ(ierr);
  ierr = MatGetSize(X,&mY,&nY);CHKERRQ(ierr);
  if (mX != mY || nX != nY) SETERRQ4(PETSC_ERR_ARG_SIZ,"Non conforming matrices: %d %d first %d %d second",mX,mY,nX,nY);

  ierr = MatScale(a,Y);CHKERRQ(ierr);
  ierr = MatAXPY(&one,X,Y,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatComputeExplicitOperator"
/*@
    MatComputeExplicitOperator - Computes the explicit matrix

    Collective on Mat

    Input Parameter:
.   inmat - the matrix

    Output Parameter:
.   mat - the explict preconditioned operator

    Notes:
    This computation is done by applying the operators to columns of the 
    identity matrix.

    Currently, this routine uses a dense matrix format when 1 processor
    is used and a sparse format otherwise.  This routine is costly in general,
    and is recommended for use only with relatively small systems.

    Level: advanced
   
.keywords: Mat, compute, explicit, operator

@*/
int MatComputeExplicitOperator(Mat inmat,Mat *mat)
{
  Vec      in,out;
  int      ierr,i,M,m,size,*rows,start,end;
  MPI_Comm comm;
  PetscScalar   *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmat,MAT_COOKIE);
  PetscValidPointer(mat);

  comm = inmat->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = MatGetLocalSize(inmat,&m,0);CHKERRQ(ierr);
  ierr = MatGetSize(inmat,&M,0);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,m,M,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(in,&out);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  if (size == 1) {
    ierr = MatCreateSeqDense(comm,M,M,PETSC_NULL,mat);CHKERRQ(ierr);
  } else {
    ierr = MatCreateMPIAIJ(comm,m,m,M,M,0,0,0,0,mat);CHKERRQ(ierr);
  }

  for (i=0; i<M; i++) {

    ierr = VecSet(&zero,in);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = MatMult(inmat,in,out);CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

