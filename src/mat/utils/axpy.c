#define PETSCMAT_DLL

#include "private/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY"
/*@
   MatAXPY - Computes Y = a*X + Y.

   Collective on Mat

   Input Parameters:
+  a - the scalar multiplier
.  X - the first matrix
.  Y - the second matrix
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN 
         or SUBSET_NONZERO_PATTERN (nonzeros of X is a subset of Y's)

   Notes:
     Will only be efficient if one has the SAME_NONZERO_PATTERN or SUBSET_NONZERO_PATTERN

   Level: intermediate

.keywords: matrix, add

.seealso: MatAYPX()
 @*/
PetscErrorCode PETSCMAT_DLLEXPORT MatAXPY(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  PetscInt       m1,m2,n1,n2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE,3); 
  PetscValidHeaderSpecific(Y,MAT_COOKIE,1);

  ierr = MatGetSize(X,&m1,&n1);CHKERRQ(ierr);
  ierr = MatGetSize(Y,&m2,&n2);CHKERRQ(ierr);
  if (m1 != m2 || n1 != n2) SETERRQ4(PETSC_ERR_ARG_SIZ,"Non conforming matrix add: %D %D %D %D",m1,m2,n1,n2);

  if (Y->ops->axpy) {
    ierr = (*Y->ops->axpy)(Y,a,X,str);CHKERRQ(ierr);
  } else {
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_Basic"
PetscErrorCode MatAXPY_Basic(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscInt          i,start,end,j,ncols,m,n;
  PetscErrorCode    ierr;
  const PetscInt    *row;
  PetscScalar       *val;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatGetSize(X,&m,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(X,&start,&end);CHKERRQ(ierr);
  if (a == 1.0) {
    for (i = start; i < end; i++) {
      ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMalloc((n+1)*sizeof(PetscScalar),&val);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      for (j=0; j<ncols; j++) {
	val[j] = a*vals[j];
      }
      ierr = MatSetValues(Y,1,&i,ncols,row,val,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
    }
    ierr = PetscFree(val);CHKERRQ(ierr);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatShift(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  PetscInt       i,start,end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE,1);
  if (!Y->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (Y->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(Y);CHKERRQ(ierr);

  if (Y->ops->shift) {
    ierr = (*Y->ops->shift)(Y,a);CHKERRQ(ierr);
  } else {
    PetscScalar alpha = a;
    ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatSetValues(Y,1,&i,1,&i,&alpha,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalSet_Default"
PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalSet_Default(Mat Y,Vec D,InsertMode is)
{
  PetscErrorCode ierr;
  PetscInt       i,start,end,vstart,vend;
  PetscScalar    *v;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(D,&vstart,&vend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
  if (vstart != start || vend != end) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Vector ownership range not compatible with matrix: %D %D vec %D %D mat",vstart,vend,start,end);
  }
  ierr = VecGetArray(D,&v);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    ierr = MatSetValues(Y,1,&i,1,&i,v+i-start,is);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(D,&v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalSet(Mat Y,Vec D,InsertMode is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE,1);
  PetscValidHeaderSpecific(D,VEC_COOKIE,2);
  if (Y->ops->diagonalset) {
    ierr = (*Y->ops->diagonalset)(Y,D,is);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalSet_Default(Y,D,is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAYPX"
/*@
   MatAYPX - Computes Y = a*Y + X.

   Collective on Mat

   Input Parameters:
+  a - the PetscScalar multiplier
.  Y - the first matrix
.  X - the second matrix
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or SUBSET_NONZERO_PATTERN 

   Notes:
     Will only be efficient if one has the SAME_NONZERO_PATTERN or SUBSET_NONZERO_PATTERN

   Level: intermediate

.keywords: matrix, add

.seealso: MatAXPY()
 @*/
PetscErrorCode PETSCMAT_DLLEXPORT MatAYPX(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscScalar    one = 1.0;
  PetscErrorCode ierr;
  PetscInt       mX,mY,nX,nY;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE,2);
  PetscValidHeaderSpecific(Y,MAT_COOKIE,1);

  ierr = MatGetSize(X,&mX,&nX);CHKERRQ(ierr);
  ierr = MatGetSize(X,&mY,&nY);CHKERRQ(ierr);
  if (mX != mY || nX != nY) SETERRQ4(PETSC_ERR_ARG_SIZ,"Non conforming matrices: %D %D first %D %D second",mX,mY,nX,nY);

  ierr = MatScale(Y,a);CHKERRQ(ierr);
  ierr = MatAXPY(Y,one,X,str);CHKERRQ(ierr)
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
PetscErrorCode PETSCMAT_DLLEXPORT MatComputeExplicitOperator(Mat inmat,Mat *mat)
{
  Vec            in,out;
  PetscErrorCode ierr;
  PetscInt       i,m,n,M,N,*rows,start,end;
  MPI_Comm       comm;
  PetscScalar    *array,zero = 0.0,one = 1.0;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmat,MAT_COOKIE,1);
  PetscValidPointer(mat,2);

  comm = ((PetscObject)inmat)->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = MatGetLocalSize(inmat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(inmat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetVecs(inmat,&in,&out);CHKERRQ(ierr);
  ierr = VecSetOption(in,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(out,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(*mat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*mat,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*mat,n,PETSC_NULL,N-n,PETSC_NULL);CHKERRQ(ierr);
  }

  for (i=0; i<N; i++) {

    ierr = VecSet(in,zero);CHKERRQ(ierr);
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

/* Get the map xtoy which is used by MatAXPY() in the case of SUBSET_NONZERO_PATTERN */
#undef __FUNCT__  
#define __FUNCT__ "MatAXPYGetxtoy_Private"
PetscErrorCode MatAXPYGetxtoy_Private(PetscInt m,PetscInt *xi,PetscInt *xj,PetscInt *xgarray, PetscInt *yi,PetscInt *yj,PetscInt *ygarray, PetscInt **xtoy)
{
  PetscErrorCode ierr;
  PetscInt       row,i,nz,xcol,ycol,jx,jy,*x2y;

  PetscFunctionBegin;
  ierr = PetscMalloc(xi[m]*sizeof(PetscInt),&x2y);CHKERRQ(ierr);
  i = 0;    
  for (row=0; row<m; row++){
    nz = xi[1] - xi[0];
    jy = 0;
    for (jx=0; jx<nz; jx++,jy++){
      if (xgarray && ygarray){
        xcol = xgarray[xj[*xi + jx]];
        ycol = ygarray[yj[*yi + jy]];  
      } else {
        xcol = xj[*xi + jx];
        ycol = yj[*yi + jy];  /* col index for y */
      }
      while ( ycol < xcol ) {
        jy++; 
        if (ygarray){
          ycol = ygarray[yj[*yi + jy]]; 
        } else {
          ycol = yj[*yi + jy]; 
        }
      }
      if (xcol != ycol) SETERRQ2(PETSC_ERR_ARG_WRONG,"X matrix entry (%D,%D) is not in Y matrix",row,ycol);
      x2y[i++] = *yi + jy;
    }
    xi++; yi++;
  }
  *xtoy = x2y;
  PetscFunctionReturn(0);
}
