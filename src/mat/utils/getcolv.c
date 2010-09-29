#define PETSCMAT_DLL

#include "private/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnVector"
/*@
   MatGetColumnVector - Gets the values from a given column of a matrix.

   Not Collective

   Input Parameters:
+  A - the matrix
.  yy - the vector
-  c - the column requested (in global numbering)

   Level: advanced

   Notes:
   Each processor for which this is called gets the values for its rows.

   Since PETSc matrices are usually stored in compressed row format, this routine
   will generally be slow.

   The vector must have the same parallel row layout as the matrix.

   Contributed by: Denis Vanderstraeten

.keywords: matrix, column, get 

.seealso: MatGetRow(), MatGetDiagonal()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetColumnVector(Mat A,Vec yy,PetscInt col)
{
  PetscScalar        *y;
  const PetscScalar  *v;
  PetscErrorCode     ierr;
  PetscInt           i,j,nz,N,Rs,Re,rs,re;
  const PetscInt     *idx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1); 
  PetscValidHeaderSpecific(yy,VEC_COOKIE,2); 
  if (col < 0)  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Requested negative column: %D",col);
  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  if (col >= N)  SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Requested column %D larger than number columns in matrix %D",col,N);
  ierr = MatGetOwnershipRange(A,&Rs,&Re);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);
  if (Rs != rs || Re != re) SETERRQ4(PETSC_ERR_ARG_INCOMP,"Matrix %D %D does not have same ownership range (size) as vector %D %D",Rs,Re,rs,re);

  if (A->ops->getcolumnvector) {
    ierr = (*A->ops->getcolumnvector)(A,yy,col);CHKERRQ(ierr);
  } else {
    ierr = VecSet(yy,0.0);CHKERRQ(ierr);
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    
    for (i=Rs; i<Re; i++) {
      ierr = MatGetRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
      if (nz && idx[0] <= col) {
	/*
          Should use faster search here 
	*/
	for (j=0; j<nz; j++) {
	  if (idx[j] >= col) {
	    if (idx[j] == col) y[i-rs] = v[j];
	    break;
	  }
	}
      }
      ierr = MatRestoreRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}

#include "../src/mat/impls/aij/seq/aij.h"

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_SeqAIJ"
PetscErrorCode MatGetColumnNorms_SeqAIJ(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,m,n;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscMemzero(norms,n*sizeof(PetscReal));CHKERRQ(ierr);  
  if (type == NORM_2) {
    for (i=0; i<aij->i[m]; i++) {
      norms[aij->j[i]] += PetscAbsScalar(aij->a[i]*aij->a[i]);
    }
  } else if (type == NORM_1) {
    for (i=0; i<aij->i[m]; i++) {
      norms[aij->j[i]] += PetscAbsScalar(aij->a[i]);
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<aij->i[m]; i++) {
      norms[aij->j[i]] = PetscMax(PetscAbsScalar(aij->a[i]),norms[aij->j[i]]);
    }
  } else SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown NormType");

  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = sqrt(norms[i]);
  }
  PetscFunctionReturn(0);
}

#include "../src/mat/impls/aij/mpi/mpiaij.h"

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_MPIAIJ"
PetscErrorCode MatGetColumnNorms_MPIAIJ(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)A->data;
  PetscInt       i,n,*garray = aij->garray;
  Mat_SeqAIJ     *a_aij = (Mat_SeqAIJ*) aij->A->data;
  Mat_SeqAIJ     *b_aij = (Mat_SeqAIJ*) aij->B->data;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMemzero(work,n*sizeof(PetscReal));CHKERRQ(ierr);  
  if (type == NORM_2) {
    for (i=0; i<a_aij->i[aij->A->rmap->n]; i++) {
      work[A->cmap->rstart + a_aij->j[i]] += PetscAbsScalar(a_aij->a[i]*a_aij->a[i]);
    }
    for (i=0; i<b_aij->i[aij->B->rmap->n]; i++) {
      work[garray[b_aij->j[i]]] += PetscAbsScalar(b_aij->a[i]*b_aij->a[i]);
    }
  } else if (type == NORM_1) {
    for (i=0; i<a_aij->i[aij->A->rmap->n]; i++) {
      work[A->cmap->rstart + a_aij->j[i]] += PetscAbsScalar(a_aij->a[i]);
    }
    for (i=0; i<b_aij->i[aij->B->rmap->n]; i++) {
      work[garray[b_aij->j[i]]] += PetscAbsScalar(b_aij->a[i]);
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<a_aij->i[aij->A->rmap->n]; i++) {
      work[A->cmap->rstart + a_aij->j[i]] = PetscMax(PetscAbsScalar(a_aij->a[i]), work[A->cmap->rstart + a_aij->j[i]]);
    }
    for (i=0; i<b_aij->i[aij->B->rmap->n]; i++) {
      work[garray[b_aij->j[i]]] = PetscMax(PetscAbsScalar(b_aij->a[i]),work[garray[b_aij->j[i]]]);
    }

  } else SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown NormType");
  if (type == NORM_INFINITY) {
    ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPI_MAX,A->hdr.comm);CHKERRQ(ierr);
  } else {
    ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPI_SUM,A->hdr.comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = sqrt(norms[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_SeqDense"
PetscErrorCode MatGetColumnNorms_SeqDense(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m,n;
  PetscScalar    *a;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscMemzero(norms,n*sizeof(PetscReal));CHKERRQ(ierr);  
  ierr = MatGetArray(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++ ){ 
      for (j=0; j<m; j++) {
	norms[i] += PetscAbsScalar(a[j]*a[j]);
      }
      a += m;
    }
  } else if (type == NORM_1) {
    for (i=0; i<n; i++ ){ 
      for (j=0; j<m; j++) {
	norms[i] += PetscAbsScalar(a[j]);
      }
      a += m;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<n; i++ ){ 
      for (j=0; j<m; j++) {
	norms[i] = PetscMax(PetscAbsScalar(a[j]),norms[i]);
      }
      a += m;
    }
  } else SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown NormType");
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = sqrt(norms[i]);
  }
  PetscFunctionReturn(0);
}

#include "../src/mat/impls/dense/mpi/mpidense.h"
#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_MPIDense"
PetscErrorCode MatGetColumnNorms_MPIDense(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = MatGetColumnNorms_SeqDense(a->A,type,work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) work[i] *= work[i];
  }
  if (type == NORM_INFINITY) {
    ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPI_MAX,A->hdr.comm);CHKERRQ(ierr);
  } else {
    ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPI_SUM,A->hdr.comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = sqrt(norms[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms"
/*@
    MatGetColumnNorms - Gets the 2 norms of each column of a sparse or dense matrix.

  Input Parameter:
+  A - the matrix
-  type - NORM_2, NORM_1 or NORM_INFINITY

  Output Parameter:
.  norms - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes: Each process has ALL the column norms after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: MatGetColumns()

@*/
PetscErrorCode MatGetColumnNorms(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatGetColumnNorms_SeqAIJ(A,type,norms);CHKERRQ(ierr);
  } else {
    ierr = PetscTypeCompare((PetscObject)A,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatGetColumnNorms_SeqDense(A,type,norms);CHKERRQ(ierr);
    } else {
      ierr = PetscTypeCompare((PetscObject)A,MATMPIDENSE,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = MatGetColumnNorms_MPIDense(A,type,norms);CHKERRQ(ierr);
      } else {
        ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
        if (flg) {
          ierr = MatGetColumnNorms_MPIAIJ(A,type,norms);CHKERRQ(ierr);
        } else SETERRQ(PETSC_ERR_SUP,"Not coded for this matrix type");
      }
    } 
  }
  PetscFunctionReturn(0);
}
  
