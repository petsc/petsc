
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/matimpl.h>
#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPISBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode    ierr;
  Mat               M;
  Mat_MPIAIJ        *mpimat = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ        *Aa     = (Mat_SeqAIJ*)mpimat->A->data,*Ba = (Mat_SeqAIJ*)mpimat->B->data;
  PetscInt          *d_nnz,*o_nnz;
  PetscInt          i,j,nz;
  PetscInt          m,n,lm,ln;
  PetscInt          rstart,rend,bs=PetscAbs(A->rmap->bs);
  const PetscScalar *vwork;
  const PetscInt    *cwork;

  PetscFunctionBegin;
  PetscAssertFalse(!A->symmetric && !A->hermitian,PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"Matrix must be symmetric or hermitian. Call MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE) or MatSetOption(mat,MAT_HERMITIAN,PETSC_TRUE)");
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&lm,&ln);CHKERRQ(ierr);
    ierr = PetscMalloc2(lm/bs,&d_nnz,lm/bs,&o_nnz);CHKERRQ(ierr);

    ierr = MatMarkDiagonal_SeqAIJ(mpimat->A);CHKERRQ(ierr);
    for (i=0; i<lm/bs; i++) {
      if (Aa->i[i*bs+1] == Aa->diag[i*bs]) { /* misses diagonal entry */
        d_nnz[i] = (Aa->i[i*bs+1] - Aa->i[i*bs])/bs;
      } else {
        d_nnz[i] = (Aa->i[i*bs+1] - Aa->diag[i*bs])/bs;
      }
      o_nnz[i] = (Ba->i[i*bs+1] - Ba->i[i*bs])/bs;
    }

    ierr = MatCreate(PetscObjectComm((PetscObject)A),&M);CHKERRQ(ierr);
    ierr = MatSetSizes(M,lm,ln,m,n);CHKERRQ(ierr);
    ierr = MatSetType(M,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(M,bs,0,d_nnz);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(M,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
    ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  } else M = *newmat;

  if (bs == 1) {
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
      if (nz) {
        j = 0;
        while (cwork[j] < i) {
          j++; nz--;
        }
        ierr = MatSetValues(M,1,&i,nz,cwork+j,vwork+j,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatSetOption(M,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
    /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
    /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
    ierr = MatConvert_Basic(A,newtype,MAT_REUSE_MATRIX,&M);CHKERRQ(ierr);
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&M);CHKERRQ(ierr);
  } else *newmat = M;
  PetscFunctionReturn(0);
}
/* contributed by Dahai Guo <dhguo@ncsa.uiuc.edu> April 2011 */
PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPISBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode    ierr;
  Mat               M;
  Mat_MPIBAIJ       *mpimat = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ       *Aa     = (Mat_SeqBAIJ*)mpimat->A->data,*Ba = (Mat_SeqBAIJ*)mpimat->B->data;
  PetscInt          *d_nnz,*o_nnz;
  PetscInt          i,nz;
  PetscInt          m,n,lm,ln;
  PetscInt          rstart,rend;
  const PetscScalar *vwork;
  const PetscInt    *cwork;
  PetscInt          bs = A->rmap->bs;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&lm,&ln);CHKERRQ(ierr);
    ierr = PetscMalloc2(lm/bs,&d_nnz,lm/bs,&o_nnz);CHKERRQ(ierr);

    ierr = MatMarkDiagonal_SeqBAIJ(mpimat->A);CHKERRQ(ierr);
    for (i=0; i<lm/bs; i++) {
      d_nnz[i] = Aa->i[i+1] - Aa->diag[i];
      o_nnz[i] = Ba->i[i+1] - Ba->i[i];
    }

    ierr = MatCreate(PetscObjectComm((PetscObject)A),&M);CHKERRQ(ierr);
    ierr = MatSetSizes(M,lm,ln,m,n);CHKERRQ(ierr);
    ierr = MatSetType(M,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(M,bs,0,d_nnz);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(M,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);

    ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  } else M = *newmat;

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatSetOption(M,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&M);CHKERRQ(ierr);
  } else *newmat = M;
  PetscFunctionReturn(0);
}
