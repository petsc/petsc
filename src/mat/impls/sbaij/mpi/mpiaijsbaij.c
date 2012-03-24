
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc-private/matimpl.h>
#include <petscmat.h>

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIAIJ_MPISBAIJ"
PetscErrorCode  MatConvert_MPIAIJ_MPISBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode     ierr;
  Mat                M;
  Mat_MPIAIJ         *mpimat = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ         *Aa = (Mat_SeqAIJ*)mpimat->A->data,*Ba = (Mat_SeqAIJ*)mpimat->B->data; 
  PetscInt           *d_nnz,*o_nnz;
  PetscInt           i,j,nz;
  PetscInt           m,n,lm,ln;
  PetscInt           rstart,rend;
  const PetscScalar  *vwork;
  const PetscInt     *cwork;

  PetscFunctionBegin;
  if (!A->symmetric) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_USER,"Matrix must be symmetric. Call MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE)");
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&lm,&ln);CHKERRQ(ierr);
  ierr = PetscMalloc2(lm,PetscInt,&d_nnz,lm,PetscInt,&o_nnz);CHKERRQ(ierr);

  ierr = MatMarkDiagonal_SeqAIJ(mpimat->A);CHKERRQ(ierr);
  for(i=0;i<lm;i++){
    d_nnz[i] = Aa->i[i+1] - Aa->diag[i];
    o_nnz[i] = Ba->i[i+1] - Ba->i[i];
  }

  ierr = MatCreate(((PetscObject)A)->comm,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,lm,ln,m,n);CHKERRQ(ierr);
  ierr = MatSetType(M,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(M,1,0,d_nnz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(M,1,0,d_nnz,0,o_nnz);CHKERRQ(ierr);

  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for(i=rstart;i<rend;i++){
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    j = 0;
    while (cwork[j] < i){ j++; nz--;}
    ierr = MatSetValues(M,1,&i,nz,cwork+j,vwork+j,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,M);CHKERRQ(ierr);
  } else {
    *newmat = M;
  }
  PetscFunctionReturn(0);
}
/* contributed by Dahai Guo <dhguo@ncsa.uiuc.edu> April 2011 */
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIBAIJ_MPISBAIJ"
PetscErrorCode MatConvert_MPIBAIJ_MPISBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode     ierr;
  Mat                M;
  Mat_MPIBAIJ        *mpimat = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ        *Aa = (Mat_SeqBAIJ*)mpimat->A->data,*Ba = (Mat_SeqBAIJ*)mpimat->B->data; 
  PetscInt           *d_nnz,*o_nnz;
  PetscInt           i,j,nz;
  PetscInt           m,n,lm,ln;
  PetscInt           rstart,rend;
  const PetscScalar  *vwork;
  const PetscInt     *cwork;
  PetscInt           bs = A->rmap->bs;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&lm,&ln);CHKERRQ(ierr);
  ierr = PetscMalloc2(lm/bs,PetscInt,&d_nnz,lm/bs,PetscInt,&o_nnz);CHKERRQ(ierr);
  
  ierr = MatMarkDiagonal_SeqBAIJ(mpimat->A);CHKERRQ(ierr);
  for(i=0;i<lm/bs;i++){
    d_nnz[i] = Aa->i[i+1] - Aa->diag[i]; 
    o_nnz[i] = Ba->i[i+1] - Ba->i[i];
  }

  ierr = MatCreate(((PetscObject)A)->comm,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,lm,ln,m,n);CHKERRQ(ierr);
  ierr = MatSetType(M,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(M,bs,0,d_nnz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(M,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);

  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr); 
  ierr = MatSetOption(M,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr); 
  for(i=rstart;i<rend;i++){
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    j = 0;
    ierr = MatSetValues(M,1,&i,nz,cwork+j,vwork+j,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,M);CHKERRQ(ierr);
  } else {
    *newmat = M;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
