#define PETSCMAT_DLL

#include "mpisbaij.h" /*I "petscmat.h" I*/
#include "../src/mat/impls/aij/mpi/mpiaij.h"
#include "private/matimpl.h"
#include "petscmat.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIAIJ_MPISBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_MPISBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode     ierr;
  Mat                M;
  PetscInt           *d_nnz,*o_nnz;
  PetscInt           i,j,k,nz;
  PetscInt           m,n,lm,ln;
  PetscInt           rstart,rend;
  const PetscScalar  *vwork;
  const PetscInt     *cwork;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&lm,&ln);CHKERRQ(ierr);
  ierr = PetscMalloc2(lm*sizeof(PetscInt),PetscInt,&d_nnz,lm*sizeof(PetscInt),PetscInt,&o_nnz);CHKERRQ(ierr);

  k = 0;
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for(i=rstart;i<rend;i++){
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    d_nnz[k] = o_nnz[k] = 0;
    for(j=0;j<nz;j++){
      if(cwork[j] >= rstart && cwork[j] <= rend-1){ /* diagonal portion */
	if(cwork[j] >= i) d_nnz[k] += 1;
      } else {
	/* insert values only in upper triangular portion */
	if(cwork[j] >= rend) o_nnz[k] += 1;  /* off-diagonal portion */
      }
    }
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    k++;
  }

  ierr = MatCreate(((PetscObject)A)->comm,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,lm,ln,m,n);CHKERRQ(ierr);
  ierr = MatSetType(M,newtype);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(M,1,0,d_nnz,0,o_nnz);CHKERRQ(ierr);

  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

  for(i=rstart;i<rend;i++){
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    j = 0;
    while (cwork[j] < i){ j++; nz--;}
    ierr = MatSetValues(M,1,&i,nz,cwork+j,vwork+j,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    k++;
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A->hermitian){
    ierr = MatSetOption(M,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  }

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,M);CHKERRQ(ierr);
  } else {
    *newmat = M;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
