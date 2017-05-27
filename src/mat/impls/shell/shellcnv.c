
#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

PETSC_INTERN PetscErrorCode MatConvert_Shell(Mat oldmat, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat;
  Vec            in,out;
  PetscErrorCode ierr;
  PetscInt       i,M,m,*rows,start,end;
  MPI_Comm       comm;
  PetscScalar    *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)oldmat,&comm);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(oldmat,&start,&end);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,end-start,PETSC_DECIDE,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = PetscMalloc1(m+1,&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) rows[i] = start + i;

  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,M,M,M);CHKERRQ(ierr);
  ierr = MatSetType(mat,newtype);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(mat,oldmat,oldmat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);

  for (i=0; i<M; i++) {
    ierr = VecSet(in,zero);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = MatMult(oldmat,in,out);CHKERRQ(ierr);

    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(&in);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(oldmat,&mat);CHKERRQ(ierr);
  } else {
    *newmat = mat;
  }
  PetscFunctionReturn(0);
}



