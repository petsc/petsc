
#include "src/mat/matimpl.h"        /*I "petscmat.h" I*/
#include "vecimpl.h"  
  
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_Shell"
PetscErrorCode MatConvert_Shell(Mat oldmat,const MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat;
  Vec            in,out;
  PetscErrorCode ierr;
  PetscInt       i,M,m,*rows,start,end;
  MPI_Comm       comm;
  PetscScalar    *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  comm = oldmat->comm;

  ierr = MatGetOwnershipRange(oldmat,&start,&end);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,end-start,PETSC_DECIDE,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  ierr = MatCreate(comm,m,M,M,M,&mat);CHKERRQ(ierr);
  ierr = MatSetType(mat,newtype);CHKERRQ(ierr);
  ierr = MatSetBlockSize(mat,oldmat->bs);CHKERRQ(ierr);

  for (i=0; i<M; i++) {
    ierr = VecSet(&zero,in);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = MatMult(oldmat,in,out);CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatDestroy(oldmat);CHKERRQ(ierr);
  }
  *newmat = mat;
  PetscFunctionReturn(0);
}



