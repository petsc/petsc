static char help[] = "Tests VecPlaceArray().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  PetscInt          n=5,bs;
  PetscBool         cuda;
  Vec               x,x1,x2;
  const PetscScalar *px;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create vector of length 2*n */
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,2*n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* create two vectors of length n without array */
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQCUDA,&cuda);CHKERRQ(ierr);
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
  if (cuda) {
    ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1);CHKERRQ(ierr);
    ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x1);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n,NULL,&x2);CHKERRQ(ierr);
  }

  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+n);CHKERRQ(ierr);
  ierr = VecSet(x1,1.0);CHKERRQ(ierr);
  ierr = VecSet(x2,2.0);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);

  ierr = VecView(x,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     testset:
       output_file: output/ex60_1.out
       test:
         suffix: 1
       test:
         suffix: 1_cuda
         args: -vec_type cuda
         filter: sed -e 's/seqcuda/seq/'

TEST*/
