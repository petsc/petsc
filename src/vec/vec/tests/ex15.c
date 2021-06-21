
static char help[] = "Tests VecSetValuesBlocked() on sequential vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 9,bs = 3,indices[2],i;
  PetscScalar    values[6];
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with one processor");

  /* create vector */
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRQ(ierr);
  ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);

  for (i=0; i<6; i++) values[i] = 4.0*i;
  indices[0] = 0;
  indices[1] = 2;

  ierr = VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
      Resulting vector should be 0 4 8  0 0 0 12 16 20
  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* test insertion with negative indices */
  ierr = VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0; i<6; i++) values[i] = -4.0*i;
  indices[0] = -1;
  indices[1] = 2;

  ierr = VecSetValuesBlocked(x,2,indices,values,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
      Resulting vector should be 0 4 8  0 0 0 0 0 0
  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
