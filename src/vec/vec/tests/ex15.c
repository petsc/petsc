
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with one processor");

  /* create vector */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&x));
  CHKERRQ(VecSetSizes(x,n,n));
  CHKERRQ(VecSetBlockSize(x,bs));
  CHKERRQ(VecSetType(x,VECSEQ));

  for (i=0; i<6; i++) values[i] = 4.0*i;
  indices[0] = 0;
  indices[1] = 2;

  CHKERRQ(VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /*
      Resulting vector should be 0 4 8  0 0 0 12 16 20
  */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* test insertion with negative indices */
  CHKERRQ(VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  for (i=0; i<6; i++) values[i] = -4.0*i;
  indices[0] = -1;
  indices[1] = 2;

  CHKERRQ(VecSetValuesBlocked(x,2,indices,values,ADD_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /*
      Resulting vector should be 0 4 8  0 0 0 0 0 0
  */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&x));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
