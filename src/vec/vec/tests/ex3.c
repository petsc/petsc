
static char help[] = "Tests parallel vector assembly.  Input arguments are\n\
  -n <length> : local vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscInt       n   = 5,idx;
  PetscScalar    one = 1.0,two = 2.0,three = 3.0;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  if (n < 5) n = 5;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(size < 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with at least two processors");

  /* create two vector */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(y));
  CHKERRQ(VecSet(x,one));
  CHKERRQ(VecSet(y,two));

  if (rank == 1) {
    idx = 2; CHKERRQ(VecSetValues(y,1,&idx,&three,INSERT_VALUES));
    idx = 0; CHKERRQ(VecSetValues(y,1,&idx,&two,INSERT_VALUES));
    idx = 0; CHKERRQ(VecSetValues(y,1,&idx,&one,INSERT_VALUES));
  } else {
    idx = 7; CHKERRQ(VecSetValues(y,1,&idx,&three,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));

  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2

     test:
       suffix: 2
       nsize: 2
       args: -vec_view ascii::ascii_info

TEST*/
