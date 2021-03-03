
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  if (n < 5) n = 5;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  if (size < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with at least two processors");

  /* create two vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);

  if (rank == 1) {
    idx = 2; ierr = VecSetValues(y,1,&idx,&three,INSERT_VALUES);CHKERRQ(ierr);
    idx = 0; ierr = VecSetValues(y,1,&idx,&two,INSERT_VALUES);CHKERRQ(ierr);
    idx = 0; ierr = VecSetValues(y,1,&idx,&one,INSERT_VALUES);CHKERRQ(ierr);
  } else {
    idx = 7; ierr = VecSetValues(y,1,&idx,&three,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

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
