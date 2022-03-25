
static char help[] = "Tests parallel vector assembly.  Input arguments are\n\
  -n <length> : local vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       n   = 5,idx;
  PetscScalar    one = 1.0,two = 2.0,three = 3.0;
  Vec            x,y;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  if (n < 5) n = 5;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(size < 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with at least two processors");

  /* create two vector */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,n,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecSet(x,one));
  PetscCall(VecSet(y,two));

  if (rank == 1) {
    idx = 2; PetscCall(VecSetValues(y,1,&idx,&three,INSERT_VALUES));
    idx = 0; PetscCall(VecSetValues(y,1,&idx,&two,INSERT_VALUES));
    idx = 0; PetscCall(VecSetValues(y,1,&idx,&one,INSERT_VALUES));
  } else {
    idx = 7; PetscCall(VecSetValues(y,1,&idx,&three,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

     test:
       suffix: 2
       nsize: 2
       args: -vec_view ascii::ascii_info

TEST*/
