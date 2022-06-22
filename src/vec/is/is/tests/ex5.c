
static char help[] = "Tests ISLocalToGlobalMappingGetInfo.()\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  PetscMPIInt            size,rank;
  PetscInt               nlocal,local[5],nneigh,*neigh,**ineigh,*numneigh;
  ISLocalToGlobalMapping mapping;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 3,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with three processors");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    nlocal = 4; local[0] = 0; local[1] = 3; local[2] = 2; local[3] = 1;
  } else if (rank == 1) {
    nlocal = 4; local[0] = 3; local[1] = 5; local[2] = 4; local[3] = 2;
  } else {
    nlocal = 4; local[0] = 7; local[1] = 6; local[2] = 5; local[3] = 3;
  }
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,nlocal,local,PETSC_COPY_VALUES,&mapping));
  PetscCall(ISLocalToGlobalMappingGetInfo(mapping,&nneigh,&neigh,&numneigh,&ineigh));
  PetscCall(ISLocalToGlobalMappingRestoreInfo(mapping,&nneigh,&neigh,&numneigh,&ineigh));
  PetscCall(ISLocalToGlobalMappingDestroy(&mapping));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      output_file: output/ex1_1.out

TEST*/
