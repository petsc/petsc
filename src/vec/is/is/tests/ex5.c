
static char help[] = "Tests ISLocalToGlobalMappingGetInfo.()\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscMPIInt            size,rank;
  PetscInt               nlocal,local[5],nneigh,*neigh,**ineigh,*numneigh;
  ISLocalToGlobalMapping mapping;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 3,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with three processors");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    nlocal = 4; local[0] = 0; local[1] = 3; local[2] = 2; local[3] = 1;
  } else if (rank == 1) {
    nlocal = 4; local[0] = 3; local[1] = 5; local[2] = 4; local[3] = 2;
  } else {
    nlocal = 4; local[0] = 7; local[1] = 6; local[2] = 5; local[3] = 3;
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,nlocal,local,PETSC_COPY_VALUES,&mapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetInfo(mapping,&nneigh,&neigh,&numneigh,&ineigh);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreInfo(mapping,&nneigh,&neigh,&numneigh,&ineigh);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      output_file: output/ex1_1.out

TEST*/
