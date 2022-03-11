
static char help[] = "Tests PetscGlobalMinMax\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       li[2],gi[2] = {-1, -1};
  PetscReal      lr[2],gr[2] = {-1., -1.};

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  li[0] = 4 + rank;
  li[1] = -3 + size - rank;
  ierr = PetscGlobalMinMaxInt(PETSC_COMM_WORLD,li,gi);CHKERRQ(ierr);
  if (gi[0] != 4 || gi[1] != -3+size) { ierr = PetscPrintf(PETSC_COMM_SELF,"1) Error MIN/MAX %" PetscInt_FMT " %" PetscInt_FMT "\n",gi[0],gi[1]);CHKERRQ(ierr); }
  ierr = PetscGlobalMinMaxInt(PETSC_COMM_WORLD,li,li);CHKERRQ(ierr);
  if (li[0] != gi[0] || li[1] != gi[1]) { ierr = PetscPrintf(PETSC_COMM_SELF,"2) Error MIN/MAX %" PetscInt_FMT " %" PetscInt_FMT "\n",li[0],li[1]);CHKERRQ(ierr); }

  if (rank == 0) {
    li[0] = PETSC_MAX_INT;
    li[1] = PETSC_MIN_INT;
  } else if (rank == 1) {
    li[0] = PETSC_MIN_INT;
    li[1] = PETSC_MAX_INT;
  }

  ierr = PetscGlobalMinMaxInt(PETSC_COMM_WORLD,li,gi);CHKERRQ(ierr);
  if (gi[0] > li[0] || gi[1] < li[1]) { ierr = PetscPrintf(PETSC_COMM_SELF,"3) Error MIN/MAX %" PetscInt_FMT " %" PetscInt_FMT "\n",gi[0],gi[1]);CHKERRQ(ierr); }

  lr[0] = 4.0 + rank;
  lr[1] = -3.0 + size - rank;
  ierr = PetscGlobalMinMaxReal(PETSC_COMM_WORLD,lr,gr);CHKERRQ(ierr);
  if (gr[0] != 4.0 || gr[1] != -3.0+size) { ierr = PetscPrintf(PETSC_COMM_SELF,"4) Error MIN/MAX %g %g\n",(double)gr[0],(double)gr[1]);CHKERRQ(ierr); }
  ierr = PetscGlobalMinMaxReal(PETSC_COMM_WORLD,lr,lr);CHKERRQ(ierr);
  if (lr[0] != gr[0] || lr[1] != gr[1]) { ierr = PetscPrintf(PETSC_COMM_SELF,"5) Error MIN/MAX %g %g\n",(double)lr[0],(double)li[1]);CHKERRQ(ierr); }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     output_file: output/ex58_1.out

   test:
     suffix: 2
     output_file: output/ex58_1.out
     nsize: 2

TEST*/
