static char help[] = "Tests ISRenumber.\n\n";

#include <petscis.h>

PetscErrorCode TestRenumber(IS is, IS mult)
{
  IS             nis;
  PetscInt       N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PetscObjectComm((PetscObject)is),"\n-----------------\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)is),"\nInitial\n");CHKERRQ(ierr);
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  if (mult) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)is),"\nMult\n");CHKERRQ(ierr);
    ierr = ISView(mult,NULL);CHKERRQ(ierr);
  }
  ierr = ISRenumber(is,mult,&N,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)is),"\nRenumbered, unique entries %" PetscInt_FMT "\n",N);CHKERRQ(ierr);
  ierr = ISRenumber(is,mult,NULL,&nis);CHKERRQ(ierr);
  ierr = ISView(nis,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&nis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  IS              is;
  PetscErrorCode  ierr;
  PetscMPIInt     size, rank;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  for (PetscInt c = 0; c < 3; c++) {
    IS mult = NULL;

    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,0,&is);CHKERRQ(ierr);
    if (c) {
      PetscInt n;
      ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_WORLD,n,c-2,0,&mult);CHKERRQ(ierr);
    }
    ierr = TestRenumber(is,mult);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = ISDestroy(&mult);CHKERRQ(ierr);

    ierr = ISCreateStride(PETSC_COMM_WORLD,2,-rank-1,-4,&is);CHKERRQ(ierr);
    if (c) {
      PetscInt n;
      ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_WORLD,n,c-2,0,&mult);CHKERRQ(ierr);
    }
    ierr = TestRenumber(is,mult);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = ISDestroy(&mult);CHKERRQ(ierr);

    ierr = ISCreateStride(PETSC_COMM_WORLD,10,4+rank,2,&is);CHKERRQ(ierr);
    if (c) {
      PetscInt n;
      ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_WORLD,n,c-2,1,&mult);CHKERRQ(ierr);
    }
    ierr = TestRenumber(is,mult);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = ISDestroy(&mult);CHKERRQ(ierr);

    ierr = ISCreateStride(PETSC_COMM_WORLD,10,-rank-1,2,&is);CHKERRQ(ierr);
    if (c) {
      PetscInt n;
      ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_WORLD,n,c-2,1,&mult);CHKERRQ(ierr);
    }
    ierr = TestRenumber(is,mult);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = ISDestroy(&mult);CHKERRQ(ierr);
  }
  /* Finalize */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1
    nsize: {{1 2}separate output}

TEST*/
