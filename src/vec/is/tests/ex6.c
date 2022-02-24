static char help[] = "Tests ISRenumber.\n\n";

#include <petscis.h>

PetscErrorCode TestRenumber(IS is, IS mult)
{
  IS       nis;
  PetscInt N;

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)is),"\n-----------------\n"));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)is),"\nInitial\n"));
  CHKERRQ(ISView(is,NULL));
  if (mult) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)is),"\nMult\n"));
    CHKERRQ(ISView(mult,NULL));
  }
  CHKERRQ(ISRenumber(is,mult,&N,NULL));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)is),"\nRenumbered, unique entries %" PetscInt_FMT "\n",N));
  CHKERRQ(ISRenumber(is,mult,NULL,&nis));
  CHKERRQ(ISView(nis,NULL));
  CHKERRQ(ISDestroy(&nis));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  IS              is;
  PetscErrorCode  ierr;
  PetscMPIInt     size, rank;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  for (PetscInt c = 0; c < 3; c++) {
    IS mult = NULL;

    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,0,0,0,&is));
    if (c) {
      PetscInt n;
      CHKERRQ(ISGetLocalSize(is,&n));
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,c-2,0,&mult));
    }
    CHKERRQ(TestRenumber(is,mult));
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(ISDestroy(&mult));

    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,2,-rank-1,-4,&is));
    if (c) {
      PetscInt n;
      CHKERRQ(ISGetLocalSize(is,&n));
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,c-2,0,&mult));
    }
    CHKERRQ(TestRenumber(is,mult));
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(ISDestroy(&mult));

    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,10,4+rank,2,&is));
    if (c) {
      PetscInt n;
      CHKERRQ(ISGetLocalSize(is,&n));
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,c-2,1,&mult));
    }
    CHKERRQ(TestRenumber(is,mult));
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(ISDestroy(&mult));

    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,10,-rank-1,2,&is));
    if (c) {
      PetscInt n;
      CHKERRQ(ISGetLocalSize(is,&n));
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,c-2,1,&mult));
    }
    CHKERRQ(TestRenumber(is,mult));
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(ISDestroy(&mult));
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
