static char help[] = "Test MatCreateNest with block sizes.\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                    A, B, C, mats[2];
  ISLocalToGlobalMapping cmap, rmap;
  const PetscInt         indices[1] = {0};
  PetscMPIInt            size;
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Only coded for one process");
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 2));
  CHKERRQ(MatSetBlockSizes(A, 1, 2));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 2, 1, indices,PETSC_COPY_VALUES, &cmap));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 1, indices,PETSC_COPY_VALUES, &rmap));
  CHKERRQ(MatSetLocalToGlobalMapping(A, rmap, cmap));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &B));
  CHKERRQ(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  CHKERRQ(MatSetBlockSizes(A, 1, 1));
  CHKERRQ(MatSetType(B,MATAIJ));
  CHKERRQ(MatSetLocalToGlobalMapping(B, rmap, rmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&rmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&cmap));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetUp(B));
  mats[0] = A;
  mats[1] = B;
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD, 1, NULL, 2, NULL, mats,&C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(C, NULL));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:

TEST*/
