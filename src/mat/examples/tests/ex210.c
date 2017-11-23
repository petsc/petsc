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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Only coded for one process");
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 2);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(A, 1, 2);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 2, 1, indices,PETSC_COPY_VALUES, &cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 1, indices,PETSC_COPY_VALUES, &rmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A, rmap, cmap);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &B);CHKERRQ(ierr);
  ierr = MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 1, 1);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(A, 1, 1);CHKERRQ(ierr);
  ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B, rmap, rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  mats[0] = A;
  mats[1] = B;
  ierr = MatCreateNest(PETSC_COMM_WORLD, 1, NULL, 2, NULL, mats,&C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C, NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:

TEST*/
