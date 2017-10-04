#include <petscksp.h>


// Mat Q:
// 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 
// 1.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 
// 1.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 
// 1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00

// 1.0000000000000000e+00 2.0000000000000000e+00 0.0000000000000000e+00 
// 1.0000000000000000e+00 2.0000000000000000e+00 1.0000000000000000e+00 
// 1.0000000000000000e+00 3.0000000000000000e+00 0.0000000000000000e+00 
// 1.0000000000000000e+00 3.0000000000000000e+00 1.0000000000000000e+00 

PetscErrorCode fill(Mat m, Vec v) {
  PetscInt       idxn[3] = {0, 1, 2};
  PetscInt       localRows = 0;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);CHKERRQ(ierr);  

  if (rank == 1 || rank == 2) localRows = 4;
  if (size == 1) localRows = 8;
  ierr = MatSetSizes(m, localRows, PETSC_DECIDE, PETSC_DECIDE, 3);CHKERRQ(ierr);
  ierr = VecSetSizes(v, localRows, PETSC_DECIDE);CHKERRQ(ierr);

  ierr = MatSetFromOptions(m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = MatSetUp(m);CHKERRQ(ierr);

  if (size == 1) {
    localRows = 8;


    PetscInt idxm1[4] = {0, 1, 2, 3};
    PetscScalar values1[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};

    ierr = MatSetValues(m, 4, idxm1, 3, idxn, values1, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 0, 1.1, INSERT_VALUES); VecSetValue(v, 1, 2.5, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 2, 3, INSERT_VALUES); VecSetValue(v, 3, 4, INSERT_VALUES);CHKERRQ(ierr);

    PetscInt idxm2[4] = {4, 5, 6, 7};
    PetscScalar values2[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    ierr = MatSetValues(m, 4, idxm2, 3, idxn, values2, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 4, 5, INSERT_VALUES); VecSetValue(v, 5, 6, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 6, 7, INSERT_VALUES); VecSetValue(v, 7, 8, INSERT_VALUES);CHKERRQ(ierr);

  }else if (rank == 1) {
    PetscInt idxm[4] = {0, 1, 2, 3};
    PetscScalar values[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};

    ierr = MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 0, 1.1, INSERT_VALUES); VecSetValue(v, 1, 2.5, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 2, 3, INSERT_VALUES); VecSetValue(v, 3, 4, INSERT_VALUES);CHKERRQ(ierr);
  }
  else if (rank == 2) {
    PetscInt idxm[4] = {4, 5, 6, 7};
    PetscScalar values[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    ierr = MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 4, 5, INSERT_VALUES); VecSetValue(v, 5, 6, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 6, 7, INSERT_VALUES); VecSetValue(v, 7, 8, INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc, char** argv)
{
  Mat            Q;
  Vec            v, a;
  KSP            QRsolver;
  PC             pc;
  PetscViewer    viewerQ, viewerV;
  PetscBool      load;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &v);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &Q);CHKERRQ(ierr);
  ierr = MatSetType(Q, MATDENSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrixQ", FILE_MODE_READ, &viewerQ);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "in", FILE_MODE_READ, &viewerV);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL, NULL, "-load", &load);CHKERRQ(ierr);
  if (load) {
    ierr = MatLoad(Q, viewerQ);CHKERRQ(ierr);
    ierr = VecLoad(v, viewerV);CHKERRQ(ierr);
  } else {
    ierr = fill(Q, v);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewerQ);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerV);CHKERRQ(ierr);

  ierr = MatCreateVecs(Q, &a, NULL);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &QRsolver);CHKERRQ(ierr);
  ierr = KSPGetPC(QRsolver, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = KSPSetType(QRsolver, KSPLSQR);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(QRsolver);CHKERRQ(ierr);
  ierr = KSPSetOperators(QRsolver, Q, Q);CHKERRQ(ierr);
  ierr = MatViewFromOptions(Q, NULL, "-sys_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(a, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = KSPSolve(QRsolver, v, a);CHKERRQ(ierr);
  ierr = KSPDestroy(&QRsolver);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
