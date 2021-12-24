
static char help[] = "Test MatAXPY()\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,C1,C2,CU;
  PetscScalar    v;
  PetscInt       Ii,J,Istart,Iend;
  PetscErrorCode ierr;
  PetscInt       i,j,m = 3,n;
  PetscMPIInt    size;
  PetscBool      mat_nonsymmetric = PETSC_FALSE,flg;
  MatInfo        info;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = 2*size;

  /* Set flag if we are doing a nonsymmetric problem; the default is symmetric. */
  ierr = PetscOptionsGetBool(NULL,NULL,"-mat_nonsym",&mat_nonsymmetric,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C,5,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,5,NULL,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  /* Make the matrix nonsymmetric if desired */
  if (mat_nonsymmetric) {
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.5; i = Ii/n;
      if (i>1) {J = Ii-n-1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    }
  } else {
    ierr = MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)C,"C");CHKERRQ(ierr);
  ierr = MatViewFromOptions(C,NULL,"-view");CHKERRQ(ierr);

  /* C1 = 2.0*C1 + C, C1 is anti-diagonal and has different non-zeros than C */
  ierr = MatCreate(PETSC_COMM_WORLD,&C1);CHKERRQ(ierr);
  ierr = MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C1);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C1,1,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C1,1,NULL,1,NULL);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = 1.0;
    i = m*n - Ii -1;
    j = Ii;
    ierr = MatSetValues(C1,1,&i,1,&j,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)C1,"C1");CHKERRQ(ierr);
  ierr = MatViewFromOptions(C1,NULL,"-view");CHKERRQ(ierr);
  ierr = MatDuplicate(C1,MAT_COPY_VALUES,&CU);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C1,2.0,C,DIFFERENT_NONZERO_PATTERN)...\n");CHKERRQ(ierr);
  ierr = MatAXPY(C1,2.0,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(CU,2.0,C,UNKNOWN_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatGetInfo(C1,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," C1: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded);CHKERRQ(ierr);
  ierr = MatViewFromOptions(C1,NULL,"-view");CHKERRQ(ierr);
  ierr = MatMultEqual(CU,C1,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error UNKNOWN_NONZERO_PATTERN (supposedly DIFFERENT_NONZERO_PATTERN)\n");CHKERRQ(ierr);
    ierr = MatViewFromOptions(CU,NULL,"-view");CHKERRQ(ierr);
  }
  ierr = MatDestroy(&CU);CHKERRQ(ierr);

  /* Secondly, compute C1 = 2.0*C2 + C1, C2 has non-zero pattern of C */
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C2);CHKERRQ(ierr);
  ierr = MatDuplicate(C1,MAT_COPY_VALUES,&CU);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) {
    v    = 1.0;
    ierr = MatSetValues(C2,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)C2,"C2");CHKERRQ(ierr);
  ierr = MatViewFromOptions(C2,NULL,"-view");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C1,2.0,C2,SUBSET_NONZERO_PATTERN)...\n");CHKERRQ(ierr);
  ierr = MatAXPY(C1,2.0,C2,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(CU,2.0,C2,UNKNOWN_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatGetInfo(C1,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," C1: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded);CHKERRQ(ierr);
  ierr = MatViewFromOptions(C1,NULL,"-view");CHKERRQ(ierr);
  ierr = MatMultEqual(CU,C1,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error UNKNOWN_NONZERO_PATTERN (supposedly SUBSET_NONZERO_PATTERN)\n");CHKERRQ(ierr);
    ierr = MatViewFromOptions(CU,NULL,"-view");CHKERRQ(ierr);
  }
  ierr = MatDestroy(&CU);CHKERRQ(ierr);

  /* Test SAME_NONZERO_PATTERN computing C2 = C2 + 2.0 * C */
  ierr = MatDuplicate(C2,MAT_COPY_VALUES,&CU);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C2,2.0,C,SAME_NONZERO_PATTERN)...\n");CHKERRQ(ierr);
  ierr = MatAXPY(C2,2.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(CU,2.0,C,UNKNOWN_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatGetInfo(C2,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," C2: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded);CHKERRQ(ierr);
  ierr = MatViewFromOptions(C2,NULL,"-view");CHKERRQ(ierr);
  ierr = MatMultEqual(CU,C2,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error UNKNOWN_NONZERO_PATTERN (supposedly SUBSET_NONZERO_PATTERN)\n");CHKERRQ(ierr);
    ierr = MatViewFromOptions(CU,NULL,"-view");CHKERRQ(ierr);
  }
  ierr = MatDestroy(&CU);CHKERRQ(ierr);

  ierr = MatDestroy(&C1);CHKERRQ(ierr);
  ierr = MatDestroy(&C2);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: 1
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view
     diff_args: -j

   test:
     output_file: output/ex132_1.out
     requires: cuda
     suffix: 1_cuda
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijcusparse
     diff_args: -j

   test:
     output_file: output/ex132_1.out
     requires: kokkos_kernels
     suffix: 1_kokkos
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijkokkos
     diff_args: -j

   test:
     suffix: 2
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_nonsym
     diff_args: -j

   test:
     output_file: output/ex132_2.out
     requires: cuda
     suffix: 2_cuda
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijcusparse -mat_nonsym
     diff_args: -j

   test:
     output_file: output/ex132_2.out
     requires: kokkos_kernels
     suffix: 2_kokkos
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijkokkos -mat_nonsym
     diff_args: -j

   test:
     nsize: 2
     suffix: 1_par
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view
     diff_args: -j

   test:
     nsize: 2
     output_file: output/ex132_1_par.out
     requires: cuda
     suffix: 1_par_cuda
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijcusparse
     diff_args: -j

   test:
     nsize: 2
     output_file: output/ex132_1_par.out
     requires: !sycl kokkos_kernels
     suffix: 1_par_kokkos
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijkokkos
     diff_args: -j

   test:
     nsize: 2
     suffix: 2_par
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_nonsym
     diff_args: -j

   test:
     nsize: 2
     output_file: output/ex132_2_par.out
     requires: cuda
     suffix: 2_par_cuda
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijcusparse -mat_nonsym
     diff_args: -j

   test:
     nsize: 2
     output_file: output/ex132_2_par.out
     requires: !sycl kokkos_kernels
     suffix: 2_par_kokkos
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijkokkos -mat_nonsym -use_gpu_aware_mpi {{0 1}}
     diff_args: -j

TEST*/
