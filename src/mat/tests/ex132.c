
static char help[] = "Test MatAXPY()\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,C1,C2,CU;
  PetscScalar    v;
  PetscInt       Ii,J,Istart,Iend;
  PetscInt       i,j,m = 3,n;
  PetscMPIInt    size;
  PetscBool      mat_nonsymmetric = PETSC_FALSE,flg;
  MatInfo        info;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* Set flag if we are doing a nonsymmetric problem; the default is symmetric. */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-mat_nonsym",&mat_nonsymmetric,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSeqAIJSetPreallocation(C,5,NULL));
  PetscCall(MatMPIAIJSetPreallocation(C,5,NULL,5,NULL));

  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }

  /* Make the matrix nonsymmetric if desired */
  if (mat_nonsymmetric) {
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.5; i = Ii/n;
      if (i>1) {J = Ii-n-1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES));}
    }
  } else {
    PetscCall(MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE));
    PetscCall(MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)C,"C"));
  PetscCall(MatViewFromOptions(C,NULL,"-view"));

  /* C1 = 2.0*C1 + C, C1 is anti-diagonal and has different non-zeros than C */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C1));
  PetscCall(MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C1));
  PetscCall(MatSeqAIJSetPreallocation(C1,1,NULL));
  PetscCall(MatMPIAIJSetPreallocation(C1,1,NULL,1,NULL));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = 1.0;
    i = m*n - Ii -1;
    j = Ii;
    PetscCall(MatSetValues(C1,1,&i,1,&j,&v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)C1,"C1"));
  PetscCall(MatViewFromOptions(C1,NULL,"-view"));
  PetscCall(MatDuplicate(C1,MAT_COPY_VALUES,&CU));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C1,2.0,C,DIFFERENT_NONZERO_PATTERN)...\n"));
  PetscCall(MatAXPY(C1,2.0,C,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAXPY(CU,2.0,C,UNKNOWN_NONZERO_PATTERN));
  PetscCall(MatGetInfo(C1,MAT_GLOBAL_SUM,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," C1: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded));
  PetscCall(MatViewFromOptions(C1,NULL,"-view"));
  PetscCall(MatMultEqual(CU,C1,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error UNKNOWN_NONZERO_PATTERN (supposedly DIFFERENT_NONZERO_PATTERN)\n"));
    PetscCall(MatViewFromOptions(CU,NULL,"-view"));
  }
  PetscCall(MatDestroy(&CU));

  /* Secondly, compute C1 = 2.0*C2 + C1, C2 has non-zero pattern of C */
  PetscCall(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C2));
  PetscCall(MatDuplicate(C1,MAT_COPY_VALUES,&CU));

  for (Ii=Istart; Ii<Iend; Ii++) {
    v    = 1.0;
    PetscCall(MatSetValues(C2,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C2,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C2,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)C2,"C2"));
  PetscCall(MatViewFromOptions(C2,NULL,"-view"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C1,2.0,C2,SUBSET_NONZERO_PATTERN)...\n"));
  PetscCall(MatAXPY(C1,2.0,C2,SUBSET_NONZERO_PATTERN));
  PetscCall(MatAXPY(CU,2.0,C2,UNKNOWN_NONZERO_PATTERN));
  PetscCall(MatGetInfo(C1,MAT_GLOBAL_SUM,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," C1: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded));
  PetscCall(MatViewFromOptions(C1,NULL,"-view"));
  PetscCall(MatMultEqual(CU,C1,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error UNKNOWN_NONZERO_PATTERN (supposedly SUBSET_NONZERO_PATTERN)\n"));
    PetscCall(MatViewFromOptions(CU,NULL,"-view"));
  }
  PetscCall(MatDestroy(&CU));

  /* Test SAME_NONZERO_PATTERN computing C2 = C2 + 2.0 * C */
  PetscCall(MatDuplicate(C2,MAT_COPY_VALUES,&CU));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C2,2.0,C,SAME_NONZERO_PATTERN)...\n"));
  PetscCall(MatAXPY(C2,2.0,C,SAME_NONZERO_PATTERN));
  PetscCall(MatAXPY(CU,2.0,C,UNKNOWN_NONZERO_PATTERN));
  PetscCall(MatGetInfo(C2,MAT_GLOBAL_SUM,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," C2: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded));
  PetscCall(MatViewFromOptions(C2,NULL,"-view"));
  PetscCall(MatMultEqual(CU,C2,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error UNKNOWN_NONZERO_PATTERN (supposedly SUBSET_NONZERO_PATTERN)\n"));
    PetscCall(MatViewFromOptions(CU,NULL,"-view"));
  }
  PetscCall(MatDestroy(&CU));

  PetscCall(MatDestroy(&C1));
  PetscCall(MatDestroy(&C2));
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
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

   testset:
     nsize: 2
     output_file: output/ex132_2_par.out
     requires: !sycl kokkos_kernels
     filter: grep -v " type:" | grep -v "Mat Object"
     args: -view -mat_type aijkokkos -mat_nonsym
     diff_args: -j
     test:
       suffix: 2_par_kokkos_no_gpu_aware
       args: -use_gpu_aware_mpi 0
     test:
       requires: defined(HAVE_MPI_GPU_AWARE)
       suffix: 2_par_kokkos_gpu_aware
       args: -use_gpu_aware_mpi 1

TEST*/
