static char help[] = "Reads a PETSc matrix from a file and solves a linear system \n\
using the aijcusparse class. Input parameters are:\n\
  -f <input_file> : the file to load\n\n";

/*
  This code can be used to test PETSc interface to other packages.\n\
  Examples of command line options:       \n\
   ./ex43 -f DATAFILESPATH/matrices/cfd.2.10 -mat_cusparse_mult_storage_format ell  \n\
   ./ex43 -f DATAFILESPATH/matrices/shallow_water1 -ksp_type cg -pc_type icc -mat_cusparse_mult_storage_format ell  \n\
   \n\n";
*/

#include <petscksp.h>

int main(int argc,char **argv)
{
  KSP                ksp;
  Mat                A;
  Vec                X,B;
  PetscInt           m, its;
  PetscReal          norm;
  char               file[PETSC_MAX_PATH_LEN];
  PetscBool          flg;
  PetscViewer        fd;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  /* Load the data from a file */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /* Build the matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,fd));

  /* Build the vectors */
  PetscCall(MatGetLocalSize(A,&m,NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&B));
  PetscCall(VecSetSizes(B,m,PETSC_DECIDE));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&X));
  PetscCall(VecSetSizes(X,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(B));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecSet(B,1.0));

  /* Build the KSP */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetType(ksp,KSPGMRES));
  PetscCall(KSPSetTolerances(ksp,1.0e-12,PETSC_DEFAULT,PETSC_DEFAULT,100));
  PetscCall(KSPSetFromOptions(ksp));

  /* Solve */
  PetscCall(KSPSolve(ksp,B,X));

  /* print out norm and the number of iterations */
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(KSPGetResidualNorm(ksp,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3" PetscInt_FMT "\n",its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %1.5g\n",(double)norm));

  /* Cleanup */
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: cuda datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) !CUDA_VERSION_11PLUS
      args: -f ${DATAFILESPATH}/matrices/cfd.2.10 -mat_type seqaijcusparse -pc_factor_mat_solver_type cusparse -mat_cusparse_storage_format ell -vec_type cuda -pc_type ilu

   test:
      suffix: 2
      requires: cuda datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) !CUDA_VERSION_11PLUS
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type seqaijcusparse -pc_factor_mat_solver_type cusparse -mat_cusparse_storage_format hyb -vec_type cuda -ksp_type cg -pc_type icc

   testset:
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.10 -ksp_type bicg -pc_type ilu

      test:
         suffix: 3
         requires: cuda
         args: -mat_type seqaijcusparse -pc_factor_mat_solver_type cusparse -mat_cusparse_storage_format csr -vec_type cuda
      test:
         suffix: 4
         requires: cuda
         args: -mat_type seqaijcusparse -pc_factor_mat_solver_type cusparse -mat_cusparse_storage_format csr -vec_type cuda -pc_factor_mat_ordering_type nd
      test: # Test MatSolveTranspose
         suffix: 3_kokkos
         requires: !sycl kokkos_kernels
         args: -mat_type seqaijkokkos -pc_factor_mat_solver_type kokkos -vec_type kokkos
         output_file: output/ex43_3.out

   testset:
      nsize: 2
      requires: cuda datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) !CUDA_VERSION_11PLUS
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type mpiaijcusparse -mat_cusparse_mult_diag_storage_format hyb -pc_type none -vec_type cuda
      test:
        suffix: 5
        args: -use_gpu_aware_mpi 0
      test:
        suffix: 5_gpu_aware_mpi
        output_file: output/ex43_5.out

   test:
      suffix: 6
      requires: cuda datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type seqaijcusparse -pc_type none -vec_type cuda

   testset:
      nsize: 2
      requires: cuda datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type mpiaijcusparse -pc_type none -vec_type cuda

      test:
        suffix: 7
        args: -use_gpu_aware_mpi 0
      test:
        suffix: 7_gpu_aware_mpi
        output_file: output/ex43_7.out

   test:
      suffix: 8
      requires: viennacl datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type seqaijviennacl -pc_type none -vec_type viennacl
      output_file: output/ex43_6.out

   test:
      suffix: 9
      nsize: 2
      requires: viennacl datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type mpiaijviennacl -pc_type none -vec_type viennacl
      output_file: output/ex43_7.out

   test:
      suffix: 10
      nsize: 2
      requires: !sycl kokkos_kernels datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -mat_type aijkokkos -vec_type kokkos

TEST*/
