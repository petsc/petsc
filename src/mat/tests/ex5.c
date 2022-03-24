
static char help[] = "Tests MatMult(), MatMultAdd(), MatMultTranspose().\n\
Also MatMultTransposeAdd(), MatScale(), MatGetDiagonal(), and MatDiagonalScale().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  Vec            s,u,w,x,y,z;
  PetscInt       i,j,m = 8,n,rstart,rend,vstart,vend;
  PetscScalar    one = 1.0,negone = -1.0,v,alpha=0.1;
  PetscReal      norm, tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscBool      flg;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-rectA",&flg));
  if (flg) n += 2;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-rectB",&flg));
  if (flg) n -= 2;

  /* ---------- Assemble matrix and vectors ----------- */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&z));
  CHKERRQ(VecDuplicate(x,&w));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(y));
  CHKERRQ(VecDuplicate(y,&u));
  CHKERRQ(VecDuplicate(y,&s));
  CHKERRQ(VecGetOwnershipRange(y,&vstart,&vend));

  /* Assembly */
  for (i=rstart; i<rend; i++) {
    v    = 100*(i+1);
    CHKERRQ(VecSetValues(z,1,&i,&v,INSERT_VALUES));
    for (j=0; j<n; j++) {
      v    = 10*(i+1)+j+1;
      CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }

  /* Flush off proc Vec values and do more assembly */
  CHKERRQ(VecAssemblyBegin(z));
  for (i=vstart; i<vend; i++) {
    v    = one*((PetscReal)i);
    CHKERRQ(VecSetValues(y,1,&i,&v,INSERT_VALUES));
    v    = 100.0*i;
    CHKERRQ(VecSetValues(u,1,&i,&v,INSERT_VALUES));
  }

  /* Flush off proc Mat values and do more assembly */
  CHKERRQ(MatAssemblyBegin(C,MAT_FLUSH_ASSEMBLY));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10*(i+1)+j+1;
      CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  /* Try overlap Coomunication with the next stage XXXSetValues */
  CHKERRQ(VecAssemblyEnd(z));

  CHKERRQ(MatAssemblyEnd(C,MAT_FLUSH_ASSEMBLY));
  CHKMEMQ;
  /* The Assembly for the second Stage */
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));
  CHKERRQ(MatScale(C,alpha));
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"testing MatMult()\n"));
  CHKMEMQ;
  CHKERRQ(MatMult(C,y,x));
  CHKMEMQ;
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"testing MatMultAdd()\n"));
  CHKERRQ(MatMultAdd(C,y,z,w));
  CHKERRQ(VecAXPY(x,one,z));
  CHKERRQ(VecAXPY(x,negone,w));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error difference = %g\n",(double)norm));
  }

  /* ------- Test MatMultTranspose(), MatMultTransposeAdd() ------- */

  for (i=rstart; i<rend; i++) {
    v    = one*((PetscReal)i);
    CHKERRQ(VecSetValues(x,1,&i,&v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"testing MatMultTranspose()\n"));
  CHKERRQ(MatMultTranspose(C,x,y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"testing MatMultTransposeAdd()\n"));
  CHKERRQ(MatMultTransposeAdd(C,x,u,s));
  CHKERRQ(VecAXPY(y,one,u));
  CHKERRQ(VecAXPY(y,negone,s));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error difference = %g\n",(double)norm));
  }

  /* -------------------- Test MatGetDiagonal() ------------------ */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"testing MatGetDiagonal(), MatDiagonalScale()\n"));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecSet(x,one));
  CHKERRQ(MatGetDiagonal(C,x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  for (i=vstart; i<vend; i++) {
    v    = one*((PetscReal)(i+1));
    CHKERRQ(VecSetValues(y,1,&i,&v,INSERT_VALUES));
  }

  /* -------------------- Test () MatDiagonalScale ------------------ */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_diagonalscale",&flg));
  if (flg) {
    CHKERRQ(MatDiagonalScale(C,x,y));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  }
  /* Free data structures */
  CHKERRQ(VecDestroy(&u)); CHKERRQ(VecDestroy(&s));
  CHKERRQ(VecDestroy(&w)); CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y)); CHKERRQ(VecDestroy(&z));
  CHKERRQ(MatDestroy(&C));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 11_A
      args: -mat_type seqaij -rectA
      filter: grep -v type

   test:
      args: -mat_type seqdense -rectA
      suffix: 12_A

   test:
      args: -mat_type seqaij -rectB
      suffix: 11_B
      filter: grep -v type

   test:
      args: -mat_type seqdense -rectB
      suffix: 12_B

   test:
      suffix: 21
      args: -mat_type mpiaij
      filter: grep -v type

   test:
      suffix: 22
      args: -mat_type mpidense

   test:
      suffix: 23
      nsize: 3
      args: -mat_type mpiaij
      filter: grep -v type

   test:
      suffix: 24
      nsize: 3
      args: -mat_type mpidense

   test:
      suffix: 2_aijcusparse_1
      args: -mat_type mpiaijcusparse -vec_type cuda
      filter: grep -v type
      output_file: output/ex5_21.out
      requires: cuda

   test:
      nsize: 3
      suffix: 2_aijcusparse_2
      filter: grep -v type
      args: -mat_type mpiaijcusparse -vec_type cuda
      args: -sf_type {{basic neighbor}}
      output_file: output/ex5_23.out
      requires: cuda

   test:
      nsize: 3
      suffix: 2_aijcusparse_3
      filter: grep -v type
      args: -mat_type mpiaijcusparse -vec_type cuda
      args: -sf_type {{basic neighbor}}
      output_file: output/ex5_23.out
      requires: cuda defined(PETSC_HAVE_MPI_GPU_AWARE)

   test:
      suffix: 31
      args: -mat_type mpiaij -test_diagonalscale
      filter: grep -v type

   test:
      suffix: 32
      args: -mat_type mpibaij -test_diagonalscale
      filter: grep -v Mat_

   test:
      suffix: 33
      nsize: 3
      args: -mat_type mpiaij -test_diagonalscale
      filter: grep -v type

   test:
      suffix: 34
      nsize: 3
      args: -mat_type mpibaij -test_diagonalscale
      filter: grep -v Mat_

   test:
      suffix: 3_aijcusparse_1
      args: -mat_type mpiaijcusparse -vec_type cuda -test_diagonalscale
      filter: grep -v type
      output_file: output/ex5_31.out
      requires: cuda

   test:
      suffix: 3_aijcusparse_2
      nsize: 3
      args: -mat_type mpiaijcusparse -vec_type cuda -test_diagonalscale
      filter: grep -v type
      output_file: output/ex5_33.out
      requires: cuda

   test:
      suffix: 3_kokkos
      nsize: 3
      args: -mat_type mpiaijkokkos -vec_type kokkos -test_diagonalscale
      filter: grep -v type
      output_file: output/ex5_33.out
      requires: !sycl kokkos_kernels

   test:
      suffix: aijcusparse_1
      args: -mat_type seqaijcusparse -vec_type cuda -rectA
      filter: grep -v type
      output_file: output/ex5_11_A.out
      requires: cuda

   test:
      suffix: aijcusparse_2
      args: -mat_type seqaijcusparse -vec_type cuda -rectB
      filter: grep -v type
      output_file: output/ex5_11_B.out
      requires: cuda

   test:
      suffix: sell_1
      args: -mat_type sell
      output_file: output/ex5_41.out

   test:
      suffix: sell_2
      nsize: 3
      args: -mat_type sell
      output_file: output/ex5_43.out

   test:
      suffix: sell_3
      args: -mat_type sell -test_diagonalscale
      output_file: output/ex5_51.out

   test:
      suffix: sell_4
      nsize: 3
      args: -mat_type sell -test_diagonalscale
      output_file: output/ex5_53.out

TEST*/
