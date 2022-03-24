static char help[] = "Example of using MatPreallocator\n\n";

/*T
   Concepts: Mat^matrix preallocation
   Processors: n
T*/

#include <petscmat.h>

PetscErrorCode ex1_nonsquare_bs1(void)
{
  Mat            A,preallocator;
  PetscInt       M,N,m,n,bs;

  /*
     Create the Jacobian matrix
  */
  PetscFunctionBegin;
  M = 10;
  N = 8;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetBlockSize(A,1));
  CHKERRQ(MatSetFromOptions(A));

  /*
     Get the sizes of the jacobian.
  */
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatGetBlockSize(A,&bs));

  /*
     Create a preallocator matrix with sizes (local and global) matching the jacobian A.
  */
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&preallocator));
  CHKERRQ(MatSetType(preallocator,MATPREALLOCATOR));
  CHKERRQ(MatSetSizes(preallocator,m,n,M,N));
  CHKERRQ(MatSetBlockSize(preallocator,bs));
  CHKERRQ(MatSetUp(preallocator));

  /*
     Insert non-zero pattern (e.g. perform a sweep over the grid).
     You can use MatSetValues(), MatSetValuesBlocked() or MatSetValue().
  */
  {
    PetscInt    ii,jj;
    PetscScalar vv = 0.0;

    ii = 3; jj = 3;
    CHKERRQ(MatSetValues(preallocator,1,&ii,1,&jj,&vv,INSERT_VALUES));

    ii = 7; jj = 4;
    CHKERRQ(MatSetValues(preallocator,1,&ii,1,&jj,&vv,INSERT_VALUES));

    ii = 9; jj = 7;
    CHKERRQ(MatSetValue(preallocator,ii,jj,vv,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY));

  /*
     Push the non-zero pattern defined within preallocator into A.
     Internally this will call MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE).
     The arg fill = PETSC_TRUE will insert zeros in the matrix A automatically.
  */
  CHKERRQ(MatPreallocatorPreallocate(preallocator,PETSC_TRUE,A));

  /*
     We no longer require the preallocator object so destroy it.
  */
  CHKERRQ(MatDestroy(&preallocator));

  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Insert non-zero values into A.
  */
  {
    PetscInt    ii,jj;
    PetscScalar vv;

    ii = 3; jj = 3; vv = 0.3;
    CHKERRQ(MatSetValue(A,ii,jj,vv,INSERT_VALUES));

    ii = 7; jj = 4; vv = 3.3;
    CHKERRQ(MatSetValues(A,1,&ii,1,&jj,&vv,INSERT_VALUES));

    ii = 9; jj = 7; vv = 4.3;
    CHKERRQ(MatSetValues(A,1,&ii,1,&jj,&vv,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

PetscErrorCode ex2_square_bsvariable(void)
{
  Mat            A,preallocator;
  PetscInt       M,N,m,n,bs = 1;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-block_size",&bs,NULL));

  /*
     Create the Jacobian matrix.
  */
  M = 10 * bs;
  N = 10 * bs;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetBlockSize(A,bs));
  CHKERRQ(MatSetFromOptions(A));

  /*
     Get the sizes of the jacobian.
  */
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatGetBlockSize(A,&bs));

  /*
     Create a preallocator matrix with dimensions matching the jacobian A.
  */
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&preallocator));
  CHKERRQ(MatSetType(preallocator,MATPREALLOCATOR));
  CHKERRQ(MatSetSizes(preallocator,m,n,M,N));
  CHKERRQ(MatSetBlockSize(preallocator,bs));
  CHKERRQ(MatSetUp(preallocator));

  /*
     Insert non-zero pattern (e.g. perform a sweep over the grid).
     You can use MatSetValues(), MatSetValuesBlocked() or MatSetValue().
  */
  {
    PetscInt  ii,jj;
    PetscScalar *vv;

    CHKERRQ(PetscCalloc1(bs*bs,&vv));

    ii = 0; jj = 9;
    CHKERRQ(MatSetValue(preallocator,ii,jj,vv[0],INSERT_VALUES));

    ii = 0; jj = 0;
    CHKERRQ(MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 2; jj = 4;
    CHKERRQ(MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 4; jj = 2;
    CHKERRQ(MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 4; jj = 4;
    CHKERRQ(MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 9; jj = 9;
    CHKERRQ(MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES));

    CHKERRQ(PetscFree(vv));
  }
  CHKERRQ(MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY));

  /*
     Push non-zero pattern defined from preallocator into A.
     Internally this will call MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE).
     The arg fill = PETSC_TRUE will insert zeros in the matrix A automatically.
  */
  CHKERRQ(MatPreallocatorPreallocate(preallocator,PETSC_TRUE,A));

  /*
     We no longer require the preallocator object so destroy it.
  */
  CHKERRQ(MatDestroy(&preallocator));

  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  {
    PetscInt    ii,jj;
    PetscScalar *vv;

    CHKERRQ(PetscCalloc1(bs*bs,&vv));
    for (ii=0; ii<bs*bs; ii++) {
      vv[ii] = (PetscReal)(ii+1);
    }

    ii = 0; jj = 9;
    CHKERRQ(MatSetValue(A,ii,jj,33.3,INSERT_VALUES));

    ii = 0; jj = 0;
    CHKERRQ(MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 2; jj = 4;
    CHKERRQ(MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 4; jj = 2;
    CHKERRQ(MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 4; jj = 4;
    CHKERRQ(MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES));

    ii = 9; jj = 9;
    CHKERRQ(MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES));

    CHKERRQ(PetscFree(vv));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscInt testid = 0;
  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test_id",&testid,NULL));
  switch (testid) {
    case 0:
      CHKERRQ(ex1_nonsquare_bs1());
      break;
    case 1:
      CHKERRQ(ex2_square_bsvariable());
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid value for -test_id. Must be {0,1}");
  }
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: t0_a_aij
     nsize: 1
     args: -test_id 0 -mat_type aij

   test:
     suffix: t0_b_aij
     nsize: 6
     args: -test_id 0 -mat_type aij

   test:
     suffix: t1_a_aij
     nsize: 1
     args: -test_id 1 -mat_type aij

   test:
     suffix: t2_a_baij
     nsize: 1
     args: -test_id 1 -mat_type baij

   test:
     suffix: t3_a_sbaij
     nsize: 1
     args: -test_id 1 -mat_type sbaij

   test:
     suffix: t4_a_aij_bs3
     nsize: 1
     args: -test_id 1 -mat_type aij -block_size 3

   test:
     suffix: t5_a_baij_bs3
     nsize: 1
     args: -test_id 1 -mat_type baij -block_size 3

   test:
     suffix: t6_a_sbaij_bs3
     nsize: 1
     args: -test_id 1 -mat_type sbaij -block_size 3

   test:
     suffix: t4_b_aij_bs3
     nsize: 6
     args: -test_id 1 -mat_type aij -block_size 3

   test:
     suffix: t5_b_baij_bs3
     nsize: 6
     args: -test_id 1 -mat_type baij -block_size 3
     filter: grep -v Mat_

   test:
     suffix: t6_b_sbaij_bs3
     nsize: 6
     args: -test_id 1 -mat_type sbaij -block_size 3
     filter: grep -v Mat_

TEST*/
