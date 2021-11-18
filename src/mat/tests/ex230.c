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
  PetscErrorCode ierr;

  /*
     Create the Jacobian matrix
  */
  PetscFunctionBegin;
  M = 10;
  N = 8;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  /*
     Get the sizes of the jacobian.
  */
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);

  /*
     Create a preallocator matrix with sizes (local and global) matching the jacobian A.
  */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&preallocator);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(preallocator,bs);CHKERRQ(ierr);
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);

  /*
     Insert non-zero pattern (e.g. perform a sweep over the grid).
     You can use MatSetValues(), MatSetValuesBlocked() or MatSetValue().
  */
  {
    PetscInt    ii,jj;
    PetscScalar vv = 0.0;

    ii = 3; jj = 3;
    ierr = MatSetValues(preallocator,1,&ii,1,&jj,&vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 7; jj = 4;
    ierr = MatSetValues(preallocator,1,&ii,1,&jj,&vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 9; jj = 7;
    ierr = MatSetValue(preallocator,ii,jj,vv,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Push the non-zero pattern defined within preallocator into A.
     Internally this will call MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE).
     The arg fill = PETSC_TRUE will insert zeros in the matrix A automatically.
  */
  ierr = MatPreallocatorPreallocate(preallocator,PETSC_TRUE,A);CHKERRQ(ierr);

  /*
     We no longer require the preallocator object so destroy it.
  */
  ierr = MatDestroy(&preallocator);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Insert non-zero values into A.
  */
  {
    PetscInt    ii,jj;
    PetscScalar vv;

    ii = 3; jj = 3; vv = 0.3;
    ierr = MatSetValue(A,ii,jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 7; jj = 4; vv = 3.3;
    ierr = MatSetValues(A,1,&ii,1,&jj,&vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 9; jj = 7; vv = 4.3;
    ierr = MatSetValues(A,1,&ii,1,&jj,&vv,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ex2_square_bsvariable(void)
{
  Mat            A,preallocator;
  PetscInt       M,N,m,n,bs = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(NULL,NULL,"-block_size",&bs,NULL);CHKERRQ(ierr);

  /*
     Create the Jacobian matrix.
  */
  M = 10 * bs;
  N = 10 * bs;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  /*
     Get the sizes of the jacobian.
  */
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);

  /*
     Create a preallocator matrix with dimensions matching the jacobian A.
  */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&preallocator);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(preallocator,bs);CHKERRQ(ierr);
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);

  /*
     Insert non-zero pattern (e.g. perform a sweep over the grid).
     You can use MatSetValues(), MatSetValuesBlocked() or MatSetValue().
  */
  {
    PetscInt  ii,jj;
    PetscScalar *vv;

    ierr = PetscCalloc1(bs*bs,&vv);CHKERRQ(ierr);

    ii = 0; jj = 9;
    ierr = MatSetValue(preallocator,ii,jj,vv[0],INSERT_VALUES);CHKERRQ(ierr);

    ii = 0; jj = 0;
    ierr = MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 2; jj = 4;
    ierr = MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 4; jj = 2;
    ierr = MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 4; jj = 4;
    ierr = MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 9; jj = 9;
    ierr = MatSetValuesBlocked(preallocator,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ierr = PetscFree(vv);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Push non-zero pattern defined from preallocator into A.
     Internally this will call MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE).
     The arg fill = PETSC_TRUE will insert zeros in the matrix A automatically.
  */
  ierr = MatPreallocatorPreallocate(preallocator,PETSC_TRUE,A);CHKERRQ(ierr);

  /*
     We no longer require the preallocator object so destroy it.
  */
  ierr = MatDestroy(&preallocator);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  {
    PetscInt    ii,jj;
    PetscScalar *vv;

    ierr = PetscCalloc1(bs*bs,&vv);CHKERRQ(ierr);
    for (ii=0; ii<bs*bs; ii++) {
      vv[ii] = (PetscReal)(ii+1);
    }

    ii = 0; jj = 9;
    ierr = MatSetValue(A,ii,jj,33.3,INSERT_VALUES);CHKERRQ(ierr);

    ii = 0; jj = 0;
    ierr = MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 2; jj = 4;
    ierr = MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 4; jj = 2;
    ierr = MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 4; jj = 4;
    ierr = MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ii = 9; jj = 9;
    ierr = MatSetValuesBlocked(A,1,&ii,1,&jj,vv,INSERT_VALUES);CHKERRQ(ierr);

    ierr = PetscFree(vv);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscErrorCode ierr;
  PetscInt testid = 0;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-test_id",&testid,NULL);CHKERRQ(ierr);
  switch (testid) {
    case 0:
      ierr = ex1_nonsquare_bs1();CHKERRQ(ierr);
      break;
    case 1:
      ierr = ex2_square_bsvariable();CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid value for -test_id. Must be {0,1}");
  }
  ierr = PetscFinalize();
  return ierr;
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
