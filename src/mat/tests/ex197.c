static char help[] = "Test MatMultHermitianTranspose() and MatMultHermitianTransposeAdd().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,C;
  Vec            x,y,ys;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    v;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  i = 0; j = 0; v = 2.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0; j = 1; v = 3.0 + 4.0*PETSC_i;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1; j = 0; v = 5.0 + 6.0*PETSC_i;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1; j = 1; v = 7.0 + 8.0*PETSC_i;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&ys);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&x);CHKERRQ(ierr);

  i = 0; v = 10.0 + 11.0*PETSC_i;
  ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1; v = 100.0 + 120.0*PETSC_i;
  ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = MatMultHermitianTranspose(A,x,y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatMultHermitianTransposeAdd(A,x,y,ys);CHKERRQ(ierr);
  ierr = VecView(ys,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatCreateHermitianTranspose(A,&C);CHKERRQ(ierr);
  ierr = MatMultHermitianTransposeEqual(B,C,4,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"B^Hx != C^Hx");
  ierr = MatMultHermitianTransposeAddEqual(B,C,4,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"y+B^Hx != y+C^Hx");
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&ys);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: complex
   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
