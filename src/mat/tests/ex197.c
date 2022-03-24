static char help[] = "Test MatMultHermitianTranspose() and MatMultHermitianTransposeAdd().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat         A,B,C;
  Vec         x,y,ys;
  PetscInt    i,j;
  PetscScalar v;
  PetscBool   flg;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  i = 0; j = 0; v = 2.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i = 0; j = 1; v = 3.0 + 4.0*PETSC_i;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i = 1; j = 0; v = 5.0 + 6.0*PETSC_i;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i = 1; j = 1; v = 7.0 + 8.0*PETSC_i;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,PETSC_DECIDE,2));
  CHKERRQ(VecSetFromOptions(y));
  CHKERRQ(VecDuplicate(y,&ys));
  CHKERRQ(VecDuplicate(y,&x));

  i = 0; v = 10.0 + 11.0*PETSC_i;
  CHKERRQ(VecSetValues(x,1,&i,&v,INSERT_VALUES));
  i = 1; v = 100.0 + 120.0*PETSC_i;
  CHKERRQ(VecSetValues(x,1,&i,&v,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(MatMultHermitianTranspose(A,x,y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatMultHermitianTransposeAdd(A,x,y,ys));
  CHKERRQ(VecView(ys,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatCreateHermitianTranspose(A,&C));
  CHKERRQ(MatMultHermitianTransposeEqual(B,C,4,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"B^Hx != C^Hx");
  CHKERRQ(MatMultHermitianTransposeAddEqual(B,C,4,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"y+B^Hx != y+C^Hx");
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(MatDestroy(&A));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&ys));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: complex
   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
