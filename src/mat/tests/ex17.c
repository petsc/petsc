
static char help[] = "Tests the use of MatSolveTranspose().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 5,n = 5,Ii,J;
  PetscScalar    v,five = 5.0,one = 1.0;
  IS             isrow,row,col;
  Vec            x,u,b;
  PetscReal      norm;
  MatFactorInfo  info;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C));
  CHKERRQ(MatSetUp(C));

  /* create the matrix for the five point stencil, YET AGAIN*/
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow));
  CHKERRQ(MatZeroRowsIS(C,isrow,five,0,0));

  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m*n,&u));
  CHKERRQ(VecDuplicate(u,&x));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecSet(u,one));

  CHKERRQ(MatMultTranspose(C,u,b));

  /* Set default ordering to be Quotient Minimum Degree; also read
     orderings from the options database */
  CHKERRQ(MatGetOrdering(C,MATORDERINGQMD,&row,&col));

  CHKERRQ(MatFactorInfoInitialize(&info));
  CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&A));
  CHKERRQ(MatLUFactorSymbolic(A,C,row,col,&info));
  CHKERRQ(MatLUFactorNumeric(A,C,&info));
  CHKERRQ(MatSolveTranspose(A,b,x));

  CHKERRQ(ISView(row,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",(double)norm));

  CHKERRQ(ISDestroy(&row));
  CHKERRQ(ISDestroy(&col));
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
