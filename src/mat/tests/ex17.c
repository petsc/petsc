
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C));
  PetscCall(MatSetUp(C));

  /* create the matrix for the five point stencil, YET AGAIN*/
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  PetscCall(ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow));
  PetscCall(MatZeroRowsIS(C,isrow,five,0,0));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m*n,&u));
  PetscCall(VecDuplicate(u,&x));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecSet(u,one));

  PetscCall(MatMultTranspose(C,u,b));

  /* Set default ordering to be Quotient Minimum Degree; also read
     orderings from the options database */
  PetscCall(MatGetOrdering(C,MATORDERINGQMD,&row,&col));

  PetscCall(MatFactorInfoInitialize(&info));
  PetscCall(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&A));
  PetscCall(MatLUFactorSymbolic(A,C,row,col,&info));
  PetscCall(MatLUFactorNumeric(A,C,&info));
  PetscCall(MatSolveTranspose(A,b,x));

  PetscCall(ISView(row,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",(double)norm));

  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  PetscCall(ISDestroy(&isrow));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
