
static char help[] = "Tests matrix factorization.  Note that most users should\n\
employ the KSP  interface to the linear solvers instead of using the factorization\n\
routines directly.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,LU;
  MatInfo        info;
  PetscInt       i,j,m,n,Ii,J;
  PetscScalar    v,one = 1.0;
  IS             perm,iperm;
  Vec            x,u,b;
  PetscReal      norm,fill;
  MatFactorInfo  luinfo;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Mat test ex7 options","Mat");
  m = 3; n = 3; fill = 2.0;
  PetscCall(PetscOptionsInt("-m","Number of rows in grid",NULL,m,&m,NULL));
  PetscCall(PetscOptionsInt("-n","Number of columns in grid",NULL,n,&n,NULL));
  PetscCall(PetscOptionsReal("-fill","Expected fill ratio for factorization",NULL,fill,&fill,NULL));

  PetscOptionsEnd();

  /* Create the matrix for the five point stencil, YET AGAIN */
  PetscCall(MatCreate(PETSC_COMM_SELF,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
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
  PetscCall(MatGetOrdering(C,MATORDERINGRCM,&perm,&iperm));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISView(perm,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatFactorInfoInitialize(&luinfo));

  luinfo.fill          = fill;
  luinfo.dtcol         = 0.0;
  luinfo.zeropivot     = 1.e-14;
  luinfo.pivotinblocks = 1.0;

  PetscCall(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&LU));
  PetscCall(MatLUFactorSymbolic(LU,C,perm,iperm,&luinfo));
  PetscCall(MatLUFactorNumeric(LU,C,&luinfo));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m*n,&u));
  PetscCall(VecSet(u,one));
  PetscCall(VecDuplicate(u,&x));
  PetscCall(VecDuplicate(u,&b));

  PetscCall(MatMult(C,u,b));
  PetscCall(MatSolve(LU,b,x));

  PetscCall(VecView(b,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",(double)norm));

  PetscCall(MatGetInfo(C,MAT_LOCAL,&info));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"original matrix nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used));
  PetscCall(MatGetInfo(LU,MAT_LOCAL,&info));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"factored matrix nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used));

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&LU));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      filter: grep -v " MPI process"

   test:
      suffix: 2
      args: -m 1 -n 1 -fill 0.49
      filter: grep -v " MPI process"

TEST*/
