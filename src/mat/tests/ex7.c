
static char help[] = "Tests matrix factorization.  Note that most users should\n\
employ the KSP  interface to the linear solvers instead of using the factorization\n\
routines directly.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,LU;
  MatInfo        info;
  PetscInt       i,j,m,n,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v,one = 1.0;
  IS             perm,iperm;
  Vec            x,u,b;
  PetscReal      norm,fill;
  MatFactorInfo  luinfo;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Mat test ex7 options","Mat");CHKERRQ(ierr);
  m = 3; n = 3; fill = 2.0;
  CHKERRQ(PetscOptionsInt("-m","Number of rows in grid",NULL,m,&m,NULL));
  CHKERRQ(PetscOptionsInt("-n","Number of columns in grid",NULL,n,&n,NULL));
  CHKERRQ(PetscOptionsReal("-fill","Expected fill ratio for factorization",NULL,fill,&fill,NULL));

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create the matrix for the five point stencil, YET AGAIN */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
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
  CHKERRQ(MatGetOrdering(C,MATORDERINGRCM,&perm,&iperm));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISView(perm,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(MatFactorInfoInitialize(&luinfo));

  luinfo.fill          = fill;
  luinfo.dtcol         = 0.0;
  luinfo.zeropivot     = 1.e-14;
  luinfo.pivotinblocks = 1.0;

  CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&LU));
  CHKERRQ(MatLUFactorSymbolic(LU,C,perm,iperm,&luinfo));
  CHKERRQ(MatLUFactorNumeric(LU,C,&luinfo));

  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m*n,&u));
  CHKERRQ(VecSet(u,one));
  CHKERRQ(VecDuplicate(u,&x));
  CHKERRQ(VecDuplicate(u,&b));

  CHKERRQ(MatMult(C,u,b));
  CHKERRQ(MatSolve(LU,b,x));

  CHKERRQ(VecView(b,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",(double)norm));

  CHKERRQ(MatGetInfo(C,MAT_LOCAL,&info));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"original matrix nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used));
  CHKERRQ(MatGetInfo(LU,MAT_LOCAL,&info));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"factored matrix nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used));

  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&LU));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      filter: grep -v "MPI processes"

   test:
      suffix: 2
      args: -m 1 -n 1 -fill 0.49
      filter: grep -v "MPI processes"

TEST*/
