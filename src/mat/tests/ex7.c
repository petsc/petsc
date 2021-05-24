
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

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Mat test ex7 options","Mat");CHKERRQ(ierr);
  m = 3; n = 3; fill = 2.0;
  ierr = PetscOptionsInt("-m","Number of rows in grid",NULL,m,&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","Number of columns in grid",NULL,n,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-fill","Expected fill ratio for factorization",NULL,fill,&fill,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetOrdering(C,MATORDERINGRCM,&perm,&iperm);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&luinfo);CHKERRQ(ierr);

  luinfo.fill          = fill;
  luinfo.dtcol         = 0.0;
  luinfo.zeropivot     = 1.e-14;
  luinfo.pivotinblocks = 1.0;

  ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&LU);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(LU,C,perm,iperm,&luinfo);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(LU,C,&luinfo);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m*n,&u);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);

  ierr = MatMult(C,u,b);CHKERRQ(ierr);
  ierr = MatSolve(LU,b,x);CHKERRQ(ierr);

  ierr = VecView(b,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",(double)norm);CHKERRQ(ierr);

  ierr = MatGetInfo(C,MAT_LOCAL,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"original matrix nonzeros = %D\n",(PetscInt)info.nz_used);CHKERRQ(ierr);
  ierr = MatGetInfo(LU,MAT_LOCAL,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"factored matrix nonzeros = %D\n",(PetscInt)info.nz_used);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&LU);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
