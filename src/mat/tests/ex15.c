
static char help[] = "Tests MatNorm(), MatLUFactor(), MatSolve() and MatSolveAdd().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 3,n = 3,Ii,J;
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscScalar    v;
  IS             perm,iperm;
  Vec            x,u,b,y;
  PetscReal      norm,tol=PETSC_SMALL;
  MatFactorInfo  info;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-symmetric",&flg));
  if (flg) {  /* Treat matrix as symmetric only if we set this flag */
    CHKERRQ(MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE));
    CHKERRQ(MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  }

  /* Create the matrix for the five point stencil, YET AGAIN */
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
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m*n,&u));
  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(VecDuplicate(u,&x));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(u,&y));
  CHKERRQ(MatMult(C,u,b));
  CHKERRQ(VecCopy(b,y));
  CHKERRQ(VecScale(y,2.0));

  CHKERRQ(MatNorm(C,NORM_FROBENIUS,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Frobenius norm of matrix %g\n",(double)norm));
  CHKERRQ(MatNorm(C,NORM_1,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"One  norm of matrix %g\n",(double)norm));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Infinity norm of matrix %g\n",(double)norm));

  CHKERRQ(MatFactorInfoInitialize(&info));
  info.fill          = 2.0;
  info.dtcol         = 0.0;
  info.zeropivot     = 1.e-14;
  info.pivotinblocks = 1.0;

  CHKERRQ(MatLUFactor(C,perm,iperm,&info));

  /* Test MatSolve */
  CHKERRQ(MatSolve(C,b,x));
  CHKERRQ(VecView(b,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatSolve: Norm of error %g\n",(double)norm));
  }

  /* Test MatSolveAdd */
  CHKERRQ(MatSolveAdd(C,b,y,x));
  CHKERRQ(VecAXPY(x,-1.0,y));
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatSolveAdd(): Norm of error %g\n",(double)norm));
  }

  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
