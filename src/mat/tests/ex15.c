
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-symmetric",&flg);CHKERRQ(ierr);
  if (flg) {  /* Treat matrix as symmetric only if we set this flag */
    ierr = MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* Create the matrix for the five point stencil, YET AGAIN */
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
  ierr = VecCreateSeq(PETSC_COMM_SELF,m*n,&u);CHKERRQ(ierr);
  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&y);CHKERRQ(ierr);
  ierr = MatMult(C,u,b);CHKERRQ(ierr);
  ierr = VecCopy(b,y);CHKERRQ(ierr);
  ierr = VecScale(y,2.0);CHKERRQ(ierr);

  ierr = MatNorm(C,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Frobenius norm of matrix %g\n",(double)norm);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"One  norm of matrix %g\n",(double)norm);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Infinity norm of matrix %g\n",(double)norm);CHKERRQ(ierr);

  ierr               = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  info.fill          = 2.0;
  info.dtcol         = 0.0;
  info.zeropivot     = 1.e-14;
  info.pivotinblocks = 1.0;

  ierr = MatLUFactor(C,perm,iperm,&info);CHKERRQ(ierr);

  /* Test MatSolve */
  ierr = MatSolve(C,b,x);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolve: Norm of error %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Test MatSolveAdd */
  ierr = MatSolveAdd(C,b,y,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,y);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolveAdd(): Norm of error %g\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
