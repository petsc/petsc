
static char help[] = "Tests MatNorm(), MatLUFactor(), MatSolve() and MatSolveAdd().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 3,n = 3,Ii,J;
  PetscBool      flg;
  PetscScalar    v;
  IS             perm,iperm;
  Vec            x,u,b,y;
  PetscReal      norm,tol=PETSC_SMALL;
  MatFactorInfo  info;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-symmetric",&flg));
  if (flg) {  /* Treat matrix as symmetric only if we set this flag */
    PetscCall(MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE));
    PetscCall(MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  }

  /* Create the matrix for the five point stencil, YET AGAIN */
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
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m*n,&u));
  PetscCall(VecSet(u,1.0));
  PetscCall(VecDuplicate(u,&x));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(u,&y));
  PetscCall(MatMult(C,u,b));
  PetscCall(VecCopy(b,y));
  PetscCall(VecScale(y,2.0));

  PetscCall(MatNorm(C,NORM_FROBENIUS,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Frobenius norm of matrix %g\n",(double)norm));
  PetscCall(MatNorm(C,NORM_1,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"One  norm of matrix %g\n",(double)norm));
  PetscCall(MatNorm(C,NORM_INFINITY,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Infinity norm of matrix %g\n",(double)norm));

  PetscCall(MatFactorInfoInitialize(&info));
  info.fill          = 2.0;
  info.dtcol         = 0.0;
  info.zeropivot     = 1.e-14;
  info.pivotinblocks = 1.0;

  PetscCall(MatLUFactor(C,perm,iperm,&info));

  /* Test MatSolve */
  PetscCall(MatSolve(C,b,x));
  PetscCall(VecView(b,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatSolve: Norm of error %g\n",(double)norm));
  }

  /* Test MatSolveAdd */
  PetscCall(MatSolveAdd(C,b,y,x));
  PetscCall(VecAXPY(x,-1.0,y));
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatSolveAdd(): Norm of error %g\n",(double)norm));
  }

  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
