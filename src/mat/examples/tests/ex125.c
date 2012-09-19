
static char help[] = "Tests MatSolve() and MatMatSolve() (interface to superlu_dist).\n\
Example: mpiexec -n <np> ./ex125 -f <matrix binary file> -nrhs 4 \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    rank,nproc;
  PetscInt       i,m,n,nfact,nsolve,nrhs,k,ipack=0;
  PetscScalar    *array,rval;
  PetscReal      norm,tol=1.e-12;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscRandom    rand;
  PetscBool      flg,testMatSolve=PETSC_TRUE,testMatMatSolve=PETSC_TRUE;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nproc);CHKERRQ(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (m != n) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
  }

  /* Create dense matrix C and X; C holds true solution with identical colums */
  nrhs = 2;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nrhs",&nrhs,PETSC_NULL);CHKERRQ(ierr);
  if (!rank) printf("ex125: nrhs %d\n",nrhs);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatSetRandom(C,rand);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Test LU Factorization */
  ierr = MatGetOrdering(A,MATORDERINGND,&perm,&iperm);CHKERRQ(ierr);
  //ierr = ISView(perm,PETSC_VIEWER_STDOUT_WORLD);
  //ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_solver_package",&ipack,PETSC_NULL);CHKERRQ(ierr);
  switch (ipack){
  case 0:
#ifdef PETSC_HAVE_SUPERLU
    if (!rank) printf(" SUPERLU LU:\n");
    ierr = MatGetFactor(A,MATSOLVERSUPERLU,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    break;
#endif
  case 1:
#ifdef PETSC_HAVE_SUPERLU_DIST
    if (!rank) printf(" SUPERLU_DIST LU:\n");
    ierr = MatGetFactor(A,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    break;
#endif
  case 2:
#ifdef PETSC_HAVE_MUMPS
    if (!rank) printf(" MUMPS LU:\n");
    ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    {
      /* test mumps options */
      PetscInt icntl_7 = 5;
      ierr = MatMumpsSetIcntl(F,7,icntl_7);CHKERRQ(ierr);
    }
    break;
#endif
  default:
    if (!rank) printf(" PETSC LU:\n");
    ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  }

  info.fill = 5.0;
  ierr = MatLUFactorSymbolic(F,A,perm,iperm,&info);CHKERRQ(ierr);

  for (nfact = 0; nfact < 2; nfact++){
    if (!rank) printf(" %d-the LU numfactorization \n",nfact);
    ierr = MatLUFactorNumeric(F,A,&info);CHKERRQ(ierr);

    /* Test MatMatSolve() */
    /*
    if ((ipack == 0 || ipack == 2) && testMatMatSolve){
      printf("   MatMatSolve() is not implemented for this package. Skip the testing.\n");
      testMatMatSolve = PETSC_FALSE;
    }
     */
    if (testMatMatSolve){
      if (!nfact){
        ierr = MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS);CHKERRQ(ierr);
      } else {
        ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS);CHKERRQ(ierr);
      }
      for (nsolve = 0; nsolve < 2; nsolve++){
        if (!rank) printf("   %d-the MatMatSolve \n",nsolve);
        ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);

        /* Check the error */
        ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
        if (norm > tol){
          if (!rank){
            ierr = PetscPrintf(PETSC_COMM_SELF,"1st MatMatSolve: Norm of error %g, nsolve %d\n",norm,nsolve);CHKERRQ(ierr);
          }
        }
      }
    }

    /* Test MatSolve() */
    if (testMatSolve){
      for (nsolve = 0; nsolve < 2; nsolve++){
        ierr = VecGetArray(x,&array);CHKERRQ(ierr);
        for (i=0; i<m; i++){
          ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
          array[i] = rval;
        }
        ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
        ierr = VecCopy(x,u);CHKERRQ(ierr);
        ierr = MatMult(A,x,b);CHKERRQ(ierr);

        if (!rank) printf("   %d-the MatSolve \n",nsolve);
        ierr = MatSolve(F,b,x);CHKERRQ(ierr);

        /* Check the error */
        ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
        ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
        if (norm > tol){
          ierr = MatMult(A,x,u);CHKERRQ(ierr); /* u = A*x */
          PetscReal resi;
          ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);  /* u <- (-1.0)b + u */
          ierr = VecNorm(u,NORM_2,&resi);CHKERRQ(ierr);
          if (!rank){
            ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolve: Norm of error %g, resi %g, LU numfact %d\n",norm,resi,nfact);CHKERRQ(ierr);
          }
        }
      }
    }
  }

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  if (testMatMatSolve){
    ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  }

  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
