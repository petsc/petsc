 
static char help[] = "Tests PLAPACK interface.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,C1,F; 
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    rank,nproc;
  PetscInt       i,M = 10,m,n,nfact,nsolve;
  PetscScalar    *array,rval;
  PetscReal      norm;
  PetscTruth     flg;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscRandom    rand;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nproc);CHKERRQ(ierr);

#ifdef PETSC_HAVE_PLAPACK
  /* Create matrix and vectors */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetType(C,MATPLAPACK);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(C);CHKERRQ(ierr); 
  
  ierr = MatGetLocalSize(C,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_ERR_ARG_WRONG,"Matrix local size m %d must equal n %d",m,n);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Assembly */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatGetArray(C,&array);CHKERRQ(ierr);
  for (i=0; i<m*M; i++){
    ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    array[i] = rval; 
  }
  ierr = MatRestoreArray(C,&array);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);   
  /*if (!rank) {printf("main, C: \n");}
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Test MatDuplicate() */
  ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr); 

  /* Test LU Factorization */
  ierr = MatGetOrdering(C1,MATORDERING_NATURAL,&perm,&iperm);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(C1,perm,iperm,&info,&F);CHKERRQ(ierr);
  for (nfact = 0; nfact < 2; nfact++){
    if (!rank) printf(" LU nfact %d\n",nfact);
    ierr = MatLUFactorNumeric(C1,&info,&F);CHKERRQ(ierr);

    /* Test MatSolve() */
    for (nsolve = 0; nsolve < 5; nsolve++){
      ierr = VecGetArray(x,&array);CHKERRQ(ierr);
      for (i=0; i<m; i++){
        ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
        array[i] = rval; 
      }
      ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
      ierr = VecCopy(x,u);CHKERRQ(ierr); 
      ierr = MatMult(C,x,b);CHKERRQ(ierr);

      ierr = MatSolve(F,b,x);CHKERRQ(ierr); 

      /* Check the error */
      ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      if (!rank){
        ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %A\n",norm);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatDestroy(C1);CHKERRQ(ierr);
  ierr = MatDestroy(F);CHKERRQ(ierr);

  /* Test Cholesky Factorization */
  ierr = MatTranspose(C,&C1);CHKERRQ(ierr); /* C1 = C^T */
  ierr = MatAXPY(C,1.0,C1,SAME_NONZERO_PATTERN);CHKERRQ(ierr); /* make C symmetric: C <- C + C^T */
  ierr = MatShift(C,M);CHKERRQ(ierr);  /* make C positive definite */
  ierr = MatDestroy(C1);CHKERRQ(ierr);
  
  ierr = MatSetOption(C,MAT_SYMMETRIC);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_SYMMETRY_ETERNAL);CHKERRQ(ierr); 

  ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(C,perm,&info,&F);CHKERRQ(ierr);
  for (nfact = 0; nfact < 2; nfact++){
    if (!rank) printf(" Cholesky nfact %d\n",nfact);
    ierr = MatCholeskyFactorNumeric(C1,&info,&F);CHKERRQ(ierr);

    /* Test MatSolve() */
    for (nsolve = 0; nsolve < 5; nsolve++){
      ierr = VecGetArray(x,&array);CHKERRQ(ierr);
      for (i=0; i<m; i++){
        ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
        array[i] = rval; 
      }
      ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
      ierr = VecCopy(x,u);CHKERRQ(ierr); 
      ierr = MatMult(C,x,b);CHKERRQ(ierr);

      ierr = MatSolve(F,b,x);CHKERRQ(ierr); 

      /* Check the error */
      ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      if (!rank){
        ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %A\n",norm);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatDestroy(C1);CHKERRQ(ierr);
  ierr = MatDestroy(F);CHKERRQ(ierr);

  /* Free data structures */
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr); 
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr); 
  ierr = MatDestroy(C);CHKERRQ(ierr); 

#else
  if (!rank) printf("This example needs PLAPLAPACK\n");
#endif
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
