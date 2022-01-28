
static char help[] = "Tests external Clique direct solvers. Simplified from ex130.c\n\
Example: mpiexec -n <np> ./ex168 -f <matrix binary file> \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,F;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       m,n,nfact;
  PetscReal      norm,tol=1.e-12,Anorm;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscBool      flg,testMatSolve=PETSC_TRUE;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecLoad(b,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  PetscAssertFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
  ierr = MatNorm(A,NORM_INFINITY,&Anorm);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Test Cholesky Factorization */
  ierr = MatGetOrdering(A,MATORDERINGNATURAL,&perm,&iperm);CHKERRQ(ierr);

  if (rank == 0) printf(" Clique Cholesky:\n");
  ierr = MatGetFactor(A,MATSOLVERCLIQUE,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);

  info.fill = 5.0;
  ierr      = MatCholeskyFactorSymbolic(F,A,perm,&info);CHKERRQ(ierr);

  for (nfact = 0; nfact < 1; nfact++) {
    if (rank == 0) printf(" %d-the Cholesky numfactorization \n",nfact);
    ierr = MatCholeskyFactorNumeric(F,A,&info);CHKERRQ(ierr);

    /* Test MatSolve() */
    if (testMatSolve && nfact == 2) {
      ierr = MatSolve(F,b,x);CHKERRQ(ierr);

      /* Check the residual */
      ierr = MatMult(A,x,u);CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
      ierr = VecNorm(u,NORM_INFINITY,&norm);CHKERRQ(ierr);
      /* if (norm > tol) { */
      if (rank == 0) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolve: rel residual %g/%g = %g, LU numfact %d\n",norm,Anorm,norm/Anorm,nfact);CHKERRQ(ierr);
      }
      /*} */
    }
  }

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
