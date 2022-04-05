
static char help[] = "Tests external Clique direct solvers. Simplified from ex130.c\n\
Example: mpiexec -n <np> ./ex168 -f <matrix binary file> \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,F;
  Vec            u,x,b;
  PetscMPIInt    rank,size;
  PetscInt       m,n,nfact;
  PetscReal      norm,tol=1.e-12,Anorm;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscBool      flg,testMatSolve=PETSC_TRUE;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Determine file from which we read the matrix A */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecLoad(b,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
  PetscCall(MatNorm(A,NORM_INFINITY,&Anorm));

  /* Create vectors */
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecDuplicate(x,&u)); /* save the true solution */

  /* Test Cholesky Factorization */
  PetscCall(MatGetOrdering(A,MATORDERINGNATURAL,&perm,&iperm));

  if (rank == 0) printf(" Clique Cholesky:\n");
  PetscCall(MatGetFactor(A,MATSOLVERCLIQUE,MAT_FACTOR_CHOLESKY,&F));

  info.fill = 5.0;
  PetscCall(MatCholeskyFactorSymbolic(F,A,perm,&info));

  for (nfact = 0; nfact < 1; nfact++) {
    if (rank == 0) printf(" %d-the Cholesky numfactorization \n",nfact);
    PetscCall(MatCholeskyFactorNumeric(F,A,&info));

    /* Test MatSolve() */
    if (testMatSolve && nfact == 2) {
      PetscCall(MatSolve(F,b,x));

      /* Check the residual */
      PetscCall(MatMult(A,x,u));
      PetscCall(VecAXPY(u,-1.0,b));
      PetscCall(VecNorm(u,NORM_INFINITY,&norm));
      /* if (norm > tol) { */
      if (rank == 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatSolve: rel residual %g/%g = %g, LU numfact %d\n",norm,Anorm,norm/Anorm,nfact));
      }
      /*} */
    }
  }

  /* Free data structures */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&F));
  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
}
