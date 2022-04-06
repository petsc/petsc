
static char help[] = "Test procedural KSPSetFromOptions() or at runtime; Test PCREDUNDANT.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x, b, u;     /* approx solution, RHS, exact solution */
  Mat            A;           /* linear system matrix */
  KSP            ksp;         /* linear solver context */
  PC             pc;          /* preconditioner context */
  PetscReal      norm;        /* norm of solution error */
  PetscInt       i,n = 10,col[3],its,rstart,rend,nlocal;
  PetscScalar    one = 1.0,value[3];
  PetscBool      TEST_PROCEDURAL=PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-procedural",&TEST_PROCEDURAL,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u));

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

  PetscCall(VecGetOwnershipRange(x,&rstart,&rend));
  PetscCall(VecGetLocalSize(x,&nlocal));

  /* Create a tridiagonal matrix. See ../tutorials/ex23.c */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,nlocal,nlocal,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  /* Assemble matrix */
  if (!rstart) {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
    PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Set exact solution; then compute right-hand-side vector. */
  PetscCall(VecSet(u,one));
  PetscCall(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  if (TEST_PROCEDURAL) {
    /* Example of runtime options: '-pc_redundant_number 3 -redundant_ksp_type gmres -redundant_pc_type bjacobi' */
    PetscMPIInt size,rank,subsize;
    Mat         A_redundant;
    KSP         innerksp;
    PC          innerpc;
    MPI_Comm    subcomm;

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCREDUNDANT));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    PetscCheck(size > 2,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE, "Num of processes %d must greater than 2",size);
    PetscCall(PCRedundantSetNumber(pc,size-2));
    PetscCall(KSPSetFromOptions(ksp));

    /* Get subcommunicator and redundant matrix */
    PetscCall(KSPSetUp(ksp));
    PetscCall(PCRedundantGetKSP(pc,&innerksp));
    PetscCall(KSPGetPC(innerksp,&innerpc));
    PetscCall(PCGetOperators(innerpc,NULL,&A_redundant));
    PetscCall(PetscObjectGetComm((PetscObject)A_redundant,&subcomm));
    PetscCallMPI(MPI_Comm_size(subcomm,&subsize));
    if (subsize==1 && rank == 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"A_redundant:\n"));
      PetscCall(MatView(A_redundant,PETSC_VIEWER_STDOUT_SELF));
    }
  } else {
    PetscCall(KSPSetFromOptions(ksp));
  }

  /*  Solve linear system */
  PetscCall(KSPSolve(ksp,b,x));

  /* Check the error */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %" PetscInt_FMT "\n",(double)norm,its));

  /* Free work space. */
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b)); PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: 3
      output_file: output/ex28.out

    test:
      suffix: 2
      args:  -procedural -pc_redundant_number 3 -redundant_ksp_type gmres -redundant_pc_type bjacobi
      nsize: 3

    test:
      suffix: 3
      args:  -procedural -pc_redundant_number 3 -redundant_ksp_type gmres -redundant_pc_type bjacobi
      nsize: 5

TEST*/
