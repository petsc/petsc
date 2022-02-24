
static char help[] = "Test procedural KSPSetFromOptions() or at runtime; Test PCREDUNDANT.\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Processors: n
T*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x, b, u;     /* approx solution, RHS, exact solution */
  Mat            A;           /* linear system matrix */
  KSP            ksp;         /* linear solver context */
  PC             pc;          /* preconditioner context */
  PetscReal      norm;        /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 10,col[3],its,rstart,rend,nlocal;
  PetscScalar    one = 1.0,value[3];
  PetscBool      TEST_PROCEDURAL=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-procedural",&TEST_PROCEDURAL,NULL));

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
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

  CHKERRQ(VecGetOwnershipRange(x,&rstart,&rend));
  CHKERRQ(VecGetLocalSize(x,&nlocal));

  /* Create a tridiagonal matrix. See ../tutorials/ex23.c */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,nlocal,nlocal,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  /* Assemble matrix */
  if (!rstart) {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
    CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Set exact solution; then compute right-hand-side vector. */
  CHKERRQ(VecSet(u,one));
  CHKERRQ(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));

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

    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCREDUNDANT));
    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    PetscCheckFalse(size < 3,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Num of processes %d must greater than 2",size);
    CHKERRQ(PCRedundantSetNumber(pc,size-2));
    CHKERRQ(KSPSetFromOptions(ksp));

    /* Get subcommunicator and redundant matrix */
    CHKERRQ(KSPSetUp(ksp));
    CHKERRQ(PCRedundantGetKSP(pc,&innerksp));
    CHKERRQ(KSPGetPC(innerksp,&innerpc));
    CHKERRQ(PCGetOperators(innerpc,NULL,&A_redundant));
    CHKERRQ(PetscObjectGetComm((PetscObject)A_redundant,&subcomm));
    CHKERRMPI(MPI_Comm_size(subcomm,&subsize));
    if (subsize==1 && rank == 0) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"A_redundant:\n"));
      CHKERRQ(MatView(A_redundant,PETSC_VIEWER_STDOUT_SELF));
    }
  } else {
    CHKERRQ(KSPSetFromOptions(ksp));
  }

  /*  Solve linear system */
  CHKERRQ(KSPSolve(ksp,b,x));

  /* Check the error */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /* Free work space. */
  CHKERRQ(VecDestroy(&x)); CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b)); CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
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
