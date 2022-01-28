
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-procedural",&TEST_PROCEDURAL,NULL);CHKERRQ(ierr);

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
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);

  /* Create a tridiagonal matrix. See ../tutorials/ex23.c */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  /* Assemble matrix */
  if (!rstart) {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Set exact solution; then compute right-hand-side vector. */
  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

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

    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
    PetscAssertFalse(size < 3,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Num of processes %d must greater than 2",size);
    ierr = PCRedundantSetNumber(pc,size-2);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /* Get subcommunicator and redundant matrix */
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = PCRedundantGetKSP(pc,&innerksp);CHKERRQ(ierr);
    ierr = KSPGetPC(innerksp,&innerpc);CHKERRQ(ierr);
    ierr = PCGetOperators(innerpc,NULL,&A_redundant);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)A_redundant,&subcomm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(subcomm,&subsize);CHKERRMPI(ierr);
    if (subsize==1 && rank == 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"A_redundant:\n");CHKERRQ(ierr);
      ierr = MatView(A_redundant,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  } else {
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  }

  /*  Solve linear system */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* Check the error */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  /* Free work space. */
  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
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
