
static char help[] = "Bilinear elements on the unit square for Laplacian.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

/* Declare user-defined routines */
extern PetscErrorCode FormElementStiffness(PetscReal,PetscScalar*);
extern PetscErrorCode FormElementRhs(PetscScalar,PetscScalar,PetscReal,PetscScalar*);

int main(int argc,char **args)
{
  Vec            u,b,ustar; /* approx solution, RHS, exact solution */
  Mat            A;           /* linear system matrix */
  KSP            ksp;         /* Krylov subspace method context */
  PetscInt       N;           /* dimension of system (global) */
  PetscInt       M;           /* number of elements (global) */
  PetscMPIInt    rank;        /* processor rank */
  PetscMPIInt    size;        /* size of communicator */
  PetscScalar    Ke[16];      /* element matrix */
  PetscScalar    r[4];        /* element vector */
  PetscReal      h;           /* mesh width */
  PetscReal      norm;        /* norm of solution error */
  PetscScalar    x,y;
  PetscInt       idx[4],count,*rows,i,m = 5,start,end,its;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N    = (m+1)*(m+1);
  M    = m*m;
  h    = 1.0/m;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Au = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create stiffness matrix
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A,9,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,9,NULL,8,NULL));
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /*
     Assemble matrix
  */
  PetscCall(FormElementStiffness(h*h,Ke));
  for (i=start; i<end; i++) {
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    PetscCall(MatSetValues(A,4,idx,4,idx,Ke,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Create right-hand-side and solution vectors
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(PetscObjectSetName((PetscObject)u,"Approx. Solution"));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(PetscObjectSetName((PetscObject)b,"Right hand side"));
  PetscCall(VecDuplicate(b,&ustar));
  PetscCall(VecSet(u,0.0));
  PetscCall(VecSet(b,0.0));

  /*
     Assemble right-hand-side vector
  */
  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    x = h*(i % m); y = h*(i/m);
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    PetscCall(FormElementRhs(x,y,h*h,r));
    PetscCall(VecSetValues(b,4,idx,r,ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /*
     Modify matrix and right-hand-side for Dirichlet boundary conditions
  */
  PetscCall(PetscMalloc1(4*m,&rows));
  for (i=0; i<m+1; i++) {
    rows[i] = i; /* bottom */
    rows[3*m - 1 +i] = m*(m+1) + i; /* top */
  }
  count = m+1; /* left side */
  for (i=m+1; i<m*(m+1); i+= m+1) rows[count++] = i;
  count = 2*m; /* left side */
  for (i=2*m+1; i<m*(m+1); i+= m+1) rows[count++] = i;
  for (i=0; i<4*m; i++) {
    y = h*(rows[i]/(m+1));
    PetscCall(VecSetValues(u,1,&rows[i],&y,INSERT_VALUES));
    PetscCall(VecSetValues(b,1,&rows[i],&y,INSERT_VALUES));
  }
  PetscCall(MatZeroRows(A,4*m,rows,1.0,0,0));
  PetscCall(PetscFree(rows));

  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPSolve(ksp,b,u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Check error */
  PetscCall(VecGetOwnershipRange(ustar,&start,&end));
  for (i=start; i<end; i++) {
    y = h*(i/(m+1));
    PetscCall(VecSetValues(ustar,1,&i,&y,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(ustar));
  PetscCall(VecAssemblyEnd(ustar));
  PetscCall(VecAXPY(u,-1.0,ustar));
  PetscCall(VecNorm(u,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)(norm*h),its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp)); PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&ustar)); PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/* --------------------------------------------------------------------- */
/* element stiffness for Laplacian */
PetscErrorCode FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  PetscFunctionBeginUser;
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode FormElementRhs(PetscScalar x,PetscScalar y,PetscReal H,PetscScalar *r)
{
  PetscFunctionBeginUser;
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0;
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -ksp_monitor_short

TEST*/
