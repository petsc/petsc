/*$Id: ex23.c,v 1.1 2000/09/10 21:58:08 curfman Exp curfman $*/

/* Program usage:  mpirun ex23 [-help] [all PETSc options] */

static char help[] = "Solves a tridiagonal linear system with SLES.\n\n";

/*T
   Concepts: SLES^Solving a system of linear equations (basic parallel example);
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESView(); SLESGetKSP(); SLESGetPC();
   Routines: KSPSetTolerances(); PCSetType();
   Processors: n
T*/

/* 
  Include "petscsles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding uniprocessor example is ex1.c
*/
#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* linear solver context */
  PC      pc;           /* preconditioner context */
  KSP     ksp;          /* Krylov subspace method context */
  double  norm;         /* norm of solution error */
  int     ierr,i,n = 10,col[3],its,rstart,rend;
  Scalar  neg_one = -1.0,one = 1.0,value[3];

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);

  /* 
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good 
     performance.  Since preallocation is not possible via the generic
     matrix creation routine MatCreate(), we recommend for practical 
     problems instead to use the creation routine for a particular matrix
     format, e.g.,
         MatCreateMPIAIJ() - sequential AIJ (compressed sparse row)
         MatCreateMPIBAIJ() - block AIJ
     See the matrix chapter of the users manual for details.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A);CHKERRA(ierr);

  /* 
     Assemble matrix.  

     The linear system is distributed across the processors by 
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.  
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh.  Set the
     boundary values for the problem domain. */

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRA(ierr);

  if (!rstart) {
    rstart = 1;
    i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);
  }
  if (rend == n) {
    rend = n-1;
    i = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0; 
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one,u);CHKERRA(ierr);
  ierr = MatMult(A,u,b);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* 
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the SLES context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       SLESSetFromOptions();
  */
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Solve linear system
  */
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr); 

  /* 
     View solver info; we could instead use the option -sles_view to
     print this info to the screen at the conclusion of SLESSolve().
  */
  ierr = SLESView(sles,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Check the error
  */
  ierr = VecAXPY(&neg_one,u,x);CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A, Iterations %d\n",norm,its);CHKERRA(ierr);
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x);CHKERRA(ierr); ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr); ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = SLESDestroy(sles);CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  PetscFinalize();
  return 0;
}
