#ifndef lint
static char vcid[] = "$Id: ex5.c,v 1.22 1996/07/08 22:20:55 bsmith Exp $";
#endif

static char help[] = "Illustrates use of the block Jacobi preconditioner for\n\
solving a linear system in parallel with SLES.  The code indicates the\n\
procedure for using different linear solvers on the individual blocks.\n\n";

/*T
   Concepts: SLES; solving linear equations
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions()
   Routines: SLESSolve(); SLESView()
   Processors: n
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* SLES context */
  SLES    *subsles;     /* array of local SLES contexts on this processor */
  PC      pc;           /* PC context */
  PC      subpc;        /* PC context for subdomain */
  KSP     subksp;       /* KSP context for subdomain */
  PCType  pctype;       /* preconditioning technique */
  double  norm;         /* norm of solution error */
  int       i, j, I, J, ierr, m = 3, n = 2;
  int       rank, size, its, nlocal, first, Istart, Iend,flg;
  Scalar    v, zero = 0.0, one = 1.0, none = -1.0;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);  n = 2*size;

  /* Create and assemble matrix */
  ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,
         0,PETSC_NULL,0,PETSC_NULL,&A); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    v = 4.0; MatSetValues(A,1,&I,1,&I,&v,ADD_VALUES);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create vectors for exact solution, approx solution, and RHS */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* Create SLES context and set operators */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);
         CHKERRA(ierr);

  /* Set default preconditioner */
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetType(pc,PCBJACOBI); CHKERRA(ierr);

  /* Set options */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* Set local solvers for Block Jacobi method.  This code is intended as
     a simple illustration of setting different linear solvers for the
     individual blocks.  These choices are obviously not recommended for
     solving this particular problem. */
  ierr = PCGetType(pc,&pctype,PETSC_NULL); CHKERRA(ierr);
  if (pctype == PCBJACOBI) {
    /* Note that SLESSetUp() MUST be called before PCBJacobiGetSubSLES(). */
    ierr = SLESSetUp(sles,x,b); CHKERRA(ierr);
    ierr = PCBJacobiGetSubSLES(pc,&nlocal,&first,&subsles); CHKERRA(ierr);
    ierr = SLESGetPC(subsles[0],&subpc); CHKERRA(ierr);
    ierr = SLESGetKSP(subsles[0],&subksp); CHKERRA(ierr);
    if (rank == 0) {
      ierr = PCSetType(subpc,PCILU); CHKERRA(ierr);
      ierr = KSPSetTolerances(subksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,
             PETSC_DEFAULT); CHKERRA(ierr);
    } else {
      ierr = PCSetType(subpc,PCJACOBI); CHKERRA(ierr);
      ierr = KSPSetTolerances(subksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,
             PETSC_DEFAULT); CHKERRA(ierr);
    }
  }
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-noslesview",&flg); CHKERRA(ierr);
  if (!flg) {
    ierr = SLESView(sles,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  }

  /* Check the error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* Destroy work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
