From eltgroth@h4p.llnl.gov Fri Apr  4 11:41:45 1997
Received: from poptart.llnl.gov (poptart.llnl.gov [128.115.18.61]) by antares.mcs.anl.gov (8.6.10/8.6.10)  with ESMTP
	id LAA14601 for <bsmith@mcs.anl.gov>; Fri, 4 Apr 1997 11:41:44 -0600
Received: from h4p.llnl.gov (h4p.llnl.gov [134.9.1.6])
          by poptart.llnl.gov (8.8.5/LLNL-3.0) with SMTP
   id JAA18869 for <bsmith@mcs.anl.gov>; Fri, 4 Apr 1997 09:45:18 -0800 (PST)
Received:  by h4p.llnl.gov (5.61/CRI-80.1 )
	id AA08399; Fri, 4 Apr 97 09:45:18 -0800
Date: Fri, 4 Apr 97 09:45:18 -0800
From: Peter Eltgroth <eltgroth@h4p.llnl.gov>
Message-Id: <9704041745.AA08399@h4p.llnl.gov>
To: bsmith@mcs.anl.gov
Subject: pge_test0.c
Status: RO

You may want to rip out the local timing calls ...

#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.2 1996/09/17 20:08:22 balay Exp $";
#endif

static char help[] = "Solves a linear system in parallel with SLES.\n\n";

/*T
   Concepts: SLES (solving linear equations)
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESView(); SLESGetKSP(); SLESGetPC();
   Routines: KSPSetTolerances(); PCSetType();
   Processors: n
 *
 * Code completely rewritten by Eltgroth March 1997 to test
 * 3D linear system solve.
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

int slesex(int,char**);

int main( int argc, char **argv )
{
    MPI_Init( &argc, &argv );
    slesex(argc,argv);
    MPI_Finalize();
    return 0;
}

int slesex(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* linear solver context */
  PC      pc;           /* preconditioner context */
  KSP     ksp;          /* Krylov subspace method context */
  double  norm;         /* norm of solution error */
  int     i, j, I, J, Istart, Iend, ierr, m = 8, n = 7, its, flg;
  int     my_id;
  Scalar  v, one = 1.0, none = -1.0;

#include <sys/times.h>
#include <time.h>

  struct tms before, after;
  clock_t utime, stime, startime, endtime;

  /*
  **  Below makes each processor run independently; thus
  **  not a real parallel test.
  ierr = PetscSetCommWorld(MPI_COMM_SELF); CHKERRA(ierr);
  */

  PetscInitialize(&argc,&args,(char *)0,help);

  MPI_Comm_rank(MPI_COMM_WORLD,&my_id);

  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* TIMING STARTS */
  startime = times(&before);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,m*n,m*n,&A); CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);

  /*
  fprintf(stdout, "Process %d has indices %d to %d\n", my_id, Istart, Iend);
  */

  /* 
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global row and columns of matrix entries.
   */
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    v = 4.0; MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition,
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* 
     Create parallel vectors.
      - When using VecCreate(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime. 
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr); 
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles); CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);

  /* 
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the SLES context,
       we can then directly directly call any KSP and PC routines
       to set various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       SLESSetFromOptions();
  */
  ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  /*
  ierr = PCSetType(pc,PCJACOBI); CHKERRA(ierr);
  */
  ierr = PCSetType(pc,PCNONE); CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,
         PETSC_DEFAULT); CHKERRA(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* TIMING FINISHES */

  endtime = times(&after);

  utime = after.tms_utime - before.tms_utime;
  stime = after.tms_stime - before.tms_stime;

  /*
  printf("\nCPU time used in user space = %f sec or %ld clock ticks\n",
            (float)utime/(float)CLK_TCK, utime);
  printf("CPU time used by the system = %f sec or %ld clock ticks\n",
            (float)stime/(float)CLK_TCK, stime);
  printf("Wall clock time used by process = %f sec ",
            (float)(endtime - startime)/(float)CLK_TCK);
  printf("or %ld clock ticks\n", endtime - startime);
  */

  printf("Wall clock time used by process %d is %f sec.\n", my_id,
            (float)(endtime - startime)/(float)CLK_TCK);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

