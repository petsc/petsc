#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex8.c,v 1.36 1999/10/13 20:38:22 bsmith Exp bsmith $";
#endif

static char help[] = "Illustrates use of the preconditioner ASM (Additive\n\
Schwarz Method) for solving a linear system in parallel with SLES.  The\n\
code indicates the procedure for setting user-defined subdomains.  Input\n\
parameters include:\n\
  -user_set_subdomain_solvers:  User explicitly sets subdomain solvers\n\
  -user_set_subdomains:  Activate user-defined subdomains\n\n";

/*
   Note:  This example focuses on setting the subdomains for the ASM 
   preconditioner for a problem on a 2D rectangular grid.  See ex1.c
   and ex2.c for more detailed comments on the basic usage of SLES
   (including working with matrices and vectors).

   The ASM preconditioner is fully parallel, but currently the routine
   PCASMCreateSubDomains2D(), which is used in this example to demonstrate
   user-defined subdomains (activated via -user_set_subdomains), is
   uniprocessor only.

   This matrix in this linear system arises from the discretized Laplacian,
   and thus is not very interesting in terms of experimenting with variants
   of the ASM preconditioner.  
*/

/*T
   Concepts: SLES^Using the Additive Schwarz Method (ASM) with user-defined subdomains
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions(); SLESSolve();
   Routines: PCSetType(); PCASMCreateSubdomains2D(); PCASMSetLocalSubdomains();
   Routines: PCASMSetOverlap(); PCASMGetSubSLES();
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

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     x, b, u;                 /* approx solution, RHS, exact solution */
  Mat     A;                       /* linear system matrix */
  SLES    sles;                    /* linear solver context */
  PC      pc;                      /* PC context */
  IS      *is;                     /* array of index sets that define the subdomains */
  int     overlap = 1;             /* width of subdomain overlap */
  int     Nsub;                    /* number of subdomains */
  int     user_subdomains;         /* flag - 1 indicates user-defined subdomains */
  int     m = 15, n = 17;          /* mesh dimensions in x- and y- directions */
  int     M = 2, N = 1;            /* number of subdomains in x- and y- directions */
  int     i, j, its, I, J, ierr, Istart, Iend, size, flg;
  Scalar  v,  one = 1.0;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-overlap",&overlap,&flg);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-user_set_subdomains",&user_subdomains);CHKERRA(ierr);

  /* -------------------------------------------------------------------
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     ------------------------------------------------------------------- */

  /* 
     Assemble the matrix for the five point stencil, YET AGAIN 
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
    if ( i<m-1 ) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
    if ( j>0 )   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
    if ( j<n-1 ) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* 
     Create and set vectors 
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&b);CHKERRA(ierr);
  ierr = VecSetFromOptions(b);CHKERRA(ierr);
  ierr = VecDuplicate(b,&u);CHKERRA(ierr);
  ierr = VecDuplicate(b,&x);CHKERRA(ierr);
  ierr = VecSet(&one,u);CHKERRA(ierr);
  ierr = MatMult(A,u,b);CHKERRA(ierr);

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
     Set the default preconditioner for this program to be ASM
  */
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCASM);CHKERRA(ierr);

  /* -------------------------------------------------------------------
                  Define the problem decomposition
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       Basic method, should be sufficient for the needs of many users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Set the overlap, using the default PETSc decomposition via
         PCASMSetOverlap(pc,overlap);
     Could instead use the option -pc_asm_overlap <ovl> 

     Set the total number of blocks via -pc_asm_blocks <blks>
     Note:  The ASM default is to use 1 block per processor.  To
     experiment on a single processor with various overlaps, you
     must specify use of multiple blocks!
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       More advanced method, setting user-defined subdomains
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Firstly, create index sets that define the subdomains.  The utility
     routine PCASMCreateSubdomains2D() is a simple example (that currently
     supports 1 processor only!).  More generally, the user should write
     a custom routine for a particular problem geometry.

     Then call either PCASMSetLocalSubdomains() or PCASMSetTotalSubdomains()
     to set the subdomains for the ASM preconditioner.
  */

  if (!user_subdomains) { /* basic version */
    ierr = PCASMSetOverlap(pc,overlap);CHKERRA(ierr);
  } else { /* advanced version */
    if (size != 1) SETERRA(1,0,
      "PCASMCreateSubdomains() is currently a uniprocessor routine only!");
    ierr = PCASMCreateSubdomains2D(m,n,M,N,1,overlap,&Nsub,&is);CHKERRA(ierr);
    ierr = PCASMSetLocalSubdomains(pc,Nsub,is);CHKERRA(ierr);
  }

  /* -------------------------------------------------------------------
                Set the linear solvers for the subblocks
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       Basic method, should be sufficient for the needs of most users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     By default, the ASM preconditioner uses the same solver on each
     block of the problem.  To set the same solver options on all blocks,
     use the prefix -sub before the usual PC and KSP options, e.g.,
          -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Advanced method, setting different solvers for various blocks.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Note that each block's SLES context is completely independent of
     the others, and the full range of uniprocessor SLES options is
     available for each block.

     - Use PCASMGetSubSLES() to extract the array of SLES contexts for
       the local blocks.
     - See ex7.c for a simple example of setting different linear solvers
       for the individual blocks for the block Jacobi method (which is
       equivalent to the ASM method with zero overlap).
  */

  ierr = OptionsHasName(PETSC_NULL,"-user_set_subdomain_solvers",&flg);CHKERRA(ierr);
  if (flg) {
    SLES       *subsles;       /* array of SLES contexts for local subblocks */
    int        nlocal, first;  /* number of local subblocks, first local subblock */
    KSP        subksp;         /* KSP context for subblock */
    PC         subpc;          /* PC context for subblock */
    PetscTruth isasm;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"User explicitly sets subdomain solvers.\n");CHKERRA(ierr);

    /* 
       Set runtime options
    */
    ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

    /* 
       Flag an error if PCTYPE is changed from the runtime options
     */
    ierr = PetscTypeCompare((PetscObject)pc,PCASM,&isasm);CHKERRA(ierr);
    if (isasm) {
      SETERRA(1,0,"Cannot Change the PCTYPE when manually changing the subdomain solver settings");
    }
    /* 
       Call SLESSetUp() to set the block Jacobi data structures (including
       creation of an internal SLES context for each block).

       Note: SLESSetUp() MUST be called before PCASMGetSubSLES().
    */
    ierr = SLESSetUp(sles,x,b);CHKERRA(ierr);

    /*
       Extract the array of SLES contexts for the local blocks
    */
    ierr = PCASMGetSubSLES(pc,&nlocal,&first,&subsles);CHKERRA(ierr);

    /*
       Loop over the local blocks, setting various SLES options
       for each block.  
    */
    for (i=0; i<nlocal; i++) {
      ierr = SLESGetPC(subsles[i],&subpc);CHKERRA(ierr);
      ierr = SLESGetKSP(subsles[i],&subksp);CHKERRA(ierr);
      ierr = PCSetType(subpc,PCILU);CHKERRA(ierr);
      ierr = KSPSetType(subksp,KSPGMRES);CHKERRA(ierr);
      ierr = KSPSetTolerances(subksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);
    }
  } else {
    /* 
       Set runtime options
    */
    ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  }

  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */

  if (user_subdomains) {
    for ( i=0; i<Nsub; i++ ) {
      ierr = ISDestroy(is[i]);CHKERRA(ierr);
    }
    ierr = PetscFree(is);CHKERRA(ierr);
  }
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
