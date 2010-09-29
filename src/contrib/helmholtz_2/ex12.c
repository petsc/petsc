#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex12.c,v 1.27 2000/05/13 20:57:34 bsmith Exp $";
#endif

static char help[] = "This parallel code is designed for the solution of linear systems\n\
discretized on a 2D logically rectangular grid.  Currently, we support 1 model problem,\n\
a Helmholtz equation in a half-plane.  Input parameters include:\n\
  -problem <number> : currently only problem #1 supported\n\
  -print_grid : print grid information to stdout\n\
  -print_system : print linear system matrix and vector to stdout\n\
  -print_solution : print solution vector to stdout\n\
  -print_debug : print deugging information\n\
  -modify_submat : activate routine to modify submatrices\n\
  -N_eta <N_eta>, -N_xi <N_xi> : number of processors in eta and xi directions\n\
  -m_eta <m_eta>, -m_xi <m_xi> : number of grid points in eta and xi directions\n\
  -xi_max <xi_max> ; maximum xi value\n\
  -amp <amp> : amplitude of scattered acoustic field\n\
  -mach <mach> : free-stream Mach number\n\
  -k1 <k1> : parameter k1\n\n";

/*T
   Concepts: SLES^Solving a Helmholtz equation (advanced parallel example);
   Concepts: Helmholtz equation
   Concepts: DA^Using distributed arrays;
   Concepts: Complex numbers; 
   Concepts: Matrices^Preallocating matrix memory
   Concepts: Error Handling^Using the macro __FUNC__ to define routine names;
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSetUp(); SLESSetUpOnBlocks();
   Routines: SLESSolve(); SLESView(); SLESGetPC(); SLESGetKSP();
   Routines: KSPSetTolerances(); PCSetModifySubMatrices();
   Routines: MatCreateSeqAIJ(); MatCreateMPIAIJ();
   Routines: DACreate2d(); DADestroy(); DACreateGlobalVector(); DAView();
   Routines: DAGetCorners(); DAGetGhostCorners(); DAGetGlobalIndices();
   Routines: ISCreateGeneral(); ISDestroy(); MatZeroRows();
   Routines: ViewerSetFormat();
T*/

/* 
   Include "da.h" so that we can use distributed arrays (DAs).
   Include "sles.h" so that we can use SLES solvers.  Note that this file
   automatically includes:
     petscsys.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners

   We also include "dfvec.h" so that we can use "discrete function" vectors,
   which enable use to perform various actions that use information about
   vector component ordering.  In particular, we use DFVecView() for
   viewing vectors in parallel using the same ordering of components that
   would be used for 1 processor, regardless of the processor layout.
*/

#include "da.h"
#include "sles.h"
#include "dfvec.h"

/* 
   ---------------------
    Compiling the code:
   ---------------------
     This code uses the complex numbers version of PETSc, so configure must
     be run to set complex numbers

   ----------------------------------
    Code organization and extension:
   ----------------------------------
     This code is designed for the parallel solution of linear systems
     discretized on a 2D logically rectangular grid.  We currently specify a
     single model problem (discussed below); additional linear problems for 2D
     regular grids (with the same stencil) can easily be added by merely
     providing routines (analogous to "FormSystem1") to compute the matrix
     and right-hand-side vector that define each linear system.  To define
     problems with different stencils or multiple degrees of freedom per node,
     the call to DACreate2d() should be modified accordingly.

   ------------------
    Model Problem 1:
   ------------------
       Reference: "Numerical Solution of Periodic Vortical Flows about
       a Thin Airfoil", J. Scott and H. Atassi, AIAA paper 89-1691,
       AIAA 24th Thermophysics Conference, June 12-14, 1989.

       This program was written by David Keyes and Lois Curfman McInnes.

       Relative to this reference, we set k_3 to zero (two-dimensional 
       problem).  Amplitude "amp" is a_2 of the paper.  Unlike the paper, 
       our computational grid is uniform in this initial cut.  (Since this
       results in stretched farfield cells that lose their high frequency
       resolving ability, this will be fixed shortly.)
  
       As noted below, the downstream symmetry boundary has a nonlocal
       condition, which is presently implemented with a dense subcolumn
       of the system matrix.  A subdiagonal, lying within the standard
       stencil pattern, could be used instead, and probably should be used
       for better properties with a level-based fill preconditioner. (This
       is a subject for investigation.)

       We use the eta/xi coordinate system, as described in the above
       reference.  The full grid (including boundary on all sides) is
       given below, so that the global system size is m_xi * m_eta.


           m_xi - 1  |------------|
                     |            |
                     |            |
                     |            |
                   0 --------------
                     0         m_eta - 1


       The bottom boundary maps to the airfoil slit surface in the physical
       plane, and is a closed segment.  The top boundary maps to the farfield
       boundary, and is an open segment.  The lateral boundaries map to the
       upstream and downstream portions of the symmetry boundary in the physical
       plane, and include the "corner" points of the farfield boundary.

       Current formulation:
        - uniform grid
        - standard 2nd-order finite difference discretization in the domain's
          interior, and 1st-order discretization of boundary conditions
        - Improvements are forthcoming, so stay tuned!
 */

/* User-defined application context, named in honor of Hafiz Atassi */
typedef struct {
  int      problem;           /* model problem number */
  int      m_eta, m_xi;       /* global dimensions in eta and xi directions */
  int      m_dim, m_ldim;     /* global, local system size */
  DA       da;                /* distributed array */
  Vec      phi;               /* solution vector */
  MPI_Comm comm;              /* communicator */
  int      rank, size;        /* rank, size of communicator */
  double   xi_max;            /* maximum xi value */
  double   h_eta, h_xi;       /* spacing in eta and xi directions */
  double   mach;              /* Mach number */
  double   amp;               /* parameters used for system evaluation */
  double   pi;                
  double   rh_eta_sq;         
  double   rh_xi_sq; 
  double   k1Dbeta_sq; 
  double   ampDbeta;
  int      print_debug;      /* flag - if 1, print debugging info */
} Atassi;

/* Declare user-defined routines */
int UserMatrixCreate1(Atassi*,Mat*);
int FormSystem1(Atassi*,Mat,Vec);
int UserDetermineMatrixNonzeros(Atassi*,MatType,int**,int**);
int ModifySubmatrices1(PC,int,IS*,IS*,Mat*,void*);

/* 
   User-defined routines.  Note that immediately before each routine below,
   we define the macro __FUNC__ to be a string containing the routine name.
   If defined, this macro is used in the PETSc error handlers to provide a
   complete traceback of routine names.  All PETSc library routines use this
   macro, and users can optionally employ it as well in their application
   codes.  Note that users can get a traceback of PETSc errors regardless of
   whether they define __FUNC__ in application codes; this macro merely
   provides the added traceback detail of the application routine names.
*/

#define sqr(x) ((x)*(x))

#undef __FUNC__  
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     b;            /* solution, RHS vectors */
  Vec     b2, localv;        /* work vectors */
  Mat     A;                 /* linear system matrix */
  SLES    sles;              /* linear solver context */
  PC      pc;                /* preconditioner context */
  KSP     ksp;               /* Krylov subspace context */
  Atassi  user;              /* user-defined work context */
  int     N_eta, N_xi;       /* number of processors in eta and xi directions */
  Scalar  none = -1.0;
  double  k1, beta_sq, norm;
  int     ierr, its, flg;

  /*
      Initialize PETSc
  */
  PetscInitialize(&argc,&args,(char *)0,help);
#if !defined(USE_PETSC_COMPLEX)
  SETERRA(1,0,"This example requires complex numbers");
#endif

  /*
      Set default viewer to cause matrices to be printed in 
     a standard format; independent of the underlying data structure.
  */
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL);CHKERRA(ierr);

  /* 
     Set problem parameters
  */

  user.comm       = PETSC_COMM_WORLD;
  user.problem    = 1;
  user.m_eta      = 7;
  user.m_xi       = 7;
  user.amp        = 1.0;
  k1              = 1.0;
  user.mach       = 0.5;
  user.pi         = 4.0 * atan(1.0);
  user.xi_max     = user.pi/2.0;
  ierr = OptionsGetInt(PETSC_NULL,"-problem",&user.problem,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m_eta",&user.m_eta,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m_xi",&user.m_xi,&flg);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-amp",&user.amp,&flg);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-mach",&user.mach,&flg);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-k1",&k1,&flg);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-xi_max",&user.xi_max,&flg);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-print_debug",&user.print_debug);CHKERRA(ierr);
  user.m_dim      = user.m_eta * user.m_xi;
  user.h_eta      = 1.0/(user.m_eta - 1);
  user.h_xi       = user.xi_max/(user.m_xi - 1);
  user.rh_eta_sq  = 1.0/sqr(user.h_eta);
  user.rh_xi_sq   = 1.0/sqr(user.h_xi);
  beta_sq         = 1 - sqr(user.mach);
  user.k1Dbeta_sq = k1/beta_sq;
  user.ampDbeta   = user.amp/sqrt(beta_sq);

  /* Create distributed array (DA) and vectors */
  MPI_Comm_size(user.comm,&user.size);
  MPI_Comm_rank(user.comm,&user.rank);
  N_xi = PETSC_DECIDE; N_eta = PETSC_DECIDE;
  ierr = OptionsGetInt(PETSC_NULL,"-N_eta",&N_eta,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N_xi",&N_xi,&flg);CHKERRA(ierr);
  if (N_eta*N_xi != user.size && (N_eta != PETSC_DECIDE || N_xi != PETSC_DECIDE)) {
    SETERRA(1,0,"Incompatible number of processors:  N_eta * N_xi != size");
  }

  /* Note: Although the ghost width overlap is 0 for this problem, we need to
     create a DA with width 1, so that each processor generates the local-to-global
     mapping for its neighbors in the north/south/east/west (needed for
     matrix assembly for the 5-point, 2D finite difference stencil). This
     mapping is needed when we determine the global column numbers for
     grid points on a processor edge.
  */
  ierr = DACreate2d(user.comm,DA_NONPERIODIC,DA_STENCIL_STAR,user.m_eta,
                    user.m_xi,N_eta,N_xi,1,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(user.da,&user.phi);CHKERRA(ierr);
  ierr = VecGetLocalSize(user.phi,&user.m_ldim);CHKERRA(ierr);
  ierr = VecDuplicate(user.phi,&b);CHKERRA(ierr);
  ierr = VecDuplicate(user.phi,&b2);CHKERRA(ierr);


  /* 
     Assemble linear system
  */
  switch (user.problem) {
    case 1:
      /* Create matrix data structure */
      ierr = UserMatrixCreate1(&user,&A);CHKERRA(ierr);

      /* Compute matrix and vector that define linear system */
      ierr = FormSystem1(&user,A,b);CHKERRA(ierr); break;
    default:
      SETERRA(1,0,"Only problem #1 currently supported");
  }

  /*
      Create linear solver context; set linear system matrix 
  */
  ierr = SLESCreate(user.comm,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /*
      Set convergence tolerance default; can be overwritten with command-line option 
  */
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* 
     Set routine for modifying submarices that arise in certain preconditioners
      (block Jacobi, ASM, and block Gauss-Seidel)
  */
  ierr = OptionsHasName(PETSC_NULL,"-modify_submat",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
    ierr = PCSetModifySubMatrices(pc,ModifySubmatrices1,(void*)&user);CHKERRA(ierr);
  }

  /*
     Explicitly call SLESSetUp() and SLESSetUpOnBlocks() to
     enable more detailed profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     SLESSolve() if they haven't been called already. 
  */
  ierr = SLESSetUp(sles,b,user.phi);CHKERRA(ierr);
  ierr = SLESSetUpOnBlocks(sles);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,user.phi,&its);CHKERRA(ierr);

  /* 
     View info about linear solver; could use runtime option -sles_view instead 
  */
  ierr = SLESView(sles,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-print_solution",&flg);CHKERRA(ierr);
  if (flg) {
    PetscPrintf(user.comm,"solution vector\n");
    ierr = DFVecView(user.phi,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }

  /*
     Check norm of residual
  */
  ierr = MatMult(A,user.phi,b2);CHKERRA(ierr);
  ierr = VecAXPY(b2,-1.0,b);CHKERRA(ierr);
  ierr  = VecNorm(b2,NORM_2,&norm);CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of RHS difference=%g, Iterations=%d\n",norm,its);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of RHS difference < 1.e-12, Iterations=%d\n",its);

  /*
      Destroy all the PETSc objects created
  */
  ierr = VecDestroy(user.phi);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(b2);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = DACreateLocalVector(user.da,&localv);CHKERRA(ierr);
  ierr = VecDestroy(localv);CHKERRA(ierr);
  ierr = DADestroy(user.da);CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).  See PetscFinalize()
         manpage for more information.
  */
  PetscFinalize();
  return 0;
}
/* -------------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "UserMatrixCreate1"
/*
   UserMatrixCreate1 - Creates matrix data structure, selecting a particular format
   at runtime.  This routine is just a customized version of the generic PETSc
   routine MatCreate() to enable preallocation of matrix memory.  See the manpage
   for MatCreate() for runtime options.

   Input Parameter:
   user - user-defined application context

   Output Parameter:
   mat - newly created matrix

   Notes:
   For now, we consider only the basic PETSc matrix formats: MATSEQAIJ, MATMPIAIJ.
   Additional formats are also available.

   Preallocation of matrix memory is crucial for fast matrix assembly!!
   See the users manual for details.  Use the option -info to print
   info about matrix memory allocation.
 */
int UserMatrixCreate1(Atassi *user,Mat *mat)
{
  Mat        A;
  MatType    mtype;
  int        ierr, *nnz_d, *nnz_o;
  PetscBool  flg;

  ierr = MatGetTypeFromOptions(user->comm,PETSC_NULL,&mtype,&flg);CHKERRQ(ierr);

  /*
     Determine precise nonzero structure of matrix so that we can preallocate
     memory.  We could alternatively use rough estimates of the number of nonzeros
     per row. 
  */
  ierr = UserDetermineMatrixNonzeros(user,mtype,&nnz_d,&nnz_o);CHKERRQ(ierr);

  if (mtype == MATSEQAIJ) {
    ierr = MatCreateSeqAIJ(user->comm,user->m_dim,user->m_dim,PETSC_NULL,nnz_d,&A);CHKERRQ(ierr);
  } 
  else if (mtype == MATMPIAIJ) {
    ierr = MatCreateMPIAIJ(user->comm,user->m_ldim,user->m_ldim,user->m_dim,
                           user->m_dim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&A);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(user->comm,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,user->m_dim,user->m_dim);CHKERRQ(ierr);
  }
  ierr = PetscFree(nnz_d);CHKERRQ(ierr);
  *mat = A;
  return 0;
}
/* -------------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "UserDetermineMatrixNonzeros"
/*
   UserDetermineMatrixNonzeros - Precompute amount of space for matrix preallocation,
   to enable fast matrix assembly without continual dynamic memory allocation.
   This code just mimics the matrix evaluation code to determine nonzero locations.

   Input Parameters:
.  user - user-defined application context
.  mtype - matrix type

   Output Parameters:
.  nnz_d - number of nonzeros in various rows (diagonal part)
.  nnz_o - number of nonzeros in various rows (off-diagonal part)

 */
int UserDetermineMatrixNonzeros(Atassi *user,MatType mtype,int **nz_d,int **nz_o)
{
  int xs, ys, xe, ye;     /* local grid starting/ending values (no ghost points) */
  int xsi, ysi, xei, yei; /* local interior grid starting/ending values (no ghost points) */
  int xm, ym;             /* local grid widths (no ghost points) */
  int Xm, Ym;             /* local grid widths (including ghost points) */
  int Xs, Ys;             /* local ghost point starting values */
  int nloc, *ltog;        /* local-to-global mapping (including ghost points!) */
  int te;                 /* trailing edge point */
  int istart, iend;       /* starting/ending local row numbers */
  int *nnz_d;             /* number of nonzeros (diagonal part of parallel matrix) */
  int *nnz_o;             /* number of nonzeros - (off-diagonal part of parallel matrix) */
  int lrow_g;             /* local row number, including ghost points */
  int lrow_ng;            /* local row number, not including ghost pooints */
  int col[5];             /* work array - column numbers */
  int m_eta = user->m_eta, m_xi = user->m_xi, m_ldim = user->m_ldim;
  int ierr, i, j, m;

  nnz_o = PETSC_NULL; nnz_d = PETSC_NULL;
  if (mtype == MATSEQAIJ) {
    nnz_d = (int *) PetscMalloc(m_ldim * sizeof(int));CHKPTRQ(nnz_d);
    PetscMemzero(nnz_d,m_ldim * sizeof(int));
    istart = 0; iend = m_ldim;
  } else if (mtype == MATMPIAIJ) {
    nnz_d = (int *) PetscMalloc(2*m_ldim * sizeof(int));CHKPTRQ(nnz_d);
    PetscMemzero(nnz_d,2*m_ldim * sizeof(int));
    nnz_o = nnz_d + m_ldim;
    /* Note: vector and matrix distribution is identical */
    ierr = VecGetOwnershipRange(user->phi,&istart,&iend);CHKERRQ(ierr);
  } SETERRQ(PETSC_COMM_SELF,1,0,"UserDetermineMatrixNonzeros: Code not yet written for this type");
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);
  *nz_o = nnz_o; *nz_d = nnz_d;

  /* Get corners and global indices */
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  xe = xs + xm;
  ye = ys + ym;

  /* Define interior grid points (excluding boundary values) */
  if (xe == m_eta) xei = xe - 1;
  else             xei = xe;
  if (ye == m_xi)  yei = ye - 1;
  else             yei = ye;
  if (xs == 0)     xsi = xs + 1;
  else             xsi = xs;
  if (ys == 0)     ysi = ys + 1;
  else             ysi = ys;

  /*
     We build 2 arrays (each of length m_ldim, where m_ldim is the
     number of local rows of the matrix) that specify the number of
     nonzeros for each row of the local submatrix.  Internally, the
     MATMPIAIJ matrix format subdivides ech processor's local 
     submatrix into a 2 parts:
       - diagonal part (square submatrix of the local
            rows and corresponding columns)
       - off-diagonal part (rectangular submatrix of the local
            rows and all other columns)
     We thus specify information for these 2 sections in
     nnz_d and nnz_o, respectively.

     lrow_ng - local row number, not including ghost values,
               used to find the index in the nnz_d nnz_o arrays
     lrow_g  - local row number, including ghost values,
               used in conjunction with the local-to-global
               mapping to determine the global column numbers
               for the parallel grid.  Recall that due to grid
               point reordering with DAs, we must always work
               with the local grid points, and then transform 
               them to the new global numbering with the "ltog"
               mapping (via DAGetGlobalIndices()).  We cannot work
               directly with the global numbers for the original
               uniprocessor grid!
  */

  if (user->problem == 1) {
    /* Interior part of matrix */
    for (j=ysi; j<yei; j++) {
      for (i=xsi; i<xei; i++) {
        lrow_g  = (j-Ys)*Xm + i-Xs; 
        lrow_ng = (j-ys)*xm + i-xs; 
        col[0]  = ltog[lrow_g - Xm];
        col[1]  = ltog[lrow_g - 1];
        col[2]  = ltog[lrow_g];
        col[3]  = ltog[lrow_g + 1];
        col[4]  = ltog[lrow_g + Xm];
        for (m=0; m<5; m++) {
          if (col[m] >= istart && col[m] < iend) nnz_d[lrow_ng]++;
          else                                   nnz_o[lrow_ng]++;
        }
      }
    }

    /* Downstream: i=0 */
    te = 0;   /* Define trailing edge point: te = (0,0) */
              /* Possible alternative: Set to south neighbor instead of te=(0,0) */
    if (xs == 0) {
      for (j=ysi; j<ye; j++) {
        lrow_g  = (j-Ys)*Xm - Xs;
        lrow_ng = (j-ys)*xm - xs;
        col[0]  = ltog[lrow_g];
        col[1]  = te;
        for (m=0; m<2; m++) {
          if (col[m] >= istart && col[m] < iend) nnz_d[lrow_ng]++;
          else                                   nnz_o[lrow_ng]++;
        }
      }
    }

    /* Upstream: i=xe-1 */
    if (xe == m_eta) {
      for (j=ysi; j<ye; j++) {
        lrow_g  = (j-Ys)*Xm + xe-1-Xs;
        lrow_ng = (j-ys)*xm + xe-1-xs;
        col[0]  = ltog[lrow_g];
        if (col[0] >= istart && col[0] < iend) nnz_d[lrow_ng]++;
        else                                   nnz_o[lrow_ng]++;
      }
    }

    /* Airfoil slit: j=0 */
    if (ys == 0) {
      for (i=xs; i<xe; i++) {
        lrow_g  = -Ys*Xm + i-Xs; 
        lrow_ng = -ys*xm + i-xs; 
        col[0]  = ltog[lrow_g];
        col[1]  = ltog[lrow_g + Xm];
        for (m=0; m<2; m++) {
          if (col[m] >= istart && col[m] < iend) nnz_d[lrow_ng]++;
          else                                   nnz_o[lrow_ng]++;
        }
      }
    }

    /* Farfield: j=ye-1 */
    j = ye-1;
    if (ye == m_xi) {
      for (i=xsi; i<xei; i++) {
        lrow_g  = (ye-1-Ys)*Xm + i-Xs; 
        lrow_ng = (ye-1-ys)*xm + i-xs; 
        col[0]  = ltog[lrow_g - 2*Xm];
        col[1]  = ltog[lrow_g - Xm];
        col[2]  = ltog[lrow_g];
        for (m=0; m<3; m++) {
          if (col[m] >= istart && col[m] < iend) nnz_d[lrow_ng]++;
          else                                   nnz_o[lrow_ng]++;
        }
      }
    }
  } else SETERRQ(PETSC_COMM_SELF,1,0,"UserDetermineMatrixNonzeros: Only problem 1 has been coded so far!");

  return 0;
  }
/* -------------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "FormSystem1"
/*
   FormSystem1 - Evaluates matrix and vector for Helmholtz model problem #1,
   described above.

   Input Parameters:
.  user - user-defined application context
.  A - matrix data structure
.  b - right-hand-side vector data structure

   Output Parameters:
.  A - fully assembled matrix
.  b - fully assembled right-hand-side vector

   Current formulation:
    - uniform grid
    - mapped problem domain
    - standard 2nd-order finite difference discretization in the domain's
      interior, and 1st-order discretization of boundary conditions
    - Future improvements in the problem formulation are forthcoming;
      stay tuned!

   Notes:
   Due to grid point reordering with DAs, we must always work
   with the local grid points, and then transform them to the new
   global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
   We cannot work directly with the global numbers for the original
   uniprocessor grid!

   See manpage for MatAssemblyEnd() for runtime options, such as
   -mat_view_draw to draw nonzero structure of matrix.
 */
int FormSystem1(Atassi *user,Mat A,Vec b)
{
  int     te;                 /* trailing edge point */
  int     xs, ys, xe, ye;     /* local grid starting/ending values (no ghost points) */
  int     xsi, ysi, xei, yei; /* local interior grid starting/ending values 
                                 (no ghost points) */
  int     xm, ym;             /* local grid widths (no ghost points) */
  int     Xm, Ym;             /* local grid widths (including ghost points) */
  int     Xs, Ys;             /* local ghost point starting values */
  int     nloc, *ltog;        /* local-to-global mapping (including ghost points!) */
  int     lrow;               /* local row number, including ghost points, used
                                 in conjunction with ltog mapping */
  int     grow;               /* global row number */
  int     col[5];             /* work array to stash global column numbers */
  int     flg, ierr, i, j;
  int     m_eta = user->m_eta, m_xi = user->m_xi;
  double  pi = user->pi, mach = user->mach, h_xi = user->h_xi, h_eta = user->h_eta;
  double  k1Dbeta_sq = user->k1Dbeta_sq, ampDbeta = user->ampDbeta, rh_xi;
  double  rh_eta_sq = user->rh_eta_sq, rh_xi_sq = user->rh_xi_sq, one = 1.0, two = 2.0;
  Scalar  zero = 0.0, val, c2, c1, c, v[5];

  rh_xi = 1.0/h_xi;

  /* Get corners and global indices */
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  xe = xs + xm;
  ye = ys + ym;

  ierr = OptionsHasName(PETSC_NULL,"-print_grid",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = DAView(user->da,VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    PetscPrintf(user->comm,"global grid: %d X %d ==> global vector dimension %d\n",
      user->m_eta,user->m_xi,user->m_dim); fflush(stdout);
    PetscSequentialPhaseBegin(user->comm,1);
    printf("[%d] local grid %d X %d ==> local vector dimension %d\n",user->rank,xm,ym,user->m_ldim);
    fflush(stdout);
    PetscSequentialPhaseEnd(user->comm,1);
  }

  /* Define interior grid points (excluding boundary values) */
  if (xe == m_eta) xei = xe - 1;
  else             xei = xe;
  if (ye == m_xi)  yei = ye - 1;
  else             yei = ye;
  if (xs == 0)     xsi = xs + 1;
  else             xsi = xs;
  if (ys == 0)     ysi = ys + 1;
  else             ysi = ys;

  /* Evaluate interior part of matrix */
  c = sqr(pi * mach * k1Dbeta_sq);
  for (j=ysi; j<yei; j++) {
    for (i=xsi; i<xei; i++) {
      lrow   = (j-Ys)*Xm + i-Xs; 
      grow   = ltog[lrow];
      col[0] = ltog[lrow - Xm];
      col[1] = ltog[lrow - 1];
      col[2] = grow;
      col[3] = ltog[lrow + 1];
      col[4] = ltog[lrow + Xm];
      v[0]   = rh_xi_sq;
      v[1]   = rh_eta_sq;
      v[2]   = -two * (rh_eta_sq + rh_xi_sq) +
                c*(sqr(sin(pi*i*h_eta)) + sqr(sinh(pi*j*h_xi)) );
      v[3]   = rh_eta_sq;
      v[4]   = rh_xi_sq;
      ierr = MatSetValues(A,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValues(b,1,&grow,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Evaluate matrix and vector components for grid edges */

  /* Downstream: i=0 */
  te = 0;   /* Define trailing edge point: te = (0,0) */
            /* Possible alternative: Set to south neighbor instead of te=(0,0) */
  if (xs == 0) {
    for (j=ysi; j<ye; j++) {
      lrow   = (j-Ys)*Xm - Xs;
      grow   = ltog[lrow];
      col[0] = grow;
      col[1] = te;
      v[0]   = -one;
      v[1]   = exp(PETSC_i * k1Dbeta_sq * (cosh(pi*j*h_xi) - one));
      ierr   = MatSetValues(A,1,&grow,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr   = VecSetValues(b,1,&grow,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Upstream: i=xe-1 */
  if (xe == m_eta) {
    for (j=ysi; j<ye; j++) {
      lrow = (j-Ys)*Xm + xe-1-Xs;
      grow = ltog[lrow];
      v[0] = -one;
      ierr = MatSetValues(A,1,&grow,1,&grow,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValues(b,1,&grow,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Airfoil slit: j=0 */
  if (ys == 0) {
    for (i=xs; i<xe; i++) {
      lrow   = -Ys*Xm + i-Xs; 
      grow   = ltog[lrow];
      col[0] = grow;
      col[1] = ltog[lrow + Xm];
      v[0]   = -rh_xi;
      v[1]   = rh_xi;
      val    = -pi*ampDbeta * sin(pi*i*h_eta) *
                exp(PETSC_i * k1Dbeta_sq * cos(pi*i*h_eta));
      ierr   = MatSetValues(A,1,&grow,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr   = VecSetValues(b,1,&grow,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Farfield: j=ye-1 */
  j = ye-1;
  if (ye == m_xi) {
    for (i=xsi; i<xei; i++) {
      c2 = rh_xi_sq * cos(pi*i*h_eta) / sqr(pi* sin(pi*i*h_eta) * cosh(pi*j*h_xi));
      c1 = rh_xi * k1Dbeta_sq * (mach * cos(pi*i*h_eta) + one) /
          (pi* sin(pi*i*h_eta) * cosh(pi*j*h_xi));
      lrow   = (ye-1-Ys)*Xm + i-Xs; 
      grow   = ltog[lrow];
      col[0] = ltog[lrow - 2*Xm];
      col[1] = ltog[lrow - Xm];
      col[2] = ltog[lrow];
      v[0]   = c2;
      v[1]   = -2.0*c2 + PETSC_i * c1;
      v[2]   = c2 - PETSC_i * c1 - sqr(k1Dbeta_sq)*mach;
      /* v[2]   = c2 - PETSC_i * c1 - sqr(k1Dbeta_sq*mach); */
      ierr   = MatSetValues(A,1,&grow,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr   = VecSetValues(b,1,&grow,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-print_system",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DFVecView(b,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  return 0;
}
/* -------------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "ModifySubmatrices1"
/*
   ModifySubmatrices1 - Modifies the submatrices that arise in certain
   preconditioners (block Jacobi, ASM) (for
   example, to set alternative boundary conditions for the subdomains).
   This routine is set by calling PCSetModifySubMatrices() within the
   main program.

   Input Parameters:
.  pc - the preconditioner context
.  nsub - the number of local submatrices
.  row - an array of index sets that contain the global row numbers
         that comprise each local submatrix
.  col - an array of index sets that contain the global column numbers
         that comprise each local submatrix
.  submat - array of local submatrices
.  ctx - optional user-defined context for private data for the 
         user-defined func routine (may be null)

   Output Parameter:
.  submat - array of local submatrices (the entries of which may
            have been modified)

   Notes:
   The basic submatrices are extracted from the preconditioner matrix as usual;
   this routine allows the user to modify the entries within the local submatrices
   as needed in a particular application before they are used for the local solves.

   This code is intended merely to demonstrate the process of modifying
   the local submatrices; these particular choices for matrix entries are
   NOT representative of what would be appropriate in Helmholtz problems
   (or any particular problems, for that matter).
*/
int ModifySubmatrices1(PC pc,int nsub,IS *row,IS *col,Mat *submat,void *dummy)
{
  Atassi *user = (Atassi*)dummy;
  int    i, ierr, m, n, lrow, lcol;
  IS     is;
  Scalar one = 1.0, val;

  /* Note that one can refer to any data within the user-defined context,
    as set by the call to PCSetModifySubMatrices() in the main program. */
  if (user->print_debug){
    PetscPrintf(user->comm,"grid spacing: h_eta = %g, h_xi = %g\n",user->h_eta,user->h_xi);
  }

  /* 
     Loop over local submatrices
  */
  for (i=0; i<nsub; i++) {
    ierr = MatGetSize(submat[i],&m,&n);CHKERRQ(ierr);
    if (user->print_debug) {
      printf("[%d] changing submatrix %d of %d local submatrices: dimension %d X %d\n",
              user->rank,i+1,nsub,m,n);
    }
    if (m) m--;

    /* 
       Create an index set to define certain rows numbers that we then set
       to zero. Note that each processor creates its own local index set using
       the communicator PETSC_COMM_SELF.
    */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,1,&m,&is);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(submat[i],is,&one);CHKERRQ(ierr);
    ierr = ISDestroy(is);CHKERRQ(ierr);

    /*
       Reassemble the submatrix 
    */
    lrow = 1; lcol = 1; val = 0.5;
    ierr = MatSetValues(submat[i],1,&lrow,1,&lcol,&val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(submat[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submat[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  return 0;
}
