
/***************************************************************************

  Application Description: 3D Euler Flow

  This particular file contains the driver program and some support routines
  for the PETSc interface to the Julianne code for the 3D Euler model.
  The problem domain is a logically rectangular C-grid, where there are
  five degrees of freedom per node (corresponding to density, vector momentum,
  and internal energy).  The standard, 7-point 3D stencil is used via finite
  volumes.

  The driver program and most PETSc interface routines are written in C,
  while the application code, as modified only slightly from the original
  Julianne program, remains in Fortran.

  We use the PETSc nonlinear solvers (SNES) with pseudo-transient continuation
  to solve the nonlinear system.  We parallelize the grid using the PETSc
  distributed arrays (DAs).

  This code supports two variants for treating boundary conditions:
    - explicitly - only interior grid points are part of Newton systems;
                   boundary conditions are applied explicitly after
                   each Newton update
    - implicitly - both interior and boundary grid points contribute to
                   the Newton systems
  The code can also interface to the original (uniprocessor) Julianne solver
  that uses explicit boundary conditions.

 ***************************************************************************/

static char help[] = "This program solves a 3D Euler problem, using either the\n\
original Julianne code's sequential solver or the parallel PETSc nonlinear solvers.\n\
Runtime options include:\n\
  -Nx <nx> -Ny <ny> -Nz <nz> : Number of processors in the x-, y-, z-directions\n\
  -problem <1,2,3,4>         : 1(50x10x10 grid), 2(98x18x18 grid), 3(194x34x34 grid),\n\
                               4(data structure test)\n\
  -matrix_free               : Use matrix-free Newton-Krylov method\n\
  -snes_mf_err <err>         : Choose differencing parameter for SNES matrix-free method\n\
  -post                      : Print post-processing info (currently uniproc version only)\n\
  -angle <angle_in_degrees>  : angle of attack (default is 3.06 degrees)\n\
  -jfreq <it>                : frequency of forming Jacobian (once every <it> iterations)\n\
  -implicit                  : use fully implicit formulation of boundary conditions\n\
  -cfl_advance               : use advancing CFL number\n\
  -f_red <fraction>          : reduce the function norm by this fraction before advancing CFL\n\
  -no_output                 : do not print any output during SNES solve (intended for use\n\
                               during timing runs)\n\n";

static char help2[] = "Options for Julianne solver:\n\
  -julianne                  : Use original Julianne solver (uniprocessor only)\n\
  -julianne_rtol [rtol]      : Set convergence tolerance for Julianne solver\n\n";

static char help3[] = "Options for VRML viewing:\n\
  -vrmlevenhue               : Use uniform colors for VRML output\n\
  -vrmlnolod                 : Do not use LOD for VRML output\n\
  -dump_freq                 : Frequency for dumping output (default is every 10 iterations)\n\
  -dump_vrml_layers [num]    : Dump pressure contours for [num] layers\n\
  -dump_vrml_cut_y           : Dump pressure contours, slicing in y-direction (around the wing),\n\
                               instead of the default slices in the z-direction (through the wing)\n\
  -dump_vrml_different_files : Dump VRML output into a file (corresponding to iteration number)\n\
                               rather than the default of dumping continually into a single file\n\
  -dump_general              : Dump various fields into files (euler.[iteration].out) for\n\
                               later processing\n\n";

static char help4[] = "Options for debugging and matrix dumping:\n\
  -printg                    : Print grid information\n\
  -mat_dump -cfl_switch [cfl]: Dump linear system corresponding to this CFL in binary to file\n\
                               'euler.dat'; then exit.\n\
These shouldn't be needed anymore but were useful in writing the parallel code\n\
  -printv                    : Print various vectors\n\
  -debug                     : Activate debugging printouts (dump matices, index sets, etc.).\n\
                               Since this option dumps lots of data, it should be used for just\n\
                               a few iterations.\n\
  -bctest                    : Test scatters for boundary conditions\n\n";

/***************************************************************************/

/* user-defined include file for the Euler application */
#include "user.h"
#include "src/fortran/custom/zpetsc.h"

int main(int argc,char **argv)
{
  MPI_Comm comm;                  /* communicator */
  SNES     snes;                  /* SNES context */
  SLES     sles;                  /* SLES context */
  KSP      ksp;                   /* KSP context */
  Euler    *app;                  /* user-defined context */
  Viewer   view;                  /* viewer for printing vectors */
  /* char     stagename[2][16]; */     /* names of profiling stages */
  int      fort_app;              /* Fortran interface user context */
  int      log;                   /* flag indicating use of logging */
  int      solve_with_julianne;   /* flag indicating use of original Julianne solver */
  /* int      init1, init2, init3; */   /* event numbers for application initialization */
  int      its, ierr, flg, stage;

  /* Set Defaults */
  int      maxksp = 25;           /* max number of KSP iterations */
  int      post_process = 0;      /* flag indicating post-processing */
  int      total_stages = 1;      /* number of times to run nonlinear solver */
  int      log_stage_0 = 0;       /* are we doing dummy solve for logging stage 0? */
  Scalar   ksprtol = 1.0e-2;      /* KSP relative convergence tolerance */
  double   rtol = 1.e-12;         /* SNES relative convergence tolerance */
  double   time1, tsolve;         /* time for solution process */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize PETSc and print help information
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = MPI_COMM_WORLD;
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRA(ierr);
  if (flg) {PetscPrintf(comm,help2);PetscPrintf(comm,help3);PetscPrintf(comm,help4);}

  /* -----------------------------------------------------------
                  Beginning of nonlinear solver loop
     ----------------------------------------------------------- */
  /* 
     If using -log_summary to monitor performance, then loop through the
     nonlinear solve 2 times.  
      - The intention here is to preload and solve a small system;
        then load another (larger) system and solve it as well.
        This process preloads the instructions with the smaller
        system so that more accurate performance monitoring (via
        -log_summary) can be done with the larger one (that actually
        is the system of interest). 
     See the "Performance" chapter of the PETSc users manual for details.

     If not using -log_summary, then we just do 1 nonlinear solve.
  */

  ierr = OptionsHasName(PETSC_NULL,"-log_summary",&log); CHKERRA(ierr);

  /* 
      temporarily deactivate user-defined events and multiple logging stages
  */
  /*
  if (log) total_stages = 2;
  ierr = PLogEventRegister(&init1,"DA, Scatter Init","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&init2,"Mesh Setup      ","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&init3,"Julianne Init   ","Red:"); CHKERRA(ierr);
  */
  for ( stage=0; stage<total_stages; stage++ ) {
  
  /*
     Begin profiling next stage
  */
  /*
  PLogStagePush(stage);
  sprintf(stagename[stage],"Solve %d",stage);
  PLogStageRegister(stage,stagename[stage]);
  if (log && !stage) log_stage_0 = 1;
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize distributed arrays and vector scatters; allocate work space
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsHasName(PETSC_NULL,"-julianne",&solve_with_julianne); CHKERRA(ierr);
  /* PLogEventBegin(init1,0,0,0,0); */
  ierr = UserCreateEuler(comm,solve_with_julianne,log_stage_0,&app); CHKERRA(ierr);
  /* PLogEventEnd(init1,0,0,0,0); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Read mesh and convert to parallel version
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* PLogEventBegin(init2,0,0,0,0); */
  ierr = UserSetGrid(app); CHKERRA(ierr); 
  /* PLogEventEnd(init2,0,0,0,0); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     If desired, use original (uniprocessor) solver in Julianne code
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* for dummy logging stage 1, form the Jacobian every 2 iterations */
  /* if (log && stage==0) app->jfreq = 2; */

  if (solve_with_julianne) {
    if (app->size != 1) SETERRA(1,1,"Original code runs on 1 processor only.");
    ierr = OptionsGetDouble(PETSC_NULL,"-julianne_rtol",&rtol,&flg); CHKERRA(ierr);
    /* PLogEventBegin(init3,0,0,0,0); */
    time1 = PetscGetTime();
    ierr = julianne_(&solve_with_julianne,0,&app->cfl,
           &rtol,app->b1,app->b2,
           app->b3,app->b4,app->b5,app->b6,app->diag,app->dt,
           app->r,app->ru,app->rv,app->rw,app->e,app->p,
           app->dr,app->dru,app->drv,app->drw,app->de,
           app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,app->xc,app->yc,app->zc,
           app->f1,app->g1,app->h1,
           app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,
           &app->angle,&app->jfreq); CHKERRA(ierr);
    tsolve = PetscGetTime() - time1;
    /* PLogEventEnd(init3,0,0,0,0); */
    PetscPrintf(comm,"Julianne solution time = %g seconds\n",tsolve);
    if (app->dump_general) {ierr = MonitorDumpGeneralJulianne(app); CHKERRA(ierr);}
    PetscFinalize();
    return 0;
  } 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Otherwise, initialize application data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Read entire grid and initialize data */
  /* PLogEventBegin(init2,0,0,0,0); */
  fort_app = PetscFromPointer(app);
  ierr = julianne_(&solve_with_julianne,&fort_app,&app->cfl,
         &rtol,app->b1,
         app->b2,app->b3,app->b4,app->b5,app->b6,app->diag,app->dt,
         app->r,app->ru,app->rv,app->rw,app->e,app->p,
         app->dr,app->dru,app->drv,app->drw,app->de,
         app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
         app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
         app->aiz,app->ajz,app->akz,app->xc,app->yc,app->zc,
         app->f1,app->g1,app->h1,
         app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,
         &app->angle,&app->jfreq); CHKERRA(ierr);
 /* PLogEventEnd(init2,0,0,0,0); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(comm,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);
  ierr = SNESSetType(snes,SNES_EQ_LS); CHKERRA(ierr);
  ierr = SNESSetLineSearch(snes,SNESNoLineSearch); CHKERRA(ierr); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set various routines
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = UserSetJacobian(snes,app); CHKERRA(ierr);
  ierr = SNESSetFunction(snes,app->F,ComputeFunction,app); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,JulianneMonitor,app); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Set default options; these must precede SNESSetFromOptions() so that
     they can be overridden at runtime if desired */

  ierr = SNESGetSLES(snes,&sles); CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);
  ierr = KSPSetType(ksp,KSPGMRES); CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,ksprtol,PETSC_DEFAULT,PETSC_DEFAULT,maxksp); CHKERRA(ierr);
  ierr = KSPGMRESSetRestart(ksp,maxksp+1); CHKERRA(ierr);

  /* Set runtime options (e.g. -snes_rtol <rtol> -ksp_type <type>) */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* We use just a few iterations if doing the "dummy" logging phase 0 */
  if (log && stage==0) {ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,
                                   PETSC_DEFAULT,3,PETSC_DEFAULT); CHKERRA(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = InitialGuess(snes,app,app->X); CHKERRA(ierr);

  /* Transfer application data to vector X and view if desired */
  if (app->print_vecs) {
    ierr = ViewerFileOpenASCII(comm,"init.out",&view); CHKERRA(ierr);
    ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = DFVecView(app->X,view); CHKERRA(ierr);
    ierr = ViewerDestroy(view); CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESSolve(snes,app->X,&its); CHKERRA(ierr);
  PetscPrintf(comm,"number of Newton iterations = %d\n\n",its);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Do post-processing
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Calculate physical quantities of interest */
  ierr = OptionsHasName(PETSC_NULL,"-post",&post_process); CHKERRA(ierr);
  if (post_process && app->size == 1) {
    SETERRA(1,1,"Pvar post processing is temporarily disabled.\n");
    /* ierr = pvar_(app->r,app->ru,app->rv,app->rw,app->e,app->p,
       app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
       app->aiz,app->ajz,app->akz,app->x,app->y,app->z); CHKERRA(ierr);
       */
  }

  ierr = OptionsHasName(PETSC_NULL,"-tecplot",&flg); CHKERRA(ierr);
  if (flg) {
    ierr = TECPLOTMonitor(snes,app->X,app); CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free data structures 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = UserDestroyEuler(app); CHKERRA(ierr);

  /* Conclude profiling this stage */
  /* PLogStagePop();
  MPI_Barrier(comm); */

  /* -----------------------------------------------------------
                      End of nonlinear solver loop
     ----------------------------------------------------------- */
  }

  PetscFinalize();
  return 0;
}
/***************************************************************************/
/*
   UserSetJacobian - Forms Jacobian matrix context and sets Jacobian
   evaluation routine.  We also compute space for preallocation of
   matrix memory.

   Input Parameters:
   snes - SNES context
   app - application-defined context

   ----------------
    General Notes:
   ----------------
    Although it is not required to preallocate matrix memory, this step
    is crucial for efficient matrix assembly.  See the "Matrices" chapter
    of the PETSc users manual for detailed information on this topic.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserSetJacobian(SNES snes,Euler *app)
{
  MatType  mtype = MATSEQAIJ;      /* matrix format */
  MPI_Comm comm = app->comm;      /* comunicator */
  Mat      J;                      /* Jacobian matrix context */
  int      ldim = app->ldim;	   /* local dimension of vectors and matrix */
  int      gdim = app->gdim;	   /* global dimension of vectors and matrix */
  int      nc = app->nc;	   /* size of matrix blocks (DoF per node) */
  int      istart, iend;           /* range of locally owned matrix rows */
  int      *nnz_d = 0, *nnz_o = 0; /* arrays for preallocating matrix memory */
  int      wkdim;                  /* dimension of nnz_d and nnz_o */
  int      ierr, flg;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     First, compute amount of space for matrix preallocation, to enable
     fast matrix assembly without continual dynamic memory allocation.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatGetTypeFromOptions(comm,PETSC_NULL,&mtype,&flg); CHKERRQ(ierr);

  /* Row-based matrix formats */
  if (mtype == MATSEQAIJ || mtype == MATMPIAIJ || mtype == MATMPIROWBS)
    wkdim = app->ldim;
  else if (mtype == MATSEQBAIJ || mtype == MATMPIBAIJ) /* block row formats */
    wkdim = app->lbkdim;
  else SETERRQ(1,1,"UserSetJacobian:Matrix format not currently supported.");

  /* Allocate work arrays */
  nnz_d = (int *)PetscMalloc(2*wkdim * sizeof(int)); CHKPTRQ(nnz_d);
  PetscMemzero(nnz_d,2*wkdim * sizeof(int));
  nnz_o = nnz_d + wkdim;

  /* Note that vector and matrix partitionings are the same (see note below) */
  ierr = VecGetOwnershipRange(app->X,&istart,&iend); CHKERRQ(ierr);

  /* We mimic the matrix assembly code to determine precise locations 
     of nonzero matrix entries */
  ierr = nzmat_(&mtype,&nc,&istart,&iend,app->is1,app->ltog,&app->nloc,
                &wkdim,nnz_d,nnz_o); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Form Jacobian matrix data structure
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note:  For the parallel case, vectors and matrices MUST be partitioned
     accordingly.  When using distributed arrays (DAs) to create vectors,
     the DAs determine the problem partitioning.  We must explicitly
     specify the local matrix dimensions upon its creation for compatibility
     with the vector distribution.  Thus, the generic MatCreate() routine
     is NOT sufficient when working with distributed arrays.
  */

  if (mtype == MATSEQAIJ) {
    /* Rough estimate of nonzeros per row is:  nd * nc = 7 * 5 = 35 */
    /* ierr = MatCreateSeqAIJ(comm,gdim,gdim,nd*nc,PETSC_NULL,&J); CHKERRQ(ierr); */
       ierr = MatCreateSeqAIJ(comm,gdim,gdim,PETSC_NULL,nnz_d,&J); CHKERRQ(ierr);
  } 
  else if (mtype == MATMPIAIJ) {
    ierr = MatCreateMPIAIJ(comm,ldim,ldim,gdim,
                           gdim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&J); CHKERRQ(ierr);
  } 
  else if (mtype == MATMPIROWBS) {
    ierr = MatCreateMPIRowbs(comm,ldim,gdim,PETSC_NULL,nnz_d,
                             PETSC_NULL,&J); CHKERRQ(ierr);
  }
  else if (mtype == MATSEQBAIJ) {
    /* Rough estimate of block nonzeros per row is:  # of diagonals, nd */
    /* ierr = MatCreateSeqBAIJ(comm,nc,gdim,gdim,nd,PETSC_NULL,&J); CHKERRQ(ierr); */
    ierr = MatCreateSeqBAIJ(comm,nc,gdim,gdim,PETSC_NULL,nnz_d,&J); CHKERRQ(ierr);
  } 
  else if (mtype == MATMPIBAIJ) {
    ierr = MatCreateMPIBAIJ(comm,nc,ldim,ldim,
           gdim,gdim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&J); CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"UserSetJacobian:Matrix format not currently supported.");
  }
  if (nnz_d) PetscFree(nnz_d);
  app->J = J;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Set data structures and routine for Jacobian evaluation 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsHasName(PETSC_NULL,"-matrix_free",&app->matrix_free); CHKERRQ(ierr);
  if (!app->matrix_free) {
    /* Use explicit (approx) Jacobian to define Newton system and preconditioner */
    ierr = SNESSetJacobian(snes,J,J,ComputeJacobian,app); CHKERRQ(ierr);
    PetscPrintf(comm,"Not using matrix-free KSP method\n"); 
  } else {
    /* Use matrix-free Jacobian to define Newton system; use explicit (approx)
       Jacobian for preconditioner */
   double err_rel;
   if (app->bctype != IMPLICIT) SETERRQ(1,1,"UserSetJacobian: Matrix-free method requires implicit BCs!");
   ierr = UserMatrixFreeMatCreate(snes,app,app->X,&app->Jmf); CHKERRQ(ierr);
   ierr = SNESSetJacobian(snes,app->Jmf,J,ComputeJacobian,app); CHKERRQ(ierr);
   ierr = OptionsGetDouble(PETSC_NULL,"-snes_mf_err",&err_rel,&flg); CHKERRQ(ierr);
   PetscPrintf(comm,"Using matrix-free KSP method (snes_mf_err = %g)\n",err_rel);
  }
  return 0;
}
/***************************************************************************/
/* 
    UserDestroyEuler - Destroys the user-defined application context.

   Input Parameter:
   app - application-defined context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserDestroyEuler(Euler *app)
{
  int ierr;
  if (app->J) {ierr = MatDestroy(app->J); CHKERRQ(ierr);}
  if (app->matrix_free) {ierr = MatDestroy(app->Jmf); CHKERRQ(ierr);}
  if (app->Fvrml) {ierr = VecDestroy(app->Fvrml); CHKERRQ(ierr);}
  ierr = VecDestroy(app->X); CHKERRQ(ierr);
  ierr = VecDestroy(app->Xbc); CHKERRQ(ierr);
  ierr = VecDestroy(app->F); CHKERRQ(ierr);
  ierr = VecDestroy(app->localX); CHKERRQ(ierr);
  ierr = VecDestroy(app->localDX); CHKERRQ(ierr);
  ierr = DADestroy(app->da); CHKERRQ(ierr);
  ierr = VecScatterDestroy(app->Xbcscatter); CHKERRQ(ierr);
  PetscFree(app->label);
  if (app->is1) PetscFree(app->is1);
  if (app->bctype != IMPLICIT || app->dump_vrml || app->dump_general) {
    ierr = VecScatterDestroy(app->Pbcscatter); CHKERRQ(ierr);
    ierr = DADestroy(app->da1); CHKERRQ(ierr);
    ierr = VecDestroy(app->Pbc); CHKERRQ(ierr);
    ierr = VecDestroy(app->P); CHKERRQ(ierr);
    ierr = VecDestroy(app->localP); CHKERRQ(ierr);
  } else {
    if (!app->mat_assemble_direct) PetscFree(app->b1bc);
  }
  if (app->bctype == IMPLICIT) {
    PetscFree(app->fbcri1); PetscFree(app->fbcrj1); PetscFree(app->fbcrk1);
  }

  /* Free misc work space for Fortran arrays */
  if (app->dr)      PetscFree(app->dr);
  if (app->dt)      PetscFree(app->dt);
  if (app->diag)    PetscFree(app->diag);
  if (app->b1)      PetscFree(app->b1);
  if (app->work_p)  PetscFree(app->work_p);
  if (app->f1)      PetscFree(app->f1);
  if (app->sp)      PetscFree(app->sp);
  if (app->sadai)   PetscFree(app->sadai);
  if (app->bl)      PetscFree(app->bl);
  if (app->xc)      PetscFree(app->xc);

  /* Close Fortran file */
  if (!app->no_output) cleanup_();

  PetscFree(app);
  return 0;
}
/***************************************************************************/
/* 
   ComputeJacobian - Computes Jacobian matrix.  The user sets this routine
   by calling SNESSetJacobian().

   Input Parameters:
   X     - input vector
   jac   - Jacobian matrix
   pjac  - preconditioner matrix (same as Jacobian matrix except when
           we use matrix-free Newton-Krylov methods)
   flag  - flag indicating information about the preconditioner matrix
           structure.  See SLESSetOperators() for important information 
           about setting this flag.
   ptr   - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
   pjac  - fully assembled preconditioner matrix
   flag  - flag indicating information about the preconditioner matrix structure

  -----------------------
   Notes for Euler code:
  -----------------------
   This routine supports two appropaches for matrix assembly:
     (1) Assembling the Jacobian directly into the PETSc matrix data 
         structure (the default approach -- more efficient)
     (2) Forming the matrix in the original Eagle data structures and
         converting these later to asssemble a PETSc matrix.  This approach
         is expensive in terms of both memory and time and is retained only
         as a demonstration of how to quickly revise an existing code that
         uses the Eagle format.

   This routine supports two modes of handling boundary conditions:
     (1) explicitly - only interior grid points are part of Newton systems
     (2) implicitly - both interior and boundary grid points contribute to
                      the Newton systems
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int ComputeJacobian(SNES snes,Vec X,Mat *jac,Mat *pjac,MatStructure *flag,void *ptr)
{
  Euler *app = (Euler *)ptr;
  MatType type;                /* matrix format */
  int     skip;	               /* if (skip) then retain the previous Jacobian */
  int     iter;                /* nonlinear solver iteration number */
  int     fortmat, flg, ierr;

  ierr = PetscCObjectToFortranObject(*pjac,&fortmat); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);

  /* Convert vector.  If using explicit boundary conditions, this passes along
     any changes in X due to their application to the component arrays in the
     Julianne code */
  ierr = UnpackWork(app,app->r,app->ru,app->rv,app->rw,app->e,X); CHKERRQ(ierr);

  /* Form Jacobian matrix */
  if (app->bctype == IMPLICIT) {

    /* Either assemble the matrix directly (the more efficient route) ... */
    if (app->mat_assemble_direct) {
      /* We must zero the diagonal block here, since this is not done within jformdt2 */
      PetscMemzero(app->diag,app->diag_len);
      ierr = jformdt2_(&skip,&app->epsbc,app->ltog,&app->nloc,&fortmat,app->is1,
              app->b1bc,app->b2bc,app->b3bc,app->b2bc_tmp,app->diag,
              app->dt,app->r,app->ru,app->rv,app->rw,app->e,app->p,
              app->r_bc,app->ru_bc,app->rv_bc,app->rw_bc,app->e_bc,app->p_bc,
              app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
              app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
              app->aiz,app->ajz,app->akz,app->f1,app->g1,app->h1,
              app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,&iter,
	   app->fbcri1, app->fbcrui1, app->fbcrvi1, app->fbcrwi1, app->fbcei1,
           app->fbcri2, app->fbcrui2, app->fbcrvi2, app->fbcrwi2, app->fbcei2,
           app->fbcrj1, app->fbcruj1, app->fbcrvj1, app->fbcrwj1, app->fbcej1,
           app->fbcrj2, app->fbcruj2, app->fbcrvj2, app->fbcrwj2, app->fbcej2,
           app->fbcrk1, app->fbcruk1, app->fbcrvk1, app->fbcrwk1, app->fbcek1,
           app->fbcrk2, app->fbcruk2, app->fbcrvk2, app->fbcrwk2, app->fbcek2); CHKERRQ(ierr);

    /* Or store the matrix in the intermediate Eagle format for later conversion ... */
    } else {
      ierr = jformdt_(&skip,&app->epsbc,app->b1,app->b2,app->b3,app->b4,app->b5,app->b6,
              app->b1bc,app->b2bc,app->b3bc,app->b2bc_tmp,
              app->diag,app->dt,app->r,app->ru,app->rv,app->rw,app->e,app->p,
              app->r_bc,app->ru_bc,app->rv_bc,app->rw_bc,app->e_bc,app->p_bc,
              app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
              app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
              app->aiz,app->ajz,app->akz,app->f1,app->g1,app->h1,
              app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,
	   app->fbcri1, app->fbcrui1, app->fbcrvi1, app->fbcrwi1, app->fbcei1,
           app->fbcri2, app->fbcrui2, app->fbcrvi2, app->fbcrwi2, app->fbcei2,
           app->fbcrj1, app->fbcruj1, app->fbcrvj1, app->fbcrwj1, app->fbcej1,
           app->fbcrj2, app->fbcruj2, app->fbcrvj2, app->fbcrwj2, app->fbcej2,
           app->fbcrk1, app->fbcruk1, app->fbcrvk1, app->fbcrwk1, app->fbcek1,
           app->fbcrk2, app->fbcruk2, app->fbcrvk2, app->fbcrwk2, app->fbcek2); CHKERRQ(ierr);
    }
 } else {
    /* Either assemble the matrix directly (the more efficient route) ... */
    if (app->mat_assemble_direct) {
      /* We must zero the diagonal block here, since this is not done within jform2 */
      PetscMemzero(app->diag,app->diag_len);
      ierr = jform2_(&skip,app->ltog,&app->nloc,&fortmat,app->diag,
              app->dt,app->r,app->ru,app->rv,app->rw,app->e,app->p,
              app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
              app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
              app->aiz,app->ajz,app->akz,app->f1,app->g1,app->h1,
              app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2); CHKERRQ(ierr);

    /* Or store the matrix in the intermediate Eagle format for later conversion ... */
    } else {
      ierr = jform_(&skip,app->b1,app->b2,app->b3,app->b4,app->b5,app->b6,
              app->diag,app->dt,app->r,app->ru,app->rv,app->rw,app->e,app->p,
              app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
              app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
              app->aiz,app->ajz,app->akz,app->f1,app->g1,app->h1,
              app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2); CHKERRQ(ierr);
    }
  }

  /* For better efficiency, we recompute the Jacobian matrix only every few 
     nonlinear iterations (as set by the option -jfreq).
     The Fortran matrix formation routine sets skip=1 if we  If we do not recompute the Jacobian matrix, then indicate this to the
     nonlinear solver so that the current preconditioner will be retained */
  if (skip) {
    *flag = SAME_PRECONDITIONER;
    return 0;
  }

  /* Convert Jacobian from Eagle format if direct assembly hasn't been done */
  if (!app->mat_assemble_direct) {
    if (!app->no_output) PetscPrintf(app->comm,"Building PETSc matrix ...\n");
    ierr = MatGetType(*pjac,&type,PETSC_NULL); CHKERRQ(ierr);
    ierr = buildmat_(&fortmat,&app->sctype,app->is1,&iter,
         app->b1,app->b2,app->b3,app->b4,app->b5,app->b6,app->diag,
         app->dt,app->ltog,&app->nloc,
         app->b1bc,app->b2bc,app->b3bc,app->b2bc_tmp); CHKERRQ(ierr);
  } 

  /* Finish the matrix assembly process.  For the Euler code, the matrix
     assembly is done completely locally, so no message-pasing is performed
     during these phases. */
  ierr = MatAssemblyBegin(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Indicate that the preconditioner matrix has the same nonzero
     structure each time it is formed */
  *flag = SAME_NONZERO_PATTERN;

  /* View matrix (for debugging only) */
  if (app->print_debug) {
    /*
    char filename[64]; Viewer view; MatType mtype;
    ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
    sprintf(filename,"mat.%d.out",iter);
    ierr = ViewerFileOpenASCII(app->comm,filename,&view); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = MatGetType(*pjac,&mtype,PETSC_NULL); CHKERRQ(ierr);
    if (mtype == MATMPIAIJ) {ierr = MatViewDFVec_MPIAIJ(*pjac,X,view); CHKERRQ(ierr);}
    else                    {ierr = MatView(*pjac,view); CHKERRQ(ierr);}
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
    */
    /*    PetscFinalize(); exit(0); */
  }

  /* Dump Jacobian and residual in binary format to file euler.dat 
     (for use in separate experiments with linear system solves) */
  ierr = OptionsHasName(PETSC_NULL,"-mat_dump",&flg); CHKERRQ(ierr);
  if (flg && app->cfl_switch <= app->cfl) {
    Viewer viewer;
    PetscPrintf(app->comm,"writing matrix in binary to euler.dat ...\n"); 
    ierr = ViewerFileOpenBinary(app->comm,"euler.dat",BINARY_CREATE,&viewer); 
           CHKERRQ(ierr);
    ierr = MatView(*pjac,viewer); CHKERRQ(ierr);

    ierr = ComputeFunction(snes,X,app->F,ptr); CHKERRQ(ierr);
    PetscPrintf(app->comm,"writing vector in binary to euler.dat ...\n"); 
    ierr = VecView(app->F,viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    PetscFinalize(); exit(0);
  }

  /* Check matrix-vector products, run with -matrix_free and -debug option */
  if (app->matrix_free && app->print_debug) {
    int    loc, i, m, j ,k, ijkx, jkx, ijkxl, jkxl, *ltog, dc[5];
    Vec    y1, y2, x1;
    Viewer view2;
    Scalar *y1a, *y2a, one = 1.0, di, diff, md[5];

    ierr = DAGetGlobalIndices(app->da,&loc,&ltog); CHKERRQ(ierr);
    ierr = VecDuplicate(X,&y1); CHKERRQ(ierr);
    ierr = VecDuplicate(X,&x1); CHKERRQ(ierr);
    ierr = VecDuplicate(X,&y2); CHKERRQ(ierr);
    for (k=app->zs; k<app->ze; k++) {
      for (j=app->ys; j<app->ye; j++) {
        jkx  = j*app->mx + k*app->mx*app->my;
        jkxl = (j-app->gys)*app->gxm + (k-app->gzs)*app->gxm*app->gym;
        for (i=app->xs; i<app->xe; i++) {
          ijkx  = (jkx + i)*app->nc;
          ijkxl = (jkxl + i-app->gxs)*app->nc;
          for (m=0;m<app->nc;m++) {
            di = one*ijkx;
            loc = ltog[ijkxl];
            ierr = VecSetValues(x1,1,&loc,&di,INSERT_VALUES); CHKERRQ(ierr);
         printf("[%d] k=%d, j=%d, i=%d, ijkx=%d, ijkxl=%d\n",app->rank,k,j,i,ijkx,ijkxl);
            ijkx++; ijkxl++;
          }
        }
      }
    }
    ierr = VecAssemblyBegin(x1); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x1); CHKERRQ(ierr);
    ierr = ViewerFileOpenASCII(app->comm,"x1.out",&view2); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = DFVecView(x1,view2); CHKERRQ(ierr);
    ierr = ViewerDestroy(view2); CHKERRQ(ierr);

    ierr = MatMult(*pjac,x1,y1); CHKERRQ(ierr);
    ierr = ViewerFileOpenASCII(app->comm,"v1.out",&view2); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = DFVecView(y1,view2); CHKERRQ(ierr);
    ierr = ViewerDestroy(view2); CHKERRQ(ierr);

    ierr = MatMult(*jac,x1,y2); CHKERRQ(ierr);
    ierr = ViewerFileOpenASCII(app->comm,"v2.out",&view2); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = DFVecView(y2,view2); CHKERRQ(ierr);
    ierr = ViewerDestroy(view2); CHKERRQ(ierr);

    ierr = VecGetArray(y1,&y1a); CHKERRQ(ierr);
    ierr = VecGetArray(y2,&y2a); CHKERRQ(ierr);

    for (m=0;m<app->nc;m++) {
      dc[m] = 0;
      md[m] = 0;
      for (k=app->zs; k<app->ze; k++) {
        for (j=app->ys; j<app->ye; j++) {
          jkx = (j-app->ys)*app->xm + (k-app->zs)*app->xm*app->ym;
          for (i=app->xs; i<app->xe; i++) {
            ijkx = (jkx + i-app->xs)*app->nc + m;
            diff = (PetscAbsScalar(y1a[ijkx])-PetscAbsScalar(y2a[ijkx])) /
                  PetscAbsScalar(y1a[ijkx]);
            if (diff > 0.1) {
              printf("k=%d, j=%d, i=%d, m=%d, ijkx=%d,     diff=%6.3e       y1=%6.3e,   y2=%6.3e\n",
                      k,j,i,m,ijkx,diff,y1a[ijkx],y2a[ijkx]);
              if (diff > md[m]) md[m] = diff;
              dc[m]++;
            }
	  }
        }
      }
    }
    printf("[%d] maxdiff = %g, %g, %g, %g, %g\n\
    dcount = %d, %d, %d, %d, %d\n",
           app->rank,md[0],md[1],md[2],md[3],md[4],dc[0],dc[1],dc[2],dc[3],dc[4]);

    ierr = VecDestroy(x1); CHKERRQ(ierr);
    ierr = VecDestroy(y1); CHKERRQ(ierr);
    ierr = VecDestroy(y2); CHKERRQ(ierr);

    PetscFinalize(); exit(0);
  }

  if (!app->no_output) PetscPrintf(app->comm,"Done building PETSc matrix.\n");
  return 0;
}
/***************************************************************************/
/*
   ComputeFunction - Evaluates the nonlinear function, F(X).  

   Input Parameters:
   snes - the SNES context
   X    - input vector
   ptr  - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
   F    - fully assembled vector containing the nonlinear function

  -----------------------
   Notes for Euler code:
  -----------------------
   Correspondence between SNES vectors and Julianne work arrays is: 
      X    : (dr,dru,drv,drw,de) in Julianne code
      F(X) : (r,ru,rv,rw,e) in Julianne code
   We pack/unpack these work arrays with the routines PackWork()
   and UnpackWork().

   This routine supports two modes of handling boundary conditions:
     (1) explicitly - only interior grid points are part of Newton systems
     (2) implicitly - both interior and boundary grid points contribute to
                      the Newton systems
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int ComputeFunction(SNES snes,Vec X,Vec F, void *ptr)
{
  Euler *app = (Euler *)ptr;
  int     ierr, base_unit, fortvec, iter;
  Scalar  zero = 0.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Do setup (not required for the first function evaluation)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->first_time_resid) {
    app->first_time_resid = 0;
  } else { 
    /* Convert current iterate to Julianne format */
    ierr = PackWork(app,X,app->localX,
                    app->r,app->ru,app->rv,app->rw,app->e); CHKERRQ(ierr);

    /* Compute pressures */
    ierr = jpressure_(app->r,app->ru,app->rv,app->rw,app->e,app->p); CHKERRQ(ierr);

    /* Apply boundary conditions for explicit case, or do the scatters needed
       for the implicit formulation */
    if (app->bctype != IMPLICIT) {
      ierr = BoundaryConditionsExplicit(app,X); CHKERRQ(ierr);
    } else {
      ierr = BoundaryConditionsImplicit(app,X); CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Compute nonlinear function in Julianne work arrays
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* If using implicit BC's, compute function components for boundary grid points */
  if (app->bctype == IMPLICIT) {
    ierr = residbc_(app->r,app->ru,app->rv,app->rw,app->e,app->p,
           app->r_bc,app->ru_bc,app->rv_bc,app->rw_bc,app->e_bc,app->p_bc,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,
           app->fbcri1, app->fbcrui1, app->fbcrvi1, app->fbcrwi1, app->fbcei1,
           app->fbcri2, app->fbcrui2, app->fbcrvi2, app->fbcrwi2, app->fbcei2,
           app->fbcrj1, app->fbcruj1, app->fbcrvj1, app->fbcrwj1, app->fbcej1,
           app->fbcrj2, app->fbcruj2, app->fbcrvj2, app->fbcrwj2, app->fbcej2,
           app->fbcrk1, app->fbcruk1, app->fbcrvk1, app->fbcrwk1, app->fbcek1,
           app->fbcrk2, app->fbcruk2, app->fbcrvk2, app->fbcrwk2, app->fbcek2); CHKERRQ(ierr);
  }

  /* Compute function components for interior grid points. */
  ierr = resid_(app->r,app->ru,app->rv,app->rw,app->e,app->p,
         app->dr,app->dru,app->drv,app->drw,app->de,
         app->br,app->bl,app->be,
         app->sadai,app->sadaj,app->sadak,
         app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
         app->aiz,app->ajz,app->akz,
         app->f1,app->g1,app->h1,
         app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2); CHKERRQ(ierr);

  /* Compute pseudo-transient continuation array, dt.  Really need to
     recalculate dt only when the iterates change.  */
  if (app->sctype == DT_MULT && !app->matrix_free_mult) {
    eigenv_(app->dt,app->r,app->ru,app->rv,app->rw,app->e,app->p,
         app->sadai,app->sadaj,app->sadak,
         app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
         app->aiz,app->ajz,app->akz);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Assemble vector F(X) by converting from Julianne work arrays 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->bctype == IMPLICIT || app->bctype == IMPLICIT_SIZE) {

    /* Initialize vector to zero */
    ierr = VecSet(&zero,F); CHKERRQ(ierr);

    /* Transform F to a Fortran vector */
    ierr = PetscCObjectToFortranObject(F,&fortvec); CHKERRQ(ierr);

    /* Build F(X) */
    ierr = rbuild_(&fortvec, &app->sctype, app->dt, app->dr, app->dru, app->drv,
           app->drw, app->de, app->ltog, &app->nloc,
	   app->fbcri1, app->fbcrui1, app->fbcrvi1, app->fbcrwi1, app->fbcei1,
           app->fbcri2, app->fbcrui2, app->fbcrvi2, app->fbcrwi2, app->fbcei2,
           app->fbcrj1, app->fbcruj1, app->fbcrvj1, app->fbcrwj1, app->fbcej1,
           app->fbcrj2, app->fbcruj2, app->fbcrvj2, app->fbcrwj2, app->fbcej2,
           app->fbcrk1, app->fbcruk1, app->fbcrvk1, app->fbcrwk1, app->fbcek1,
           app->fbcrk2, app->fbcruk2, app->fbcrvk2, app->fbcrwk2, app->fbcek2); CHKERRQ(ierr);

  } else if (app->bctype == EXPLICIT) {
    /* Scale if necessary; then build Function */
    if (app->sctype == DT_MULT) {
      ierr = rscale_(app->dt,app->dr,app->dru,app->drv,app->drw,app->de); CHKERRQ(ierr);
    }
    ierr = UnpackWork(app,app->dr,app->dru,app->drv,app->drw,app->de,F); CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
          Optional output for debugging and visualizing solution 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Output various fields into a general-format file for later viewing */
  if (app->dump_general) {
    ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
    if (!(iter%app->dump_freq)) ierr = MonitorDumpGeneral(snes,X,app);
  }

  /* Output various fields directly into a VRML-format file */
  if (app->dump_vrml) {
    ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
    if (!(iter%app->dump_freq)) ierr = MonitorDumpVRML(snes,X,F,app);
    MonitorDumpIter(iter);
  }

  if (app->print_debug) { /* Print Fortran arrays */
    base_unit = 60;
    printgjul_(app->dr,app->dru,app->drv,app->drw,app->de,app->p,&base_unit);
    base_unit = 70;
    printgjul_(app->r,app->ru,app->rv,app->rw,app->e,app->p,&base_unit);
  }

  return 0;
}
/***************************************************************************/
/*
   InitialGuess - Copies initial guess to vector X from work arrays
                  in Julianne code.

   Input Parameters:
   snes - SNES context
   app  - user-defined application context
   X    - current iterate
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int InitialGuess(SNES snes,Euler *app,Vec X)
{
  int ierr;

  /* Convert Fortran work arrays to vector X */
  ierr = UnpackWork(app,app->r,app->ru,app->rv,app->rw,app->e,X); CHKERRQ(ierr);

  /* If testing scatters for boundary conditions, then replace actual
     initial values with a test vector. */
  if (app->bc_test) {
    ierr = GridTest(app); CHKERRQ(ierr);
    ierr = PackWork(app,X,app->localX,
                    app->r,app->ru,app->rv,app->rw,app->e); CHKERRQ(ierr);
  }

  /* Apply boundary conditions */
  ierr = BoundaryConditionsExplicit(app,X); CHKERRQ(ierr);

  /* Destroy pressure scatters for boundary conditions, since we needed
     them only to computate the initial guess */
  if (app->bctype == IMPLICIT && !app->dump_vrml && !app->dump_general) {
    ierr = VecScatterDestroy(app->Pbcscatter); CHKERRQ(ierr);
    ierr = DADestroy(app->da1); CHKERRQ(ierr);
    ierr = VecDestroy(app->Pbc); CHKERRQ(ierr);
    ierr = VecDestroy(app->P); CHKERRQ(ierr);
    ierr = VecDestroy(app->localP); CHKERRQ(ierr);
  }

  /* Exit if we're just testing scatters for boundary conditions */
  if (app->bc_test) {UserDestroyEuler(app); PetscFinalize(); exit(0);}

  return 0;
}
/***************************************************************************/
/* 
   UserCreateEuler - Defines the application-specific data structure and
   initializes the PETSc distributed arrays.

   Input Parameters:
   comm - MPI communicator
   solve_with_julianne
   log_stage_0 - are we doing the initial dummy setup for log_summary?

   Output Parameter:
   newapp - user-defined application context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserCreateEuler(MPI_Comm comm,int solve_with_julianne,int log_stage_0,Euler **newapp)
{
  Euler *app;
  int   ni1;    	 /* x-direction grid dimension */
  int   nj1;	         /* y-direction grid dimension */
  int   nk1;	         /* z-direction grid dimension */
  int   Nx, Ny, Nz;   /* number of procs in each direction */
  int   Nlocal, ierr, flg, llen, llenb, fort_comm, problem = 1, nc;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Create user-defined application context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  app = PetscNew(Euler); CHKPTRQ(app);
  PetscMemzero(app,sizeof(Euler));
  app->comm = comm;
  MPI_Comm_size(comm,&app->size);
  MPI_Comm_rank(comm,&app->rank);
  if (solve_with_julianne && app->size != 1)
    SETERRQ(1,1,"UserCreateEuler: Julianne solver is uniprocessor only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Set problem parameters and flags 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsGetInt(PETSC_NULL,"-problem",&problem,&flg); CHKERRQ(ierr);
  /* force use of problem 1 for dummy logging stage 0 */
  /* if (log_stage_0) problem = 1; */
  switch (problem) {
    case 1:
    /* full grid dimensions, including all boundaries */
      ni1 = 50; nj1 = 10; nk1 = 10;
    /* wing points, used to define BC scatters.  These are analogs
       in C of Fortran points in input (shifted by -2 for explicit formulation) 
       from m6c: Fortran: itl=10, itu=40, ile=25, ktip=6 */
      app->ktip = 4; app->itl = 8; app->itu = 38; app->ile = 23;   
      app->epsbc = 1.e-7;
      break;
    case 2:
      /* from m6f: Fortran: itl=19, itu=79, ile=49, ktip=11 */
      ni1 = 98; nj1 = 18; nk1 = 18;
      app->ktip = 9; app->itl = 17; app->itu = 77; app->ile = 47;   
      app->epsbc = 1.e-7;
      break;
    case 3:
      /* from m6n: Fortran: itl=37, itu=157, ile=97, ktip=21 */
      ni1 = 194; nj1 = 34; nk1 = 34;
      app->ktip = 19; app->itl = 35; app->itu = 155; app->ile = 95;   
      app->epsbc = 1.e-7;
      break;
    case 4:
      /* test case for PETSc grid manipulations only! */
      ni1 = 4; nj1 = 5; nk1 = 6;
      app->ktip = 0; app->itl = 0; app->itu = 0; app->ile = 0;
      ierr = OptionsGetInt(PETSC_NULL,"-ni1",&ni1,&flg); CHKERRQ(ierr);
      ierr = OptionsGetInt(PETSC_NULL,"-nj1",&nj1,&flg); CHKERRQ(ierr);
      ierr = OptionsGetInt(PETSC_NULL,"-nk1",&nk1,&flg); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(1,1,"UserCreateEuler:Unsupported problem, only 1,2,3 or 4 supported");
  }

  /* Set various defaults */
  app->cfl                 = 0;        /* Initial CFL is set within Julianne code */
  app->cfl_switch          = 10.0;     /* CFL at which to dump binary linear system */
  if (problem == 1)
    app->f_reduction       = 0.5;      /* fnorm reduction ratio before beginning to advance CFL */
  else
    app->f_reduction       = 0.3;
  app->cfl_max             = 100000.0; /* maximum CFL value */
  app->cfl_advance         = 0;        /* flag - by default we don't advance CFL */
  app->angle               = 3.06;     /* default angle of attack = 3.06 degrees */
  app->mat_assemble_direct = 1;        /* by default, we assemble Jacobian directly */
  app->jfreq               = 10;       /* default frequency of computing Jacobian matrix */
  app->no_output           = 0;        /* flag - by default print some output as program runs */

  /* Override default with runtime options */
  ierr = OptionsHasName(PETSC_NULL,"-cfl_advance",&app->cfl_advance); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_switch",&app->cfl_switch,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_max",&app->cfl_max,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-f_red",&app->f_reduction,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-epsbc",&app->epsbc,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-angle",&app->angle,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-jfreq",&app->jfreq,&flg); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-no_output",&app->no_output); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_assemble_last",&flg); CHKERRQ(ierr);
  if (flg) app->mat_assemble_direct = 0;
  if (app->mat_assemble_direct) PetscPrintf(comm,"Problem %d (%dx%dx%d grid), assembling PETSc matrix directly: angle of attack = %g, epsbc = %g\n",problem,ni1,nj1,nk1,app->angle,app->epsbc);
  else PetscPrintf(comm,"Problem %d (%dx%dx%d grid), ssembling PETSc matrix via translation of Eagle format: angle of attack = %g, epsbc = %g\n",problem,ni1,nj1,nk1,app->angle,app->epsbc);

  app->ni1              = ni1;
  app->nj1              = nj1;
  app->nk1              = nk1;
  app->ni               = ni1 - 1;
  app->nj               = nj1 - 1;
  app->nk               = nk1 - 1;
  app->nim              = ni1 - 2;
  app->njm              = nj1 - 2;
  app->nkm              = nk1 - 2;
  app->nc	        = 5;
  app->nd	        = 7;
  app->matrix_free      = 0;
  app->matrix_free_mult = 0;
  app->first_time_resid = 1;
  app->J                = 0;
  app->Jmf              = 0;
  app->Fvrml            = 0;
  nc = app->nc;

  if (app->dump_vrml) dump_angle_vrml(app->angle);

  /* Set problem type of formulation */
  ierr = OptionsHasName(PETSC_NULL,"-dt_mult",&flg); CHKERRQ(ierr);
  if (flg == 1) {
    app->sctype = DT_MULT;
    PetscPrintf(comm,"Pseudo-transiant variant: Multiply system by dt\n");
  } else {
    PetscPrintf(comm,"Pseudo-transiant variant: Regular use of dt\n");
    app->sctype = DT_DIV;
  }
  ierr = OptionsHasName(PETSC_NULL,"-implicit",&flg); CHKERRQ(ierr);
  if (flg == 1) {
    app->bctype = IMPLICIT;
    app->ktip++; app->itl++; app->itu++; app->ile++;
    app->mx = app->ni1, app->my = app->nj1, app->mz = app->nk1;
    PetscPrintf(comm,"Using fully implicit formulation: mx=%d, my=%d, mz=%d\n",
      app->mx,app->my,app->mz);
  } else { /* interior grid points only */
    ierr = OptionsHasName(PETSC_NULL,"-implicit_size",&flg); CHKERRQ(ierr);
    if (flg == 1) {
      app->bctype = IMPLICIT_SIZE;
      app->ktip++; app->itl++; app->itu++; app->ile++;
      app->mx = app->ni1, app->my = app->nj1, app->mz = app->nk1;
      PetscPrintf(comm,"Using default implicit/explicit formulation with implicit system size: mx=%d, my=%d, mz=%d\n",
        app->mx,app->my,app->mz);
    } else {
      app->bctype = EXPLICIT;
      app->mx = app->nim, app->my = app->njm, app->mz = app->nkm;
      PetscPrintf(comm,"Using default implicit/explicit formulation: mx=%d, my=%d, mz=%d\n",
        app->mx,app->my,app->mz);
    }
  }

  /* Monitoring information */
  app->label = (char **) PetscMalloc(nc*sizeof(char*)); CHKPTRQ(app->label);
  app->label[0] = "Density";
  app->label[1] = "Velocity: x-component";
  app->label[2] = "Velocity: y-component";
  app->label[3] = "Velocity: z-component";
  app->label[4] = "Internal Energy";

  /* Set various debugging flags */
  ierr = OptionsHasName(PETSC_NULL,"-printv",&flg); CHKERRQ(ierr);
  if (flg) {
    app->print_vecs = 1; PetscPrintf(comm,"Printing vectors\n");
  } else app->print_vecs = 0;
  ierr = OptionsHasName(PETSC_NULL,"-debug",&flg); CHKERRQ(ierr);
  if (flg) {
    app->print_debug = 1; PetscPrintf(comm,"Activating debugging printouts\n");
  } else app->print_debug = 0;
  ierr = OptionsHasName(PETSC_NULL,"-printg",&flg); CHKERRQ(ierr);
  if (flg) app->print_grid = 1; else app->print_grid = 0;
  ierr = OptionsHasName(PETSC_NULL,"-bctest",&flg); CHKERRQ(ierr);
  if (flg) app->bc_test = 1; else app->bc_test = 0;
  ierr = OptionsHasName(PETSC_NULL,"-dump_general",&app->dump_general); CHKERRQ(ierr);
  if (app->dump_general) PetscPrintf(comm,"Dumping fields for general viewing\n");

  /* various VRML options */

  /* for backward compatibility still support -dump_vrml (with default being pressure field) */
  ierr = OptionsHasName(PETSC_NULL,"-dump_vrml",&app->dump_vrml); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-dump_vrml_residual",&app->dump_vrml_residual); CHKERRQ(ierr);
  if (app->dump_vrml_residual) {
    PetscPrintf(comm,"Dumping residual for VRML viewing\n");
    app->dump_vrml = 1;
  }
  ierr = OptionsHasName(PETSC_NULL,"-dump_vrml_pressure",&app->dump_vrml_pressure); CHKERRQ(ierr);
  if (app->dump_vrml_pressure) {
    PetscPrintf(comm,"Dumping pressure for VRML viewing\n");
    app->dump_vrml = 1;
  }
  if (app->dump_vrml && !app->dump_vrml_residual && !app->dump_vrml_pressure) {
    app->dump_vrml_pressure = 1;  /* set default VRML output */
    PetscPrintf(comm,"Dumping pressure for VRML viewing\n");
  }
  app->dump_freq = 10;
  ierr = OptionsGetInt(PETSC_NULL,"-dump_freq",&app->dump_freq,&flg); CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create DA and associated vectors 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Create initial distributed array and set processor configuration */
  Nx=PETSC_DECIDE; Ny=PETSC_DECIDE; Nz=PETSC_DECIDE;
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg); CHKERRQ(ierr);
  /* really need better test here */
  if (Nx*Ny*Nz != app->size && (Nx != PETSC_DECIDE && Ny != PETSC_DECIDE && Nz != PETSC_DECIDE))
    SETERRQ(1,1,"Incompatible number of processors:  Nx * Ny * Nz != size");
  app->Nx = Nx; app->Ny = Ny; app->Nz = Nz;
  ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,app->mx,app->my,app->mz,
         app->Nx,app->Ny,app->Nz,nc,2,&app->da); CHKERRQ(ierr);

  /* Get global and local vectors */
  ierr = DAGetDistributedVector(app->da,&app->X); CHKERRQ(ierr);
  ierr = VecDuplicate(app->X,&app->Xbc); CHKERRQ(ierr);
  ierr = VecDuplicate(app->X,&app->F); CHKERRQ(ierr);
  ierr = DAGetLocalVector(app->da,&app->localX); CHKERRQ(ierr);
  ierr = VecDuplicate(app->localX,&app->localDX); CHKERRQ(ierr);

  /* Get local grid information */
  ierr = DAGetCorners(app->da,&app->xs,&app->ys,&app->zs,
                      &app->xm,&app->ym,&app->zm); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(app->da,&app->gxs,&app->gys,&app->gzs,
                      &app->gxm,&app->gym,&app->gzm); CHKERRQ(ierr);

  /* Get local->global mapping, used for matrix and vector assembly */
  ierr = DAGetGlobalIndices(app->da,&app->nloc,&app->ltog); CHKERRQ(ierr);

  /* Print grid info */
  if (app->print_grid) {
    int xs, ys, zs, xm, ym, zm;
    ierr = DAView(app->da,VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    ierr = DAGetCorners(app->da,&xs,&ys,&zs,&xm,&ym,&zm); CHKERRQ(ierr);
    PetscPrintf(comm,"global grid: %d X %d X %d with %d components per node ==> global vector dimension %d\n",
      app->mx,app->my,app->mz,nc,nc*app->mx*app->my*app->mz); fflush(stdout);
    ierr = VecGetLocalSize(app->X,&Nlocal); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(comm,1);
    printf("[%d] local grid %d X %d X %d with %d components per node ==> local vector dimension %d\n",
      app->rank,xm,ym,zm,nc,Nlocal);
    fflush(stdout);
    PetscSequentialPhaseEnd(comm,1);
  }

  /* Create pressure work vector, used for explicit boundary conditions */
  ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,app->mx,app->my,app->mz,
         app->Nx,app->Ny,app->Nz,1,2,&app->da1); CHKERRQ(ierr);
  ierr = DAGetDistributedVector(app->da1,&app->P); CHKERRQ(ierr);
  ierr = VecDuplicate(app->P,&app->Pbc); CHKERRQ(ierr);
  ierr = DAGetLocalVector(app->da1,&app->localP); CHKERRQ(ierr);

  app->xe     = app->xs + app->xm;
  app->ye     = app->ys + app->ym;
  app->ze     = app->zs + app->zm;
  app->gxe    = app->gxs + app->gxm;
  app->gye    = app->gys + app->gym;
  app->gze    = app->gzs + app->gzm;
  app->gdim   = app->mx * app->my * app->mz * nc;
  app->ldim   = app->xm * app->ym * app->zm * nc;
  app->lbkdim = app->xm * app->ym * app->zm;

  ierr = UserSetGridParameters(app); CHKERRQ(ierr);
  if (problem == 4) { /* parallel grid test */
    ierr = GridTest(app); CHKERRQ(ierr);
    ierr = UserDestroyEuler(app); CHKERRQ(ierr);
    PetscFinalize(); exit(0);
  } else {
    ierr = OptionsHasName(PETSC_NULL,"-gridtest",&flg); CHKERRQ(ierr);
    if (flg) {ierr = GridTest(app); CHKERRQ(ierr);}
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Setup parallel grid for Fortran code 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  fort_comm = PetscFromPointerComm(comm);
  ierr = parsetup_(&fort_comm, &app->print_grid, &app->no_output,
            &app->bctype, &app->rank, &app->size, &problem,
            &app->gxsf, &app->gysf, &app->gzsf,
            &app->gxef, &app->gyef, &app->gzef, 
            &app->xsf, &app->ysf, &app->zsf, 
            &app->xef, &app->yef, &app->zef,
            &app->xefm1, &app->yefm1, &app->zefm1, 
            &app->xefp1, &app->yefp1, &app->zefp1, 
            &app->gxefp1, &app->gyefp1, &app->gzefp1,
            &app->xsf1, &app->ysf1, &app->zsf1, 
            &app->gxsf1, &app->gysf1, &app->gzsf1,
            &app->xsf2, &app->ysf2, &app->zsf2, 
            &app->gxsf2, &app->gysf2, &app->gzsf2,
            &app->gxsfw, &app->gysfw, &app->gzsfw,
            &app->gxefw, &app->gyefw, &app->gzefw,
            &app->gxm, &app->gym, &app->gzm,
            &app->xef01, &app->yef01, &app->zef01,
            &app->gxef01, &app->gyef01, &app->gzef01); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Allocate local Fortran work space
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Residual vector boundary work space */
  if (app->bctype == IMPLICIT) {
    int lx, ly, lz;
    lx = app->xef01 - app->xsf2 + 1;
    ly = app->yef01 - app->ysf2 + 1;
    lz = app->zef01 - app->zsf2 + 1;
    llen  = ly * lz;
    llenb = llen * 2 * nc * sizeof(Scalar);
    app->fbcri1  = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->fbcri1);
    PetscMemzero(app->fbcri1,llenb);
    app->fbcrui1 = app->fbcri1  + llen;
    app->fbcrvi1 = app->fbcrui1 + llen;
    app->fbcrwi1 = app->fbcrvi1 + llen;
    app->fbcei1  = app->fbcrwi1 + llen;
    app->fbcri2  = app->fbcei1  + llen;
    app->fbcrui2 = app->fbcri2  + llen;
    app->fbcrvi2 = app->fbcrui2 + llen;
    app->fbcrwi2 = app->fbcrvi2 + llen;
    app->fbcei2  = app->fbcrwi2 + llen;

    llen  = lx * lz;
    llenb = llen * 2 * nc * sizeof(Scalar);
    app->fbcrj1 = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->fbcrj1);
    PetscMemzero(app->fbcrj1,llenb);
    app->fbcruj1 = app->fbcrj1  + llen;
    app->fbcrvj1 = app->fbcruj1 + llen;
    app->fbcrwj1 = app->fbcrvj1 + llen;
    app->fbcej1  = app->fbcrwj1 + llen;
    app->fbcrj2  = app->fbcej1  + llen;
    app->fbcruj2 = app->fbcrj2  + llen;
    app->fbcrvj2 = app->fbcruj2 + llen;
    app->fbcrwj2 = app->fbcrvj2 + llen;
    app->fbcej2  = app->fbcrwj2 + llen;

    llen  = lx * ly;
    llenb = llen * 2 * nc * sizeof(Scalar);
    app->fbcrk1 = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->fbcrk1);
    PetscMemzero(app->fbcrk1,llenb);
    app->fbcruk1 = app->fbcrk1  + llen;
    app->fbcrvk1 = app->fbcruk1 + llen;
    app->fbcrwk1 = app->fbcrvk1 + llen;
    app->fbcek1  = app->fbcrwk1 + llen;
    app->fbcrk2  = app->fbcek1  + llen;
    app->fbcruk2 = app->fbcrk2  + llen;
    app->fbcrvk2 = app->fbcruk2 + llen;
    app->fbcrwk2 = app->fbcrvk2 + llen;
    app->fbcek2  = app->fbcrwk2 + llen;
  }

  /* Fortran work arrays for vectors */
  llen = (app->gxefp1 - app->gxsf1+1) * (app->gyefp1 - app->gysf1+1) 
            * (app->gzefp1 - app->gzsf1+1);
  app->dr = (Scalar *)PetscMalloc(17*llen*sizeof(Scalar)); CHKPTRQ(app->dr);

  app->dru   = app->dr    + llen; /* components of F vector */
  app->drv   = app->dru   + llen;
  app->drw   = app->drv   + llen;
  app->de    = app->drw   + llen;
  app->r     = app->de    + llen; /* components of X vector */
  app->ru    = app->r     + llen;
  app->rv    = app->ru    + llen;
  app->rw    = app->rv    + llen;
  app->e     = app->rw    + llen;
  app->p     = app->e     + llen; /* pressure */
  app->p_bc  = app->p     + llen; /* parallel bc work space */
  app->r_bc  = app->p_bc  + llen;
  app->ru_bc = app->r_bc  + llen;
  app->rv_bc = app->ru_bc + llen;
  app->rw_bc = app->rv_bc + llen;
  app->e_bc  = app->rw_bc + llen;

  /* Fortran work arrays for matrix (diagonal) blocks */
  llen = (app->xefp1 - app->gxsf1 + 1) * (app->yefp1 - app->gysf1 + 1) 
          * (app->zefp1 - app->gzsf1 + 1) * nc * nc;
  if (!app->mat_assemble_direct || solve_with_julianne) {
    llenb = 6*llen*sizeof(Scalar);
    app->b1 = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->b1);
    PetscMemzero(app->b1,llenb);
    app->b2 = app->b1 + llen;
    app->b3 = app->b2 + llen;
    app->b4 = app->b3 + llen;
    app->b5 = app->b4 + llen;
    app->b6 = app->b5 + llen;
    if (app->bctype == IMPLICIT) {
      llen = app->xm * app->zm * nc * nc * 8
              + app->xm * app->ym * nc * nc * 4
              + app->zm * app->ym * nc * nc * 4;
      llenb = llen*sizeof(Scalar);
      app->b1bc     = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->b1bc);
      PetscMemzero(app->b1bc,llenb);
      app->b2bc     = app->b1bc + app->ym * app->zm * nc * nc * 4;
      app->b2bc_tmp = app->b2bc + app->xm * app->zm * nc * nc * 4;
      app->b3bc     = app->b2bc_tmp + app->xm * app->zm * nc * nc * 4;
    }
  }

  /* Work space for pseudo-transient continuation term, dt */
  llenb = ((app->xefp1 - app->xsf1 + 1) * (app->yefp1 - app->ysf1 + 1) 
          * (app->zefp1 - app->zsf1 + 1)) * sizeof(Scalar);
  app->dt = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->dt);
  PetscMemzero(app->dt,llenb);

  /* Work space for building main diagonal block of Jacobian */
  llenb = (app->xef01 - app->xsf1 + 1) * (app->yef01 - app->ysf1 + 1) 
          * (app->zef01 - app->zsf1 + 1) * nc * nc * sizeof(Scalar);
  app->diag = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->diag);
  PetscMemzero(app->diag,llenb);
  app->diag_len = llenb;

  /* Fortran work arrays for eigen info */
  llen = (app->gxefp1 - app->gxsf1 + 1) * (app->gyefp1 - app->gysf1 + 1) 
            * (app->gzefp1 - app->gzsf1 + 1) * nc * nc;
  llenb = 3*llen*sizeof(Scalar);
  app->bl = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->bl);
  PetscMemzero(app->bl,llenb);
  app->br = app->bl + llen;
  app->be = app->br + llen;

  /* Fortran work arrays for mesh metrics */
  llen = (app->gxef01 - app->gxsf1+1) * (app->gyef01 - app->gysf1+1) 
            * (app->gzef01 - app->gzsf1+1);
  llenb = 12*llen*sizeof(Scalar);
  app->sadai = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->sadai);
  PetscMemzero(app->sadai,llenb);
  app->sadaj = app->sadai + llen;
  app->sadak = app->sadaj + llen;
  app->aix   = app->sadak + llen;
  app->ajx   = app->aix   + llen;
  app->akx   = app->ajx   + llen;
  app->aiy   = app->akx   + llen;
  app->ajy   = app->aiy   + llen;
  app->aky   = app->ajy   + llen;
  app->aiz   = app->aky   + llen;
  app->ajz   = app->aiz   + llen;
  app->akz   = app->ajz   + llen;

  /* More Fortran work arrays */
  llen = nc * (app->xef01 - app->gxsf1+1) * (app->yef01 - app->gysf1+1) 
            * (app->zef01 - app->gzsf1+1);
  llenb = 3*llen*sizeof(Scalar);
  app->f1 = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->f1);
  app->g1 = app->f1 + llen;
  app->h1 = app->g1 + llen;

  llen = nc * app->ni;
  llenb = 6*llen*sizeof(Scalar);
  app->sp  = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->sp);
  app->sp1 = app->sp  + llen;
  app->sp2 = app->sp1 + llen;
  app->sm  = app->sp2 + llen;
  app->sm1 = app->sm  + llen;
  app->sm2 = app->sm1 + llen;

  /* Mesh coordinates */
  llen  = app->ni * app->nj * app->nk;
  llenb = llen * 3 * sizeof(Scalar);
  app->xc = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->xc);
  app->yc = app->xc + llen;
  app->zc = app->yc + llen;

  /* Misc Fortran work space */
  llenb = 2*(app->size+1)*sizeof(Scalar);
  app->work_p = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->work_p);
  PetscMemzero(app->work_p,llenb);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Set up scatters for certain BCs
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  app->Xbcscatter = 0; app->Pbcscatter = 0;
  ierr = BCScatterSetUp(app); CHKERRQ(ierr);

  *newapp = app;
  return 0;
}
/***************************************************************************/
/*
   UserSetGridParameters - Sets various grid parameters within the application
   context.

   Input Parameter:
   u - user-defined application context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserSetGridParameters(Euler *u)
{
  /* 
     Define Fortran grid points. Shifts between Fortran/C for the grid:
       - explicit boundary condition formulation:
         PETSc code works with the interior grid points only
           C:       i=0,i<ni-1; j=0,j<nj-1; k=0,k<nk-1
           Fortran: i=2,ni;     j=2,nj;     k=2,nk
       - implicit boundary condition formulation:
         PETSc code works with the interior grid and boundary points
           C:       i=0,i<ni+1; j=0,j<nj+1; k=0,k<nk+1
           Fortran: i=1,ni+1;   j=1,nj+1;   k=1,nk+1
  */
  if (u->bctype == EXPLICIT) {
    u->xsf  = u->xs+2; 
    u->ysf  = u->ys+2;
    u->zsf  = u->zs+2;
    u->gxsf = u->gxs+2;
    u->gysf = u->gys+2;
    u->gzsf = u->gzs+2;
    u->xef  = u->xe+1;
    u->yef  = u->ye+1;
    u->zef  = u->ze+1;
    u->gxef = u->gxe+1;
    u->gyef = u->gye+1;
    u->gzef = u->gze+1;
  } else {
    u->xsf  = u->xs+1; 
    u->ysf  = u->ys+1;
    u->zsf  = u->zs+1;
    u->gxsf = u->gxs+1;
    u->gysf = u->gys+1;
    u->gzsf = u->gzs+1;
    u->xef  = u->xe;
    u->yef  = u->ye;
    u->zef  = u->ze;
    u->gxef = u->gxe;
    u->gyef = u->gye;
    u->gzef = u->gze;
  }

  /* Use in Fortran code to get specific points */
  if (u->xe == u->mx) { 
    if (u->bctype == EXPLICIT) {
      u->xefm1  = u->xef-1;    /* points mx-1, my-1, mz-1 */
      u->xef01  = u->xef;      /* points mx, my, mz */
      u->xefp1  = u->xef+1;    /* points mx+1, my+1, mz+1 */
      u->gxef01 = u->gxef;     /* ghost points mx, my, mz */ 
      u->gxefp1 = u->gxef+1;   /* ghost points mx+1, my+1, mz+1 */ 
      u->gxefw  = u->gxef;     /* ending ghost point - 1 */ 
      u->xei    = u->xe;
      u->gxei   = u->gxe;
    } else {
      u->xefm1  = u->xef-2;    /* points mx-1, my-1, mz-1 */
      u->xef01  = u->xef-1;    /* points mx, my, mz */
      u->xefp1  = u->xef;      /* points mx+1, my+1, mz+1 */
      u->gxef01 = u->gxef-1;   /* ghost points mx, my, mz */ 
      u->gxefp1 = u->gxef;     /* ghost points mx+1, my+1, mz+1 */ 
      u->gxefw  = u->gxef01;   /* ending ghost point - 1 */ 
      u->xei    = u->xe-1;
      u->gxei   = u->gxe-1;
    }
  } else {
      u->xefm1  = u->xef;
      u->xef01  = u->xef;   
      u->xefp1  = u->xef;   
      u->gxef01 = u->gxef;
      u->gxefp1 = u->gxef;
      u->gxefw  = u->gxef-1;   
      u->xei    = u->xe;
      u->gxei   = u->gxe;
  }
  if (u->ye == u->my) {
    if (u->bctype == EXPLICIT) {
      u->yefm1  = u->yef-1;
      u->yef01  = u->yef;
      u->yefp1  = u->yef+1;
      u->gyef01 = u->gyef;
      u->gyefp1 = u->gyef+1;
      u->gyefw  = u->gyef;
      u->yei    = u->ye;
      u->gyei   = u->gye;
    } else {
      u->yefm1  = u->yef-2;
      u->yef01  = u->yef-1;
      u->yefp1  = u->yef;
      u->gyef01 = u->gyef-1;
      u->gyefp1 = u->gyef;
      u->gyefw  = u->gyef01;
      u->yei    = u->ye-1;
      u->gyei   = u->gye-1;
    }
  } else {
    u->yefm1  = u->yef;
    u->yef01  = u->yef;
    u->yefp1  = u->yef; 
    u->gyef01 = u->gyef;  
    u->gyefp1 = u->gyef;
    u->gyefw  = u->gyef-1;   
    u->yei    = u->ye;
    u->gyei   = u->gye;
  }
  if (u->ze == u->mz) {
    if (u->bctype == EXPLICIT) {
      u->zefm1  = u->zef-1;
      u->zef01  = u->zef;
      u->zefp1  = u->zef+1;
      u->gzef01 = u->gzef;
      u->gzefp1 = u->gzef+1;
      u->gzefw  = u->gzef;
      u->zei    = u->ze;
      u->gzei   = u->gze;
    } else {
      u->zefm1  = u->zef-2;
      u->zef01  = u->zef-1;
      u->zefp1  = u->zef;
      u->gzef01 = u->gzef-1;
      u->gzefp1 = u->gzef;
      u->gzefw  = u->gzef01;
      u->zei    = u->ze-1;
      u->gzei   = u->gze-1;
    }
  } else {
    u->zefm1  = u->zef;
    u->zef01  = u->zef;
    u->zefp1  = u->zef;   
    u->gzef01 = u->gzef;
    u->gzefp1 = u->gzef;
    u->gzefw  = u->gzef-1;   
    u->zei    = u->ze;
    u->gzei   = u->gze;
  }

  if (u->xs == 0) { 
    u->xsf1  = 1;         /* grid points:  x=1, y=1, z=1 */
    u->xsf2  = 2;         /* grid points:  x=2, y=2, z=2 */
    u->gxsf1 = 1;         /* ghost points: x=1, y=1, z=1 */ 
    u->gxsf2 = 2;         /* ghost points: x=2, y=2, z=2 */ 
    u->gxsfw = u->gxsf;   /* starting ghost point + 1 */
    if (u->bctype == EXPLICIT) {
      u->xsi  = u->xs;
      u->gxsi = u->gxs;
    } else {
      u->xsi  = u->xs+1;
      u->gxsi = u->gxs+1;
    }
  } else {
    u->xsf1  = u->xsf;
    u->xsf2  = u->xsf;
    u->gxsf1 = u->gxsf;
    u->gxsf2 = u->gxsf;
    u->gxsfw = u->gxsf+1;
    u->xsi   = u->xs;
    u->gxsi  = u->gxs;
  }
  if (u->ys == 0) {
    u->ysf1  = 1;
    u->ysf2  = 2;
    u->gysf1 = 1;
    u->gysf2 = 2;
    u->gysfw = u->gysf;
    if (u->bctype == EXPLICIT) {
      u->ysi  = u->ys;
      u->gysi = u->gys;
    } else {
      u->ysi  = u->ys+1;
      u->gysi = u->gys+1;
    }
  } else {
    u->ysf1  = u->ysf;
    u->ysf2  = u->ysf;
    u->gysf1 = u->gysf;
    u->gysf2 = u->gysf;
    u->gysfw = u->gysf+1;
    u->ysi   = u->ys;
    u->gysi  = u->gys;
  }
  if (u->zs == 0) {
    u->zsf1  = 1;
    u->zsf2  = 2;
    u->gzsf1 = 1;
    u->gzsf2 = 2;
    u->gzsfw = u->gzsf;
    if (u->bctype == EXPLICIT) {
      u->zsi  = u->zs;
      u->gzsi = u->gzs;
    } else {
      u->zsi  = u->zs+1;
      u->gzsi = u->gzs+1;
    }
  } else {
    u->zsf1  = u->zsf;
    u->zsf2  = u->zsf;
    u->gzsf1 = u->gzsf;
    u->gzsf2 = u->gzsf;
    u->gzsfw = u->gzsf+1;
    u->zsi   = u->zs;
    u->gzsi  = u->gzs;
  }

  u->xmfp1 = u->xefp1 - u->xsf1 + 1; /* widths for Fortran */
  u->ymfp1 = u->yefp1 - u->ysf1 + 1;
  u->zmfp1 = u->zefp1 - u->zsf1 + 1;
  u->gxmfp1 = u->gxefp1 - u->gxsf1 + 1; /* ghost widths for Fortran */
  u->gymfp1 = u->gyefp1 - u->gysf1 + 1;
  u->gzmfp1 = u->gzefp1 - u->gzsf1 + 1;

  if (u->print_grid) {
    PetscSequentialPhaseBegin(u->comm,1);
    fprintf(stdout,"[%d] Grid points:\n\
     xs=%d, xsi=%d, xe=%d, xei=%d, xm=%d, xmfp1=%d\n\
     ys=%d, ysi=%d, ye=%d, yei=%d, ym=%d, ymfp1=%d\n\
     zs=%d, zsi=%d, ze=%d, zei=%d, zm=%d, zmfp1=%d\n\
   Ghost points:\n\
     gxs=%d, gxsi=%d, gxe=%d, gxei=%d, gxm=%d, gxmfp1=%d\n\
     gys=%d, gysi=%d, gye=%d, gyei=%d, gym=%d, gymfp1=%d\n\
     gzs=%d, gzsi=%d, gze=%d, gzei=%d, gzm=%d, gzmfp1=%d\n",
     u->rank,u->xs,u->xsi,u->xe,u->xei,u->xm,u->xmfp1,
     u->ys,u->ysi,u->ye,u->yei,u->ym,u->ymfp1,
     u->zs,u->zsi,u->ze,u->zei,u->zm,u->zmfp1,
     u->gxs,u->gxsi,u->gxe,u->gxei,u->gxm,u->gxmfp1,
     u->gys,u->gysi,u->gye,u->gyei,u->gym,u->gymfp1,
     u->gzs,u->gzsi,u->gze,u->gzei,u->gzm,u->gzmfp1);
    fflush(stdout);
    PetscSequentialPhaseEnd(u->comm,1);
  }
  return 0;
}
/***************************************************************************/
/* 
   UserSetGrid - Reads mesh and sets up local portion of grid.
 */
int UserSetGrid(Euler *app)
{
  Scalar *xt, *yt, *zt;
  int    llen, i, j ,k, gxs1, gxe01, gys1, gye01, gzs1, gze01;
  int    mx_l, my_l, mz_l, mx_g, my_g, mz_g, ict_g, ict_l, ierr;
  int    itl, itu, ile, ktip;

  ierr = readmesh_(&itl,&itu,&ile,&ktip,app->xc,app->yc,app->zc); CHKERRQ(ierr);
  if ((app->bctype == EXPLICIT 
          && (app->ktip+2 != ktip || app->itl+2 != itl 
             || app->itu+2 != itu || app->ile+2 != ile)) ||
      ((app->bctype == IMPLICIT || app->bctype == IMPLICIT_SIZE)
          && (app->ktip+1 != ktip || app->itl+1 != itl 
            || app->itu+1 != itu || app->ile+1 != ile)))
     SETERRQ(1,1,"UserSetGrid: Conflicting wing parameters");

  /* Create local mesh and free global mesh if using > 1 processor;
     otherwise, return. */
  if (app->size > 1) {
    mx_l = (app->gxef01 - app->gxsf1 + 1); mx_g = app->ni1-1;
    my_l = (app->gyef01 - app->gysf1 + 1); my_g = app->nj1-1;
    mz_l = (app->gzef01 - app->gzsf1 + 1); mz_g = app->nk1-1;

    llen = mx_l * my_l * mz_l;
    xt = (Scalar *)PetscMalloc(llen * 3 * sizeof(Scalar)); CHKPTRQ(xt);
    yt = xt + llen;
    zt = yt + llen;

    gxs1 = app->gxsf1-1; gxe01 = app->gxef01;
    gys1 = app->gysf1-1; gye01 = app->gyef01;
    gzs1 = app->gzsf1-1; gze01 = app->gzef01;

    for (k=gzs1; k<gze01; k++) {
      for (j=gys1; j<gye01; j++) {
        for (i=gxs1; i<gxe01; i++) {
          ict_l = (k-gzs1)*mx_l*my_l + (j-gys1)*mx_l + i-gxs1;
          ict_g = k*mx_g*my_g + j*mx_g + i;
          xt[ict_l] = app->xc[ict_g];
          yt[ict_l] = app->yc[ict_g];
          zt[ict_l] = app->zc[ict_g];
        }
      }
    }
    PetscFree(app->xc);
    app->xc = xt;
    app->yc = yt;
    app->zc = zt;
  }
  return 0;
}
