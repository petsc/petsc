
static char help[] =
"This program solves a 2D full potential flow problem in parallel.\n\
A finite difference approximation with the usual 9-point stencil is\n\
used to discretize the boundary value problem to obtain a nonlinear\n\
system of equations, which is then solved with the nonlinear solvers\n\
within PETSc.  The runtime options include:\n\
  -snes_mf_operator :    use matrix-free Newton-Krylov method (still explicitly\n\
                    forming the preconditioning matrix)\n\
  -print_param :    print problem parameters and grid information\n\
  -print_solution : print solution to stdout\n\
  -print_output :   print misc. output to various files\n\
  -mat_fd_coloring_freq <val>,    <val> = frequency of evaluating Jacobian (or precond)\n\
\n\
Problem parameters include:\n\
  -mx <xg>,    <xg> = number of grid p oints in the x-direction\n\
  -my <yg>,    <yg> = number of grid points in the y-direction\n\
  -x0 <xs>,    <xs> = physical domain value starting in x-direction\n\
  -x1 <xe>,    <xe> = physical domain value ending in x-direction\n\
  -y0 <ys>,    <ys> = physical domain value starting in y-direction\n\
  -y1 <ye>,    <ye> = physical domain value ending in y-direction\n\
  -mach <ma>,  <ma> = free stream Mach number\n\
  -qinf <qi>,  <qi> = parameter for equations\n\
  -Nx <nx>,    <nx> = number of processors in x-direction\n\
  -Ny <ny>,    <ny> = number of processors in y-direction\n\
\n\
Debugging options:\n\
  -user_monitor : activate monitoring routine that prints the residual\n\
                  vector and Jacobian matrix to files at each iteration\n\n";

#include "pfull.h"

/*
    main - This program controls the potential flow application code. 

    Useful Options Database Keys:
      -mat_fd_coloring_freq <f>  frequency at which Jacobians are recomputed.
      -snes_mf_operator          use a matrix free multiplication (for when using lagged Jacobians)
      -mat_fd_coloring_draw      display coloring of matrix

      -help                      prints detailed message about various runtime options
      -version                   prints version of PETSc being used
      -log_info                  prints verbose info about the solvers, data structures, etc.
      -log_summary               generates a summary of performance data
      -snes_view                 prints details about the SNES solver 
      -trdump                    dumps unused memory at program's conclusion
      -optionsleft               prints all runtime options, specifying any that have
                                    not been used during the program's execution
      -nox                       deactivates all x-window graphics
      -start_in_debugger         starts all processes in debugger
      -on_error_attach_debugger  attaches debugger if an error is encountered

      See manpages of various routines for many more runtime options!

    Currently this code works on a single grid; the "grid" datastructure is 
  separated from the "application" data structure to allow future expansion for
  grid sequencing.

*/

/*
     Note: This sample code is not great and could be enhanced in many ways;
   so just because something is done a particular way in this code it does not 
   mean it has to be done that way.
*/

/*
      To test: 
       - make BOPT=g
       - make BOPT=g runpfull1  (tests on one processor)
       - make BOPT=g runpfull4  (tests on four processors)

      For some graphics
      
         mpirun -np 1 pfull -options_file in.small     -draw_pause -1 
         mpirun -np 1 pfull -options_file in.graphics  -draw_pause -1

   The -draw_pause -1 causes it to wait after each graphic until you click in
the window with the righ most mouse button.

*/
int main( int argc, char **argv )
{
  AppCtx     user;             /* full potential application context */
  GridCtx    *grid;            /* grid information (for each level in multigrid/level methods) */

  SNES       snes;             /* nonlinear solver context */

  int        Nx, Ny;           /* number of processors in x- and y-directions */
  int        mx, my;           /* coarse mesh parameters */

  Viewer     viewer1, viewer2, viewer3;
  Draw       win_solution, win_mach, win_pressure;
  DrawLG     lg;

  ISColoring iscoloring;      /* coloring of matrix, for computation of jacobian */

  Scalar     xd, yd, *pressure, zero = 0.0;
  int        i, flag, ierr, its, nlocals, nglobals; 

  /* ---------------------------------------------------------------------
     Initialize PETSc and set default viewer format 
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = OptionsReject("-snes_mf","Use -snes_mf_operator\n");CHKERRA(ierr);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL);CHKERRA(ierr);

  /* ----------------------------------------------------------------------
       Initialize partioning across processors
     ---------------------------------------------------------------------- */

  /* set partitioning of array across processors */
  Nx = PETSC_DECIDE; ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flag); CHKERRA(ierr);
  Ny = PETSC_DECIDE; ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flag); CHKERRA(ierr);

  /* --------------------------------------------------------------------
        Set problem parameters 
     -------------------------------------------------------------------- */
  user.xx0         = -5.0;
  user.xx1         = 5.0;
  user.yy0         = 0.0;
  user.yy1         = 10.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-x0",&user.xx0,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-y0",&user.yy0,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-x1",&user.xx1,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-y1",&user.yy1,&flag); CHKERRA(ierr);
  user.comm        = PETSC_COMM_WORLD;
  user.mach        = 0.1;
  user.Qinf        = 1.0;
  user.nc          = 1;
  ierr = OptionsGetDouble(PETSC_NULL,"-mach",&user.mach,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-qinf",&user.Qinf,&flag); CHKERRA(ierr);


  /* ----------------------------------------------------------------------
       Initialize grid size parameters
     ---------------------------------------------------------------------- */
  user.Nlevels = 1; 

  mx = 10;  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,&flag); CHKERRA(ierr);
  my = 10;  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,&flag); CHKERRA(ierr);
  if (mx < 2 || my < 2) SETERRA(1,0,"mx, my >=2 only");

  grid              = &user.grids[0];
  grid->mx          = mx;
  grid->my          = my;
  grid->hx          = (user.xx1-user.xx0) / (double)(grid->mx-1);
  grid->hy          = (user.yy1-user.yy0) / (double)(grid->my-1);

  /* ----------------------------------------------------------------------
         Create DA to manage parallel array
     ---------------------------------------------------------------------- */

  ierr = DACreate2d(user.comm,DA_NONPERIODIC,DA_STENCIL_BOX, 
                      grid->mx,grid->my,Nx,Ny,user.nc,1,0,0,&grid->da); CHKERRA(ierr);
  ierr = DAGetCorners(grid->da,&grid->xs,&grid->ys,PETSC_NULL,&grid->xm,&grid->ym,
                        PETSC_NULL);CHKERRA(ierr);
  ierr = DAGetGhostCorners(grid->da,&grid->Xs,&grid->Ys,PETSC_NULL,&grid->Xm,&grid->Ym,
                             PETSC_NULL);CHKERRA(ierr);
  grid->xe = grid->xs + grid->xm;
  grid->ye = grid->ys + grid->ym;
  grid->Xe = grid->Xs + grid->Xm;
  grid->Ye = grid->Ys + grid->Ym;

  /* ----------------------------------------------------------------------
             Create vector data structures  
     ---------------------------------------------------------------------- */

  /* Extract global and local vectors from DA; duplicate for remaining work vectors */
  ierr     = DAGetDistributedVector(grid->da,&grid->globalX); CHKERRA(ierr);
  ierr     = DAGetLocalVector(grid->da,&grid->localX); CHKERRA(ierr);

  nglobals = 4; /* number of parallel work vectors */
  ierr     = VecDuplicateVecs(grid->globalX,nglobals,&grid->vec_g); CHKERRA(ierr);
  grid->globalF      = grid->vec_g[0];
  grid->globalMach   = grid->vec_g[1];
  grid->jj2          = grid->vec_g[2];
  grid->x2           = grid->vec_g[3];

  nlocals = 5; /* number of sequential (ghosted) work vectors */
  ierr    = VecDuplicateVecs(grid->localX,nlocals,&grid->vec_l); CHKERRA(ierr);
  grid->localF       = grid->vec_l[0];
  grid->localMach    = grid->vec_l[1];
  grid->localDensity = grid->vec_l[2];
  grid->localXbak    = grid->vec_l[3];
  grid->localFbak    = grid->vec_l[4];

  ierr = VecCreateMPI(user.comm,PETSC_DECIDE,user.nc*grid->mx,&grid->globalPressure);CHKERRA(ierr);
  ierr = VecSet(&zero,grid->globalPressure); CHKERRA(ierr);
  ierr = VecGetLocalSize(grid->globalX,&grid->ldim); CHKERRA(ierr);
  ierr = VecGetSize(grid->globalX,&grid->gdim); CHKERRA(ierr);

  /* -------------------------------------------------------------
       Set up coloring information needed for finite difference
       computation of sparse Jacobian
  ----------------------------------------------------------------*/
  ierr = DAGetColoring2dBox(grid->da,&iscoloring,&grid->J); CHKERRQ(ierr);
  ierr = MatFDColoringCreate(grid->J,iscoloring,&grid->fdcoloring); CHKERRQ(ierr); 
  ierr = MatFDColoringSetFunction(grid->fdcoloring,(int (*)(void *,Vec,Vec,void *))Function_PotentialFlow,&user);
         CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(grid->fdcoloring); CHKERRQ(ierr); 
  ierr = ISColoringDestroy(iscoloring); CHKERRQ(ierr);


  /* -------------------------------------------------------------
      Print grid info and problem parameters 
  ----------------------------------------------------------------*/
  ierr = OptionsHasName(PETSC_NULL,"-print_param",&flag); CHKERRA(ierr);
  if (flag) {
    int rank;
    MPI_Comm_rank(user.comm,&rank);
    PetscPrintf(user.comm,"problem domain corners: xx0=%g, xx1=%g, yy0=%g, yy1=%g\n",user.xx0,user.xx1,user.yy0,user.yy1);
    PetscPrintf(user.comm,"Mach number = %g, Qinf = %g\n",user.mach,user.Qinf);
    ierr = DAView(grid->da,VIEWER_STDOUT_SELF); CHKERRA(ierr);
    PetscPrintf(user.comm,"global grid: %d X %d with %d component(s) per node ==> global vector dimension %d\n",
                grid->mx,grid->my,user.nc,grid->gdim); fflush(stdout);
    PetscSequentialPhaseBegin(user.comm,1);
    printf("[%d] local grid %d X %d with %d component(s) per node ==> local vector dimension %d\n",
           rank,grid->xm,grid->ym,user.nc,grid->ldim);
    printf("[%d] xs=%d, xe=%d, Xs=%d, Xe=%d, ys=%d, ye=%d, Ys=%d, Ye=%d\n",
           rank,grid->xs,grid->xe,grid->Xs,grid->Xe,grid->ys,grid->ye,grid->Ys,grid->Ye);
    fflush(stdout);
    PetscSequentialPhaseEnd(user.comm,1);
  }

  /* ----------------------------------------------------------------------
         Create nonlinear SNES solver, set various routines and options.
               Also, create data structure for Jacobian matrix.
     ---------------------------------------------------------------------- */

  ierr = SNESCreate(user.comm,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* 
      Set default nonlinear solver method 
      (can be overridden with runtime option -snes_type tr) 
  */
  ierr = SNESSetType(snes,SNES_EQ_LS); CHKERRA(ierr);

  /* 
      Set the routine that evaluates the "function"
  */
  ierr = SNESSetFunction(snes,grid->globalF,Function_PotentialFlow,&user);CHKERRA(ierr);

  /*
      Set the routine that evaluates the "Jacobian"
  */
  ierr = SNESSetJacobian(snes,grid->J,grid->J,SNESDefaultComputeJacobianWithColoring,
                             grid->fdcoloring); CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-user_monitor",&flag); CHKERRA(ierr);
  if (flag) {ierr = SNESSetMonitor(snes,UserMonitor,&user); CHKERRA(ierr);}

  /* 
     Set runtime solution options 
  */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* ----------------------------------------------------------------------
         Compute Initial Guess
     ---------------------------------------------------------------------- */

  ierr = InitialGuess_PotentialFlow(&user,grid->globalX); CHKERRA(ierr);

  /* ----------------------------------------------------------------------
         Solve nonlinear system
     ---------------------------------------------------------------------- */

  ierr = SNESSolve(snes,grid->globalX,&its);  CHKERRA(ierr);
  PetscPrintf(user.comm,"Number of Newton iterations = %d\n", its );

  /* ----------------------------------------------------------------------
         Interpret solution
     ---------------------------------------------------------------------- */

  ierr = OptionsHasName(PETSC_NULL,"-print_solution",&flag); CHKERRA(ierr);
  if (flag) {ierr = DFVecView(grid->globalX,VIEWER_STDOUT_WORLD); CHKERRA(ierr);}

  ierr = OptionsHasName(PETSC_NULL,"-print_output",&flag); CHKERRA(ierr);
  if (flag) {
    ierr = ViewerFileOpenASCII(user.comm,"outmach",&viewer1); CHKERRA(ierr); 
    ierr = ViewerFileOpenASCII(user.comm,"outpre",&viewer2); CHKERRA(ierr); 
    ierr = ViewerFileOpenASCII(user.comm,"outpot",&viewer3); CHKERRA(ierr);
    ierr = ViewerSetFormat(viewer1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = ViewerSetFormat(viewer2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = ViewerSetFormat(viewer3,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = DFVecView(grid->globalMach,viewer1); CHKERRA(ierr);
    ierr = DFVecView(grid->globalPressure,viewer2); CHKERRA(ierr);
    ierr = DFVecView(grid->globalX,viewer3); CHKERRA(ierr);
    ierr = ViewerDestroy(viewer1); CHKERRA(ierr);
    ierr = ViewerDestroy(viewer2); CHKERRA(ierr);
    ierr = ViewerDestroy(viewer3); CHKERRA(ierr);
  }

  /* Draw contour plot of solution and Mach number */
  ierr = DrawOpenX(user.comm,0,"Solution",0,0,300,300,&win_solution); CHKERRA(ierr);
  ierr = DrawTensorContour(win_solution,grid->mx,grid->my,0,0,grid->globalX); CHKERRA(ierr);
  ierr = DrawSyncFlush(win_solution); CHKERRA(ierr);

  ierr = DrawOpenX(user.comm,0,"Mach Number",310,0,300,300,&win_mach); CHKERRA(ierr);
  ierr = DrawTensorContour(win_mach,grid->mx,grid->my,0,0,grid->globalMach); CHKERRA(ierr);
  ierr = DrawSyncFlush(win_mach); CHKERRA(ierr);

  /* Draw line graph of pressure */
  ierr = OptionsHasName(PETSC_NULL,"-nox",&flag); CHKERRA(ierr);
  if (!flag) {
    ierr = DrawOpenX(user.comm,0,"Pressure",620,0,300,300,&win_pressure); CHKERRA(ierr);
    VecGetArray(grid->globalPressure, &pressure);
    ierr = DrawLGCreate(win_pressure,1,&lg); CHKERRA(ierr);
    xd   = -5;
    yd   = pressure[1];
    ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);
    for (i=1; i<grid->mx-1; i++) {
      xd   = (user.xx1 - user.xx0)/(double)(grid->mx - 1) * i + user.xx0;
      yd   = pressure[i];
      ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);
    }
    xd   = 5;
    yd   = pressure[grid->mx-2];
    ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);

    ierr = VecRestoreArray(grid->globalPressure,&pressure); CHKERRA(ierr);
    ierr = DrawLGDraw(lg); CHKERRA(ierr);
    ierr = DrawFlush(win_pressure); CHKERRA(ierr); 

    ierr = DrawDestroy(win_pressure); CHKERRA(ierr);
    ierr = DrawLGDestroy(lg); CHKERRA(ierr);
  }

  /* ----------------------------------------------------------------
        Free data structures 
     ---------------------------------------------------------------- */
 
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = VecDestroyVecs(grid->vec_g,nglobals); CHKERRA(ierr);
  ierr = VecDestroyVecs(grid->vec_l,nlocals); CHKERRA(ierr);
  ierr = VecDestroy(grid->globalX); CHKERRA(ierr);
  ierr = VecDestroy(grid->globalPressure); CHKERRA(ierr);
  ierr = VecDestroy(grid->localX); CHKERRA(ierr);
  ierr = MatDestroy(grid->J); CHKERRA(ierr);
  ierr = MatFDColoringDestroy(grid->fdcoloring); CHKERRA(ierr);  
  ierr = DADestroy(grid->da); CHKERRA(ierr);

  if (win_solution) {ierr = DrawDestroy(win_solution); CHKERRA(ierr); }
  if (win_mach)     {ierr = DrawDestroy(win_mach); CHKERRA(ierr);}

  PetscFinalize();
  return 0;
}

/* --------------------------------------------------------------- */
/* 
   UserMonitor - User-defined monitoring routine for nonlinear solver.
   This routine is primarily intended for use in debugging.

   Input Parameters:
.  snes  - SNES context
.  its   - current iteration number
.  fnorm - norm of current function
.  dummy - optional user-defined context

   Notes:
   This monitoring routine should be set by calling SNESSetMonitor()
   before calling SNESSolve().
 */
int UserMonitor(SNES snes,int its,double fnorm,void *dummy)
{
  MPI_Comm comm;
  Viewer   view1;
  Vec      F;
  Mat      Jprec;
  char     filename[64];
  int      ierr;

  PetscObjectGetComm((PetscObject)snes,&comm);
  PetscPrintf(comm,"iter = %d, Function norm %g \n",its,fnorm);

  /* Print residual vector */
  sprintf(filename,"res.%d.out",its);
  ierr = ViewerFileOpenASCII(MPI_COMM_WORLD,filename,&view1); CHKERRQ(ierr);
  ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  ierr = DFVecView(F,view1); CHKERRQ(ierr);
  ierr = ViewerDestroy(view1); CHKERRQ(ierr);

  /* Print preconditioner matrix (which here also serves as the Jacobian
     matrix approximation if this is not a matrix-free variant) */
  if (its) {
    sprintf(filename,"jac.%d.out",its);
    ierr = ViewerFileOpenASCII(MPI_COMM_WORLD,filename,&view1); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,PETSC_NULL,&Jprec,PETSC_NULL); CHKERRQ(ierr);
    ierr = MatView(Jprec,view1); CHKERRQ(ierr);
    ierr = ViewerDestroy(view1); CHKERRQ(ierr);
  }
  return 0;
}













