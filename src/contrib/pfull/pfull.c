
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
  -jfreq <val>,    <val> = frequency of evaluating Jacobian (or precond)\n\
\n\
Problem parameters include:\n\
  -mx <xg>,    <xg> = number of grid points in the x-direction\n\
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
  -snes_fd :      use SNESDefaultComputeJacobian() to form Jacobian (instead\n\
                  of user-defined sparse variant)\n\
  -user_monitor : activate monitoring routine that prints the residual\n\
                  vector and Jacobian matrix to files at each iteration\n\n";

#include "puser.h"

/*
    main - This program controls the potential flow application code. 

    Useful Options Database Keys:
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
 */
int main( int argc, char **argv )
{
  AppCtx   user;             /* user-defined application context */
  GridCtx  *grid;

  SNES     snes;             /* nonlinear solver context */

  int      Nx, Ny;           /* number of processors in x- and y-directions */
  int      mx, my;           /* coarse mesh parameters */

  Viewer   viewer1, viewer2, viewer3;
  Draw     win_solution, win_mach, win_pressure;
  DrawLG   lg;

  Scalar   xd, yd, *pressure, zero = 0.0;
  int      i, flag, ierr, its, nlocals, nglobals; 

  /* ----------------------------------------------------------------------
       Initialize problem parameters
     ---------------------------------------------------------------------- */

  /* Initialize PETSc and set default viewer format */
  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,ASCII_FORMAT_COMMON,PETSC_NULL);CHKERRA(ierr);

  /* set partitioning of array across processors */
  Nx = PETSC_DECIDE; ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flag); CHKERRA(ierr);
  Ny = PETSC_DECIDE; ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flag); CHKERRA(ierr);

  /* Set problem parameters */
  user.xx0         = -5.0;
  user.xx1         = 5.0;
  user.yy0         = 0.0;
  user.yy1         = 10.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-x0",&user.xx0,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-y0",&user.yy0,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-x1",&user.xx1,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-y1",&user.yy1,&flag); CHKERRA(ierr);
  user.comm        = MPI_COMM_WORLD;
  user.mach        = 0.1;
  user.Qinf        = 1.0;
  user.jfreq       = 1;
  user.nc          = 1;
  ierr = OptionsGetInt(PETSC_NULL,"-jfreq",&user.jfreq,&flag); CHKERRA(ierr);
  if (user.jfreq < 1) SETERRA(1,"jfreq >= 1 only");
  ierr = OptionsGetDouble(PETSC_NULL,"-mach",&user.mach,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-qinf",&user.Qinf,&flag); CHKERRA(ierr);

  user.Nlevels = 1; ierr = OptionsGetInt(PETSC_NULL,"-Nlevels",&user.Nlevels,&flag); CHKERRA(ierr);
  user.Nstep   = 1; ierr = OptionsGetInt(PETSC_NULL,"-Nstep",&user.Nstep,&flag); CHKERRA(ierr);

  /* ----------------------------------------------------------------------
       Initialize grid parameters
     ---------------------------------------------------------------------- */

  mx = 4;  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,&flag); CHKERRA(ierr);
  my = 4;  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,&flag); CHKERRA(ierr);
  if (mx < 2 || my < 2) SETERRA(1,"mx, my >=2 only");

  for ( i=0; i<user.Nlevels; i++ ) {
    grid              = &user.grids[i];
    grid->mx          = mx;
    grid->my          = my;
    grid->hx          = (user.xx1-user.xx0) / (double)(grid->mx-1);
    grid->hy          = (user.yy1-user.yy0) / (double)(grid->my-1);

    mx  = pow(2.0,(double) user.Nstep)*mx - pow(2.0,(double) (user.Nstep)) + 1;
    my  = pow(2.0,(double) user.Nstep)*my - pow(2.0,(double) (user.Nstep)) + 1;

    /* ----------------------------------------------------------------------
         Create DA to manage parallel array
       ---------------------------------------------------------------------- */

    ierr = DACreate2d(user.comm,DA_NONPERIODIC,DA_STENCIL_BOX, 
                      grid->mx,grid->my,Nx,Ny,user.nc,1,&grid->da); CHKERRA(ierr);
    ierr = DAGetCorners(grid->da,&grid->xs,&grid->ys,PETSC_NULL,&grid->xm,&grid->ym,PETSC_NULL);CHKERRA(ierr);
    ierr = DAGetGhostCorners(grid->da,&grid->Xs,&grid->Ys,PETSC_NULL,&grid->Xm,&grid->Ym,PETSC_NULL); CHKERRA(ierr);
    grid->xe = grid->xs + grid->xm;
    grid->ye = grid->ys + grid->ym;
    grid->Xe = grid->Xs + grid->Xm;
    grid->Ye = grid->Ys + grid->Ym;

    /* ----------------------------------------------------------------------
             Create vector data structures  
       ---------------------------------------------------------------------- */

    /* Extract global and local vectors from DA; then duplicate for remaining work vectors */
    ierr     = DAGetDistributedVector(grid->da,&grid->globalX); CHKERRA(ierr);
    ierr     = DAGetLocalVector(grid->da,&grid->localX); CHKERRA(ierr);

    nglobals = 4;
    ierr     = VecDuplicateVecs(grid->globalX,nglobals,&grid->vec_g); CHKERRA(ierr);
    grid->globalF      = grid->vec_g[0];
    grid->globalMach   = grid->vec_g[1];
    grid->jj2          = grid->vec_g[2];
    grid->x2           = grid->vec_g[3];

    nlocals = 5; 
    ierr    = VecDuplicateVecs(grid->localX,nlocals,&grid->vec_l); CHKERRA(ierr);
    grid->localF       = grid->vec_l[0];
    grid->localMach    = grid->vec_l[1];
    grid->localDensity = grid->vec_l[2];
    grid->localXbak    = grid->vec_l[3];
    grid->localFbak    = grid->vec_l[4];

    ierr = VecCreateMPI(user.comm,PETSC_DECIDE,grid->mx,&grid->globalPressure); CHKERRA(ierr);
    ierr = VecSet(&zero,grid->globalPressure); CHKERRA(ierr);
    ierr = VecGetLocalSize(grid->globalX,&grid->ldim); CHKERRA(ierr);
    ierr = VecGetSize(grid->globalX,&grid->gdim); CHKERRA(ierr);
  }

  /* Print grid info and problem parameters */
  ierr = OptionsHasName(PETSC_NULL,"-print_param",&flag); CHKERRA(ierr);
  if (flag) {
    int rank;
    MPI_Comm_rank(user.comm,&rank);
    PetscPrintf(user.comm,"Potential flow: compute new Jacobian every %d iteration(s), Jacobian = preconditioner matrix\n",user.jfreq);
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

  /* Create nonlinear solver */
  ierr = SNESCreate(user.comm,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* Set default method (can be overridden with runtime option -snes_type <type>) */
  ierr = SNESSetType(snes,SNES_EQ_LS); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetFunction(snes,grid->globalF,Function_PotentialFlow,(void *)&user); CHKERRA(ierr);
  ierr = UserSetJacobian(snes,&user); CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-user_monitor",&flag); CHKERRA(ierr);
  if (flag) ierr = SNESSetMonitor(snes,UserMonitor,&user);
  else      ierr = SNESSetMonitor(snes,SNESDefaultMonitor,PETSC_NULL); CHKERRA(ierr);

  /* Set runtime options */
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
    ierr = ViewerSetFormat(viewer1,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = ViewerSetFormat(viewer2,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = ViewerSetFormat(viewer3,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);
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
 
  for ( i=0; i<user.Nlevels; i++ ) {
    grid              = &user.grids[i];
    ierr = VecDestroyVecs(grid->vec_g,nglobals); CHKERRA(ierr);
    ierr = VecDestroyVecs(grid->vec_l,nlocals); CHKERRA(ierr);
    ierr = VecDestroy(grid->globalX); CHKERRA(ierr);
    ierr = VecDestroy(grid->globalPressure); CHKERRA(ierr);
    ierr = VecDestroy(grid->localX); CHKERRA(ierr);
    ierr = DADestroy(grid->da); CHKERRA(ierr);
    ierr = MatDestroy(grid->J); CHKERRA(ierr);
  }

  ierr = SNESDestroy(snes); CHKERRA(ierr);
  if (win_solution) {ierr = DrawDestroy(win_solution); CHKERRA(ierr); }
  if (win_mach)     {ierr = DrawDestroy(win_mach); CHKERRA(ierr);}

  PetscFinalize();
  return 0;
}
/* ----------------------------------------------------------------- */
/*
   UserSetJacobian - Forms Jacobian matrix context and sets Jacobian
   evaluation routine.

   Input Parameters:
.  snes - SNES context
.  user - user-defined application context

   Notes:
   This routine preallocates matrix memory space for the Jacobian
   that arises for nonlinear problems when using the standard
   9-point finite difference stencil in 2D.  This code is writen
   for the specialized case of 1 degree of freedom per node (for
   the potential flow application); extensions to block data structures
   and multiple degrees of freedom per node are straightforward.
  
   Matrix memory preallocation such as in this routine is not necessary
   for matrix creation, since PETSc dynamically allocates memory when
   needed.  However, preallocation is crucial for fast matrix assembly!
   See the users manual and the file petsc/Performance for details.

   Useful Options Database Keys:
     -log_info        prints info about matrix memory allocation
     -mat_view_draw   draws sparsity pattern of matrix

     See manpage for MatAssemblyEnd() for additional options.
 */
int UserSetJacobian(SNES snes,AppCtx *user)
{
  GridCtx *grid;
  MatType mtype = MATSEQAIJ;                 /* matrix format */
  Mat     J;                                 /* Jacobian and/or preconditioner */
  int     ierr, c, flag, i, j, k, ye, xe;
  int     lrow, row, Xs, Ys, xm;
  int     ys, xs, is, ie, *nnz_d, *nnz_o;
  int     n_south, n_north, n_east, n_west, n_sw, n_nw, n_se, n_ne;
  int     mx1, my1, *ltog, nloc, Xm;

  /*
     Loop over all levels building the Jacobian matrix data structure for each level 
  */
  for ( k=0; k<user->Nlevels; k++ ) {
    grid   = &user->grids[k];
    mx1    = grid->mx-1;
    my1    = grid->my-1;
    ye     = grid->ye;
    xe     = grid->xe;
    Xs     = grid->Xs;
    Xm     = grid->Xm;
    Ys     = grid->Ys;
    xm     = grid->xm;
    ys     = grid->ys;
    xs     = grid->xs;
    nnz_d  = 0;
    nnz_o  = 0;



    /* ---------------- Preallocate matrix space ----------------- */
    /* First, precompute amount of space for matrix preallocation, to enable
       fast matrix assembly without continual dynamic memory allocation.

       IMPORTANT: This is not a particularly efficient way to compute this,
       having the if test inside a double loop. More efficiently you would have 
       seperate loops arount the boundary and in the interior!

       For this problem we need only a few basic formats.  Additional formats
       are available (in particular, use block formats for problems with multiple
       degrees of freedom per node!)

       Note: We over-allocate space here, since this is based on the stencil
       pattern only (not currently accounting for more sparsity due to boundary
       conditions). 
    */
    ierr = MatGetTypeFromOptions(user->comm,PETSC_NULL,&mtype,&flag); CHKERRQ(ierr);
    if (mtype == MATSEQAIJ || mtype == MATMPIROWBS) {
      nnz_d = (int *)PetscMalloc(grid->ldim * sizeof(int)); CHKPTRQ(nnz_d);
      PetscMemzero(nnz_d,grid->ldim * sizeof(int));
      for (j=ys; j<ye; j++) {
        if (j>0)   n_south = 1; else n_south = 0;
        if (j<my1) n_north = 1; else n_north = 0;
        for (i=xs; i<xe; i++) {
          if (i>0)            n_west = 1; else n_west = 0;
          if (i<mx1)          n_east = 1; else n_east = 0;
          if (i>0 && j>0)     n_sw   = 1; else n_sw   = 0;
          if (i>0 && j<my1)   n_nw   = 1; else n_nw   = 0;
          if (i<mx1 && j>0)   n_se   = 1; else n_se   = 0;
          if (i<mx1 && j<my1) n_ne   = 1; else n_ne   = 0;
          nnz_d[(j-ys)*xm + i-xs] = 1 + n_south + n_north + n_east + n_west + n_sw + n_nw + n_se + n_ne;
        }
      }
    } else if (mtype == MATMPIAIJ) {
      nnz_d = (int *)PetscMalloc(2*grid->ldim * sizeof(int)); CHKPTRQ(nnz_d);
      nnz_o = nnz_d + grid->ldim;
      PetscMemzero(nnz_d,2*grid->ldim * sizeof(int));
      /* Note: vector and matrix partitionings are identical */
      ierr = VecGetOwnershipRange(grid->globalX,&is,&ie); CHKERRQ(ierr);
      ierr = DAGetGlobalIndices(grid->da,&nloc,&ltog); CHKERRQ(ierr);
      for (j=ys; j<ye; j++) {
        for (i=xs; i<xe; i++) {
          row  = (j-Ys)*Xm + i-Xs;
          lrow = (j-ys)*xm + i-xs;
          if (i>0 && j>0)     {c = ltog[row - Xm - 1]; /* south-west */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
          if (j>0)            {c = ltog[row - Xm];     /* south */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
          if (i<mx1 && j>0)   {c = ltog[row - Xm + 1]; /* south-east */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
          if (i>0)            {c = ltog[row - 1];      /* west */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
                               c = ltog[row];          /* center */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;
          if (i<mx1)          {c = ltog[row + 1];      /* east */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
          if (i>0 && j<my1)   {c = ltog[row + Xm - 1]; /* north-west */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
          if (j<my1)          {c = ltog[row + Xm];     /* north */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
          if (i<mx1 && j<my1) {c = ltog[row + Xm + 1]; /* north-east */
                               if (c>=is && c<ie) nnz_d[lrow]++; else nnz_o[lrow]++;}
        }
      }
    } else SETERRQ(1,"UserSetJacobian: preallocation for matrix format not coded yet!");

    /* -------- Create data structure for Jacobian matrix --------- */
    /* 
     Notes:

     - We MUST specify the local matrix dimensions for the parallel matrix formats
     so that the matrix partitions match the vector partitions generated by the
     distributed array (DA).

      - To reduce the size of the executable, we do NOT use the generic matrix
     creation routine MatCreate(), which loads all possible PETSc matrix formats.
     While using MatCreate() is a good first step when beginnning to write a new 
     application code, the user should generally progress to using only the formats
     most appropriate for the application at hand.

      - For this problem (potential flow), we are concerned now only with sparse,
     row-based data structures; additional matrix formats can easily be added 
     (e.g., block variants for problems with multiple degrees of freedom per node).
     Here, we use:
         MATSEQAIJ - default uniprocessor sparse format
         MATMPIAIJ - default parallel sparse format
         MATMPIROWBS - parallel format needed for use of the parallel ILU(0)
                       preconditioner within BlockSolve95
     */

    if (mtype == MATSEQAIJ) {
      ierr = MatCreateSeqAIJ(user->comm,grid->gdim,grid->gdim,PETSC_NULL,nnz_d,&J); CHKERRQ(ierr);
    } 
    else if (mtype == MATMPIAIJ) {
      ierr = MatCreateMPIAIJ(user->comm,grid->ldim,grid->ldim,grid->gdim,
                             grid->gdim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&J); CHKERRQ(ierr);
    } 
    else if (mtype == MATMPIROWBS) {
      ierr = MatCreateMPIRowbs(user->comm,grid->ldim,grid->gdim,PETSC_NULL,nnz_d,
                               PETSC_NULL,&J); CHKERRQ(ierr);
    } else SETERRQ(1,"UserSetJacobian: interface for matrix format not coded yet!");

    grid->J = J;
    if (nnz_d) PetscFree(nnz_d);
  }

    /* -------- Specify Jacobian info for use by SNES solver (finest level only) --------- */
  
    ierr = SNESSetJacobian(snes,J,J,Jacobian_PotentialFlow,user); CHKERRQ(ierr);
    return 0;
}

/* --------------------------------------------------------------- */
/* 
   UserMonitor - User-defined monitoring routine for nonlinear solver.
   This routine is primarily intended for use in debugging.

   Input Parameters:
.  snes - SNES context
.  its - current iteration number
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
  ierr = ViewerSetFormat(view1,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  ierr = DFVecView(F,view1); CHKERRQ(ierr);
  ierr = ViewerDestroy(view1); CHKERRQ(ierr);

  /* Print preconditioner matrix (which here also serves as the Jacobian
     matrix approximation if this is not a matrix-free variant) */
  if (its) {
    sprintf(filename,"jac.%d.out",its);
    ierr = ViewerFileOpenASCII(MPI_COMM_WORLD,filename,&view1); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view1,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,PETSC_NULL,&Jprec,PETSC_NULL); CHKERRQ(ierr);
    ierr = MatView(Jprec,view1); CHKERRQ(ierr);
    ierr = ViewerDestroy(view1); CHKERRQ(ierr);
  }
  return 0;
}
