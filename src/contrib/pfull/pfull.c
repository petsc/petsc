
static char help[] =
"This program solves a 2D full potential flow problem in parallel.\n\
A finite difference approximation with the usual 9-point stencil is\n\
used to discretize the boundary value problem to obtain a nonlinear\n\
system of equations, which is then solved with the nonlinear solvers\n\
within PETSc.  The runtime options include:\n\
  -matrix_free : use matrix-free Newton-Krylov method (still explicitly\n\
                    forming the preconditioning matrix)\n\
  -print_param : print problem parameters and grid information\n\
  -print_solution : print solution to stdout\n\
  -print_output : print misc. output to various files\n\
  -jfreq <val>, where <val> = frequency of evaluating Jacobian (or precond)\n\
\n\
Problem parameters include:\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -x0 <xs>, where <xs> = physical domain value starting in x-direction\n\
  -x1 <xe>, where <xe> = physical domain value ending in x-direction\n\
  -y0 <ys>, where <ys> = physical domain value starting in y-direction\n\
  -y1 <ye>, where <ye> = physical domain value ending in y-direction\n\
  -mach <ma>, where <ma> = free stream Mach number\n\
  -qinf <qi>, where <qi> = parameter for equations\n\
  -Nx <nx>, where <nx> = number of processors in x-direction\n\
  -Ny <ny>, where <ny> = number of processors in y-direction\n\
\n\
Debugging options:\n\
  -snes_fd : use SNESDefaultComputeJacobian() to form Jacobian (instead\n\
             of user-defined sparse variant)\n\
  -user_monitor : activate monitoring routine that prints the residual\n\
                  vector and Jacobian matrix to files at each iteration\n\n";

#include "puser.h"
int TestMatrix(AppCtx*,Mat);

/*
    main - This program controls the potential flow application code. 

    Cool Options Database Keys:
      -help : prints detailed message about various runtime options
      -version : prints version of PETSc being used
      -log_info : prints verbose info about the solvers, data structures, etc.
      -log_summary : generates a summary of performance data
      -snes_view : prints details about the SNES solver 
      -trdump : dumps unused memory at program's conclusion
      -optionsleft :  prints all runtime options, specifying any that have
                      not been used during the program's execution
      -nox : deactivates all x-window graphics
      -start_in_debugger : starts all processes in debugger
      -on_error_attach_debugger : attaches debugger if an error is encountered

      See manpages of various routines for lots of additional runtime options!
 */
int main( int argc, char **argv )
{
  SNES     snes;             /* nonlinear solver context */
  Vec      X;                /* solution vector */
  Vec      *vec_g, *vec_l;   /* global, local vectors */
  AppCtx   user;             /* user-defined application context */
  int      Nx, Ny;           /* number of processors in x- and y-directions */
  MPI_Comm comm;             /* communicator */
  Viewer   viewer1, viewer2, viewer3;
  Draw     win_solution, win_mach, win_pressure;
  DrawLG   lg;
  Scalar   xd, yd, *pressure, zero = 0.0;
  int      mx, my, i, flag, ierr, its, nlocals, nglobals; 
  double   xx0, xx1, yy0, yy1;

  /* ----------------------------------------------------------------------
       Phase 1: Initialize problem and create DA to manage parallel grid
     ---------------------------------------------------------------------- */

  /* Initialize PETSc and default viewer format */
  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);

  /* Set problem parameters */
  PetscMemzero(&user,sizeof(AppCtx));
  mx               = 4;
  my               = 4;
  xx0              = -5.0;
  xx1              = 5.0;
  yy0              = 0.0;
  yy1              = 10.0;
  user.nc          = 1;
  user.M           = 0.1;
  user.Qinf        = 1.0;
  user.jfreq       = 1;
  user.matrix_free = 0;
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,&flag); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,&flag); CHKERRA(ierr);
  if (mx < 2 || my < 2) SETERRA(1,"mx, my >=2 only");
  ierr = OptionsGetInt(PETSC_NULL,"-jfreq",&user.jfreq,&flag); CHKERRA(ierr);
  if (user.jfreq < 1) SETERRA(1,"jfreq >= 1 only");
  ierr = OptionsGetDouble(PETSC_NULL,"-x0",&xx0,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-y0",&yy0,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-x1",&xx1,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-y1",&yy1,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-mach",&user.M,&flag); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-qinf",&user.Qinf,&flag); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-matrix_free",&user.matrix_free); CHKERRQ(ierr);
  user.xx0 = xx0; user.xx1 = xx1; user.yy0 = yy0; user.yy1 = yy1;
  user.mx = mx; user.my = my;
  user.hx = (xx1-xx0) / (double)(mx-1);
  user.hy = (yy1-yy0) / (double)(my-1);

  user.comm = comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&user.rank);
  MPI_Comm_size(comm,&user.size);

  /* Create distributed array (DA) */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flag); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flag); CHKERRA(ierr);
  if (Nx*Ny != user.size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE))
    SETERRA(1,"Incompatible number of processors:  Nx * Ny != size");
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX, 
         user.mx,user.my,Nx,Ny,user.nc,1,&user.da); CHKERRA(ierr);
  ierr = DAGetCorners(user.da,&user.xs,&user.ys,PETSC_NULL,&user.xm,&user.ym,PETSC_NULL); CHKERRA(ierr);
  ierr = DAGetGhostCorners(user.da,&user.Xs,&user.Ys,PETSC_NULL,&user.Xm,&user.Ym,PETSC_NULL); CHKERRA(ierr);
  user.xe = user.xs + user.xm;
  user.ye = user.ys + user.ym;
  user.Xe = user.Xs + user.Xm;
  user.Ye = user.Ys + user.Ym;

  /* ----------------------------------------------------------------------
             Phase 2:  Create vector data structures  
     ---------------------------------------------------------------------- */

  /* Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types */
  ierr = DAGetDistributedVector(user.da,&X); CHKERRA(ierr);
  ierr = DAGetLocalVector(user.da,&user.localX); CHKERRA(ierr);
  user.X = X;
  nglobals = 5;
  ierr = VecDuplicateVecs(X,nglobals,&vec_g); CHKERRA(ierr);
  user.F            = vec_g[0];
  user.Fcopy        = vec_g[1];
  user.globalMach   = vec_g[2];
  user.jj2          = vec_g[3];
  user.x2           = vec_g[4];
  nlocals = 5; 
  ierr = VecDuplicateVecs(user.localX,nlocals,&vec_l); CHKERRA(ierr);
  user.localF       = vec_l[0];
  user.localMach    = vec_l[1];
  user.localDensity = vec_l[2];
  user.localXbak    = vec_l[3];
  user.localFbak    = vec_l[4];
  ierr = VecCreateMPI(comm,PETSC_DECIDE,user.mx,&user.globalPressure); CHKERRA(ierr);
  ierr = VecSet(&zero,user.globalPressure); CHKERRA(ierr);
  ierr = VecGetLocalSize(X,&user.ldim); CHKERRA(ierr);
  ierr = VecGetSize(X,&user.gdim); CHKERRA(ierr);

  /* Print grid info and problem parameters */
  ierr = OptionsHasName(PETSC_NULL,"-print_param",&flag); CHKERRA(ierr);
  if (flag) {
    if (user.matrix_free)
      PetscPrintf(comm,"Potential flow: matrix-free version, compute new preconditioner every %d iteration(s)\n",user.jfreq);
    else
      PetscPrintf(comm,"Potential flow: compute new Jacobian every %d iteration(s), Jacobian = preconditioner matrix\n",user.jfreq);
    PetscPrintf(comm,"problem domain corners: xx0=%g, xx1=%g, yy0=%g, yy1=%g\n",xx0,xx1,yy0,yy1);
    PetscPrintf(comm,"Mach number = %g, Qinf = %g\n",user.M,user.Qinf);
    ierr = DAView(user.da,VIEWER_STDOUT_SELF); CHKERRA(ierr);
    PetscPrintf(comm,"global grid: %d X %d with %d component(s) per node ==> global vector dimension %d\n",
      user.mx,user.my,user.nc,user.gdim); fflush(stdout);
    PetscSequentialPhaseBegin(comm,1);
    printf("[%d] local grid %d X %d with %d component(s) per node ==> local vector dimension %d\n",
      user.rank,user.xm,user.ym,user.nc,user.ldim);
    printf("[%d] xs=%d, xe=%d, Xs=%d, Xe=%d, ys=%d, ye=%d, Ys=%d, Ye=%d\n",
      user.rank,user.xs,user.xe,user.Xs,user.Xe,user.ys,user.ye,user.Ys,user.Ye);
    fflush(stdout);
    PetscSequentialPhaseEnd(comm,1);
  }

  /* ----------------------------------------------------------------------
       Phase 3:  Create SNES solver, set various routines and options.
                 Also, create data structure for Jacobian matrix.
     ---------------------------------------------------------------------- */

  /* Create nonlinear solver */
  ierr = SNESCreate(comm,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* Set default method (can be overridden with runtime option -snes_type <type>) */
  ierr = SNESSetType(snes,SNES_EQ_LS); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetFunction(snes,user.F,Function_PotentialFlow,(void *)&user); CHKERRA(ierr);
  ierr = UserSetJacobian(snes,&user); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-user_monitor",&flag); CHKERRA(ierr);
  if (flag) ierr = SNESSetMonitor(snes,UserMonitor,(void*)&user);
  else      ierr = SNESSetMonitor(snes,SNESDefaultMonitor,PETSC_NULL); CHKERRA(ierr);

  /* Set runtime options */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* ----------------------------------------------------------------------
        Phase 4:  Solve nonlinear system
     ---------------------------------------------------------------------- */

  /* Compute initial guess, then solve the nonlinear system */
  ierr = InitialGuess_PotentialFlow(&user,X); CHKERRA(ierr);
  ierr = SNESSolve(snes,X,&its);  CHKERRA(ierr);
  PetscPrintf(comm,"Number of Newton iterations = %d\n", its );

  /* ----------------------------------------------------------------------
        Phase 5:  Interpret solution
     ---------------------------------------------------------------------- */

  ierr = OptionsHasName(PETSC_NULL,"-print_solution",&flag); CHKERRA(ierr);
  if (flag) {ierr = DFVecView(X,VIEWER_STDOUT_WORLD); CHKERRA(ierr);}

  ierr = OptionsHasName(PETSC_NULL,"-print_output",&flag); CHKERRA(ierr);
  if (flag) {
      ierr = ViewerFileOpenASCII(comm,"outmach",&viewer1); CHKERRA(ierr); 
      ierr = ViewerFileOpenASCII(comm,"outpre",&viewer2); CHKERRA(ierr); 
      ierr = ViewerFileOpenASCII(comm,"outpot",&viewer3); CHKERRA(ierr);
      ierr = ViewerSetFormat(viewer1,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);
      ierr = ViewerSetFormat(viewer2,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);
      ierr = ViewerSetFormat(viewer3,ASCII_FORMAT_COMMON,PETSC_NULL); CHKERRA(ierr);
      ierr = DFVecView(user.globalMach,viewer1); CHKERRA(ierr);
      ierr = DFVecView(user.globalPressure,viewer2); CHKERRA(ierr);
      ierr = DFVecView(X,viewer3); CHKERRA(ierr);
      ierr = ViewerDestroy(viewer1); CHKERRA(ierr);
      ierr = ViewerDestroy(viewer2); CHKERRA(ierr);
      ierr = ViewerDestroy(viewer3); CHKERRA(ierr);
  }

  /* Draw contour plot of solution and Mach number */
  ierr = DrawOpenX(comm,0,"Solution",0,0,300,300,&win_solution); CHKERRA(ierr);
  ierr = DrawTensorContour(win_solution,user.mx,user.my,0,0,X); CHKERRA(ierr);
  ierr = DrawSyncFlush(win_solution); CHKERRA(ierr);
  ierr = DrawOpenX(comm,0,"Mach Number",310,0,300,300,&win_mach); CHKERRA(ierr);
  ierr = DrawTensorContour(win_mach,user.mx,user.my,0,0,user.globalMach); CHKERRA(ierr);
  ierr = DrawSyncFlush(win_mach); CHKERRA(ierr);

  /* Draw line graph of pressure */
  ierr = OptionsHasName(PETSC_NULL,"-nox",&flag); CHKERRA(ierr);
  if (!flag) {
      ierr = DrawOpenX(comm,0,"Pressure",620,0,300,300,&win_pressure); CHKERRA(ierr);
      VecGetArray(user.globalPressure, &pressure);
      ierr = DrawLGCreate(win_pressure,1,&lg); CHKERRA(ierr);
      xd = -5;
      yd = pressure[1];
      ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);
      for (i=1; i<user.mx-1; i++) {
         xd = (user.xx1 - user.xx0)/(double)(user.mx - 1) * i + user.xx0;
         yd = pressure[i];
         ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);
      }
      xd = 5;
      yd = pressure[user.mx-2];
      ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);

      ierr = VecRestoreArray(user.globalPressure,&pressure); CHKERRA(ierr);
      ierr = DrawLGDraw(lg); CHKERRA(ierr);
      ierr = DrawFlush(win_pressure); CHKERRA(ierr); 
      PetscSleep(600);

      ierr = DrawDestroy(win_pressure); CHKERRA(ierr);
      ierr = DrawLGDestroy(lg); CHKERRA(ierr);
  }

  /* ----------------------------------------------------------------
        Phase 6:  Free data structures 
     ---------------------------------------------------------------- */
 
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = VecDestroyVecs(vec_g,nglobals); CHKERRA(ierr);
  ierr = VecDestroyVecs(vec_l,nlocals); CHKERRA(ierr);
  ierr = VecDestroy(user.X); CHKERRA(ierr);
  ierr = VecDestroy(user.globalPressure); CHKERRA(ierr);
  ierr = VecDestroy(user.localX); CHKERRA(ierr);
  ierr = DADestroy(user.da); CHKERRA(ierr);
  ierr = MatDestroy(user.J); CHKERRA(ierr);
  if (user.Jmf) ierr = MatDestroy(user.Jmf); CHKERRA(ierr);
  if (win_solution) {ierr = DrawDestroy(win_solution); CHKERRA(ierr); }
  if (win_mach) {ierr = DrawDestroy(win_mach); CHKERRA(ierr);}

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

   Cool Options Database Keys:
     -log_info - prints info about matrix memory allocation
     -mat_view_draw - draws sparsity pattern of matrix
     See manpage for MatAssemblyEnd() for additional options.
 */
int UserSetJacobian(SNES snes,AppCtx *user)
{
  MatType mtype = MATSEQAIJ;     /* matrix format */
  Mat     J;                     /* Jacobian and/or preconditioner */
  int     ierr, c, flag, i, j, ye = user->ye, xe = user->xe;
  int     lrow, row, Xs = user->Xs, Ys = user->Ys, xm = user->xm;
  int     ys = user->ys, xs = user->xs, is, ie, *nnz_d = 0, *nnz_o = 0;
  int     n_south, n_north, n_east, n_west, n_sw, n_nw, n_se, n_ne;
  int     mx1 = user->mx-1, my1 = user->my-1, *ltog, nloc, nz_d, Xm = user->Xm;

  /* ---------------- Phase 1:  Preallocate matrix space ----------------- */

  /* First, precompute amount of space for matrix preallocation, to enable
     fast matrix assembly without continual dynamic memory allocation.

     For this problem we need only a few basic formats.  Additional formats
     are available (in particular, use block formats for problems with multiple
     degrees of freedom per node!)

     Note: We over-allocate space here, since this is based on the stencil
     pattern only (not currently accounting for more sparsity due to boundary
     conditions.  We can make this more accurate later.
   */
  ierr = MatGetTypeFromOptions(user->comm,PETSC_NULL,&mtype,&flag); CHKERRQ(ierr);
  if (mtype == MATSEQAIJ || mtype == MATMPIROWBS) {
    nnz_d = (int *)PetscMalloc(user->ldim * sizeof(int)); CHKPTRQ(nnz_d);
    PetscMemzero(nnz_d,user->ldim * sizeof(int));
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
        nz_d = 1 + n_south + n_north + n_east + n_west + n_sw + n_nw + n_se + n_ne;
        nnz_d[(j-ys)*xm + i-xs] = nz_d;
      }
    }
  } else if (mtype == MATMPIAIJ) {
    nnz_d = (int *)PetscMalloc(2*user->ldim * sizeof(int)); CHKPTRQ(nnz_d);
    PetscMemzero(nnz_d,2*user->ldim * sizeof(int));
    nnz_o = nnz_d + user->ldim;
    /* Note: vector and matrix partitionings are identical */
    ierr = VecGetOwnershipRange(user->X,&is,&ie); CHKERRQ(ierr);
    ierr = DAGetGlobalIndices(user->da,&nloc,&ltog); CHKERRQ(ierr);
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

  /* -------- Phase 2: Create data structure for Jacobian matrix --------- */
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
    /* Rough estimate of nonzeros per row is nd * nc = 9 * 1 = 9 */
    /* ierr = MatCreateSeqAIJ(user->comm,user->gdim,user->gdim,nd,PETSC_NULL,&J); CHKERRQ(ierr); */
    ierr = MatCreateSeqAIJ(user->comm,user->gdim,user->gdim,PETSC_NULL,nnz_d,&J); CHKERRQ(ierr);
  } 
  else if (mtype == MATMPIAIJ) {
    ierr = MatCreateMPIAIJ(user->comm,user->ldim,user->ldim,user->gdim,
                           user->gdim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&J); CHKERRQ(ierr);
  } 
  else if (mtype == MATMPIROWBS) {
    ierr = MatCreateMPIRowbs(user->comm,user->ldim,user->gdim,PETSC_NULL,nnz_d,
                             PETSC_NULL,&J); CHKERRQ(ierr);
  } else SETERRQ(1,"UserSetJacobian: interface for matrix format not coded yet!");
  user->J = J;
  if (nnz_d) PetscFree(nnz_d);

  ierr = OptionsHasName(PETSC_NULL,"-test_matrix",&flag); CHKERRQ(ierr);  
  if (flag) {ierr = TestMatrix(user,J); CHKERRQ(ierr);}

  /* -------- Phase 3: Specify Jacobian info for use by SNES solver --------- */

  if (user->matrix_free) {
    /* Use matrix-free Jacobian to define Newton system; use explicit (approx)
       Jacobian for preconditioner */
    ierr = SNESDefaultMatrixFreeMatCreate(snes,user->X,&user->Jmf); CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,user->Jmf,J,Jacobian_PotentialFlow,user); CHKERRQ(ierr);
  } else {
    /* Use explicit (approx) Jacobian to define Newton system and preconditioner */
    ierr = SNESSetJacobian(snes,J,J,Jacobian_PotentialFlow,user); CHKERRQ(ierr);
  }
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
/* --------------------------------------------------------------- */
/*
   TestMatrix - Forms a test matrix to test work with the 9-point
   stencil in 2D.  This routine is for debugging purposes only.

   Input Parameters:
.  user - user-defined application context
.  A - matrix 

   Notes:
   See manpage for MatAssemblyEnd() for options to examine matrix info.
*/
int TestMatrix(AppCtx *user,Mat A)
{
  Scalar val[9];
  int ierr, ict, i, j, nloc, *ltog, row, col[9];
  int xs = user->xs, ys= user->ys, xe = user->xe, ye = user->ye;
  int Xs = user->Xs, Ys = user->Ys, Xm = user->Xm;
  int mx1 = user->mx-1, my1 = user->my-1, l, grow;

  PetscPrintf(user->comm,"TestMatrix: Forming test matrix, then exiting PETSc");
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog); CHKERRQ(ierr);
  for (l=0; l<9; l++) val[l] = 1.0;
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
      row  = (j-Ys)*Xm + i-Xs;
      grow = ltog[row];
      ict = 0;
      if (i>0 && j>0)     col[ict++] = ltog[row - Xm - 1]; /* south-west */
      if (j>0)            col[ict++] = ltog[row - Xm];     /* south */
      if (i<mx1 && j>0)   col[ict++] = ltog[row - Xm + 1]; /* south-east */
      if (i>0)            col[ict++] = ltog[row - 1];      /* west */
                          col[ict++] = ltog[row];          /* center */
      if (i<mx1)          col[ict++] = ltog[row + 1];      /* east */
      if (i>0 && j<my1)   col[ict++] = ltog[row + Xm - 1]; /* north-west */
      if (j<my1)          col[ict++] = ltog[row + Xm];     /* north */
      if (i<mx1 && j<my1) col[ict++] = ltog[row + Xm + 1]; /* north-east */
      ierr = MatSetValues(A,1,&grow,ict,col,val,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFinalize(); exit(0);
  return 0;
}
