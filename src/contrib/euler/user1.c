#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: user1.c,v 1.69 1997/10/19 19:50:13 curfman Exp curfman $";
#endif

/***************************************************************************

  Application Description: 3D Euler Flow

  This file contains the driver program and some support routines for
  the PETSc interface to the Julianne code for a 3D Euler model.  The
  problem domain is a logically regular C-grid, where there are five
  degrees of freedom per node (corresponding to density, vector momentum,
  and internal energy).  The standard, 7-point 3D stencil is used via
  finite volumes.

  The driver program and most PETSc interface routines are written in C,
  while the application code, as modified only slightly from the original
  Julianne program, remains in Fortran.

  We use the PETSc nonlinear solvers (SNES) with pseudo-transient continuation
  to solve the nonlinear system.  We parallelize the grid using the PETSc
  distributed arrays (DAs).  Note that this current version of code does not
  use the higher level TS component of PETSc to handle the time stepping,
  although it certainly could and it will soon be modified to do so.

  This program supports only implicit treatment of boundary conditions, where
  both interior and boundary grid points contribute to the Newton systems.
  Another version of code also supports explicit boundary conditions (via
  the PETSc solvers) and provides an interface to the original (uniprocessor) 
  Julianne solver that also uses explicit boundary conditions.  We eliminated
  support for the explicit boundary conditions to reduce complexity of this
  code.

 ***************************************************************************/

static char help[] = "This program solves a 3D Euler problem, using either the\n\
original Julianne code's sequential solver or the parallel PETSc nonlinear solvers.\n\
Runtime options include:\n\
  -Nx <nx> -Ny <ny> -Nz <nz> : Number of processors in the x-, y-, z-directions\n\
  -problem <1,2,3,4,5,6>       : 1(50x10x10 grid), 2(98x18x18 grid), 3(194x34x34 grid),\n\
                               4(data structure test)\n\
                               5(duct flow 50x10x10), 6(98x18x18)\n\
  -mm_type <euler,fp,hybrid,hybrid_e,hybrid_f> : multi-model variant\n\
  -dim2                      : use 2D problem only\n\
  -angle <angle_in_degrees>  : angle of attack (default is 3.06 degrees)\n\
  -matrix_free               : Use matrix-free Newton-Krylov method\n\
    -pc_ilu_in_place         : When using matrix-free KSP with ILU(0), do so in-place\n\
    -sub_pc_ilu_in_place     : When using matrix-free KSP with ILU(0) for subblocks, do so in-place\n\
  -post                      : Compute post-processing data\n\
  -cfl_advance               : Use advancing CFL number\n\
  -cfl_max_incr              : Maximum ratio for advancing CFL number at any given step\n\
  -cfl_max_decr              : Maximum ratio for decreasing CFL number at any given step\n\
  -global_timestep           : Use global timestepping instead of the default local version\n\
  -cfl_snes_it <it>          : Number of SNES iterations at each CFL step\n\
  -f_red <fraction>          : Reduce the function norm by this fraction before advancing CFL\n\
  -use_jratio                : Use ratio of fnorm decrease for detecting when to form Jacobian\n\
  -jratio                    : Set ratio of fnorm decrease for detecting when to form Jacobian\n\
  -jfreq <it>                : frequency of forming Jacobian (once every <it> iterations)\n\
  -eps_jac <eps>             : Choose differencing parameter for FD Jacobian approx\n\
  -jac_no_wake               : Don't use wake boundary conditions in the preconditioning Jacobian\n\
  -jac_snes_fd               : Use PETSc sparse finite difference approximation of Jacobian\n\
                               (the default for problems 5 and higher)\n\
  -no_output                 : Do not print any output during SNES solve (intended for use\n\
                               during timing runs)\n\n";

static char help3[] = "Options for VRML viewing:\n\
  -vrmlevenhue               : Use uniform colors for VRML output\n\
  -vrmlnolod                 : Do not use LOD for VRML output\n\
  -dump_freq                 : Frequency for dumping output (default is every 10 iterations)\n\
  -dump_vrml_layers [num]    : Dump pressure contours for [num] layers\n\
  -dump_vrml_cut_z           : Dump pressure contours, slicing in z-direction (through the wing)\n\
                               instead of the default slices in the y-direction (around the wing),\n\
  -dump_vrml_different_files : Dump VRML output into a file (corresponding to iteration number)\n\
                               rather than the default of dumping continually into a single file\n\
  -dump_general              : Dump various fields into files (euler.[iteration].out) for\n\
                               later processing\n\n";

static char help4[] = "Options for debugging and matrix dumping:\n\
  -printg                    : Print grid information\n\
  -mat_dump -cfl_switch [cfl]: Dump linear system corresponding to this CFL in binary to file\n\
                               'euler.dat'; then exit.\n\
These shouldn't be needed anymore but were useful in testing the parallel code\n\
  -printv                    : Print various vectors\n\
  -debug                     : Activate debugging printouts (dump matices, index sets, etc.).\n\
                               Since this option dumps lots of data, it should be used for just\n\
                               a few iterations.\n\
  -bctest                    : Test scatters for boundary conditions\n\
\n\
Duct problem options:\n\
  -bump <bump>               : maximum height of bump\n\
  -bump_dump                 : Dump various fields for Matlab viewing\n\n";

/***************************************************************************/

/* user-defined include file for the Euler application */
#include "user.h"
#include "src/fortran/custom/zpetsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  MPI_Comm comm;                  /* communicator */
  SNES     snes;                  /* SNES context */
  SLES     sles;                  /* SLES context */
  Euler    *app;                  /* user-defined context */
  Viewer   view;                  /* viewer for printing vectors */
  char     stagename[2][16];      /* names of profiling stages */
  int      fort_app;              /* Fortran interface user context */
  int      logging;               /* flag indicating use of logging */
  int      solve_with_julianne;   /* flag indicating use of original Julianne solver */
  int      init1, init2, init3;   /* event numbers for application initialization */
  int      len, its, ierr, flg, stage, pprint, wing;
  /* char     filename[64], outstring[64]; */

  /* Set Defaults */
  int      total_stages = 1;      /* number of times to run nonlinear solver */
  int      log_stage_0 = 0;       /* are we doing dummy solve for logging stage 0? */
  int      maxsnes;               /* maximum number of SNES iterations */
  double   rtol = 1.e-10;         /* SNES relative convergence tolerance */
  double   time1, tsolve;         /* time for solution process */
  Scalar   c_lift, c_drag;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize PETSc and print help information
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = MPI_COMM_WORLD;
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRA(ierr);
  if (flg) {PetscPrintf(comm,help3);PetscPrintf(comm,help4);}

  /* Temporarily deactivate these events */
  ierr = PLogEventDeactivate(VEC_SetValues); CHKERRA(ierr);
  ierr = PLogEventMPEDeactivate(VEC_SetValues); CHKERRA(ierr);
  ierr = PLogEventDeactivate(MAT_SetValues); CHKERRA(ierr);
  ierr = PLogEventMPEDeactivate(MAT_SetValues); CHKERRA(ierr);

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

  ierr = OptionsHasName(PETSC_NULL,"-log_summary",&logging); CHKERRA(ierr);
  if (logging) total_stages = 2;
  ierr = PLogEventRegister(&init1,"DA, Scatter Init","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&init2,"Mesh Setup      ","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&init3,"Julianne Init   ","Red:"); CHKERRA(ierr);

  for ( stage=0; stage<total_stages; stage++ ) {
  
  /*
     Begin profiling next stage
  */
  PLogStagePush(stage);
  sprintf(stagename[stage],"Solve %d",stage);
  PLogStageRegister(stage,stagename[stage]);
  if (logging && !stage) log_stage_0 = 1;
  else                   log_stage_0 = 0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize distributed arrays and vector scatters; allocate work space
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsHasName(PETSC_NULL,"-julianne",&solve_with_julianne); CHKERRA(ierr);
  PLogEventBegin(init1,0,0,0,0);
  ierr = UserCreateEuler(comm,solve_with_julianne,log_stage_0,&app); CHKERRA(ierr);
  PLogEventEnd(init1,0,0,0,0);

  ierr = PLogEventRegister(&(app->event_pack),"PackWork        ","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&(app->event_unpack),"UnpackWork      ","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&(app->event_localf),"Local fct eval  ","Red:"); CHKERRA(ierr);
  ierr = PLogEventRegister(&app->event_monitor,"Monitoring      ","Red:"); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Read the mesh
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PLogEventBegin(init2,0,0,0,0);
  ierr = UserSetGrid(app); CHKERRA(ierr); 
  PLogEventEnd(init2,0,0,0,0);

  /* for dummy logging stage 1, form the Jacobian every 2 iterations */
  if (logging && stage==0) app->jfreq = 2;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     If desired, use original (uniprocessor) solver in Julianne code
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (solve_with_julianne) {
    if (app->size != 1) SETERRA(1,1,"Original code runs on 1 processor only.");
    ierr = OptionsGetDouble(PETSC_NULL,"-julianne_rtol",&rtol,&flg); CHKERRA(ierr);

    ierr = PackWork(app,app->da,app->X,app->localX,&app->xx); CHKERRQ(ierr);
    ierr = PackWork(app,app->da,app->F,app->localDX,&app->dxx); CHKERRQ(ierr);

    PLogEventBegin(init3,0,0,0,0);
    *(int*) (&fort_app) = PetscFromPointer(app);
    time1 = PetscGetTime();

    ierr = julianne_(&time1,&solve_with_julianne,&fort_app,&app->cfl,
           &rtol,&app->eps_jac,app->b1,app->b2,
           app->b3,app->b4,app->b5,app->b6,app->diag,app->dt,
           app->xx,app->p,app->dxx,
           app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,app->xc,app->yc,app->zc,
           app->f1,app->g1,app->h1,
           app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,
           &app->angle,&app->jfreq); CHKERRA(ierr);
    tsolve = PetscGetTime() - time1;
    PLogEventEnd(init3,0,0,0,0);
    PetscPrintf(comm,"Julianne solution time = %g seconds\n",tsolve);
    PetscFinalize();
    return 0;
  } 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Otherwise, initialize application data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PLogEventBegin(init2,0,0,0,0);
  *(int*) (&fort_app) = PetscFromPointer(app);
  time1 = PetscGetTime();

  solve_with_julianne = 0;
  ierr = julianne_(&time1,&solve_with_julianne,&fort_app,&app->cfl,
         &rtol,&app->eps_jac,app->b1,
         app->b2,app->b3,app->b4,app->b5,app->b6,app->diag,app->dt,
         app->xx,app->p,app->dxx,
         app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
         app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
         app->aiz,app->ajz,app->akz,app->xc,app->yc,app->zc,
         app->f1,app->g1,app->h1,
         app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,
         &app->angle,&app->jfreq); CHKERRA(ierr);
  ierr = GetWingCommunicator(app,&app->fort_wing_comm,&wing); CHKERRQ(ierr);
  PLogEventEnd(init2,0,0,0,0);

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
  ierr = SNESSetFunction(snes,app->F,ComputeFunctionBasic,app); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,MonitorEuler,app); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Set default options; these must precede SNESSetFromOptions() so that
     they can be overridden at runtime if desired */

  ierr = SNESGetSLES(snes,&sles); CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&app->ksp); CHKERRA(ierr);
  ierr = KSPSetType(app->ksp,KSPGMRES); CHKERRA(ierr);
  ierr = KSPSetTolerances(app->ksp,app->ksp_rtol_max,PETSC_DEFAULT,
         PETSC_DEFAULT,app->ksp_max_it); CHKERRA(ierr); 
  ierr = KSPGMRESSetRestart(app->ksp,app->ksp_max_it+1); CHKERRA(ierr);
  ierr = KSPGMRESSetOrthogonalization(app->ksp,
         KSPGMRESUnmodifiedGramSchmidtOrthogonalization); CHKERRA(ierr); 
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,rtol,
                                1.e-13,3000,100000); CHKERRA(ierr);

  /* Use the Eisenstat-Walker method to set KSP convergence tolerances.  Set the
     initial and maximum relative tolerance for the linear solvers to ksp_rtol_max */
  /* ierr = SNES_KSP_SetConvergenceTestEW(snes); CHKERRA(ierr);
  ierr = SNES_KSP_SetParametersEW(snes,PETSC_DEFAULT,app->ksp_rtol_max,app->ksp_rtol_max,
         PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRA(ierr); */

  /* Set runtime options (e.g. -snes_rtol <rtol> -ksp_type <type>) */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* We use just a few iterations if doing the "dummy" logging phase.  We
     call this after SNESSetFromOptions() to override any runtime options */
  if (logging && stage==0) {ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,
                                   PETSC_DEFAULT,3,PETSC_DEFAULT); CHKERRA(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set an application-specific convergence test 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESGetTolerances(snes,PETSC_NULL,PETSC_NULL,PETSC_NULL,&maxsnes,
         PETSC_NULL); CHKERRA(ierr);
  len = maxsnes + 1;

  app->farray = (Scalar *)PetscMalloc(8*len*sizeof(Scalar)); CHKPTRQ(app->farray);
  PetscMemzero(app->farray,8*len*sizeof(Scalar));
  app->favg     = app->farray   + len;
  app->flog     = app->favg     + len;
  app->ftime    = app->flog     + len;
  app->fcfl     = app->ftime    + len;
  app->lin_rtol = app->fcfl     + len;
  app->c_lift   = app->lin_rtol + len;
  app->c_drag   = app->c_lift   + len;
  app->lin_its  = (int *)PetscMalloc(2*len*sizeof(int)); CHKPTRQ(app->lin_its);
  app->nsup     = app->lin_its  + len;
  PetscMemzero(app->lin_its,2*len*sizeof(int));
  ierr = SNESSetConvergenceHistory(snes,app->farray,maxsnes); CHKERRA(ierr);
  ierr = SNESSetConvergenceTest(snes,ConvergenceTestEuler,app); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Transfer application data to vector X and view if desired */
  ierr = InitialGuess(snes,app,app->X); CHKERRA(ierr);
  if (app->print_vecs) {
    ierr = ViewerFileOpenASCII(comm,"init.out",&view); CHKERRA(ierr);
    ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRA(ierr);
    ierr = DFVecView(app->X,view); CHKERRA(ierr);
    ierr = ViewerDestroy(view); CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  app->time_init = PetscGetTime();
  ierr = SNESSolve(snes,app->X,&its); CHKERRA(ierr);
  PetscPrintf(comm,"number of Newton iterations = %d\n\n",its);

  /* Print solver parameters */
  ierr = SNESView(snes,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Do post-processing
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Print convergence stats.  Note that we save this data in arrays for
     printing only after the completion of SNESSolve(), so that the timings
     are not distorted by output during the solve. */
  /* temporarily this is printed within the monitoring routine */
  /*
  if (app->no_output) {
    int i, overlap;
    if (app->rank == 0) {
      overlap = 0;
      ierr = OptionsGetInt(PETSC_NULL,"-pc_asm_overlap",&overlap,&flg); CHKERRQ(ierr);
      if (app->problem == 1) {
        sprintf(filename,"f_m6%s_cc%d_asm%d_p%d.m","c",app->cfl_snes_it,overlap,app->size);
        sprintf(outstring,"zsnes_m6%s_cc%d_asm%d_p%d = [\n","c",app->cfl_snes_it,overlap,app->size);
      }
      else if (app->problem == 2) {
        sprintf(filename,"f_m6%s_cc%d_asm%d_p%d.m","f",app->cfl_snes_it,overlap,app->size);
        sprintf(outstring,"zsnes_m6%s_cc%d_asm%d_p%d = [\n","f",app->cfl_snes_it,overlap,app->size);
      }
      else if (app->problem == 3) {
        sprintf(filename,"f_m6%s_cc%d_asm%d_p%d.m","n",app->cfl_snes_it,overlap,app->size);
        sprintf(outstring,"zsnes_m6%s_cc%d_asm%d_p%d = [\n","n",app->cfl_snes_it,overlap,app->size);
      }
      else if (app->problem == 5) {
        sprintf(filename,"f_duct%s_asm%d_p%d.m","c",overlap,app->size);
        sprintf(outstring,"zsnes_ductc_asm%d_p%d = [\n",overlap,app->size);
      } 
      else if (app->problem == 6) {
        sprintf(filename,"f_duct%s_asm%d_p%d.m","f",overlap,app->size);
        sprintf(outstring,"zsnes_ductf_asm%d_p%d = [\n",overlap,app->size);
      } 
      else SETERRQ(1,0,"No support for this problem number");
      app->fp = fopen(filename,"w"); 
      if (!app->fp) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open output file");
      fprintf(app->fp,"%% iter, fnorm2, log(fnorm2), CFL#, time, ksp_its, ksp_rtol, c_lift, c_drag, nsup\n");
      fprintf(app->fp,outstring);
      for (i=0; i<=its; i++)
        fprintf(app->fp," %5d  %8.4e  %8.4f  %8.1f  %10.2f  %4d  %7.3e  %8.4e  %8.4e  %8d\n",
                i,app->farray[i],app->flog[i],app->fcfl[i],app->ftime[i],app->lin_its[i],
                app->lin_rtol[i],app->c_lift[i],app->c_drag[i],app->nsup[i]);
    }
  }
  */
  if (app->rank == 0) {
    fprintf(app->fp," ];\n%% Total SLES iters = %d, Total fct evals = %d, Total time = %g sec\n",app->sles_tot,app->fct_tot,app->ftime[its]);
    fclose(app->fp);
  }

  /* - - - - - Dump fields for later viewing with VRML - - - - - */

  if (app->post_process) {

    /* First pack local ghosted vector; then compute pressure and dump to file */
    ierr = PackWork(app,app->da,app->X,app->localX,&app->xx); CHKERRA(ierr);
    ierr = jpressure_(app->xx,app->p); CHKERRA(ierr);

    /* Calculate physical quantities of interest */
    pprint = 1;
    ierr = pvar_(app->xx,app->p,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,app->xc,app->yc,app->zc,&pprint,
           &c_lift,&c_drag,&app->fort_wing_comm,&app->wing); CHKERRA(ierr);

    /* Dump all fields for general viewing */
    ierr = MonitorDumpGeneral(snes,app->X,app); CHKERRA(ierr);

    /* Dump for VRML viewing */
    ierr = MonitorDumpVRML(snes,app->X,app->F,app); CHKERRA(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-bump_dump",&flg); CHKERRQ(ierr);
  if (flg && app->problem >= 5) {
    Scalar *xa;
    ierr = VecGetArray(app->X,&xa); CHKERRQ(ierr);
    its = -1;
    if (app->mmtype == MMFP) {
      ierr = VisualizeFP_Matlab(its,app,xa); CHKERRQ(ierr);
    } else if (app->mmtype == MMEULER) {
      ierr = VisualizeEuler_Matlab(its,app,xa); CHKERRQ(ierr);
    } else SETERRQ(1,0,"Option not supported yet");
    ierr = VecRestoreArray(app->X,&xa); CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free data structures 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /* problem with destroy on Origin2000 */
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = UserDestroyEuler(app); CHKERRA(ierr);

  /* Conclude profiling this stage; need a barrier before beginning profiling
     of the next stage */
  PLogStagePop();
  MPI_Barrier(comm);

  /* -----------------------------------------------------------
                      End of nonlinear solver loop
     ----------------------------------------------------------- */
  }

  PetscFinalize();
  return 0;
}
#undef __FUNC__
#define __FUNC__ "UserDestroyEuler"
/***************************************************************************/
/* 
    UserDestroyEuler - Destroys the user-defined application context.

   Input Parameter:
   app - application-defined context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserDestroyEuler(Euler *app)
{
  int  ierr;
  if (app->J) {ierr = MatDestroy(app->J); CHKERRQ(ierr);}
  if (app->matrix_free) {ierr = MatDestroy(app->Jmf); CHKERRQ(ierr);}
  if (app->Fvrml) {ierr = VecDestroy(app->Fvrml); CHKERRQ(ierr);}
  ierr = VecDestroy(app->X); CHKERRQ(ierr);
  ierr = VecDestroy(app->Xbc); CHKERRQ(ierr);
  ierr = VecDestroy(app->F); CHKERRQ(ierr);
  ierr = VecDestroy(app->localX); CHKERRQ(ierr);
  ierr = VecDestroy(app->localDX); CHKERRQ(ierr);
  ierr = VecDestroy(app->localXBC); CHKERRQ(ierr);
  ierr = DADestroy(app->da); CHKERRQ(ierr);
  ierr = VecScatterDestroy(app->Xbcscatter); CHKERRQ(ierr);
  ierr = MMDestroy(app->multimodel); CHKERRQ(ierr);
  PetscFree(app->label);
  if (app->is1) PetscFree(app->is1);

  if (app->bctype != IMPLICIT || app->dump_vrml || app->dump_general || app->post_process) {
    ierr = VecScatterDestroy(app->Pbcscatter); CHKERRQ(ierr);
    ierr = DADestroy(app->da1); CHKERRQ(ierr);
    ierr = VecDestroy(app->Pbc); CHKERRQ(ierr);
    ierr = VecDestroy(app->P); CHKERRQ(ierr);
    ierr = VecDestroy(app->localP); CHKERRQ(ierr);
  }
  if (app->bctype == IMPLICIT) {
    if (!app->mat_assemble_direct) PetscFree(app->b1bc);
  }
  ierr = VecDestroy(app->vcoord); CHKERRQ(ierr);
  if (app->fdcoloring) {ierr = MatFDColoringDestroy(app->fdcoloring); CHKERRQ(ierr);}

  /* Free misc work space for Fortran arrays */
  if (app->farray)  PetscFree(app->farray);
  if (app->dt)      PetscFree(app->dt);
  if (app->diag)    PetscFree(app->diag);
  if (app->b1)      PetscFree(app->b1);
  if (app->work_p)  PetscFree(app->work_p);
  if (app->f1)      PetscFree(app->f1);
  if (app->sp)      PetscFree(app->sp);
  if (app->sadai)   PetscFree(app->sadai);
  if (app->bl)      PetscFree(app->bl);
  if (app->lin_its) PetscFree(app->lin_its);
  if (app->p)       PetscFree(app->p);
  PetscFree(app);

  /* If fp or multimodel */
  if (app->mmtype != MMEULER) {
    ierr = VecDestroy(app->den); CHKERRQ(ierr);
    ierr = VecDestroy(app->xvel); CHKERRQ(ierr);
    ierr = VecDestroy(app->yvel); CHKERRQ(ierr);
    ierr = VecDestroy(app->zvel); CHKERRQ(ierr);
  }

  return 0;
}
#undef __FUNC__
#define __FUNC__ "ComputeFunctionNoWake"
/***************************************************************************/
/*
   ComputeFunctionNoWake - Evaluates the nonlinear function, F(X), setting
   wake BC components to 0 if the option -jac_no_wake is used and we are
   doing a Jacobian evaluation.
 */
int ComputeFunctionNoWake(SNES snes,Vec X,Vec Fvec,void *ptr)
{
  int jacform = 1;
  return ComputeFunctionCore(jacform,snes,X,Fvec,ptr); 
}

#undef __FUNC__
#define __FUNC__ "ComputeFunctionBasic"
/***************************************************************************/
/*
   ComputeFunctionBasic - Evaluates the nonlinear function, F(X).
 */
int ComputeFunctionBasic(SNES snes,Vec X,Vec Fvec,void *ptr)
{
  int jacform = 0;
  return ComputeFunctionCore(jacform,snes,X,Fvec,ptr); 
}

#undef __FUNC__
#define __FUNC__ "ComputeFunctionCore"
/***************************************************************************/
/*
   ComputeFunctionCore - Evaluates the nonlinear function, F(X).  

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
      F(X) : (dr,dru,drv,drw,de) in Julianne code
      X    : (r,ru,rv,rw,e) in Julianne code
   We pack/unpack these work arrays with the routines PackWork()
   and UnpackWork().

   This routine supports only the implicit mode of handling boundary
   conditions.    Another version of code also supports explicit boundary
   conditions; we omit this capability here to reduce code complexity.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int ComputeFunctionCore(int jacform,SNES snes,Vec X,Vec Fvec,void *ptr)
{
  Euler  *app = (Euler *)ptr;
  int    ierr, base_unit, iter;
  Scalar zero = 0.0, *fv_array;

  if (app->bctype != IMPLICIT) SETERRQ(1,0,"This version supports only implicit BCs!");
  if (jacform != 0 && jacform != 1) SETERRQ(1,0,"Must use jacform = 0 or 1 only!");
  app->fct_tot++;

  /* Initialize vector to zero.  These values will be overwritten everywhere but
     the edges of the 3D domain */
  ierr = VecSet(&zero,Fvec); CHKERRQ(ierr);
  ierr = VecGetArray(Fvec,&fv_array); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Do setup (not required for the first function evaluation)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->first_time_resid) {
    app->first_time_resid = 0;
  } else { 
    /* Convert current iterate to Julianne format */
    ierr = PackWork(app,app->da,X,app->localX,&app->xx); CHKERRQ(ierr);

    /* Do scatters for implict BCs */
    if (app->mmtype != MMFP) {
      ierr = BoundaryConditionsImplicit(app,X); CHKERRQ(ierr);
    }
  }

  if (app->mmtype == MMEULER || app->mmtype == MMHYBRID_E || app->mmtype == MMHYBRID_EF1) {
    /* As long as we're not doing just the full potential model, we must
       compute the Euler components */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
          Local computation
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    PLogEventBegin(app->event_localf,0,0,0,0);

    ierr =localfortfct_euler_(&jacform,fv_array,
           app->xx,app->p,app->xx_bc,app->p_bc,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,
           app->dxx,app->br,app->bl,app->be,app->f1,app->g1,app->h1,
           app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2); CHKERRQ(ierr);

    PLogEventEnd(app->event_localf,0,0,0,0);

    /* Compute pseudo-transient continuation array, dt.  Really need to
       recalculate dt only when the iterates change.  DT is used to modify
       the diagonal of the Jacobian matrix.  */
    if (app->sctype == DT_MULT && !app->matrix_free_mult) {
      eigenv_(app->dt,app->xx,app->p,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,&app->ts_type);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
          Assemble vector Fvec(X)
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(ACTIVATE_OLD_ASSEMBLY)
    if (app->use_vecsetvalues) {

      /* Transform Fvec to a Fortran vector */
      ierr = PetscCObjectToFortranObject(Fvec,&fortvec); CHKERRQ(ierr);

      /* Build Fvec(X) using VecSetValues() */
      ierr = rbuild_(&fortvec, &app->sctype, app->dt, app->dxx, fv_array,
           app->ltog, &app->nloc ); CHKERRQ(ierr);
    } else {
#endif
      /* Build Fvec(X) directly, without using VecSetValues() */
      ierr = rbuild_direct_euler_(fv_array, &app->sctype, app->dt, app->dxx );  CHKERRQ(ierr);
#if defined(ACTIVATE_OLD_ASSEMBLY)
    }
#endif

  } else if (app->mmtype == MMFP || app->mmtype == MMHYBRID_F) {
      /* No Euler computations needed here */
  } else SETERRQ(1,0,"Unsupported model type");

  if (app->mmtype == MMFP || app->mmtype == MMHYBRID_F || app->mmtype == MMHYBRID_EF1) {

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
          Full potential code
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* fv_array[0] = 1.e-13; */

    /* Full potential function call */
    ierr =localfortfct_fp_(&jacform,fv_array,
           app->xx,app->p,app->xx_bc,app->p_bc,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,app->dxx,
           app->den_a,app->xvel_a,app->yvel_a,app->zvel_a,
           app->xc,app->yc,app->zc); CHKERRQ(ierr); 

    /* Build Fvec(X) directly, without using VecSetValues() */
    ierr = rbuild_direct_fp_(fv_array, &app->sctype, app->dt, app->dxx );  CHKERRQ(ierr); 

  } else if (app->mmtype == MMEULER || app->mmtype == MMHYBRID_E) {
     /* no FP computations needed here */
  } else SETERRQ(1,0,"Unsupported model type");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
          Optional output for debugging and visualizing solution 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (!app->no_output) {
    /* Output various fields into a general-format file for later viewing */
    if (app->dump_general) {
      ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
      if (!(iter%app->dump_freq)) ierr = MonitorDumpGeneral(snes,X,app);
    }

    /* Output various fields directly into a VRML-format file */
    if (app->dump_vrml) {
      ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
      if (!(iter%app->dump_freq)) ierr = MonitorDumpVRML(snes,X,Fvec,app);
      MonitorDumpIter(iter);
    }

    /* Print Fortran arrays (debugging) */
    if (app->print_debug) {
      base_unit = 60;
      printgjul_(app->dxx,app->p,&base_unit);
      base_unit = 70;
      printgjul_(app->xx,app->p,&base_unit);
    }
  }

  return 0;
}
#undef __FUNC__
#define __FUNC__ "InitialGuess"
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

  ierr = UnpackWork(app,app->da,app->xx,app->localX,X); CHKERRQ(ierr);

  /* If testing scatters for boundary conditions, then replace actual
     initial values with a test vector. */
  if (app->bc_test) {
    ierr = GridTest(app); CHKERRQ(ierr);
    ierr = PackWork(app,app->da,X,app->localX,&app->xx); CHKERRQ(ierr);
  }

  if (app->mmtype != MMFP && app->mmtype != MMHYBRID_F) {
    /* PetscPrintf(app->comm,"Dummy FP: Setting all full potential Jacobian diagonal components to 1\n"); */

    /* Apply boundary conditions */
    ierr = BoundaryConditionsExplicit(app,X); CHKERRQ(ierr);

    /* Destroy pressure scatters for boundary conditions, since we needed
       them only to computate the initial guess */
    if (app->bctype == IMPLICIT && !app->dump_vrml && !app->dump_general && !app->post_process) {
      ierr = VecScatterDestroy(app->Pbcscatter); CHKERRQ(ierr);
      ierr = DADestroy(app->da1); CHKERRQ(ierr);
      ierr = VecDestroy(app->Pbc); CHKERRQ(ierr);
      ierr = VecDestroy(app->P); CHKERRQ(ierr);
      ierr = VecDestroy(app->localP); CHKERRQ(ierr);
    }

    /* Exit if we're just testing scatters for boundary conditions */
    if (app->bc_test) {UserDestroyEuler(app); PetscFinalize(); exit(0);}
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "UserCreateEuler"
/***************************************************************************/
/* 
   UserCreateEuler - Defines the application-specific data structure and
   initializes the PETSc distributed arrays.

   Input Parameters:
   comm - MPI communicator
   solve_with_julianne - are we using the Julianne solver?
   log_stage_0 - are we doing the initial dummy setup for log_summary?

   Output Parameter:
   newapp - user-defined application context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserCreateEuler(MPI_Comm comm,int solve_with_julianne,int log_stage_0,Euler **newapp)
{
  Euler *app;
  AO    ao;           /* application ordering context */
  int   ni1;          /* x-direction grid dimension */
  int   nj1;	      /* y-direction grid dimension */
  int   nk1;	      /* z-direction grid dimension */
  int   Nx, Ny, Nz;   /* number of processors in each direction */
  int   Nlocal, ierr, flg, llen, llenb, fort_comm, problem = 1, ndof, ndof_e;
  char  *mmname;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Create user-defined application context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  app = PetscNew(Euler); CHKPTRQ(app);
  PetscMemzero(app,sizeof(Euler));
  app->comm = comm;
  MPI_Comm_size(comm,&app->size);
  MPI_Comm_rank(comm,&app->rank);
  if (solve_with_julianne && app->size != 1)
    SETERRQ(1,1,"Julianne solver is uniprocessor only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Set problem parameters and flags 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsGetInt(PETSC_NULL,"-problem",&problem,&flg); CHKERRQ(ierr);
  /* force use of problem 1 for dummy logging stage 0 */
  if (log_stage_0) problem = 1;
  app->problem = problem;
  switch (problem) {
    case 1:
    /* full grid dimensions, including all boundaries */
      ni1 = 50; nj1 = 10; nk1 = 10;
    /* wing points, used to define BC scatters.  These are analogs
       in C of Fortran points in input (shifted by -2 for explicit formulation) 
       from m6c: Fortran: itl=10, itu=40, ile=25, ktip=6 */
      app->ktip = 4; app->itl = 8; app->itu = 38; app->ile = 23;   
      app->eps_jac        = 1.0e-7;
      app->eps_mf_default = 1.0e-6;
      app->cfl_snes_it    = 1;
      app->ksp_max_it     = 20;   /* max number of KSP iterations */
      app->f_reduction    = 0.3;  /* fnorm reduction before beginning to advance CFL */
      app->mm_xs          = 0;
      app->mm_xe          = 0;
      break;
    case 2:
      /* from m6f: Fortran: itl=19, itu=79, ile=49, ktip=11 */
      ni1 = 98; nj1 = 18; nk1 = 18;
      app->ktip = 9; app->itl = 17; app->itu = 77; app->ile = 47;   
      app->eps_jac        = 1.0e-7;
      app->eps_mf_default = 1.52e-5;
      app->cfl_snes_it    = 1;
      app->ksp_max_it     = 20; 
      app->f_reduction    = 0.3;
      app->mm_xs          = 0;
      app->mm_xe          = 0;
      break;
    case 3:
      /* from m6n: Fortran: itl=37, itu=157, ile=97, ktip=21 */
      ni1 = 194; nj1 = 34; nk1 = 34;
      app->ktip = 19; app->itl = 35; app->itu = 155; app->ile = 95;   
      app->eps_jac        = 1.0e-7;
      app->eps_mf_default = 1.42e-5;
      app->cfl_snes_it    = 1;
      app->ksp_max_it     = 30;
      app->f_reduction    = 0.3; 
      app->mm_xs          = 0;
      app->mm_xe          = 0;
      break;
    case 4:
      /* test case for PETSc grid manipulations only! */
      ni1 = 4; nj1 = 5; nk1 = 6;
      app->ktip = 0; app->itl = 0; app->itu = 0; app->ile = 0;
      ierr = OptionsGetInt(PETSC_NULL,"-ni1",&ni1,&flg); CHKERRQ(ierr);
      ierr = OptionsGetInt(PETSC_NULL,"-nj1",&nj1,&flg); CHKERRQ(ierr);
      ierr = OptionsGetInt(PETSC_NULL,"-nk1",&nk1,&flg); CHKERRQ(ierr);
      break;
    case 5:
    /* full grid dimensions, including all boundaries */
      ni1 = 50; nj1 = 10; nk1 = 10;
    /* wing points, used to define BC scatters.  These are analogs
       in C of Fortran points in input (shifted by -2 for explicit formulation) 
       from m6c_bump: Fortran: itl=-1, itu=33, ile=17, ktip=-1 */
      app->ktip = -3; app->itl = -3; app->itu = 31; app->ile = 15;   
      app->eps_jac        = 1.0e-7;
      app->eps_mf_default = 1.0e-6;
      app->cfl_snes_it    = 1;
      app->ksp_max_it     = 20;   /* max number of KSP iterations */
      app->f_reduction    = 0.3;  /* fnorm reduction before beginning to advance CFL */
      app->mm_xs          = 13;   /* Euler region: mm_xs <= i < mm_xe */
      app->mm_xe          = 39;   /* FP regions: i < mm_xs, i >= mm_xe */

      /* Options for duct problem */
      app->bump = 0.10;     /* default max bump height relative to channel height */
      ierr = OptionsGetDouble(PETSC_NULL,"-bump",&app->bump,&flg); CHKERRQ(ierr);
      PetscPrintf(app->comm,"Duct problem: bump = %g\n",app->bump);
      break;
    case 6:
    /* full grid dimensions, including all boundaries */
      ni1 = 98; nj1 = 18; nk1 = 18;
    /* wing points, used to define BC scatters.  These are analogs
       in C of Fortran points in input (shifted by -2 for explicit formulation) 
       from m6c_bump: Fortran: itl=-1, itu=33, ile=17, ktip=-1 */
      app->ktip = -3; app->itl = -3; app->itu = 63; app->ile = 31;   
      app->eps_jac        = 1.0e-7;
      app->eps_mf_default = 1.0e-6;
      app->cfl_snes_it    = 1;
      app->ksp_max_it     = 20;   /* max number of KSP iterations */
      app->f_reduction    = 0.3;  /* fnorm reduction before beginning to advance CFL */
      app->mm_xs          = 25;
      app->mm_xe          = 75;

      /* Options for duct problem */
      app->bump = 0.10;     /* default max bump height relative to channel height */
      ierr = OptionsGetDouble(PETSC_NULL,"-bump",&app->bump,&flg); CHKERRQ(ierr);
      PetscPrintf(app->comm,"Duct problem: bump = %g\n",app->bump);
      ierr = OptionsGetInt(PETSC_NULL,"-nk1",&nk1,&flg); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(1,1,"Unsupported problem, only 1,2,3 (M6), 4 (test), or 5,6 (duct) supported");
  }

  /* Set various defaults */
  app->sles_tot              = 0;
  app->fct_tot               = 0;
  app->fdcoloring            = 0;
  app->cfl                   = 0;        /* Initial CFL is set within Julianne code */
  app->cfl_init              = 0;        /* Initial CFL is set within Julianne code */
  app->cfl_max               = 100000.0; /* maximum CFL value */
  app->cfl_switch            = 10.0;     /* CFL at which to dump binary linear system */
  app->cfl_begin_advancement = 0;        /* flag - indicates CFL advancement has begun */
  app->cfl_max_incr          = 2.0;      /* maximum CFL increase at any given step */
  app->cfl_max_decr          = 0.1;      /* maximum CFL decrease at any given step */
  app->cfl_advance           = CONSTANT; /* flag - by default we don't advance CFL */
  app->ts_type               = LOCAL_TS; /* type of timestepping */
  app->angle                 = 3.06;     /* default angle of attack = 3.06 degrees */
  app->fstagnate_ratio       = .01;      /* stagnation detection parameter */
  app->ksp_rtol_max          = 1.0e-2;   /* maximum KSP relative convergence tolerance */
  app->ksp_rtol_min          = 1.0e-5;   /* minimum KSP relative convergence tolerance */
  app->mat_assemble_direct   = 1;        /* by default, we assemble Jacobian directly */
  app->use_vecsetvalues      = 0;        /* flag - by default assemble local vector data directly */
  app->no_output             = 0;        /* flag - by default print some output as program runs */
  app->farray                = 0;        /* work arrays */
  app->favg                  = 0;
  app->flog                  = 0;
  app->ftime                 = 0;
  app->lin_rtol              = 0;
  app->lin_its               = 0;
  app->last_its              = 0;
  app->post_process          = 0;
  app->global_grid           = 0;
  app->no_wake               = 0;

  /* control of forming new preconditioner matrices */
  app->jfreq                 = 10;       /* default frequency of computing Jacobian matrix */
  app->jratio                = 50;       /* default ration for computing new Jacobian */
  app->use_jratio            = 0;        /* flag - are we using the Jacobian ratio test? */
  app->check_solution        = 0;        /* flag - are we checking solution components? */
  app->iter_last_jac         = 0;        /* iteration at which last Jacobian precond was formed */
  app->fnorm_last_jac        = 0;        /* || F || - last time Jacobian precond was formed */

  /* Override default with runtime options */
  ierr = OptionsGetDouble(PETSC_NULL,"-ksp_rtol_max",&app->ksp_rtol_max,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-ksp_rtol_min",&app->ksp_rtol_min,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_max_incr",&app->cfl_max_incr,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_max_decr",&app->cfl_max_decr,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-cfl_snes_it",&app->cfl_snes_it,&flg); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-cfl_advance",&flg); CHKERRQ(ierr);
  if (flg) {
    app->cfl_advance = ADVANCE;
    PetscPrintf(comm,"Begin CFL advancement at iteration 12, Cfl_snes_it=%d, CFL_max_incr=%g, CFL_max_decr=%g, ksp_rtol_max=%g\n",
    app->cfl_snes_it,app->cfl_max_incr,app->cfl_max_decr,app->ksp_rtol_max);
  }
  else PetscPrintf(comm,"CFL remains constant\n");

  ierr = OptionsHasName(PETSC_NULL,"-global_timestep",&flg); CHKERRQ(ierr);
  if (flg) app->ts_type = GLOBAL_TS;
  ierr = OptionsHasName(PETSC_NULL,"-check_solution",&app->check_solution); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-use_jratio",&app->use_jratio); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-jac_no_wake",&app->no_wake); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-jratio",&app->jratio,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_switch",&app->cfl_switch,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_max",&app->cfl_max,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-f_red",&app->f_reduction,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-eps_jac",&app->eps_jac,&flg); CHKERRQ(ierr);
  app->eps_jac_inv = 1.0/app->eps_jac;
  app->bcswitch = 10;
  ierr = OptionsGetInt(PETSC_NULL,"-bc_imperm",&app->bcswitch,&flg); CHKERRQ(ierr);
  /* if (app->bcswitch > 10) app->bcswitch = 10; */
  ierr = OptionsGetDouble(PETSC_NULL,"-stagnate_ratio",&app->fstagnate_ratio,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-angle",&app->angle,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-jfreq",&app->jfreq,&flg); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-no_output",&app->no_output); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-post",&app->post_process); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-global_grid",&app->global_grid); CHKERRA(ierr);
  /* temporarily MUST use global grid! */
  app->global_grid = 1;
  /*  if (app->post_process) app->global_grid = 1; */
  if (app->global_grid) PetscPrintf(app->comm,"Using global grid (needed for post processing only)\n");

  /* 2-dimensional variant */
  ierr = OptionsHasName(PETSC_NULL,"-dim2",&app->dim2); CHKERRA(ierr);
  app->nktot = nk1;
  if (app->dim2) {
    nk1 = 3;
    PetscPrintf(app->comm,"Running 2-dimensional problem only\n");
  }

  ierr = OptionsGetInt(PETSC_NULL,"-mm_xs",&app->mm_xs,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mm_xe",&app->mm_xe,&flg); CHKERRQ(ierr);

#if defined(ACTIVATE_OLD_ASSEMBLY)
   /* 
     We currently do not support these options; the parallel code works fine for
     these cases, but they are not as efficient, so we deactivate them now.
   */
    ierr = OptionsHasName(PETSC_NULL,"-use_vecsetvalues",&app->use_vecsetvalues); CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-mat_assemble_last",&flg); CHKERRQ(ierr);
    if (flg) app->mat_assemble_direct = 0;
#endif

  if (app->mat_assemble_direct) PetscPrintf(comm,"Problem %d (%dx%dx%d grid), assembling PETSc matrix directly: angle of attack = %g, eps_jac = %g\n",problem,ni1,nj1,nk1,app->angle,app->eps_jac);
  else PetscPrintf(comm,"Problem %d (%dx%dx%d grid), assembling PETSc matrix via translation of Eagle format: angle of attack = %g, eps_jac = %g\n",problem,ni1,nj1,nk1,app->angle,app->eps_jac);
  if (app->no_wake) PetscPrintf(comm,"No wake BCs in Jacobian (preconditioner)\n");
  if (app->dump_vrml) dump_angle_vrml(app->angle);

  app->ni1              = ni1;
  app->nj1              = nj1;
  app->nk1              = nk1;
  app->ni               = ni1 - 1;
  app->nj               = nj1 - 1;
  app->nk               = nk1 - 1;
  app->nim              = ni1 - 2;
  app->njm              = nj1 - 2;
  app->nkm              = nk1 - 2;
  app->nd	        = 7;
  app->matrix_free      = 0;
  app->matrix_free_mult = 0;
  app->first_time_resid = 1;
  app->J                = 0;
  app->Jmf              = 0;
  app->Fvrml            = 0;

  /* Create multi-model context */
  ierr = MMCreate(app->comm,&app->multimodel); CHKERRQ(ierr);
  ierr = MMSetFromOptions(app->multimodel); CHKERRQ(ierr);
  ierr = MMGetNumberOfComponents(app->multimodel,&app->ndof); CHKERRQ(ierr);
  ndof = app->ndof;
  ierr = MMGetType(app->multimodel,&app->mmtype,&mmname); CHKERRQ(ierr);
  PetscPrintf(app->comm,"Multi-model: %s, enum=%d, ndof=%d, mm_xs=%d, mm_xe=%d\n",
      mmname,app->mmtype,ndof,app->mm_xs,app->mm_xe); 

  /* Set type of formulation */
  ierr = OptionsHasName(PETSC_NULL,"-dt_mult",&flg); CHKERRQ(ierr);
  if (flg == 1) {
    app->sctype = DT_MULT;
    PetscPrintf(comm,"Pseudo-transiant variant: Multiply system by dt\n");
  } else {
    PetscPrintf(comm,"Pseudo-transiant variant: Regular use of dt\n");
    app->sctype = DT_DIV;
  }
  ierr = OptionsHasName(PETSC_NULL,"-explicit",&flg); CHKERRQ(ierr);
  if (flg) SETERRQ(1,0,"This code no longer suports explicit boundary conditions!");
  if (!flg) {
    app->bctype = IMPLICIT;
    app->ktip++; app->itl++; app->itu++; app->ile++;
    app->mx = app->ni1, app->my = app->nj1, app->mz = app->nk1;
    PetscPrintf(comm,"Using fully implicit formulation: mx=%d, my=%d, mz=%d\n",
      app->mx,app->my,app->mz);
    PetscPrintf(comm,"Not reording variables for internal computation\n");
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
  app->label = (char **) PetscMalloc(6*sizeof(char*)); CHKPTRQ(app->label);
  app->label[0] = "Density";
  app->label[1] = "Velocity: x-component";
  app->label[2] = "Velocity: y-component";
  app->label[3] = "Velocity: z-component";
  app->label[4] = "Internal Energy";
  app->label[5] = "Full potential";

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
         app->Nx,app->Ny,app->Nz,ndof,2,PETSC_NULL,PETSC_NULL,PETSC_NULL,&app->da); CHKERRQ(ierr);
  ierr = DAGetAO(app->da,&ao); CHKERRQ(ierr);
  ierr = PetscCObjectToFortranObject(ao,&app->fort_ao); CHKERRQ(ierr);

  /* Get global and local vectors */
  ierr = DAGetDistributedVector(app->da,&app->X); CHKERRQ(ierr);
  ierr = VecDuplicate(app->X,&app->Xbc); CHKERRQ(ierr);
  ierr = VecDuplicate(app->X,&app->F); CHKERRQ(ierr);
  ierr = DAGetLocalVector(app->da,&app->localX); CHKERRQ(ierr);
  ierr = VecDuplicate(app->localX,&app->localDX); CHKERRQ(ierr);
  ierr = VecDuplicate(app->localX,&app->localXBC); CHKERRQ(ierr);
  ierr = VecGetArray(app->localX,&app->xx);
  ierr = VecGetArray(app->localDX,&app->dxx);
  ierr = VecGetArray(app->localXBC,&app->xx_bc);

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
      app->mx,app->my,app->mz,ndof,ndof*app->mx*app->my*app->mz); fflush(stdout);
    ierr = VecGetLocalSize(app->X,&Nlocal); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(comm,1);
    printf("[%d] local grid %d X %d X %d with %d components per node ==> local vector dimension %d\n",
      app->rank,xm,ym,zm,ndof,Nlocal);
    fflush(stdout);
    PetscSequentialPhaseEnd(comm,1);
  }

  /* Create pressure work vector, used for explicit boundary conditions */
  ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,app->mx,app->my,app->mz,
         app->Nx,app->Ny,app->Nz,1,2,PETSC_NULL,PETSC_NULL,PETSC_NULL,&app->da1); CHKERRQ(ierr);
  ierr = DAGetDistributedVector(app->da1,&app->P); CHKERRQ(ierr);
  ierr = VecDuplicate(app->P,&app->Pbc); CHKERRQ(ierr);
  ierr = DAGetLocalVector(app->da1,&app->localP); CHKERRQ(ierr);

  /* Do only for FP or multimodel */
  if (app->mmtype != MMEULER) {
    ierr = VecDuplicate(app->localP,&app->den); CHKERRQ(ierr);
    ierr = VecDuplicate(app->localP,&app->xvel); CHKERRQ(ierr);
    ierr = VecDuplicate(app->localP,&app->yvel); CHKERRQ(ierr);
    ierr = VecDuplicate(app->localP,&app->zvel); CHKERRQ(ierr);
    ierr = VecGetArray(app->den,&app->den_a); CHKERRQ(ierr);
    ierr = VecGetArray(app->xvel,&app->xvel_a); CHKERRQ(ierr);
    ierr = VecGetArray(app->yvel,&app->yvel_a); CHKERRQ(ierr);
    ierr = VecGetArray(app->zvel,&app->zvel_a); CHKERRQ(ierr);
  }

  app->xe     = app->xs + app->xm;
  app->ye     = app->ys + app->ym;
  app->ze     = app->zs + app->zm;
  app->gxe    = app->gxs + app->gxm;
  app->gye    = app->gys + app->gym;
  app->gze    = app->gzs + app->gzm;
  app->gdim   = app->mx * app->my * app->mz * ndof;
  app->ldim   = app->xm * app->ym * app->zm * ndof;
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

  ierr = MPICCommToFortranComm(comm,&fort_comm);
  ierr = parsetup_(&fort_comm, &app->print_grid, &app->no_wake, &app->no_output,
            &app->bctype, &app->rank, &app->size, &problem,
            &app->gxsf, &app->gysf, &app->gzsf,
            &app->gxef, &app->gyef, &app->gzef, 
            &app->xsf, &app->ysf, &app->zsf, 
            &app->xef, &app->yef, &app->zef,
            &app->xefm1, &app->yefm1, &app->zefm1, 
            &app->xefp1, &app->yefp1, &app->zefp1, 
            &app->gxefp1, &app->gyefp1, &app->gzefp1,
            &app->gxefm1, &app->gyefm1, &app->gzefm1,
            &app->xsf1, &app->ysf1, &app->zsf1, 
            &app->gxsf1, &app->gysf1, &app->gzsf1,
            &app->xsf2, &app->ysf2, &app->zsf2, 
            &app->gxsf2, &app->gysf2, &app->gzsf2,
            &app->gxsfw, &app->gysfw, &app->gzsfw,
            &app->gxefw, &app->gyefw, &app->gzefw,
            &app->gxm, &app->gym, &app->gzm,
            &app->xef01, &app->yef01, &app->zef01,
            &app->gxef01, &app->gyef01, &app->gzef01,
            &ndof, &app->global_grid, &app->bcswitch, 
            &app->mmtype,&app->bump,&nk1,&app->mm_xs,&app->mm_xe); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Allocate local Fortran work space
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Fortran work arrays for vectors */
  llen = (app->gxefp1 - app->gxsf1+1) * (app->gyefp1 - app->gysf1+1) 
            * (app->gzefp1 - app->gzsf1+1);
  llenb = 7*llen*sizeof(Scalar);
  app->p = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->p);
  PetscMemzero(app->p,llenb);
  app->p_bc  = app->p     + llen; /* parallel bc work space */
  app->r_bc  = app->p_bc  + llen;
  app->ru_bc = app->r_bc  + llen;
  app->rv_bc = app->ru_bc + llen;
  app->rw_bc = app->rv_bc + llen;
  app->e_bc  = app->rw_bc + llen;

  /* Fortran work arrays for matrix (diagonal) blocks */
  llen = (app->xefp1 - app->gxsf1 + 1) * (app->yefp1 - app->gysf1 + 1) 
          * (app->zefp1 - app->gzsf1 + 1) * ndof * ndof;
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
      llen = app->xm * app->zm * ndof * ndof * 8
              + app->xm * app->ym * ndof * ndof * 4
              + app->zm * app->ym * ndof * ndof * 4;
      llenb = llen*sizeof(Scalar);
      app->b1bc     = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->b1bc);
      PetscMemzero(app->b1bc,llenb);
      app->b2bc     = app->b1bc + app->ym * app->zm * ndof * ndof * 4;
      app->b2bc_tmp = app->b2bc + app->xm * app->zm * ndof * ndof * 4;
      app->b3bc     = app->b2bc_tmp + app->xm * app->zm * ndof * ndof * 4;
    }
  }

  /* Work space for pseudo-transient continuation term, dt */
  llenb = ((app->xefp1 - app->xsf1 + 1) * (app->yefp1 - app->ysf1 + 1) 
          * (app->zefp1 - app->zsf1 + 1)) * sizeof(Scalar);
  app->dt = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->dt);
  PetscMemzero(app->dt,llenb);

  /* Work space for building main diagonal block of Jacobian */
  ndof_e = 5;
  llenb = (app->xef01 - app->xsf2 + 1) * (app->yef01 - app->ysf2 + 1) 
          * (app->zef01 - app->zsf2 + 1) * ndof_e * ndof_e * sizeof(Scalar);
  app->diag = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->diag);
  PetscMemzero(app->diag,llenb);
  app->diag_len = llenb;

  /* Fortran work arrays for eigen info */
  llen = (app->gxefp1 - app->gxsf1 + 1) * (app->gyefp1 - app->gysf1 + 1) 
            * (app->gzefp1 - app->gzsf1 + 1) * ndof * ndof;
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
  llen = ndof * (app->xef01 - app->gxsf1+1) * (app->yef01 - app->gysf1+1) 
            * (app->zef01 - app->gzsf1+1);
  llenb = 3*llen*sizeof(Scalar);
  app->f1 = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->f1);
  PetscMemzero(app->f1,llenb);
  app->g1 = app->f1 + llen;
  app->h1 = app->g1 + llen;

  llen = ndof * app->ni;
  llenb = 6*llen*sizeof(Scalar);
  app->sp  = (Scalar *)PetscMalloc(llenb); CHKPTRQ(app->sp);
  PetscMemzero(app->sp,llenb);
  app->sp1 = app->sp  + llen;
  app->sp2 = app->sp1 + llen;
  app->sm  = app->sp2 + llen;
  app->sm1 = app->sm  + llen;
  app->sm2 = app->sm1 + llen;

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
