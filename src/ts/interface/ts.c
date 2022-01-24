#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscdmshell.h>
#include <petscdmda.h>
#include <petscviewer.h>
#include <petscdraw.h>
#include <petscconvest.h>

#define SkipSmallValue(a,b,tol) if (PetscAbsScalar(a)< tol || PetscAbsScalar(b)< tol) continue;

/* Logging support */
PetscClassId  TS_CLASSID, DMTS_CLASSID;
PetscLogEvent TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

const char *const TSExactFinalTimeOptions[] = {"UNSPECIFIED","STEPOVER","INTERPOLATE","MATCHSTEP","TSExactFinalTimeOption","TS_EXACTFINALTIME_",NULL};

static PetscErrorCode TSAdaptSetDefaultType(TSAdapt adapt,TSAdaptType default_type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidCharPointer(default_type,2);
  if (!((PetscObject)adapt)->type_name) {
    ierr = TSAdaptSetType(adapt,default_type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSSetFromOptions - Sets various TS parameters from user options.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
+  -ts_type <type> - TSEULER, TSBEULER, TSSUNDIALS, TSPSEUDO, TSCN, TSRK, TSTHETA, TSALPHA, TSGLLE, TSSSP, TSGLEE, TSBSYMP, TSIRK
.  -ts_save_trajectory - checkpoint the solution at each time-step
.  -ts_max_time <time> - maximum time to compute to
.  -ts_max_steps <steps> - maximum number of time-steps to take
.  -ts_init_time <time> - initial time to start computation
.  -ts_final_time <time> - final time to compute to (deprecated: use -ts_max_time)
.  -ts_dt <dt> - initial time step
.  -ts_exact_final_time <stepover,interpolate,matchstep> - whether to stop at the exact given final time and how to compute the solution at that time
.  -ts_max_snes_failures <maxfailures> - Maximum number of nonlinear solve failures allowed
.  -ts_max_reject <maxrejects> - Maximum number of step rejections before step fails
.  -ts_error_if_step_fails <true,false> - Error if no step succeeds
.  -ts_rtol <rtol> - relative tolerance for local truncation error
.  -ts_atol <atol> Absolute tolerance for local truncation error
.  -ts_rhs_jacobian_test_mult -mat_shell_test_mult_view - test the Jacobian at each iteration against finite difference with RHS function
.  -ts_rhs_jacobian_test_mult_transpose -mat_shell_test_mult_transpose_view - test the Jacobian at each iteration against finite difference with RHS function
.  -ts_adjoint_solve <yes,no> After solving the ODE/DAE solve the adjoint problem (requires -ts_save_trajectory)
.  -ts_fd_color - Use finite differences with coloring to compute IJacobian
.  -ts_monitor - print information at each timestep
.  -ts_monitor_cancel - Cancel all monitors
.  -ts_monitor_lg_solution - Monitor solution graphically
.  -ts_monitor_lg_error - Monitor error graphically
.  -ts_monitor_error - Monitors norm of error
.  -ts_monitor_lg_timestep - Monitor timestep size graphically
.  -ts_monitor_lg_timestep_log - Monitor log timestep size graphically
.  -ts_monitor_lg_snes_iterations - Monitor number nonlinear iterations for each timestep graphically
.  -ts_monitor_lg_ksp_iterations - Monitor number nonlinear iterations for each timestep graphically
.  -ts_monitor_sp_eig - Monitor eigenvalues of linearized operator graphically
.  -ts_monitor_draw_solution - Monitor solution graphically
.  -ts_monitor_draw_solution_phase  <xleft,yleft,xright,yright> - Monitor solution graphically with phase diagram, requires problem with exactly 2 degrees of freedom
.  -ts_monitor_draw_error - Monitor error graphically, requires use to have provided TSSetSolutionFunction()
.  -ts_monitor_solution [ascii binary draw][:filename][:viewerformat] - monitors the solution at each timestep
.  -ts_monitor_solution_vtk <filename.vts,filename.vtu> - Save each time step to a binary file, use filename-%%03D.vts (filename-%%03D.vtu)
-  -ts_monitor_envelope - determine maximum and minimum value of each component of the solution over the solution time

   Notes:
     See SNESSetFromOptions() and KSPSetFromOptions() for how to control the nonlinear and linear solves used by the time-stepper.

     Certain SNES options get reset for each new nonlinear solver, for example -snes_lag_jacobian <its> and -snes_lag_preconditioner <its>, in order
     to retain them over the multiple nonlinear solves that TS uses you mush also provide -snes_lag_jacobian_persists true and
     -snes_lag_preconditioner_persists true

   Developer Note:
     We should unify all the -ts_monitor options in the way that -xxx_view has been unified

   Level: beginner

.seealso: TSGetType()
@*/
PetscErrorCode  TSSetFromOptions(TS ts)
{
  PetscBool              opt,flg,tflg;
  PetscErrorCode         ierr;
  char                   monfilename[PETSC_MAX_PATH_LEN];
  PetscReal              time_step;
  TSExactFinalTimeOption eftopt;
  char                   dir[16];
  TSIFunction            ifun;
  const char             *defaultType;
  char                   typeName[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);

  ierr = TSRegisterAll();CHKERRQ(ierr);
  ierr = TSGetIFunction(ts,NULL,&ifun,NULL);CHKERRQ(ierr);

  ierr = PetscObjectOptionsBegin((PetscObject)ts);CHKERRQ(ierr);
  if (((PetscObject)ts)->type_name) defaultType = ((PetscObject)ts)->type_name;
  else defaultType = ifun ? TSBEULER : TSEULER;
  ierr = PetscOptionsFList("-ts_type","TS method","TSSetType",TSList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSSetType(ts,typeName);CHKERRQ(ierr);
  } else {
    ierr = TSSetType(ts,defaultType);CHKERRQ(ierr);
  }

  /* Handle generic TS options */
  ierr = PetscOptionsDeprecated("-ts_final_time","-ts_max_time","3.10",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_max_time","Maximum time to run to","TSSetMaxTime",ts->max_time,&ts->max_time,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ts_max_steps","Maximum number of time steps","TSSetMaxSteps",ts->max_steps,&ts->max_steps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_init_time","Initial time","TSSetTime",ts->ptime,&ts->ptime,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_dt","Initial time step","TSSetTimeStep",ts->time_step,&time_step,&flg);CHKERRQ(ierr);
  if (flg) {ierr = TSSetTimeStep(ts,time_step);CHKERRQ(ierr);}
  ierr = PetscOptionsEnum("-ts_exact_final_time","Option for handling of final time step","TSSetExactFinalTime",TSExactFinalTimeOptions,(PetscEnum)ts->exact_final_time,(PetscEnum*)&eftopt,&flg);CHKERRQ(ierr);
  if (flg) {ierr = TSSetExactFinalTime(ts,eftopt);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-ts_max_snes_failures","Maximum number of nonlinear solve failures","TSSetMaxSNESFailures",ts->max_snes_failures,&ts->max_snes_failures,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ts_max_reject","Maximum number of step rejections before step fails","TSSetMaxStepRejections",ts->max_reject,&ts->max_reject,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_error_if_step_fails","Error if no step succeeds","TSSetErrorIfStepFails",ts->errorifstepfailed,&ts->errorifstepfailed,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_rtol","Relative tolerance for local truncation error","TSSetTolerances",ts->rtol,&ts->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_atol","Absolute tolerance for local truncation error","TSSetTolerances",ts->atol,&ts->atol,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ts_rhs_jacobian_test_mult","Test the RHS Jacobian for consistency with RHS at each solve ","None",ts->testjacobian,&ts->testjacobian,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_rhs_jacobian_test_mult_transpose","Test the RHS Jacobian transpose for consistency with RHS at each solve ","None",ts->testjacobiantranspose,&ts->testjacobiantranspose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_use_splitrhsfunction","Use the split RHS function for multirate solvers ","TSSetUseSplitRHSFunction",ts->use_splitrhsfunction,&ts->use_splitrhsfunction,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  {
    PetscBool set;
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ts_saws_block","Block for SAWs memory snooper at end of TSSolve","PetscObjectSAWsBlock",((PetscObject)ts)->amspublishblock,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PetscObjectSAWsSetBlock((PetscObject)ts,flg);CHKERRQ(ierr);
    }
  }
#endif

  /* Monitor options */
  ierr = PetscOptionsInt("-ts_monitor_frequency", "Number of time steps between monitor output", "TSMonitorSetFrequency", ts->monitorFrequency, &ts->monitorFrequency, NULL);CHKERRQ(ierr);
  ierr = TSMonitorSetFromOptions(ts,"-ts_monitor","Monitor time and timestep size","TSMonitorDefault",TSMonitorDefault,NULL);CHKERRQ(ierr);
  ierr = TSMonitorSetFromOptions(ts,"-ts_monitor_extreme","Monitor extreme values of the solution","TSMonitorExtreme",TSMonitorExtreme,NULL);CHKERRQ(ierr);
  ierr = TSMonitorSetFromOptions(ts,"-ts_monitor_solution","View the solution at each timestep","TSMonitorSolution",TSMonitorSolution,NULL);CHKERRQ(ierr);
  ierr = TSMonitorSetFromOptions(ts,"-ts_dmswarm_monitor_moments","Monitor moments of particle distribution","TSDMSwarmMonitorMoments",TSDMSwarmMonitorMoments,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsString("-ts_monitor_python","Use Python function","TSMonitorSet",NULL,monfilename,sizeof(monfilename),&flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscPythonMonitorSet((PetscObject)ts,monfilename);CHKERRQ(ierr);}

  ierr = PetscOptionsName("-ts_monitor_lg_solution","Monitor solution graphically","TSMonitorLGSolution",&opt);CHKERRQ(ierr);
  if (opt) {
    PetscInt       howoften = 1;
    DM             dm;
    PetscBool      net;

    ierr = PetscOptionsInt("-ts_monitor_lg_solution","Monitor solution graphically","TSMonitorLGSolution",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)dm,DMNETWORK,&net);CHKERRQ(ierr);
    if (net) {
      TSMonitorLGCtxNetwork ctx;
      ierr = TSMonitorLGCtxNetworkCreate(ts,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,600,400,howoften,&ctx);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSMonitorLGCtxNetworkSolution,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxNetworkDestroy);CHKERRQ(ierr);
      ierr = PetscOptionsBool("-ts_monitor_lg_solution_semilogy","Plot the solution with a semi-log axis","",ctx->semilogy,&ctx->semilogy,NULL);CHKERRQ(ierr);
    } else {
      TSMonitorLGCtx ctx;
      ierr = TSMonitorLGCtxCreate(PETSC_COMM_SELF,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,howoften,&ctx);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSMonitorLGSolution,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsName("-ts_monitor_lg_error","Monitor error graphically","TSMonitorLGError",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorLGCtx ctx;
    PetscInt       howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_lg_error","Monitor error graphically","TSMonitorLGError",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorLGCtxCreate(PETSC_COMM_SELF,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorLGError,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
  }
  ierr = TSMonitorSetFromOptions(ts,"-ts_monitor_error","View the error at each timestep","TSMonitorError",TSMonitorError,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsName("-ts_monitor_lg_timestep","Monitor timestep size graphically","TSMonitorLGTimeStep",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorLGCtx ctx;
    PetscInt       howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_lg_timestep","Monitor timestep size graphically","TSMonitorLGTimeStep",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorLGCtxCreate(PetscObjectComm((PetscObject)ts),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorLGTimeStep,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-ts_monitor_lg_timestep_log","Monitor log timestep size graphically","TSMonitorLGTimeStep",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorLGCtx ctx;
    PetscInt       howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_lg_timestep_log","Monitor log timestep size graphically","TSMonitorLGTimeStep",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorLGCtxCreate(PetscObjectComm((PetscObject)ts),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorLGTimeStep,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
    ctx->semilogy = PETSC_TRUE;
  }

  ierr = PetscOptionsName("-ts_monitor_lg_snes_iterations","Monitor number nonlinear iterations for each timestep graphically","TSMonitorLGSNESIterations",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorLGCtx ctx;
    PetscInt       howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_lg_snes_iterations","Monitor number nonlinear iterations for each timestep graphically","TSMonitorLGSNESIterations",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorLGCtxCreate(PetscObjectComm((PetscObject)ts),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorLGSNESIterations,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-ts_monitor_lg_ksp_iterations","Monitor number nonlinear iterations for each timestep graphically","TSMonitorLGKSPIterations",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorLGCtx ctx;
    PetscInt       howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_lg_ksp_iterations","Monitor number nonlinear iterations for each timestep graphically","TSMonitorLGKSPIterations",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorLGCtxCreate(PetscObjectComm((PetscObject)ts),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorLGKSPIterations,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-ts_monitor_sp_eig","Monitor eigenvalues of linearized operator graphically","TSMonitorSPEig",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorSPEigCtx ctx;
    PetscInt          howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_sp_eig","Monitor eigenvalues of linearized operator graphically","TSMonitorSPEig",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorSPEigCtxCreate(PETSC_COMM_SELF,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorSPEig,ctx,(PetscErrorCode (*)(void**))TSMonitorSPEigCtxDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-ts_monitor_sp_swarm","Display particle phase from the DMSwarm","TSMonitorSPSwarm",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorSPCtx  ctx;
    PetscInt        howoften = 1, retain = 0;
    PetscBool       phase = PETSC_TRUE;

    ierr = PetscOptionsInt("-ts_monitor_sp_swarm","Display particles phase from the DMSwarm", "TSMonitorSPSwarm", howoften, &howoften, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_monitor_sp_swarm_retain", "Retain n points plotted to show trajectory, -1 for all points", "TSMonitorSPSwarm", retain, &retain, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_monitor_sp_swarm_phase", "Plot in phase space rather than coordinate space", "TSMonitorSPSwarm", phase, &phase, NULL);CHKERRQ(ierr);
    ierr = TSMonitorSPCtxCreate(PetscObjectComm((PetscObject) ts), NULL, NULL, PETSC_DECIDE, PETSC_DECIDE, 300, 300, howoften, retain, phase, &ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, TSMonitorSPSwarmSolution, ctx, (PetscErrorCode (*)(void**))TSMonitorSPCtxDestroy);CHKERRQ(ierr);
  }
  opt  = PETSC_FALSE;
  ierr = PetscOptionsName("-ts_monitor_draw_solution","Monitor solution graphically","TSMonitorDrawSolution",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscInt         howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_draw_solution","Monitor solution graphically","TSMonitorDrawSolution",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts),NULL,"Computed Solution",PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorDrawSolution,ctx,(PetscErrorCode (*)(void**))TSMonitorDrawCtxDestroy);CHKERRQ(ierr);
  }
  opt  = PETSC_FALSE;
  ierr = PetscOptionsName("-ts_monitor_draw_solution_phase","Monitor solution graphically","TSMonitorDrawSolutionPhase",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscReal        bounds[4];
    PetscInt         n = 4;
    PetscDraw        draw;
    PetscDrawAxis    axis;

    ierr = PetscOptionsRealArray("-ts_monitor_draw_solution_phase","Monitor solution graphically","TSMonitorDrawSolutionPhase",bounds,&n,NULL);CHKERRQ(ierr);
    if (n != 4) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONG,"Must provide bounding box of phase field");
    ierr = TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,300,300,1,&ctx);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(ctx->viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDrawAxis(ctx->viewer,0,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLimits(axis,bounds[0],bounds[2],bounds[1],bounds[3]);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Phase Diagram","Variable 1","Variable 2");CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorDrawSolutionPhase,ctx,(PetscErrorCode (*)(void**))TSMonitorDrawCtxDestroy);CHKERRQ(ierr);
  }
  opt  = PETSC_FALSE;
  ierr = PetscOptionsName("-ts_monitor_draw_error","Monitor error graphically","TSMonitorDrawError",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscInt         howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_draw_error","Monitor error graphically","TSMonitorDrawError",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts),NULL,"Error",PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorDrawError,ctx,(PetscErrorCode (*)(void**))TSMonitorDrawCtxDestroy);CHKERRQ(ierr);
  }
  opt  = PETSC_FALSE;
  ierr = PetscOptionsName("-ts_monitor_draw_solution_function","Monitor solution provided by TSMonitorSetSolutionFunction() graphically","TSMonitorDrawSolutionFunction",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorDrawCtx ctx;
    PetscInt         howoften = 1;

    ierr = PetscOptionsInt("-ts_monitor_draw_solution_function","Monitor solution provided by TSMonitorSetSolutionFunction() graphically","TSMonitorDrawSolutionFunction",howoften,&howoften,NULL);CHKERRQ(ierr);
    ierr = TSMonitorDrawCtxCreate(PetscObjectComm((PetscObject)ts),NULL,"Solution provided by user function",PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorDrawSolutionFunction,ctx,(PetscErrorCode (*)(void**))TSMonitorDrawCtxDestroy);CHKERRQ(ierr);
  }

  opt  = PETSC_FALSE;
  ierr = PetscOptionsString("-ts_monitor_solution_vtk","Save each time step to a binary file, use filename-%%03D.vts","TSMonitorSolutionVTK",NULL,monfilename,sizeof(monfilename),&flg);CHKERRQ(ierr);
  if (flg) {
    const char *ptr,*ptr2;
    char       *filetemplate;
    if (!monfilename[0]) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"-ts_monitor_solution_vtk requires a file template, e.g. filename-%%03D.vts");
    /* Do some cursory validation of the input. */
    ierr = PetscStrstr(monfilename,"%",(char**)&ptr);CHKERRQ(ierr);
    if (!ptr) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"-ts_monitor_solution_vtk requires a file template, e.g. filename-%%03D.vts");
    for (ptr++; ptr && *ptr; ptr++) {
      ierr = PetscStrchr("DdiouxX",*ptr,(char**)&ptr2);CHKERRQ(ierr);
      if (!ptr2 && (*ptr < '0' || '9' < *ptr)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Invalid file template argument to -ts_monitor_solution_vtk, should look like filename-%%03D.vts");
      if (ptr2) break;
    }
    ierr = PetscStrallocpy(monfilename,&filetemplate);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorSolutionVTK,filetemplate,(PetscErrorCode (*)(void**))TSMonitorSolutionVTKDestroy);CHKERRQ(ierr);
  }

  ierr = PetscOptionsString("-ts_monitor_dmda_ray","Display a ray of the solution","None","y=0",dir,sizeof(dir),&flg);CHKERRQ(ierr);
  if (flg) {
    TSMonitorDMDARayCtx *rayctx;
    int                  ray = 0;
    DMDirection          ddir;
    DM                   da;
    PetscMPIInt          rank;

    if (dir[1] != '=') SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONG,"Unknown ray %s",dir);
    if (dir[0] == 'x') ddir = DM_X;
    else if (dir[0] == 'y') ddir = DM_Y;
    else SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONG,"Unknown ray %s",dir);
    sscanf(dir+2,"%d",&ray);

    ierr = PetscInfo2(((PetscObject)ts),"Displaying DMDA ray %c = %d\n",dir[0],ray);CHKERRQ(ierr);
    ierr = PetscNew(&rayctx);CHKERRQ(ierr);
    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    ierr = DMDAGetRay(da,ddir,ray,&rayctx->ray,&rayctx->scatter);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&rank);CHKERRMPI(ierr);
    if (rank == 0) {
      ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,NULL,0,0,600,300,&rayctx->viewer);CHKERRQ(ierr);
    }
    rayctx->lgctx = NULL;
    ierr = TSMonitorSet(ts,TSMonitorDMDARay,rayctx,TSMonitorDMDARayDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsString("-ts_monitor_lg_dmda_ray","Display a ray of the solution","None","x=0",dir,sizeof(dir),&flg);CHKERRQ(ierr);
  if (flg) {
    TSMonitorDMDARayCtx *rayctx;
    int                 ray = 0;
    DMDirection         ddir;
    DM                  da;
    PetscInt            howoften = 1;

    if (dir[1] != '=') SETERRQ1(PetscObjectComm((PetscObject) ts), PETSC_ERR_ARG_WRONG, "Malformed ray %s", dir);
    if      (dir[0] == 'x') ddir = DM_X;
    else if (dir[0] == 'y') ddir = DM_Y;
    else SETERRQ1(PetscObjectComm((PetscObject) ts), PETSC_ERR_ARG_WRONG, "Unknown ray direction %s", dir);
    sscanf(dir+2, "%d", &ray);

    ierr = PetscInfo2(((PetscObject) ts),"Displaying LG DMDA ray %c = %d\n", dir[0], ray);CHKERRQ(ierr);
    ierr = PetscNew(&rayctx);CHKERRQ(ierr);
    ierr = TSGetDM(ts, &da);CHKERRQ(ierr);
    ierr = DMDAGetRay(da, ddir, ray, &rayctx->ray, &rayctx->scatter);CHKERRQ(ierr);
    ierr = TSMonitorLGCtxCreate(PETSC_COMM_SELF,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,600,400,howoften,&rayctx->lgctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, TSMonitorLGDMDARay, rayctx, TSMonitorDMDARayDestroy);CHKERRQ(ierr);
  }

  ierr = PetscOptionsName("-ts_monitor_envelope","Monitor maximum and minimum value of each component of the solution","TSMonitorEnvelope",&opt);CHKERRQ(ierr);
  if (opt) {
    TSMonitorEnvelopeCtx ctx;

    ierr = TSMonitorEnvelopeCtxCreate(ts,&ctx);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,TSMonitorEnvelope,ctx,(PetscErrorCode (*)(void**))TSMonitorEnvelopeCtxDestroy);CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-ts_monitor_cancel","Remove all monitors","TSMonitorCancel",flg,&flg,&opt);CHKERRQ(ierr);
  if (opt && flg) {ierr = TSMonitorCancel(ts);CHKERRQ(ierr);}

  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-ts_fd_color", "Use finite differences with coloring to compute IJacobian", "TSComputeJacobianDefaultColor", flg, &flg, NULL);CHKERRQ(ierr);
  if (flg) {
    DM   dm;
    DMTS tdm;

    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMGetDMTS(dm, &tdm);CHKERRQ(ierr);
    tdm->ijacobianctx = NULL;
    ierr = TSSetIJacobian(ts, NULL, NULL, TSComputeIJacobianDefaultColor, NULL);CHKERRQ(ierr);
    ierr = PetscInfo(ts, "Setting default finite difference coloring Jacobian matrix\n");CHKERRQ(ierr);
  }

  /* Handle specific TS options */
  if (ts->ops->setfromoptions) {
    ierr = (*ts->ops->setfromoptions)(PetscOptionsObject,ts);CHKERRQ(ierr);
  }

  /* Handle TSAdapt options */
  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetDefaultType(ts->adapt,ts->default_adapt_type);CHKERRQ(ierr);
  ierr = TSAdaptSetFromOptions(PetscOptionsObject,ts->adapt);CHKERRQ(ierr);

  /* TS trajectory must be set after TS, since it may use some TS options above */
  tflg = ts->trajectory ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscOptionsBool("-ts_save_trajectory","Save the solution at each timestep","TSSetSaveTrajectory",tflg,&tflg,NULL);CHKERRQ(ierr);
  if (tflg) {
    ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  }

  ierr = TSAdjointSetFromOptions(PetscOptionsObject,ts);CHKERRQ(ierr);

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ts);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (ts->trajectory) {
    ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);
  }

  /* why do we have to do this here and not during TSSetUp? */
  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  if (ts->problem_type == TS_LINEAR) {
    ierr = PetscObjectTypeCompareAny((PetscObject)ts->snes,&flg,SNESKSPONLY,SNESKSPTRANSPOSEONLY,"");CHKERRQ(ierr);
    if (!flg) { ierr = SNESSetType(ts->snes,SNESKSPONLY);CHKERRQ(ierr); }
  }
  ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSGetTrajectory - Gets the trajectory from a TS if it exists

   Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
.  tr - the TSTrajectory object, if it exists

   Note: This routine should be called after all TS options have been set

   Level: advanced

.seealso: TSGetTrajectory(), TSAdjointSolve(), TSTrajectory, TSTrajectoryCreate()

@*/
PetscErrorCode  TSGetTrajectory(TS ts,TSTrajectory *tr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *tr = ts->trajectory;
  PetscFunctionReturn(0);
}

/*@
   TSSetSaveTrajectory - Causes the TS to save its solutions as it iterates forward in time in a TSTrajectory object

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database:
+  -ts_save_trajectory - saves the trajectory to a file
-  -ts_trajectory_type type

Note: This routine should be called after all TS options have been set

    The TSTRAJECTORYVISUALIZATION files can be loaded into Python with $PETSC_DIR/lib/petsc/bin/PetscBinaryIOTrajectory.py and
   MATLAB with $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m

   Level: intermediate

.seealso: TSGetTrajectory(), TSAdjointSolve()

@*/
PetscErrorCode  TSSetSaveTrajectory(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->trajectory) {
    ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSResetTrajectory - Destroys and recreates the internal TSTrajectory object

   Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Level: intermediate

.seealso: TSGetTrajectory(), TSAdjointSolve(), TSRemoveTrajectory()

@*/
PetscErrorCode  TSResetTrajectory(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->trajectory) {
    ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
    ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSRemoveTrajectory - Destroys and removes the internal TSTrajectory object from TS

   Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Level: intermediate

.seealso: TSResetTrajectory(), TSAdjointSolve()

@*/
PetscErrorCode TSRemoveTrajectory(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->trajectory) {
    ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSComputeRHSJacobian - Computes the Jacobian matrix that has been
      set with TSSetRHSJacobian().

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  t - current timestep
-  U - input vector

   Output Parameters:
+  A - Jacobian matrix
-  B - optional preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Level: developer

.seealso:  TSSetRHSJacobian(), KSPSetOperators()
@*/
PetscErrorCode  TSComputeRHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B)
{
  PetscErrorCode   ierr;
  PetscObjectState Ustate;
  PetscObjectId    Uid;
  DM               dm;
  DMTS             tsdm;
  TSRHSJacobian    rhsjacobianfunc;
  void             *ctx;
  TSRHSFunction    rhsfunction;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscCheckSameComm(ts,1,U,3);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  ierr = DMTSGetRHSFunction(dm,&rhsfunction,NULL);CHKERRQ(ierr);
  ierr = DMTSGetRHSJacobian(dm,&rhsjacobianfunc,&ctx);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)U,&Ustate);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)U,&Uid);CHKERRQ(ierr);

  if (ts->rhsjacobian.time == t && (ts->problem_type == TS_LINEAR || (ts->rhsjacobian.Xid == Uid && ts->rhsjacobian.Xstate == Ustate)) && (rhsfunction != TSComputeRHSFunctionLinear)) PetscFunctionReturn(0);

  if (ts->rhsjacobian.shift && ts->rhsjacobian.reuse) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Should not call TSComputeRHSJacobian() on a shifted matrix (shift=%lf) when RHSJacobian is reusable.",ts->rhsjacobian.shift);
  if (rhsjacobianfunc) {
    ierr = PetscLogEventBegin(TS_JacobianEval,ts,U,A,B);CHKERRQ(ierr);
    PetscStackPush("TS user Jacobian function");
    ierr = (*rhsjacobianfunc)(ts,t,U,A,B,ctx);CHKERRQ(ierr);
    PetscStackPop;
    ts->rhsjacs++;
    ierr = PetscLogEventEnd(TS_JacobianEval,ts,U,A,B);CHKERRQ(ierr);
  } else {
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    if (B && A != B) {ierr = MatZeroEntries(B);CHKERRQ(ierr);}
  }
  ts->rhsjacobian.time  = t;
  ts->rhsjacobian.shift = 0;
  ts->rhsjacobian.scale = 1.;
  ierr                  = PetscObjectGetId((PetscObject)U,&ts->rhsjacobian.Xid);CHKERRQ(ierr);
  ierr                  = PetscObjectStateGet((PetscObject)U,&ts->rhsjacobian.Xstate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSComputeRHSFunction - Evaluates the right-hand-side function.

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  t - current time
-  U - state vector

   Output Parameter:
.  y - right hand side

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Level: developer

.seealso: TSSetRHSFunction(), TSComputeIFunction()
@*/
PetscErrorCode TSComputeRHSFunction(TS ts,PetscReal t,Vec U,Vec y)
{
  PetscErrorCode ierr;
  TSRHSFunction  rhsfunction;
  TSIFunction    ifunction;
  void           *ctx;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSFunction(dm,&rhsfunction,&ctx);CHKERRQ(ierr);
  ierr = DMTSGetIFunction(dm,&ifunction,NULL);CHKERRQ(ierr);

  if (!rhsfunction && !ifunction) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSSetRHSFunction() and / or TSSetIFunction()");

  if (rhsfunction) {
    ierr = PetscLogEventBegin(TS_FunctionEval,ts,U,y,0);CHKERRQ(ierr);
    ierr = VecLockReadPush(U);CHKERRQ(ierr);
    PetscStackPush("TS user right-hand-side function");
    ierr = (*rhsfunction)(ts,t,U,y,ctx);CHKERRQ(ierr);
    PetscStackPop;
    ierr = VecLockReadPop(U);CHKERRQ(ierr);
    ts->rhsfuncs++;
    ierr = PetscLogEventEnd(TS_FunctionEval,ts,U,y,0);CHKERRQ(ierr);
  } else {
    ierr = VecZeroEntries(y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSComputeSolutionFunction - Evaluates the solution function.

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  t - current time

   Output Parameter:
.  U - the solution

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Level: developer

.seealso: TSSetSolutionFunction(), TSSetRHSFunction(), TSComputeIFunction()
@*/
PetscErrorCode TSComputeSolutionFunction(TS ts,PetscReal t,Vec U)
{
  PetscErrorCode     ierr;
  TSSolutionFunction solutionfunction;
  void               *ctx;
  DM                 dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetSolutionFunction(dm,&solutionfunction,&ctx);CHKERRQ(ierr);

  if (solutionfunction) {
    PetscStackPush("TS user solution function");
    ierr = (*solutionfunction)(ts,t,U,ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}
/*@
   TSComputeForcingFunction - Evaluates the forcing function.

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  t - current time

   Output Parameter:
.  U - the function value

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Level: developer

.seealso: TSSetSolutionFunction(), TSSetRHSFunction(), TSComputeIFunction()
@*/
PetscErrorCode TSComputeForcingFunction(TS ts,PetscReal t,Vec U)
{
  PetscErrorCode     ierr, (*forcing)(TS,PetscReal,Vec,void*);
  void               *ctx;
  DM                 dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetForcingFunction(dm,&forcing,&ctx);CHKERRQ(ierr);

  if (forcing) {
    PetscStackPush("TS user forcing function");
    ierr = (*forcing)(ts,t,U,ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGetRHSVec_Private(TS ts,Vec *Frhs)
{
  Vec            F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *Frhs = NULL;
  ierr  = TSGetIFunction(ts,&F,NULL,NULL);CHKERRQ(ierr);
  if (!ts->Frhs) {
    ierr = VecDuplicate(F,&ts->Frhs);CHKERRQ(ierr);
  }
  *Frhs = ts->Frhs;
  PetscFunctionReturn(0);
}

PetscErrorCode TSGetRHSMats_Private(TS ts,Mat *Arhs,Mat *Brhs)
{
  Mat            A,B;
  PetscErrorCode ierr;
  TSIJacobian    ijacobian;

  PetscFunctionBegin;
  if (Arhs) *Arhs = NULL;
  if (Brhs) *Brhs = NULL;
  ierr = TSGetIJacobian(ts,&A,&B,&ijacobian,NULL);CHKERRQ(ierr);
  if (Arhs) {
    if (!ts->Arhs) {
      if (ijacobian) {
        ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&ts->Arhs);CHKERRQ(ierr);
        ierr = TSSetMatStructure(ts,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ts->Arhs = A;
        ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      }
    } else {
      PetscBool flg;
      ierr = SNESGetUseMatrixFree(ts->snes,NULL,&flg);CHKERRQ(ierr);
      /* Handle case where user provided only RHSJacobian and used -snes_mf_operator */
      if (flg && !ijacobian && ts->Arhs == ts->Brhs) {
        ierr = PetscObjectDereference((PetscObject)ts->Arhs);CHKERRQ(ierr);
        ts->Arhs = A;
        ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      }
    }
    *Arhs = ts->Arhs;
  }
  if (Brhs) {
    if (!ts->Brhs) {
      if (A != B) {
        if (ijacobian) {
          ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&ts->Brhs);CHKERRQ(ierr);
        } else {
          ts->Brhs = B;
          ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
        }
      } else {
        ierr = PetscObjectReference((PetscObject)ts->Arhs);CHKERRQ(ierr);
        ts->Brhs = ts->Arhs;
      }
    }
    *Brhs = ts->Brhs;
  }
  PetscFunctionReturn(0);
}

/*@
   TSComputeIFunction - Evaluates the DAE residual written in implicit form F(t,U,Udot)=0

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  t - current time
.  U - state vector
.  Udot - time derivative of state vector
-  imex - flag indicates if the method is IMEX so that the RHSFunction should be kept separate

   Output Parameter:
.  Y - right hand side

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   If the user did did not write their equations in implicit form, this
   function recasts them in implicit form.

   Level: developer

.seealso: TSSetIFunction(), TSComputeRHSFunction()
@*/
PetscErrorCode TSComputeIFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec Y,PetscBool imex)
{
  PetscErrorCode ierr;
  TSIFunction    ifunction;
  TSRHSFunction  rhsfunction;
  void           *ctx;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,5);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetIFunction(dm,&ifunction,&ctx);CHKERRQ(ierr);
  ierr = DMTSGetRHSFunction(dm,&rhsfunction,NULL);CHKERRQ(ierr);

  if (!rhsfunction && !ifunction) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSSetRHSFunction() and / or TSSetIFunction()");

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,U,Udot,Y);CHKERRQ(ierr);
  if (ifunction) {
    PetscStackPush("TS user implicit function");
    ierr = (*ifunction)(ts,t,U,Udot,Y,ctx);CHKERRQ(ierr);
    PetscStackPop;
    ts->ifuncs++;
  }
  if (imex) {
    if (!ifunction) {
      ierr = VecCopy(Udot,Y);CHKERRQ(ierr);
    }
  } else if (rhsfunction) {
    if (ifunction) {
      Vec Frhs;
      ierr = TSGetRHSVec_Private(ts,&Frhs);CHKERRQ(ierr);
      ierr = TSComputeRHSFunction(ts,t,U,Frhs);CHKERRQ(ierr);
      ierr = VecAXPY(Y,-1,Frhs);CHKERRQ(ierr);
    } else {
      ierr = TSComputeRHSFunction(ts,t,U,Y);CHKERRQ(ierr);
      ierr = VecAYPX(Y,-1,Udot);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(TS_FunctionEval,ts,U,Udot,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSRecoverRHSJacobian - Recover the Jacobian matrix so that one can call TSComputeRHSJacobian() on it.

   Note:
   This routine is needed when one switches from TSComputeIJacobian() to TSComputeRHSJacobian() because the Jacobian matrix may be shifted or scaled in TSComputeIJacobian().

*/
static PetscErrorCode TSRecoverRHSJacobian(TS ts,Mat A,Mat B)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (A != ts->Arhs) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Invalid Amat");
  if (B != ts->Brhs) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Invalid Bmat");

  if (ts->rhsjacobian.shift) {
    ierr = MatShift(A,-ts->rhsjacobian.shift);CHKERRQ(ierr);
  }
  if (ts->rhsjacobian.scale == -1.) {
    ierr = MatScale(A,-1);CHKERRQ(ierr);
  }
  if (B && B == ts->Brhs && A != B) {
    if (ts->rhsjacobian.shift) {
      ierr = MatShift(B,-ts->rhsjacobian.shift);CHKERRQ(ierr);
    }
    if (ts->rhsjacobian.scale == -1.) {
      ierr = MatScale(B,-1);CHKERRQ(ierr);
    }
  }
  ts->rhsjacobian.shift = 0;
  ts->rhsjacobian.scale = 1.;
  PetscFunctionReturn(0);
}

/*@
   TSComputeIJacobian - Evaluates the Jacobian of the DAE

   Collective on TS

   Input
      Input Parameters:
+  ts - the TS context
.  t - current timestep
.  U - state vector
.  Udot - time derivative of state vector
.  shift - shift to apply, see note below
-  imex - flag indicates if the method is IMEX so that the RHSJacobian should be kept separate

   Output Parameters:
+  A - Jacobian matrix
-  B - matrix from which the preconditioner is constructed; often the same as A

   Notes:
   If F(t,U,Udot)=0 is the DAE, the required Jacobian is

   dF/dU + shift*dF/dUdot

   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Level: developer

.seealso:  TSSetIJacobian()
@*/
PetscErrorCode TSComputeIJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal shift,Mat A,Mat B,PetscBool imex)
{
  PetscErrorCode ierr;
  TSIJacobian    ijacobian;
  TSRHSJacobian  rhsjacobian;
  DM             dm;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);
  PetscValidPointer(A,6);
  PetscValidHeaderSpecific(A,MAT_CLASSID,6);
  PetscValidPointer(B,7);
  PetscValidHeaderSpecific(B,MAT_CLASSID,7);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetIJacobian(dm,&ijacobian,&ctx);CHKERRQ(ierr);
  ierr = DMTSGetRHSJacobian(dm,&rhsjacobian,NULL);CHKERRQ(ierr);

  if (!rhsjacobian && !ijacobian) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSSetRHSJacobian() and / or TSSetIJacobian()");

  ierr = PetscLogEventBegin(TS_JacobianEval,ts,U,A,B);CHKERRQ(ierr);
  if (ijacobian) {
    PetscStackPush("TS user implicit Jacobian");
    ierr = (*ijacobian)(ts,t,U,Udot,shift,A,B,ctx);CHKERRQ(ierr);
    ts->ijacs++;
    PetscStackPop;
  }
  if (imex) {
    if (!ijacobian) {  /* system was written as Udot = G(t,U) */
      PetscBool assembled;
      if (rhsjacobian) {
        Mat Arhs = NULL;
        ierr = TSGetRHSMats_Private(ts,&Arhs,NULL);CHKERRQ(ierr);
        if (A == Arhs) {
          if (rhsjacobian == TSComputeRHSJacobianConstant) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Unsupported operation! cannot use TSComputeRHSJacobianConstant"); /* there is no way to reconstruct shift*M-J since J cannot be reevaluated */
          ts->rhsjacobian.time = PETSC_MIN_REAL;
        }
      }
      ierr = MatZeroEntries(A);CHKERRQ(ierr);
      ierr = MatAssembled(A,&assembled);CHKERRQ(ierr);
      if (!assembled) {
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
      ierr = MatShift(A,shift);CHKERRQ(ierr);
      if (A != B) {
        ierr = MatZeroEntries(B);CHKERRQ(ierr);
        ierr = MatAssembled(B,&assembled);CHKERRQ(ierr);
        if (!assembled) {
          ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        }
        ierr = MatShift(B,shift);CHKERRQ(ierr);
      }
    }
  } else {
    Mat Arhs = NULL,Brhs = NULL;
    if (rhsjacobian) { /* RHSJacobian needs to be converted to part of IJacobian if exists */
      ierr = TSGetRHSMats_Private(ts,&Arhs,&Brhs);CHKERRQ(ierr);
    }
    if (Arhs == A) { /* No IJacobian matrix, so we only have the RHS matrix */
      PetscObjectState Ustate;
      PetscObjectId    Uid;
      TSRHSFunction    rhsfunction;

      ierr = DMTSGetRHSFunction(dm,&rhsfunction,NULL);CHKERRQ(ierr);
      ierr = PetscObjectStateGet((PetscObject)U,&Ustate);CHKERRQ(ierr);
      ierr = PetscObjectGetId((PetscObject)U,&Uid);CHKERRQ(ierr);
      if ((rhsjacobian == TSComputeRHSJacobianConstant || (ts->rhsjacobian.time == t && (ts->problem_type == TS_LINEAR || (ts->rhsjacobian.Xid == Uid && ts->rhsjacobian.Xstate == Ustate)) && rhsfunction != TSComputeRHSFunctionLinear)) && ts->rhsjacobian.scale == -1.) { /* No need to recompute RHSJacobian */
        ierr = MatShift(A,shift-ts->rhsjacobian.shift);CHKERRQ(ierr); /* revert the old shift and add the new shift with a single call to MatShift */
        if (A != B) {
          ierr = MatShift(B,shift-ts->rhsjacobian.shift);CHKERRQ(ierr);
        }
      } else {
        PetscBool flg;

        if (ts->rhsjacobian.reuse) { /* Undo the damage */
          /* MatScale has a short path for this case.
             However, this code path is taken the first time TSComputeRHSJacobian is called
             and the matrices have not been assembled yet */
          ierr = TSRecoverRHSJacobian(ts,A,B);CHKERRQ(ierr);
        }
        ierr = TSComputeRHSJacobian(ts,t,U,A,B);CHKERRQ(ierr);
        ierr = SNESGetUseMatrixFree(ts->snes,NULL,&flg);CHKERRQ(ierr);
        /* since -snes_mf_operator uses the full SNES function it does not need to be shifted or scaled here */
        if (!flg) {
          ierr = MatScale(A,-1);CHKERRQ(ierr);
          ierr = MatShift(A,shift);CHKERRQ(ierr);
        }
        if (A != B) {
          ierr = MatScale(B,-1);CHKERRQ(ierr);
          ierr = MatShift(B,shift);CHKERRQ(ierr);
        }
      }
      ts->rhsjacobian.scale = -1;
      ts->rhsjacobian.shift = shift;
    } else if (Arhs) {          /* Both IJacobian and RHSJacobian */
      if (!ijacobian) {         /* No IJacobian provided, but we have a separate RHS matrix */
        ierr = MatZeroEntries(A);CHKERRQ(ierr);
        ierr = MatShift(A,shift);CHKERRQ(ierr);
        if (A != B) {
          ierr = MatZeroEntries(B);CHKERRQ(ierr);
          ierr = MatShift(B,shift);CHKERRQ(ierr);
        }
      }
      ierr = TSComputeRHSJacobian(ts,t,U,Arhs,Brhs);CHKERRQ(ierr);
      ierr = MatAXPY(A,-1,Arhs,ts->axpy_pattern);CHKERRQ(ierr);
      if (A != B) {
        ierr = MatAXPY(B,-1,Brhs,ts->axpy_pattern);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(TS_JacobianEval,ts,U,A,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSSetRHSFunction - Sets the routine for evaluating the function,
    where U_t = G(t,u).

    Logically Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   r - vector to put the computed right hand side (or NULL to have it created)
.   f - routine for evaluating the right-hand-side function
-   ctx - [optional] user-defined context for private data for the
          function evaluation routine (may be NULL)

    Calling sequence of f:
$     PetscErrorCode f(TS ts,PetscReal t,Vec u,Vec F,void *ctx);

+   ts - timestep context
.   t - current timestep
.   u - input vector
.   F - function vector
-   ctx - [optional] user-defined function context

    Level: beginner

    Notes:
    You must call this function or TSSetIFunction() to define your ODE. You cannot use this function when solving a DAE.

.seealso: TSSetRHSJacobian(), TSSetIJacobian(), TSSetIFunction()
@*/
PetscErrorCode  TSSetRHSFunction(TS ts,Vec r,PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  Vec            ralloc = NULL;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (r) PetscValidHeaderSpecific(r,VEC_CLASSID,2);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetRHSFunction(dm,f,ctx);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (!r && !ts->dm && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&ralloc);CHKERRQ(ierr);
    r = ralloc;
  }
  ierr = SNESSetFunction(snes,r,SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = VecDestroy(&ralloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSSetSolutionFunction - Provide a function that computes the solution of the ODE or DAE

    Logically Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   f - routine for evaluating the solution
-   ctx - [optional] user-defined context for private data for the
          function evaluation routine (may be NULL)

    Calling sequence of f:
$     PetscErrorCode f(TS ts,PetscReal t,Vec u,void *ctx);

+   t - current timestep
.   u - output vector
-   ctx - [optional] user-defined function context

    Options Database:
+  -ts_monitor_lg_error - create a graphical monitor of error history, requires user to have provided TSSetSolutionFunction()
-  -ts_monitor_draw_error - Monitor error graphically, requires user to have provided TSSetSolutionFunction()

    Notes:
    This routine is used for testing accuracy of time integration schemes when you already know the solution.
    If analytic solutions are not known for your system, consider using the Method of Manufactured Solutions to
    create closed-form solutions with non-physical forcing terms.

    For low-dimensional problems solved in serial, such as small discrete systems, TSMonitorLGError() can be used to monitor the error history.

    Level: beginner

.seealso: TSSetRHSJacobian(), TSSetIJacobian(), TSComputeSolutionFunction(), TSSetForcingFunction(), TSSetSolution(), TSGetSolution(), TSMonitorLGError(), TSMonitorDrawError()
@*/
PetscErrorCode  TSSetSolutionFunction(TS ts,PetscErrorCode (*f)(TS,PetscReal,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetSolutionFunction(dm,f,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSSetForcingFunction - Provide a function that computes a forcing term for a ODE or PDE

    Logically Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   func - routine for evaluating the forcing function
-   ctx - [optional] user-defined context for private data for the
          function evaluation routine (may be NULL)

    Calling sequence of func:
$     PetscErrorCode func (TS ts,PetscReal t,Vec f,void *ctx);

+   t - current timestep
.   f - output vector
-   ctx - [optional] user-defined function context

    Notes:
    This routine is useful for testing accuracy of time integration schemes when using the Method of Manufactured Solutions to
    create closed-form solutions with a non-physical forcing term. It allows you to use the Method of Manufactored Solution without directly editing the
    definition of the problem you are solving and hence possibly introducing bugs.

    This replaces the ODE F(u,u_t,t) = 0 the TS is solving with F(u,u_t,t) - func(t) = 0

    This forcing function does not depend on the solution to the equations, it can only depend on spatial location, time, and possibly parameters, the
    parameters can be passed in the ctx variable.

    For low-dimensional problems solved in serial, such as small discrete systems, TSMonitorLGError() can be used to monitor the error history.

    Level: beginner

.seealso: TSSetRHSJacobian(), TSSetIJacobian(), TSComputeSolutionFunction(), TSSetSolutionFunction()
@*/
PetscErrorCode  TSSetForcingFunction(TS ts,TSForcingFunction func,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetForcingFunction(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSSetRHSJacobian - Sets the function to compute the Jacobian of G,
   where U_t = G(U,t), as well as the location to store the matrix.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  Amat - (approximate) Jacobian matrix
.  Pmat - matrix from which preconditioner is to be constructed (usually the same as Amat)
.  f   - the Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL)

   Calling sequence of f:
$     PetscErrorCode f(TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx);

+  t - current timestep
.  u - input vector
.  Amat - (approximate) Jacobian matrix
.  Pmat - matrix from which preconditioner is to be constructed (usually the same as Amat)
-  ctx - [optional] user-defined context for matrix evaluation routine

   Notes:
   You must set all the diagonal entries of the matrices, if they are zero you must still set them with a zero value

   The TS solver may modify the nonzero structure and the entries of the matrices Amat and Pmat between the calls to f()
   You should not assume the values are the same in the next call to f() as you set them in the previous call.

   Level: beginner

.seealso: SNESComputeJacobianDefaultColor(), TSSetRHSFunction(), TSRHSJacobianSetReuse(), TSSetIJacobian()

@*/
PetscErrorCode  TSSetRHSJacobian(TS ts,Mat Amat,Mat Pmat,TSRHSJacobian f,void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;
  TSIJacobian    ijacobian;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_CLASSID,3);
  if (Amat) PetscCheckSameComm(ts,1,Amat,2);
  if (Pmat) PetscCheckSameComm(ts,1,Pmat,3);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetRHSJacobian(dm,f,ctx);CHKERRQ(ierr);
  ierr = DMTSGetIJacobian(dm,&ijacobian,NULL);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (!ijacobian) {
    ierr = SNESSetJacobian(snes,Amat,Pmat,SNESTSFormJacobian,ts);CHKERRQ(ierr);
  }
  if (Amat) {
    ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Arhs);CHKERRQ(ierr);
    ts->Arhs = Amat;
  }
  if (Pmat) {
    ierr = PetscObjectReference((PetscObject)Pmat);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Brhs);CHKERRQ(ierr);
    ts->Brhs = Pmat;
  }
  PetscFunctionReturn(0);
}

/*@C
   TSSetIFunction - Set the function to compute F(t,U,U_t) where F() = 0 is the DAE to be solved.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  r   - vector to hold the residual (or NULL to have it created internally)
.  f   - the function evaluation routine
-  ctx - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of f:
$     PetscErrorCode f(TS ts,PetscReal t,Vec u,Vec u_t,Vec F,ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  F   - function vector
-  ctx - [optional] user-defined context for matrix evaluation routine

   Important:
   The user MUST call either this routine or TSSetRHSFunction() to define the ODE.  When solving DAEs you must use this function.

   Level: beginner

.seealso: TSSetRHSJacobian(), TSSetRHSFunction(), TSSetIJacobian()
@*/
PetscErrorCode  TSSetIFunction(TS ts,Vec r,TSIFunction f,void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  Vec            ralloc = NULL;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (r) PetscValidHeaderSpecific(r,VEC_CLASSID,2);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetIFunction(dm,f,ctx);CHKERRQ(ierr);

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (!r && !ts->dm && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&ralloc);CHKERRQ(ierr);
    r  = ralloc;
  }
  ierr = SNESSetFunction(snes,r,SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = VecDestroy(&ralloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSGetIFunction - Returns the vector where the implicit residual is stored and the function/context to compute it.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameters:
+  r - vector to hold residual (or NULL)
.  func - the function to compute residual (or NULL)
-  ctx - the function context (or NULL)

   Level: advanced

.seealso: TSSetIFunction(), SNESGetFunction()
@*/
PetscErrorCode TSGetIFunction(TS ts,Vec *r,TSIFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,r,NULL,NULL);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetIFunction(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSGetRHSFunction - Returns the vector where the right hand side is stored and the function/context to compute it.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameters:
+  r - vector to hold computed right hand side (or NULL)
.  func - the function to compute right hand side (or NULL)
-  ctx - the function context (or NULL)

   Level: advanced

.seealso: TSSetRHSFunction(), SNESGetFunction()
@*/
PetscErrorCode TSGetRHSFunction(TS ts,Vec *r,TSRHSFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,r,NULL,NULL);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSFunction(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSSetIJacobian - Set the function to compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is the function
        provided with TSSetIFunction().

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  Amat - (approximate) Jacobian matrix
.  Pmat - matrix used to compute preconditioner (usually the same as Amat)
.  f   - the Jacobian evaluation routine
-  ctx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Calling sequence of f:
$    PetscErrorCode f(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal a,Mat Amat,Mat Pmat,void *ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  a    - shift
.  Amat - (approximate) Jacobian of F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
.  Pmat - matrix used for constructing preconditioner, usually the same as Amat
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Notes:
   The matrices Amat and Pmat are exactly the matrices that are used by SNES for the nonlinear solve.

   If you know the operator Amat has a null space you can use MatSetNullSpace() and MatSetTransposeNullSpace() to supply the null
   space to Amat and the KSP solvers will automatically use that null space as needed during the solution process.

   The matrix dF/dU + a*dF/dU_t you provide turns out to be
   the Jacobian of F(t,U,W+a*U) where F(t,U,U_t) = 0 is the DAE to be solved.
   The time integrator internally approximates U_t by W+a*U where the positive "shift"
   a and vector W depend on the integration method, step size, and past states. For example with
   the backward Euler method a = 1/dt and W = -a*U(previous timestep) so
   W + a*U = a*(U - U(previous timestep)) = (U - U(previous timestep))/dt

   You must set all the diagonal entries of the matrices, if they are zero you must still set them with a zero value

   The TS solver may modify the nonzero structure and the entries of the matrices Amat and Pmat between the calls to f()
   You should not assume the values are the same in the next call to f() as you set them in the previous call.

   Level: beginner

.seealso: TSSetIFunction(), TSSetRHSJacobian(), SNESComputeJacobianDefaultColor(), SNESComputeJacobianDefault(), TSSetRHSFunction()

@*/
PetscErrorCode  TSSetIJacobian(TS ts,Mat Amat,Mat Pmat,TSIJacobian f,void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_CLASSID,3);
  if (Amat) PetscCheckSameComm(ts,1,Amat,2);
  if (Pmat) PetscCheckSameComm(ts,1,Pmat,3);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetIJacobian(dm,f,ctx);CHKERRQ(ierr);

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,Amat,Pmat,SNESTSFormJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSRHSJacobianSetReuse - restore RHS Jacobian before re-evaluating.  Without this flag, TS will change the sign and
   shift the RHS Jacobian for a finite-time-step implicit solve, in which case the user function will need to recompute
   the entire Jacobian.  The reuse flag must be set if the evaluation function will assume that the matrix entries have
   not been changed by the TS.

   Logically Collective

   Input Parameters:
+  ts - TS context obtained from TSCreate()
-  reuse - PETSC_TRUE if the RHS Jacobian

   Level: intermediate

.seealso: TSSetRHSJacobian(), TSComputeRHSJacobianConstant()
@*/
PetscErrorCode TSRHSJacobianSetReuse(TS ts,PetscBool reuse)
{
  PetscFunctionBegin;
  ts->rhsjacobian.reuse = reuse;
  PetscFunctionReturn(0);
}

/*@C
   TSSetI2Function - Set the function to compute F(t,U,U_t,U_tt) where F = 0 is the DAE to be solved.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  F   - vector to hold the residual (or NULL to have it created internally)
.  fun - the function evaluation routine
-  ctx - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of fun:
$     PetscErrorCode fun(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,Vec F,ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  F    - function vector
-  ctx  - [optional] user-defined context for matrix evaluation routine (may be NULL)

   Level: beginner

.seealso: TSSetI2Jacobian(), TSSetIFunction(), TSCreate(), TSSetRHSFunction()
@*/
PetscErrorCode TSSetI2Function(TS ts,Vec F,TSI2Function fun,void *ctx)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (F) PetscValidHeaderSpecific(F,VEC_CLASSID,2);
  ierr = TSSetIFunction(ts,F,NULL,NULL);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetI2Function(dm,fun,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSGetI2Function - Returns the vector where the implicit residual is stored and the function/context to compute it.

  Not Collective

  Input Parameter:
. ts - the TS context

  Output Parameters:
+ r - vector to hold residual (or NULL)
. fun - the function to compute residual (or NULL)
- ctx - the function context (or NULL)

  Level: advanced

.seealso: TSSetIFunction(), SNESGetFunction(), TSCreate()
@*/
PetscErrorCode TSGetI2Function(TS ts,Vec *r,TSI2Function *fun,void **ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,r,NULL,NULL);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetI2Function(dm,fun,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSSetI2Jacobian - Set the function to compute the matrix dF/dU + v*dF/dU_t  + a*dF/dU_tt
        where F(t,U,U_t,U_tt) is the function you provided with TSSetI2Function().

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J   - Jacobian matrix
.  P   - preconditioning matrix for J (may be same as J)
.  jac - the Jacobian evaluation routine
-  ctx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$    PetscErrorCode jac(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,PetscReal v,PetscReal a,Mat J,Mat P,void *ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  v    - shift for U_t
.  a    - shift for U_tt
.  J    - Jacobian of G(U) = F(t,U,W+v*U,W'+a*U), equivalent to dF/dU + v*dF/dU_t  + a*dF/dU_tt
.  P    - preconditioning matrix for J, may be same as J
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Notes:
   The matrices J and P are exactly the matrices that are used by SNES for the nonlinear solve.

   The matrix dF/dU + v*dF/dU_t + a*dF/dU_tt you provide turns out to be
   the Jacobian of G(U) = F(t,U,W+v*U,W'+a*U) where F(t,U,U_t,U_tt) = 0 is the DAE to be solved.
   The time integrator internally approximates U_t by W+v*U and U_tt by W'+a*U  where the positive "shift"
   parameters 'v' and 'a' and vectors W, W' depend on the integration method, step size, and past states.

   Level: beginner

.seealso: TSSetI2Function(), TSGetI2Jacobian()
@*/
PetscErrorCode TSSetI2Jacobian(TS ts,Mat J,Mat P,TSI2Jacobian jac,void *ctx)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J) PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,3);
  ierr = TSSetIJacobian(ts,J,P,NULL,NULL);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetI2Jacobian(dm,jac,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSGetI2Jacobian - Returns the implicit Jacobian at the present timestep.

  Not Collective, but parallel objects are returned if TS is parallel

  Input Parameter:
. ts  - The TS context obtained from TSCreate()

  Output Parameters:
+ J  - The (approximate) Jacobian of F(t,U,U_t,U_tt)
. P - The matrix from which the preconditioner is constructed, often the same as J
. jac - The function to compute the Jacobian matrices
- ctx - User-defined context for Jacobian evaluation routine

  Notes:
    You can pass in NULL for any return argument you do not need.

  Level: advanced

.seealso: TSGetTimeStep(), TSGetMatrices(), TSGetTime(), TSGetStepNumber(), TSSetI2Jacobian(), TSGetI2Function(), TSCreate()

@*/
PetscErrorCode  TSGetI2Jacobian(TS ts,Mat *J,Mat *P,TSI2Jacobian *jac,void **ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,J,P,NULL,NULL);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetI2Jacobian(dm,jac,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSComputeI2Function - Evaluates the DAE residual written in implicit form F(t,U,U_t,U_tt) = 0

  Collective on TS

  Input Parameters:
+ ts - the TS context
. t - current time
. U - state vector
. V - time derivative of state vector (U_t)
- A - second time derivative of state vector (U_tt)

  Output Parameter:
. F - the residual vector

  Note:
  Most users should not need to explicitly call this routine, as it
  is used internally within the nonlinear solvers.

  Level: developer

.seealso: TSSetI2Function(), TSGetI2Function()
@*/
PetscErrorCode TSComputeI2Function(TS ts,PetscReal t,Vec U,Vec V,Vec A,Vec F)
{
  DM             dm;
  TSI2Function   I2Function;
  void           *ctx;
  TSRHSFunction  rhsfunction;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetI2Function(dm,&I2Function,&ctx);CHKERRQ(ierr);
  ierr = DMTSGetRHSFunction(dm,&rhsfunction,NULL);CHKERRQ(ierr);

  if (!I2Function) {
    ierr = TSComputeIFunction(ts,t,U,A,F,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,U,V,F);CHKERRQ(ierr);

  PetscStackPush("TS user implicit function");
  ierr = I2Function(ts,t,U,V,A,F,ctx);CHKERRQ(ierr);
  PetscStackPop;

  if (rhsfunction) {
    Vec Frhs;
    ierr = TSGetRHSVec_Private(ts,&Frhs);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,t,U,Frhs);CHKERRQ(ierr);
    ierr = VecAXPY(F,-1,Frhs);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(TS_FunctionEval,ts,U,V,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSComputeI2Jacobian - Evaluates the Jacobian of the DAE

  Collective on TS

  Input Parameters:
+ ts - the TS context
. t - current timestep
. U - state vector
. V - time derivative of state vector
. A - second time derivative of state vector
. shiftV - shift to apply, see note below
- shiftA - shift to apply, see note below

  Output Parameters:
+ J - Jacobian matrix
- P - optional preconditioning matrix

  Notes:
  If F(t,U,V,A)=0 is the DAE, the required Jacobian is

  dF/dU + shiftV*dF/dV + shiftA*dF/dA

  Most users should not need to explicitly call this routine, as it
  is used internally within the nonlinear solvers.

  Level: developer

.seealso:  TSSetI2Jacobian()
@*/
PetscErrorCode TSComputeI2Jacobian(TS ts,PetscReal t,Vec U,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P)
{
  DM             dm;
  TSI2Jacobian   I2Jacobian;
  void           *ctx;
  TSRHSJacobian  rhsjacobian;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(J,MAT_CLASSID,8);
  PetscValidHeaderSpecific(P,MAT_CLASSID,9);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetI2Jacobian(dm,&I2Jacobian,&ctx);CHKERRQ(ierr);
  ierr = DMTSGetRHSJacobian(dm,&rhsjacobian,NULL);CHKERRQ(ierr);

  if (!I2Jacobian) {
    ierr = TSComputeIJacobian(ts,t,U,A,shiftA,J,P,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(TS_JacobianEval,ts,U,J,P);CHKERRQ(ierr);

  PetscStackPush("TS user implicit Jacobian");
  ierr = I2Jacobian(ts,t,U,V,A,shiftV,shiftA,J,P,ctx);CHKERRQ(ierr);
  PetscStackPop;

  if (rhsjacobian) {
    Mat Jrhs,Prhs;
    ierr = TSGetRHSMats_Private(ts,&Jrhs,&Prhs);CHKERRQ(ierr);
    ierr = TSComputeRHSJacobian(ts,t,U,Jrhs,Prhs);CHKERRQ(ierr);
    ierr = MatAXPY(J,-1,Jrhs,ts->axpy_pattern);CHKERRQ(ierr);
    if (P != J) {ierr = MatAXPY(P,-1,Prhs,ts->axpy_pattern);CHKERRQ(ierr);}
  }

  ierr = PetscLogEventEnd(TS_JacobianEval,ts,U,J,P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSSetTransientVariable - sets function to transform from state to transient variables

   Logically Collective

   Input Parameters:
+  ts - time stepping context on which to change the transient variable
.  tvar - a function that transforms to transient variables
-  ctx - a context for tvar

    Calling sequence of tvar:
$     PetscErrorCode tvar(TS ts,Vec p,Vec c,void *ctx);

+   ts - timestep context
.   p - input vector (primative form)
.   c - output vector, transient variables (conservative form)
-   ctx - [optional] user-defined function context

   Level: advanced

   Notes:
   This is typically used to transform from primitive to conservative variables so that a time integrator (e.g., TSBDF)
   can be conservative.  In this context, primitive variables P are used to model the state (e.g., because they lead to
   well-conditioned formulations even in limiting cases such as low-Mach or zero porosity).  The transient variable is
   C(P), specified by calling this function.  An IFunction thus receives arguments (P, Cdot) and the IJacobian must be
   evaluated via the chain rule, as in

     dF/dP + shift * dF/dCdot dC/dP.

.seealso: DMTSSetTransientVariable(), DMTSGetTransientVariable(), TSSetIFunction(), TSSetIJacobian()
@*/
PetscErrorCode TSSetTransientVariable(TS ts,TSTransientVariable tvar,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetTransientVariable(dm,tvar,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSComputeTransientVariable - transforms state (primitive) variables to transient (conservative) variables

   Logically Collective

   Input Parameters:
+  ts - TS on which to compute
-  U - state vector to be transformed to transient variables

   Output Parameters:
.  C - transient (conservative) variable

   Developer Notes:
   If DMTSSetTransientVariable() has not been called, then C is not modified in this routine and C=NULL is allowed.
   This makes it safe to call without a guard.  One can use TSHasTransientVariable() to check if transient variables are
   being used.

   Level: developer

.seealso: DMTSSetTransientVariable(), TSComputeIFunction(), TSComputeIJacobian()
@*/
PetscErrorCode TSComputeTransientVariable(TS ts,Vec U,Vec C)
{
  PetscErrorCode ierr;
  DM             dm;
  DMTS           dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetDMTS(dm,&dmts);CHKERRQ(ierr);
  if (dmts->ops->transientvar) {
    PetscValidHeaderSpecific(C,VEC_CLASSID,3);
    ierr = (*dmts->ops->transientvar)(ts,U,C,dmts->transientvarctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSHasTransientVariable - determine whether transient variables have been set

   Logically Collective

   Input Parameters:
.  ts - TS on which to compute

   Output Parameters:
.  has - PETSC_TRUE if transient variables have been set

   Level: developer

.seealso: DMTSSetTransientVariable(), TSComputeTransientVariable()
@*/
PetscErrorCode TSHasTransientVariable(TS ts,PetscBool *has)
{
  PetscErrorCode ierr;
  DM             dm;
  DMTS           dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetDMTS(dm,&dmts);CHKERRQ(ierr);
  *has = dmts->ops->transientvar ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   TS2SetSolution - Sets the initial solution and time derivative vectors
   for use by the TS routines handling second order equations.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  u - the solution vector
-  v - the time derivative vector

   Level: beginner

@*/
PetscErrorCode  TS2SetSolution(TS ts,Vec u,Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v,VEC_CLASSID,3);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)v);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_dot);CHKERRQ(ierr);
  ts->vec_dot = v;
  PetscFunctionReturn(0);
}

/*@
   TS2GetSolution - Returns the solution and time derivative at the present timestep
   for second order equations. It is valid to call this routine inside the function
   that you are evaluating in order to move to the new timestep. This vector not
   changed until the solution at the next timestep has been calculated.

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  u - the vector containing the solution
-  v - the vector containing the time derivative

   Level: intermediate

.seealso: TS2SetSolution(), TSGetTimeStep(), TSGetTime()

@*/
PetscErrorCode  TS2GetSolution(TS ts,Vec *u,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (u) PetscValidPointer(u,2);
  if (v) PetscValidPointer(v,3);
  if (u) *u = ts->vec_sol;
  if (v) *v = ts->vec_dot;
  PetscFunctionReturn(0);
}

/*@C
  TSLoad - Loads a KSP that has been stored in binary  with KSPView().

  Collective on PetscViewer

  Input Parameters:
+ newdm - the newly loaded TS, this needs to have been created with TSCreate() or
           some related function before a call to TSLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen()

   Level: intermediate

  Notes:
   The type is determined by the data in the file, any type set into the TS before this call is ignored.

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since TSLoad() and TSView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     has not yet been determined
.ve

.seealso: PetscViewerBinaryOpen(), TSView(), MatLoad(), VecLoad()
@*/
PetscErrorCode  TSLoad(TS ts, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  PetscInt       classid;
  char           type[256];
  DMTS           sdm;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  ierr = PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
  if (classid != TS_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONG,"Not TS next in file");
  ierr = PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, type);CHKERRQ(ierr);
  if (ts->ops->load) {
    ierr = (*ts->ops->load)(ts,viewer);CHKERRQ(ierr);
  }
  ierr = DMCreate(PetscObjectComm((PetscObject)ts),&dm);CHKERRQ(ierr);
  ierr = DMLoad(dm,viewer);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ts->dm,&ts->vec_sol);CHKERRQ(ierr);
  ierr = VecLoad(ts->vec_sol,viewer);CHKERRQ(ierr);
  ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
  ierr = DMTSLoad(sdm,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif

/*@C
   TSViewFromOptions - View from Options

   Collective on TS

   Input Parameters:
+  A - the application ordering context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  TS, TSView, PetscObjectViewFromOptions(), TSCreate()
@*/
PetscErrorCode  TSViewFromOptions(TS A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,TS_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSView - Prints the TS data structure.

    Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
-   viewer - visualization context

    Options Database Key:
.   -ts_view - calls TSView() at end of TSStep()

    Notes:
    The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

    The user can open an alternative visualization context with
    PetscViewerASCIIOpen() - output to a specified file.

    In the debugger you can do "call TSView(ts,0)" to display the TS solver. (The same holds for any PETSc object viewer).

    Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  TSView(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSType         type;
  PetscBool      iascii,isstring,isundials,isbinary,isdraw;
  DMTS           sdm;
#if defined(PETSC_HAVE_SAWS)
  PetscBool      issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ts),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws);CHKERRQ(ierr);
#endif
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ts,viewer);CHKERRQ(ierr);
    if (ts->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ts->max_steps < PETSC_MAX_INT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum steps=%D\n",ts->max_steps);CHKERRQ(ierr);
    }
    if (ts->max_time < PETSC_MAX_REAL) {
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum time=%g\n",(double)ts->max_time);CHKERRQ(ierr);
    }
    if (ts->ifuncs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of I function evaluations=%D\n",ts->ifuncs);CHKERRQ(ierr);
    }
    if (ts->ijacs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of I Jacobian evaluations=%D\n",ts->ijacs);CHKERRQ(ierr);
    }
    if (ts->rhsfuncs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of RHS function evaluations=%D\n",ts->rhsfuncs);CHKERRQ(ierr);
    }
    if (ts->rhsjacs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of RHS Jacobian evaluations=%D\n",ts->rhsjacs);CHKERRQ(ierr);
    }
    if (ts->usessnes) {
      PetscBool lin;
      if (ts->problem_type == TS_NONLINEAR) {
        ierr = PetscViewerASCIIPrintf(viewer,"  total number of nonlinear solver iterations=%D\n",ts->snes_its);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",ts->ksp_its);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)ts->snes,&lin,SNESKSPONLY,SNESKSPTRANSPOSEONLY,"");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of %slinear solve failures=%D\n",lin ? "" : "non",ts->num_snes_failures);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of rejected steps=%D\n",ts->reject);CHKERRQ(ierr);
    if (ts->vrtol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using vector of relative error tolerances, ");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  using relative error tolerance of %g, ",(double)ts->rtol);CHKERRQ(ierr);
    }
    if (ts->vatol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using vector of absolute error tolerances\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  using absolute error tolerance of %g\n",(double)ts->atol);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = TSGetType(ts,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," TSType: %-7.7s",type);CHKERRQ(ierr);
    if (ts->ops->view) {ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);}
  } else if (isbinary) {
    PetscInt    classid = TS_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    if (rank == 0) {
      ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscStrncpy(type,((PetscObject)ts)->type_name,256);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,type,256,PETSC_CHAR);CHKERRQ(ierr);
    }
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
    }
    if (ts->adapt) {ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);}
    ierr = DMView(ts->dm,viewer);CHKERRQ(ierr);
    ierr = VecView(ts->vec_sol,viewer);CHKERRQ(ierr);
    ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
    ierr = DMTSView(sdm,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    PetscDraw draw;
    char      str[36];
    PetscReal x,y,bottom,h;

    ierr   = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr   = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    ierr   = PetscStrcpy(str,"TS: ");CHKERRQ(ierr);
    ierr   = PetscStrcat(str,((PetscObject)ts)->type_name);CHKERRQ(ierr);
    ierr   = PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_BLACK,PETSC_DRAW_BLACK,str,NULL,&h);CHKERRQ(ierr);
    bottom = y - h;
    ierr   = PetscDrawPushCurrentPoint(draw,x,bottom);CHKERRQ(ierr);
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
    }
    if (ts->adapt) {ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);}
    if (ts->snes)  {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
    ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;
    const char  *name;

    ierr = PetscObjectGetName((PetscObject)ts,&name);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
    if (!((PetscObject)ts)->amsmem && rank == 0) {
      char       dir[1024];

      ierr = PetscObjectViewSAWs((PetscObject)ts,viewer);CHKERRQ(ierr);
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/time_step",name);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&ts->steps,1,SAWs_READ,SAWs_INT));
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/time",name);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&ts->ptime,1,SAWs_READ,SAWs_DOUBLE));
    }
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
    }
#endif
  }
  if (ts->snes && ts->usessnes)  {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
  ierr = DMTSView(sdm,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSSUNDIALS,&isundials);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSSetApplicationContext - Sets an optional user-defined context for
   the timesteppers.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  usrP - optional user context

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

   Level: intermediate

.seealso: TSGetApplicationContext()
@*/
PetscErrorCode  TSSetApplicationContext(TS ts,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->user = usrP;
  PetscFunctionReturn(0);
}

/*@
    TSGetApplicationContext - Gets the user-defined context for the
    timestepper.

    Not Collective

    Input Parameter:
.   ts - the TS context obtained from TSCreate()

    Output Parameter:
.   usrP - user context

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

    Level: intermediate

.seealso: TSSetApplicationContext()
@*/
PetscErrorCode  TSGetApplicationContext(TS ts,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *(void**)usrP = ts->user;
  PetscFunctionReturn(0);
}

/*@
   TSGetStepNumber - Gets the number of steps completed.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  steps - number of steps completed so far

   Level: intermediate

.seealso: TSGetTime(), TSGetTimeStep(), TSSetPreStep(), TSSetPreStage(), TSSetPostStage(), TSSetPostStep()
@*/
PetscErrorCode TSGetStepNumber(TS ts,PetscInt *steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(steps,2);
  *steps = ts->steps;
  PetscFunctionReturn(0);
}

/*@
   TSSetStepNumber - Sets the number of steps completed.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context
-  steps - number of steps completed so far

   Notes:
   For most uses of the TS solvers the user need not explicitly call
   TSSetStepNumber(), as the step counter is appropriately updated in
   TSSolve()/TSStep()/TSRollBack(). Power users may call this routine to
   reinitialize timestepping by setting the step counter to zero (and time
   to the initial time) to solve a similar problem with different initial
   conditions or parameters. Other possible use case is to continue
   timestepping from a previously interrupted run in such a way that TS
   monitors will be called with a initial nonzero step counter.

   Level: advanced

.seealso: TSGetStepNumber(), TSSetTime(), TSSetTimeStep(), TSSetSolution()
@*/
PetscErrorCode TSSetStepNumber(TS ts,PetscInt steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,steps,2);
  if (steps < 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Step number must be non-negative");
  ts->steps = steps;
  PetscFunctionReturn(0);
}

/*@
   TSSetTimeStep - Allows one to reset the timestep at any time,
   useful for simple pseudo-timestepping codes.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  time_step - the size of the timestep

   Level: intermediate

.seealso: TSGetTimeStep(), TSSetTime()

@*/
PetscErrorCode  TSSetTimeStep(TS ts,PetscReal time_step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time_step,2);
  ts->time_step = time_step;
  PetscFunctionReturn(0);
}

/*@
   TSSetExactFinalTime - Determines whether to adapt the final time step to
     match the exact final time, interpolate solution to the exact final time,
     or just return at the final time TS computed.

  Logically Collective on TS

   Input Parameters:
+   ts - the time-step context
-   eftopt - exact final time option

$  TS_EXACTFINALTIME_STEPOVER    - Don't do anything if final time is exceeded
$  TS_EXACTFINALTIME_INTERPOLATE - Interpolate back to final time
$  TS_EXACTFINALTIME_MATCHSTEP - Adapt final time step to match the final time

   Options Database:
.   -ts_exact_final_time <stepover,interpolate,matchstep> - select the final step at runtime

   Warning: If you use the option TS_EXACTFINALTIME_STEPOVER the solution may be at a very different time
    then the final time you selected.

   Level: beginner

.seealso: TSExactFinalTimeOption, TSGetExactFinalTime()
@*/
PetscErrorCode TSSetExactFinalTime(TS ts,TSExactFinalTimeOption eftopt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ts,eftopt,2);
  ts->exact_final_time = eftopt;
  PetscFunctionReturn(0);
}

/*@
   TSGetExactFinalTime - Gets the exact final time option.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  eftopt - exact final time option

   Level: beginner

.seealso: TSExactFinalTimeOption, TSSetExactFinalTime()
@*/
PetscErrorCode TSGetExactFinalTime(TS ts,TSExactFinalTimeOption *eftopt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(eftopt,2);
  *eftopt = ts->exact_final_time;
  PetscFunctionReturn(0);
}

/*@
   TSGetTimeStep - Gets the current timestep size.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  dt - the current timestep size

   Level: intermediate

.seealso: TSSetTimeStep(), TSGetTime()

@*/
PetscErrorCode  TSGetTimeStep(TS ts,PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidRealPointer(dt,2);
  *dt = ts->time_step;
  PetscFunctionReturn(0);
}

/*@
   TSGetSolution - Returns the solution at the present timestep. It
   is valid to call this routine inside the function that you are evaluating
   in order to move to the new timestep. This vector not changed until
   the solution at the next timestep has been calculated.

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  v - the vector containing the solution

   Note: If you used TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); this does not return the solution at the requested
   final time. It returns the solution at the next timestep.

   Level: intermediate

.seealso: TSGetTimeStep(), TSGetTime(), TSGetSolveTime(), TSGetSolutionComponents(), TSSetSolutionFunction()

@*/
PetscErrorCode  TSGetSolution(TS ts,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(v,2);
  *v = ts->vec_sol;
  PetscFunctionReturn(0);
}

/*@
   TSGetSolutionComponents - Returns any solution components at the present
   timestep, if available for the time integration method being used.
   Solution components are quantities that share the same size and
   structure as the solution vector.

   Not Collective, but Vec returned is parallel if TS is parallel

   Parameters :
+  ts - the TS context obtained from TSCreate() (input parameter).
.  n - If v is PETSC_NULL, then the number of solution components is
       returned through n, else the n-th solution component is
       returned in v.
-  v - the vector containing the n-th solution component
       (may be PETSC_NULL to use this function to find out
        the number of solutions components).

   Level: advanced

.seealso: TSGetSolution()

@*/
PetscErrorCode  TSGetSolutionComponents(TS ts,PetscInt *n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->getsolutioncomponents) *n = 0;
  else {
    ierr = (*ts->ops->getsolutioncomponents)(ts,n,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSGetAuxSolution - Returns an auxiliary solution at the present
   timestep, if available for the time integration method being used.

   Not Collective, but Vec returned is parallel if TS is parallel

   Parameters :
+  ts - the TS context obtained from TSCreate() (input parameter).
-  v - the vector containing the auxiliary solution

   Level: intermediate

.seealso: TSGetSolution()

@*/
PetscErrorCode  TSGetAuxSolution(TS ts,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->getauxsolution) {
    ierr = (*ts->ops->getauxsolution)(ts,v);CHKERRQ(ierr);
  } else {
    ierr = VecZeroEntries(*v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSGetTimeError - Returns the estimated error vector, if the chosen
   TSType has an error estimation functionality.

   Not Collective, but Vec returned is parallel if TS is parallel

   Note: MUST call after TSSetUp()

   Parameters :
+  ts - the TS context obtained from TSCreate() (input parameter).
.  n - current estimate (n=0) or previous one (n=-1)
-  v - the vector containing the error (same size as the solution).

   Level: intermediate

.seealso: TSGetSolution(), TSSetTimeError()

@*/
PetscErrorCode  TSGetTimeError(TS ts,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->gettimeerror) {
    ierr = (*ts->ops->gettimeerror)(ts,n,v);CHKERRQ(ierr);
  } else {
    ierr = VecZeroEntries(*v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSSetTimeError - Sets the estimated error vector, if the chosen
   TSType has an error estimation functionality. This can be used
   to restart such a time integrator with a given error vector.

   Not Collective, but Vec returned is parallel if TS is parallel

   Parameters :
+  ts - the TS context obtained from TSCreate() (input parameter).
-  v - the vector containing the error (same size as the solution).

   Level: intermediate

.seealso: TSSetSolution(), TSGetTimeError)

@*/
PetscErrorCode  TSSetTimeError(TS ts,Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetUp() first");
  if (ts->ops->settimeerror) {
    ierr = (*ts->ops->settimeerror)(ts,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----- Routines to initialize and destroy a timestepper ---- */
/*@
  TSSetProblemType - Sets the type of problem to be solved.

  Not collective

  Input Parameters:
+ ts   - The TS
- type - One of TS_LINEAR, TS_NONLINEAR where these types refer to problems of the forms
.vb
         U_t - A U = 0      (linear)
         U_t - A(t) U = 0   (linear)
         F(t,U,U_t) = 0     (nonlinear)
.ve

   Level: beginner

.seealso: TSSetUp(), TSProblemType, TS
@*/
PetscErrorCode  TSSetProblemType(TS ts, TSProblemType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->problem_type = type;
  if (type == TS_LINEAR) {
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetType(snes,SNESKSPONLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  TSGetProblemType - Gets the type of problem to be solved.

  Not collective

  Input Parameter:
. ts   - The TS

  Output Parameter:
. type - One of TS_LINEAR, TS_NONLINEAR where these types refer to problems of the forms
.vb
         M U_t = A U
         M(t) U_t = A(t) U
         F(t,U,U_t)
.ve

   Level: beginner

.seealso: TSSetUp(), TSProblemType, TS
@*/
PetscErrorCode  TSGetProblemType(TS ts, TSProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidIntPointer(type,2);
  *type = ts->problem_type;
  PetscFunctionReturn(0);
}

/*
    Attempt to check/preset a default value for the exact final time option. This is needed at the beginning of TSSolve() and in TSSetUp()
*/
static PetscErrorCode TSSetExactFinalTimeDefault(TS ts)
{
  PetscErrorCode ierr;
  PetscBool      isnone;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetDefaultType(ts->adapt,ts->default_adapt_type);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&isnone);CHKERRQ(ierr);
  if (!isnone && ts->exact_final_time == TS_EXACTFINALTIME_UNSPECIFIED) {
    ts->exact_final_time = TS_EXACTFINALTIME_MATCHSTEP;
  } else if (ts->exact_final_time == TS_EXACTFINALTIME_UNSPECIFIED) {
    ts->exact_final_time = TS_EXACTFINALTIME_INTERPOLATE;
  }
  PetscFunctionReturn(0);
}

/*@
   TSSetUp - Sets up the internal data structures for the later use of a timestepper.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Notes:
   For basic use of the TS solvers the user need not explicitly call
   TSSetUp(), since these actions will automatically occur during
   the call to TSStep() or TSSolve().  However, if one wishes to control this
   phase separately, TSSetUp() should be called after TSCreate()
   and optional routines of the form TSSetXXX(), but before TSStep() and TSSolve().

   Level: advanced

.seealso: TSCreate(), TSStep(), TSDestroy(), TSSolve()
@*/
PetscErrorCode  TSSetUp(TS ts)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscErrorCode (*func)(SNES,Vec,Vec,void*);
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*);
  TSIFunction    ifun;
  TSIJacobian    ijac;
  TSI2Jacobian   i2jac;
  TSRHSJacobian  rhsjac;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->setupcalled) PetscFunctionReturn(0);

  if (!((PetscObject)ts)->type_name) {
    ierr = TSGetIFunction(ts,NULL,&ifun,NULL);CHKERRQ(ierr);
    ierr = TSSetType(ts,ifun ? TSBEULER : TSEULER);CHKERRQ(ierr);
  }

  if (!ts->vec_sol) {
    if (ts->dm) {
      ierr = DMCreateGlobalVector(ts->dm,&ts->vec_sol);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetSolution() first");
  }

  if (!ts->Jacp && ts->Jacprhs) { /* IJacobianP shares the same matrix with RHSJacobianP if only RHSJacobianP is provided */
    ierr = PetscObjectReference((PetscObject)ts->Jacprhs);CHKERRQ(ierr);
    ts->Jacp = ts->Jacprhs;
  }

  if (ts->quadraturets) {
    ierr = TSSetUp(ts->quadraturets);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecDuplicate(ts->quadraturets->vec_sol,&ts->vec_costintegrand);CHKERRQ(ierr);
  }

  ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjac,NULL);CHKERRQ(ierr);
  if (rhsjac == TSComputeRHSJacobianConstant) {
    Mat Amat,Pmat;
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,&Amat,&Pmat,NULL,NULL);CHKERRQ(ierr);
    /* Matching matrices implies that an IJacobian is NOT set, because if it had been set, the IJacobian's matrix would
     * have displaced the RHS matrix */
    if (Amat && Amat == ts->Arhs) {
      /* we need to copy the values of the matrix because for the constant Jacobian case the user will never set the numerical values in this new location */
      ierr = MatDuplicate(ts->Arhs,MAT_COPY_VALUES,&Amat);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,Amat,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&Amat);CHKERRQ(ierr);
    }
    if (Pmat && Pmat == ts->Brhs) {
      ierr = MatDuplicate(ts->Brhs,MAT_COPY_VALUES,&Pmat);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,NULL,Pmat,NULL,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&Pmat);CHKERRQ(ierr);
    }
  }

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetDefaultType(ts->adapt,ts->default_adapt_type);CHKERRQ(ierr);

  if (ts->ops->setup) {
    ierr = (*ts->ops->setup)(ts);CHKERRQ(ierr);
  }

  ierr = TSSetExactFinalTimeDefault(ts);CHKERRQ(ierr);

  /* In the case where we've set a DMTSFunction or what have you, we need the default SNESFunction
     to be set right but can't do it elsewhere due to the overreliance on ctx=ts.
   */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetFunction(dm,&func,NULL);CHKERRQ(ierr);
  if (!func) {
    ierr = DMSNESSetFunction(dm,SNESTSFormFunction,ts);CHKERRQ(ierr);
  }
  /* If the SNES doesn't have a jacobian set and the TS has an ijacobian or rhsjacobian set, set the SNES to use it.
     Otherwise, the SNES will use coloring internally to form the Jacobian.
   */
  ierr = DMSNESGetJacobian(dm,&jac,NULL);CHKERRQ(ierr);
  ierr = DMTSGetIJacobian(dm,&ijac,NULL);CHKERRQ(ierr);
  ierr = DMTSGetI2Jacobian(dm,&i2jac,NULL);CHKERRQ(ierr);
  ierr = DMTSGetRHSJacobian(dm,&rhsjac,NULL);CHKERRQ(ierr);
  if (!jac && (ijac || i2jac || rhsjac)) {
    ierr = DMSNESSetJacobian(dm,SNESTSFormJacobian,ts);CHKERRQ(ierr);
  }

  /* if time integration scheme has a starting method, call it */
  if (ts->ops->startingmethod) {
    ierr = (*ts->ops->startingmethod)(ts);CHKERRQ(ierr);
  }

  ts->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TSReset - Resets a TS context and removes any allocated Vecs and Mats.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: beginner

.seealso: TSCreate(), TSSetup(), TSDestroy()
@*/
PetscErrorCode  TSReset(TS ts)
{
  TS_RHSSplitLink ilink = ts->tsrhssplit,next;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);

  if (ts->ops->reset) {
    ierr = (*ts->ops->reset)(ts);CHKERRQ(ierr);
  }
  if (ts->snes) {ierr = SNESReset(ts->snes);CHKERRQ(ierr);}
  if (ts->adapt) {ierr = TSAdaptReset(ts->adapt);CHKERRQ(ierr);}

  ierr = MatDestroy(&ts->Arhs);CHKERRQ(ierr);
  ierr = MatDestroy(&ts->Brhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->Frhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_sol);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_dot);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vatol);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vrtol);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->nwork,&ts->work);CHKERRQ(ierr);

  ierr = MatDestroy(&ts->Jacprhs);CHKERRQ(ierr);
  ierr = MatDestroy(&ts->Jacp);CHKERRQ(ierr);
  if (ts->forward_solve) {
    ierr = TSForwardReset(ts);CHKERRQ(ierr);
  }
  if (ts->quadraturets) {
    ierr = TSReset(ts->quadraturets);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vec_costintegrand);CHKERRQ(ierr);
  }
  while (ilink) {
    next = ilink->next;
    ierr = TSDestroy(&ilink->ts);CHKERRQ(ierr);
    ierr = PetscFree(ilink->splitname);CHKERRQ(ierr);
    ierr = ISDestroy(&ilink->is);CHKERRQ(ierr);
    ierr = PetscFree(ilink);CHKERRQ(ierr);
    ilink = next;
  }
  ts->tsrhssplit = NULL;
  ts->num_rhs_splits = 0;
  ts->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   TSDestroy - Destroys the timestepper context that was created
   with TSCreate().

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: beginner

.seealso: TSCreate(), TSSetUp(), TSSolve()
@*/
PetscErrorCode  TSDestroy(TS *ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ts) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*ts,TS_CLASSID,1);
  if (--((PetscObject)(*ts))->refct > 0) {*ts = NULL; PetscFunctionReturn(0);}

  ierr = TSReset(*ts);CHKERRQ(ierr);
  ierr = TSAdjointReset(*ts);CHKERRQ(ierr);
  if ((*ts)->forward_solve) {
    ierr = TSForwardReset(*ts);CHKERRQ(ierr);
  }
  /* if memory was published with SAWs then destroy it */
  ierr = PetscObjectSAWsViewOff((PetscObject)*ts);CHKERRQ(ierr);
  if ((*ts)->ops->destroy) {ierr = (*(*ts)->ops->destroy)((*ts));CHKERRQ(ierr);}

  ierr = TSTrajectoryDestroy(&(*ts)->trajectory);CHKERRQ(ierr);

  ierr = TSAdaptDestroy(&(*ts)->adapt);CHKERRQ(ierr);
  ierr = TSEventDestroy(&(*ts)->event);CHKERRQ(ierr);

  ierr = SNESDestroy(&(*ts)->snes);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ts)->dm);CHKERRQ(ierr);
  ierr = TSMonitorCancel((*ts));CHKERRQ(ierr);
  ierr = TSAdjointMonitorCancel((*ts));CHKERRQ(ierr);

  ierr = TSDestroy(&(*ts)->quadraturets);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSGetSNES - Returns the SNES (nonlinear solver) associated with
   a TS (timestepper) context. Valid only for nonlinear problems.

   Not Collective, but SNES is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  snes - the nonlinear solver context

   Notes:
   The user can then directly manipulate the SNES context to set various
   options, etc.  Likewise, the user can then extract and manipulate the
   KSP, KSP, and PC contexts as well.

   TSGetSNES() does not work for integrators that do not use SNES; in
   this case TSGetSNES() returns NULL in snes.

   Level: beginner

@*/
PetscErrorCode  TSGetSNES(TS ts,SNES *snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(snes,2);
  if (!ts->snes) {
    ierr = SNESCreate(PetscObjectComm((PetscObject)ts),&ts->snes);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)ts->snes,((PetscObject)ts)->options);CHKERRQ(ierr);
    ierr = SNESSetFunction(ts->snes,NULL,SNESTSFormFunction,ts);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ts,(PetscObject)ts->snes);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);
    if (ts->dm) {ierr = SNESSetDM(ts->snes,ts->dm);CHKERRQ(ierr);}
    if (ts->problem_type == TS_LINEAR) {
      ierr = SNESSetType(ts->snes,SNESKSPONLY);CHKERRQ(ierr);
    }
  }
  *snes = ts->snes;
  PetscFunctionReturn(0);
}

/*@
   TSSetSNES - Set the SNES (nonlinear solver) to be used by the timestepping context

   Collective

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  snes - the nonlinear solver context

   Notes:
   Most users should have the TS created by calling TSGetSNES()

   Level: developer

@*/
PetscErrorCode TSSetSNES(TS ts,SNES snes)
{
  PetscErrorCode ierr;
  PetscErrorCode (*func)(SNES,Vec,Mat,Mat,void*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(snes,SNES_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)snes);CHKERRQ(ierr);
  ierr = SNESDestroy(&ts->snes);CHKERRQ(ierr);

  ts->snes = snes;

  ierr = SNESSetFunction(ts->snes,NULL,SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = SNESGetJacobian(ts->snes,NULL,NULL,&func,NULL);CHKERRQ(ierr);
  if (func == SNESTSFormJacobian) {
    ierr = SNESSetJacobian(ts->snes,NULL,NULL,SNESTSFormJacobian,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSGetKSP - Returns the KSP (linear solver) associated with
   a TS (timestepper) context.

   Not Collective, but KSP is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  ksp - the nonlinear solver context

   Notes:
   The user can then directly manipulate the KSP context to set various
   options, etc.  Likewise, the user can then extract and manipulate the
   KSP and PC contexts as well.

   TSGetKSP() does not work for integrators that do not use KSP;
   in this case TSGetKSP() returns NULL in ksp.

   Level: beginner

@*/
PetscErrorCode  TSGetKSP(TS ts,KSP *ksp)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!((PetscObject)ts)->type_name) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"KSP is not created yet. Call TSSetType() first");
  if (ts->problem_type != TS_LINEAR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Linear only; use TSGetSNES()");
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

/*@
   TSSetMaxSteps - Sets the maximum number of steps to use.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  maxsteps - maximum number of steps to use

   Options Database Keys:
.  -ts_max_steps <maxsteps> - Sets maxsteps

   Notes:
   The default maximum number of steps is 5000

   Level: intermediate

.seealso: TSGetMaxSteps(), TSSetMaxTime(), TSSetExactFinalTime()
@*/
PetscErrorCode TSSetMaxSteps(TS ts,PetscInt maxsteps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,maxsteps,2);
  if (maxsteps < 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of steps must be non-negative");
  ts->max_steps = maxsteps;
  PetscFunctionReturn(0);
}

/*@
   TSGetMaxSteps - Gets the maximum number of steps to use.

   Not Collective

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  maxsteps - maximum number of steps to use

   Level: advanced

.seealso: TSSetMaxSteps(), TSGetMaxTime(), TSSetMaxTime()
@*/
PetscErrorCode TSGetMaxSteps(TS ts,PetscInt *maxsteps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(maxsteps,2);
  *maxsteps = ts->max_steps;
  PetscFunctionReturn(0);
}

/*@
   TSSetMaxTime - Sets the maximum (or final) time for timestepping.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  maxtime - final time to step to

   Options Database Keys:
.  -ts_max_time <maxtime> - Sets maxtime

   Notes:
   The default maximum time is 5.0

   Level: intermediate

.seealso: TSGetMaxTime(), TSSetMaxSteps(), TSSetExactFinalTime()
@*/
PetscErrorCode TSSetMaxTime(TS ts,PetscReal maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,maxtime,2);
  ts->max_time = maxtime;
  PetscFunctionReturn(0);
}

/*@
   TSGetMaxTime - Gets the maximum (or final) time for timestepping.

   Not Collective

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  maxtime - final time to step to

   Level: advanced

.seealso: TSSetMaxTime(), TSGetMaxSteps(), TSSetMaxSteps()
@*/
PetscErrorCode TSGetMaxTime(TS ts,PetscReal *maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidRealPointer(maxtime,2);
  *maxtime = ts->max_time;
  PetscFunctionReturn(0);
}

/*@
   TSSetInitialTimeStep - Deprecated, use TSSetTime() and TSSetTimeStep().

   Level: deprecated

@*/
PetscErrorCode  TSSetInitialTimeStep(TS ts,PetscReal initial_time,PetscReal time_step)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSSetTime(ts,initial_time);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,time_step);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSGetDuration - Deprecated, use TSGetMaxSteps() and TSGetMaxTime().

   Level: deprecated

@*/
PetscErrorCode TSGetDuration(TS ts, PetscInt *maxsteps, PetscReal *maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  if (maxsteps) {
    PetscValidIntPointer(maxsteps,2);
    *maxsteps = ts->max_steps;
  }
  if (maxtime) {
    PetscValidRealPointer(maxtime,3);
    *maxtime = ts->max_time;
  }
  PetscFunctionReturn(0);
}

/*@
   TSSetDuration - Deprecated, use TSSetMaxSteps() and TSSetMaxTime().

   Level: deprecated

@*/
PetscErrorCode TSSetDuration(TS ts,PetscInt maxsteps,PetscReal maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,maxsteps,2);
  PetscValidLogicalCollectiveReal(ts,maxtime,3);
  if (maxsteps >= 0) ts->max_steps = maxsteps;
  if (maxtime != PETSC_DEFAULT) ts->max_time = maxtime;
  PetscFunctionReturn(0);
}

/*@
   TSGetTimeStepNumber - Deprecated, use TSGetStepNumber().

   Level: deprecated

@*/
PetscErrorCode TSGetTimeStepNumber(TS ts,PetscInt *steps) { return TSGetStepNumber(ts,steps); }

/*@
   TSGetTotalSteps - Deprecated, use TSGetStepNumber().

   Level: deprecated

@*/
PetscErrorCode TSGetTotalSteps(TS ts,PetscInt *steps) { return TSGetStepNumber(ts,steps); }

/*@
   TSSetSolution - Sets the initial solution vector
   for use by the TS routines.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  u - the solution vector

   Level: beginner

.seealso: TSSetSolutionFunction(), TSGetSolution(), TSCreate()
@*/
PetscErrorCode  TSSetSolution(TS ts,Vec u)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)u);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_sol);CHKERRQ(ierr);
  ts->vec_sol = u;

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMShellSetGlobalVector(dm,u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSSetPreStep - Sets the general-purpose function
  called once at the beginning of each time step.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
.   PetscErrorCode func (TS ts);

  Level: intermediate

.seealso: TSSetPreStage(), TSSetPostStage(), TSSetPostStep(), TSStep(), TSRestartStep()
@*/
PetscErrorCode  TSSetPreStep(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->prestep = func;
  PetscFunctionReturn(0);
}

/*@
  TSPreStep - Runs the user-defined pre-step function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSPreStep() is typically used within time stepping implementations,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetPreStep(), TSPreStage(), TSPostStage(), TSPostStep()
@*/
PetscErrorCode  TSPreStep(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->prestep) {
    Vec              U;
    PetscObjectState sprev,spost;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*ts->prestep),(ts));
    ierr = PetscObjectStateGet((PetscObject)U,&spost);CHKERRQ(ierr);
    if (sprev != spost) {ierr = TSRestartStep(ts);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@C
  TSSetPreStage - Sets the general-purpose function
  called once at the beginning of each stage.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
.    PetscErrorCode func(TS ts, PetscReal stagetime);

  Level: intermediate

  Note:
  There may be several stages per time step. If the solve for a given stage fails, the step may be rejected and retried.
  The time step number being computed can be queried using TSGetStepNumber() and the total size of the step being
  attempted can be obtained using TSGetTimeStep(). The time at the start of the step is available via TSGetTime().

.seealso: TSSetPostStage(), TSSetPreStep(), TSSetPostStep(), TSGetApplicationContext()
@*/
PetscErrorCode  TSSetPreStage(TS ts, PetscErrorCode (*func)(TS,PetscReal))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->prestage = func;
  PetscFunctionReturn(0);
}

/*@C
  TSSetPostStage - Sets the general-purpose function
  called once at the end of each stage.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. PetscErrorCode func(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y);

  Level: intermediate

  Note:
  There may be several stages per time step. If the solve for a given stage fails, the step may be rejected and retried.
  The time step number being computed can be queried using TSGetStepNumber() and the total size of the step being
  attempted can be obtained using TSGetTimeStep(). The time at the start of the step is available via TSGetTime().

.seealso: TSSetPreStage(), TSSetPreStep(), TSSetPostStep(), TSGetApplicationContext()
@*/
PetscErrorCode  TSSetPostStage(TS ts, PetscErrorCode (*func)(TS,PetscReal,PetscInt,Vec*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->poststage = func;
  PetscFunctionReturn(0);
}

/*@C
  TSSetPostEvaluate - Sets the general-purpose function
  called once at the end of each step evaluation.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. PetscErrorCode func(TS ts);

  Level: intermediate

  Note:
  Semantically, TSSetPostEvaluate() differs from TSSetPostStep() since the function it sets is called before event-handling
  thus guaranteeing the same solution (computed by the time-stepper) will be passed to it. On the other hand, TSPostStep()
  may be passed a different solution, possibly changed by the event handler. TSPostEvaluate() is called after the next step
  solution is evaluated allowing to modify it, if need be. The solution can be obtained with TSGetSolution(), the time step
  with TSGetTimeStep(), and the time at the start of the step is available via TSGetTime()

.seealso: TSSetPreStage(), TSSetPreStep(), TSSetPostStep(), TSGetApplicationContext()
@*/
PetscErrorCode  TSSetPostEvaluate(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->postevaluate = func;
  PetscFunctionReturn(0);
}

/*@
  TSPreStage - Runs the user-defined pre-stage function set using TSSetPreStage()

  Collective on TS

  Input Parameters:
. ts          - The TS context obtained from TSCreate()
  stagetime   - The absolute time of the current stage

  Notes:
  TSPreStage() is typically used within time stepping implementations,
  most users would not generally call this routine themselves.

  Level: developer

.seealso: TSPostStage(), TSSetPreStep(), TSPreStep(), TSPostStep()
@*/
PetscErrorCode  TSPreStage(TS ts, PetscReal stagetime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->prestage) {
    PetscStackCallStandard((*ts->prestage),(ts,stagetime));
  }
  PetscFunctionReturn(0);
}

/*@
  TSPostStage - Runs the user-defined post-stage function set using TSSetPostStage()

  Collective on TS

  Input Parameters:
. ts          - The TS context obtained from TSCreate()
  stagetime   - The absolute time of the current stage
  stageindex  - Stage number
  Y           - Array of vectors (of size = total number
                of stages) with the stage solutions

  Notes:
  TSPostStage() is typically used within time stepping implementations,
  most users would not generally call this routine themselves.

  Level: developer

.seealso: TSPreStage(), TSSetPreStep(), TSPreStep(), TSPostStep()
@*/
PetscErrorCode  TSPostStage(TS ts, PetscReal stagetime, PetscInt stageindex, Vec *Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->poststage) {
    PetscStackCallStandard((*ts->poststage),(ts,stagetime,stageindex,Y));
  }
  PetscFunctionReturn(0);
}

/*@
  TSPostEvaluate - Runs the user-defined post-evaluate function set using TSSetPostEvaluate()

  Collective on TS

  Input Parameters:
. ts          - The TS context obtained from TSCreate()

  Notes:
  TSPostEvaluate() is typically used within time stepping implementations,
  most users would not generally call this routine themselves.

  Level: developer

.seealso: TSSetPostEvaluate(), TSSetPreStep(), TSPreStep(), TSPostStep()
@*/
PetscErrorCode  TSPostEvaluate(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->postevaluate) {
    Vec              U;
    PetscObjectState sprev,spost;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*ts->postevaluate),(ts));
    ierr = PetscObjectStateGet((PetscObject)U,&spost);CHKERRQ(ierr);
    if (sprev != spost) {ierr = TSRestartStep(ts);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@C
  TSSetPostStep - Sets the general-purpose function
  called once at the end of each time step.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
$ func (TS ts);

  Notes:
  The function set by TSSetPostStep() is called after each successful step. The solution vector X
  obtained by TSGetSolution() may be different than that computed at the step end if the event handler
  locates an event and TSPostEvent() modifies it. Use TSSetPostEvaluate() if an unmodified solution is needed instead.

  Level: intermediate

.seealso: TSSetPreStep(), TSSetPreStage(), TSSetPostEvaluate(), TSGetTimeStep(), TSGetStepNumber(), TSGetTime(), TSRestartStep()
@*/
PetscErrorCode  TSSetPostStep(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->poststep = func;
  PetscFunctionReturn(0);
}

/*@
  TSPostStep - Runs the user-defined post-step function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSPostStep() is typically used within time stepping implementations,
  so most users would not generally call this routine themselves.

  Level: developer

@*/
PetscErrorCode  TSPostStep(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->poststep) {
    Vec              U;
    PetscObjectState sprev,spost;

    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&sprev);CHKERRQ(ierr);
    PetscStackCallStandard((*ts->poststep),(ts));
    ierr = PetscObjectStateGet((PetscObject)U,&spost);CHKERRQ(ierr);
    if (sprev != spost) {ierr = TSRestartStep(ts);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@
   TSInterpolate - Interpolate the solution computed during the previous step to an arbitrary location in the interval

   Collective on TS

   Input Parameters:
+  ts - time stepping context
-  t - time to interpolate to

   Output Parameter:
.  U - state at given time

   Level: intermediate

   Developer Notes:
   TSInterpolate() and the storing of previous steps/stages should be generalized to support delay differential equations and continuous adjoints.

.seealso: TSSetExactFinalTime(), TSSolve()
@*/
PetscErrorCode TSInterpolate(TS ts,PetscReal t,Vec U)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  if (t < ts->ptime_prev || t > ts->ptime) SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Requested time %g not in last time steps [%g,%g]",t,(double)ts->ptime_prev,(double)ts->ptime);
  if (!ts->ops->interpolate) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"%s does not provide interpolation",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->interpolate)(ts,t,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSStep - Steps one time step

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: developer

   Notes:
   The public interface for the ODE/DAE solvers is TSSolve(), you should almost for sure be using that routine and not this routine.

   The hook set using TSSetPreStep() is called before each attempt to take the step. In general, the time step size may
   be changed due to adaptive error controller or solve failures. Note that steps may contain multiple stages.

   This may over-step the final time provided in TSSetMaxTime() depending on the time-step used. TSSolve() interpolates to exactly the
   time provided in TSSetMaxTime(). One can use TSInterpolate() to determine an interpolated solution within the final timestep.

.seealso: TSCreate(), TSSetUp(), TSDestroy(), TSSolve(), TSSetPreStep(), TSSetPreStage(), TSSetPostStage(), TSInterpolate()
@*/
PetscErrorCode  TSStep(TS ts)
{
  PetscErrorCode   ierr;
  static PetscBool cite = PETSC_FALSE;
  PetscReal        ptime;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscCitationsRegister("@article{tspaper,\n"
                                "  title         = {{PETSc/TS}: A Modern Scalable {DAE/ODE} Solver Library},\n"
                                "  author        = {Abhyankar, Shrirang and Brown, Jed and Constantinescu, Emil and Ghosh, Debojyoti and Smith, Barry F. and Zhang, Hong},\n"
                                "  journal       = {arXiv e-preprints},\n"
                                "  eprint        = {1806.01437},\n"
                                "  archivePrefix = {arXiv},\n"
                                "  year          = {2018}\n}\n",&cite);CHKERRQ(ierr);

  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(ts->trajectory,ts);CHKERRQ(ierr);

  if (!ts->ops->step) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSStep not implemented for type '%s'",((PetscObject)ts)->type_name);
  if (ts->max_time >= PETSC_MAX_REAL && ts->max_steps == PETSC_MAX_INT) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"You must call TSSetMaxTime() or TSSetMaxSteps(), or use -ts_max_time <time> or -ts_max_steps <steps>");
  if (ts->exact_final_time == TS_EXACTFINALTIME_UNSPECIFIED) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"You must call TSSetExactFinalTime() or use -ts_exact_final_time <stepover,interpolate,matchstep> before calling TSStep()");
  if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP && !ts->adapt) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Since TS is not adaptive you cannot use TS_EXACTFINALTIME_MATCHSTEP, suggest TS_EXACTFINALTIME_INTERPOLATE");

  if (!ts->steps) ts->ptime_prev = ts->ptime;
  ptime = ts->ptime; ts->ptime_prev_rollback = ts->ptime_prev;
  ts->reason = TS_CONVERGED_ITERATING;

  ierr = PetscLogEventBegin(TS_Step,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*ts->ops->step)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_Step,ts,0,0,0);CHKERRQ(ierr);

  if (ts->reason >= 0) {
    ts->ptime_prev = ptime;
    ts->steps++;
    ts->steprollback = PETSC_FALSE;
    ts->steprestart  = PETSC_FALSE;
  }

  if (!ts->reason) {
    if (ts->steps >= ts->max_steps) ts->reason = TS_CONVERGED_ITS;
    else if (ts->ptime >= ts->max_time) ts->reason = TS_CONVERGED_TIME;
  }

  if (ts->reason < 0 && ts->errorifstepfailed && ts->reason == TS_DIVERGED_NONLINEAR_SOLVE) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s, increase -ts_max_snes_failures or make negative to attempt recovery",TSConvergedReasons[ts->reason]);
  if (ts->reason < 0 && ts->errorifstepfailed) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s",TSConvergedReasons[ts->reason]);
  PetscFunctionReturn(0);
}

/*@
   TSEvaluateWLTE - Evaluate the weighted local truncation error norm
   at the end of a time step with a given order of accuracy.

   Collective on TS

   Input Parameters:
+  ts - time stepping context
-  wnormtype - norm type, either NORM_2 or NORM_INFINITY

   Input/Output Parameter:
.  order - optional, desired order for the error evaluation or PETSC_DECIDE;
           on output, the actual order of the error evaluation

   Output Parameter:
.  wlte - the weighted local truncation error norm

   Level: advanced

   Notes:
   If the timestepper cannot evaluate the error in a particular step
   (eg. in the first step or restart steps after event handling),
   this routine returns wlte=-1.0 .

.seealso: TSStep(), TSAdapt, TSErrorWeightedNorm()
@*/
PetscErrorCode TSEvaluateWLTE(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidType(ts,1);
  PetscValidLogicalCollectiveEnum(ts,wnormtype,2);
  if (order) PetscValidIntPointer(order,3);
  if (order) PetscValidLogicalCollectiveInt(ts,*order,3);
  PetscValidRealPointer(wlte,4);
  if (wnormtype != NORM_2 && wnormtype != NORM_INFINITY) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"No support for norm type %s",NormTypes[wnormtype]);
  if (!ts->ops->evaluatewlte) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSEvaluateWLTE not implemented for type '%s'",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->evaluatewlte)(ts,wnormtype,order,wlte);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSEvaluateStep - Evaluate the solution at the end of a time step with a given order of accuracy.

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  order - desired order of accuracy
-  done - whether the step was evaluated at this order (pass NULL to generate an error if not available)

   Output Parameter:
.  U - state at the end of the current step

   Level: advanced

   Notes:
   This function cannot be called until all stages have been evaluated.
   It is normally called by adaptive controllers before a step has been accepted and may also be called by the user after TSStep() has returned.

.seealso: TSStep(), TSAdapt
@*/
PetscErrorCode TSEvaluateStep(TS ts,PetscInt order,Vec U,PetscBool *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidType(ts,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  if (!ts->ops->evaluatestep) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSEvaluateStep not implemented for type '%s'",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->evaluatestep)(ts,order,U,done);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSGetComputeInitialCondition - Get the function used to automatically compute an initial condition for the timestepping.

  Not collective

  Input Parameter:
. ts        - time stepping context

  Output Parameter:
. initConditions - The function which computes an initial condition

   Level: advanced

   Notes:
   The calling sequence for the function is
$ initCondition(TS ts, Vec u)
$ ts - The timestepping context
$ u  - The input vector in which the initial condition is stored

.seealso: TSSetComputeInitialCondition(), TSComputeInitialCondition()
@*/
PetscErrorCode TSGetComputeInitialCondition(TS ts, PetscErrorCode (**initCondition)(TS, Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(initCondition, 2);
  *initCondition = ts->ops->initcondition;
  PetscFunctionReturn(0);
}

/*@C
  TSSetComputeInitialCondition - Set the function used to automatically compute an initial condition for the timestepping.

  Logically collective on ts

  Input Parameters:
+ ts        - time stepping context
- initCondition - The function which computes an initial condition

  Level: advanced

  Calling sequence for initCondition:
$ PetscErrorCode initCondition(TS ts, Vec u)

+ ts - The timestepping context
- u  - The input vector in which the initial condition is to be stored

.seealso: TSGetComputeInitialCondition(), TSComputeInitialCondition()
@*/
PetscErrorCode TSSetComputeInitialCondition(TS ts, PetscErrorCode (*initCondition)(TS, Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidFunction(initCondition, 2);
  ts->ops->initcondition = initCondition;
  PetscFunctionReturn(0);
}

/*@
  TSComputeInitialCondition - Compute an initial condition for the timestepping using the function previously set.

  Collective on ts

  Input Parameters:
+ ts - time stepping context
- u  - The Vec to store the condition in which will be used in TSSolve()

  Level: advanced

.seealso: TSGetComputeInitialCondition(), TSSetComputeInitialCondition(), TSSolve()
@*/
PetscErrorCode TSComputeInitialCondition(TS ts, Vec u)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 2);
  if (ts->ops->initcondition) {ierr = (*ts->ops->initcondition)(ts, u);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  TSGetComputeExactError - Get the function used to automatically compute the exact error for the timestepping.

  Not collective

  Input Parameter:
. ts         - time stepping context

  Output Parameter:
. exactError - The function which computes the solution error

  Level: advanced

  Calling sequence for exactError:
$ PetscErrorCode exactError(TS ts, Vec u)

+ ts - The timestepping context
. u  - The approximate solution vector
- e  - The input vector in which the error is stored

.seealso: TSGetComputeExactError(), TSComputeExactError()
@*/
PetscErrorCode TSGetComputeExactError(TS ts, PetscErrorCode (**exactError)(TS, Vec, Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(exactError, 2);
  *exactError = ts->ops->exacterror;
  PetscFunctionReturn(0);
}

/*@C
  TSSetComputeExactError - Set the function used to automatically compute the exact error for the timestepping.

  Logically collective on ts

  Input Parameters:
+ ts         - time stepping context
- exactError - The function which computes the solution error

  Level: advanced

  Calling sequence for exactError:
$ PetscErrorCode exactError(TS ts, Vec u)

+ ts - The timestepping context
. u  - The approximate solution vector
- e  - The input vector in which the error is stored

.seealso: TSGetComputeExactError(), TSComputeExactError()
@*/
PetscErrorCode TSSetComputeExactError(TS ts, PetscErrorCode (*exactError)(TS, Vec, Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidFunction(exactError, 2);
  ts->ops->exacterror = exactError;
  PetscFunctionReturn(0);
}

/*@
  TSComputeExactError - Compute the solution error for the timestepping using the function previously set.

  Collective on ts

  Input Parameters:
+ ts - time stepping context
. u  - The approximate solution
- e  - The Vec used to store the error

  Level: advanced

.seealso: TSGetComputeInitialCondition(), TSSetComputeInitialCondition(), TSSolve()
@*/
PetscErrorCode TSComputeExactError(TS ts, Vec u, Vec e)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(e, VEC_CLASSID, 3);
  if (ts->ops->exacterror) {ierr = (*ts->ops->exacterror)(ts, u, e);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
   TSSolve - Steps the requested number of timesteps.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  u - the solution vector  (can be null if TSSetSolution() was used and TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP) was not used,
                             otherwise must contain the initial conditions and will contain the solution at the final requested time

   Level: beginner

   Notes:
   The final time returned by this function may be different from the time of the internally
   held state accessible by TSGetSolution() and TSGetTime() because the method may have
   stepped over the final time.

.seealso: TSCreate(), TSSetSolution(), TSStep(), TSGetTime(), TSGetSolveTime()
@*/
PetscErrorCode TSSolve(TS ts,Vec u)
{
  Vec               solution;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (u) PetscValidHeaderSpecific(u,VEC_CLASSID,2);

  ierr = TSSetExactFinalTimeDefault(ts);CHKERRQ(ierr);
  if (ts->exact_final_time == TS_EXACTFINALTIME_INTERPOLATE && u) {   /* Need ts->vec_sol to be distinct so it is not overwritten when we interpolate at the end */
    if (!ts->vec_sol || u == ts->vec_sol) {
      ierr = VecDuplicate(u,&solution);CHKERRQ(ierr);
      ierr = TSSetSolution(ts,solution);CHKERRQ(ierr);
      ierr = VecDestroy(&solution);CHKERRQ(ierr); /* grant ownership */
    }
    ierr = VecCopy(u,ts->vec_sol);CHKERRQ(ierr);
    if (ts->forward_solve) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Sensitivity analysis does not support the mode TS_EXACTFINALTIME_INTERPOLATE");
  } else if (u) {
    ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  }
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(ts->trajectory,ts);CHKERRQ(ierr);

  if (ts->max_time >= PETSC_MAX_REAL && ts->max_steps == PETSC_MAX_INT) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"You must call TSSetMaxTime() or TSSetMaxSteps(), or use -ts_max_time <time> or -ts_max_steps <steps>");
  if (ts->exact_final_time == TS_EXACTFINALTIME_UNSPECIFIED) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"You must call TSSetExactFinalTime() or use -ts_exact_final_time <stepover,interpolate,matchstep> before calling TSSolve()");
  if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP && !ts->adapt) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Since TS is not adaptive you cannot use TS_EXACTFINALTIME_MATCHSTEP, suggest TS_EXACTFINALTIME_INTERPOLATE");

  if (ts->forward_solve) {
    ierr = TSForwardSetUp(ts);CHKERRQ(ierr);
  }

  /* reset number of steps only when the step is not restarted. ARKIMEX
     restarts the step after an event. Resetting these counters in such case causes
     TSTrajectory to incorrectly save the output files
  */
  /* reset time step and iteration counters */
  if (!ts->steps) {
    ts->ksp_its           = 0;
    ts->snes_its          = 0;
    ts->num_snes_failures = 0;
    ts->reject            = 0;
    ts->steprestart       = PETSC_TRUE;
    ts->steprollback      = PETSC_FALSE;
    ts->rhsjacobian.time  = PETSC_MIN_REAL;
  }

  /* make sure initial time step does not overshoot final time */
  if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    PetscReal maxdt = ts->max_time-ts->ptime;
    PetscReal dt = ts->time_step;

    ts->time_step = dt >= maxdt ? maxdt : (PetscIsCloseAtTol(dt,maxdt,10*PETSC_MACHINE_EPSILON,0) ? maxdt : dt);
  }
  ts->reason = TS_CONVERGED_ITERATING;

  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;
    static PetscBool  incall = PETSC_FALSE;

    if (!incall) {
      /* Estimate the convergence rate of the time discretization */
      ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) ts),((PetscObject)ts)->options, ((PetscObject) ts)->prefix, "-ts_convergence_estimate", &viewer, &format, &flg);CHKERRQ(ierr);
      if (flg) {
        PetscConvEst conv;
        DM           dm;
        PetscReal   *alpha; /* Convergence rate of the solution error for each field in the L_2 norm */
        PetscInt     Nf;
        PetscBool    checkTemporal = PETSC_TRUE;

        incall = PETSC_TRUE;
        ierr = PetscOptionsGetBool(((PetscObject)ts)->options, ((PetscObject) ts)->prefix, "-ts_convergence_temporal", &checkTemporal, &flg);CHKERRQ(ierr);
        ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
        ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
        ierr = PetscCalloc1(PetscMax(Nf, 1), &alpha);CHKERRQ(ierr);
        ierr = PetscConvEstCreate(PetscObjectComm((PetscObject) ts), &conv);CHKERRQ(ierr);
        ierr = PetscConvEstUseTS(conv, checkTemporal);CHKERRQ(ierr);
        ierr = PetscConvEstSetSolver(conv, (PetscObject) ts);CHKERRQ(ierr);
        ierr = PetscConvEstSetFromOptions(conv);CHKERRQ(ierr);
        ierr = PetscConvEstSetUp(conv);CHKERRQ(ierr);
        ierr = PetscConvEstGetConvRate(conv, alpha);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
        ierr = PetscConvEstRateView(conv, alpha, viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        ierr = PetscConvEstDestroy(&conv);CHKERRQ(ierr);
        ierr = PetscFree(alpha);CHKERRQ(ierr);
        incall = PETSC_FALSE;
      }
    }
  }

  ierr = TSViewFromOptions(ts,NULL,"-ts_view_pre");CHKERRQ(ierr);

  if (ts->ops->solve) { /* This private interface is transitional and should be removed when all implementations are updated. */
    ierr = (*ts->ops->solve)(ts);CHKERRQ(ierr);
    if (u) {ierr = VecCopy(ts->vec_sol,u);CHKERRQ(ierr);}
    ts->solvetime = ts->ptime;
    solution = ts->vec_sol;
  } else { /* Step the requested number of timesteps. */
    if (ts->steps >= ts->max_steps) ts->reason = TS_CONVERGED_ITS;
    else if (ts->ptime >= ts->max_time) ts->reason = TS_CONVERGED_TIME;

    if (!ts->steps) {
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      ierr = TSEventInitialize(ts->event,ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }

    while (!ts->reason) {
      ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      if (!ts->steprollback) {
        ierr = TSPreStep(ts);CHKERRQ(ierr);
      }
      ierr = TSStep(ts);CHKERRQ(ierr);
      if (ts->testjacobian) {
        ierr = TSRHSJacobianTest(ts,NULL);CHKERRQ(ierr);
      }
      if (ts->testjacobiantranspose) {
        ierr = TSRHSJacobianTestTranspose(ts,NULL);CHKERRQ(ierr);
      }
      if (ts->quadraturets && ts->costintegralfwd) { /* Must evaluate the cost integral before event is handled. The cost integral value can also be rolled back. */
        if (ts->reason >= 0) ts->steps--; /* Revert the step number changed by TSStep() */
        ierr = TSForwardCostIntegral(ts);CHKERRQ(ierr);
        if (ts->reason >= 0) ts->steps++;
      }
      if (ts->forward_solve) { /* compute forward sensitivities before event handling because postevent() may change RHS and jump conditions may have to be applied */
        if (ts->reason >= 0) ts->steps--; /* Revert the step number changed by TSStep() */
        ierr = TSForwardStep(ts);CHKERRQ(ierr);
        if (ts->reason >= 0) ts->steps++;
      }
      ierr = TSPostEvaluate(ts);CHKERRQ(ierr);
      ierr = TSEventHandler(ts);CHKERRQ(ierr); /* The right-hand side may be changed due to event. Be careful with Any computation using the RHS information after this point. */
      if (ts->steprollback) {
        ierr = TSPostEvaluate(ts);CHKERRQ(ierr);
      }
      if (!ts->steprollback) {
        ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
        ierr = TSPostStep(ts);CHKERRQ(ierr);
      }
    }
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

    if (ts->exact_final_time == TS_EXACTFINALTIME_INTERPOLATE && ts->ptime > ts->max_time) {
      ierr = TSInterpolate(ts,ts->max_time,u);CHKERRQ(ierr);
      ts->solvetime = ts->max_time;
      solution = u;
      ierr = TSMonitor(ts,-1,ts->solvetime,solution);CHKERRQ(ierr);
    } else {
      if (u) {ierr = VecCopy(ts->vec_sol,u);CHKERRQ(ierr);}
      ts->solvetime = ts->ptime;
      solution = ts->vec_sol;
    }
  }

  ierr = TSViewFromOptions(ts,NULL,"-ts_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(solution,(PetscObject)ts,"-ts_view_solution");CHKERRQ(ierr);
  ierr = PetscObjectSAWsBlock((PetscObject)ts);CHKERRQ(ierr);
  if (ts->adjoint_solve) {
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSGetTime - Gets the time of the most recently completed step.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  t  - the current time. This time may not corresponds to the final time set with TSSetMaxTime(), use TSGetSolveTime().

   Level: beginner

   Note:
   When called during time step evaluation (e.g. during residual evaluation or via hooks set using TSSetPreStep(),
   TSSetPreStage(), TSSetPostStage(), or TSSetPostStep()), the time is the time at the start of the step being evaluated.

.seealso:  TSGetSolveTime(), TSSetTime(), TSGetTimeStep(), TSGetStepNumber()

@*/
PetscErrorCode  TSGetTime(TS ts,PetscReal *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidRealPointer(t,2);
  *t = ts->ptime;
  PetscFunctionReturn(0);
}

/*@
   TSGetPrevTime - Gets the starting time of the previously completed step.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  t  - the previous time

   Level: beginner

.seealso: TSGetTime(), TSGetSolveTime(), TSGetTimeStep()

@*/
PetscErrorCode  TSGetPrevTime(TS ts,PetscReal *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidRealPointer(t,2);
  *t = ts->ptime_prev;
  PetscFunctionReturn(0);
}

/*@
   TSSetTime - Allows one to reset the time.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  time - the time

   Level: intermediate

.seealso: TSGetTime(), TSSetMaxSteps()

@*/
PetscErrorCode  TSSetTime(TS ts, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t,2);
  ts->ptime = t;
  PetscFunctionReturn(0);
}

/*@C
   TSSetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Logically Collective on TS

   Input Parameters:
+  ts     - The TS context
-  prefix - The prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: TSSetFromOptions()

@*/
PetscErrorCode  TSSetOptionsPrefix(TS ts,const char prefix[])
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSAppendOptionsPrefix - Appends to the prefix used for searching for all
   TS options in the database.

   Logically Collective on TS

   Input Parameters:
+  ts     - The TS context
-  prefix - The prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: TSGetOptionsPrefix()

@*/
PetscErrorCode  TSAppendOptionsPrefix(TS ts,const char prefix[])
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESAppendOptionsPrefix(snes,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSGetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Not Collective

   Input Parameter:
.  ts - The TS context

   Output Parameter:
.  prefix - A pointer to the prefix string used

   Notes:
    On the fortran side, the user should pass in a string 'prifix' of
   sufficient length to hold the prefix.

   Level: intermediate

.seealso: TSAppendOptionsPrefix()
@*/
PetscErrorCode  TSGetOptionsPrefix(TS ts,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSGetRHSJacobian - Returns the Jacobian J at the present timestep.

   Not Collective, but parallel objects are returned if TS is parallel

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
+  Amat - The (approximate) Jacobian J of G, where U_t = G(U,t)  (or NULL)
.  Pmat - The matrix from which the preconditioner is constructed, usually the same as Amat  (or NULL)
.  func - Function to compute the Jacobian of the RHS  (or NULL)
-  ctx - User-defined context for Jacobian evaluation routine  (or NULL)

   Notes:
    You can pass in NULL for any return argument you do not need.

   Level: intermediate

.seealso: TSGetTimeStep(), TSGetMatrices(), TSGetTime(), TSGetStepNumber()

@*/
PetscErrorCode  TSGetRHSJacobian(TS ts,Mat *Amat,Mat *Pmat,TSRHSJacobian *func,void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  if (Amat || Pmat) {
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,Amat,Pmat,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSJacobian(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSGetIJacobian - Returns the implicit Jacobian at the present timestep.

   Not Collective, but parallel objects are returned if TS is parallel

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
+  Amat  - The (approximate) Jacobian of F(t,U,U_t)
.  Pmat - The matrix from which the preconditioner is constructed, often the same as Amat
.  f   - The function to compute the matrices
- ctx - User-defined context for Jacobian evaluation routine

   Notes:
    You can pass in NULL for any return argument you do not need.

   Level: advanced

.seealso: TSGetTimeStep(), TSGetRHSJacobian(), TSGetMatrices(), TSGetTime(), TSGetStepNumber()

@*/
PetscErrorCode  TSGetIJacobian(TS ts,Mat *Amat,Mat *Pmat,TSIJacobian *f,void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  if (Amat || Pmat) {
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,Amat,Pmat,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetIJacobian(dm,f,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/dmimpl.h>
/*@
   TSSetDM - Sets the DM that may be used by some nonlinear solvers or preconditioners under the TS

   Logically Collective on ts

   Input Parameters:
+  ts - the ODE integrator object
-  dm - the dm, cannot be NULL

   Notes:
   A DM can only be used for solving one problem at a time because information about the problem is stored on the DM,
   even when not using interfaces like DMTSSetIFunction().  Use DMClone() to get a distinct DM when solving
   different problems using the same function space.

   Level: intermediate

.seealso: TSGetDM(), SNESSetDM(), SNESGetDM()
@*/
PetscErrorCode  TSSetDM(TS ts,DM dm)
{
  PetscErrorCode ierr;
  SNES           snes;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  if (ts->dm) {               /* Move the DMTS context over to the new DM unless the new DM already has one */
    if (ts->dm->dmts && !dm->dmts) {
      ierr = DMCopyDMTS(ts->dm,dm);CHKERRQ(ierr);
      ierr = DMGetDMTS(ts->dm,&tsdm);CHKERRQ(ierr);
      if (tsdm->originaldm == ts->dm) { /* Grant write privileges to the replacement DM */
        tsdm->originaldm = dm;
      }
    }
    ierr = DMDestroy(&ts->dm);CHKERRQ(ierr);
  }
  ts->dm = dm;

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSGetDM - Gets the DM that may be used by some preconditioners

   Not Collective

   Input Parameter:
. ts - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate

.seealso: TSSetDM(), SNESSetDM(), SNESGetDM()
@*/
PetscErrorCode  TSGetDM(TS ts,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->dm) {
    ierr = DMShellCreate(PetscObjectComm((PetscObject)ts),&ts->dm);CHKERRQ(ierr);
    if (ts->snes) {ierr = SNESSetDM(ts->snes,ts->dm);CHKERRQ(ierr);}
  }
  *dm = ts->dm;
  PetscFunctionReturn(0);
}

/*@
   SNESTSFormFunction - Function to evaluate nonlinear residual

   Logically Collective on SNES

   Input Parameters:
+ snes - nonlinear solver
. U - the current state at which to evaluate the residual
- ctx - user context, must be a TS

   Output Parameter:
. F - the nonlinear residual

   Notes:
   This function is not normally called by users and is automatically registered with the SNES used by TS.
   It is most frequently passed to MatFDColoringSetFunction().

   Level: advanced

.seealso: SNESSetFunction(), MatFDColoringSetFunction()
@*/
PetscErrorCode  SNESTSFormFunction(SNES snes,Vec U,Vec F,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscValidHeaderSpecific(ts,TS_CLASSID,4);
  ierr = (ts->ops->snesfunction)(snes,U,F,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   SNESTSFormJacobian - Function to evaluate the Jacobian

   Collective on SNES

   Input Parameters:
+ snes - nonlinear solver
. U - the current state at which to evaluate the residual
- ctx - user context, must be a TS

   Output Parameters:
+ A - the Jacobian
- B - the preconditioning matrix (may be the same as A)

   Notes:
   This function is not normally called by users and is automatically registered with the SNES used by TS.

   Level: developer

.seealso: SNESSetJacobian()
@*/
PetscErrorCode  SNESTSFormJacobian(SNES snes,Vec U,Mat A,Mat B,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidPointer(A,3);
  PetscValidHeaderSpecific(A,MAT_CLASSID,3);
  PetscValidPointer(B,4);
  PetscValidHeaderSpecific(B,MAT_CLASSID,4);
  PetscValidHeaderSpecific(ts,TS_CLASSID,5);
  ierr = (ts->ops->snesjacobian)(snes,U,A,B,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSComputeRHSFunctionLinear - Evaluate the right hand side via the user-provided Jacobian, for linear problems Udot = A U only

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  t - time at which to evaluate
.  U - state at which to evaluate
-  ctx - context

   Output Parameter:
.  F - right hand side

   Level: intermediate

   Notes:
   This function is intended to be passed to TSSetRHSFunction() to evaluate the right hand side for linear problems.
   The matrix (and optionally the evaluation context) should be passed to TSSetRHSJacobian().

.seealso: TSSetRHSFunction(), TSSetRHSJacobian(), TSComputeRHSJacobianConstant()
@*/
PetscErrorCode TSComputeRHSFunctionLinear(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  Mat            Arhs,Brhs;

  PetscFunctionBegin;
  ierr = TSGetRHSMats_Private(ts,&Arhs,&Brhs);CHKERRQ(ierr);
  /* undo the damage caused by shifting */
  ierr = TSRecoverRHSJacobian(ts,Arhs,Brhs);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(ts,t,U,Arhs,Brhs);CHKERRQ(ierr);
  ierr = MatMult(Arhs,U,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSComputeRHSJacobianConstant - Reuses a Jacobian that is time-independent.

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  t - time at which to evaluate
.  U - state at which to evaluate
-  ctx - context

   Output Parameters:
+  A - pointer to operator
-  B - pointer to preconditioning matrix

   Level: intermediate

   Notes:
   This function is intended to be passed to TSSetRHSJacobian() to evaluate the Jacobian for linear time-independent problems.

.seealso: TSSetRHSFunction(), TSSetRHSJacobian(), TSComputeRHSFunctionLinear()
@*/
PetscErrorCode TSComputeRHSJacobianConstant(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@C
   TSComputeIFunctionLinear - Evaluate the left hand side via the user-provided Jacobian, for linear problems only

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  t - time at which to evaluate
.  U - state at which to evaluate
.  Udot - time derivative of state vector
-  ctx - context

   Output Parameter:
.  F - left hand side

   Level: intermediate

   Notes:
   The assumption here is that the left hand side is of the form A*Udot (and not A*Udot + B*U). For other cases, the
   user is required to write their own TSComputeIFunction.
   This function is intended to be passed to TSSetIFunction() to evaluate the left hand side for linear problems.
   The matrix (and optionally the evaluation context) should be passed to TSSetIJacobian().

   Note that using this function is NOT equivalent to using TSComputeRHSFunctionLinear() since that solves Udot = A U

.seealso: TSSetIFunction(), TSSetIJacobian(), TSComputeIJacobianConstant(), TSComputeRHSFunctionLinear()
@*/
PetscErrorCode TSComputeIFunctionLinear(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  Mat            A,B;

  PetscFunctionBegin;
  ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
  ierr = TSComputeIJacobian(ts,t,U,Udot,1.0,A,B,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatMult(A,Udot,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSComputeIJacobianConstant - Reuses a time-independent for a semi-implicit DAE or ODE

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  t - time at which to evaluate
.  U - state at which to evaluate
.  Udot - time derivative of state vector
.  shift - shift to apply
-  ctx - context

   Output Parameters:
+  A - pointer to operator
-  B - pointer to preconditioning matrix

   Level: advanced

   Notes:
   This function is intended to be passed to TSSetIJacobian() to evaluate the Jacobian for linear time-independent problems.

   It is only appropriate for problems of the form

$     M Udot = F(U,t)

  where M is constant and F is non-stiff.  The user must pass M to TSSetIJacobian().  The current implementation only
  works with IMEX time integration methods such as TSROSW and TSARKIMEX, since there is no support for de-constructing
  an implicit operator of the form

$    shift*M + J

  where J is the Jacobian of -F(U).  Support may be added in a future version of PETSc, but for now, the user must store
  a copy of M or reassemble it when requested.

.seealso: TSSetIFunction(), TSSetIJacobian(), TSComputeIFunctionLinear()
@*/
PetscErrorCode TSComputeIJacobianConstant(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal shift,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(A, shift / ts->ijacobian.shift);CHKERRQ(ierr);
  ts->ijacobian.shift = shift;
  PetscFunctionReturn(0);
}

/*@
   TSGetEquationType - Gets the type of the equation that TS is solving.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  equation_type - see TSEquationType

   Level: beginner

.seealso: TSSetEquationType(), TSEquationType
@*/
PetscErrorCode  TSGetEquationType(TS ts,TSEquationType *equation_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(equation_type,2);
  *equation_type = ts->equation_type;
  PetscFunctionReturn(0);
}

/*@
   TSSetEquationType - Sets the type of the equation that TS is solving.

   Not Collective

   Input Parameters:
+  ts - the TS context
-  equation_type - see TSEquationType

   Level: advanced

.seealso: TSGetEquationType(), TSEquationType
@*/
PetscErrorCode  TSSetEquationType(TS ts,TSEquationType equation_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->equation_type = equation_type;
  PetscFunctionReturn(0);
}

/*@
   TSGetConvergedReason - Gets the reason the TS iteration was stopped.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see TSConvergedReason or the
            manual pages for the individual convergence tests for complete lists

   Level: beginner

   Notes:
   Can only be called after the call to TSSolve() is complete.

.seealso: TSSetConvergenceTest(), TSConvergedReason
@*/
PetscErrorCode  TSGetConvergedReason(TS ts,TSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = ts->reason;
  PetscFunctionReturn(0);
}

/*@
   TSSetConvergedReason - Sets the reason for handling the convergence of TSSolve.

   Logically Collective; reason must contain common value

   Input Parameters:
+  ts - the TS context
-  reason - negative value indicates diverged, positive value converged, see TSConvergedReason or the
            manual pages for the individual convergence tests for complete lists

   Level: advanced

   Notes:
   Can only be called while TSSolve() is active.

.seealso: TSConvergedReason
@*/
PetscErrorCode  TSSetConvergedReason(TS ts,TSConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->reason = reason;
  PetscFunctionReturn(0);
}

/*@
   TSGetSolveTime - Gets the time after a call to TSSolve()

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  ftime - the final time. This time corresponds to the final time set with TSSetMaxTime()

   Level: beginner

   Notes:
   Can only be called after the call to TSSolve() is complete.

.seealso: TSSetConvergenceTest(), TSConvergedReason
@*/
PetscErrorCode  TSGetSolveTime(TS ts,PetscReal *ftime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(ftime,2);
  *ftime = ts->solvetime;
  PetscFunctionReturn(0);
}

/*@
   TSGetSNESIterations - Gets the total number of nonlinear iterations
   used by the time integrator.

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  nits - number of nonlinear iterations

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.seealso:  TSGetKSPIterations()
@*/
PetscErrorCode TSGetSNESIterations(TS ts,PetscInt *nits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(nits,2);
  *nits = ts->snes_its;
  PetscFunctionReturn(0);
}

/*@
   TSGetKSPIterations - Gets the total number of linear iterations
   used by the time integrator.

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.seealso:  TSGetSNESIterations(), SNESGetKSPIterations()
@*/
PetscErrorCode TSGetKSPIterations(TS ts,PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(lits,2);
  *lits = ts->ksp_its;
  PetscFunctionReturn(0);
}

/*@
   TSGetStepRejections - Gets the total number of rejected steps.

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  rejects - number of steps rejected

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetSNESFailures(), TSSetMaxSNESFailures(), TSSetErrorIfStepFails()
@*/
PetscErrorCode TSGetStepRejections(TS ts,PetscInt *rejects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(rejects,2);
  *rejects = ts->reject;
  PetscFunctionReturn(0);
}

/*@
   TSGetSNESFailures - Gets the total number of failed SNES solves

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  fails - number of failed nonlinear solves

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetStepRejections(), TSSetMaxSNESFailures()
@*/
PetscErrorCode TSGetSNESFailures(TS ts,PetscInt *fails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(fails,2);
  *fails = ts->num_snes_failures;
  PetscFunctionReturn(0);
}

/*@
   TSSetMaxStepRejections - Sets the maximum number of step rejections before a step fails

   Not Collective

   Input Parameters:
+  ts - TS context
-  rejects - maximum number of rejected steps, pass -1 for unlimited

   Notes:
   The counter is reset to zero for each step

   Options Database Key:
 .  -ts_max_reject - Maximum number of step rejections before a step fails

   Level: intermediate

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxSNESFailures(), TSGetStepRejections(), TSGetSNESFailures(), TSSetErrorIfStepFails(), TSGetConvergedReason()
@*/
PetscErrorCode TSSetMaxStepRejections(TS ts,PetscInt rejects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->max_reject = rejects;
  PetscFunctionReturn(0);
}

/*@
   TSSetMaxSNESFailures - Sets the maximum number of failed SNES solves

   Not Collective

   Input Parameters:
+  ts - TS context
-  fails - maximum number of failed nonlinear solves, pass -1 for unlimited

   Notes:
   The counter is reset to zero for each successive call to TSSolve().

   Options Database Key:
 .  -ts_max_snes_failures - Maximum number of nonlinear solve failures

   Level: intermediate

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetStepRejections(), TSGetSNESFailures(), SNESGetConvergedReason(), TSGetConvergedReason()
@*/
PetscErrorCode TSSetMaxSNESFailures(TS ts,PetscInt fails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->max_snes_failures = fails;
  PetscFunctionReturn(0);
}

/*@
   TSSetErrorIfStepFails - Error if no step succeeds

   Not Collective

   Input Parameters:
+  ts - TS context
-  err - PETSC_TRUE to error if no step succeeds, PETSC_FALSE to return without failure

   Options Database Key:
 .  -ts_error_if_step_fails - Error if no step succeeds

   Level: intermediate

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetStepRejections(), TSGetSNESFailures(), TSSetErrorIfStepFails(), TSGetConvergedReason()
@*/
PetscErrorCode TSSetErrorIfStepFails(TS ts,PetscBool err)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->errorifstepfailed = err;
  PetscFunctionReturn(0);
}

/*@
   TSGetAdapt - Get the adaptive controller context for the current method

   Collective on TS if controller has not been created yet

   Input Parameter:
.  ts - time stepping context

   Output Parameter:
.  adapt - adaptive controller

   Level: intermediate

.seealso: TSAdapt, TSAdaptSetType(), TSAdaptChoose()
@*/
PetscErrorCode TSGetAdapt(TS ts,TSAdapt *adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(adapt,2);
  if (!ts->adapt) {
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)ts),&ts->adapt);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ts,(PetscObject)ts->adapt);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->adapt,(PetscObject)ts,1);CHKERRQ(ierr);
  }
  *adapt = ts->adapt;
  PetscFunctionReturn(0);
}

/*@
   TSSetTolerances - Set tolerances for local truncation error when using adaptive controller

   Logically Collective

   Input Parameters:
+  ts - time integration context
.  atol - scalar absolute tolerances, PETSC_DECIDE to leave current value
.  vatol - vector of absolute tolerances or NULL, used in preference to atol if present
.  rtol - scalar relative tolerances, PETSC_DECIDE to leave current value
-  vrtol - vector of relative tolerances or NULL, used in preference to atol if present

   Options Database keys:
+  -ts_rtol <rtol> - relative tolerance for local truncation error
-  -ts_atol <atol> Absolute tolerance for local truncation error

   Notes:
   With PETSc's implicit schemes for DAE problems, the calculation of the local truncation error
   (LTE) includes both the differential and the algebraic variables. If one wants the LTE to be
   computed only for the differential or the algebraic part then this can be done using the vector of
   tolerances vatol. For example, by setting the tolerance vector with the desired tolerance for the
   differential part and infinity for the algebraic part, the LTE calculation will include only the
   differential variables.

   Level: beginner

.seealso: TS, TSAdapt, TSErrorWeightedNorm(), TSGetTolerances()
@*/
PetscErrorCode TSSetTolerances(TS ts,PetscReal atol,Vec vatol,PetscReal rtol,Vec vrtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (atol != PETSC_DECIDE && atol != PETSC_DEFAULT) ts->atol = atol;
  if (vatol) {
    ierr = PetscObjectReference((PetscObject)vatol);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vatol);CHKERRQ(ierr);
    ts->vatol = vatol;
  }
  if (rtol != PETSC_DECIDE && rtol != PETSC_DEFAULT) ts->rtol = rtol;
  if (vrtol) {
    ierr = PetscObjectReference((PetscObject)vrtol);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vrtol);CHKERRQ(ierr);
    ts->vrtol = vrtol;
  }
  PetscFunctionReturn(0);
}

/*@
   TSGetTolerances - Get tolerances for local truncation error when using adaptive controller

   Logically Collective

   Input Parameter:
.  ts - time integration context

   Output Parameters:
+  atol - scalar absolute tolerances, NULL to ignore
.  vatol - vector of absolute tolerances, NULL to ignore
.  rtol - scalar relative tolerances, NULL to ignore
-  vrtol - vector of relative tolerances, NULL to ignore

   Level: beginner

.seealso: TS, TSAdapt, TSErrorWeightedNorm(), TSSetTolerances()
@*/
PetscErrorCode TSGetTolerances(TS ts,PetscReal *atol,Vec *vatol,PetscReal *rtol,Vec *vrtol)
{
  PetscFunctionBegin;
  if (atol)  *atol  = ts->atol;
  if (vatol) *vatol = ts->vatol;
  if (rtol)  *rtol  = ts->rtol;
  if (vrtol) *vrtol = ts->vrtol;
  PetscFunctionReturn(0);
}

/*@
   TSErrorWeightedNorm2 - compute a weighted 2-norm of the difference between two state vectors

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  U - state vector, usually ts->vec_sol
-  Y - state vector to be compared to U

   Output Parameters:
+  norm - weighted norm, a value of 1.0 means that the error matches the tolerances
.  norma - weighted norm based on the absolute tolerance, a value of 1.0 means that the error matches the tolerances
-  normr - weighted norm based on the relative tolerance, a value of 1.0 means that the error matches the tolerances

   Level: developer

.seealso: TSErrorWeightedNorm(), TSErrorWeightedNormInfinity()
@*/
PetscErrorCode TSErrorWeightedNorm2(TS ts,Vec U,Vec Y,PetscReal *norm,PetscReal *norma,PetscReal *normr)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,N,rstart;
  PetscInt          n_loc,na_loc,nr_loc;
  PetscReal         n_glb,na_glb,nr_glb;
  const PetscScalar *u,*y;
  PetscReal         sum,suma,sumr,gsum,gsuma,gsumr,diff;
  PetscReal         tol,tola,tolr;
  PetscReal         err_loc[6],err_glb[6];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,3);
  PetscValidType(U,2);
  PetscValidType(Y,3);
  PetscCheckSameComm(U,2,Y,3);
  PetscValidPointer(norm,4);
  PetscValidPointer(norma,5);
  PetscValidPointer(normr,6);
  if (U == Y) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_IDN,"U and Y cannot be the same vector");

  ierr = VecGetSize(U,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  sum  = 0.; n_loc  = 0;
  suma = 0.; na_loc = 0;
  sumr = 0.; nr_loc = 0;
  if (ts->vatol && ts->vrtol) {
    const PetscScalar *atol,*rtol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = PetscRealPart(atol[i]);
      if (tola>0.) {
        suma  += PetscSqr(diff/tola);
        na_loc++;
      }
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(diff/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(diff/tol);
        n_loc++;
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else if (ts->vatol) {       /* vector atol, scalar rtol */
    const PetscScalar *atol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = PetscRealPart(atol[i]);
      if (tola>0.) {
        suma  += PetscSqr(diff/tola);
        na_loc++;
      }
      tolr = ts->rtol * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(diff/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(diff/tol);
        n_loc++;
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
  } else if (ts->vrtol) {       /* scalar atol, vector rtol */
    const PetscScalar *rtol;
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = ts->atol;
      if (tola>0.) {
        suma  += PetscSqr(diff/tola);
        na_loc++;
      }
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(diff/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(diff/tol);
        n_loc++;
      }
    }
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else {                      /* scalar atol, scalar rtol */
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = ts->atol;
      if (tola>0.) {
        suma  += PetscSqr(diff/tola);
        na_loc++;
      }
      tolr = ts->rtol * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(diff/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(diff/tol);
        n_loc++;
      }
    }
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);

  err_loc[0] = sum;
  err_loc[1] = suma;
  err_loc[2] = sumr;
  err_loc[3] = (PetscReal)n_loc;
  err_loc[4] = (PetscReal)na_loc;
  err_loc[5] = (PetscReal)nr_loc;

  ierr = MPIU_Allreduce(err_loc,err_glb,6,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)ts));CHKERRMPI(ierr);

  gsum   = err_glb[0];
  gsuma  = err_glb[1];
  gsumr  = err_glb[2];
  n_glb  = err_glb[3];
  na_glb = err_glb[4];
  nr_glb = err_glb[5];

  *norm  = 0.;
  if (n_glb>0.) {*norm  = PetscSqrtReal(gsum  / n_glb);}
  *norma = 0.;
  if (na_glb>0.) {*norma = PetscSqrtReal(gsuma / na_glb);}
  *normr = 0.;
  if (nr_glb>0.) {*normr = PetscSqrtReal(gsumr / nr_glb);}

  if (PetscIsInfOrNanScalar(*norm)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  if (PetscIsInfOrNanScalar(*norma)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norma");
  if (PetscIsInfOrNanScalar(*normr)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in normr");
  PetscFunctionReturn(0);
}

/*@
   TSErrorWeightedNormInfinity - compute a weighted infinity-norm of the difference between two state vectors

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  U - state vector, usually ts->vec_sol
-  Y - state vector to be compared to U

   Output Parameters:
+  norm - weighted norm, a value of 1.0 means that the error matches the tolerances
.  norma - weighted norm based on the absolute tolerance, a value of 1.0 means that the error matches the tolerances
-  normr - weighted norm based on the relative tolerance, a value of 1.0 means that the error matches the tolerances

   Level: developer

.seealso: TSErrorWeightedNorm(), TSErrorWeightedNorm2()
@*/
PetscErrorCode TSErrorWeightedNormInfinity(TS ts,Vec U,Vec Y,PetscReal *norm,PetscReal *norma,PetscReal *normr)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,N,rstart;
  const PetscScalar *u,*y;
  PetscReal         max,gmax,maxa,gmaxa,maxr,gmaxr;
  PetscReal         tol,tola,tolr,diff;
  PetscReal         err_loc[3],err_glb[3];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,3);
  PetscValidType(U,2);
  PetscValidType(Y,3);
  PetscCheckSameComm(U,2,Y,3);
  PetscValidPointer(norm,4);
  PetscValidPointer(norma,5);
  PetscValidPointer(normr,6);
  if (U == Y) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_IDN,"U and Y cannot be the same vector");

  ierr = VecGetSize(U,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);

  max=0.;
  maxa=0.;
  maxr=0.;

  if (ts->vatol && ts->vrtol) {     /* vector atol, vector rtol */
    const PetscScalar *atol,*rtol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);

    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = PetscRealPart(atol[i]);
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,diff / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,diff / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,diff / tol);
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else if (ts->vatol) {       /* vector atol, scalar rtol */
    const PetscScalar *atol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = PetscRealPart(atol[i]);
      tolr = ts->rtol  * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,diff / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,diff / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,diff / tol);
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
  } else if (ts->vrtol) {       /* scalar atol, vector rtol */
    const PetscScalar *rtol;
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);

    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = ts->atol;
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,diff / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,diff / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,diff / tol);
      }
    }
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else {                      /* scalar atol, scalar rtol */

    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      diff = PetscAbsScalar(y[i] - u[i]);
      tola = ts->atol;
      tolr = ts->rtol * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,diff / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,diff / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,diff / tol);
      }
    }
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  err_loc[0] = max;
  err_loc[1] = maxa;
  err_loc[2] = maxr;
  ierr  = MPIU_Allreduce(err_loc,err_glb,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)ts));CHKERRMPI(ierr);
  gmax   = err_glb[0];
  gmaxa  = err_glb[1];
  gmaxr  = err_glb[2];

  *norm = gmax;
  *norma = gmaxa;
  *normr = gmaxr;
  if (PetscIsInfOrNanScalar(*norm)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    if (PetscIsInfOrNanScalar(*norma)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norma");
    if (PetscIsInfOrNanScalar(*normr)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in normr");
  PetscFunctionReturn(0);
}

/*@
   TSErrorWeightedNorm - compute a weighted norm of the difference between two state vectors based on supplied absolute and relative tolerances

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  U - state vector, usually ts->vec_sol
.  Y - state vector to be compared to U
-  wnormtype - norm type, either NORM_2 or NORM_INFINITY

   Output Parameters:
+  norm  - weighted norm, a value of 1.0 achieves a balance between absolute and relative tolerances
.  norma - weighted norm, a value of 1.0 means that the error meets the absolute tolerance set by the user
-  normr - weighted norm, a value of 1.0 means that the error meets the relative tolerance set by the user

   Options Database Keys:
.  -ts_adapt_wnormtype <wnormtype> - 2, INFINITY

   Level: developer

.seealso: TSErrorWeightedNormInfinity(), TSErrorWeightedNorm2(), TSErrorWeightedENorm
@*/
PetscErrorCode TSErrorWeightedNorm(TS ts,Vec U,Vec Y,NormType wnormtype,PetscReal *norm,PetscReal *norma,PetscReal *normr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (wnormtype == NORM_2) {
    ierr = TSErrorWeightedNorm2(ts,U,Y,norm,norma,normr);CHKERRQ(ierr);
  } else if (wnormtype == NORM_INFINITY) {
    ierr = TSErrorWeightedNormInfinity(ts,U,Y,norm,norma,normr);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for norm type %s",NormTypes[wnormtype]);
  PetscFunctionReturn(0);
}

/*@
   TSErrorWeightedENorm2 - compute a weighted 2 error norm based on supplied absolute and relative tolerances

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  E - error vector
.  U - state vector, usually ts->vec_sol
-  Y - state vector, previous time step

   Output Parameters:
+  norm - weighted norm, a value of 1.0 means that the error matches the tolerances
.  norma - weighted norm based on the absolute tolerance, a value of 1.0 means that the error matches the tolerances
-  normr - weighted norm based on the relative tolerance, a value of 1.0 means that the error matches the tolerances

   Level: developer

.seealso: TSErrorWeightedENorm(), TSErrorWeightedENormInfinity()
@*/
PetscErrorCode TSErrorWeightedENorm2(TS ts,Vec E,Vec U,Vec Y,PetscReal *norm,PetscReal *norma,PetscReal *normr)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,N,rstart;
  PetscInt          n_loc,na_loc,nr_loc;
  PetscReal         n_glb,na_glb,nr_glb;
  const PetscScalar *e,*u,*y;
  PetscReal         err,sum,suma,sumr,gsum,gsuma,gsumr;
  PetscReal         tol,tola,tolr;
  PetscReal         err_loc[6],err_glb[6];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(E,VEC_CLASSID,2);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,4);
  PetscValidType(E,2);
  PetscValidType(U,3);
  PetscValidType(Y,4);
  PetscCheckSameComm(E,2,U,3);
  PetscCheckSameComm(U,3,Y,4);
  PetscValidPointer(norm,5);
  PetscValidPointer(norma,6);
  PetscValidPointer(normr,7);

  ierr = VecGetSize(E,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(E,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(E,&e);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  sum  = 0.; n_loc  = 0;
  suma = 0.; na_loc = 0;
  sumr = 0.; nr_loc = 0;
  if (ts->vatol && ts->vrtol) {
    const PetscScalar *atol,*rtol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = PetscRealPart(atol[i]);
      if (tola>0.) {
        suma  += PetscSqr(err/tola);
        na_loc++;
      }
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(err/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(err/tol);
        n_loc++;
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else if (ts->vatol) {       /* vector atol, scalar rtol */
    const PetscScalar *atol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = PetscRealPart(atol[i]);
      if (tola>0.) {
        suma  += PetscSqr(err/tola);
        na_loc++;
      }
      tolr = ts->rtol * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(err/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(err/tol);
        n_loc++;
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
  } else if (ts->vrtol) {       /* scalar atol, vector rtol */
    const PetscScalar *rtol;
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = ts->atol;
      if (tola>0.) {
        suma  += PetscSqr(err/tola);
        na_loc++;
      }
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(err/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(err/tol);
        n_loc++;
      }
    }
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else {                      /* scalar atol, scalar rtol */
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = ts->atol;
      if (tola>0.) {
        suma  += PetscSqr(err/tola);
        na_loc++;
      }
      tolr = ts->rtol * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      if (tolr>0.) {
        sumr  += PetscSqr(err/tolr);
        nr_loc++;
      }
      tol=tola+tolr;
      if (tol>0.) {
        sum  += PetscSqr(err/tol);
        n_loc++;
      }
    }
  }
  ierr = VecRestoreArrayRead(E,&e);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);

  err_loc[0] = sum;
  err_loc[1] = suma;
  err_loc[2] = sumr;
  err_loc[3] = (PetscReal)n_loc;
  err_loc[4] = (PetscReal)na_loc;
  err_loc[5] = (PetscReal)nr_loc;

  ierr = MPIU_Allreduce(err_loc,err_glb,6,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)ts));CHKERRMPI(ierr);

  gsum   = err_glb[0];
  gsuma  = err_glb[1];
  gsumr  = err_glb[2];
  n_glb  = err_glb[3];
  na_glb = err_glb[4];
  nr_glb = err_glb[5];

  *norm  = 0.;
  if (n_glb>0.) {*norm  = PetscSqrtReal(gsum  / n_glb);}
  *norma = 0.;
  if (na_glb>0.) {*norma = PetscSqrtReal(gsuma / na_glb);}
  *normr = 0.;
  if (nr_glb>0.) {*normr = PetscSqrtReal(gsumr / nr_glb);}

  if (PetscIsInfOrNanScalar(*norm)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  if (PetscIsInfOrNanScalar(*norma)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norma");
  if (PetscIsInfOrNanScalar(*normr)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in normr");
  PetscFunctionReturn(0);
}

/*@
   TSErrorWeightedENormInfinity - compute a weighted infinity error norm based on supplied absolute and relative tolerances
   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  E - error vector
.  U - state vector, usually ts->vec_sol
-  Y - state vector, previous time step

   Output Parameters:
+  norm - weighted norm, a value of 1.0 means that the error matches the tolerances
.  norma - weighted norm based on the absolute tolerance, a value of 1.0 means that the error matches the tolerances
-  normr - weighted norm based on the relative tolerance, a value of 1.0 means that the error matches the tolerances

   Level: developer

.seealso: TSErrorWeightedENorm(), TSErrorWeightedENorm2()
@*/
PetscErrorCode TSErrorWeightedENormInfinity(TS ts,Vec E,Vec U,Vec Y,PetscReal *norm,PetscReal *norma,PetscReal *normr)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,N,rstart;
  const PetscScalar *e,*u,*y;
  PetscReal         err,max,gmax,maxa,gmaxa,maxr,gmaxr;
  PetscReal         tol,tola,tolr;
  PetscReal         err_loc[3],err_glb[3];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(E,VEC_CLASSID,2);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,4);
  PetscValidType(E,2);
  PetscValidType(U,3);
  PetscValidType(Y,4);
  PetscCheckSameComm(E,2,U,3);
  PetscCheckSameComm(U,3,Y,4);
  PetscValidPointer(norm,5);
  PetscValidPointer(norma,6);
  PetscValidPointer(normr,7);

  ierr = VecGetSize(E,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(E,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(E,&e);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);

  max=0.;
  maxa=0.;
  maxr=0.;

  if (ts->vatol && ts->vrtol) {     /* vector atol, vector rtol */
    const PetscScalar *atol,*rtol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);

    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = PetscRealPart(atol[i]);
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,err / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,err / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,err / tol);
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else if (ts->vatol) {       /* vector atol, scalar rtol */
    const PetscScalar *atol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = PetscRealPart(atol[i]);
      tolr = ts->rtol  * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,err / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,err / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,err / tol);
      }
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
  } else if (ts->vrtol) {       /* scalar atol, vector rtol */
    const PetscScalar *rtol;
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);

    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = ts->atol;
      tolr = PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,err / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,err / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,err / tol);
      }
    }
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else {                      /* scalar atol, scalar rtol */

    for (i=0; i<n; i++) {
      SkipSmallValue(y[i],u[i],ts->adapt->ignore_max);
      err = PetscAbsScalar(e[i]);
      tola = ts->atol;
      tolr = ts->rtol * PetscMax(PetscAbsScalar(u[i]),PetscAbsScalar(y[i]));
      tol  = tola+tolr;
      if (tola>0.) {
        maxa = PetscMax(maxa,err / tola);
      }
      if (tolr>0.) {
        maxr = PetscMax(maxr,err / tolr);
      }
      if (tol>0.) {
        max = PetscMax(max,err / tol);
      }
    }
  }
  ierr = VecRestoreArrayRead(E,&e);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  err_loc[0] = max;
  err_loc[1] = maxa;
  err_loc[2] = maxr;
  ierr  = MPIU_Allreduce(err_loc,err_glb,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)ts));CHKERRMPI(ierr);
  gmax   = err_glb[0];
  gmaxa  = err_glb[1];
  gmaxr  = err_glb[2];

  *norm = gmax;
  *norma = gmaxa;
  *normr = gmaxr;
  if (PetscIsInfOrNanScalar(*norm)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    if (PetscIsInfOrNanScalar(*norma)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in norma");
    if (PetscIsInfOrNanScalar(*normr)) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_FP,"Infinite or not-a-number generated in normr");
  PetscFunctionReturn(0);
}

/*@
   TSErrorWeightedENorm - compute a weighted error norm based on supplied absolute and relative tolerances

   Collective on TS

   Input Parameters:
+  ts - time stepping context
.  E - error vector
.  U - state vector, usually ts->vec_sol
.  Y - state vector, previous time step
-  wnormtype - norm type, either NORM_2 or NORM_INFINITY

   Output Parameters:
+  norm  - weighted norm, a value of 1.0 achieves a balance between absolute and relative tolerances
.  norma - weighted norm, a value of 1.0 means that the error meets the absolute tolerance set by the user
-  normr - weighted norm, a value of 1.0 means that the error meets the relative tolerance set by the user

   Options Database Keys:
.  -ts_adapt_wnormtype <wnormtype> - 2, INFINITY

   Level: developer

.seealso: TSErrorWeightedENormInfinity(), TSErrorWeightedENorm2(), TSErrorWeightedNormInfinity(), TSErrorWeightedNorm2()
@*/
PetscErrorCode TSErrorWeightedENorm(TS ts,Vec E,Vec U,Vec Y,NormType wnormtype,PetscReal *norm,PetscReal *norma,PetscReal *normr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (wnormtype == NORM_2) {
    ierr = TSErrorWeightedENorm2(ts,E,U,Y,norm,norma,normr);CHKERRQ(ierr);
  } else if (wnormtype == NORM_INFINITY) {
    ierr = TSErrorWeightedENormInfinity(ts,E,U,Y,norm,norma,normr);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for norm type %s",NormTypes[wnormtype]);
  PetscFunctionReturn(0);
}

/*@
   TSSetCFLTimeLocal - Set the local CFL constraint relative to forward Euler

   Logically Collective on TS

   Input Parameters:
+  ts - time stepping context
-  cfltime - maximum stable time step if using forward Euler (value can be different on each process)

   Note:
   After calling this function, the global CFL time can be obtained by calling TSGetCFLTime()

   Level: intermediate

.seealso: TSGetCFLTime(), TSADAPTCFL
@*/
PetscErrorCode TSSetCFLTimeLocal(TS ts,PetscReal cfltime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->cfltime_local = cfltime;
  ts->cfltime       = -1.;
  PetscFunctionReturn(0);
}

/*@
   TSGetCFLTime - Get the maximum stable time step according to CFL criteria applied to forward Euler

   Collective on TS

   Input Parameter:
.  ts - time stepping context

   Output Parameter:
.  cfltime - maximum stable time step for forward Euler

   Level: advanced

.seealso: TSSetCFLTimeLocal()
@*/
PetscErrorCode TSGetCFLTime(TS ts,PetscReal *cfltime)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->cfltime < 0) {
    ierr = MPIU_Allreduce(&ts->cfltime_local,&ts->cfltime,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)ts));CHKERRMPI(ierr);
  }
  *cfltime = ts->cfltime;
  PetscFunctionReturn(0);
}

/*@
   TSVISetVariableBounds - Sets the lower and upper bounds for the solution vector. xl <= x <= xu

   Input Parameters:
+  ts   - the TS context.
.  xl   - lower bound.
-  xu   - upper bound.

   Notes:
   If this routine is not called then the lower and upper bounds are set to
   PETSC_NINFINITY and PETSC_INFINITY respectively during SNESSetUp().

   Level: advanced

@*/
PetscErrorCode TSVISetVariableBounds(TS ts, Vec xl, Vec xu)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSComputeLinearStability - computes the linear stability function at a point

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  xr,xi - real and imaginary part of input arguments

   Output Parameters:
.  yr,yi - real and imaginary part of function value

   Level: developer

.seealso: TSSetRHSFunction(), TSComputeIFunction()
@*/
PetscErrorCode TSComputeLinearStability(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->ops->linearstability) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Linearized stability function not provided for this method");
  ierr = (*ts->ops->linearstability)(ts,xr,xi,yr,yi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSRestartStep - Flags the solver to restart the next step

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: advanced

   Notes:
   Multistep methods like BDF or Runge-Kutta methods with FSAL property require restarting the solver in the event of
   discontinuities. These discontinuities may be introduced as a consequence of explicitly modifications to the solution
   vector (which PETSc attempts to detect and handle) or problem coefficients (which PETSc is not able to detect). For
   the sake of correctness and maximum safety, users are expected to call TSRestart() whenever they introduce
   discontinuities in callback routines (e.g. prestep and poststep routines, or implicit/rhs function routines with
   discontinuous source terms).

.seealso: TSSolve(), TSSetPreStep(), TSSetPostStep()
@*/
PetscErrorCode TSRestartStep(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->steprestart = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TSRollBack - Rolls back one time step

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: advanced

.seealso: TSCreate(), TSSetUp(), TSDestroy(), TSSolve(), TSSetPreStep(), TSSetPreStage(), TSInterpolate()
@*/
PetscErrorCode  TSRollBack(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  if (ts->steprollback) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TSRollBack already called");
  if (!ts->ops->rollback) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRollBack not implemented for type '%s'",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->rollback)(ts);CHKERRQ(ierr);
  ts->time_step = ts->ptime - ts->ptime_prev;
  ts->ptime = ts->ptime_prev;
  ts->ptime_prev = ts->ptime_prev_rollback;
  ts->steps--;
  ts->steprollback = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TSGetStages - Get the number of stages and stage values

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  ns - the number of stages
-  Y - the current stage vectors

   Level: advanced

   Notes: Both ns and Y can be NULL.

.seealso: TSCreate()
@*/
PetscErrorCode  TSGetStages(TS ts,PetscInt *ns,Vec **Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  if (ns) PetscValidPointer(ns,2);
  if (Y) PetscValidPointer(Y,3);
  if (!ts->ops->getstages) {
    if (ns) *ns = 0;
    if (Y) *Y = NULL;
  } else {
    ierr = (*ts->ops->getstages)(ts,ns,Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  TSComputeIJacobianDefaultColor - Computes the Jacobian using finite differences and coloring to exploit matrix sparsity.

  Collective on SNES

  Input Parameters:
+ ts - the TS context
. t - current timestep
. U - state vector
. Udot - time derivative of state vector
. shift - shift to apply, see note below
- ctx - an optional user context

  Output Parameters:
+ J - Jacobian matrix (not altered in this routine)
- B - newly computed Jacobian matrix to use with preconditioner (generally the same as J)

  Level: intermediate

  Notes:
  If F(t,U,Udot)=0 is the DAE, the required Jacobian is

  dF/dU + shift*dF/dUdot

  Most users should not need to explicitly call this routine, as it
  is used internally within the nonlinear solvers.

  This will first try to get the coloring from the DM.  If the DM type has no coloring
  routine, then it will try to get the coloring from the matrix.  This requires that the
  matrix have nonzero entries precomputed.

.seealso: TSSetIJacobian(), MatFDColoringCreate(), MatFDColoringSetFunction()
@*/
PetscErrorCode TSComputeIJacobianDefaultColor(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal shift,Mat J,Mat B,void *ctx)
{
  SNES           snes;
  MatFDColoring  color;
  PetscBool      hascolor, matcolor = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject)ts)->options,((PetscObject) ts)->prefix, "-ts_fd_color_use_mat", &matcolor, NULL);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) B, "TSMatFDColoring", (PetscObject *) &color);CHKERRQ(ierr);
  if (!color) {
    DM         dm;
    ISColoring iscoloring;

    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMHasColoring(dm, &hascolor);CHKERRQ(ierr);
    if (hascolor && !matcolor) {
      ierr = DMCreateColoring(dm, IS_COLORING_GLOBAL, &iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B, iscoloring, &color);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(color, (PetscErrorCode (*)(void)) SNESTSFormFunction, (void *) ts);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B, iscoloring, color);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    } else {
      MatColoring mc;

      ierr = MatColoringCreate(B, &mc);CHKERRQ(ierr);
      ierr = MatColoringSetDistance(mc, 2);CHKERRQ(ierr);
      ierr = MatColoringSetType(mc, MATCOLORINGSL);CHKERRQ(ierr);
      ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
      ierr = MatColoringApply(mc, &iscoloring);CHKERRQ(ierr);
      ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B, iscoloring, &color);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(color, (PetscErrorCode (*)(void)) SNESTSFormFunction, (void *) ts);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B, iscoloring, color);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    }
    ierr = PetscObjectCompose((PetscObject) B, "TSMatFDColoring", (PetscObject) color);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject) color);CHKERRQ(ierr);
  }
  ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
  ierr = MatFDColoringApply(B, color, U, snes);CHKERRQ(ierr);
  if (J != B) {
    ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    TSSetFunctionDomainError - Set a function that tests if the current state vector is valid

    Input Parameters:
+    ts - the TS context
-    func - function called within TSFunctionDomainError

    Calling sequence of func:
$     PetscErrorCode func(TS ts,PetscReal time,Vec state,PetscBool reject)

+   ts - the TS context
.   time - the current time (of the stage)
.   state - the state to check if it is valid
-   reject - (output parameter) PETSC_FALSE if the state is acceptable, PETSC_TRUE if not acceptable

    Level: intermediate

    Notes:
      If an implicit ODE solver is being used then, in addition to providing this routine, the
      user's code should call SNESSetFunctionDomainError() when domain errors occur during
      function evaluations where the functions are provided by TSSetIFunction() or TSSetRHSFunction().
      Use TSGetSNES() to obtain the SNES object

    Developer Notes:
      The naming of this function is inconsistent with the SNESSetFunctionDomainError()
      since one takes a function pointer and the other does not.

.seealso: TSAdaptCheckStage(), TSFunctionDomainError(), SNESSetFunctionDomainError(), TSGetSNES()
@*/

PetscErrorCode TSSetFunctionDomainError(TS ts, PetscErrorCode (*func)(TS,PetscReal,Vec,PetscBool*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->functiondomainerror = func;
  PetscFunctionReturn(0);
}

/*@
    TSFunctionDomainError - Checks if the current state is valid

    Input Parameters:
+    ts - the TS context
.    stagetime - time of the simulation
-    Y - state vector to check.

    Output Parameter:
.    accept - Set to PETSC_FALSE if the current state vector is valid.

    Note:
    This function is called by the TS integration routines and calls the user provided function (set with TSSetFunctionDomainError())
    to check if the current state is valid.

    Level: developer

.seealso: TSSetFunctionDomainError()
@*/
PetscErrorCode TSFunctionDomainError(TS ts,PetscReal stagetime,Vec Y,PetscBool* accept)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *accept = PETSC_TRUE;
  if (ts->functiondomainerror) {
    PetscStackCallStandard((*ts->functiondomainerror),(ts,stagetime,Y,accept));
  }
  PetscFunctionReturn(0);
}

/*@C
  TSClone - This function clones a time step object.

  Collective

  Input Parameter:
. tsin    - The input TS

  Output Parameter:
. tsout   - The output TS (cloned)

  Notes:
  This function is used to create a clone of a TS object. It is used in ARKIMEX for initializing the slope for first stage explicit methods. It will likely be replaced in the future with a mechanism of switching methods on the fly.

  When using TSDestroy() on a clone the user has to first reset the correct TS reference in the embedded SNES object: e.g.: by running SNES snes_dup=NULL; TSGetSNES(ts,&snes_dup); TSSetSNES(ts,snes_dup);

  Level: developer

.seealso: TSCreate(), TSSetType(), TSSetUp(), TSDestroy(), TSSetProblemType()
@*/
PetscErrorCode  TSClone(TS tsin, TS *tsout)
{
  TS             t;
  PetscErrorCode ierr;
  SNES           snes_start;
  DM             dm;
  TSType         type;

  PetscFunctionBegin;
  PetscValidPointer(tsin,1);
  *tsout = NULL;

  ierr = PetscHeaderCreate(t, TS_CLASSID, "TS", "Time stepping", "TS", PetscObjectComm((PetscObject)tsin), TSDestroy, TSView);CHKERRQ(ierr);

  /* General TS description */
  t->numbermonitors    = 0;
  t->monitorFrequency  = 1;
  t->setupcalled       = 0;
  t->ksp_its           = 0;
  t->snes_its          = 0;
  t->nwork             = 0;
  t->rhsjacobian.time  = PETSC_MIN_REAL;
  t->rhsjacobian.scale = 1.;
  t->ijacobian.shift   = 1.;

  ierr = TSGetSNES(tsin,&snes_start);CHKERRQ(ierr);
  ierr = TSSetSNES(t,snes_start);CHKERRQ(ierr);

  ierr = TSGetDM(tsin,&dm);CHKERRQ(ierr);
  ierr = TSSetDM(t,dm);CHKERRQ(ierr);

  t->adapt = tsin->adapt;
  ierr = PetscObjectReference((PetscObject)t->adapt);CHKERRQ(ierr);

  t->trajectory = tsin->trajectory;
  ierr = PetscObjectReference((PetscObject)t->trajectory);CHKERRQ(ierr);

  t->event = tsin->event;
  if (t->event) t->event->refct++;

  t->problem_type      = tsin->problem_type;
  t->ptime             = tsin->ptime;
  t->ptime_prev        = tsin->ptime_prev;
  t->time_step         = tsin->time_step;
  t->max_time          = tsin->max_time;
  t->steps             = tsin->steps;
  t->max_steps         = tsin->max_steps;
  t->equation_type     = tsin->equation_type;
  t->atol              = tsin->atol;
  t->rtol              = tsin->rtol;
  t->max_snes_failures = tsin->max_snes_failures;
  t->max_reject        = tsin->max_reject;
  t->errorifstepfailed = tsin->errorifstepfailed;

  ierr = TSGetType(tsin,&type);CHKERRQ(ierr);
  ierr = TSSetType(t,type);CHKERRQ(ierr);

  t->vec_sol           = NULL;

  t->cfltime          = tsin->cfltime;
  t->cfltime_local    = tsin->cfltime_local;
  t->exact_final_time = tsin->exact_final_time;

  ierr = PetscMemcpy(t->ops,tsin->ops,sizeof(struct _TSOps));CHKERRQ(ierr);

  if (((PetscObject)tsin)->fortran_func_pointers) {
    PetscInt i;
    ierr = PetscMalloc((10)*sizeof(void(*)(void)),&((PetscObject)t)->fortran_func_pointers);CHKERRQ(ierr);
    for (i=0; i<10; i++) {
      ((PetscObject)t)->fortran_func_pointers[i] = ((PetscObject)tsin)->fortran_func_pointers[i];
    }
  }
  *tsout = t;
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSWrapperFunction_TSRHSJacobianTest(void* ctx,Vec x,Vec y)
{
  PetscErrorCode ierr;
  TS             ts = (TS) ctx;

  PetscFunctionBegin;
  ierr = TSComputeRHSFunction(ts,0,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    TSRHSJacobianTest - Compares the multiply routine provided to the MATSHELL with differencing on the TS given RHS function.

   Logically Collective on TS

    Input Parameters:
    TS - the time stepping routine

   Output Parameter:
.   flg - PETSC_TRUE if the multiply is likely correct

   Options Database:
 .   -ts_rhs_jacobian_test_mult -mat_shell_test_mult_view - run the test at each timestep of the integrator

   Level: advanced

   Notes:
    This only works for problems defined only the RHS function and Jacobian NOT IFunction and IJacobian

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation(), MatShellTestMultTranspose(), TSRHSJacobianTestTranspose()
@*/
PetscErrorCode  TSRHSJacobianTest(TS ts,PetscBool *flg)
{
  Mat            J,B;
  PetscErrorCode ierr;
  TSRHSJacobian  func;
  void*          ctx;

  PetscFunctionBegin;
  ierr = TSGetRHSJacobian(ts,&J,&B,&func,&ctx);CHKERRQ(ierr);
  ierr = (*func)(ts,0.0,ts->vec_sol,J,B,ctx);CHKERRQ(ierr);
  ierr = MatShellTestMult(J,RHSWrapperFunction_TSRHSJacobianTest,ts->vec_sol,ts,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSRHSJacobianTestTranspose - Compares the multiply transpose routine provided to the MATSHELL with differencing on the TS given RHS function.

   Logically Collective on TS

    Input Parameters:
    TS - the time stepping routine

   Output Parameter:
.   flg - PETSC_TRUE if the multiply is likely correct

   Options Database:
.   -ts_rhs_jacobian_test_mult_transpose -mat_shell_test_mult_transpose_view - run the test at each timestep of the integrator

   Notes:
    This only works for problems defined only the RHS function and Jacobian NOT IFunction and IJacobian

   Level: advanced

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation(), MatShellTestMultTranspose(), TSRHSJacobianTest()
@*/
PetscErrorCode  TSRHSJacobianTestTranspose(TS ts,PetscBool *flg)
{
  Mat            J,B;
  PetscErrorCode ierr;
  void           *ctx;
  TSRHSJacobian  func;

  PetscFunctionBegin;
  ierr = TSGetRHSJacobian(ts,&J,&B,&func,&ctx);CHKERRQ(ierr);
  ierr = (*func)(ts,0.0,ts->vec_sol,J,B,ctx);CHKERRQ(ierr);
  ierr = MatShellTestMultTranspose(J,RHSWrapperFunction_TSRHSJacobianTest,ts->vec_sol,ts,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSSetUseSplitRHSFunction - Use the split RHSFunction when a multirate method is used.

  Logically collective

  Input Parameters:
+  ts - timestepping context
-  use_splitrhsfunction - PETSC_TRUE indicates that the split RHSFunction will be used

  Options Database:
.   -ts_use_splitrhsfunction - <true,false>

  Notes:
    This is only useful for multirate methods

  Level: intermediate

.seealso: TSGetUseSplitRHSFunction()
@*/
PetscErrorCode TSSetUseSplitRHSFunction(TS ts, PetscBool use_splitrhsfunction)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->use_splitrhsfunction = use_splitrhsfunction;
  PetscFunctionReturn(0);
}

/*@
  TSGetUseSplitRHSFunction - Gets whether to use the split RHSFunction when a multirate method is used.

  Not collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  use_splitrhsfunction - PETSC_TRUE indicates that the split RHSFunction will be used

  Level: intermediate

.seealso: TSSetUseSplitRHSFunction()
@*/
PetscErrorCode TSGetUseSplitRHSFunction(TS ts, PetscBool *use_splitrhsfunction)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *use_splitrhsfunction = ts->use_splitrhsfunction;
  PetscFunctionReturn(0);
}

/*@
    TSSetMatStructure - sets the relationship between the nonzero structure of the RHS Jacobian matrix to the IJacobian matrix.

   Logically  Collective on ts

   Input Parameters:
+  ts - the time-stepper
-  str - the structure (the default is UNKNOWN_NONZERO_PATTERN)

   Level: intermediate

   Notes:
     When the relationship between the nonzero structures is known and supplied the solution process can be much faster

.seealso: MatAXPY(), MatStructure
 @*/
PetscErrorCode TSSetMatStructure(TS ts,MatStructure str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->axpy_pattern = str;
  PetscFunctionReturn(0);
}
