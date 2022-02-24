
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscdraw.h>

/* ------------------------------------------------------------------------*/
struct _n_TSMonitorSPEigCtx {
  PetscDrawSP drawsp;
  KSP         ksp;
  PetscInt    howoften;  /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscBool   computeexplicitly;
  MPI_Comm    comm;
  PetscRandom rand;
  PetscReal   xmin,xmax,ymin,ymax;
};

/*@C
   TSMonitorSPEigCtxCreate - Creates a context for use with TS to monitor the eigenvalues of the linearized operator

   Collective on TS

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of the window
.  m, n - the screen width and height in pixels
-  howoften - if positive then determines the frequency of the plotting, if -1 then only at the final time

   Output Parameter:
.  ctx - the context

   Options Database Key:
.  -ts_monitor_sp_eig - plot egienvalues of linearized right hand side

   Notes:
   Use TSMonitorSPEigCtxDestroy() to destroy.

   Currently only works if the Jacobian is provided explicitly.

   Currently only works for ODEs u_t - F(t,u) = 0; that is with no mass matrix.

   Level: intermediate

.seealso: TSMonitorSPEigTimeStep(), TSMonitorSet(), TSMonitorLGSolution(), TSMonitorLGError()

@*/
PetscErrorCode  TSMonitorSPEigCtxCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TSMonitorSPEigCtx *ctx)
{
  PetscDraw      win;
  PC             pc;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(ctx));
  CHKERRQ(PetscRandomCreate(comm,&(*ctx)->rand));
  CHKERRQ(PetscRandomSetFromOptions((*ctx)->rand));
  CHKERRQ(PetscDrawCreate(comm,host,label,x,y,m,n,&win));
  CHKERRQ(PetscDrawSetFromOptions(win));
  CHKERRQ(PetscDrawSPCreate(win,1,&(*ctx)->drawsp));
  CHKERRQ(KSPCreate(comm,&(*ctx)->ksp));
  CHKERRQ(KSPSetOptionsPrefix((*ctx)->ksp,"ts_monitor_sp_eig_")); /* this is wrong, used use also prefix from the TS */
  CHKERRQ(KSPSetType((*ctx)->ksp,KSPGMRES));
  CHKERRQ(KSPGMRESSetRestart((*ctx)->ksp,200));
  CHKERRQ(KSPSetTolerances((*ctx)->ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,200));
  CHKERRQ(KSPSetComputeSingularValues((*ctx)->ksp,PETSC_TRUE));
  CHKERRQ(KSPSetFromOptions((*ctx)->ksp));
  CHKERRQ(KSPGetPC((*ctx)->ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));

  (*ctx)->howoften          = howoften;
  (*ctx)->computeexplicitly = PETSC_FALSE;

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ts_monitor_sp_eig_explicitly",&(*ctx)->computeexplicitly,NULL));

  (*ctx)->comm = comm;
  (*ctx)->xmin = -2.1;
  (*ctx)->xmax = 1.1;
  (*ctx)->ymin = -1.1;
  (*ctx)->ymax = 1.1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLinearStabilityIndicator(TS ts, PetscReal xr,PetscReal xi,PetscBool *flg)
{
  PetscReal      yr,yi;

  PetscFunctionBegin;
  CHKERRQ(TSComputeLinearStability(ts,xr,xi,&yr,&yi));
  if ((yr*yr + yi*yi) <= 1.0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorSPEig(TS ts,PetscInt step,PetscReal ptime,Vec v,void *monctx)
{
  TSMonitorSPEigCtx ctx = (TSMonitorSPEigCtx) monctx;
  KSP               ksp = ctx->ksp;
  PetscInt          n,N,nits,neig,i,its = 200;
  PetscReal         *r,*c,time_step_save;
  PetscDrawSP       drawsp = ctx->drawsp;
  Mat               A,B;
  Vec               xdot;
  SNES              snes;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!step) PetscFunctionReturn(0);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason)) {
    CHKERRQ(VecDuplicate(v,&xdot));
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESGetJacobian(snes,&A,&B,NULL,NULL));
    CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B));
    /*
       This doesn't work because methods keep and use internal information about the shift so it
       seems we would need code for each method to trick the correct Jacobian in being computed.
     */
    time_step_save = ts->time_step;
    ts->time_step  = PETSC_MAX_REAL;

    CHKERRQ(SNESComputeJacobian(snes,v,A,B));

    ts->time_step  = time_step_save;

    CHKERRQ(KSPSetOperators(ksp,B,B));
    CHKERRQ(VecGetSize(v,&n));
    if (n < 200) its = n;
    CHKERRQ(KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,its));
    CHKERRQ(VecSetRandom(xdot,ctx->rand));
    CHKERRQ(KSPSolve(ksp,xdot,xdot));
    CHKERRQ(VecDestroy(&xdot));
    CHKERRQ(KSPGetIterationNumber(ksp,&nits));
    N    = nits+2;

    if (nits) {
      PetscDraw     draw;
      PetscReal     pause;
      PetscDrawAxis axis;
      PetscReal     xmin,xmax,ymin,ymax;

      CHKERRQ(PetscDrawSPReset(drawsp));
      CHKERRQ(PetscDrawSPSetLimits(drawsp,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax));
      CHKERRQ(PetscMalloc2(PetscMax(n,N),&r,PetscMax(n,N),&c));
      if (ctx->computeexplicitly) {
        CHKERRQ(KSPComputeEigenvaluesExplicitly(ksp,n,r,c));
        neig = n;
      } else {
        CHKERRQ(KSPComputeEigenvalues(ksp,N,r,c,&neig));
      }
      /* We used the positive operator to be able to reuse KSPs that require positive definiteness, now flip the spectrum as is conventional for ODEs */
      for (i=0; i<neig; i++) r[i] = -r[i];
      for (i=0; i<neig; i++) {
        if (ts->ops->linearstability) {
          PetscReal fr,fi;
          CHKERRQ(TSComputeLinearStability(ts,r[i],c[i],&fr,&fi));
          if ((fr*fr + fi*fi) > 1.0) {
            CHKERRQ(PetscPrintf(ctx->comm,"Linearized Eigenvalue %g + %g i linear stability function %g norm indicates unstable scheme \n",(double)r[i],(double)c[i],(double)(fr*fr + fi*fi)));
          }
        }
        CHKERRQ(PetscDrawSPAddPoint(drawsp,r+i,c+i));
      }
      CHKERRQ(PetscFree2(r,c));
      CHKERRQ(PetscDrawSPGetDraw(drawsp,&draw));
      CHKERRQ(PetscDrawGetPause(draw,&pause));
      CHKERRQ(PetscDrawSetPause(draw,0.0));
      CHKERRQ(PetscDrawSPDraw(drawsp,PETSC_TRUE));
      CHKERRQ(PetscDrawSetPause(draw,pause));
      if (ts->ops->linearstability) {
        CHKERRQ(PetscDrawSPGetAxis(drawsp,&axis));
        CHKERRQ(PetscDrawAxisGetLimits(axis,&xmin,&xmax,&ymin,&ymax));
        CHKERRQ(PetscDrawIndicatorFunction(draw,xmin,xmax,ymin,ymax,PETSC_DRAW_CYAN,(PetscErrorCode (*)(void*,PetscReal,PetscReal,PetscBool*))TSLinearStabilityIndicator,ts));
        CHKERRQ(PetscDrawSPDraw(drawsp,PETSC_FALSE));
      }
      CHKERRQ(PetscDrawSPSave(drawsp));
    }
    CHKERRQ(MatDestroy(&B));
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSPEigCtxDestroy - Destroys a scatter plot context that was created with TSMonitorSPEigCtxCreate().

   Collective on TSMonitorSPEigCtx

   Input Parameter:
.  ctx - the monitor context

   Level: intermediate

.seealso: TSMonitorSPEigCtxCreate(),  TSMonitorSet(), TSMonitorSPEig();
@*/
PetscErrorCode  TSMonitorSPEigCtxDestroy(TSMonitorSPEigCtx *ctx)
{
  PetscDraw      draw;

  PetscFunctionBegin;
  CHKERRQ(PetscDrawSPGetDraw((*ctx)->drawsp,&draw));
  CHKERRQ(PetscDrawDestroy(&draw));
  CHKERRQ(PetscDrawSPDestroy(&(*ctx)->drawsp));
  CHKERRQ(KSPDestroy(&(*ctx)->ksp));
  CHKERRQ(PetscRandomDestroy(&(*ctx)->rand));
  CHKERRQ(PetscFree(*ctx));
  PetscFunctionReturn(0);
}
