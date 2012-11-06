#define PETSC_DESIRE_COMPLEX
#include <petsc-private/tsimpl.h>        /*I "petscts.h"  I*/

/* ------------------------------------------------------------------------*/
struct _n_TSMonitorSPEigCtx {
  PetscDrawSP drawsp;
  KSP         ksp;
  PetscInt    howoften;  /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscBool   computeexplicitly;
  MPI_Comm    comm;
  PetscRandom rand;
  PetscScalar shift;
  PetscReal   xmin,xmax,ymin,ymax;
};


#undef __FUNCT__
#define __FUNCT__ "TSMonitorSPEigCtxCreate"
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

.keywords: TS, monitor, line graph, residual, seealso

.seealso: TSMonitorSPEigTimeStep(), TSMonitorSet(), TSMonitorLGSolution(), TSMonitorLGError()

@*/
PetscErrorCode  TSMonitorSPEigCtxCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TSMonitorSPEigCtx *ctx)
{
  PetscDraw      win;
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_TSMonitorSPEigCtx,ctx);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&(*ctx)->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions((*ctx)->rand);CHKERRQ(ierr);
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(win);CHKERRQ(ierr);
  ierr = PetscDrawSPCreate(win,1,&(*ctx)->drawsp);CHKERRQ(ierr);
  ierr = KSPCreate(comm,&(*ctx)->ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix((*ctx)->ksp,"ts_monitor_sp_eig_");CHKERRQ(ierr); /* this is wrong, used use also prefix from the TS */
  ierr = KSPSetType((*ctx)->ksp,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGMRESSetRestart((*ctx)->ksp,200);CHKERRQ(ierr);
  ierr = KSPSetTolerances((*ctx)->ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,200);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues((*ctx)->ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions((*ctx)->ksp);CHKERRQ(ierr);
  ierr = KSPGetPC((*ctx)->ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  (*ctx)->howoften          = howoften;
  (*ctx)->computeexplicitly = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-ts_monitor_sp_eig_explicitly",&(*ctx)->computeexplicitly,PETSC_NULL);CHKERRQ(ierr);
  (*ctx)->shift             = 0.0;
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-ts_monitor_sp_eig_shift",&(*ctx)->shift,PETSC_NULL);CHKERRQ(ierr);
  (*ctx)->comm              = comm;
  (*ctx)->xmin = -2.1;
  (*ctx)->xmax = 1.1;
  (*ctx)->ymin = -1.1;
  (*ctx)->ymax = 1.1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSLinearStabilityIndicator"
static PetscErrorCode TSLinearStabilityIndicator(TS  ts, PetscReal xr,PetscReal xi,PetscBool *flg) 
{
  PetscErrorCode ierr;
  PetscReal      yr,yi;

  PetscFunctionBegin;
  ierr = TSComputeLinearStability(ts,xr,xi,&yr,&yi);CHKERRQ(ierr);
  if ((yr*yr + yi*yi) <= 1.0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSPEig"
PetscErrorCode TSMonitorSPEig(TS ts,PetscInt step,PetscReal ptime,Vec v,void *monctx)
{
  TSMonitorSPEigCtx ctx = (TSMonitorSPEigCtx) monctx;
  PetscErrorCode    ierr;
  KSP               ksp = ctx->ksp;
  PetscInt          n,N,nits,neig,i,its = 200;
  PetscReal         *r,*c;
  PetscDrawSP       drawsp = ctx->drawsp;
  MatStructure      structure;
  Mat               A,B;
  Vec               xdot;
  SNES              snes;

  PetscFunctionBegin;
  if (!step) PetscFunctionReturn(0);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && (step == -1))){
    ierr = VecDuplicate(v,&xdot);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,&A,&B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B);CHKERRQ(ierr);
    /* 
       This doesn't work because methods keep and use internal information about the shift so it 
       seems we would need code for each method to trick the correct Jacobian in being computed.
    ts->time_step = PETSC_MAX_REAL;
    ierr = SNESComputeJacobian(snes,v,&A,&B,&structure);CHKERRQ(ierr); 
    */
    ierr = TSComputeIJacobian(ts,ptime,v,xdot,0.0,&B,&B,&structure,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatScale(B,-1.0*ts->time_step);CHKERRQ(ierr);
    ierr = MatShift(B,ctx->shift);CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp,B,B,structure);CHKERRQ(ierr);
    ierr = VecGetSize(v,&n);CHKERRQ(ierr);
    if (n < 200) its = n;
    ierr = KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,its);CHKERRQ(ierr);
    ierr = VecSetRandom(xdot,ctx->rand);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,xdot,xdot);CHKERRQ(ierr);
    ierr = VecDestroy(&xdot);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&nits);CHKERRQ(ierr);
    N = nits+2;

    if (nits) {
      PetscDraw     draw;
      PetscReal     pause;
      PetscDrawAxis axis;
      PetscReal     xmin,xmax,ymin,ymax;

      ierr = PetscDrawSPReset(drawsp);CHKERRQ(ierr);
      ierr = PetscDrawSPSetLimits(drawsp,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax);CHKERRQ(ierr);
      ierr = PetscMalloc2(PetscMax(n,N),PetscReal,&r,PetscMax(n,N),PetscReal,&c);CHKERRQ(ierr);
      if (ctx->computeexplicitly) {
        ierr = KSPComputeEigenvaluesExplicitly(ksp,n,r,c);CHKERRQ(ierr);
        neig = n;
      } else {
        ierr = KSPComputeEigenvalues(ksp,N,r,c,&neig);CHKERRQ(ierr);
      }
      for (i=0; i<neig; i++) {
        r[i] -= PetscAbsScalar(ctx->shift);
        if ((r[i]*r[i] + c[i]*c[i]) > 1.0) {
           ierr = PetscPrintf(ctx->comm,"Linearized Eigenvalue %g + %g i norm indicates unstable scheme \n",r[i],c[i]);CHKERRQ(ierr);
        }
        ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);
      }
      ierr = PetscFree2(r,c);CHKERRQ(ierr);
      ierr = PetscDrawSPGetDraw(drawsp,&draw);CHKERRQ(ierr);
      ierr = PetscDrawGetPause(draw,&pause);CHKERRQ(ierr);
      ierr = PetscDrawSetPause(draw,0.0);CHKERRQ(ierr);
      ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscDrawSetPause(draw,pause);CHKERRQ(ierr);

      if (ts->ops->linearstability) {
        ierr = PetscDrawSPGetAxis(drawsp,&axis);CHKERRQ(ierr);
        ierr = PetscDrawAxisGetLimits(axis,&xmin,&xmax,&ymin,&ymax);CHKERRQ(ierr);
        ierr = PetscDrawIndicatorFunction(draw,xmin,xmax,ymin,ymax,PETSC_DRAW_CYAN,(PetscErrorCode (*)(void*,PetscReal,PetscReal,PetscBool*))TSLinearStabilityIndicator,ts);CHKERRQ(ierr);
        ierr = PetscDrawSPDraw(drawsp,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSPEigCtxDestroy"
/*@C
   TSMonitorSPEigCtxDestroy - Destroys a scatter plot context that was created with TSMonitorSPEigCtxCreate().

   Collective on TSMonitorSPEigCtx

   Input Parameter:
.  ctx - the monitor context

   Level: intermediate

.keywords: TS, monitor, line graph, destroy

.seealso: TSMonitorSPEigCtxCreate(),  TSMonitorSet(), TSMonitorSPEig();
@*/
PetscErrorCode  TSMonitorSPEigCtxDestroy(TSMonitorSPEigCtx *ctx)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSPGetDraw((*ctx)->drawsp,&draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscDrawSPDestroy(&(*ctx)->drawsp);CHKERRQ(ierr);
  ierr = KSPDestroy(&(*ctx)->ksp);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*ctx)->rand);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



