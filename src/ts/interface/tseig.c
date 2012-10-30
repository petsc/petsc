
#include <petsc-private/tsimpl.h>        /*I "petscts.h"  I*/

/* ------------------------------------------------------------------------*/
struct _n_TSMonitorSPEigCtx {
  PetscDrawSP drawsp;
  KSP         ksp;
  PetscRandom rand;
  PetscInt    howoften;  /* when > 0 uses step % howoften, when negative only final solution plotted */
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
.  -ts_monitor_eig - plot egienvalues of linearized right hand side

   Notes:
   Use TSMonitorSPEigCtxDestroy() to destroy.

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
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(win);CHKERRQ(ierr);
  ierr = PetscDrawSPCreate(win,1,&(*ctx)->drawsp);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&(*ctx)->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions((*ctx)->rand);CHKERRQ(ierr);
  ierr = KSPCreate(comm,&(*ctx)->ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix((*ctx)->ksp,"ts_monitor_eig_");CHKERRQ(ierr); /* this is wrong, used use also prefix from the TS */
  ierr = KSPSetType((*ctx)->ksp,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGMRESSetRestart((*ctx)->ksp,200);CHKERRQ(ierr);
  ierr = KSPSetTolerances((*ctx)->ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,200);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues((*ctx)->ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions((*ctx)->ksp);CHKERRQ(ierr);
  ierr = KSPGetPC((*ctx)->ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  (*ctx)->howoften = howoften;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSPEig"
PetscErrorCode TSMonitorSPEig(TS ts,PetscInt step,PetscReal ptime,Vec v,void *monctx)
{
  TSMonitorSPEigCtx ctx = (TSMonitorSPEigCtx) monctx;
  PetscErrorCode    ierr;
  KSP               ksp = ctx->ksp;
  PetscInt          n,nits,neig,i;
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
    ierr = SNESComputeJacobian(snes,v,&A,&B,&structure);CHKERRQ(ierr);
    /*    ierr = TSComputeIJacobian(ts,ptime,v,xdot,1.0,&A,&B,&structure,PETSC_FALSE);CHKERRQ(ierr);   */
    ierr = MatScale(A,-1.0);CHKERRQ(ierr);
    if (A != B) {
      ierr = MatScale(B,-1.0);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ksp,A,B,structure);CHKERRQ(ierr);
    ierr = VecSetRandom(xdot,ctx->rand);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,xdot,xdot);CHKERRQ(ierr);
    ierr = VecDestroy(&xdot);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&nits);CHKERRQ(ierr);
    n = nits+2;

    if (nits) {
      ierr = PetscMalloc2(n,PetscReal,&r,n,PetscReal,&c);CHKERRQ(ierr);
      ierr = KSPComputeEigenvalues(ksp,n,r,c,&neig);CHKERRQ(ierr);
      for (i=0; i<neig; i++) {
        ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);
      }
      ierr = PetscDrawSPDraw(drawsp);CHKERRQ(ierr);
      ierr = PetscFree2(r,c);CHKERRQ(ierr);
    }
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

