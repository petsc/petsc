
static char help[] = "Solves PDE optimization problem of ex22.c with AD for adjoint.\n\n";

#include "petscda.h"
#include "petscpf.h"
#include "petscmg.h"
#include "petscsnes.h"
#include "petscdmmg.h"

/*

              Minimize F(w,u) such that G(w,u) = 0

         L(w,u,lambda) = F(w,u) + lambda^T G(w,u)

       w - design variables (what we change to get an optimal solution)
       u - state variables (i.e. the PDE solution)
       lambda - the Lagrange multipliers

            U = (w u lambda)

       fu, fw, flambda contain the gradient of L(w,u,lambda)

            FU = (fw fu flambda)

       In this example the PDE is 
                             Uxx - u^2 = 2, 
                            u(0) = w(0), thus this is the free parameter
                            u(1) = 0
       the function we wish to minimize is 
                            \integral u^{2}

       The exact solution for u is given by u(x) = x*x - 1.25*x + .25

       Use the usual centered finite differences.

       Note we treat the problem as non-linear though it happens to be linear

       The lambda and u are NOT interlaced.

          We optionally provide a preconditioner on each level from the operator

              (1   0   0)
              (0   J   0)
              (0   0   J')

  
*/


extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode PDEFormFunctionLocal(DALocalInfo*,PetscScalar*,PetscScalar*,PassiveScalar*);

typedef struct {
  Mat        J;           /* Jacobian of PDE system */
  KSP       ksp;        /* Solver for that Jacobian */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "myPCApply"
PetscErrorCode myPCApply(PC pc,Vec x,Vec y)
{
  Vec            xu,xlambda,yu,ylambda;
  PetscScalar    *xw,*yw;
  PetscErrorCode ierr;
  DMMG           dmmg;
  DMComposite    packer;
  AppCtx         *appctx;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&dmmg);CHKERRQ(ierr);
  packer = (DMComposite)dmmg->dm;
  appctx = (AppCtx*)dmmg->user;
  ierr = DMCompositeGetAccess(packer,x,&xw,&xu,&xlambda);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,y,&yw,&yu,&ylambda);CHKERRQ(ierr);
  if (yw && xw) {
    yw[0] = xw[0];
  }
  ierr = KSPSolve(appctx->ksp,xu,yu);CHKERRQ(ierr);

  ierr = KSPSolveTranspose(appctx->ksp,xlambda,ylambda);CHKERRQ(ierr);
  /*  ierr = VecCopy(xu,yu);CHKERRQ(ierr);
      ierr = VecCopy(xlambda,ylambda);CHKERRQ(ierr); */
  ierr = DMCompositeRestoreAccess(packer,x,&xw,&xu,&xlambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(packer,y,&yw,&yu,&ylambda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "myPCView"
PetscErrorCode myPCView(PC pc,PetscViewer v)
{
  PetscErrorCode ierr;
  DMMG           dmmg;
  AppCtx         *appctx;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&dmmg);CHKERRQ(ierr);
  appctx = (AppCtx*)dmmg->user;
  ierr = KSPView(appctx->ksp,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       nlevels,i,j;
  DA             da;
  DMMG           *dmmg;
  DMComposite        packer;
  AppCtx         *appctx;
  ISColoring     iscoloring;
  PetscTruth     bdp;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  /* Hardwire several options; can be changed at command line */
  ierr = PetscOptionsSetValue("-dmmg_grid_sequence",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-ksp_type","fgmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-ksp_max_it","5");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_mg_type","full");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_coarse_ksp_type","gmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_levels_ksp_type","gmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_coarse_ksp_max_it","6");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_levels_ksp_max_it","3");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mat_mffd_type","wp");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mat_mffd_compute_normu","no");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_ls","basic");CHKERRQ(ierr); 
  ierr = PetscOptionsSetValue("-dmmg_jacobian_mf_fd",0);CHKERRQ(ierr); 
  /* ierr = PetscOptionsSetValue("-snes_ls","basicnonorms");CHKERRQ(ierr); */
  ierr = PetscOptionsInsert(&argc,&argv,PETSC_NULL);CHKERRQ(ierr);   

  /* create DMComposite object to manage composite vector */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = DMCompositeAddArray(packer,0,1);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,-5,1,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,(DM)da);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /* create nonlinear multi-level solver */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)packer);CHKERRQ(ierr);
  ierr = DMCompositeDestroy(packer);CHKERRQ(ierr);

  /* Create Jacobian of PDE function for each level */
  nlevels = DMMGGetLevels(dmmg);
  for (i=0; i<nlevels; i++) {
    packer = (DMComposite)dmmg[i]->dm;
    ierr   = DMCompositeGetEntries(packer,PETSC_NULL,&da,PETSC_NULL);CHKERRQ(ierr);
    ierr   = PetscNew(AppCtx,&appctx);CHKERRQ(ierr);
    ierr   = DAGetColoring(da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);
    ierr   = DAGetMatrix(da,MATAIJ,&appctx->J);CHKERRQ(ierr);
    ierr   = MatSetColoring(appctx->J,iscoloring);CHKERRQ(ierr);
    ierr   = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    ierr   = DASetLocalFunction(da,(DALocalFunction1)PDEFormFunctionLocal);CHKERRQ(ierr);
    ierr   = DASetLocalAdicFunction(da,ad_PDEFormFunctionLocal);CHKERRQ(ierr);
    dmmg[i]->user = (void*)appctx;
  }

  ierr = DMMGSetSNES(dmmg,FormFunction,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-bdp",&bdp);CHKERRQ(ierr);
  if (bdp) {
    for (i=0; i<nlevels; i++) {
      KSP  ksp;
      PC   pc,mpc;

      appctx = (AppCtx*) dmmg[i]->user;
      ierr   = KSPCreate(PETSC_COMM_WORLD,&appctx->ksp);CHKERRQ(ierr);
      ierr   = KSPSetOptionsPrefix(appctx->ksp,"bdp_");CHKERRQ(ierr);
      ierr   = KSPSetFromOptions(appctx->ksp);CHKERRQ(ierr);

      ierr = SNESGetKSP(dmmg[i]->snes,&ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      for (j=0; j<=i; j++) {
	ierr = PCMGGetSmoother(pc,j,&ksp);CHKERRQ(ierr);
	ierr = KSPGetPC(ksp,&mpc);CHKERRQ(ierr);
	ierr = PCSetType(mpc,PCSHELL);CHKERRQ(ierr);
	ierr = PCShellSetContext(mpc,dmmg[j]);CHKERRQ(ierr);
	ierr = PCShellSetApply(mpc,myPCApply);CHKERRQ(ierr);
	ierr = PCShellSetView(mpc,myPCView);CHKERRQ(ierr);
      }
    }
  }

  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  for (i=0; i<nlevels; i++) {
    appctx = (AppCtx*)dmmg[i]->user;
    ierr   = MatDestroy(appctx->J);CHKERRQ(ierr);
    if (appctx->ksp) {ierr = KSPDestroy(appctx->ksp);CHKERRQ(ierr);}
    ierr   = PetscFree(appctx);CHKERRQ(ierr);  
  }
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
/*
     Enforces the PDE on the grid
     This local function acts on the ghosted version of U (accessed via DAGetLocalVector())
     BUT the global, nonghosted version of FU

     Process adiC(36): PDEFormFunctionLocal
*/
#undef __FUNCT__
#define __FUNCT__ "PDEFormFunctionLocal"
PetscErrorCode PDEFormFunctionLocal(DALocalInfo *info,PetscScalar *u,PetscScalar *fu,PassiveScalar *w)
{
  PetscInt       xs = info->xs,xm = info->xm,i,mx = info->mx;
  PetscScalar    d,h;
  PetscErrorCode ierr;

  d    = mx-1.0;
  h    = 1.0/d;

  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)    fu[i]   = 2.0*d*(u[i] - w[0]) + h*u[i]*u[i];
    else if (i == mx-1) fu[i]   = 2.0*d*u[i] + h*u[i]*u[i];
    else                fu[i]   = -(d*(u[i+1] - 2.0*u[i] + u[i-1]) - 2.0*h) + h*u[i]*u[i];
  } 

  ierr = PetscLogFlops(9.0*mx);CHKERRQ(ierr);
  return 0;
}

/*
      Evaluates FU = Gradiant(L(w,u,lambda))

      This is the function that is usually passed to the SNESSetJacobian() or DMMGSetSNES() and
    defines the nonlinear set of equations that are to be solved.

     This local function acts on the ghosted version of U (accessed via DMCompositeGetLocalVectors() and
   DMCompositeScatter()) BUT the global, nonghosted version of FU (via DMCompositeAccess()).

     This function uses PDEFormFunction() to enforce the PDE constraint equations and its adjoint
   for the Lagrange multiplier equations

*/
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  DMMG           dmmg = (DMMG)dummy;
  PetscErrorCode ierr;
  PetscInt       xs,xm,i,N,nredundant;
  PetscScalar    *u,*w,*fw,*fu,*lambda,*flambda,d,h,h2;
  Vec            vu,vlambda,vfu,vflambda,vglambda;
  DA             da;
  DMComposite        packer = (DMComposite)dmmg->dm;
  PetscTruth     useadic = PETSC_TRUE;
#if defined(PETSC_HAVE_ADIC)
  AppCtx         *appctx = (AppCtx*)dmmg->user;
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_ADIC)
  ierr = PetscOptionsHasName(0,"-useadic",&skipadic);CHKERRQ(ierr);
#endif

  ierr = DMCompositeGetEntries(packer,&nredundant,&da,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  d    = (N-1.0);
  h    = 1.0/d;
  h2   = 2.0*h;

  ierr = DMCompositeGetLocalVectors(packer,&w,&vu,&vlambda);CHKERRQ(ierr);
  ierr = DMCompositeScatter(packer,U,w,vu,vlambda);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,FU,&fw,&vfu,&vflambda);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,U,0,0,&vglambda);CHKERRQ(ierr);

  /* G() */
  ierr = DAFormFunction1(da,vu,vfu,w);CHKERRQ(ierr);

#if defined(PETSC_HAVE_ADIC)
  if (useadic) { 
    /* lambda^T G_u() */
    ierr = DAComputeJacobian1WithAdic(da,vu,appctx->J,w);CHKERRQ(ierr);  
    if (appctx->ksp) {
      ierr = KSPSetOperators(appctx->ksp,appctx->J,appctx->J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = MatMultTranspose(appctx->J,vglambda,vflambda);CHKERRQ(ierr); 
  }
#endif

  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu,&fu);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vlambda,&lambda);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vflambda,&flambda);CHKERRQ(ierr);

  /* L_w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -2.*d*lambda[0];
  }

  /* lambda^T G_u() */
  if (!useadic) {
    for (i=xs; i<xs+xm; i++) {
      if      (i == 0)   flambda[0]   = 2.*d*lambda[0]   - d*lambda[1] + h2*lambda[0]*u[0];
      else if (i == 1)   flambda[1]   = 2.*d*lambda[1]   - d*lambda[2] + h2*lambda[1]*u[1];
      else if (i == N-1) flambda[N-1] = 2.*d*lambda[N-1] - d*lambda[N-2] + h2*lambda[N-1]*u[N-1];
      else if (i == N-2) flambda[N-2] = 2.*d*lambda[N-2] - d*lambda[N-3] + h2*lambda[N-2]*u[N-2];
      else               flambda[i]   = - d*(lambda[i+1] - 2.0*lambda[i] + lambda[i-1]) + h2*lambda[i]*u[i];
    }  
  }

  /* F_u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   flambda[0]   +=    h*u[0];
    else if (i == 1)   flambda[1]   +=    h2*u[1];
    else if (i == N-1) flambda[N-1] +=    h*u[N-1];
    else if (i == N-2) flambda[N-2] +=    h2*u[N-2];
    else               flambda[i]   +=    h2*u[i];
  } 

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu,&fu);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vlambda,&lambda);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vflambda,&flambda);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(packer,&w,&vu,&vlambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(packer,FU,&fw,&vfu,&vflambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(packer,U,0,0,&vglambda);CHKERRQ(ierr);

  ierr = PetscLogFlops(9.0*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






