/*$Id: ex24.c,v 1.19 2001/04/30 03:52:57 bsmith Exp bsmith $*/

static char help[] = "Solves PDE optimization problem of ex22.c with finite differences for adjoint.\n\n";

#include "petscda.h"
#include "petscpf.h"
#include "petscsnes.h"

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
                             Uxx = 2, 
                            u(0) = w(0), thus this is the free parameter
                            u(1) = 0
       the function we wish to minimize is 
                            \integral u^{2}

       The exact solution for u is given by u(x) = x*x - 1.25*x + .25

       Use the usual centered finite differences.

       Note we treat the problem as non-linear though it happens to be linear

       The lambda and u are NOT interlaced.
*/

typedef int (*DALocalFunction1)(DALocalInfo*,void*,void*,void*);
extern int FormFunction(SNES,Vec,Vec,void*);
extern int PDEFormFunction(DA,int (*)(DALocalInfo*,void*,void*,void*),Vec,Vec,void*);
extern int PDEFormJacobian(DA,ISColoring,int (*)(DALocalInfo*,void*,void*,void*),Vec,Mat,void*);

typedef struct {
  ISColoring iscoloring;  /* coloring of grid used for computing Jacobian of PDE */
  Mat        J;         /* Jacobian of PDE system */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int        ierr,nlevels,i;
  DA         da;
  DMMG       *dmmg;
  VecPack    packer;
  AppCtx     *appctx;

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
  ierr = PetscOptionsSetValue("-snes_mf_type","wp");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_mf_compute_norma","no");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_mf_compute_normu","no");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_eq_ls","basic");CHKERRQ(ierr);
  /* ierr = PetscOptionsSetValue("-snes_eq_ls","basicnonorms");CHKERRQ(ierr); */
  ierr = PetscOptionsInsert(&argc,&argv,PETSC_NULL);CHKERRQ(ierr);   
  /* Create a global vector from a da arrays */
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,-5,1,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = VecPackCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = VecPackAddArray(packer,1);CHKERRQ(ierr);
  ierr = VecPackAddDA(packer,da);CHKERRQ(ierr);
  ierr = VecPackAddDA(packer,da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /* create nonlinear multi-level solver */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetUseMatrixFree(dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)packer);CHKERRQ(ierr);
  ierr = VecPackDestroy(packer);CHKERRQ(ierr);

  /* Create Jacobian of PDE function for each level */
  nlevels = DMMGGetLevels(dmmg);
  for (i=0; i<nlevels; i++) {
    packer = (VecPack)dmmg[i]->dm;
    ierr   = VecPackGetEntries(packer,PETSC_NULL,&da,PETSC_NULL);CHKERRQ(ierr);
    ierr   = PetscNew(AppCtx,&appctx);CHKERRQ(ierr);
    ierr   = DAGetColoring(da,IS_COLORING_GHOSTED,MATMPIAIJ,&appctx->iscoloring,&appctx->J);CHKERRQ(ierr);
    ierr   = MatSetColoring(appctx->J,appctx->iscoloring);CHKERRQ(ierr);
    dmmg[i]->user = (void*)appctx;
  }

  ierr = DMMGSetSNES(dmmg,FormFunction,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  /* ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr); */
  for (i=0; i<nlevels; i++) {
    appctx = (AppCtx*)dmmg[i]->user;
    ierr   = MatDestroy(appctx->J);CHKERRQ(ierr);
    ierr   = ISColoringDestroy(appctx->iscoloring);CHKERRQ(ierr);
  }
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
/*
     Enforces the PDE on the grid
     This local function acts on the ghosted version of U (accessed via DAGetLocalVector())
     BUT the global, nonghosted version of FU

     Process adiC: PDEFormFunctionLocal
*/
#undef __FUNCT__
#define __FUNCT__ "PDEFormFunctionLocal"
int PDEFormFunctionLocal(DALocalInfo *info,Scalar *u,Scalar *fu,PassiveScalar *w)
{
  int     ierr,xs = info->xs,xm = info->xm,i,mx = info->mx;
  Scalar  d,h;

  d    = mx-1.0;
  h    = 1.0/d;

  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)    fu[i]   = 2.0*d*(u[i] - w[0]) + h*u[i]*u[i];
    else if (i == mx-1) fu[i]   = 2.0*d*u[i] + h*u[i]*u[i];
    else                fu[i]   = -(d*(u[i+1] - 2.0*u[i] + u[i-1]) - 2.0*h) + h*u[i]*u[i];
  } 

  PetscLogFlops(9*mx);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PDEFormFunction"
int PDEFormFunction(DA da,int (*lf)(DALocalInfo*,void*,void*,void*),Vec vu,Vec vfu,void *w)
{
  int         ierr;
  void        *u,*fu;
  DALocalInfo info;

  PetscFunctionBegin;

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,(void**)&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu,(void**)&fu);CHKERRQ(ierr);

  ierr = (*lf)(&info,u,fu,w);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da,vu,(void**)&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu,(void**)&fu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ADIC)

/* #include "adic_utils.h" */

#undef __FUNCT__
#define __FUNCT__ "PDEFormJacobian"
int PDEFormJacobian(DA da,ISColoring iscoloring,int (*lf)(DALocalInfo*,void*,void*,void*),Vec vu,Mat J,void *w)
{
  int         ierr,gtdof,tdof;
  Scalar      *u,*ustart;
  DALocalInfo info;
  void        *ad_u,*ad_f,*ad_ustart,*ad_fstart;

  PetscFunctionBegin;

  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* get space for derivative objects.  */
  ierr = DAGetADArray(da,PETSC_TRUE,(void **)&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DAGetADArray(da,PETSC_FALSE,(void **)&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  ierr = VecGetArray(vu,&ustart);CHKERRQ(ierr);
  my_AD_SetValArray(((DERIV_TYPE*)ad_ustart),gtdof,ustart);
  ierr = VecRestoreArray(vu,&ustart);CHKERRQ(ierr);

  my_AD_ResetIndep();
  my_AD_SetIndepArrayColored(ad_ustart,gtdof,iscoloring->colors);
  my_AD_IncrementTotalGradSize(iscoloring->n);
  my_AD_SetIndepDone();

  ierr = (*lf)(&info,ad_u,ad_f,w);CHKERRQ(ierr);

  /* stick the values into the matrix */
  ierr = MatSetValuesAD(J,(Scalar**)ad_fstart);CHKERRQ(ierr);

  /* return space for derivative objects.  */
  ierr = DARestoreADArray(da,PETSC_TRUE,(void **)&ad_u,&ad_ustart,&gtdof);CHKERRQ(ierr);
  ierr = DARestoreADArray(da,PETSC_FALSE,(void **)&ad_f,&ad_fstart,&tdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

/*
      Evaluates FU = Gradiant(L(w,u,lambda))

     This local function acts on the ghosted version of U (accessed via VecPackGetLocalVectors() and
   VecPackScatter()) BUT the global, nonghosted version of FU (via VecPackAccess()).

     This function uses PDEFormFunction() to enforce the PDE constraint equations and its adjoint
   for the Lagrange multiplier equations

*/
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
int FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  DMMG    dmmg = (DMMG)dummy;
  int     ierr,xs,xm,i,N,nredundant;
  Scalar  *u,*w,*fw,*fu,*lambda,*flambda,d,h,h2;
  Vec     vu,vlambda,vfu,vflambda,vglambda;
  DA      da,dadummy;
  VecPack packer = (VecPack)dmmg->dm;
  AppCtx  *appctx = (AppCtx*)dmmg->user;
  PetscTruth skipadic;

  PetscFunctionBegin;
  ierr = VecPackGetEntries(packer,&nredundant,&da,&dadummy);CHKERRQ(ierr);
  ierr = VecPackGetLocalVectors(packer,&w,&vu,&vlambda);CHKERRQ(ierr);
  ierr = VecPackScatter(packer,U,w,vu,vlambda);CHKERRQ(ierr);
  ierr = VecPackGetAccess(packer,FU,&fw,&vfu,&vflambda);CHKERRQ(ierr);
  ierr = VecPackGetAccess(packer,U,0,0,&vglambda);CHKERRQ(ierr);

  /* Evaluate the Jacobian of PDEFormFunction() */
  ierr = PDEFormJacobian(da,appctx->iscoloring,(int (*)(DALocalInfo*,void*,void*,void*))ad_PDEFormFunctionLocal,vu,appctx->J,w);CHKERRQ(ierr);
  ierr = MatMultTranspose(appctx->J,vglambda,vflambda);CHKERRQ(ierr); 

  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB);
  /*  ierr = MatView((Mat)dmmg->user,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* derivative of constraint portion of L() w.r.t. u */
  ierr = PDEFormFunction(da,(int (*)(DALocalInfo*,void*,void*,void*))PDEFormFunctionLocal,vu,vfu,w);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,(void**)&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu,(void**)&fu);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vlambda,(void**)&lambda);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vflambda,(void**)&flambda);CHKERRQ(ierr);
  d    = (N-1.0);
  h    = 1.0/d;
  h2   = 2.0*h;

  /* derivative of L() w.r.t. w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -2.*d*lambda[0];
  }

  ierr = PetscOptionsHasName(0,"-skipadic",&skipadic);CHKERRQ(ierr);
  if (skipadic) {
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   flambda[0]   = 2.*d*lambda[0]   - d*lambda[1] + h2*lambda[0]*u[0];
    else if (i == 1)   flambda[1]   = 2.*d*lambda[1]   - d*lambda[2] + h2*lambda[1]*u[1];
    else if (i == N-1) flambda[N-1] = 2.*d*lambda[N-1] - d*lambda[N-2] + h2*lambda[N-1]*u[N-1];
    else if (i == N-2) flambda[N-2] = 2.*d*lambda[N-2] - d*lambda[N-3] + h2*lambda[N-2]*u[N-2];
    else               flambda[i]   = - d*(lambda[i+1] - 2.0*lambda[i] + lambda[i-1]) + h2*lambda[i]*u[i];
  }  
  }

  /* derivative of function part of L() w.r.t. u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   flambda[0]   +=    h*u[0];
    else if (i == 1)   flambda[1]   +=    h2*u[1];
    else if (i == N-1) flambda[N-1] +=    h*u[N-1];
    else if (i == N-2) flambda[N-2] +=    h2*u[N-2];
    else               flambda[i]   +=    h2*u[i];
  } 

  ierr = DAVecRestoreArray(da,vu,(void**)&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu,(void**)&fu);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vlambda,(void**)&lambda);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vflambda,(void**)&flambda);CHKERRQ(ierr);


  ierr = VecPackRestoreLocalVectors(packer,&w,&vu,&vlambda);CHKERRQ(ierr);
  ierr = VecPackRestoreAccess(packer,FU,&fw,&vfu,&vflambda);CHKERRQ(ierr);
  ierr = VecPackRestoreAccess(packer,U,0,0,&vglambda);CHKERRQ(ierr);

  PetscLogFlops(9*N);
  PetscFunctionReturn(0);
}






