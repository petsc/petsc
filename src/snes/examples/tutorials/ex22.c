
static char help[] = "Solves PDE optimization problem.\n\n";

#include "petscda.h"
#include "petscpf.h"
#include "petscsnes.h"
#include "petscdmmg.h"

/*

       w - design variables (what we change to get an optimal solution)
       u - state variables (i.e. the PDE solution)
       lambda - the Lagrange multipliers

            U = (w [u_0 lambda_0 u_1 lambda_1 .....])

       fu, fw, flambda contain the gradient of L(w,u,lambda)

            FU = (fw [fu_0 flambda_0 .....])

       In this example the PDE is 
                             Uxx = 2, 
                            u(0) = w(0), thus this is the free parameter
                            u(1) = 0
       the function we wish to minimize is 
                            \integral u^{2}

       The exact solution for u is given by u(x) = x*x - 1.25*x + .25

       Use the usual centered finite differences.

       Note we treat the problem as non-linear though it happens to be linear

       See ex21.c for the same code, but that does NOT interlaces the u and the lambda

       The vectors u_lambda and fu_lambda contain the u and the lambda interlaced
*/

typedef struct {
  PetscViewer  u_lambda_viewer;
  PetscViewer  fu_lambda_viewer;
} UserCtx;

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);

/*
    Uses full multigrid preconditioner with GMRES (with no preconditioner inside the GMRES) as the 
  smoother on all levels. This is because (1) in the matrix free case no matrix entries are 
  available for doing Jacobi or SOR preconditioning and (2) the explicit matrix case the diagonal
  entry for the control variable is zero which means default SOR will not work.

*/
char  common_options[]      = "-dmmg_grid_sequence \
                               -dmmg_nlevels 5 \
                               -mg_levels_pc_type none \
                               -mg_coarse_pc_type none \
                               -pc_mg_type full \
                               -mg_coarse_ksp_type gmres \
                               -mg_levels_ksp_type gmres \
                               -mg_coarse_ksp_max_it 6 \
                               -mg_levels_ksp_max_it 3";

char  matrix_free_options[] = "-mat_mffd_compute_normu no \
                               -mat_mffd_type wp \
                               -dmmg_jacobian_mf_fd";

/*
    Currently only global coloring is supported with DMComposite
*/
char  matrix_based_options[] = "-dmmg_iscoloring_type global";

/*
     The -use_matrix_based version does not work! This is because the DMComposite code cannot determine the nonzero
  pattern of the Jacobian since the coupling between the boundary condition (array variable) and DA variables is problem 
  dependent. To get the explicit Jacobian correct you would need to use the DMCompositeSetCoupling() to indicate the extra nonzero 
  pattern and run with -dmmg_coloring_from_mat.
*/


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  UserCtx        user;
  DA             da;
  DMMG           *dmmg;
  DMComposite    packer;
  PetscTruth     use_matrix_based = PETSC_FALSE,use_monitor = PETSC_FALSE;
  PetscInt       i;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = PetscOptionsSetFromOptions();CHKERRQ(ierr);

  /* Hardwire several options; can be changed at command line */
  ierr = PetscOptionsInsertString(common_options);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-use_matrix_based",&use_matrix_based,PETSC_IGNORE);CHKERRQ(ierr);
  if (use_matrix_based) {
    ierr = PetscOptionsInsertString(matrix_based_options);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsInsertString(matrix_free_options);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInsert(&argc,&argv,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-use_monitor",&use_monitor,PETSC_IGNORE);CHKERRQ(ierr);

  /* Create a global vector that includes a single redundant array and two da arrays */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = DMCompositeAddArray(packer,0,1);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,-5,2,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,(DM)da);CHKERRQ(ierr);


  /* create nonlinear multi-level solver */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,&user,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)packer);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg,FormFunction,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  if (use_monitor) {
    /* create graphics windows */
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"u_lambda - state variables and Lagrange multipliers",-1,-1,-1,-1,&user.u_lambda_viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"fu_lambda - derivate w.r.t. state variables and Lagrange multipliers",-1,-1,-1,-1,&user.fu_lambda_viewer);CHKERRQ(ierr);
    for (i=0; i<DMMGGetLevels(dmmg); i++) {
      ierr = SNESMonitorSet(dmmg[i]->snes,Monitor,dmmg[i],0);CHKERRQ(ierr); 
    }
  }

  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = DMCompositeDestroy(packer);CHKERRQ(ierr);
  if (use_monitor) {
    ierr = PetscViewerDestroy(user.u_lambda_viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(user.fu_lambda_viewer);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

typedef struct {
  PetscScalar u;
  PetscScalar lambda;
} ULambda;
 
/*
      Evaluates FU = Gradiant(L(w,u,lambda))

     This local function acts on the ghosted version of U (accessed via DMCompositeGetLocalVectors() and
   DMCompositeScatter()) BUT the global, nonghosted version of FU (via DMCompositeGetAccess()).

*/
PetscErrorCode FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  DMMG           dmmg = (DMMG)dummy;
  PetscErrorCode ierr;
  PetscInt       xs,xm,i,N,nredundant;
  ULambda        *u_lambda,*fu_lambda;
  PetscScalar    d,h,*w,*fw;
  Vec            vu_lambda,vfu_lambda;
  DA             da;
  DMComposite        packer = (DMComposite)dmmg->dm;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(packer,&nredundant,&da);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(packer,&w,&vu_lambda);CHKERRQ(ierr);
  ierr = DMCompositeScatter(packer,U,w,vu_lambda);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,FU,&fw,&vfu_lambda);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu_lambda,&u_lambda);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu_lambda,&fu_lambda);CHKERRQ(ierr);
  d    = N-1.0;
  h    = 1.0/d;

  /* derivative of L() w.r.t. w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -2.0*d*u_lambda[0].lambda;
  }

  /* derivative of L() w.r.t. u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu_lambda[0].lambda   =    h*u_lambda[0].u   + 2.*d*u_lambda[0].lambda   - d*u_lambda[1].lambda;
    else if (i == 1)   fu_lambda[1].lambda   = 2.*h*u_lambda[1].u   + 2.*d*u_lambda[1].lambda   - d*u_lambda[2].lambda;
    else if (i == N-1) fu_lambda[N-1].lambda =    h*u_lambda[N-1].u + 2.*d*u_lambda[N-1].lambda - d*u_lambda[N-2].lambda;
    else if (i == N-2) fu_lambda[N-2].lambda = 2.*h*u_lambda[N-2].u + 2.*d*u_lambda[N-2].lambda - d*u_lambda[N-3].lambda;
    else               fu_lambda[i].lambda   = 2.*h*u_lambda[i].u   - d*(u_lambda[i+1].lambda - 2.0*u_lambda[i].lambda + u_lambda[i-1].lambda);
  } 

  /* derivative of L() w.r.t. lambda */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu_lambda[0].u   = 2.0*d*(u_lambda[0].u - w[0]);
    else if (i == N-1) fu_lambda[N-1].u = 2.0*d*u_lambda[N-1].u;
    else               fu_lambda[i].u   = -(d*(u_lambda[i+1].u - 2.0*u_lambda[i].u + u_lambda[i-1].u) - 2.0*h);
  } 

  ierr = DAVecRestoreArray(da,vu_lambda,&u_lambda);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu_lambda,&fu_lambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(packer,&w,&vu_lambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(packer,FU,&fw,&vfu_lambda);CHKERRQ(ierr);
  ierr = PetscLogFlops(13.0*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    Computes the exact solution
*/
PetscErrorCode u_solution(void *dummy,PetscInt n,PetscScalar *x,PetscScalar *u)
{
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    u[2*i] = x[i]*x[i] - 1.25*x[i] + .25;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ExactSolution(DMComposite packer,Vec U) 
{
  PF             pf;
  Vec            x,u_global;
  PetscScalar    *w;
  DA             da;
  PetscErrorCode ierr;
  PetscInt       m;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(packer,&m,&da);CHKERRQ(ierr);

  ierr = PFCreate(PETSC_COMM_WORLD,1,2,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFQUICK,(void*)u_solution);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&x);CHKERRQ(ierr);
  if (!x) {
    ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&x);CHKERRQ(ierr);
  }
  ierr = DMCompositeGetAccess(packer,U,&w,&u_global,0);CHKERRQ(ierr);
  if (w) w[0] = .25;
  ierr = PFApplyVec(pf,x,u_global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(packer,U,&w,&u_global,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal rnorm,void *dummy)
{
  DMMG           dmmg = (DMMG)dummy;
  UserCtx        *user = (UserCtx*)dmmg->user;
  PetscErrorCode ierr;
  PetscInt       m,N;
  PetscScalar    *w,*dw;
  Vec            u_lambda,U,F,Uexact;
  DMComposite        packer = (DMComposite)dmmg->dm;
  PetscReal      norm;
  DA             da;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->u_lambda_viewer); 
  ierr = DMCompositeRestoreAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);

  ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,F,&w,&u_lambda);CHKERRQ(ierr);
  /* ierr = VecView(u_lambda,user->fu_lambda_viewer); */
  ierr = DMCompositeRestoreAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);

  ierr = DMCompositeGetEntries(packer,&m,&da);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Uexact);CHKERRQ(ierr);
  ierr = ExactSolution(packer,Uexact);CHKERRQ(ierr);
  ierr = VecAXPY(Uexact,-1.0,U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(packer,Uexact,&dw,&u_lambda);CHKERRQ(ierr);
  ierr = VecStrideNorm(u_lambda,0,NORM_2,&norm);CHKERRQ(ierr);
  norm = norm/sqrt(N-1.);
  if (dw) ierr = PetscPrintf(dmmg->comm,"Norm of error %G Error at x = 0 %G\n",norm,PetscRealPart(dw[0]));CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->fu_lambda_viewer);
  ierr = DMCompositeRestoreAccess(packer,Uexact,&dw,&u_lambda);CHKERRQ(ierr);
  ierr = VecDestroy(Uexact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








