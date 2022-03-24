
static const char help[] = "Solves PDE optimization problem using full-space method, interlaces state and adjoint variables.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmredundant.h>
#include <petscdmcomposite.h>
#include <petscpf.h>
#include <petscsnes.h>
#include <petsc/private/dmimpl.h>

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
  PetscViewer u_lambda_viewer;
  PetscViewer fu_lambda_viewer;
} UserCtx;

extern PetscErrorCode ComputeFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode ComputeJacobian_MF(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);

/*
    Uses full multigrid preconditioner with GMRES (with no preconditioner inside the GMRES) as the
  smoother on all levels. This is because (1) in the matrix free case no matrix entries are
  available for doing Jacobi or SOR preconditioning and (2) the explicit matrix case the diagonal
  entry for the control variable is zero which means default SOR will not work.

*/
char common_options[] = "-ksp_type fgmres\
                         -snes_grid_sequence 2 \
                         -pc_type mg\
                         -mg_levels_pc_type none \
                         -mg_coarse_pc_type none \
                         -pc_mg_type full \
                         -mg_coarse_ksp_type gmres \
                         -mg_levels_ksp_type gmres \
                         -mg_coarse_ksp_max_it 6 \
                         -mg_levels_ksp_max_it 3";

char matrix_free_options[] = "-mat_mffd_compute_normu no \
                              -mat_mffd_type wp";

extern PetscErrorCode DMCreateMatrix_MF(DM,Mat*);

int main(int argc,char **argv)
{
  UserCtx        user;
  DM             red,da;
  SNES           snes;
  DM             packer;
  PetscBool      use_monitor = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  /* Hardwire several options; can be changed at command line */
  CHKERRQ(PetscOptionsInsertString(NULL,common_options));
  CHKERRQ(PetscOptionsInsertString(NULL,matrix_free_options));
  CHKERRQ(PetscOptionsInsert(NULL,&argc,&argv,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-use_monitor",&use_monitor,PETSC_IGNORE));

  /* Create a global vector that includes a single redundant array and two da arrays */
  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD,&packer));
  CHKERRQ(DMRedundantCreate(PETSC_COMM_WORLD,0,1,&red));
  CHKERRQ(DMSetOptionsPrefix(red,"red_"));
  CHKERRQ(DMCompositeAddDM(packer,red));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,5,2,1,NULL,&da));
  CHKERRQ(DMSetOptionsPrefix(red,"da_"));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetFieldName(da,0,"u"));
  CHKERRQ(DMDASetFieldName(da,1,"lambda"));
  CHKERRQ(DMCompositeAddDM(packer,(DM)da));
  CHKERRQ(DMSetApplicationContext(packer,&user));

  packer->ops->creatematrix = DMCreateMatrix_MF;

  /* create nonlinear multi-level solver */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetDM(snes,packer));
  CHKERRQ(SNESSetFunction(snes,NULL,ComputeFunction,NULL));
  CHKERRQ(SNESSetJacobian(snes,NULL, NULL,ComputeJacobian_MF,NULL));

  CHKERRQ(SNESSetFromOptions(snes));

  if (use_monitor) {
    /* create graphics windows */
    CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"u_lambda - state variables and Lagrange multipliers",-1,-1,-1,-1,&user.u_lambda_viewer));
    CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"fu_lambda - derivate w.r.t. state variables and Lagrange multipliers",-1,-1,-1,-1,&user.fu_lambda_viewer));
    CHKERRQ(SNESMonitorSet(snes,Monitor,0,0));
  }

  CHKERRQ(SNESSolve(snes,NULL,NULL));
  CHKERRQ(SNESDestroy(&snes));

  CHKERRQ(DMDestroy(&red));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(DMDestroy(&packer));
  if (use_monitor) {
    CHKERRQ(PetscViewerDestroy(&user.u_lambda_viewer));
    CHKERRQ(PetscViewerDestroy(&user.fu_lambda_viewer));
  }
  CHKERRQ(PetscFinalize());
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
PetscErrorCode ComputeFunction(SNES snes,Vec U,Vec FU,void *ctx)
{
  PetscInt       xs,xm,i,N;
  ULambda        *u_lambda,*fu_lambda;
  PetscScalar    d,h,*w,*fw;
  Vec            vw,vfw,vu_lambda,vfu_lambda;
  DM             packer,red,da;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetDM(U, &packer));
  CHKERRQ(DMCompositeGetEntries(packer,&red,&da));
  CHKERRQ(DMCompositeGetLocalVectors(packer,&vw,&vu_lambda));
  CHKERRQ(DMCompositeScatter(packer,U,vw,vu_lambda));
  CHKERRQ(DMCompositeGetAccess(packer,FU,&vfw,&vfu_lambda));

  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  CHKERRQ(DMDAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(VecGetArray(vw,&w));
  CHKERRQ(VecGetArray(vfw,&fw));
  CHKERRQ(DMDAVecGetArray(da,vu_lambda,&u_lambda));
  CHKERRQ(DMDAVecGetArray(da,vfu_lambda,&fu_lambda));
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

  CHKERRQ(VecRestoreArray(vw,&w));
  CHKERRQ(VecRestoreArray(vfw,&fw));
  CHKERRQ(DMDAVecRestoreArray(da,vu_lambda,&u_lambda));
  CHKERRQ(DMDAVecRestoreArray(da,vfu_lambda,&fu_lambda));
  CHKERRQ(DMCompositeRestoreLocalVectors(packer,&vw,&vu_lambda));
  CHKERRQ(DMCompositeRestoreAccess(packer,FU,&vfw,&vfu_lambda));
  CHKERRQ(PetscLogFlops(13.0*N));
  PetscFunctionReturn(0);
}

/*
    Computes the exact solution
*/
PetscErrorCode u_solution(void *dummy,PetscInt n,const PetscScalar *x,PetscScalar *u)
{
  PetscInt i;

  PetscFunctionBeginUser;
  for (i=0; i<n; i++) u[2*i] = x[i]*x[i] - 1.25*x[i] + .25;
  PetscFunctionReturn(0);
}

PetscErrorCode ExactSolution(DM packer,Vec U)
{
  PF             pf;
  Vec            x,u_global;
  PetscScalar    *w;
  DM             da;
  PetscInt       m;

  PetscFunctionBeginUser;
  CHKERRQ(DMCompositeGetEntries(packer,&m,&da));

  CHKERRQ(PFCreate(PETSC_COMM_WORLD,1,2,&pf));
  /* The cast through PETSC_UINTPTR_T is so that compilers will warn about casting to void * from void(*)(void) */
  CHKERRQ(PFSetType(pf,PFQUICK,(void*)(PETSC_UINTPTR_T)u_solution));
  CHKERRQ(DMGetCoordinates(da,&x));
  if (!x) {
    CHKERRQ(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));
    CHKERRQ(DMGetCoordinates(da,&x));
  }
  CHKERRQ(DMCompositeGetAccess(packer,U,&w,&u_global,0));
  if (w) w[0] = .25;
  CHKERRQ(PFApplyVec(pf,x,u_global));
  CHKERRQ(PFDestroy(&pf));
  CHKERRQ(DMCompositeRestoreAccess(packer,U,&w,&u_global,0));
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal rnorm,void *dummy)
{
  UserCtx        *user;
  PetscInt       m,N;
  PetscScalar    *w,*dw;
  Vec            u_lambda,U,F,Uexact;
  DM             packer;
  PetscReal      norm;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetDM(snes,&packer));
  CHKERRQ(DMGetApplicationContext(packer,&user));
  CHKERRQ(SNESGetSolution(snes,&U));
  CHKERRQ(DMCompositeGetAccess(packer,U,&w,&u_lambda));
  CHKERRQ(VecView(u_lambda,user->u_lambda_viewer));
  CHKERRQ(DMCompositeRestoreAccess(packer,U,&w,&u_lambda));

  CHKERRQ(SNESGetFunction(snes,&F,0,0));
  CHKERRQ(DMCompositeGetAccess(packer,F,&w,&u_lambda));
  /* ierr = VecView(u_lambda,user->fu_lambda_viewer); */
  CHKERRQ(DMCompositeRestoreAccess(packer,U,&w,&u_lambda));

  CHKERRQ(DMCompositeGetEntries(packer,&m,&da));
  CHKERRQ(DMDAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(VecDuplicate(U,&Uexact));
  CHKERRQ(ExactSolution(packer,Uexact));
  CHKERRQ(VecAXPY(Uexact,-1.0,U));
  CHKERRQ(DMCompositeGetAccess(packer,Uexact,&dw,&u_lambda));
  CHKERRQ(VecStrideNorm(u_lambda,0,NORM_2,&norm));
  norm = norm/PetscSqrtReal((PetscReal)N-1.);
  if (dw) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Error at x = 0 %g\n",(double)norm,(double)PetscRealPart(dw[0])));
  CHKERRQ(VecView(u_lambda,user->fu_lambda_viewer));
  CHKERRQ(DMCompositeRestoreAccess(packer,Uexact,&dw,&u_lambda));
  CHKERRQ(VecDestroy(&Uexact));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_MF(DM packer,Mat *A)
{
  Vec            t;
  PetscInt       m;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetGlobalVector(packer,&t));
  CHKERRQ(VecGetLocalSize(t,&m));
  CHKERRQ(DMRestoreGlobalVector(packer,&t));
  CHKERRQ(MatCreateMFFD(PETSC_COMM_WORLD,m,m,PETSC_DETERMINE,PETSC_DETERMINE,A));
  CHKERRQ(MatSetUp(*A));
  CHKERRQ(MatSetDM(*A,packer));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeJacobian_MF(SNES snes,Vec x,Mat A,Mat B,void *ctx)
{
  PetscFunctionBeginUser;
  CHKERRQ(MatMFFDSetFunction(A,(PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunction,snes));
  CHKERRQ(MatMFFDSetBase(A,x,NULL));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 2
      args: -da_grid_x 10 -snes_converged_reason -ksp_converged_reason -snes_view
      requires: !single

TEST*/
