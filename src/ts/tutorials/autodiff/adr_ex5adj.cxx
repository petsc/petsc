static char help[] = "Demonstrates adjoint sensitivity analysis for Reaction-Diffusion Equations.\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: Automatic differentiation using ADOL-C
*/
/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf

   REQUIRES configuration of PETSc with option --download-colpack

   For documentation on ColPack, see
     $PETSC_ARCH/externalpackages/git.colpack/README.md
*/
/* ------------------------------------------------------------------------
  See ../advection-diffusion-reaction/ex5 for a description of the problem
  ------------------------------------------------------------------------- */

/*
  Runtime options:

    Solver:
      -forwardonly       - Run the forward simulation without adjoint.
      -implicitform      - Provide IFunction and IJacobian to TS, if not set, RHSFunction and RHSJacobian will be used.
      -aijpc             - Set the preconditioner matrix to be aij (the Jacobian matrix can be of a different type such as ELL).

    Jacobian generation:
      -jacobian_by_hand  - Use the hand-coded Jacobian of ex5.c, rather than generating it automatically.
      -no_annotation     - Do not annotate ADOL-C active variables. (Should be used alongside -jacobian_by_hand.)
      -adolc_sparse      - Calculate Jacobian in compressed format, using ADOL-C sparse functionality.
      -adolc_sparse_view - Print sparsity pattern used by -adolc_sparse option.
 */
/*
  NOTE: If -adolc_sparse option is used, at least four processors should be used, so that no processor owns boundaries which are
        identified by the periodic boundary conditions. The number of grid points in both x- and y-directions should be multiples
        of 5, in order for the 5-point stencil to be cleanly parallelised.
*/

#include <petscdmda.h>
#include <petscts.h>
#include "adolc-utils/drivers.cxx"
#include <adolc/adolc.h>

/* (Passive) field for the two variables */
typedef struct {
  PetscScalar u,v;
} Field;

/* Active field for the two variables */
typedef struct {
  adouble u,v;
} AField;

/* Application context */
typedef struct {
  PetscReal D1,D2,gamma,kappa;
  AField    **u_a,**f_a;
  PetscBool aijpc;
  AdolcCtx  *adctx; /* Automatic differentation support */
} AppCtx;

extern PetscErrorCode InitialConditions(DM da,Vec U);
extern PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y);
extern PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr);
extern PetscErrorCode IFunctionActive(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr);
extern PetscErrorCode RHSFunctionActive(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr);
extern PetscErrorCode RHSFunctionPassive(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr);
extern PetscErrorCode IJacobianByHand(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx);
extern PetscErrorCode IJacobianAdolc(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx);
extern PetscErrorCode RHSJacobianByHand(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx);
extern PetscErrorCode RHSJacobianAdolc(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx);

int main(int argc,char **argv)
{
  TS             ts;
  Vec            x,r,xdot;
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  AdolcCtx       *adctx;
  Vec            lambda[1];
  PetscBool      forwardonly=PETSC_FALSE,implicitform=PETSC_FALSE,byhand=PETSC_FALSE;
  PetscInt       gxm,gym,i,dofs = 2,ctrl[3] = {0,0,0};
  PetscScalar    **Seed = NULL,**Rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscNew(&adctx);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL);CHKERRQ(ierr);
  appctx.aijpc = PETSC_FALSE,adctx->no_an = PETSC_FALSE,adctx->sparse = PETSC_FALSE,adctx->sparse_view = PETSC_FALSE;adctx->sparse_view_done = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-aijpc",&appctx.aijpc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&adctx->no_an,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&adctx->sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&adctx->sparse_view,NULL);CHKERRQ(ierr);
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  appctx.adctx = adctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,65,65,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xdot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  if (!implicitform) {
    ierr = TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&appctx);CHKERRQ(ierr);
  } else {
    ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx);CHKERRQ(ierr);
  }

  if (!adctx->no_an) {
    ierr = DMDAGetGhostCorners(da,NULL,NULL,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
    adctx->m = dofs*gxm*gym;adctx->n = dofs*gxm*gym; /* Number of dependent and independent variables */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Trace function(s) just once
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (!implicitform) {
      ierr = RHSFunctionActive(ts,1.0,x,r,&appctx);CHKERRQ(ierr);
    } else {
      ierr = IFunctionActive(ts,1.0,x,xdot,r,&appctx);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      In the case where ADOL-C generates the Jacobian in compressed format,
      seed and recovery matrices are required. Since the sparsity structure
      of the Jacobian does not change over the course of the time
      integration, we can save computational effort by only generating
      these objects once.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (adctx->sparse) {
      /*
         Generate sparsity pattern

         One way to do this is to use the Jacobian pattern driver `jac_pat`
         provided by ColPack. Otherwise, it is the responsibility of the user
         to obtain the sparsity pattern.
      */
      ierr = PetscMalloc1(adctx->n,&u_vec);CHKERRQ(ierr);
      JP = (unsigned int **) malloc(adctx->m*sizeof(unsigned int*));
      jac_pat(1,adctx->m,adctx->n,u_vec,JP,ctrl);
      if (adctx->sparse_view) {
        ierr = PrintSparsity(MPI_COMM_WORLD,adctx->m,JP);CHKERRQ(ierr);
      }

      /*
        Extract a column colouring

        For problems using DMDA, colourings can be extracted directly, as
        follows. There are also ColPack drivers available. Otherwise, it is the
        responsibility of the user to obtain a suitable colouring.
      */
      ierr = DMCreateColoring(da,IS_COLORING_LOCAL,&iscoloring);CHKERRQ(ierr);
      ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&adctx->p,NULL);CHKERRQ(ierr);

      /* Generate seed matrix to propagate through the forward mode of AD */
      ierr = AdolcMalloc2(adctx->n,adctx->p,&Seed);CHKERRQ(ierr);
      ierr = GenerateSeedMatrix(iscoloring,Seed);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

      /*
        Generate recovery matrix, which is used to recover the Jacobian from
        compressed format */
      ierr = AdolcMalloc2(adctx->m,adctx->p,&Rec);CHKERRQ(ierr);
      ierr = GetRecoveryMatrix(Seed,JP,adctx->m,adctx->p,Rec);CHKERRQ(ierr);

      /* Store results and free workspace */
      adctx->Rec = Rec;
      for (i=0;i<adctx->m;i++)
        free(JP[i]);
      free(JP);
      ierr = PetscFree(u_vec);CHKERRQ(ierr);

    } else {

      /*
        In 'full' Jacobian mode, propagate an identity matrix through the
        forward mode of AD.
      */
      adctx->p = adctx->n;
      ierr = AdolcMalloc2(adctx->n,adctx->p,&Seed);CHKERRQ(ierr);
      ierr = Identity(adctx->n,Seed);CHKERRQ(ierr);
    }
    adctx->Seed = Seed;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!implicitform) {
    if (!byhand) {
      ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianAdolc,&appctx);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianByHand,&appctx);CHKERRQ(ierr);
    }
  } else {
    if (appctx.aijpc) {
      Mat A,B;

      ierr = DMSetMatType(da,MATSELL);CHKERRQ(ierr);
      ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
      ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      /* FIXME do we need to change viewer to display matrix in natural ordering as DMCreateMatrix_DA does? */
      if (!byhand) {
        ierr = TSSetIJacobian(ts,A,B,IJacobianAdolc,&appctx);CHKERRQ(ierr);
      } else {
        ierr = TSSetIJacobian(ts,A,B,IJacobianByHand,&appctx);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
    } else {
      if (!byhand) {
        ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianAdolc,&appctx);CHKERRQ(ierr);
      } else {
        ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianByHand,&appctx);CHKERRQ(ierr);
      }
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Have the TS save its trajectory so that TSAdjointSolve() may be used
    and set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!forwardonly) {
    ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,200.0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,0.5);CHKERRQ(ierr);
  } else {
    ierr = TSSetMaxTime(ts,2000.0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,10);CHKERRQ(ierr);
  }
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  if (!forwardonly) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Start the Adjoint model
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = VecDuplicate(x,&lambda[0]);CHKERRQ(ierr);
    /*   Reset initial conditions for the adjoint integration */
    ierr = InitializeLambda(da,lambda[0],0.5,0.5);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ts,1,lambda,NULL);CHKERRQ(ierr);
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (!adctx->no_an) {
    if (adctx->sparse)
      ierr = AdolcFree2(Rec);CHKERRQ(ierr);
    ierr = AdolcFree2(Seed);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if ((1.0 <= x) && (x <= 1.5) && (1.0 <= y) && (y <= 1.5)) u[j][i].v = .25*PetscPowReal(PetscSinReal(4.0*PETSC_PI*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y)
{
   PetscInt i,j,Mx,My,xs,ys,xm,ym;
   PetscErrorCode ierr;
   Field **l;
   PetscFunctionBegin;

   ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
   /* locate the global i index for x and j index for y */
   i = (PetscInt)(x*(Mx-1));
   j = (PetscInt)(y*(My-1));
   ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

   if (xs <= i && i < xs+xm && ys <= j && j < ys+ym) {
     /* the i,j vertex is on this process */
     ierr = DMDAVecGetArray(da,lambda,&l);CHKERRQ(ierr);
     l[j][i].u = 1.0;
     l[j][i].v = 1.0;
     ierr = DMDAVecRestoreArray(da,lambda,&l);CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}

PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); sy = 1.0/(hy*hy);

  /* Get local grid boundaries */
  xs = info->xs; xm = info->xm; ys = info->ys; ym = info->ym;

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      f[j][i].u = udot[j][i].u - appctx->D1*(uxx + uyy) + uc*vc*vc - appctx->gamma*(1.0 - uc);
      f[j][i].v = udot[j][i].v - appctx->D2*(vxx + vyy) - uc*vc*vc + (appctx->gamma + appctx->kappa)*vc;
    }
  }
  ierr = PetscLogFlops(16.0*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IFunctionActive(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  DMDALocalInfo  info;
  Field          **u,**f,**udot;
  Vec            localU;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,sx,sy;
  adouble        uc,uxx,uyy,vc,vxx,vyy;
  AField         **f_a,*f_c,**u_a,*u_c;
  PetscScalar    dummy;

  PetscFunctionBegin;

  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(info.mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info.my); sy = 1.0/(hy*hy);
  xs = info.xs; xm = info.xm; gxs = info.gxs; gxm = info.gxm;
  ys = info.ys; ym = info.ym; gys = info.gys; gym = info.gym;

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Udot,&udot);CHKERRQ(ierr);

  /*
    Create contiguous 1-arrays of AFields

    NOTE: Memory for ADOL-C active variables (such as adouble and AField)
          cannot be allocated using PetscMalloc, as this does not call the
          relevant class constructor. Instead, we use the C++ keyword `new`.
  */
  u_c = new AField[info.gxm*info.gym];
  f_c = new AField[info.gxm*info.gym];

  /* Create corresponding 2-arrays of AFields */
  u_a = new AField*[info.gym];
  f_a = new AField*[info.gym];

  /* Align indices between array types to endow 2d array with ghost points */
  ierr = GiveGhostPoints(da,u_c,&u_a);CHKERRQ(ierr);
  ierr = GiveGhostPoints(da,f_c,&f_a);CHKERRQ(ierr);

  trace_on(1);  /* Start of active section on tape 1 */

  /*
    Mark independence

    NOTE: Ghost points are marked as independent, in place of the points they represent on
          other processors / on other boundaries.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_a[j][i].u <<= u[j][i].u;
      u_a[j][i].v <<= u[j][i].v;
    }
  }

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u_a[j][i].u;
      uxx       = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*sy;
      vc        = u_a[j][i].v;
      vxx       = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*sy;
      f_a[j][i].u = udot[j][i].u - appctx->D1*(uxx + uyy) + uc*vc*vc - appctx->gamma*(1.0 - uc);
      f_a[j][i].v = udot[j][i].v - appctx->D2*(vxx + vyy) - uc*vc*vc + (appctx->gamma + appctx->kappa)*vc;
    }
  }

  /*
    Mark dependence

    NOTE: Marking dependence of dummy variables makes the index notation much simpler when forming
          the Jacobian later.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      if ((i < xs) || (i >= xs+xm) || (j < ys) || (j >= ys+ym)) {
        f_a[j][i].u >>= dummy;
        f_a[j][i].v >>= dummy;
      } else {
        f_a[j][i].u >>= f[j][i].u;
        f_a[j][i].v >>= f[j][i].v;
      }
    }
  }
  trace_off();  /* End of active section */
  ierr = PetscLogFlops(16.0*xm*ym);CHKERRQ(ierr);

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Udot,&udot);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);

  /* Destroy AFields appropriately */
  f_a += info.gys;
  u_a += info.gys;
  delete[] f_a;
  delete[] u_a;
  delete[] f_c;
  delete[] u_c;

  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunctionPassive(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;
  Field          **u,**f;
  Vec            localU,localF;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My);sy = 1.0/(hy*hy);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localF);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr); // NOTE (1): See (2) below
  ierr = DMGlobalToLocalBegin(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localF,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;
    }
  }

  /*
     Gather global vector, using the 2-step process
        DMLocalToGlobalBegin(),DMLocalToGlobalEnd().

     NOTE (2): We need to use ADD_VALUES if boundaries are not of type DM_BOUNDARY_NONE or
               DM_BOUNDARY_GHOSTED, meaning we should also zero F before addition (see (1) above).
  */
  ierr = DMLocalToGlobalBegin(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localF,ADD_VALUES,F);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localF,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscLogFlops(16.0*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunctionActive(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym,Mx,My;
  PetscReal      hx,hy,sx,sy;
  AField         **f_a,*f_c,**u_a,*u_c;
  adouble        uc,uxx,uyy,vc,vxx,vyy;
  Field          **u,**f;
  Vec            localU,localF;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My);sy = 1.0/(hy*hy);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localF);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr); // NOTE (1): See (2) below
  ierr = DMGlobalToLocalBegin(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localF,&f);CHKERRQ(ierr);

  /*
     Get local and ghosted grid boundaries
  */
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
    Create contiguous 1-arrays of AFields

    NOTE: Memory for ADOL-C active variables (such as adouble and AField)
          cannot be allocated using PetscMalloc, as this does not call the
          relevant class constructor. Instead, we use the C++ keyword `new`.
  */
  u_c = new AField[gxm*gym];
  f_c = new AField[gxm*gym];

  /* Create corresponding 2-arrays of AFields */
  u_a = new AField*[gym];
  f_a = new AField*[gym];

  /* Align indices between array types to endow 2d array with ghost points */
  ierr = GiveGhostPoints(da,u_c,&u_a);CHKERRQ(ierr);
  ierr = GiveGhostPoints(da,f_c,&f_a);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  trace_on(1);  // ----------------------------------------------- Start of active section

  /*
    Mark independence

    NOTE: Ghost points are marked as independent, in place of the points they represent on
          other processors / on other boundaries.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_a[j][i].u <<= u[j][i].u;
      u_a[j][i].v <<= u[j][i].v;
    }
  }

  /*
    Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc          = u_a[j][i].u;
      uxx         = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*sx;
      uyy         = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*sy;
      vc          = u_a[j][i].v;
      vxx         = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*sx;
      vyy         = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*sy;
      f_a[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f_a[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;
    }
  }
  /*
    Mark dependence

    NOTE: Ghost points are marked as dependent in order to vastly simplify index notation
          during Jacobian assembly.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      f_a[j][i].u >>= f[j][i].u;
      f_a[j][i].v >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section

  /* Test zeroth order scalar evaluation in ADOL-C gives the same result */
//  if (appctx->adctx->zos) {
//    ierr = TestZOS2d(da,f,u,appctx);CHKERRQ(ierr); // FIXME: Why does this give nonzero?
//  }

  /*
     Gather global vector, using the 2-step process
        DMLocalToGlobalBegin(),DMLocalToGlobalEnd().

     NOTE (2): We need to use ADD_VALUES if boundaries are not of type DM_BOUNDARY_NONE or
               DM_BOUNDARY_GHOSTED, meaning we should also zero F before addition (see (1) above).
  */
  ierr = DMLocalToGlobalBegin(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localF,ADD_VALUES,F);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localF,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscLogFlops(16.0*xm*ym);CHKERRQ(ierr);

  /* Destroy AFields appropriately */
  f_a += gys;
  u_a += gys;
  delete[] f_a;
  delete[] u_a;
  delete[] f_c;
  delete[] u_c;

  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianAdolc(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  AppCtx            *appctx = (AppCtx*)ctx;
  DM                da;
  PetscErrorCode    ierr;
  const PetscScalar *u_vec;
  Vec               localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = VecGetArrayRead(localU,&u_vec);CHKERRQ(ierr);

  /*
    Compute Jacobian
  */
  ierr = PetscAdolcComputeIJacobianLocalIDMass(1,A,u_vec,a,appctx->adctx);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = VecRestoreArrayRead(localU,&u_vec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianByHand(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  Field          **u;
  Vec            localU;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {

    stencil[0].j = j-1;
    stencil[1].j = j+1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0; rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;

      /*      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;

      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
       f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);*/

      stencil[0].i = i; stencil[0].c = 0; entries[0] = -appctx->D1*sy;
      stencil[1].i = i; stencil[1].c = 0; entries[1] = -appctx->D1*sy;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = -appctx->D1*sx;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = -appctx->D1*sx;
      stencil[4].i = i; stencil[4].c = 0; entries[4] = 2.0*appctx->D1*(sx + sy) + vc*vc + appctx->gamma + a;
      stencil[5].i = i; stencil[5].c = 1; entries[5] = 2.0*uc*vc;
      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      if (appctx->aijpc) {
        ierr = MatSetValuesStencil(B,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      }
      stencil[0].c = 1; entries[0] = -appctx->D2*sy;
      stencil[1].c = 1; entries[1] = -appctx->D2*sy;
      stencil[2].c = 1; entries[2] = -appctx->D2*sx;
      stencil[3].c = 1; entries[3] = -appctx->D2*sx;
      stencil[4].c = 1; entries[4] = 2.0*appctx->D2*(sx + sy) - 2.0*uc*vc + appctx->gamma + appctx->kappa + a;
      stencil[5].c = 0; entries[5] = -vc*vc;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      if (appctx->aijpc) {
        ierr = MatSetValuesStencil(B,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc; */
    }
  }

  /*
     Restore vectors
  */
  ierr = PetscLogFlops(19.0*xm*ym);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  if (appctx->aijpc) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobianByHand(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  Field          **u;
  Vec            localU;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My); sy = 1.0/(hy*hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {

    stencil[0].j = j-1;
    stencil[1].j = j+1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0; rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;

      /*      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;

      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
       f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);*/

      stencil[0].i = i; stencil[0].c = 0; entries[0] = appctx->D1*sy;
      stencil[1].i = i; stencil[1].c = 0; entries[1] = appctx->D1*sy;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = appctx->D1*sx;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = appctx->D1*sx;
      stencil[4].i = i; stencil[4].c = 0; entries[4] = -2.0*appctx->D1*(sx + sy) - vc*vc - appctx->gamma;
      stencil[5].i = i; stencil[5].c = 1; entries[5] = -2.0*uc*vc;
      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);

      stencil[0].c = 1; entries[0] = appctx->D2*sy;
      stencil[1].c = 1; entries[1] = appctx->D2*sy;
      stencil[2].c = 1; entries[2] = appctx->D2*sx;
      stencil[3].c = 1; entries[3] = appctx->D2*sx;
      stencil[4].c = 1; entries[4] = -2.0*appctx->D2*(sx + sy) + 2.0*uc*vc - appctx->gamma - appctx->kappa;
      stencil[5].c = 0; entries[5] = vc*vc;
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      /* f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc; */
    }
  }

  /*
     Restore vectors
  */
  ierr = PetscLogFlops(19.0*xm*ym);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  if (appctx->aijpc) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobianAdolc(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             da;
  PetscErrorCode ierr;
  PetscScalar    *u_vec;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = VecGetArray(localU,&u_vec);CHKERRQ(ierr);

  /*
    Compute Jacobian
  */
  ierr = PetscAdolcComputeRHSJacobianLocal(1,A,u_vec,appctx->adctx);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(localU,&u_vec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

  build:
    requires: double !complex adolc colpack

  test:
    suffix: 1
    nsize: 1
    args: -ts_max_steps 1 -da_grid_x 12 -da_grid_y 12 -snes_test_jacobian
    output_file: output/adr_ex5adj_1.out

  test:
    suffix: 2
    nsize: 1
    args: -ts_max_steps 1 -da_grid_x 12 -da_grid_y 12 -snes_test_jacobian -implicitform
    output_file: output/adr_ex5adj_2.out

  test:
    suffix: 3
    nsize: 4
    args: -ts_max_steps 10 -da_grid_x 12 -da_grid_y 12 -ts_monitor -ts_adjoint_monitor
    output_file: output/adr_ex5adj_3.out

  test:
    suffix: 4
    nsize: 4
    args: -ts_max_steps 10 -da_grid_x 12 -da_grid_y 12 -ts_monitor -ts_adjoint_monitor -implicitform
    output_file: output/adr_ex5adj_4.out

  testset:
    suffix: 5
    nsize: 4
    args: -ts_max_steps 10 -da_grid_x 15 -da_grid_y 15 -ts_monitor -ts_adjoint_monitor -adolc_sparse
    output_file: output/adr_ex5adj_5.out

  testset:
    suffix: 6
    nsize: 4
    args: -ts_max_steps 10 -da_grid_x 15 -da_grid_y 15 -ts_monitor -ts_adjoint_monitor -adolc_sparse -implicitform
    output_file: output/adr_ex5adj_6.out

TEST*/
