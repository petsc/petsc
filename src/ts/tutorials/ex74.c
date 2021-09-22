static char help[] = "Solves the constant-coefficient 1D heat equation \n\
with an Implicit Runge-Kutta method using MatKAIJ.                  \n\
                                                                    \n\
    du      d^2 u                                                   \n\
    --  = a ----- ; 0 <= x <= 1;                                    \n\
    dt      dx^2                                                    \n\
                                                                    \n\
  with periodic boundary conditions                                 \n\
                                                                    \n\
2nd order central discretization in space:                          \n\
                                                                    \n\
   [ d^2 u ]     u_{i+1} - 2u_i + u_{i-1}                           \n\
   [ ----- ]  =  ------------------------                           \n\
   [ dx^2  ]i              h^2                                      \n\
                                                                    \n\
    i = grid index;    h = x_{i+1}-x_i (Uniform)                    \n\
    0 <= i < n         h = 1.0/n                                    \n\
                                                                    \n\
Thus,                                                               \n\
                                                                    \n\
   du                                                               \n\
   --  = Ju;  J = (a/h^2) tridiagonal(1,-2,1)_n                     \n\
   dt                                                               \n\
                                                                    \n\
This example is a TS version of the KSP ex74.c tutorial.            \n";

#include <petscts.h>

typedef enum {
  PHYSICS_DIFFUSION,
  PHYSICS_ADVECTION
} PhysicsType;
const char *const PhysicsTypes[] = {"DIFFUSION","ADVECTION","PhysicsType","PHYSICS_",NULL};

typedef struct Context {
  PetscReal     a;              /* diffusion coefficient      */
  PetscReal     xmin,xmax;      /* domain bounds              */
  PetscInt      imax;           /* number of grid points      */
  PhysicsType   physics_type;
} UserContext;

static PetscErrorCode ExactSolution(Vec,void*,PetscReal);
static PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc, char **argv)
{
  TS             ts;
  Mat            A;
  Vec            u,uex;
  UserContext    ctxt;
  PetscReal      err,ftime;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* default value */
  ctxt.a       = 0.1;
  ctxt.xmin    = 0.0;
  ctxt.xmax    = 1.0;
  ctxt.imax    = 40;
  ctxt.physics_type = PHYSICS_DIFFUSION;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"IRK options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-a","diffusion coefficient","<1.0>",ctxt.a,&ctxt.a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-imax","grid size","<20>",ctxt.imax,&ctxt.imax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","xmin","<0.0>",ctxt.xmin,&ctxt.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","xmax","<1.0>",ctxt.xmax,&ctxt.xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-physics_type","Type of process to discretize","",PhysicsTypes,(PetscEnum)ctxt.physics_type,(PetscEnum*)&ctxt.physics_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* allocate and initialize solution vector and exact solution */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,ctxt.imax);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uex);CHKERRQ(ierr);
  /* initial solution */
  ierr = ExactSolution(u,&ctxt,0.0);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,ctxt.imax,ctxt.imax);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* Create and set options for TS */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.125);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,10);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&ctxt);CHKERRQ(ierr);
  ierr = RHSJacobian(ts,0,u,A,A,&ctxt);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&ctxt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,u);CHKERRQ(ierr);

  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  /* exact   solution */
  ierr = ExactSolution(uex,&ctxt,ftime);CHKERRQ(ierr);

  /* Calculate error in final solution */
  ierr = VecAYPX(uex,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(uex,NORM_2,&err);CHKERRQ(ierr);
  err  = PetscSqrtReal(err*err/((PetscReal)ctxt.imax));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 norm of the numerical error = %g (time=%g)\n",(double)err,(double)ftime);CHKERRQ(ierr);

  /* Free up memory */
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&uex);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ExactSolution(Vec u,void *c,PetscReal t)
{
  UserContext     *ctxt = (UserContext*) c;
  PetscErrorCode  ierr;
  PetscInt        i,is,ie;
  PetscScalar     *uarr;
  PetscReal       x,dx,a=ctxt->a,pi=PETSC_PI;

  PetscFunctionBegin;
  dx = (ctxt->xmax - ctxt->xmin)/((PetscReal) ctxt->imax);
  ierr = VecGetOwnershipRange(u,&is,&ie);CHKERRQ(ierr);
  ierr = VecGetArray(u,&uarr);CHKERRQ(ierr);
  for (i=is; i<ie; i++) {
    x          = i * dx;
    switch (ctxt->physics_type) {
    case PHYSICS_DIFFUSION:
      uarr[i-is] = PetscExpScalar(-4.0*pi*pi*a*t)*PetscSinScalar(2*pi*x);
      break;
    case PHYSICS_ADVECTION:
      uarr[i-is] = PetscSinScalar(2*pi*(x - a*t));
      break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for physics type %s",PhysicsTypes[ctxt->physics_type]);
    }
  }
  ierr = VecRestoreArray(u,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat J,Mat Jpre,void *ctx)
{
  UserContext    *user = (UserContext*) ctx;
  PetscInt       matis,matie,i;
  PetscReal      dx,dx2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dx = (user->xmax - user->xmin)/((PetscReal)user->imax); dx2 = dx*dx;
  ierr = MatGetOwnershipRange(J,&matis,&matie);CHKERRQ(ierr);
  for (i=matis; i<matie; i++) {
    PetscScalar values[3];
    PetscInt    col[3];
    switch (user->physics_type) {
    case PHYSICS_DIFFUSION:
      values[0] = user->a*1.0/dx2;
      values[1] = -user->a*2.0/dx2;
      values[2] = user->a*1.0/dx2;
      break;
    case PHYSICS_ADVECTION:
      values[0] = user->a*.5/dx;
      values[1] = 0.;
      values[2] = -user->a*.5/dx;
      break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for physics type %s",PhysicsTypes[user->physics_type]);
    }
    /* periodic boundaries */
    if (i == 0) {
      col[0] = user->imax-1;
      col[1] = i;
      col[2] = i+1;
    } else if (i == user->imax-1) {
      col[0] = i-1;
      col[1] = i;
      col[2] = 0;
    } else {
      col[0] = i-1;
      col[1] = i;
      col[2] = i+1;
    }
    ierr = MatSetValues(J,1,&i,3,col,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

  test:
    requires: double
    suffix: 1
    nsize: {{1 2}}
    args: -ts_max_steps 5 -ts_monitor -ksp_monitor_short -pc_type pbjacobi -ksp_atol 1e-6 -ts_type irk -ts_irk_nstages 2

  test:
    requires: double
    suffix: 2
    args: -ts_max_steps 5 -ts_monitor -ksp_monitor_short -pc_type pbjacobi -ksp_atol 1e-6 -ts_type irk -ts_irk_nstages 3

  test:
    requires: hpddm double
    suffix: 3
    args: -ts_max_steps 5 -ts_monitor -ksp_monitor_short -pc_type pbjacobi -ksp_atol 1e-6 -ts_type irk -ts_irk_nstages 3 -ksp_view_final_residual -ksp_hpddm_type bgcrodr -ksp_type hpddm
TEST*/
