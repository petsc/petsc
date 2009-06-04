
static char help[] = "This program is a replica of ex6.c except that it does 2 solves to avoid paging.\n\
This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations in parallel, using 2-dimensional distributed arrays.\n\
The 2-dim Bratu (SFI - solid fuel ignition) test problem is used, where\n\
analytic formation of the Jacobian is the default.  The command line\n\
options are:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*  
    1) Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
*/

#include "petscsnes.h"
#include "petscda.h"

/* User-defined application context */
typedef struct {
   PetscReal   param;         /* test problem parameter */
   PetscInt    mx,my;         /* discretization in x, y directions */
   Vec         localX,localF; /* ghosted local vector */
   DA          da;            /* distributed array data structure */
} AppCtx;

extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*),FormInitialGuess1(AppCtx*,Vec);
extern PetscErrorCode FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                      /* nonlinear solver */
  const SNESType type = SNESLS;             /* nonlinear solution method */
  Vec            x,r;                       /* solution, residual vectors */
  Mat            J;                         /* Jacobian matrix */
  AppCtx         user;                      /* user-defined work context */
  PetscInt       i,its,N,Nx = PETSC_DECIDE,Ny = PETSC_DECIDE;
  PetscErrorCode ierr;
  PetscTruth     matrix_free = PETSC_FALSE;
  PetscMPIInt    size; 
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stages[2];
#endif

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscLogStageRegister("stage 1",&stages[0]);
  ierr = PetscLogStageRegister("stage 2",&stages[1]);
  for (i=0; i<2; i++) {
    ierr = PetscLogStagePush(stages[i]);CHKERRQ(ierr);
    user.mx = 4; user.my = 4; user.param = 6.0;
    
    if (i!=0) {
      ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetReal(PETSC_NULL,"-par",&user.param,PETSC_NULL);CHKERRQ(ierr);
      if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
        SETERRQ(1,"Lambda is out of range");
      }
    }
    N = user.mx*user.my;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRQ(ierr);
    if (Nx*Ny != size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE))
      SETERRQ(1,"Incompatible number of processors:  Nx * Ny != size");
    
    /* Set up distributed array */
    ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.mx,user.my,Nx,Ny,1,1,
                      PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(ierr);
    ierr = DACreateGlobalVector(user.da,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
    ierr = DACreateLocalVector(user.da,&user.localX);CHKERRQ(ierr);
    ierr = VecDuplicate(user.localX,&user.localF);CHKERRQ(ierr);

    /* Create nonlinear solver and set function evaluation routine */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetType(snes,type);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction1,&user);CHKERRQ(ierr);

    /* Set default Jacobian evaluation routine.  User can override with:
       -snes_mf : matrix-free Newton-Krylov method with no preconditioning
       (unless user explicitly sets preconditioner) 
       -snes_fd : default finite differencing approximation of Jacobian
       */
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-snes_mf",&matrix_free,PETSC_NULL);CHKERRQ(ierr);
    if (!matrix_free) {
      PetscTruth matrix_free_operator = PETSC_FALSE;
      ierr = PetscOptionsGetTruth(PETSC_NULL,"-snes_mf_operator",&matrix_free_operator,PETSC_NULL);CHKERRQ(ierr);
      if (matrix_free_operator) matrix_free = PETSC_FALSE;
    }
    if (!matrix_free) {
      ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
      ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
      ierr = MatSetFromOptions(J);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,J,J,FormJacobian1,&user);CHKERRQ(ierr);
    }

    /* Set PetscOptions, then solve nonlinear system */
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = FormInitialGuess1(&user,x);CHKERRQ(ierr);
    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D\n",its);CHKERRQ(ierr);

  /* Free data structures */
    if (!matrix_free) {
      ierr = MatDestroy(J);CHKERRQ(ierr);
    }
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = VecDestroy(r);CHKERRQ(ierr);
    ierr = VecDestroy(user.localX);CHKERRQ(ierr);
    ierr = VecDestroy(user.localF);CHKERRQ(ierr);
    ierr = SNESDestroy(snes);CHKERRQ(ierr);
    ierr = DADestroy(user.da);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}/* --------------------  Form initial approximation ----------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess1"
PetscErrorCode FormInitialGuess1(AppCtx *user,Vec X)
{
  PetscInt       i,j,row,mx,my,xs,ys,xm,ym,Xm,Ym,Xs,Ys;
  PetscErrorCode ierr;
  PetscReal      one = 1.0,lambda,temp1,temp,hx,hy;
  PetscScalar    *x;
  Vec            localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);

  /* Get ghost points */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);

  /* Compute initial guess */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Xm; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp)); 
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
} /* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
PetscErrorCode FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym;
  PetscReal      two = 2.0,one = 1.0,lambda,hx,hy,hxdhy,hydhx,sc;
  PetscScalar    u,uxx,uyy,*x,*f;
  Vec            localX = user->localX,localF = user->localF; 

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      uxx = (two*u - x[row-1] - x[row+1])*hydhx;
      uyy = (two*u - x[row-Xm] - x[row+Xm])*hxdhy;
      f[row] = uxx + uyy - sc*PetscExpScalar(u);
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm);CHKERRQ(ierr);
  return 0; 
} /* --------------------  Evaluate Jacobian F'(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian1"
PetscErrorCode FormJacobian1(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  Mat            jac = *J;
  PetscErrorCode ierr;
  PetscInt       i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5];
  PetscInt       nloc,*ltog,grow;
  PetscScalar    two = 2.0,one = 1.0,lambda,v[5],hx,hy,hxdhy,hydhx,sc,*x;
  Vec            localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        ierr = MatSetValues(jac,1,&grow,1,&grow,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      v[0] = -hxdhy; col[0] = ltog[row - Xm];
      v[1] = -hydhx; col[1] = ltog[row - 1];
      v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = grow;
      v[3] = -hydhx; col[3] = ltog[row + 1];
      v[4] = -hxdhy; col[4] = ltog[row + Xm];
      ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
