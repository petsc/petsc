
static char help[] = ".\n";

/*

        u_t =  u_xx + R(u)

      Where u(t,x,i)    for i=0, .... N-1 where i+1 represents the void size

      ex9.c is the 2d version of this code
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined data structures and routines
*/

/* AppCtx */
typedef struct {
  PetscInt N;               /* number of dofs */
} AppCtx;

extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode InitialConditions(DM,Vec);
extern PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);


int main(int argc,char **argv)
{
  TS             ts;                  /* nonlinear solver */
  Vec            U;                   /* solution, residual vectors */
  Mat            J;                   /* Jacobian matrix */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         user;
  PetscInt       i;
  char           Name[16];

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  user.N = 1;
  ierr   = PetscOptionsGetInt(NULL,NULL,"-N",&user.N,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR,-8,user.N,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  for (i=0; i<user.N; i++) {
    ierr = PetscSNPrintf(Name,16,"Void size %d",(int)(i+1));
    ierr = DMDASetFieldName(da,i,Name);CHKERRQ(ierr);
  }

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&U);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,IJacobian,&user);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,U);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   IFunction - Evaluates nonlinear function, F(U).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,c,Mx,xs,xm,N;
  PetscReal      hx,sx,x;
  PetscScalar    uxx;
  PetscScalar    **u,**f,**udot;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&N,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

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
  ierr = DMDAVecGetArrayDOF(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;

    /*  diffusion term */
    for (c=0; c<N; c++) {
      uxx     = (-2.0*u[i][c] + u[i-1][c] + u[i+1][c])*sx;
      f[i][c] = udot[i][c] - uxx;
    }

    /* reaction terms */

    for (c=0; c<N/3; c++) {
      f[i][c]   +=  500*u[i][c]*u[i][c] + 500*u[i][c]*u[i][c+1];
      f[i][c+1] += -500*u[i][c]*u[i][c] + 500*u[i][c]*u[i][c+1];
      f[i][c+2] -=                        500*u[i][c]*u[i][c+1];
    }


    /* forcing term */

    f[i][0] -= 5*PetscExpScalar((1.0 - x)*(1.0 - x));

  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayDOF(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,c,Mx,xs,xm,nc;
  DM             da;
  MatStencil     col[3],row;
  PetscScalar    vals[3],hx,sx;
  AppCtx         *user = (AppCtx*)ctx;
  PetscInt       N     = user->N;
  PetscScalar    **u;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  ierr = DMDAVecGetArrayDOF(da,U,&u);CHKERRQ(ierr);

  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    for (c=0; c<N; c++) {
      nc        = 0;
      row.c     = c; row.i = i;
      col[nc].c = c; col[nc].i = i-1; vals[nc++] = -sx;
      col[nc].c = c; col[nc].i = i;   vals[nc++] = 2.0*sx + a;
      col[nc].c = c; col[nc].i = i+1; vals[nc++] = -sx;
      ierr      = MatSetValuesStencil(Jpre,1,&row,nc,col,vals,ADD_VALUES);CHKERRQ(ierr);
    }

    for (c=0; c<N/3; c++) {
      nc        = 0;
      row.c     = c;   row.i = i;
      col[nc].c = c;   col[nc].i = i; vals[nc++] = 1000*u[i][c] + 500*u[i][c+1];
      col[nc].c = c+1; col[nc].i = i; vals[nc++] =  500*u[i][c];
      ierr      = MatSetValuesStencil(Jpre,1,&row,nc,col,vals,ADD_VALUES);CHKERRQ(ierr);

      nc        = 0;
      row.c     = c+1; row.i = i;
      col[nc].c = c;   col[nc].i = i; vals[nc++] = -1000*u[i][c] + 500*u[i][c+1];
      col[nc].c = c+1; col[nc].i = i; vals[nc++] =   500*u[i][c];
      ierr      = MatSetValuesStencil(Jpre,1,&row,nc,col,vals,ADD_VALUES);CHKERRQ(ierr);

      nc        = 0;
      row.c     = c+2; row.i = i;
      col[nc].c = c;   col[nc].i = i; vals[nc++] =  -500*u[i][c+1];
      col[nc].c = c+1; col[nc].i = i; vals[nc++] =  -500*u[i][c];
      ierr      = MatSetValuesStencil(Jpre,1,&row,nc,col,vals,ADD_VALUES);CHKERRQ(ierr);

    }
  }
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayDOF(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,c,xs,xm,Mx,N;
  PetscScalar    **u;
  PetscReal      hx,x;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&N,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)(Mx-1);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayDOF(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    for (c=0; c<N; c++) u[i][c] = 0.0; /*PetscCosScalar(PETSC_PI*x);*/
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayDOF(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


