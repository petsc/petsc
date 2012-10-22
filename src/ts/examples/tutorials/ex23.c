static char help[] = "Cahn-Hilliard-2d problem for constant mobility and triangular elements.\n\
Runtime options include:\n\
-xmin <xmin>\n\
-xmax <xmax>\n\
-ymin <ymin>\n\
-gamma <gamma>\n\
-theta_c <theta_c>\n\
-implicit <0,1> treat theta_c*M_0*u explicitly/implicitly\n\n";

/*
    Run with for example: -pc_type mg -pc_mg_galerkin -T .01 -da_grid_x 65 -da_grid_y 65 -pc_mg_levels 4 -ksp_type fgmres -snes_atol 1.e-14 -mat_no_inode
 */

#include "petscts.h"
#include "petscdmda.h"

typedef struct{
  PetscReal   dt,T; /* Time step and end time */
  DM          da;
  Mat         M; /* Mass matrix */
  Mat         S; /* stiffness matrix */
  Mat         M_0;
  Vec         q,u,work1;
  PetscScalar gamma,theta_c; /* physics parameters */
  PetscReal   xmin,xmax,ymin,ymax;
  PetscBool   tsmonitor;
  PetscInt    implicit; /* Evaluate theta_c*Mo*u impliicitly or explicitly */
}AppCtx;

PetscErrorCode GetParams(AppCtx*);
PetscErrorCode SetVariableBounds(DM,Vec,Vec);
PetscErrorCode SetUpMatrices(AppCtx*);
PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode SetInitialGuess(Vec,AppCtx*);
PetscErrorCode Update_q(TS);
PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x,r;  /* Solution and residual vectors */
  TS             ts;   /* Timestepping object */
  AppCtx         user; /* Application context */
  Vec            xl,xu; /* Upper and lower bounds on variables */
  Mat            J;

  PetscInitialize(&argc,&argv, (char*)0, help);

  /* Get physics and time parameters */
  ierr = GetParams(&user);CHKERRQ(ierr);
  /* Create a 2D DA with dof = 2 */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,2,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(ierr);
  /* Set Element type (triangular) */
  ierr = DMDASetElementType(user.da,DMDA_ELEMENT_P1);CHKERRQ(ierr);

  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(user.da,user.xmin,user.xmax,user.ymin,user.ymax,0.0,1.0);CHKERRQ(ierr);

  /* Get global vector x from DM and duplicate vectors r,xl,xu */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu);CHKERRQ(ierr);
  if (!user.implicit) {
    ierr = VecDuplicate(x,&user.q);CHKERRQ(ierr);
  }

  /* Get mass,stiffness, and jacobian matrix structure from the da */
  ierr = DMCreateMatrix(user.da,MATAIJ,&user.M);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.da,MATAIJ,&user.S);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.da,MATAIJ,&J);CHKERRQ(ierr);
  /* Form the mass,stiffness matrices and matrix M_0 */
  ierr = SetUpMatrices(&user);CHKERRQ(ierr);

  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);


  /* Set Function evaluation and jacobian evaluation routines */
  ierr = TSSetIFunction(ts,r,FormIFunction,(void*)&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  //  ierr = TSThetaSetTheta(ts,1.0);CHKERRQ(ierr);/* Explicit Euler */
  ierr = TSSetDuration(ts,PETSC_DEFAULT,user.T);CHKERRQ(ierr);
  /* Set lower and upper bound vectors */
  ierr = SetVariableBounds(user.da,xl,xu);CHKERRQ(ierr);
  ierr = TSVISetVariableBounds(ts,xl,xu);CHKERRQ(ierr);

  ierr = SetInitialGuess(x,&user);CHKERRQ(ierr);

  if (!user.implicit) {
    ierr = TSSetApplicationContext(ts,&user);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
    ierr = Update_q(ts);CHKERRQ(ierr);
    ierr = TSSetPostStep(ts,Update_q);
  }

  if (user.tsmonitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = TSSetInitialTimeStep(ts,0.0,user.dt);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);


  /* Run the timestepping solver */
  ierr = TSSolve(ts,x,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&xl);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  if (!user.implicit) {
    ierr = VecDestroy(&user.q);CHKERRQ(ierr);
    ierr = VecDestroy(&user.u);CHKERRQ(ierr);
    ierr = VecDestroy(&user.work1);CHKERRQ(ierr);
    ierr = MatDestroy(&user.M_0);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&user.M);CHKERRQ(ierr);
  ierr = MatDestroy(&user.S);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Update_q"
PetscErrorCode Update_q(TS ts)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  Vec            x;
  PetscScalar    *q_arr,*w_arr;
  PetscInt       i,n;
  PetscScalar    scale;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&x);CHKERRQ(ierr);
  ierr = VecStrideGather(x,1,user->u,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatMult(user->M_0,user->u,user->work1);CHKERRQ(ierr);
  scale = -(1.0 - user->implicit)*user->theta_c;
  ierr = VecScale(user->work1,scale);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->u,&n);CHKERRQ(ierr);
  ierr = VecGetArray(user->q,&q_arr);CHKERRQ(ierr);
  ierr = VecGetArray(user->work1,&w_arr);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    q_arr[2*i+1] = w_arr[i];
  }
  ierr = VecRestoreArray(user->q,&q_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->work1,&w_arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialGuess"
PetscErrorCode SetInitialGuess(Vec X,AppCtx* user)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt        n,i;
  PetscRandom     rand;
  PetscScalar     value;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  /* Set initial guess, only set value for 2nd dof */
  for (i=0;i<n/2;i++) {
    ierr = PetscRandomGetValue(rand,&value);CHKERRQ(ierr);
    x[2*i+1] = -0.4 + 0.05*value*(value - 0.5);
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static void Gausspoints(PetscScalar *xx,PetscScalar *yy,PetscScalar *w,PetscScalar *x,PetscScalar *y)
{

  xx[0] = 2.0/3.0*x[0] + 1.0/6.0*x[1] + 1.0/6.0*x[2];
  xx[1] = 1.0/6.0*x[0] + 2.0/3.0*x[1] + 1.0/6.0*x[2];
  xx[2] = 1.0/6.0*x[0] + 1.0/6.0*x[1] + 2.0/3.0*x[2];

  yy[0] = 2.0/3.0*y[0] + 1.0/6.0*y[1] + 1.0/6.0*y[2];
  yy[1] = 1.0/6.0*y[0] + 2.0/3.0*y[1] + 1.0/6.0*y[2];
  yy[2] = 1.0/6.0*y[0] + 1.0/6.0*y[1] + 2.0/3.0*y[2];

  *w = PetscAbsScalar(x[0]*(y[2]-y[1]) + x[2]*(y[1]-y[0]) + x[1]*(y[0]-y[2]))/6.0;

}

static void ShapefunctionsT3(PetscScalar *phi,PetscScalar phider[][2],PetscScalar xx,PetscScalar yy,PetscScalar *x,PetscScalar *y)
{
  PetscScalar area,a1,a2,a3,b1,b2,b3,c1,c2,c3,pp;

  /* Area of the triangle */
  area = 1.0/2.0*PetscAbsScalar(x[0]*(y[2]-y[1]) + x[2]*(y[1]-y[0]) + x[1]*(y[0]-y[2]));

  a1 = x[1]*y[2]-x[2]*y[1]; a2 = x[2]*y[0]-x[0]*y[2]; a3 = x[0]*y[1]-x[1]*y[0];
  b1 = y[1]-y[2]; b2 = y[2]-y[0]; b3 = y[0]-y[1];
  c1 = x[2]-x[1]; c2 = x[0]-x[2]; c3 = x[1]-x[0];
  pp = 1.0/(2.0*area);

  /* shape functions */
  phi[0] = pp*(a1 + b1*xx + c1*yy);
  phi[1] = pp*(a2 + b2*xx + c2*yy);
  phi[2] = pp*(a3 + b3*xx + c3*yy);

  /* shape functions derivatives */
  phider[0][0] = pp*b1; phider[0][1] = pp*c1;
  phider[1][0] = pp*b2; phider[1][1] = pp*c2;
  phider[2][0] = pp*b3; phider[2][1] = pp*c3;

}

#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
PetscErrorCode FormIFunction(TS ts,PetscReal t, Vec X,Vec Xdot,Vec F,void* ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatMult(user->M,Xdot,F);CHKERRQ(ierr);
  ierr = MatMultAdd(user->S,X,F,F);CHKERRQ(ierr);
  if (!user->implicit) {
    ierr = VecAXPY(F,1.0,user->q);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat *J,Mat *B,MatStructure *flg,void *ctx)
{
  PetscErrorCode   ierr;
  AppCtx           *user=(AppCtx*)ctx;
  static PetscBool copied = PETSC_FALSE;

  PetscFunctionBegin;
  /* for active set method the matrix does not get changed, so do not need to copy each time,
     if the active set remains the same for several solves the preconditioner does not need to be rebuilt*/
  *flg = SAME_PRECONDITIONER;
  if (!copied) {
    ierr = MatCopy(user->S,*J,*flg);CHKERRQ(ierr);
    ierr = MatAXPY(*J,a,user->M,*flg);CHKERRQ(ierr);
    copied = PETSC_TRUE;
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //  ierr = MatView(*J,0);CHKERRQ(ierr);
  // SETERRQ(PETSC_COMM_WORLD,1,"Stopped here\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetVariableBounds"
PetscErrorCode SetVariableBounds(DM da,Vec xl,Vec xu)
{
  PetscErrorCode ierr;
  PetscScalar    ***l,***u;
  PetscInt       xs,xm,ys,ym;
  PetscInt       j,i;

  PetscFunctionBegin;
  ierr = DMDAVecGetArrayDOF(da,xl,&l);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,xu,&u);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  for (j=ys; j < ys+ym; j++) {
    for (i=xs; i < xs+xm;i++) {
      l[j][i][0] = -SNES_VI_INF;
      l[j][i][1] = -1.0;
      u[j][i][0] = SNES_VI_INF;
      u[j][i][1] = 1.0;
    }
  }

  ierr = DMDAVecRestoreArrayDOF(da,xl,&l);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,xu,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetParams"
PetscErrorCode GetParams(AppCtx* user)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;

  /* Set default parameters */
  user->tsmonitor = PETSC_FALSE;
  user->xmin = 0.0; user->xmax = 1.0;
  user->ymin = 0.0; user->ymax = 1.0;
  user->T = 0.2;    user->dt = 0.0001;
  user->gamma = 3.2E-4; user->theta_c = 1;
  user->implicit = 0;

  ierr = PetscOptionsGetBool(PETSC_NULL,"-monitor",&user->tsmonitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmin",&user->xmin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-xmax",&user->xmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-ymin",&user->ymin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-ymax",&user->ymax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-gamma",&user->gamma,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-theta_c",&user->theta_c,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-implicit",&user->implicit,&flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUpMatrices"
PetscErrorCode SetUpMatrices(AppCtx* user)
{
  PetscErrorCode    ierr;
  PetscInt          nele,nen,i;
  const PetscInt    *ele;
  Vec               coords;
  const PetscScalar *_coords;
  PetscScalar       x[3],y[3],xx[3],yy[3],w;
  PetscInt          idx[3];
  PetscScalar       phi[3],phider[3][2];
  PetscScalar       eM_0[3][3],eM_2[3][3];
  Mat               M=user->M;
  Mat               S=user->S;
  PetscScalar       gamma=user->gamma,theta_c=user->theta_c;
  PetscInt          implicit = user->implicit;

  PetscFunctionBegin;
  /* Get ghosted coordinates */
  ierr = DMGetCoordinatesLocal(user->da,&coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords,&_coords);CHKERRQ(ierr);

  /* Get local element info */
  ierr = DMDAGetElements(user->da,&nele,&nen,&ele);CHKERRQ(ierr);
  for (i=0;i < nele;i++) {
    idx[0] = ele[3*i]; idx[1] = ele[3*i+1]; idx[2] = ele[3*i+2];
    x[0] = _coords[2*idx[0]]; y[0] = _coords[2*idx[0]+1];
    x[1] = _coords[2*idx[1]]; y[1] = _coords[2*idx[1]+1];
    x[2] = _coords[2*idx[2]]; y[2] = _coords[2*idx[2]+1];

    ierr = PetscMemzero(xx,3*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(yy,3*sizeof(PetscScalar));CHKERRQ(ierr);
    Gausspoints(xx,yy,&w,x,y);

    eM_0[0][0]=eM_0[0][1]=eM_0[0][2]=0.0;
    eM_0[1][0]=eM_0[1][1]=eM_0[1][2]=0.0;
    eM_0[2][0]=eM_0[2][1]=eM_0[2][2]=0.0;
    eM_2[0][0]=eM_2[0][1]=eM_2[0][2]=0.0;
    eM_2[1][0]=eM_2[1][1]=eM_2[1][2]=0.0;
    eM_2[2][0]=eM_2[2][1]=eM_2[2][2]=0.0;

    PetscInt m;
    for (m=0;m<3;m++) {
      ierr = PetscMemzero(phi,3*sizeof(PetscScalar));CHKERRQ(ierr);
      phider[0][0]=phider[0][1]=0.0;
      phider[1][0]=phider[1][1]=0.0;
      phider[2][0]=phider[2][1]=0.0;

      ShapefunctionsT3(phi,phider,xx[m],yy[m],x,y);

      PetscInt j,k;
      for (j=0;j<3;j++) {
	for (k=0;k<3;k++) {
	  eM_0[k][j] += phi[j]*phi[k]*w;
	  eM_2[k][j] += phider[j][0]*phider[k][0]*w + phider[j][1]*phider[k][1]*w;
	}
      }
    }
    PetscInt    row,cols[6],r;
    PetscScalar vals[6];

    for (r=0;r<3;r++) {
      row = 2*idx[r];
      /* Insert values in the mass matrix */
      cols[0] = 2*idx[0]+1;     vals[0] = eM_0[r][0];
      cols[1] = 2*idx[1]+1;     vals[1] = eM_0[r][1];
      cols[2] = 2*idx[2]+1;     vals[2] = eM_0[r][2];

      ierr = MatSetValuesLocal(M,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);

      /* Insert values in the stiffness matrix */
      cols[0] = 2*idx[0];   vals[0] = eM_2[r][0];
      cols[1] = 2*idx[1];   vals[1] = eM_2[r][1];
      cols[2] = 2*idx[2];   vals[2] = eM_2[r][2];

      ierr = MatSetValuesLocal(S,1,&row,3,cols,vals,ADD_VALUES);CHKERRQ(ierr);

      row = 2*idx[r]+1;
      cols[0] = 2*idx[0];     vals[0] = -eM_0[r][0];
      cols[1] = 2*idx[0]+1;   vals[1] = gamma*eM_2[r][0]-implicit*theta_c*eM_0[r][0];
      cols[2] = 2*idx[1];     vals[2] = -eM_0[r][1];
      cols[3] = 2*idx[1]+1;   vals[3] = gamma*eM_2[r][1]-implicit*theta_c*eM_0[r][1];
      cols[4] = 2*idx[2];     vals[4] = -eM_0[r][2];
      cols[5] = 2*idx[2]+1;   vals[5] = gamma*eM_2[r][2]-implicit*theta_c*eM_2[r][2];

      /* Insert values in matrix M for 2nd dof */
      ierr = MatSetValuesLocal(S,1,&row,6,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = DMDARestoreElements(user->da,&nele,&nen,&ele);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coords,&_coords);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (!implicit) {
    /* Create ISs to extract matrix M_0 from M */
    PetscInt n,rstart;
    IS       isrow,iscol;

    ierr = MatGetLocalSize(M,&n,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(M,&rstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,n/2,rstart,2,&isrow);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,n/2,rstart+1,2,&iscol);CHKERRQ(ierr);

    /* Extract M_0 from M */
    ierr = MatGetSubMatrix(M,isrow,iscol,MAT_INITIAL_MATRIX,&user->M_0);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&user->u);CHKERRQ(ierr);
    ierr = VecSetSizes(user->u,n/2,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(user->u);CHKERRQ(ierr);
    ierr = VecDuplicate(user->u,&user->work1);CHKERRQ(ierr);

    ierr = ISDestroy(&isrow);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt steps,PetscReal time,Vec x,void* mctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution vector at t = %5.4f\n",time,steps);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
