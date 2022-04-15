static char help[] = "Time-dependent PDE in 2d for calculating joint PDF. \n";
/*
   p_t = -x_t*p_x -y_t*p_y + f(t)*p_yy
   xmin < x < xmax, ymin < y < ymax;
   x_t = (y - ws)  y_t = (ws/2H)*(Pm - Pmax*sin(x))

   Boundary conditions: -bc_type 0 => Zero dirichlet boundary
                        -bc_type 1 => Steady state boundary condition
   Steady state boundary condition found by setting p_t = 0
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscScalar ws;   /* Synchronous speed */
  PetscScalar H;    /* Inertia constant */
  PetscScalar D;    /* Damping constant */
  PetscScalar Pmax; /* Maximum power output of generator */
  PetscScalar PM_min; /* Mean mechanical power input */
  PetscScalar lambda; /* correlation time */
  PetscScalar q;      /* noise strength */
  PetscScalar mux;    /* Initial average angle */
  PetscScalar sigmax; /* Standard deviation of initial angle */
  PetscScalar muy;    /* Average speed */
  PetscScalar sigmay; /* standard deviation of initial speed */
  PetscScalar rho;    /* Cross-correlation coefficient */
  PetscScalar t0;     /* Initial time */
  PetscScalar tmax;   /* Final time */
  PetscScalar xmin;   /* left boundary of angle */
  PetscScalar xmax;   /* right boundary of angle */
  PetscScalar ymin;   /* bottom boundary of speed */
  PetscScalar ymax;   /* top boundary of speed */
  PetscScalar dx;     /* x step size */
  PetscScalar dy;     /* y step size */
  PetscInt    bc; /* Boundary conditions */
  PetscScalar disper_coe; /* Dispersion coefficient */
  DM          da;
} AppCtx;

PetscErrorCode Parameter_settings(AppCtx*);
PetscErrorCode ini_bou(Vec,AppCtx*);
PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PetscErrorCode PostStep(TS);

int main(int argc, char **argv)
{
  Vec            x;  /* Solution vector */
  TS             ts;   /* Time-stepping context */
  AppCtx         user; /* Application context */
  Mat            J;
  PetscViewer    viewer;

  PetscCall(PetscInitialize(&argc,&argv,"petscopt_ex6", help));
  /* Get physics and time parameters */
  PetscCall(Parameter_settings(&user));
  /* Create a 2D DA with dof = 1 */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.da));
  PetscCall(DMSetFromOptions(user.da));
  PetscCall(DMSetUp(user.da));
  /* Set x and y coordinates */
  PetscCall(DMDASetUniformCoordinates(user.da,user.xmin,user.xmax,user.ymin,user.ymax,0.0,1.0));

  /* Get global vector x from DM  */
  PetscCall(DMCreateGlobalVector(user.da,&x));

  PetscCall(ini_bou(x,&user));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ini_x",FILE_MODE_WRITE,&viewer));
  PetscCall(VecView(x,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Get Jacobian matrix structure from the da */
  PetscCall(DMSetMatType(user.da,MATAIJ));
  PetscCall(DMCreateMatrix(user.da,&J));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetIFunction(ts,NULL,IFunction,&user));
  PetscCall(TSSetIJacobian(ts,J,J,IJacobian,&user));
  PetscCall(TSSetApplicationContext(ts,&user));
  PetscCall(TSSetMaxTime(ts,user.tmax));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTime(ts,user.t0));
  PetscCall(TSSetTimeStep(ts,.005));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetPostStep(ts,PostStep));
  PetscCall(TSSolve(ts,x));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fin_x",FILE_MODE_WRITE,&viewer));
  PetscCall(VecView(x,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&J));
  PetscCall(DMDestroy(&user.da));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode PostStep(TS ts)
{
  Vec            X;
  AppCtx         *user;
  PetscScalar    sum;
  PetscReal      t;

  PetscFunctionBegin;
  PetscCall(TSGetApplicationContext(ts,&user));
  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&X));
  PetscCall(VecSum(X,&sum));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"sum(p)*dw*dtheta at t = %3.2f = %3.6f\n",(double)t,(double)(sum*user->dx*user->dy)));
  PetscFunctionReturn(0);
}

PetscErrorCode ini_bou(Vec X,AppCtx* user)
{
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p;
  Vec            gc;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  PetscScalar    xi,yi;
  PetscScalar    sigmax=user->sigmax,sigmay=user->sigmay;
  PetscScalar    rho   =user->rho;
  PetscScalar    mux   =user->mux,muy=user->muy;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  user->dx = (user->xmax - user->xmin)/(M-1); user->dy = (user->ymax - user->ymin)/(N-1);
  PetscCall(DMGetCoordinateDM(user->da,&cda));
  PetscCall(DMGetCoordinates(user->da,&gc));
  PetscCall(DMDAVecGetArray(cda,gc,&coors));
  PetscCall(DMDAVecGetArray(user->da,X,&p));
  PetscCall(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      xi = coors[j][i].x; yi = coors[j][i].y;
      if (i == 0 || j == 0 || i == M-1 || j == N-1) p[j][i] = 0.0;
      else p[j][i] = (0.5/(PETSC_PI*sigmax*sigmay*PetscSqrtScalar(1.0-rho*rho)))*PetscExpScalar(-0.5/(1-rho*rho)*(PetscPowScalar((xi-mux)/sigmax,2) + PetscPowScalar((yi-muy)/sigmay,2) - 2*rho*(xi-mux)*(yi-muy)/(sigmax*sigmay)));
    }
  }
  /*  p[N/2+N%2][M/2+M%2] = 1/(user->dx*user->dy); */

  PetscCall(DMDAVecRestoreArray(cda,gc,&coors));
  PetscCall(DMDAVecRestoreArray(user->da,X,&p));
  PetscFunctionReturn(0);
}

/* First advection term */
PetscErrorCode adv1(PetscScalar **p,PetscScalar y,PetscInt i,PetscInt j,PetscInt M,PetscScalar *p1,AppCtx *user)
{
  PetscScalar f;
  /*  PetscScalar v1,v2,v3,v4,v5,s1,s2,s3; */
  PetscFunctionBegin;
  /*  if (i > 2 && i < M-2) {
    v1 = (y-user->ws)*(p[j][i-2] - p[j][i-3])/user->dx;
    v2 = (y-user->ws)*(p[j][i-1] - p[j][i-2])/user->dx;
    v3 = (y-user->ws)*(p[j][i] - p[j][i-1])/user->dx;
    v4 = (y-user->ws)*(p[j][i+1] - p[j][i])/user->dx;
    v5 = (y-user->ws)*(p[j][i+1] - p[j][i+2])/user->dx;

    s1 = v1/3.0 - (7.0/6.0)*v2 + (11.0/6.0)*v3;
    s2 =-v2/6.0 + (5.0/6.0)*v3 + (1.0/3.0)*v4;
    s3 = v3/3.0 + (5.0/6.0)*v4 - (1.0/6.0)*v5;

    *p1 = 0.1*s1 + 0.6*s2 + 0.3*s3;
    } else *p1 = 0.0; */
  f   =  (y - user->ws);
  *p1 = f*(p[j][i+1] - p[j][i-1])/(2*user->dx);
  PetscFunctionReturn(0);
}

/* Second advection term */
PetscErrorCode adv2(PetscScalar **p,PetscScalar x,PetscInt i,PetscInt j,PetscInt N,PetscScalar *p2,AppCtx *user)
{
  PetscScalar f;
  /*  PetscScalar v1,v2,v3,v4,v5,s1,s2,s3; */
  PetscFunctionBegin;
  /*  if (j > 2 && j < N-2) {
    v1 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j-2][i] - p[j-3][i])/user->dy;
    v2 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j-1][i] - p[j-2][i])/user->dy;
    v3 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j][i] - p[j-1][i])/user->dy;
    v4 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j+1][i] - p[j][i])/user->dy;
    v5 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j+2][i] - p[j+1][i])/user->dy;

    s1 = v1/3.0 - (7.0/6.0)*v2 + (11.0/6.0)*v3;
    s2 =-v2/6.0 + (5.0/6.0)*v3 + (1.0/3.0)*v4;
    s3 = v3/3.0 + (5.0/6.0)*v4 - (1.0/6.0)*v5;

    *p2 = 0.1*s1 + 0.6*s2 + 0.3*s3;
    } else *p2 = 0.0; */
  f   = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(x));
  *p2 = f*(p[j+1][i] - p[j-1][i])/(2*user->dy);
  PetscFunctionReturn(0);
}

/* Diffusion term */
PetscErrorCode diffuse(PetscScalar **p,PetscInt i,PetscInt j,PetscReal t,PetscScalar *p_diff,AppCtx * user)
{
  PetscFunctionBeginUser;

  *p_diff = user->disper_coe*((p[j-1][i] - 2*p[j][i] + p[j+1][i])/(user->dy*user->dy));
  PetscFunctionReturn(0);
}

PetscErrorCode BoundaryConditions(PetscScalar **p,DMDACoor2d **coors,PetscInt i,PetscInt j,PetscInt M, PetscInt N,PetscScalar **f,AppCtx *user)
{
  PetscScalar fwc,fthetac;
  PetscScalar w=coors[j][i].y,theta=coors[j][i].x;

  PetscFunctionBeginUser;
  if (user->bc == 0) { /* Natural boundary condition */
    f[j][i] = p[j][i];
  } else { /* Steady state boundary condition */
    fthetac = user->ws/(2*user->H)*(user->PM_min - user->Pmax*PetscSinScalar(theta));
    fwc = (w*w/2.0 - user->ws*w);
    if (i == 0 && j == 0) { /* left bottom corner */
      f[j][i] = fwc*(p[j][i+1] - p[j][i])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j][i])/user->dy;
    } else if (i == 0 && j == N-1) { /* right bottom corner */
      f[j][i] = fwc*(p[j][i+1] - p[j][i])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j][i] - p[j-1][i])/user->dy;
    } else if (i == M-1 && j == 0) { /* left top corner */
      f[j][i] = fwc*(p[j][i] - p[j][i-1])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j][i])/user->dy;
    } else if (i == M-1 && j == N-1) { /* right top corner */
      f[j][i] = fwc*(p[j][i] - p[j][i-1])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j][i] - p[j-1][i])/user->dy;
    } else if (i == 0) { /* Bottom edge */
      f[j][i] = fwc*(p[j][i+1] - p[j][i])/(user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j-1][i])/(2*user->dy);
    } else if (i == M-1) { /* Top edge */
      f[j][i] = fwc*(p[j][i] - p[j][i-1])/(user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j-1][i])/(2*user->dy);
    } else if (j == 0) { /* Left edge */
      f[j][i] = fwc*(p[j][i+1] - p[j][i-1])/(2*user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j][i])/(user->dy);
    } else if (j == N-1) { /* Right edge */
      f[j][i] = fwc*(p[j][i+1] - p[j][i-1])/(2*user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j][i] - p[j-1][i])/(user->dy);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  AppCtx         *user=(AppCtx*)ctx;
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p,**f,**pdot;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  Vec            localX,gc,localXdot;
  PetscScalar    p_adv1,p_adv2,p_diff;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMGetCoordinateDM(user->da,&cda));
  PetscCall(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));

  PetscCall(DMGetLocalVector(user->da,&localX));
  PetscCall(DMGetLocalVector(user->da,&localXdot));

  PetscCall(DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalBegin(user->da,Xdot,INSERT_VALUES,localXdot));
  PetscCall(DMGlobalToLocalEnd(user->da,Xdot,INSERT_VALUES,localXdot));

  PetscCall(DMGetCoordinatesLocal(user->da,&gc));

  PetscCall(DMDAVecGetArrayRead(cda,gc,&coors));
  PetscCall(DMDAVecGetArrayRead(user->da,localX,&p));
  PetscCall(DMDAVecGetArrayRead(user->da,localXdot,&pdot));
  PetscCall(DMDAVecGetArray(user->da,F,&f));

  user->disper_coe = PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      if (i == 0 || j == 0 || i == M-1 || j == N-1) {
        PetscCall(BoundaryConditions(p,coors,i,j,M,N,f,user));
      } else {
        PetscCall(adv1(p,coors[j][i].y,i,j,M,&p_adv1,user));
        PetscCall(adv2(p,coors[j][i].x,i,j,N,&p_adv2,user));
        PetscCall(diffuse(p,i,j,t,&p_diff,user));
        f[j][i] = -p_adv1 - p_adv2 + p_diff - pdot[j][i];
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(user->da,localX,&p));
  PetscCall(DMDAVecRestoreArrayRead(user->da,localX,&pdot));
  PetscCall(DMRestoreLocalVector(user->da,&localX));
  PetscCall(DMRestoreLocalVector(user->da,&localXdot));
  PetscCall(DMDAVecRestoreArray(user->da,F,&f));
  PetscCall(DMDAVecRestoreArrayRead(cda,gc,&coors));

  PetscFunctionReturn(0);
}

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  AppCtx         *user=(AppCtx*)ctx;
  DM             cda;
  DMDACoor2d     **coors;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  Vec            gc;
  PetscScalar    val[5],xi,yi;
  MatStencil     row,col[5];
  PetscScalar    c1,c3,c5;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMGetCoordinateDM(user->da,&cda));
  PetscCall(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));

  PetscCall(DMGetCoordinatesLocal(user->da,&gc));
  PetscCall(DMDAVecGetArrayRead(cda,gc,&coors));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      PetscInt nc = 0;
      xi = coors[j][i].x; yi = coors[j][i].y;
      row.i = i; row.j = j;
      if (i == 0 || j == 0 || i == M-1 || j == N-1) {
        if (user->bc == 0) {
          col[nc].i = i; col[nc].j = j; val[nc++] = 1.0;
        } else {
          PetscScalar fthetac,fwc;
          fthetac = user->ws/(2*user->H)*(user->PM_min - user->Pmax*PetscSinScalar(xi));
          fwc     = (yi*yi/2.0 - user->ws*yi);
          if (i==0 && j==0) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -fwc/user->dx + fthetac + user->disper_coe/user->dy;
          } else if (i==0 && j == N-1) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/user->dx;
            col[nc].i = i;   col[nc].j = j-1; val[nc++] = user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -fwc/user->dx + fthetac - user->disper_coe/user->dy;
          } else if (i== M-1 && j == 0) {
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] =  fwc/user->dx + fthetac + user->disper_coe/user->dy;
          } else if (i == M-1 && j == N-1) {
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/user->dx;
            col[nc].i = i;   col[nc].j = j-1; val[nc++] =  user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] =  fwc/user->dx + fthetac - user->disper_coe/user->dy;
          } else if (i==0) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j-1; val[nc++] =  user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -fwc/user->dx + fthetac;
          } else if (i == M-1) {
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j-1; val[nc++] =  user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j;   val[nc++] = fwc/user->dx + fthetac;
          } else if (j==0) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/(2*user->dx);
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/(2*user->dx);
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = user->disper_coe/user->dy + fthetac;
          } else if (j == N-1) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/(2*user->dx);
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/(2*user->dx);
            col[nc].i = i;   col[nc].j = j-1; val[nc++] = user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -user->disper_coe/user->dy + fthetac;
          }
        }
      } else {
        c1        = (yi-user->ws)/(2*user->dx);
        c3        = (user->ws/(2.0*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(xi))/(2*user->dy);
        c5        = (PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda)))/(user->dy*user->dy);
        col[nc].i = i-1; col[nc].j = j;   val[nc++] = c1;
        col[nc].i = i+1; col[nc].j = j;   val[nc++] = -c1;
        col[nc].i = i;   col[nc].j = j-1; val[nc++] = c3 + c5;
        col[nc].i = i;   col[nc].j = j+1; val[nc++] = -c3 + c5;
        col[nc].i = i;   col[nc].j = j;   val[nc++] = -2*c5 -a;
      }
      PetscCall(MatSetValuesStencil(Jpre,1,&row,nc,col,val,INSERT_VALUES));
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(cda,gc,&coors));

  PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Parameter_settings(AppCtx *user)
{
  PetscBool      flg;

  PetscFunctionBeginUser;

  /* Set default parameters */
  user->ws     = 1.0;
  user->H      = 5.0;  user->Pmax   = 2.1;
  user->PM_min = 1.0;  user->lambda = 0.1;
  user->q      = 1.0;  user->mux    = PetscAsinScalar(user->PM_min/user->Pmax);
  user->sigmax = 0.1;
  user->sigmay = 0.1;  user->rho  = 0.0;
  user->t0     = 0.0;  user->tmax = 2.0;
  user->xmin   = -1.0; user->xmax = 10.0;
  user->ymin   = -1.0; user->ymax = 10.0;
  user->bc     = 0;

  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-ws",&user->ws,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Inertia",&user->H,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-Pmax",&user->Pmax,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-PM_min",&user->PM_min,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-lambda",&user->lambda,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-q",&user->q,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-mux",&user->mux,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sigmax",&user->sigmax,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-muy",&user->muy,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sigmay",&user->sigmay,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-rho",&user->rho,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-t0",&user->t0,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-tmax",&user->tmax,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-xmin",&user->xmin,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-xmax",&user->xmax,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-ymin",&user->ymin,&flg));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-ymax",&user->ymax,&flg));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-bc_type",&user->bc,&flg));
  user->muy = user->ws;
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -nox -ts_max_steps 2
      localrunfiles: petscopt_ex6
      timeoutfactor: 4

TEST*/
