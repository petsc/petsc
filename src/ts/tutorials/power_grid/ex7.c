
static char help[] = "Time-dependent PDE in 2d for calculating joint PDF. \n";
/*
   p_t = -x_t*p_x -y_t*p_y + f(t)*p_yy
   xmin < x < xmax, ymin < y < ymax;

   Boundary conditions Neumman using mirror values

   Note that x_t and y_t in the above are given functions of x and y; they are not derivatives of x and y.
   x_t = (y - ws)  y_t = (ws/2H)*(Pm - Pmax*sin(x))

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
  PetscScalar xmin;   /* left boundary of angle */
  PetscScalar xmax;   /* right boundary of angle */
  PetscScalar ymin;   /* bottom boundary of speed */
  PetscScalar ymax;   /* top boundary of speed */
  PetscScalar dx;     /* x step size */
  PetscScalar dy;     /* y step size */
  PetscScalar disper_coe; /* Dispersion coefficient */
  DM          da;
  PetscInt    st_width; /* Stencil width */
  DMBoundaryType bx; /* x boundary type */
  DMBoundaryType by; /* y boundary type */
  PetscBool        nonoiseinitial;
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
  PetscViewer    viewer;

  CHKERRQ(PetscInitialize(&argc,&argv,"petscopt_ex7", help));

  /* Get physics and time parameters */
  CHKERRQ(Parameter_settings(&user));
  /* Create a 2D DA with dof = 1 */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,user.bx,user.by,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,user.st_width,NULL,NULL,&user.da));
  CHKERRQ(DMSetFromOptions(user.da));
  CHKERRQ(DMSetUp(user.da));
  /* Set x and y coordinates */
  CHKERRQ(DMDASetUniformCoordinates(user.da,user.xmin,user.xmax,user.ymin,user.ymax,0,0));
  CHKERRQ(DMDASetCoordinateName(user.da,0,"X - the angle"));
  CHKERRQ(DMDASetCoordinateName(user.da,1,"Y - the speed"));

  /* Get global vector x from DM  */
  CHKERRQ(DMCreateGlobalVector(user.da,&x));

  CHKERRQ(ini_bou(x,&user));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ini_x",FILE_MODE_WRITE,&viewer));
  CHKERRQ(VecView(x,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetDM(ts,user.da));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSARKIMEX));
  CHKERRQ(TSSetIFunction(ts,NULL,IFunction,&user));
  /*  CHKERRQ(TSSetIJacobian(ts,NULL,NULL,IJacobian,&user));  */
  CHKERRQ(TSSetApplicationContext(ts,&user));
  CHKERRQ(TSSetTimeStep(ts,.005));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetPostStep(ts,PostStep));
  CHKERRQ(TSSolve(ts,x));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fin_x",FILE_MODE_WRITE,&viewer));
  CHKERRQ(VecView(x,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(DMDestroy(&user.da));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode PostStep(TS ts)
{
  Vec            X,gc;
  AppCtx         *user;
  PetscScalar    sum = 0,asum;
  PetscReal      t,**p;
  DMDACoor2d     **coors;
  DM             cda;
  PetscInt       i,j,xs,ys,xm,ym;

  PetscFunctionBegin;
  CHKERRQ(TSGetApplicationContext(ts,&user));
  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&X));

  CHKERRQ(DMGetCoordinateDM(user->da,&cda));
  CHKERRQ(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));
  CHKERRQ(DMGetCoordinates(user->da,&gc));
  CHKERRQ(DMDAVecGetArrayRead(cda,gc,&coors));
  CHKERRQ(DMDAVecGetArrayRead(user->da,X,&p));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      if (coors[j][i].y < 5) sum += p[j][i];
    }
  }
  CHKERRQ(DMDAVecRestoreArrayRead(cda,gc,&coors));
  CHKERRQ(DMDAVecRestoreArrayRead(user->da,X,&p));
  CHKERRMPI(MPI_Allreduce(&sum,&asum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ts)));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sum(p)*dw*dtheta at t = %f = %f\n",(double)t,(double)(asum)));
  if (sum  < 1.0e-2) {
    CHKERRQ(TSSetConvergedReason(ts,TS_CONVERGED_USER));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Exiting TS as the integral of PDF is almost zero\n"));
  }
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
  PetscScalar    muy=user->muy,mux;
  PetscMPIInt    rank;
  PetscScalar    sum;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  user->dx = (user->xmax - user->xmin)/(M-1); user->dy = (user->ymax - user->ymin)/(N-1);
  CHKERRQ(DMGetCoordinateDM(user->da,&cda));
  CHKERRQ(DMGetCoordinates(user->da,&gc));
  CHKERRQ(DMDAVecGetArray(cda,gc,&coors));
  CHKERRQ(DMDAVecGetArray(user->da,X,&p));
  CHKERRQ(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));

  /* mux and muy need to be grid points in the x and y-direction otherwise the solution goes unstable
     muy is set by choosing the y domain, no. of grid points along y-direction so that muy is a grid point
     in the y-direction. We only modify mux here
  */
  mux = user->mux = coors[0][M/2+10].x; /* For -pi < x < pi, this should be some angle between 0 and pi/2 */
  if (user->nonoiseinitial) {
    for (i=xs; i < xs+xm; i++) {
      for (j=ys; j < ys+ym; j++) {
        xi = coors[j][i].x; yi = coors[j][i].y;
        if ((xi == mux) && (yi == muy)) {
          p[j][i] = 1.0;
        }
      }
    }
  } else {
    /* Change PM_min accordingly */
    user->PM_min = user->Pmax*PetscSinScalar(mux);
    for (i=xs; i < xs+xm; i++) {
      for (j=ys; j < ys+ym; j++) {
        xi = coors[j][i].x; yi = coors[j][i].y;
        p[j][i] = (0.5/(PETSC_PI*sigmax*sigmay*PetscSqrtScalar(1.0-rho*rho)))*PetscExpScalar(-0.5/(1-rho*rho)*(PetscPowScalar((xi-mux)/sigmax,2) + PetscPowScalar((yi-muy)/sigmay,2) - 2*rho*(xi-mux)*(yi-muy)/(sigmax*sigmay)));
      }
    }
  }
  CHKERRQ(DMDAVecRestoreArray(cda,gc,&coors));
  CHKERRQ(DMDAVecRestoreArray(user->da,X,&p));
  CHKERRQ(VecSum(X,&sum));
  CHKERRQ(VecScale(X,1.0/sum));
  PetscFunctionReturn(0);
}

/* First advection term */
PetscErrorCode adv1(PetscScalar **p,PetscScalar y,PetscInt i,PetscInt j,PetscInt M,PetscScalar *p1,AppCtx *user)
{
  PetscScalar f,fpos,fneg;
  PetscFunctionBegin;
  f   =  (y - user->ws);
  fpos = PetscMax(f,0);
  fneg = PetscMin(f,0);
  if (user->st_width == 1) {
    *p1 = fpos*(p[j][i] - p[j][i-1])/user->dx + fneg*(p[j][i+1] - p[j][i])/user->dx;
  } else if (user->st_width == 2) {
    *p1 = fpos*(3*p[j][i] - 4*p[j][i-1] + p[j][i-2])/(2*user->dx) + fneg*(-p[j][i+2] + 4*p[j][i+1] - 3*p[j][i])/(2*user->dx);
  } else if (user->st_width == 3) {
    *p1 = fpos*(2*p[j][i+1] + 3*p[j][i] - 6*p[j][i-1] + p[j][i-2])/(6*user->dx) + fneg*(-p[j][i+2] + 6*p[j][i+1] - 3*p[j][i] - 2*p[j][i-1])/(6*user->dx);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No support for wider stencils");
  PetscFunctionReturn(0);
}

/* Second advection term */
PetscErrorCode adv2(PetscScalar **p,PetscScalar x,PetscInt i,PetscInt j,PetscInt N,PetscScalar *p2,AppCtx *user)
{
  PetscScalar f,fpos,fneg;
  PetscFunctionBegin;
  f   = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(x));
  fpos = PetscMax(f,0);
  fneg = PetscMin(f,0);
  if (user->st_width == 1) {
    *p2 = fpos*(p[j][i] - p[j-1][i])/user->dy + fneg*(p[j+1][i] - p[j][i])/user->dy;
  } else if (user->st_width ==2) {
    *p2 = fpos*(3*p[j][i] - 4*p[j-1][i] + p[j-2][i])/(2*user->dy) + fneg*(-p[j+2][i] + 4*p[j+1][i] - 3*p[j][i])/(2*user->dy);
  } else if (user->st_width == 3) {
    *p2 = fpos*(2*p[j+1][i] + 3*p[j][i] - 6*p[j-1][i] + p[j-2][i])/(6*user->dy) + fneg*(-p[j+2][i] + 6*p[j+1][i] - 3*p[j][i] - 2*p[j-1][i])/(6*user->dy);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No support for wider stencils");
  PetscFunctionReturn(0);
}

/* Diffusion term */
PetscErrorCode diffuse(PetscScalar **p,PetscInt i,PetscInt j,PetscReal t,PetscScalar *p_diff,AppCtx * user)
{
  PetscFunctionBeginUser;
  if (user->st_width == 1) {
    *p_diff = user->disper_coe*((p[j-1][i] - 2*p[j][i] + p[j+1][i])/(user->dy*user->dy));
  } else if (user->st_width == 2) {
    *p_diff = user->disper_coe*((-p[j-2][i] + 16*p[j-1][i] - 30*p[j][i] + 16*p[j+1][i] - p[j+2][i])/(12.0*user->dy*user->dy));
  } else if (user->st_width == 3) {
    *p_diff = user->disper_coe*((2*p[j-3][i] - 27*p[j-2][i] + 270*p[j-1][i] - 490*p[j][i] + 270*p[j+1][i] - 27*p[j+2][i] + 2*p[j+3][i])/(180.0*user->dy*user->dy));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No support for wider stencils");
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
  CHKERRQ(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  CHKERRQ(DMGetCoordinateDM(user->da,&cda));
  CHKERRQ(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));

  CHKERRQ(DMGetLocalVector(user->da,&localX));
  CHKERRQ(DMGetLocalVector(user->da,&localXdot));

  CHKERRQ(DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalBegin(user->da,Xdot,INSERT_VALUES,localXdot));
  CHKERRQ(DMGlobalToLocalEnd(user->da,Xdot,INSERT_VALUES,localXdot));

  CHKERRQ(DMGetCoordinatesLocal(user->da,&gc));

  CHKERRQ(DMDAVecGetArrayRead(cda,gc,&coors));
  CHKERRQ(DMDAVecGetArrayRead(user->da,localX,&p));
  CHKERRQ(DMDAVecGetArrayRead(user->da,localXdot,&pdot));
  CHKERRQ(DMDAVecGetArray(user->da,F,&f));

  user->disper_coe = PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      CHKERRQ(adv1(p,coors[j][i].y,i,j,M,&p_adv1,user));
      CHKERRQ(adv2(p,coors[j][i].x,i,j,N,&p_adv2,user));
      CHKERRQ(diffuse(p,i,j,t,&p_diff,user));
      f[j][i] = -p_adv1 - p_adv2  + p_diff - pdot[j][i];
    }
  }
  CHKERRQ(DMDAVecRestoreArrayRead(user->da,localX,&p));
  CHKERRQ(DMDAVecRestoreArrayRead(user->da,localX,&pdot));
  CHKERRQ(DMRestoreLocalVector(user->da,&localX));
  CHKERRQ(DMRestoreLocalVector(user->da,&localXdot));
  CHKERRQ(DMDAVecRestoreArray(user->da,F,&f));
  CHKERRQ(DMDAVecRestoreArrayRead(cda,gc,&coors));

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
  PetscScalar    c1,c3,c5,c1pos,c1neg,c3pos,c3neg;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  CHKERRQ(DMGetCoordinateDM(user->da,&cda));
  CHKERRQ(DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0));

  CHKERRQ(DMGetCoordinatesLocal(user->da,&gc));
  CHKERRQ(DMDAVecGetArrayRead(cda,gc,&coors));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      PetscInt nc = 0;
      xi = coors[j][i].x; yi = coors[j][i].y;
      row.i = i; row.j = j;
      c1        = (yi-user->ws)/user->dx;
      c1pos    = PetscMax(c1,0);
      c1neg    = PetscMin(c1,0);
      c3        = (user->ws/(2.0*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(xi))/user->dy;
      c3pos    = PetscMax(c3,0);
      c3neg    = PetscMin(c3,0);
      c5        = (PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda)))/(user->dy*user->dy);
      col[nc].i = i-1; col[nc].j = j;   val[nc++] = c1pos;
      col[nc].i = i+1; col[nc].j = j;   val[nc++] = -c1neg;
      col[nc].i = i;   col[nc].j = j-1; val[nc++] = c3pos + c5;
      col[nc].i = i;   col[nc].j = j+1; val[nc++] = -c3neg + c5;
      col[nc].i = i;   col[nc].j = j;   val[nc++] = -c1pos + c1neg -c3pos + c3neg -2*c5 -a;
      CHKERRQ(MatSetValuesStencil(Jpre,1,&row,nc,col,val,INSERT_VALUES));
    }
  }
  CHKERRQ(DMDAVecRestoreArrayRead(cda,gc,&coors));

  CHKERRQ(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Parameter_settings(AppCtx *user)
{
  PetscBool      flg;

  PetscFunctionBeginUser;

  /* Set default parameters */
  user->ws     = 1.0;
  user->H      = 5.0;
  user->Pmax   = 2.1;
  user->PM_min = 1.0;
  user->lambda = 0.1;
  user->q      = 1.0;
  user->mux    = PetscAsinScalar(user->PM_min/user->Pmax);
  user->sigmax = 0.1;
  user->sigmay = 0.1;
  user->rho    = 0.0;
  user->xmin   = -PETSC_PI;
  user->xmax   =  PETSC_PI;
  user->bx     = DM_BOUNDARY_PERIODIC;
  user->by     = DM_BOUNDARY_MIRROR;
  user->nonoiseinitial = PETSC_FALSE;

  /*
     ymin of -3 seems to let the unstable solution move up and leave a zero in its wake
     with an ymin of -1 the wake is never exactly zero
  */
  user->ymin   = -3.0;
  user->ymax   = 10.0;
  user->st_width = 1;

  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-ws",&user->ws,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-Inertia",&user->H,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-Pmax",&user->Pmax,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-PM_min",&user->PM_min,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-lambda",&user->lambda,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-q",&user->q,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-mux",&user->mux,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-sigmax",&user->sigmax,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-muy",&user->muy,&flg));
  if (flg == 0) {
    user->muy = user->ws;
  }
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-sigmay",&user->sigmay,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-rho",&user->rho,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-xmin",&user->xmin,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-xmax",&user->xmax,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-ymin",&user->ymin,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-ymax",&user->ymax,&flg));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&user->st_width,&flg));
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-bx",DMBoundaryTypes,(PetscEnum*)&user->bx,&flg));
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-by",DMBoundaryTypes,(PetscEnum*)&user->by,&flg));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-nonoiseinitial",&user->nonoiseinitial,&flg));

  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -ts_max_steps 2
      localrunfiles: petscopt_ex7

   test:
      suffix: 2
      args: -ts_max_steps 2 -snes_mf_operator
      output_file: output/ex7_1.out
      localrunfiles: petscopt_ex7
      timeoutfactor: 2

   test:
      suffix: 3
      args: -ts_max_steps 2 -snes_mf -pc_type none
      output_file: output/ex7_1.out
      localrunfiles: petscopt_ex7
      timeoutfactor: 2

TEST*/
