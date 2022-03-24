static char help[] = "Time-dependent PDE in 2d for calculating joint PDF. \n";
/*
   p_t = -x_t*p_x -y_t*p_y + f(t)*p_yy
   xmin < x < xmax, ymin < y < ymax;

   Boundary conditions:
   Zero dirichlet in y using ghosted values
   Periodic in x

   Note that x_t and y_t in the above are given functions of x and y; they are not derivatives of x and y.
   x_t = (y - ws)
   y_t = (ws/2H)*(Pm - Pmax*sin(x) - D*(w - ws))

   In this example, we can see the effect of a fault, that zeroes the electrical power output
   Pmax*sin(x), on the PDF. The fault on/off times can be controlled by options -tf and -tcl respectively.

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
  PetscScalar Pmax,Pmax_s; /* Maximum power output of generator */
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
  PetscReal        tf,tcl; /* Fault incidence and clearing times */
} AppCtx;

PetscErrorCode Parameter_settings(AppCtx*);
PetscErrorCode ini_bou(Vec,AppCtx*);
PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PetscErrorCode PostStep(TS);

int main(int argc, char **argv)
{
  Vec         x;                /* Solution vector */
  TS          ts;               /* Time-stepping context */
  AppCtx      user;             /* Application context */
  PetscViewer viewer;

  CHKERRQ(PetscInitialize(&argc,&argv,"petscopt_ex8", help));

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
  /*  CHKERRQ(TSSetIJacobian(ts,NULL,NULL,IJacobian,&user)); */
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
  Vec          X;
  AppCtx      *user;
  PetscReal    t;
  PetscScalar  asum;

  PetscFunctionBegin;
  CHKERRQ(TSGetApplicationContext(ts,&user));
  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&X));
  /*
  if (t >= .2) {
    CHKERRQ(TSGetSolution(ts,&X));
    CHKERRQ(VecView(X,PETSC_VIEWER_BINARY_WORLD));
    exit(0);
     results in initial conditions after fault in binaryoutput
  }*/

  if ((t > user->tf) && (t < user->tcl)) user->Pmax = 0.0; /* A short-circuit that drives the electrical power output (Pmax*sin(delta)) to zero */
  else user->Pmax = user->Pmax_s;

  CHKERRQ(VecSum(X,&asum));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sum(p) at t = %f = %f\n",(double)t,(double)(asum)));
  PetscFunctionReturn(0);
}

PetscErrorCode ini_bou(Vec X,AppCtx* user)
{
  DM            cda;
  DMDACoor2d  **coors;
  PetscScalar **p;
  Vec           gc;
  PetscInt      M,N,Ir,J;
  PetscMPIInt   rank;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  user->dx = (user->xmax - user->xmin)/(M-1); user->dy = (user->ymax - user->ymin)/(N-1);
  CHKERRQ(DMGetCoordinateDM(user->da,&cda));
  CHKERRQ(DMGetCoordinates(user->da,&gc));
  CHKERRQ(DMDAVecGetArrayRead(cda,gc,&coors));
  CHKERRQ(DMDAVecGetArray(user->da,X,&p));

  /* Point mass at (mux,muy) */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Original user->mux = %f, user->muy = %f\n",user->mux,user->muy));
  CHKERRQ(DMDAGetLogicalCoordinate(user->da,user->mux,user->muy,0.0,&Ir,&J,NULL,&user->mux,&user->muy,NULL));
  user->PM_min = user->Pmax*PetscSinScalar(user->mux);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Corrected user->mux = %f, user->muy = %f user->PM_min = %f,user->dx = %f\n",user->mux,user->muy,user->PM_min,user->dx));
  if (Ir > -1 && J > -1) {
    p[J][Ir] = 1.0;
  }

  CHKERRQ(DMDAVecRestoreArrayRead(cda,gc,&coors));
  CHKERRQ(DMDAVecRestoreArray(user->da,X,&p));
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
  }
  /* *p1 = f*(p[j][i+1] - p[j][i-1])/user->dx;*/
  PetscFunctionReturn(0);
}

/* Second advection term */
PetscErrorCode adv2(PetscScalar **p,PetscScalar x,PetscScalar y,PetscInt i,PetscInt j,PetscInt N,PetscScalar *p2,AppCtx *user)
{
  PetscScalar f,fpos,fneg;
  PetscFunctionBegin;
  f   = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(x) - user->D*(y - user->ws));
  fpos = PetscMax(f,0);
  fneg = PetscMin(f,0);
  if (user->st_width == 1) {
    *p2 = fpos*(p[j][i] - p[j-1][i])/user->dy + fneg*(p[j+1][i] - p[j][i])/user->dy;
  } else if (user->st_width ==2) {
    *p2 = fpos*(3*p[j][i] - 4*p[j-1][i] + p[j-2][i])/(2*user->dy) + fneg*(-p[j+2][i] + 4*p[j+1][i] - 3*p[j][i])/(2*user->dy);
  } else if (user->st_width == 3) {
    *p2 = fpos*(2*p[j+1][i] + 3*p[j][i] - 6*p[j-1][i] + p[j-2][i])/(6*user->dy) + fneg*(-p[j+2][i] + 6*p[j+1][i] - 3*p[j][i] - 2*p[j-1][i])/(6*user->dy);
  }

  /* *p2 = f*(p[j+1][i] - p[j-1][i])/user->dy;*/
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
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  AppCtx       *user   = (AppCtx*)ctx;
  DM            cda;
  DMDACoor2d  **coors;
  PetscScalar **p,**f,**pdot;
  PetscInt      i,j;
  PetscInt      xs,ys,xm,ym,M,N;
  Vec           localX,gc,localXdot;
  PetscScalar   p_adv1 = 0.0,p_adv2 = 0.0,p_diff = 0;
  PetscScalar   diffuse1,gamma;

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

  gamma = user->D*user->ws/(2*user->H);
  diffuse1 = user->lambda*user->lambda*user->q/(user->lambda*gamma+1)*(1.0 - PetscExpScalar(-t*(gamma+1.0)/user->lambda));
  user->disper_coe = user->ws*user->ws/(4*user->H*user->H)*diffuse1;

  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      CHKERRQ(adv1(p,coors[j][i].y,i,j,M,&p_adv1,user));
      CHKERRQ(adv2(p,coors[j][i].x,coors[j][i].y,i,j,N,&p_adv2,user));
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
      c3        = (user->ws/(2.0*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(xi) - user->D*(yi - user->ws))/user->dy;
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
  PetscBool flg;

  PetscFunctionBeginUser;
  /* Set default parameters */
  user->ws     = 1.0;
  user->H      = 5.0;
  user->D      = 0.0;
  user->Pmax = user->Pmax_s  = 2.1;
  user->PM_min = 1.0;
  user->lambda = 0.1;
  user->q      = 1.0;
  user->mux    = PetscAsinScalar(user->PM_min/user->Pmax);
  user->sigmax = 0.1;
  user->sigmay = 0.1;
  user->rho    = 0.0;
  user->xmin   = -PETSC_PI;
  user->xmax   = PETSC_PI;
  user->bx     = DM_BOUNDARY_PERIODIC;
  user->by     = DM_BOUNDARY_GHOSTED;
  user->tf = user->tcl = -1;
  user->ymin   = -2.0;
  user->ymax   = 2.0;
  user->st_width = 1;

  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-ws",&user->ws,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-Inertia",&user->H,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-D",&user->D,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-Pmax",&user->Pmax,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-PM_min",&user->PM_min,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-lambda",&user->lambda,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-q",&user->q,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-mux",&user->mux,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-muy",&user->muy,&flg));
  if (flg == 0) user->muy = user->ws;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-xmin",&user->xmin,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-xmax",&user->xmax,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-ymin",&user->ymin,&flg));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-ymax",&user->ymax,&flg));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&user->st_width,&flg));
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-bx",DMBoundaryTypes,(PetscEnum*)&user->bx,&flg));
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-by",DMBoundaryTypes,(PetscEnum*)&user->by,&flg));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tf",&user->tf,&flg));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tcl",&user->tcl,&flg));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex x

   test:
      args: -ts_max_steps 1
      localrunfiles: petscopt_ex8
      timeoutfactor: 3

TEST*/
