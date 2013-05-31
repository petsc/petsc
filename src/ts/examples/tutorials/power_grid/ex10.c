static char help[] = "Time-dependent PDE in 2d for calculating joint PDF. \n";
/*
   p_t = -\delta_t*p_\delta -\omega_t*p_\omega + f(t)*p_\omega\omega
   xmin < x < deltamax, omegamin < y < omegamax;

   Boundary conditions:
   Zero dirichlet in \omega using ghosted values
   Periodic in \delta

   Note that \delta_t and \omega_t in the above are given functions of \delta and \omega; they are not derivatives of \delta and \omega. 
   \delta_t = \omega_s*(\omega - 1.0)  
   \omega_t = (1/2H)*(Pm - Pmax*sin(\delta) - D*(\omega - 1.0))

   In this example, the following fault scenario is simulated
   t < tf            -> Pmax = E*V/0.545
   t > tf && t < tcl -> Pmax = 0
   t > tcl           -> Pmax = E*V/0.745

   The variable coefficient advection equations are expressed in conservative form with two spatial discretization schemes available.
   i) 1st order upwinding scheme
  ii) 3rd order upwinding scheme with Koren flux limiter (From the book 'Numerical Solution of Time-Dependent Advection-Diffision-Reaction
                                                          Equations' Pg. 216)

   An explicit time discretization scheme is used.
							  
   Example runs (In these example runs, the grid is zoomed in so the periodic boundary conditions on \delta don't make sense. Since the solution stays off from
                 the boundary, this is not really a problem)
   ------------
   1st order upwinding scheme
   ./ex10 -ts_type ssp -ts_dt 0.001 -discretization UPWIND1 -deltamin 0.0 -deltamax 1.2 -omegamin 0.99 -omegamax 1.01 -ts_final_time 5.0
   3rd order upwinding scheme
   ./ex10 -ts_type ssp -ts_dt 0.001 -discretization UPWIND3_WITH_FLUXLIMITER -deltamin 0.0 -deltamax 1.2 -omegamin 0.99 -omegamax 1.01 -ts_final_time 5.0

*/

#include <petscdmda.h>
#include <petscts.h>

static const char *const BoundaryTypes[] = {"NONE","GHOSTED","MIRROR","PERIODIC","DMDABoundaryType","DMDA_BOUNDARY_",0};
typedef enum {DMDA_DISCRETIZATION_UPWIND1,DMDA_DISCRETIZATION_UPWIND3_WITH_FLUXLIMITER,DMDA_DISCRETIZATION_WENO5} DMDADiscretizationType;
static const char *const DiscretizationTypes[] = {"UPWIND1","UPWIND3_WITH_FLUXLIMITER","WENO5","DMDADiscretizationType","DMDA_DISCRETIZATION_",0};
typedef enum{FLUX_LIMITER_KOREN} FluxLimiterType;
static const char *const FluxLimiterTypes[] = {"KOREN","FluxLimiterType","FLUX_LIMITER_",0};


/*
   User-defined data structures and routines
*/
typedef struct {
  PetscScalar omega_s;   /* Synchronous speed */
  PetscScalar H;    /* Inertia constant */
  PetscScalar D;    /* Damping constant */
  PetscScalar Pmax,Pmax_s; /* Maximum power output of generator */
  PetscScalar E,V,X;  /* Internal voltage, terminal voltage, and total reactance */
  PetscScalar PM_min; /* Mean mechanical power input */
  PetscScalar lambda; /* correlation time */
  PetscScalar q;      /* noise strength */
  PetscScalar mu_delta;    /* Initial average angle */
  PetscScalar sigma_delta; /* Standard deviation of initial angle */
  PetscScalar mu_w;    /* Average speed */
  PetscScalar sigma_w; /* standard deviation of initial speed */
  PetscScalar rho;    /* Cross-correlation coefficient */
  PetscScalar deltamin;   /* left boundary of angle */
  PetscScalar deltamax;   /* right boundary of angle */
  PetscScalar omegamin;   /* bottom boundary of speed */
  PetscScalar omegamax;   /* top boundary of speed */
  PetscScalar ddelta;     /* x step size */
  PetscScalar domega;     /* y step size */
  PetscScalar disper_coe; /* Dispersion coefficient */
  DM          da;
  DMDABoundaryType bdelta; /* x boundary type */
  DMDABoundaryType bomega; /* y boundary type */
  PetscReal        tf,tcl; /* Fault incidence and clearing times */
  DMDADiscretizationType dis; /* Spatial discretization scheme */
  PetscInt               st_width; /* Stencil width */
  FluxLimiterType ftype;
  PetscViewer     binv;
  PetscInt        howoften;
} AppCtx;

PetscErrorCode Parameter_settings(AppCtx*);
PetscErrorCode ini_bou(Vec,AppCtx*);
PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
PetscErrorCode PostStep(TS);
PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x;  /* Solution vector */
  TS             ts;   /* Time-stepping context */
  AppCtx         user; /* Application context */
  PetscViewer    viewer;

  PetscInitialize(&argc,&argv,"petscopt_ex10", help);

  /* Get physics and time parameters */
  ierr = Parameter_settings(&user);CHKERRQ(ierr);
  /* Create a 2D DA with dof = 1 */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,user.bdelta,user.bomega,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,user.st_width,NULL,NULL,&user.da);CHKERRQ(ierr);
  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(user.da,user.deltamin,user.deltamax,user.omegamin,user.omegamax,0,0);CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(user.da,0,"X - the angle");
  ierr = DMDASetCoordinateName(user.da,1,"Y - the speed");

  /* Get global vector x from DM  */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);

  ierr = ini_bou(x,&user);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ini_x",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ex10output",FILE_MODE_WRITE,&user.binv);CHKERRQ(ierr);
  user.howoften = 1;
  ierr = PetscOptionsGetInt(NULL,"-howoften",&user.howoften,NULL);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.005);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,PostStep);CHKERRQ(ierr);
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&user.binv);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Corrected user->mu_delta = %f, user->mu_w = %f user->PM_min = %f,user->ddelta = %f\n",user.mu_delta,user.mu_w,user.PM_min,user.ddelta);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;
  PetscFunctionBegin;
  if (step % user->howoften == 0) {
    ierr = VecView(X,user->binv);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PostStep"
PetscErrorCode PostStep(TS ts)
{
  PetscErrorCode ierr;
  Vec            X;
  AppCtx         *user;
  PetscReal      t;
  PetscScalar    asum;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);

  if ((t > user->tf) && (t < user->tcl)) user->Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else if (t >= user->tcl) user->Pmax = user->E/0.745;

  ierr = VecSum(X,&asum);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"sum(p) at t = %f = %f\n",(double)t,(double)(asum));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ini_bou"
PetscErrorCode ini_bou(Vec X,AppCtx* user)
{
  PetscErrorCode ierr;
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p;
  Vec            gc;
  PetscInt       M,N,I,J;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  user->ddelta = (user->deltamax - user->deltamin)/(M-1);
  user->domega = (user->omegamax - user->omegamin)/(N-1);

  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(user->da,&gc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,X,&p);CHKERRQ(ierr);

  /* Point mass at (mu_delta,mu_w) */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Original user->mu_delta = %f, user->mu_w = %f\n",user->mu_delta,user->mu_w);CHKERRQ(ierr);
  ierr = DMDAGetLogicalCoordinate(user->da,user->mu_delta,user->mu_w,0.0,&I,&J,NULL,&user->mu_delta,&user->mu_w,NULL);CHKERRQ(ierr);
  user->PM_min = user->Pmax*sin(user->mu_delta);
  if (I > -1 && J > -1) {
    p[J][I] = 1.0;
  }

  ierr = DMDAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,X,&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "flux_limiter"
PETSC_STATIC_INLINE PetscErrorCode flux_limiter(PetscScalar theta,PetscScalar *psi,AppCtx *user)
{
  PetscFunctionBegin;
  if (user->ftype == FLUX_LIMITER_KOREN) {
    /* Koren 1993 */
    *psi = PetscMax(0,PetscMin(1,PetscMin(1./3. + theta/6.,theta)));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WENO5"
PETSC_STATIC_INLINE PetscErrorCode WENO5(PetscScalar m3,PetscScalar m2,PetscScalar m1, PetscScalar p1, PetscScalar p2,PetscScalar *f)
{
  PetscReal f1,f2,f3;
  PetscReal b1,b2,b3;
  PetscReal c1=0.3, c2 = 0.6, c3 = 0.1;
  PetscReal a1,a2,a3, a_sum_inv, w1, w2, w3;
  PetscReal one_sixth  = 1.0/6.0;
  PetscReal one_fourth = 1.0/4.0;
  PetscReal frac_13_12 = 13.0/12.0;

  PetscFunctionBegin;

  f1 = (2*one_sixth)*m1 + (5*one_sixth)*p1 - (one_sixth)*p2;
  f2 = (-one_sixth)*m2 + (5.0*one_sixth)*m1 + (2*one_sixth)*p1;
  f3 = (2*one_sixth)*m3 - (7.0*one_sixth)*m2 + (11.0*one_sixth)*m1;

  b1 = frac_13_12*(m1-2*p1+p2)*(m1-2*p1+p2) + one_fourth*(3*m1-4*p1+p2)*(3*m1-4*p1+p2);
  b2 = frac_13_12*(m2-2*m1+p1)*(m2-2*m1+p1) + one_fourth*(m2-p1)*(m2-p1);
  b3 = frac_13_12*(m3-2*m2+m1)*(m3-2*m2+m1) + one_fourth*(m3-4*m2+3*m1)*(m3-4*m2+3*m1);

  a1 = c1/PetscPowScalar(b1+1e-6,2);
  a2 = c2/PetscPowScalar(b2+1e-6,2);
  a3 = c3/PetscPowScalar(b3+1e-6,2);

  a_sum_inv = 1.0 / (a1 + a2 + a3);
  w1 = a1 * a_sum_inv;
  w2 = a2 * a_sum_inv;
  w3 = a3 * a_sum_inv;

  *f = w1*f1 + w2*f2 + w3*f3;

  PetscFunctionReturn(0);
}

/* First advection term */
#undef __FUNCT__
#define __FUNCT__ "adv1"
PetscErrorCode adv1(PetscScalar **p,PetscScalar y,PetscInt i,PetscInt j,PetscInt M,PetscScalar *dp_ddelta,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    f, fminushalf, fplushalf;
  PetscFunctionBegin;
  f   =  user->omega_s*(y - 1.0); /* f = f(i-1/2) = f(i+1/2) */
  if (user->dis == DMDA_DISCRETIZATION_UPWIND1) {
    if (f >= 0.) {
      fminushalf    = f*p[j][i-1];      
      fplushalf    = f*p[j][i];
    } else {
      fminushalf    = f*p[j][i];
      fplushalf    = f*p[j][i+1];
    }
    *dp_ddelta = (fminushalf - fplushalf)/user->ddelta;
  } else if (user->dis == DMDA_DISCRETIZATION_UPWIND3_WITH_FLUXLIMITER) {
    PetscScalar psi,theta;
    if (f >= 0.) {
      theta = (p[j][i-1] - p[j][i-2])/(p[j][i] - p[j][i-1]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      fminushalf    = f*(p[j][i-1] + psi*(p[j][i] - p[j][i-1]));
      
      theta = (p[j][i] - p[j][i-1])/(p[j][i+1] - p[j][i]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      psi   = PetscMax(0,PetscMin(1,PetscMin(1./3. + theta/6.,theta)));
      fplushalf    = f*(p[j][i] + psi*(p[j][i+1] - p[j][i]));
    } else {
      theta = (p[j][i] - p[j][i+1])/(p[j][i-1] - p[j][i]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      fminushalf    = f*(p[j][i] + psi*(p[j][i-1] - p[j][i]));

      theta = (p[j][i+1] - p[j][i+2])/(p[j][i] - p[j][i+1]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      fplushalf    =  f*(p[j][i+1] + psi*(p[j][i] - p[j][i+1]));
    }
    *dp_ddelta = (fminushalf - fplushalf)/user->ddelta;
  } else if (user->dis == DMDA_DISCRETIZATION_WENO5) {
    if (f >= 0.) {
      ierr = WENO5(p[j][i-3],p[j][i-2],p[j][i-1],p[j][i],p[j][i+1],&fminushalf);CHKERRQ(ierr);
      ierr = WENO5(p[j][i-2],p[j][i-1],p[j][i],p[j][i+1],p[j][i+2],&fplushalf);CHKERRQ(ierr);
      fminushalf = f*fminushalf;
      fplushalf = f*fplushalf;
    } else {
      ierr = WENO5(p[j][i+2],p[j][i+1],p[j][i],p[j][i-1],p[j][i-2],&fminushalf);CHKERRQ(ierr);
      ierr = WENO5(p[j][i+3],p[j][i+2],p[j][i+1],p[j][i],p[j][i-1],&fplushalf);CHKERRQ(ierr);
      fminushalf = f*fminushalf;
      fplushalf = f*fplushalf;
    }
    *dp_ddelta = (fminushalf - fplushalf)/user->ddelta;
  }
  PetscFunctionReturn(0);
}

/* Second advection term */
#undef __FUNCT__
#define __FUNCT__ "adv2"
PetscErrorCode adv2(PetscScalar **p,PetscScalar x,PetscScalar y,PetscInt i,PetscInt j,PetscInt N,PetscScalar *dp_dw,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    fminushalf,fplushalf;
  PetscFunctionBegin;
  fminushalf   = (1.0/(2*user->H))*(user->PM_min - user->Pmax*sin(x) - user->D*(y-user->domega/2. - 1.0));
  fplushalf    = (1.0/(2*user->H))*(user->PM_min - user->Pmax*sin(x) - user->D*(y+user->domega/2. - 1.0));

  if (user->dis == DMDA_DISCRETIZATION_UPWIND1) {
    PetscScalar f1,f2;
    if (fminushalf > 0.) f1 = fminushalf*p[j-1][i];
    else f1 = fminushalf*p[j][i];
    if (fplushalf > 0.) f2 = fplushalf*p[j][i];
    else f2 = fplushalf*p[j+1][i];
    *dp_dw = (f1 - f2)/user->domega;
  } else if (user->dis == DMDA_DISCRETIZATION_UPWIND3_WITH_FLUXLIMITER) {
    PetscScalar f1,f2,theta,psi;
    if (fminushalf > 0.) {
      theta = (p[j-1][i] - p[j-2][i])/(p[j][i] - p[j-1][i]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      f1    = fminushalf*(p[j-1][i] + psi*(p[j][i] - p[j-1][i]));
    } else {
      theta = (p[j][i] - p[j+1][i])/(p[j-1][i] - p[j][i]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      f1    = fminushalf*(p[j][i] + psi*(p[j-1][i] - p[j][i]));
    }
    
    if (fplushalf >= 0.) {
      theta = (p[j][i] - p[j-1][i])/(p[j+1][i] - p[j][i]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      f2    = fplushalf*(p[j][i] + psi*(p[j+1][i] - p[j][i]));
    } else {
      theta = (p[j+1][i] - p[j+2][i])/(p[j][i] - p[j+1][i]);
      ierr = flux_limiter(theta,&psi,user);CHKERRQ(ierr);
      f2    =  fplushalf*(p[j+1][i] + psi*(p[j][i] - p[j+1][i]));
    }
    *dp_dw = (f1 - f2)/user->domega;
  } else if (user->dis == DMDA_DISCRETIZATION_WENO5) {
    PetscScalar f1,f2;
    if (fminushalf > 0.) {
      ierr = WENO5(p[j-3][i],p[j-2][i],p[j-1][i],p[j][i],p[j+1][i],&f1);CHKERRQ(ierr);
      f1    = fminushalf*f1;
    } else {
      ierr = WENO5(p[j+2][i],p[j+1][i],p[j][i],p[j-1][i],p[j-2][i],&f1);CHKERRQ(ierr);
      f1    = fminushalf*f1;
    }
    
    if (fplushalf >= 0.) {
      ierr = WENO5(p[j-2][i],p[j-1][i],p[j][i],p[j+1][i],p[j+2][i],&f2);CHKERRQ(ierr);
      f2    = fplushalf*f2;
    } else {
      ierr = WENO5(p[j+3][i],p[j+2][i],p[j+1][i],p[j][i],p[j-1][i],&f2);CHKERRQ(ierr);
      f2    =  fplushalf*f2;
    }
    *dp_dw = (f1 - f2)/user->domega;
  }    
  PetscFunctionReturn(0);
}

/* Diffusion term */
#undef __FUNCT__
#define __FUNCT__ "diffuse"
PetscErrorCode diffuse(PetscScalar **p,PetscInt i,PetscInt j,PetscReal t,PetscScalar *p_diff,AppCtx * user)
{
  PetscFunctionBeginUser;
  *p_diff = -user->disper_coe*((p[j-1][i] - 2*p[j][i] + p[j+1][i])/(user->domega*user->domega));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p,**f;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  Vec            localX,gc;
  PetscScalar    p_adv1,p_adv2,p_diff;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(user->da,&gc);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,F,&f);CHKERRQ(ierr);

  PetscScalar diffuse1,gamma;
  gamma = user->D*1.0/(2*user->H);
  diffuse1 = user->lambda*user->lambda*user->q/(user->lambda*gamma+1)*(1.0 - PetscExpScalar(-t*(gamma + 1.0/user->lambda)));
  user->disper_coe = 1.0/(4*user->H*user->H)*diffuse1;

  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      ierr = adv1(p,coors[j][i].y,i,j,M,&p_adv1,user);CHKERRQ(ierr);
      ierr = adv2(p,coors[j][i].x,coors[j][i].y,i,j,N,&p_adv2,user);CHKERRQ(ierr);
      ierr = diffuse(p,i,j,t,&p_diff,user);CHKERRQ(ierr);
      f[j][i] = +p_adv1 + p_adv2 - p_diff;
    }
  }
  ierr = DMDAVecRestoreArray(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Parameter_settings"
PetscErrorCode Parameter_settings(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBeginUser;

  /* Set default parameters */
  user->omega_s     = 2*PETSC_PI*60;
  user->H      = 5.0;
  user->D      = 5.0;
  user->E      = 1.1358;
  user->V      = 1.0;
  user->X      = 0.545;
  user->PM_min = 0.9;
  user->lambda = 0.1;
  user->q      = 1.0;
  user->sigma_delta = 0.1;
  user->sigma_w = 0.1;
  user->rho    = 0.0;
  user->deltamin   = -PETSC_PI;
  user->deltamax   = PETSC_PI;
  user->bdelta     = DMDA_BOUNDARY_PERIODIC;
  user->bomega     = DMDA_BOUNDARY_GHOSTED;
  user->dis    = DMDA_DISCRETIZATION_UPWIND3_WITH_FLUXLIMITER;
  user->ftype  = FLUX_LIMITER_KOREN;
  user->st_width = 2;
  user->tf = user->tcl = -1;
  user->omegamin   = 0.5;
  user->omegamax   = 1.5;

  ierr = PetscOptionsGetScalar(NULL,"-Inertia",&user->H,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-E",&user->E,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-V",&user->V,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-X",&user->X,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-D",&user->D,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-PM_min",&user->PM_min,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-lambda",&user->lambda,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-q",&user->q,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-mu_delta",&user->mu_delta,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-mu_w",&user->mu_w,&flg);CHKERRQ(ierr);

  user->Pmax = user->Pmax_s  = user->E*user->V/user->X;
  user->mu_delta    = asin(user->PM_min/user->Pmax);
  user->mu_w    = 1.0;

  ierr = PetscOptionsGetScalar(NULL,"-deltamin",&user->deltamin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-deltamax",&user->deltamax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-omegamin",&user->omegamin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-omegamax",&user->omegamax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum("","-bdelta",BoundaryTypes,(PetscEnum*)&user->bdelta,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum("","-bomega",BoundaryTypes,(PetscEnum*)&user->bomega,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum("","-discretization",DiscretizationTypes,(PetscEnum*)&user->dis,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum("","-flux_limiter",FluxLimiterTypes,(PetscEnum*)&user->ftype,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-tf",&user->tf,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-tcl",&user->tcl,&flg);CHKERRQ(ierr);
  if ( user->dis == DMDA_DISCRETIZATION_UPWIND1) user->st_width = 1;
  if (user->dis == DMDA_DISCRETIZATION_WENO5) user->st_width = 3;
  PetscFunctionReturn(0);
}
