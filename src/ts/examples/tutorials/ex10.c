static const char help[] = "1D nonequilibrium radiation diffusion with Saha ionization model\n\n";

/*
  This example implements the model described in

    Rauenzahn, Mousseau, Knoll. "Temporal accuracy of the nonequilibrium radiation diffusion
    equations employing a Saha ionization model" 2005.

  The paper discusses three examples, the first two are nondimensional with a simple
  ionization model.  The third example is fully dimensional and uses the Saha ionization
  model with realistic parameters.
*/

#include "petscts.h"
#include "petscda.h"

typedef enum {BC_DIRICHLET,BC_NEUMANN,BC_ROBIN} BCType;
static const char *const BCTypes[] = {"DIRICHLET","NEUMANN","ROBIN","BCType","BC_",0};

typedef struct {
  PetscScalar E;                /* radiation energy */
  PetscScalar T;                /* material temperature */
} RDNode;

typedef struct _n_RD *RD;

struct _n_RD {
  void (*MaterialEnergy)(RD,const RDNode*,const RDNode*,PetscScalar*,PetscScalar*);
  DA da;
  PetscTruth monitor_residual;
  PetscTruth fd_jacobian;
  PetscInt   initial;
  BCType     leftbc;
  PetscTruth view_draw;
  char       view_binary[PETSC_MAX_PATH_LEN];
  PetscTruth endpoint;

  struct {
    PetscReal meter,kilogram,second,Kelvin; /* Fundamental units */
    PetscReal Joule,Watt;                   /* Derived units */
  } unit;
  /* model constants, see Table 2 and RDCreate() */
  PetscReal rho,K_R,K_p,I_H,m_p,m_e,h,k,c,sigma_b,beta,gamma;

  PetscReal Eapplied;           /* Radiation flux from the left */
  PetscReal L;                  /* Length of domain */
  PetscReal final_time;
};

#undef __FUNCT__  
#define __FUNCT__ "RDDestroy"
static PetscErrorCode RDDestroy(RD rd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DADestroy(rd->da);CHKERRQ(ierr);
  ierr = PetscFree(rd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The paper has a time derivative for material energy (Eq 2) which is a dependent variable (computable from temperature
 * and density through an uninvertible relation).  Computing this derivative is trivial for trapezoid rule (used in the
 * paper), but does not generalize nicely to higher order integrators.  Here we use the implicit form which provides
 * time derivatives of the independent variables (radiation energy and temperature), so we must compute the time
 * derivative of material energy ourselves (could be done using AD).
 *
 * There are multiple ionization models, this interface dispatches to the one currently in use.
 */
static void RDMaterialEnergy(RD rd,const RDNode *n,const RDNode *nt,PetscScalar *Em,PetscScalar *Em_t)
{ rd->MaterialEnergy(rd,n,nt,Em,Em_t); }

/* Solves a quadratic equation while propagating tangents */
static void QuadraticSolve(PetscScalar a,PetscScalar a_t,PetscScalar b,PetscScalar b_t,PetscScalar c,PetscScalar c_t,PetscScalar *x,PetscScalar *x_t)
{
  PetscScalar
    disc   = b*b - 4*a*c,
    disc_t = 2*b*b_t - 4*a_t*c - 4*a*c_t,
    num    = -b + PetscSqrtScalar(disc), /* choose positive sign */
    num_t  = -b_t + 0.5/PetscSqrtScalar(disc)*disc_t,
    den    = 2*a,
    den_t  = 2*a_t;
  *x   = num/den;
  *x_t = (num_t*den - num*den_t) / PetscSqr(den);
}

/* The primary model presented in the paper */
static void RDMaterialEnergy_Saha(RD rd,const RDNode *n,const RDNode *nt,PetscScalar *inEm,PetscScalar *inEm_t)
{
  PetscScalar Em,Em_t,alpha,alpha_t,
    T     = n->T,
    T_t   = nt->T,
    chi   = rd->I_H / (rd->k * T),
    chi_t = -chi / T * T_t,
    a     = 1.,
    a_t   = 0,
    b     = 4 * rd->m_p / rd->rho * pow(2. * PETSC_PI * rd->m_e * rd->I_H / PetscSqr(rd->h),1.5) * PetscExpScalar(-chi) * PetscPowScalar(chi,1.5), /* Eq 7 */
    b_t   = -b*chi_t + 1.5*b/chi*chi_t,
    c     = -b,
    c_t   = -b_t;
  QuadraticSolve(a,a_t,b,b_t,c,c_t,&alpha,&alpha_t);
  Em   = rd->k * T / rd->m_p * (1.5*(1+alpha) + alpha*chi); /* Eq 6 */
  Em_t = Em / T * T_t + rd->k * T / rd->m_p * (1.5*alpha_t + alpha_t*chi + alpha*chi_t);
  if (inEm)   *inEm   = Em;
  if (inEm_t) *inEm_t = Em_t;
}
/* Reduced ionization model, Eq 30 */
static void RDMaterialEnergy_Reduced(RD rd,const RDNode *n,const RDNode *nt,PetscScalar *Em,PetscScalar *Em_t)
{
  PetscScalar alpha,alpha_t,
    T = n->T,
    T_t = nt->T,
    chi = -0.3 / T,
    chi_t = -chi / T * T_t,
    a = 1.,
    a_t = 0.,
    b = PetscExpScalar(chi),
    b_t = b*chi_t,
    c = -b,
    c_t = -b_t;
  QuadraticSolve(a,a_t,b,b_t,c,c_t,&alpha,&alpha_t);
  if (Em)   *Em   = (1+alpha)*T + 0.3*alpha;
  if (Em_t) *Em_t = alpha_t*T + (1+alpha)*T_t + 0.3*alpha_t;
}

static void RDSigma_R(RD rd,RDNode *n,PetscScalar *sigma_R)
{*sigma_R = rd->K_R * rd->rho / PetscPowScalar(n->T,rd->gamma);}

static void RDDiffusionCoefficient(RD rd,RDNode *n,RDNode *nx,PetscScalar *D_r)
{
  PetscScalar sigma_R;
  RDSigma_R(rd,n,&sigma_R);
  *D_r = rd->c / (3. * rd->rho * sigma_R + PetscAbsScalar(nx->E) / n->E);
}

#undef __FUNCT__  
#define __FUNCT__ "RDStateView"
static PetscErrorCode RDStateView(RD rd,Vec X,Vec Xdot,Vec F)
{
  PetscErrorCode ierr;
  DALocalInfo info;
  PetscInt i;
  RDNode *x,*xdot,*f;
  MPI_Comm comm = ((PetscObject)rd->da)->comm;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,X,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,F,&f);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    ierr = PetscSynchronizedPrintf(comm,"x[%d] (%10.2g,%10.2g) (%10.2g,%10.2g) (%10.2g,%10.2g)\n",i,x[i].E,x[i].T,xdot[i].E,xdot[i].T,f[i].E,f[i].T);CHKERRQ(ierr);
  }
  ierr = DAVecRestoreArray(rd->da,X,&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(rd->da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(rd->da,F,&f);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscScalar RDRadiation(RD rd,RDNode *n)
{
  PetscScalar sigma_p = rd->K_p * rd->rho / PetscPowScalar(n->T,rd->beta);
  return sigma_p * rd->c * rd->rho * (4. * rd->sigma_b * PetscSqr(PetscSqr(n->T)) / rd->c - n->E);
}

static PetscScalar RDDiffusion(RD rd,PetscReal hx,RDNode *x,PetscInt i)
{
  RDNode n_L,nx_L,n_R,nx_R;
  PetscScalar D_L,D_R;

  n_L.E = 0.5*(x[i-1].E + x[i].E);
  n_L.T = 0.5*(x[i-1].T + x[i].T);
  nx_L.E = (x[i].E - x[i-1].E)/hx;
  nx_L.T = (x[i].T - x[i-1].T)/hx;
  RDDiffusionCoefficient(rd,&n_L,&nx_L,&D_L);

  n_R.E = 0.5*(x[i].E + x[i+1].E);
  n_R.T = 0.5*(x[i].T + x[i+1].T);
  nx_R.E = (x[i+1].E - x[i].E)/hx;
  nx_R.T = (x[i+1].T - x[i].T)/hx;
  RDDiffusionCoefficient(rd,&n_R,&nx_R,&D_R);
  return (D_R*nx_R.E - D_L*nx_L.E)/hx;
}

#undef __FUNCT__  
#define __FUNCT__ "RDIFunction"
static PetscErrorCode RDIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RD             rd = (RD)ctx;
  RDNode         *x,*x0,*xdot,*f;
  Vec            X0loc,Xloc,Xloc_t;
  PetscReal      hx,min,Theta,dt;
  PetscTruth     istheta;
  DALocalInfo    info;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = VecMin(X,PETSC_NULL,&min);CHKERRQ(ierr);
  if (min < 0) {
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetFunctionDomainError(snes);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = DAGetLocalVector(rd->da,&Xloc);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(rd->da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(rd->da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  ierr = DAGetLocalVector(rd->da,&Xloc_t);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(rd->da,Xdot,INSERT_VALUES,Xloc_t);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(rd->da,Xdot,INSERT_VALUES,Xloc_t);CHKERRQ(ierr);

  /*
    The following is a hack to subvert TSTHETA which is like an implicit midpoint method to behave more like a trapezoid
    rule.  These methods have equivalent linear stability, but the nonlinear stability is somewhat different.  The
    radiation system is inconvenient to write in explicit form because the ionization model is "on the left".
   */
  ierr = PetscTypeCompare((PetscObject)ts,TSTHETA,&istheta);CHKERRQ(ierr);
  if (istheta && rd->endpoint) {
    ierr = TSThetaGetTheta(ts,&Theta);CHKERRQ(ierr);
  } else {
    Theta = 1.;
  }
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = DAGetLocalVector(rd->da,&X0loc);CHKERRQ(ierr);
  ierr = VecWAXPY(X0loc,-Theta*dt,Xloc_t,Xloc);CHKERRQ(ierr); /* back out the value at the start of this step */
  if (rd->endpoint) {
    ierr = VecWAXPY(Xloc,dt,Xloc_t,X0loc);CHKERRQ(ierr);      /* move the abscissa to the end of the step */
  }

  ierr = DAVecGetArray(rd->da,Xloc,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,X0loc,&x0);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,F,&f);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);

  hx = rd->L / info.mx;

  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal rho = rd->rho;
    PetscScalar Em_t,rad;

    rad = (1.-Theta)*RDRadiation(rd,&x0[i]) + Theta*RDRadiation(rd,&x[i]);
    if (rd->endpoint) {
      PetscScalar Em0,Em1;
      RDMaterialEnergy(rd,&x0[i],&xdot[i],&Em0,PETSC_NULL);
      RDMaterialEnergy(rd,&x[i],&xdot[i],&Em1,PETSC_NULL);
      Em_t = (Em1 - Em0) / dt;
    } else {
      RDMaterialEnergy(rd,&x[i],&xdot[i],PETSC_NULL,&Em_t);
    }

    /*
      In the following, residuals are multiplied by the volume element (hx).

      In the "endpoint" version, the boundary conditions are enforced at the end of the
      step (note that the boundary conditions are not time-dependent).
    */
    if (i == 0) {               /* Left boundary condition */
      RDNode n;
      PetscScalar sigma_R;
      n.E = 0.5*(x[i].E + x[i+1].E);
      n.T = 0.5*(x[i].T + x[i+1].T);
      RDSigma_R(rd,&n,&sigma_R);
      switch (rd->leftbc) {
        case BC_ROBIN:
          f[i].E = hx*(x[i].E - (2./(3.*rho*sigma_R) * (x[i+1].E - x[i].E)/hx) - rd->Eapplied);
          break;
        case BC_NEUMANN:
          f[i].E = x[i+1].E - x[i].E;
          break;
        default: SETERRQ1(PETSC_ERR_SUP,"Case %D",rd->initial);
      }
    } else if (i == info.mx-1) { /* Right boundary */
      f[i].E = x[i].E - x[i-1].E; /* Homogeneous Neumann */
    } else {
      PetscScalar diff = (1.-Theta)*RDDiffusion(rd,hx,x0,i) + Theta*RDDiffusion(rd,hx,x,i);
      f[i].E = hx*(xdot[i].E - diff - rad);
    }
    /* The temperature equation does not have boundary conditions */
    f[i].T = hx*(rho*Em_t + rad);
  }
  ierr = DAVecRestoreArray(rd->da,Xloc,&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(rd->da,X0loc,&x0);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(rd->da,Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(rd->da,F,&f);CHKERRQ(ierr);

  ierr = DARestoreLocalVector(rd->da,&Xloc);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(rd->da,&X0loc);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(rd->da,&Xloc_t);CHKERRQ(ierr);

  if (rd->monitor_residual) {ierr = RDStateView(rd,X,Xdot,F);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Temperature that is in equilibrium with the radiation density */
static PetscScalar RDRadiationTemperature(RD rd,PetscScalar E)
{ return pow(E*rd->c/(4.*rd->sigma_b),0.25); }

static PetscErrorCode RDInitialState(RD rd,Vec X)
{
  DALocalInfo info;
  PetscInt i;
  RDNode *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(rd->da,X,&x);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal coord = i*rd->L/(info.mx-1);
    switch (rd->initial) {
      case 1:
        x[i].E = 0.001;
        x[i].T = RDRadiationTemperature(rd,x[i].E);
        break;
      case 2:
        x[i].E = 0.001 + 100.*PetscExpScalar(-PetscSqr(coord/0.1));
        x[i].T = RDRadiationTemperature(rd,x[i].E);
        break;
      case 3:
        x[i].E = 7.56e-2 * rd->unit.Joule / pow(rd->unit.meter,3);
        x[i].T = RDRadiationTemperature(rd,x[i].E);
        printf("T %g  dT %g\n",x[i].T,x[i].T - 3160 * rd->unit.Kelvin);
        break;
      default: SETERRQ1(PETSC_ERR_SUP,"No initial state %d",rd->initial);
    }
  }
  ierr = DAVecRestoreArray(rd->da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "RDView"
static PetscErrorCode RDView(RD rd,Vec X,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Vec            Y;
  RDNode         *x;
  PetscScalar    *y;
  PetscInt       i,m,M;
  const PetscInt *lx;
  DA             da;

  PetscFunctionBegin;
  /*
    Create a DA (one dof per node, zero stencil width, same layout) to hold Trad
    (radiation temperature).  It is not necessary to create a DA for this, but this way
    output and visualization will have meaningful variable names and correct scales.
  */
  ierr = DAGetInfo(rd->da,0, &M,0,0, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetOwnershipRanges(rd->da,&lx,0,0);CHKERRQ(ierr);
  ierr = DACreate1d(((PetscObject)rd->da)->comm,DA_NONPERIODIC,M,1,0,lx,&da);CHKERRQ(ierr);
  ierr = DASetUniformCoordinates(da,0.,rd->L,0.,0.,0.,0.);CHKERRQ(ierr);
  ierr = DASetFieldName(da,0,"T_rad");CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&Y);CHKERRQ(ierr);

  /* Compute the radiation temperature from the solution at each node */
  ierr = VecGetLocalSize(Y,&m);CHKERRQ(ierr);
  ierr = VecGetArray(X,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    y[i] = RDRadiationTemperature(rd,x[i].E);
  }
  ierr = VecRestoreArray(X,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);

  ierr = VecView(Y,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(Y);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "RDCreate"
static PetscErrorCode RDCreate(MPI_Comm comm,RD *inrd)
{
  PetscErrorCode ierr;
  RD             rd;
  PetscReal      meter,kilogram,second,Kelvin,Joule,Watt;

  PetscFunctionBegin;
  *inrd = 0;
  ierr = PetscNew(struct _n_RD,&rd);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Options for nonequilibrium radiation-diffusion with RD ionization",PETSC_NULL);CHKERRQ(ierr);
  {
    /* Fundamental units */
    rd->unit.kilogram = 1.;
    rd->unit.meter    = 1.;
    rd->unit.second   = 1.;
    rd->unit.Kelvin   = 1.;
    ierr = PetscOptionsReal("-rd_unit_meter","Length of 1 meter in nondimensional units","",rd->unit.meter,&rd->unit.meter,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_unit_kilogram","Mass of 1 kilogram in nondimensional units","",rd->unit.kilogram,&rd->unit.kilogram,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_unit_second","Time of a second in nondimensional units","",rd->unit.second,&rd->unit.second,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_unit_Kelvin","Temperature of a Kelvin in nondimensional units","",rd->unit.Kelvin,&rd->unit.Kelvin,0);CHKERRQ(ierr);
    /* Derived units */
    rd->unit.Joule = rd->unit.kilogram*PetscSqr(rd->unit.meter/rd->unit.second);
    rd->unit.Watt  = rd->unit.Joule/rd->unit.second;
    /* Local aliases */
    meter    = rd->unit.meter;
    kilogram = rd->unit.kilogram;
    second   = rd->unit.second;
    Kelvin   = rd->unit.Kelvin;
    Joule    = rd->unit.Joule;
    Watt     = rd->unit.Watt;

    ierr = PetscOptionsTruth("-rd_monitor_residual","Display residuals every time they are evaluated","",rd->monitor_residual,&rd->monitor_residual,0);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-rd_fd_jacobian","Use a finite difference Jacobian (ghosted and colored)","",rd->fd_jacobian,&rd->fd_jacobian,0);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-rd_initial","Initial condition (1=Marshak, 2=Blast, 3=Marshak+)","",rd->initial,&rd->initial,0);CHKERRQ(ierr);
    switch (rd->initial) {
      case 1:
        rd->leftbc     = BC_ROBIN;
        rd->Eapplied   = 4 * rd->unit.Joule / pow(rd->unit.meter,3.);
        rd->L          = 1. * rd->unit.meter;
        rd->beta       = 3.0;
        rd->gamma      = 3.0;
        rd->final_time = 3 * second;
        break;
      case 2:
        rd->leftbc     = BC_NEUMANN;
        rd->Eapplied   = 0.;
        rd->L          = 1. * rd->unit.meter;
        rd->beta       = 3.0;
        rd->gamma      = 3.0;
        rd->final_time = 1 * second;
        break;
      case 3:
        rd->leftbc     = BC_ROBIN;
        rd->Eapplied   = 7.503e6 * rd->unit.Joule / pow(rd->unit.meter,3);
        rd->L          = 5. * rd->unit.meter;
        rd->beta       = 3.5;
        rd->gamma      = 3.5;
        rd->final_time = 20e-9 * second;
        break;
      default: SETERRQ1(PETSC_ERR_SUP,"Initial %D",rd->initial);
    }
    ierr = PetscOptionsEnum("-rd_leftbc","Left boundary condition","",BCTypes,(PetscEnum)rd->leftbc,(PetscEnum*)&rd->leftbc,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_E_applied","Radiation flux at left end of domain","",rd->Eapplied,&rd->Eapplied,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_beta","Thermal exponent for photon absorption","",rd->beta,&rd->beta,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_gamma","Thermal exponent for diffusion coefficient","",rd->gamma,&rd->gamma,0);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-rd_view_draw","Draw final solution","",rd->view_draw,&rd->view_draw,0);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-rd_endpoint","Discretize using endpoints (like trapezoid rule) instead of midpoint","",rd->endpoint,&rd->endpoint,0);CHKERRQ(ierr);
    ierr = PetscOptionsString("-rd_view_binary","File name to hold final solution","",rd->view_binary,rd->view_binary,sizeof(rd->view_binary),0);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  switch (rd->initial) {
    case 1:
    case 2:
      rd->rho     = 1.;
      rd->c       = 1.;
      rd->K_R     = 1.;
      rd->K_p     = 1.;
      rd->sigma_b = 0.25;
      rd->MaterialEnergy = RDMaterialEnergy_Reduced;
      break;
    case 3:
      /* Table 2 */
      rd->rho     = 1.17e-3 * kilogram / (meter*meter*meter);                      /* density */
      rd->K_R     = 7.44e18 * pow(meter,5.) * pow(Kelvin,3.5) * pow(kilogram,-2.); /*  */
      rd->K_p     = 2.33e20 * pow(meter,5.) * pow(Kelvin,3.5) * pow(kilogram,-2.); /*  */
      rd->I_H     = 2.179e-18 * Joule;                                             /* Hydrogen ionization potential */
      rd->m_p     = 1.673e-27 * kilogram;                                          /* proton mass */
      rd->m_e     = 9.109e-31 * kilogram;                                          /* electron mass */
      rd->h       = 6.626e-34 * Joule * second;                                    /* Planck's constant */
      rd->k       = 1.381e-23 * Joule / Kelvin;                                    /* Boltzman constant */
      rd->c       = 3.00e8 * meter / second;                                       /* speed of light */
      rd->sigma_b = 5.67e-8 * Watt * pow(meter,-2.) * pow(Kelvin,-4.);             /* Stefan-Boltzman constant */
      rd->MaterialEnergy = RDMaterialEnergy_Saha;
      break;
  }


  ierr = DACreate1d(comm,DA_NONPERIODIC,-20,sizeof(RDNode)/sizeof(PetscScalar),1,PETSC_NULL,&rd->da);CHKERRQ(ierr);
  ierr = DASetFieldName(rd->da,0,"E");CHKERRQ(ierr);
  ierr = DASetFieldName(rd->da,1,"T");CHKERRQ(ierr);
  ierr = DASetUniformCoordinates(rd->da,0.,1.,0.,0.,0.,0.);CHKERRQ(ierr);

  *inrd = rd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  RD             rd;
  TS             ts;
  Vec            X;
  Mat            A,B;
  PetscInt       steps;
  PetscReal      ftime;
  MatFDColoring  matfdcoloring;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = RDCreate(PETSC_COMM_WORLD,&rd);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(rd->da,&X);CHKERRQ(ierr);
  ierr = DAGetMatrix(rd->da,MATAIJ,&B);CHKERRQ(ierr);
  ierr = RDInitialState(rd,X);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,RDIFunction,rd);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,B,B,0,rd);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,rd->final_time);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.,1e-3);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  A = B;
  if (rd->fd_jacobian) {
    SNES           snes;
    ISColoring     iscoloring;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = DAGetColoring(rd->da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(B,iscoloring,&matfdcoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode(*)(void))SNESTSFormFunction,ts);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,A,B,SNESDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr);
  }

  ierr = TSStep(ts,&steps,&ftime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Steps %D  final time %G\n",steps,ftime);CHKERRQ(ierr);
  if (rd->view_draw) {
    ierr = RDView(rd,X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  if (rd->view_binary[0]) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,rd->view_binary,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = RDView(rd,X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  if (matfdcoloring) {ierr = MatFDColoringDestroy(matfdcoloring);CHKERRQ(ierr);}
  ierr = VecDestroy(X);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = RDDestroy(rd);CHKERRQ(ierr);
  ierr = TSDestroy(ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
