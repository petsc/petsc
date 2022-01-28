static const char help[] = "1D nonequilibrium radiation diffusion with Saha ionization model.\n\n";

/*
  This example implements the model described in

    Rauenzahn, Mousseau, Knoll. "Temporal accuracy of the nonequilibrium radiation diffusion
    equations employing a Saha ionization model" 2005.

  The paper discusses three examples, the first two are nondimensional with a simple
  ionization model.  The third example is fully dimensional and uses the Saha ionization
  model with realistic parameters.
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>

typedef enum {BC_DIRICHLET,BC_NEUMANN,BC_ROBIN} BCType;
static const char *const BCTypes[] = {"DIRICHLET","NEUMANN","ROBIN","BCType","BC_",0};
typedef enum {JACOBIAN_ANALYTIC,JACOBIAN_MATRIXFREE,JACOBIAN_FD_COLORING,JACOBIAN_FD_FULL} JacobianType;
static const char *const JacobianTypes[] = {"ANALYTIC","MATRIXFREE","FD_COLORING","FD_FULL","JacobianType","FD_",0};
typedef enum {DISCRETIZATION_FD,DISCRETIZATION_FE} DiscretizationType;
static const char *const DiscretizationTypes[] = {"FD","FE","DiscretizationType","DISCRETIZATION_",0};
typedef enum {QUADRATURE_GAUSS1,QUADRATURE_GAUSS2,QUADRATURE_GAUSS3,QUADRATURE_GAUSS4,QUADRATURE_LOBATTO2,QUADRATURE_LOBATTO3} QuadratureType;
static const char *const QuadratureTypes[] = {"GAUSS1","GAUSS2","GAUSS3","GAUSS4","LOBATTO2","LOBATTO3","QuadratureType","QUADRATURE_",0};

typedef struct {
  PetscScalar E;                /* radiation energy */
  PetscScalar T;                /* material temperature */
} RDNode;

typedef struct {
  PetscReal meter,kilogram,second,Kelvin; /* Fundamental units */
  PetscReal Joule,Watt;                   /* Derived units */
} RDUnit;

typedef struct _n_RD *RD;

struct _n_RD {
  void               (*MaterialEnergy)(RD,const RDNode*,PetscScalar*,RDNode*);
  DM                 da;
  PetscBool          monitor_residual;
  DiscretizationType discretization;
  QuadratureType     quadrature;
  JacobianType       jacobian;
  PetscInt           initial;
  BCType             leftbc;
  PetscBool          view_draw;
  char               view_binary[PETSC_MAX_PATH_LEN];
  PetscBool          test_diff;
  PetscBool          endpoint;
  PetscBool          bclimit;
  PetscBool          bcmidpoint;
  RDUnit             unit;

  /* model constants, see Table 2 and RDCreate() */
  PetscReal rho,K_R,K_p,I_H,m_p,m_e,h,k,c,sigma_b,beta,gamma;

  /* Domain and boundary conditions */
  PetscReal Eapplied;           /* Radiation flux from the left */
  PetscReal L;                  /* Length of domain */
  PetscReal final_time;
};

static PetscErrorCode RDDestroy(RD *rd)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMDestroy(&(*rd)->da);CHKERRQ(ierr);
  ierr = PetscFree(*rd);CHKERRQ(ierr);
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
static void RDMaterialEnergy(RD rd,const RDNode *n,PetscScalar *Em,RDNode *dEm) { rd->MaterialEnergy(rd,n,Em,dEm); }

/* Solves a quadratic equation while propagating tangents */
static void QuadraticSolve(PetscScalar a,PetscScalar a_t,PetscScalar b,PetscScalar b_t,PetscScalar c,PetscScalar c_t,PetscScalar *x,PetscScalar *x_t)
{
  PetscScalar
    disc   = b*b - 4.*a*c,
    disc_t = 2.*b*b_t - 4.*a_t*c - 4.*a*c_t,
    num    = -b + PetscSqrtScalar(disc), /* choose positive sign */
    num_t  = -b_t + 0.5/PetscSqrtScalar(disc)*disc_t,
    den    = 2.*a,
    den_t  = 2.*a_t;
  *x   = num/den;
  *x_t = (num_t*den - num*den_t) / PetscSqr(den);
}

/* The primary model presented in the paper */
static void RDMaterialEnergy_Saha(RD rd,const RDNode *n,PetscScalar *inEm,RDNode *dEm)
{
  PetscScalar Em,alpha,alpha_t,
              T     = n->T,
              T_t   = 1.,
              chi   = rd->I_H / (rd->k * T),
              chi_t = -chi / T * T_t,
              a     = 1.,
              a_t   = 0,
              b     = 4. * rd->m_p / rd->rho * PetscPowScalarReal(2. * PETSC_PI * rd->m_e * rd->I_H / PetscSqr(rd->h),1.5) * PetscExpScalar(-chi) * PetscPowScalarReal(chi,1.5), /* Eq 7 */
              b_t   = -b*chi_t + 1.5*b/chi*chi_t,
              c     = -b,
              c_t   = -b_t;
  QuadraticSolve(a,a_t,b,b_t,c,c_t,&alpha,&alpha_t);       /* Solve Eq 7 for alpha */
  Em = rd->k * T / rd->m_p * (1.5*(1.+alpha) + alpha*chi); /* Eq 6 */
  if (inEm) *inEm = Em;
  if (dEm) {
    dEm->E = 0;
    dEm->T = Em / T * T_t + rd->k * T / rd->m_p * (1.5*alpha_t + alpha_t*chi + alpha*chi_t);
  }
}
/* Reduced ionization model, Eq 30 */
static void RDMaterialEnergy_Reduced(RD rd,const RDNode *n,PetscScalar *Em,RDNode *dEm)
{
  PetscScalar alpha,alpha_t,
              T     = n->T,
              T_t   = 1.,
              chi   = -0.3 / T,
              chi_t = -chi / T * T_t,
              a     = 1.,
              a_t   = 0.,
              b     = PetscExpScalar(chi),
              b_t   = b*chi_t,
              c     = -b,
              c_t   = -b_t;
  QuadraticSolve(a,a_t,b,b_t,c,c_t,&alpha,&alpha_t);
  if (Em) *Em = (1.+alpha)*T + 0.3*alpha;
  if (dEm) {
    dEm->E = 0;
    dEm->T = alpha_t*T + (1.+alpha)*T_t + 0.3*alpha_t;
  }
}

/* Eq 5 */
static void RDSigma_R(RD rd,RDNode *n,PetscScalar *sigma_R,RDNode *dsigma_R)
{
  *sigma_R    = rd->K_R * rd->rho * PetscPowScalar(n->T,-rd->gamma);
  dsigma_R->E = 0;
  dsigma_R->T = -rd->gamma * (*sigma_R) / n->T;
}

/* Eq 4 */
static void RDDiffusionCoefficient(RD rd,PetscBool limit,RDNode *n,RDNode *nx,PetscScalar *D_R,RDNode *dD_R,RDNode *dxD_R)
{
  PetscScalar sigma_R,denom;
  RDNode      dsigma_R,ddenom,dxdenom;

  RDSigma_R(rd,n,&sigma_R,&dsigma_R);
  denom     = 3. * rd->rho * sigma_R + (int)limit * PetscAbsScalar(nx->E) / n->E;
  ddenom.E  = -(int)limit * PetscAbsScalar(nx->E) / PetscSqr(n->E);
  ddenom.T  = 3. * rd->rho * dsigma_R.T;
  dxdenom.E = (int)limit * (PetscRealPart(nx->E)<0 ? -1. : 1.) / n->E;
  dxdenom.T = 0;
  *D_R      = rd->c / denom;
  if (dD_R) {
    dD_R->E = -rd->c / PetscSqr(denom) * ddenom.E;
    dD_R->T = -rd->c / PetscSqr(denom) * ddenom.T;
  }
  if (dxD_R) {
    dxD_R->E = -rd->c / PetscSqr(denom) * dxdenom.E;
    dxD_R->T = -rd->c / PetscSqr(denom) * dxdenom.T;
  }
}

static PetscErrorCode RDStateView(RD rd,Vec X,Vec Xdot,Vec F)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i;
  const RDNode   *x,*xdot,*f;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)rd->da,&comm);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(rd->da,X,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(rd->da,Xdot,(void*)&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(rd->da,F,(void*)&f);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    ierr = PetscSynchronizedPrintf(comm,"x[%D] (%10.2G,%10.2G) (%10.2G,%10.2G) (%10.2G,%10.2G)\n",i,PetscRealPart(x[i].E),PetscRealPart(x[i].T),
                                   PetscRealPart(xdot[i].E),PetscRealPart(xdot[i].T), PetscRealPart(f[i].E),PetscRealPart(f[i].T));CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayRead(rd->da,X,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(rd->da,Xdot,(void*)&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(rd->da,F,(void*)&f);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscScalar RDRadiation(RD rd,const RDNode *n,RDNode *dn)
{
  PetscScalar sigma_p   = rd->K_p * rd->rho * PetscPowScalar(n->T,-rd->beta),
              sigma_p_T = -rd->beta * sigma_p / n->T,
              tmp       = 4.* rd->sigma_b*PetscSqr(PetscSqr(n->T)) / rd->c - n->E,
              tmp_E     = -1.,
              tmp_T     = 4. * rd->sigma_b * 4 * n->T*(PetscSqr(n->T)) / rd->c,
              rad       = sigma_p * rd->c * rd->rho * tmp,
              rad_E     = sigma_p * rd->c * rd->rho * tmp_E,
              rad_T     = rd->c * rd->rho * (sigma_p_T * tmp + sigma_p * tmp_T);
  if (dn) {
    dn->E = rad_E;
    dn->T = rad_T;
  }
  return rad;
}

static PetscScalar RDDiffusion(RD rd,PetscReal hx,const RDNode x[],PetscInt i,RDNode d[])
{
  PetscReal   ihx = 1./hx;
  RDNode      n_L,nx_L,n_R,nx_R,dD_L,dxD_L,dD_R,dxD_R,dfluxL[2],dfluxR[2];
  PetscScalar D_L,D_R,fluxL,fluxR;

  n_L.E  = 0.5*(x[i-1].E + x[i].E);
  n_L.T  = 0.5*(x[i-1].T + x[i].T);
  nx_L.E = (x[i].E - x[i-1].E)/hx;
  nx_L.T = (x[i].T - x[i-1].T)/hx;
  RDDiffusionCoefficient(rd,PETSC_TRUE,&n_L,&nx_L,&D_L,&dD_L,&dxD_L);
  fluxL       = D_L*nx_L.E;
  dfluxL[0].E = -ihx*D_L + (0.5*dD_L.E - ihx*dxD_L.E)*nx_L.E;
  dfluxL[1].E = +ihx*D_L + (0.5*dD_L.E + ihx*dxD_L.E)*nx_L.E;
  dfluxL[0].T = (0.5*dD_L.T - ihx*dxD_L.T)*nx_L.E;
  dfluxL[1].T = (0.5*dD_L.T + ihx*dxD_L.T)*nx_L.E;

  n_R.E  = 0.5*(x[i].E + x[i+1].E);
  n_R.T  = 0.5*(x[i].T + x[i+1].T);
  nx_R.E = (x[i+1].E - x[i].E)/hx;
  nx_R.T = (x[i+1].T - x[i].T)/hx;
  RDDiffusionCoefficient(rd,PETSC_TRUE,&n_R,&nx_R,&D_R,&dD_R,&dxD_R);
  fluxR       = D_R*nx_R.E;
  dfluxR[0].E = -ihx*D_R + (0.5*dD_R.E - ihx*dxD_R.E)*nx_R.E;
  dfluxR[1].E = +ihx*D_R + (0.5*dD_R.E + ihx*dxD_R.E)*nx_R.E;
  dfluxR[0].T = (0.5*dD_R.T - ihx*dxD_R.T)*nx_R.E;
  dfluxR[1].T = (0.5*dD_R.T + ihx*dxD_R.T)*nx_R.E;

  if (d) {
    d[0].E = -ihx*dfluxL[0].E;
    d[0].T = -ihx*dfluxL[0].T;
    d[1].E =  ihx*(dfluxR[0].E - dfluxL[1].E);
    d[1].T =  ihx*(dfluxR[0].T - dfluxL[1].T);
    d[2].E =  ihx*dfluxR[1].E;
    d[2].T =  ihx*dfluxR[1].T;
  }
  return ihx*(fluxR - fluxL);
}

static PetscErrorCode RDGetLocalArrays(RD rd,TS ts,Vec X,Vec Xdot,PetscReal *Theta,PetscReal *dt,Vec *X0loc,RDNode **x0,Vec *Xloc,RDNode **x,Vec *Xloc_t,RDNode **xdot)
{
  PetscErrorCode ierr;
  PetscBool      istheta;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(rd->da,X0loc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(rd->da,Xloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(rd->da,Xloc_t);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(rd->da,X,INSERT_VALUES,*Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(rd->da,X,INSERT_VALUES,*Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(rd->da,Xdot,INSERT_VALUES,*Xloc_t);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(rd->da,Xdot,INSERT_VALUES,*Xloc_t);CHKERRQ(ierr);

  /*
    The following is a hack to subvert TSTHETA which is like an implicit midpoint method to behave more like a trapezoid
    rule.  These methods have equivalent linear stability, but the nonlinear stability is somewhat different.  The
    radiation system is inconvenient to write in explicit form because the ionization model is "on the left".
   */
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSTHETA,&istheta);CHKERRQ(ierr);
  if (istheta && rd->endpoint) {
    ierr = TSThetaGetTheta(ts,Theta);CHKERRQ(ierr);
  } else *Theta = 1.;

  ierr = TSGetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = VecWAXPY(*X0loc,-(*Theta)*(*dt),*Xloc_t,*Xloc);CHKERRQ(ierr); /* back out the value at the start of this step */
  if (rd->endpoint) {
    ierr = VecWAXPY(*Xloc,*dt,*Xloc_t,*X0loc);CHKERRQ(ierr);      /* move the abscissa to the end of the step */
  }

  ierr = DMDAVecGetArray(rd->da,*X0loc,x0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(rd->da,*Xloc,x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(rd->da,*Xloc_t,xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RDRestoreLocalArrays(RD rd,Vec *X0loc,RDNode **x0,Vec *Xloc,RDNode **x,Vec *Xloc_t,RDNode **xdot)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMDAVecRestoreArray(rd->da,*X0loc,x0);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(rd->da,*Xloc,x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(rd->da,*Xloc_t,xdot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(rd->da,X0loc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(rd->da,Xloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(rd->da,Xloc_t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RDCheckDomain_Private(RD rd,TS ts,Vec X,PetscBool  *in)
{
  PetscErrorCode ierr;
  PetscInt       minloc;
  PetscReal      min;

  PetscFunctionBeginUser;
  ierr = VecMin(X,&minloc,&min);CHKERRQ(ierr);
  if (min < 0) {
    SNES snes;
    *in  = PETSC_FALSE;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetFunctionDomainError(snes);CHKERRQ(ierr);
    ierr = PetscInfo3(ts,"Domain violation at %D field %D value %g\n",minloc/2,minloc%2,(double)min);CHKERRQ(ierr);
  } else *in = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Energy and temperature must remain positive */
#define RDCheckDomain(rd,ts,X) do {                                \
    PetscErrorCode _ierr;                                          \
    PetscBool      _in;                                                \
    _ierr = RDCheckDomain_Private(rd,ts,X,&_in);CHKERRQ(_ierr);    \
    if (!_in) PetscFunctionReturn(0);                              \
  } while (0)

static PetscErrorCode RDIFunction_FD(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RD             rd = (RD)ctx;
  RDNode         *x,*x0,*xdot,*f;
  Vec            X0loc,Xloc,Xloc_t;
  PetscReal      hx,Theta,dt;
  DMDALocalInfo  info;
  PetscInt       i;

  PetscFunctionBeginUser;
  RDCheckDomain(rd,ts,X);
  ierr = RDGetLocalArrays(rd,ts,X,Xdot,&Theta,&dt,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(rd->da,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  VecZeroEntries(F);

  hx = rd->L / (info.mx-1);

  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal   rho = rd->rho;
    PetscScalar Em_t,rad;

    rad = (1.-Theta)*RDRadiation(rd,&x0[i],0) + Theta*RDRadiation(rd,&x[i],0);
    if (rd->endpoint) {
      PetscScalar Em0,Em1;
      RDMaterialEnergy(rd,&x0[i],&Em0,NULL);
      RDMaterialEnergy(rd,&x[i],&Em1,NULL);
      Em_t = (Em1 - Em0) / dt;
    } else {
      RDNode dEm;
      RDMaterialEnergy(rd,&x[i],NULL,&dEm);
      Em_t = dEm.E * xdot[i].E + dEm.T * xdot[i].T;
    }
    /* Residuals are multiplied by the volume element (hx).  */
    /* The temperature equation does not have boundary conditions */
    f[i].T = hx*(rho*Em_t + rad);

    if (i == 0) {               /* Left boundary condition */
      PetscScalar D_R,bcTheta = rd->bcmidpoint ? Theta : 1.;
      RDNode      n, nx;

      n.E  =  (1.-bcTheta)*x0[0].E + bcTheta*x[0].E;
      n.T  =  (1.-bcTheta)*x0[0].T + bcTheta*x[0].T;
      nx.E = ((1.-bcTheta)*(x0[1].E-x0[0].E) + bcTheta*(x[1].E-x[0].E))/hx;
      nx.T = ((1.-bcTheta)*(x0[1].T-x0[0].T) + bcTheta*(x[1].T-x[0].T))/hx;
      switch (rd->leftbc) {
      case BC_ROBIN:
        RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D_R,0,0);
        f[0].E = hx*(n.E - 2. * D_R * nx.E - rd->Eapplied);
        break;
      case BC_NEUMANN:
        f[0].E = x[1].E - x[0].E;
        break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %D",rd->initial);
      }
    } else if (i == info.mx-1) { /* Right boundary */
      f[i].E = x[i].E - x[i-1].E; /* Homogeneous Neumann */
    } else {
      PetscScalar diff = (1.-Theta)*RDDiffusion(rd,hx,x0,i,0) + Theta*RDDiffusion(rd,hx,x,i,0);
      f[i].E = hx*(xdot[i].E - diff - rad);
    }
  }
  ierr = RDRestoreLocalArrays(rd,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(rd->da,F,&f);CHKERRQ(ierr);
  if (rd->monitor_residual) {ierr = RDStateView(rd,X,Xdot,F);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode RDIJacobian_FD(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  RD             rd = (RD)ctx;
  RDNode         *x,*x0,*xdot;
  Vec            X0loc,Xloc,Xloc_t;
  PetscReal      hx,Theta,dt;
  DMDALocalInfo  info;
  PetscInt       i;

  PetscFunctionBeginUser;
  RDCheckDomain(rd,ts,X);
  ierr = RDGetLocalArrays(rd,ts,X,Xdot,&Theta,&dt,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  hx   = rd->L / (info.mx-1);
  ierr = MatZeroEntries(B);CHKERRQ(ierr);

  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscInt                  col[3];
    PetscReal                 rho = rd->rho;
    PetscScalar /*Em_t,rad,*/ K[2][6];
    RDNode                    dEm_t,drad;

    /*rad = (1.-Theta)* */ RDRadiation(rd,&x0[i],0); /* + Theta* */ RDRadiation(rd,&x[i],&drad);

    if (rd->endpoint) {
      PetscScalar Em0,Em1;
      RDNode      dEm1;
      RDMaterialEnergy(rd,&x0[i],&Em0,NULL);
      RDMaterialEnergy(rd,&x[i],&Em1,&dEm1);
      /*Em_t = (Em1 - Em0) / (Theta*dt);*/
      dEm_t.E = dEm1.E / (Theta*dt);
      dEm_t.T = dEm1.T / (Theta*dt);
    } else {
      const PetscScalar epsilon = x[i].T * PETSC_SQRT_MACHINE_EPSILON;
      RDNode            n1;
      RDNode            dEm,dEm1;
      PetscScalar       Em_TT;

      n1.E = x[i].E;
      n1.T = x[i].T+epsilon;
      RDMaterialEnergy(rd,&x[i],NULL,&dEm);
      RDMaterialEnergy(rd,&n1,NULL,&dEm1);
      /* The Jacobian needs another derivative.  We finite difference here instead of
       * propagating second derivatives through the ionization model. */
      Em_TT = (dEm1.T - dEm.T) / epsilon;
      /*Em_t = dEm.E * xdot[i].E + dEm.T * xdot[i].T;*/
      dEm_t.E = dEm.E * a;
      dEm_t.T = dEm.T * a + Em_TT * xdot[i].T;
    }

    ierr = PetscMemzero(K,sizeof(K));CHKERRQ(ierr);
    /* Residuals are multiplied by the volume element (hx).  */
    if (i == 0) {
      PetscScalar D,bcTheta = rd->bcmidpoint ? Theta : 1.;
      RDNode      n, nx;
      RDNode      dD,dxD;

      n.E  = (1.-bcTheta)*x0[0].E + bcTheta*x[0].E;
      n.T  = (1.-bcTheta)*x0[0].T + bcTheta*x[0].T;
      nx.E = ((1.-bcTheta)*(x0[1].E-x0[0].E) + bcTheta*(x[1].E-x[0].E))/hx;
      nx.T = ((1.-bcTheta)*(x0[1].T-x0[0].T) + bcTheta*(x[1].T-x[0].T))/hx;
      switch (rd->leftbc) {
      case BC_ROBIN:
        RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D,&dD,&dxD);
        K[0][1*2+0] = (bcTheta/Theta)*hx*(1. -2.*D*(-1./hx) - 2.*nx.E*dD.E + 2.*nx.E*dxD.E/hx);
        K[0][1*2+1] = (bcTheta/Theta)*hx*(-2.*nx.E*dD.T);
        K[0][2*2+0] = (bcTheta/Theta)*hx*(-2.*D*(1./hx) - 2.*nx.E*dD.E - 2.*nx.E*dxD.E/hx);
        break;
      case BC_NEUMANN:
        K[0][1*2+0] = -1./Theta;
        K[0][2*2+0] = 1./Theta;
        break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %D",rd->initial);
      }
    } else if (i == info.mx-1) {
      K[0][0*2+0] = -1./Theta;
      K[0][1*2+0] = 1./Theta;
    } else {
      /*PetscScalar diff;*/
      RDNode ddiff[3];
      /*diff = (1.-Theta)*RDDiffusion(rd,hx,x0,i,0) + Theta* */ RDDiffusion(rd,hx,x,i,ddiff);
      K[0][0*2+0] = -hx*ddiff[0].E;
      K[0][0*2+1] = -hx*ddiff[0].T;
      K[0][1*2+0] = hx*(a - ddiff[1].E - drad.E);
      K[0][1*2+1] = hx*(-ddiff[1].T - drad.T);
      K[0][2*2+0] = -hx*ddiff[2].E;
      K[0][2*2+1] = -hx*ddiff[2].T;
    }

    K[1][1*2+0] = hx*(rho*dEm_t.E + drad.E);
    K[1][1*2+1] = hx*(rho*dEm_t.T + drad.T);

    col[0] = i-1;
    col[1] = i;
    col[2] = i+1<info.mx ? i+1 : -1;
    ierr   = MatSetValuesBlocked(B,1,&i,3,col,&K[0][0],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = RDRestoreLocalArrays(rd,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Evaluate interpolants and derivatives at a select quadrature point */
static void RDEvaluate(PetscReal interp[][2],PetscReal deriv[][2],PetscInt q,const RDNode x[],PetscInt i,RDNode *n,RDNode *nx)
{
  PetscInt j;
  n->E = 0; n->T = 0; nx->E = 0; nx->T = 0;
  for (j=0; j<2; j++) {
    n->E  += interp[q][j] * x[i+j].E;
    n->T  += interp[q][j] * x[i+j].T;
    nx->E += deriv[q][j] * x[i+j].E;
    nx->T += deriv[q][j] * x[i+j].T;
  }
}

/*
 Various quadrature rules.  The nonlinear terms are non-polynomial so no standard quadrature will be exact.
*/
static PetscErrorCode RDGetQuadrature(RD rd,PetscReal hx,PetscInt *nq,PetscReal weight[],PetscReal interp[][2],PetscReal deriv[][2])
{
  PetscInt        q,j;
  const PetscReal *refweight,(*refinterp)[2],(*refderiv)[2];

  PetscFunctionBeginUser;
  switch (rd->quadrature) {
  case QUADRATURE_GAUSS1: {
    static const PetscReal ww[1] = {1.},ii[1][2] = {{0.5,0.5}},dd[1][2] = {{-1.,1.}};
    *nq = 1; refweight = ww; refinterp = ii; refderiv = dd;
  } break;
  case QUADRATURE_GAUSS2: {
    static const PetscReal ii[2][2] = {{0.78867513459481287,0.21132486540518713},{0.21132486540518713,0.78867513459481287}},dd[2][2] = {{-1.,1.},{-1.,1.}},ww[2] = {0.5,0.5};
    *nq = 2; refweight = ww; refinterp = ii; refderiv = dd;
  } break;
  case QUADRATURE_GAUSS3: {
    static const PetscReal ii[3][2] = {{0.8872983346207417,0.1127016653792583},{0.5,0.5},{0.1127016653792583,0.8872983346207417}},
                           dd[3][2] = {{-1,1},{-1,1},{-1,1}},ww[3] = {5./18,8./18,5./18};
    *nq = 3; refweight = ww; refinterp = ii; refderiv = dd;
  } break;
  case QUADRATURE_GAUSS4: {
    static const PetscReal ii[][2] = {{0.93056815579702623,0.069431844202973658},
                                      {0.66999052179242813,0.33000947820757187},
                                      {0.33000947820757187,0.66999052179242813},
                                      {0.069431844202973658,0.93056815579702623}},
                           dd[][2] = {{-1,1},{-1,1},{-1,1},{-1,1}},ww[] = {0.17392742256872692,0.3260725774312731,0.3260725774312731,0.17392742256872692};

    *nq = 4; refweight = ww; refinterp = ii; refderiv = dd;
  } break;
  case QUADRATURE_LOBATTO2: {
    static const PetscReal ii[2][2] = {{1.,0.},{0.,1.}},dd[2][2] = {{-1.,1.},{-1.,1.}},ww[2] = {0.5,0.5};
    *nq = 2; refweight = ww; refinterp = ii; refderiv = dd;
  } break;
  case QUADRATURE_LOBATTO3: {
    static const PetscReal ii[3][2] = {{1,0},{0.5,0.5},{0,1}},dd[3][2] = {{-1,1},{-1,1},{-1,1}},ww[3] = {1./6,4./6,1./6};
    *nq = 3; refweight = ww; refinterp = ii; refderiv = dd;
  } break;
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown quadrature %d",(int)rd->quadrature);
  }

  for (q=0; q<*nq; q++) {
    weight[q] = refweight[q] * hx;
    for (j=0; j<2; j++) {
      interp[q][j] = refinterp[q][j];
      deriv[q][j]  = refderiv[q][j] / hx;
    }
  }
  PetscFunctionReturn(0);
}

/*
 Finite element version
*/
static PetscErrorCode RDIFunction_FE(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RD             rd = (RD)ctx;
  RDNode         *x,*x0,*xdot,*f;
  Vec            X0loc,Xloc,Xloc_t,Floc;
  PetscReal      hx,Theta,dt,weight[5],interp[5][2],deriv[5][2];
  DMDALocalInfo  info;
  PetscInt       i,j,q,nq;

  PetscFunctionBeginUser;
  RDCheckDomain(rd,ts,X);
  ierr = RDGetLocalArrays(rd,ts,X,Xdot,&Theta,&dt,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);

  ierr = DMGetLocalVector(rd->da,&Floc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(rd->da,Floc,&f);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);

  /* Set up shape functions and quadrature for elements (assumes a uniform grid) */
  hx   = rd->L / (info.mx-1);
  ierr = RDGetQuadrature(rd,hx,&nq,weight,interp,deriv);CHKERRQ(ierr);

  for (i=info.xs; i<PetscMin(info.xs+info.xm,info.mx-1); i++) {
    for (q=0; q<nq; q++) {
      PetscReal   rho = rd->rho;
      PetscScalar Em_t,rad,D_R,D0_R;
      RDNode      n,n0,nx,n0x,nt,ntx;
      RDEvaluate(interp,deriv,q,x,i,&n,&nx);
      RDEvaluate(interp,deriv,q,x0,i,&n0,&n0x);
      RDEvaluate(interp,deriv,q,xdot,i,&nt,&ntx);

      rad = (1.-Theta)*RDRadiation(rd,&n0,0) + Theta*RDRadiation(rd,&n,0);
      if (rd->endpoint) {
        PetscScalar Em0,Em1;
        RDMaterialEnergy(rd,&n0,&Em0,NULL);
        RDMaterialEnergy(rd,&n,&Em1,NULL);
        Em_t = (Em1 - Em0) / dt;
      } else {
        RDNode dEm;
        RDMaterialEnergy(rd,&n,NULL,&dEm);
        Em_t = dEm.E * nt.E + dEm.T * nt.T;
      }
      RDDiffusionCoefficient(rd,PETSC_TRUE,&n0,&n0x,&D0_R,0,0);
      RDDiffusionCoefficient(rd,PETSC_TRUE,&n,&nx,&D_R,0,0);
      for (j=0; j<2; j++) {
        f[i+j].E += (deriv[q][j] * weight[q] * ((1.-Theta)*D0_R*n0x.E + Theta*D_R*nx.E)
                     + interp[q][j] * weight[q] * (nt.E - rad));
        f[i+j].T += interp[q][j] * weight[q] * (rho * Em_t + rad);
      }
    }
  }
  if (info.xs == 0) {
    switch (rd->leftbc) {
    case BC_ROBIN: {
      PetscScalar D_R,D_R_bc;
      PetscReal   ratio,bcTheta = rd->bcmidpoint ? Theta : 1.;
      RDNode      n, nx;

      n.E  = (1-bcTheta)*x0[0].E + bcTheta*x[0].E;
      n.T  = (1-bcTheta)*x0[0].T + bcTheta*x[0].T;
      nx.E = (x[1].E-x[0].E)/hx;
      nx.T = (x[1].T-x[0].T)/hx;
      RDDiffusionCoefficient(rd,PETSC_TRUE,&n,&nx,&D_R,0,0);
      RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D_R_bc,0,0);
      ratio = PetscRealPart(D_R/D_R_bc);
      if (ratio > 1.) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Limited diffusivity is greater than unlimited");
      if (ratio < 1e-3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Heavily limited diffusivity");
      f[0].E += -ratio*0.5*(rd->Eapplied - n.E);
    } break;
    case BC_NEUMANN:
      /* homogeneous Neumann is the natural condition */
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %D",rd->initial);
    }
  }

  ierr = RDRestoreLocalArrays(rd,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(rd->da,Floc,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(rd->da,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(rd->da,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(rd->da,&Floc);CHKERRQ(ierr);

  if (rd->monitor_residual) {ierr = RDStateView(rd,X,Xdot,F);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode RDIJacobian_FE(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  RD             rd = (RD)ctx;
  RDNode         *x,*x0,*xdot;
  Vec            X0loc,Xloc,Xloc_t;
  PetscReal      hx,Theta,dt,weight[5],interp[5][2],deriv[5][2];
  DMDALocalInfo  info;
  PetscInt       i,j,k,q,nq;
  PetscScalar    K[4][4];

  PetscFunctionBeginUser;
  RDCheckDomain(rd,ts,X);
  ierr = RDGetLocalArrays(rd,ts,X,Xdot,&Theta,&dt,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  hx   = rd->L / (info.mx-1);
  ierr = RDGetQuadrature(rd,hx,&nq,weight,interp,deriv);CHKERRQ(ierr);
  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  for (i=info.xs; i<PetscMin(info.xs+info.xm,info.mx-1); i++) {
    PetscInt rc[2];

    rc[0] = i; rc[1] = i+1;
    ierr  = PetscMemzero(K,sizeof(K));CHKERRQ(ierr);
    for (q=0; q<nq; q++) {
      PetscScalar              D_R;
      PETSC_UNUSED PetscScalar rad;
      RDNode                   n,nx,nt,ntx,drad,dD_R,dxD_R,dEm;
      RDEvaluate(interp,deriv,q,x,i,&n,&nx);
      RDEvaluate(interp,deriv,q,xdot,i,&nt,&ntx);
      rad = RDRadiation(rd,&n,&drad);
      RDDiffusionCoefficient(rd,PETSC_TRUE,&n,&nx,&D_R,&dD_R,&dxD_R);
      RDMaterialEnergy(rd,&n,NULL,&dEm);
      for (j=0; j<2; j++) {
        for (k=0; k<2; k++) {
          K[j*2+0][k*2+0] += (+interp[q][j] * weight[q] * (a - drad.E) * interp[q][k]
                              + deriv[q][j] * weight[q] * ((D_R + dxD_R.E * nx.E) * deriv[q][k] + dD_R.E * nx.E * interp[q][k]));
          K[j*2+0][k*2+1] += (+interp[q][j] * weight[q] * (-drad.T * interp[q][k])
                              + deriv[q][j] * weight[q] * (dxD_R.T * deriv[q][k] + dD_R.T * interp[q][k]) * nx.E);
          K[j*2+1][k*2+0] +=   interp[q][j] * weight[q] * drad.E * interp[q][k];
          K[j*2+1][k*2+1] +=   interp[q][j] * weight[q] * (a * rd->rho * dEm.T + drad.T) * interp[q][k];
        }
      }
    }
    ierr = MatSetValuesBlocked(B,2,rc,2,rc,&K[0][0],ADD_VALUES);CHKERRQ(ierr);
  }
  if (info.xs == 0) {
    switch (rd->leftbc) {
    case BC_ROBIN: {
      PetscScalar D_R,D_R_bc;
      PetscReal   ratio;
      RDNode      n, nx;

      n.E  = (1-Theta)*x0[0].E + Theta*x[0].E;
      n.T  = (1-Theta)*x0[0].T + Theta*x[0].T;
      nx.E = (x[1].E-x[0].E)/hx;
      nx.T = (x[1].T-x[0].T)/hx;
      RDDiffusionCoefficient(rd,PETSC_TRUE,&n,&nx,&D_R,0,0);
      RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D_R_bc,0,0);
      ratio = PetscRealPart(D_R/D_R_bc);
      ierr  = MatSetValue(B,0,0,ratio*0.5,ADD_VALUES);CHKERRQ(ierr);
    } break;
    case BC_NEUMANN:
      /* homogeneous Neumann is the natural condition */
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %D",rd->initial);
    }
  }

  ierr = RDRestoreLocalArrays(rd,&X0loc,&x0,&Xloc,&x,&Xloc_t,&xdot);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Temperature that is in equilibrium with the radiation density */
static PetscScalar RDRadiationTemperature(RD rd,PetscScalar E) { return PetscPowScalar(E*rd->c/(4.*rd->sigma_b),0.25); }

static PetscErrorCode RDInitialState(RD rd,Vec X)
{
  DMDALocalInfo  info;
  PetscInt       i;
  RDNode         *x;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(rd->da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(rd->da,X,&x);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal coord = i*rd->L/(info.mx-1);
    switch (rd->initial) {
    case 1:
      x[i].E = 0.001;
      x[i].T = RDRadiationTemperature(rd,x[i].E);
      break;
    case 2:
      x[i].E = 0.001 + 100.*PetscExpReal(-PetscSqr(coord/0.1));
      x[i].T = RDRadiationTemperature(rd,x[i].E);
      break;
    case 3:
      x[i].E = 7.56e-2 * rd->unit.Joule / PetscPowScalarInt(rd->unit.meter,3);
      x[i].T = RDRadiationTemperature(rd,x[i].E);
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No initial state %D",rd->initial);
    }
  }
  ierr = DMDAVecRestoreArray(rd->da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RDView(RD rd,Vec X,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Vec            Y;
  const RDNode   *x;
  PetscScalar    *y;
  PetscInt       i,m,M;
  const PetscInt *lx;
  DM             da;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  /*
    Create a DMDA (one dof per node, zero stencil width, same layout) to hold Trad
    (radiation temperature).  It is not necessary to create a DMDA for this, but this way
    output and visualization will have meaningful variable names and correct scales.
  */
  ierr = DMDAGetInfo(rd->da,0, &M,0,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(rd->da,&lx,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)rd->da,&comm);CHKERRQ(ierr);
  ierr = DMDACreate1d(comm,DM_BOUNDARY_NONE,M,1,0,lx,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.,rd->L,0.,0.,0.,0.);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"T_rad");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&Y);CHKERRQ(ierr);

  /* Compute the radiation temperature from the solution at each node */
  ierr = VecGetLocalSize(Y,&m);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) y[i] = RDRadiationTemperature(rd,x[i].E);
  ierr = VecRestoreArrayRead(X,(const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);

  ierr = VecView(Y,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RDTestDifferentiation(RD rd)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;
  RDNode         n,nx;
  PetscScalar    epsilon;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)rd->da,&comm);CHKERRQ(ierr);
  epsilon = 1e-8;
  {
    RDNode      dEm,fdEm;
    PetscScalar T0 = 1000.,T1 = T0*(1.+epsilon),Em0,Em1;
    n.E = 1.;
    n.T = T0;
    rd->MaterialEnergy(rd,&n,&Em0,&dEm);
    n.E = 1.+epsilon;
    n.T = T0;
    rd->MaterialEnergy(rd,&n,&Em1,0);
    fdEm.E = (Em1-Em0)/epsilon;
    n.E = 1.;
    n.T = T1;
    rd->MaterialEnergy(rd,&n,&Em1,0);
    fdEm.T = (Em1-Em0)/(T0*epsilon);
    ierr = PetscPrintf(comm,"dEm {%g,%g}, fdEm {%g,%g}, diff {%g,%g}\n",(double)PetscRealPart(dEm.E),(double)PetscRealPart(dEm.T),
                       (double)PetscRealPart(fdEm.E),(double)PetscRealPart(fdEm.T),(double)PetscRealPart(dEm.E-fdEm.E),(double)PetscRealPart(dEm.T-fdEm.T));CHKERRQ(ierr);
  }
  {
    PetscScalar D0,D;
    RDNode      dD,dxD,fdD,fdxD;
    n.E = 1.;          n.T = 1.;           nx.E = 1.;          n.T = 1.;
    RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D0,&dD,&dxD);
    n.E = 1.+epsilon;  n.T = 1.;           nx.E = 1.;          n.T = 1.;
    RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D,0,0); fdD.E = (D-D0)/epsilon;
    n.E = 1;           n.T = 1.+epsilon;   nx.E = 1.;          n.T = 1.;
    RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D,0,0); fdD.T = (D-D0)/epsilon;
    n.E = 1;           n.T = 1.;           nx.E = 1.+epsilon;  n.T = 1.;
    RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D,0,0); fdxD.E = (D-D0)/epsilon;
    n.E = 1;           n.T = 1.;           nx.E = 1.;          n.T = 1.+epsilon;
    RDDiffusionCoefficient(rd,rd->bclimit,&n,&nx,&D,0,0); fdxD.T = (D-D0)/epsilon;
    ierr = PetscPrintf(comm,"dD {%g,%g}, fdD {%g,%g}, diff {%g,%g}\n",(double)PetscRealPart(dD.E),(double)PetscRealPart(dD.T),
                       (double)PetscRealPart(fdD.E),(double)PetscRealPart(fdD.T),(double)PetscRealPart(dD.E-fdD.E),(double)PetscRealPart(dD.T-fdD.T));CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"dxD {%g,%g}, fdxD {%g,%g}, diffx {%g,%g}\n",(double)PetscRealPart(dxD.E),(double)PetscRealPart(dxD.T),
                       (double)PetscRealPart(fdxD.E),(double)PetscRealPart(fdxD.T),(double)PetscRealPart(dxD.E-fdxD.E),(double)PetscRealPart(dxD.T-fdxD.T));CHKERRQ(ierr);
  }
  {
    PetscInt    i;
    PetscReal   hx = 1.;
    PetscScalar a0;
    RDNode      n0[3],n1[3],d[3],fd[3];

    n0[0].E = 1.;
    n0[0].T = 1.;
    n0[1].E = 5.;
    n0[1].T = 3.;
    n0[2].E = 4.;
    n0[2].T = 2.;
    a0 = RDDiffusion(rd,hx,n0,1,d);
    for (i=0; i<3; i++) {
      ierr    = PetscMemcpy(n1,n0,sizeof(n0));CHKERRQ(ierr); n1[i].E += epsilon;
      fd[i].E = (RDDiffusion(rd,hx,n1,1,0)-a0)/epsilon;
      ierr    = PetscMemcpy(n1,n0,sizeof(n0));CHKERRQ(ierr); n1[i].T += epsilon;
      fd[i].T = (RDDiffusion(rd,hx,n1,1,0)-a0)/epsilon;
      ierr    = PetscPrintf(comm,"ddiff[%D] {%g,%g}, fd {%g %g}, diff {%g,%g}\n",i,(double)PetscRealPart(d[i].E),(double)PetscRealPart(d[i].T),
                            (double)PetscRealPart(fd[i].E),(double)PetscRealPart(fd[i].T),(double)PetscRealPart(d[i].E-fd[i].E),(double)PetscRealPart(d[i].T-fd[i].T));CHKERRQ(ierr);
    }
  }
  {
    PetscScalar rad0,rad;
    RDNode      drad,fdrad;
    n.E  = 1.;         n.T = 1.;
    rad0 = RDRadiation(rd,&n,&drad);
    n.E  = 1.+epsilon; n.T = 1.;
    rad  = RDRadiation(rd,&n,0); fdrad.E = (rad-rad0)/epsilon;
    n.E  = 1.;         n.T = 1.+epsilon;
    rad  = RDRadiation(rd,&n,0); fdrad.T = (rad-rad0)/epsilon;
    ierr = PetscPrintf(comm,"drad {%g,%g}, fdrad {%g,%g}, diff {%g,%g}\n",(double)PetscRealPart(drad.E),(double)PetscRealPart(drad.T),
                       (double)PetscRealPart(fdrad.E),(double)PetscRealPart(fdrad.T),(double)PetscRealPart(drad.E-drad.E),(double)PetscRealPart(drad.T-fdrad.T));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RDCreate(MPI_Comm comm,RD *inrd)
{
  PetscErrorCode ierr;
  RD             rd;
  PetscReal      meter=0,kilogram=0,second=0,Kelvin=0,Joule=0,Watt=0;

  PetscFunctionBeginUser;
  *inrd = 0;
  ierr  = PetscNew(&rd);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Options for nonequilibrium radiation-diffusion with RD ionization",NULL);CHKERRQ(ierr);
  {
    rd->initial = 1;
    ierr = PetscOptionsInt("-rd_initial","Initial condition (1=Marshak, 2=Blast, 3=Marshak+)","",rd->initial,&rd->initial,0);CHKERRQ(ierr);
    switch (rd->initial) {
    case 1:
    case 2:
      rd->unit.kilogram = 1.;
      rd->unit.meter    = 1.;
      rd->unit.second   = 1.;
      rd->unit.Kelvin   = 1.;
      break;
    case 3:
      rd->unit.kilogram = 1.e12;
      rd->unit.meter    = 1.;
      rd->unit.second   = 1.e9;
      rd->unit.Kelvin   = 1.;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown initial condition %d",rd->initial);
    }
    /* Fundamental units */
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

    ierr = PetscOptionsBool("-rd_monitor_residual","Display residuals every time they are evaluated","",rd->monitor_residual,&rd->monitor_residual,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-rd_discretization","Discretization type","",DiscretizationTypes,(PetscEnum)rd->discretization,(PetscEnum*)&rd->discretization,NULL);CHKERRQ(ierr);
    if (rd->discretization == DISCRETIZATION_FE) {
      rd->quadrature = QUADRATURE_GAUSS2;
      ierr = PetscOptionsEnum("-rd_quadrature","Finite element quadrature","",QuadratureTypes,(PetscEnum)rd->quadrature,(PetscEnum*)&rd->quadrature,NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-rd_jacobian","Type of finite difference Jacobian","",JacobianTypes,(PetscEnum)rd->jacobian,(PetscEnum*)&rd->jacobian,NULL);CHKERRQ(ierr);
    switch (rd->initial) {
    case 1:
      rd->leftbc     = BC_ROBIN;
      rd->Eapplied   = 4 * rd->unit.Joule / PetscPowRealInt(rd->unit.meter,3);
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
      rd->Eapplied   = 7.503e6 * rd->unit.Joule / PetscPowRealInt(rd->unit.meter,3);
      rd->L          = 5. * rd->unit.meter;
      rd->beta       = 3.5;
      rd->gamma      = 3.5;
      rd->final_time = 20e-9 * second;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Initial %D",rd->initial);
    }
    ierr = PetscOptionsEnum("-rd_leftbc","Left boundary condition","",BCTypes,(PetscEnum)rd->leftbc,(PetscEnum*)&rd->leftbc,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_E_applied","Radiation flux at left end of domain","",rd->Eapplied,&rd->Eapplied,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_beta","Thermal exponent for photon absorption","",rd->beta,&rd->beta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rd_gamma","Thermal exponent for diffusion coefficient","",rd->gamma,&rd->gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-rd_view_draw","Draw final solution","",rd->view_draw,&rd->view_draw,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-rd_endpoint","Discretize using endpoints (like trapezoid rule) instead of midpoint","",rd->endpoint,&rd->endpoint,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-rd_bcmidpoint","Impose the boundary condition at the midpoint (Theta) of the interval","",rd->bcmidpoint,&rd->bcmidpoint,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-rd_bclimit","Limit diffusion coefficient in definition of Robin boundary condition","",rd->bclimit,&rd->bclimit,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-rd_test_diff","Test differentiation in constitutive relations","",rd->test_diff,&rd->test_diff,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-rd_view_binary","File name to hold final solution","",rd->view_binary,rd->view_binary,sizeof(rd->view_binary),NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  switch (rd->initial) {
  case 1:
  case 2:
    rd->rho            = 1.;
    rd->c              = 1.;
    rd->K_R            = 1.;
    rd->K_p            = 1.;
    rd->sigma_b        = 0.25;
    rd->MaterialEnergy = RDMaterialEnergy_Reduced;
    break;
  case 3:
    /* Table 2 */
    rd->rho     = 1.17e-3 * kilogram / (meter*meter*meter);                      /* density */
    rd->K_R     = 7.44e18 * PetscPowRealInt(meter,5) * PetscPowReal(Kelvin,3.5) * PetscPowRealInt(kilogram,-2); /*  */
    rd->K_p     = 2.33e20 * PetscPowRealInt(meter,5) * PetscPowReal(Kelvin,3.5) * PetscPowRealInt(kilogram,-2); /*  */
    rd->I_H     = 2.179e-18 * Joule;                                             /* Hydrogen ionization potential */
    rd->m_p     = 1.673e-27 * kilogram;                                          /* proton mass */
    rd->m_e     = 9.109e-31 * kilogram;                                          /* electron mass */
    rd->h       = 6.626e-34 * Joule * second;                                    /* Planck's constant */
    rd->k       = 1.381e-23 * Joule / Kelvin;                                    /* Boltzman constant */
    rd->c       = 3.00e8 * meter / second;                                       /* speed of light */
    rd->sigma_b = 5.67e-8 * Watt * PetscPowRealInt(meter,-2) * PetscPowRealInt(Kelvin,-4);             /* Stefan-Boltzman constant */
    rd->MaterialEnergy = RDMaterialEnergy_Saha;
    break;
  }

  ierr = DMDACreate1d(comm,DM_BOUNDARY_NONE,20,sizeof(RDNode)/sizeof(PetscScalar),1,NULL,&rd->da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(rd->da);CHKERRQ(ierr);
  ierr = DMSetUp(rd->da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(rd->da,0,"E");CHKERRQ(ierr);
  ierr = DMDASetFieldName(rd->da,1,"T");CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(rd->da,0.,1.,0.,0.,0.,0.);CHKERRQ(ierr);

  *inrd = rd;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  RD             rd;
  TS             ts;
  SNES           snes;
  Vec            X;
  Mat            A,B;
  PetscInt       steps;
  PetscReal      ftime;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = RDCreate(PETSC_COMM_WORLD,&rd);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(rd->da,&X);CHKERRQ(ierr);
  ierr = DMSetMatType(rd->da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(rd->da,&B);CHKERRQ(ierr);
  ierr = RDInitialState(rd,X);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetDM(ts,rd->da);CHKERRQ(ierr);
  switch (rd->discretization) {
  case DISCRETIZATION_FD:
    ierr = TSSetIFunction(ts,NULL,RDIFunction_FD,rd);CHKERRQ(ierr);
    if (rd->jacobian == JACOBIAN_ANALYTIC) {ierr = TSSetIJacobian(ts,B,B,RDIJacobian_FD,rd);CHKERRQ(ierr);}
    break;
  case DISCRETIZATION_FE:
    ierr = TSSetIFunction(ts,NULL,RDIFunction_FE,rd);CHKERRQ(ierr);
    if (rd->jacobian == JACOBIAN_ANALYTIC) {ierr = TSSetIJacobian(ts,B,B,RDIJacobian_FE,rd);CHKERRQ(ierr);}
    break;
  }
  ierr = TSSetMaxTime(ts,rd->final_time);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-3);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  A = B;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  switch (rd->jacobian) {
  case JACOBIAN_ANALYTIC:
    break;
  case JACOBIAN_MATRIXFREE:
    break;
  case JACOBIAN_FD_COLORING: {
    ierr = SNESSetJacobian(snes,A,B,SNESComputeJacobianDefaultColor,0);CHKERRQ(ierr);
  } break;
  case JACOBIAN_FD_FULL:
    ierr = SNESSetJacobian(snes,A,B,SNESComputeJacobianDefault,ts);CHKERRQ(ierr);
    break;
  }

  if (rd->test_diff) {
    ierr = RDTestDifferentiation(rd);CHKERRQ(ierr);
  }
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Steps %D  final time %g\n",steps,(double)ftime);CHKERRQ(ierr);
  if (rd->view_draw) {
    ierr = RDView(rd,X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  if (rd->view_binary[0]) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,rd->view_binary,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = RDView(rd,X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = RDDestroy(&rd);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/*TEST

    test:
      args: -da_grid_x 20 -rd_initial 1 -rd_discretization fd -rd_jacobian fd_coloring -rd_endpoint -ts_max_time 1 -ts_dt 2e-1 -ts_theta_initial_guess_extrapolate 0 -ts_monitor -snes_monitor_short -ksp_monitor_short
      requires: !single

    test:
      suffix: 2
      args: -da_grid_x 20 -rd_initial 1 -rd_discretization fe -rd_quadrature lobatto2 -rd_jacobian fd_coloring -rd_endpoint -ts_max_time 1 -ts_dt 2e-1 -ts_theta_initial_guess_extrapolate 0 -ts_monitor -snes_monitor_short -ksp_monitor_short
      requires: !single

    test:
      suffix: 3
      nsize: 2
      args: -da_grid_x 20 -rd_initial 1 -rd_discretization fd -rd_jacobian analytic -rd_endpoint -ts_max_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 -ts_monitor -snes_monitor_short -ksp_monitor_short
      requires: !single

TEST*/
