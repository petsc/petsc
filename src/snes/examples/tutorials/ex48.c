static const char help[] = "Toy hydrostatic ice flow with multigrid in 3D\n\
\n\
Solves the hydrostatic (aka Blatter/Pattyn/First Order) equations for ice sheet flow\n\
using multigrid.  The ice uses a power-law rheology with \"Glen\" exponent 3 (corresponds\n\
to p=1.33 in a p-Laplacian).  The focus is on ISMIP-HOM experiments which assume periodic\n\
boundary conditions in the x- and y-directions.\n\
\n\
Equations are rescaled so that the domain size and solution are O(1), details of this scaling\n\
can be controlled by the options -units_meter, -units_second, and -units_kilogram\n\
\n\n";

/*
The equations for horizontal velocity (u,v) are

  - [eta (4 u_x + 2 v_y)]_x - [eta (u_y + v_x)]_y - [eta u_z]_z + rho g s_x = 0
  - [eta (4 v_y + 2 u_x)]_y - [eta (u_y + v_x)]_x - [eta v_z]_z + rho g s_y = 0

where

  eta = B/2 (epsilon + gamma)^((p-2)/2)

is the nonlinear effective viscosity with regularization epsilon and hardness parameter B,
written in terms of the second invariant

  gamma = u_x^2 + v_y^2 + u_x v_y + (1/4) (u_y + v_x)^2 + (1/4) u_z^2 + (1/4) v_z^2

The surface boundary conditions are the natural conditions.  The basal boundary conditions
are either no-slip, or Navier (linear) slip with spatially variant friction coefficient beta^2.

The discretization is Q1 finite elements, managed by a DA.  The grid is never distorted in the
map (x,y) plane, but the bed and surface may be bumpy.  This is handled as usual in FEM, through
the Jacobian of the coordinate transformation from a reference element to the physical element.

Since ice-flow is tightly coupled in the z-direction (within columns), the DA is managed
specially so that columns are never distributed, and are always contiguous in memory.
This amounts to reversing the meaning of X,Y,Z compared to the DA's internal interpretation,
and then indexing as vec[i][j][k].  The exotic coarse spaces require 2D DAs which are made to
use compatible domain decomposition relative to the 3D DAs.
*/

#include <petscdmmg.h>
#include <ctype.h>

static const PetscReal HexQWeights[8] = {1,1,1,1,1,1,1,1};
static const PetscReal HexQNodes[]    = {-0.57735026918962573, 0.57735026918962573};
#define G 0.57735026918962573
#define H (0.5*(1.+G))
#define L (0.5*(1.-G))
#define M (-0.5)
#define P (0.5)
/* Special quadrature: Lobatto in horizontal, Gauss in vertical */
static const PetscReal HexQInterp[8][8] = {{H,0,0,0,L,0,0,0},
                                           {0,H,0,0,0,L,0,0},
                                           {0,0,H,0,0,0,L,0},
                                           {0,0,0,H,0,0,0,L},
                                           {L,0,0,0,H,0,0,0},
                                           {0,L,0,0,0,H,0,0},
                                           {0,0,L,0,0,0,H,0},
                                           {0,0,0,L,0,0,0,H}};
static const PetscReal HexQDeriv[8][8][3] = {
  {{M*H,M*H,M},{P*H,0,0}  ,{0,0,0}    ,{0,P*H,0}  ,{M*L,M*L,P},{P*L,0,0}  ,{0,0,0}    ,{0,P*L,0}  },
  {{M*H,0,0}  ,{P*H,M*H,M},{0,P*H,0}  ,{0,0,0}    ,{M*L,0,0}  ,{P*L,M*L,P},{0,P*L,0}  ,{0,0,0}    },
  {{0,0,0}    ,{0,M*H,0}  ,{P*H,P*H,M},{M*H,0,0}  ,{0,0,0}    ,{0,M*L,0}  ,{P*L,P*L,P},{M*L,0,0}  },
  {{0,M*H,0}  ,{0,0,0}    ,{P*H,0,0}  ,{M*H,P*H,M},{0,M*L,0}  ,{0,0,0}    ,{P*L,0,0}  ,{M*L,P*L,M}},
  {{M*L,M*L,M},{P*L,0,0}  ,{0,0,0}    ,{0,P*L,0}  ,{M*H,M*H,P},{P*H,0,0}  ,{0,0,0}    ,{0,P*H,0}  },
  {{M*L,0,0}  ,{P*L,M*L,M},{0,P*L,0}  ,{0,0,0}    ,{M*H,0,0}  ,{P*H,M*H,P},{0,P*H,0}  ,{0,0,0}    },
  {{0,0,0}    ,{0,M*L,0}  ,{P*L,P*L,M},{M*L,0,0}  ,{0,0,0}    ,{0,M*H,0}  ,{P*H,P*H,P},{M*H,0,0}  },
  {{0,M*L,0}  ,{0,0,0}    ,{P*L,0,0}  ,{M*L,P*L,M},{0,M*H,0}  ,{0,0,0}    ,{P*H,0,0}  ,{M*H,P*H,P}}};
/* Standard 2x2 Gauss quadrature for the bottom layer. */
static const PetscReal QuadQInterp[4][4] = {{H*H,L*H,L*L,H*L},
                                            {L*H,H*H,H*L,L*L},
                                            {L*L,H*L,H*H,L*H},
                                            {H*L,L*L,L*H,H*H}};
static const PetscReal QuadQDeriv[4][4][2] = {
  {{M*H,M*H},{P*H,M*L},{P*L,P*L},{M*L,P*H}},
  {{M*H,M*L},{P*H,M*H},{P*L,P*H},{M*L,P*L}},
  {{M*L,M*L},{P*L,M*H},{P*H,P*H},{M*H,P*L}},
  {{M*L,M*H},{P*L,M*L},{P*H,P*L},{M*H,P*H}}};
#undef G
#undef H
#undef L
#undef M
#undef P

#define HexExtract(x,i,j,k,n) do {              \
    (n)[0] = (x)[i][j][k];                      \
    (n)[1] = (x)[i+1][j][k];                    \
    (n)[2] = (x)[i+1][j+1][k];                  \
    (n)[3] = (x)[i][j+1][k];                    \
    (n)[4] = (x)[i][j][k+1];                    \
    (n)[5] = (x)[i+1][j][k+1];                  \
    (n)[6] = (x)[i+1][j+1][k+1];                \
    (n)[7] = (x)[i][j+1][k+1];                  \
  } while (0)

#define HexExtractRef(x,i,j,k,n) do {           \
    (n)[0] = &(x)[i][j][k];                     \
    (n)[1] = &(x)[i+1][j][k];                   \
    (n)[2] = &(x)[i+1][j+1][k];                 \
    (n)[3] = &(x)[i][j+1][k];                   \
    (n)[4] = &(x)[i][j][k+1];                   \
    (n)[5] = &(x)[i+1][j][k+1];                 \
    (n)[6] = &(x)[i+1][j+1][k+1];               \
    (n)[7] = &(x)[i][j+1][k+1];                 \
  } while (0)

#define QuadExtract(x,i,j,n) do {               \
    (n)[0] = (x)[i][j];                         \
    (n)[1] = (x)[i+1][j];                       \
    (n)[2] = (x)[i+1][j+1];                     \
    (n)[3] = (x)[i][j+1];                       \
  } while (0)

static PetscScalar Sqr(PetscScalar a) {return a*a;}

static void HexGrad(const PetscReal dphi[][3],const PetscReal zn[],PetscReal dz[])
{
  PetscInt i;
  dz[0] = dz[1] = dz[2] = 0;
  for (i=0; i<8; i++) {
    dz[0] += dphi[i][0] * zn[i];
    dz[1] += dphi[i][1] * zn[i];
    dz[2] += dphi[i][2] * zn[i];
  }
}

static void HexComputeGeometry(PetscInt q,PetscReal hx,PetscReal hy,const PetscReal dz[],PetscReal phi[],PetscReal dphi[][3],PetscReal *jw)
{
  const PetscReal
    jac[3][3] = {{hx/2,0,0}, {0,hy/2,0}, {dz[0],dz[1],dz[2]}}
  ,ijac[3][3] = {{1/jac[0][0],0,0}, {0,1/jac[1][1],0}, {-jac[2][0]/(jac[0][0]*jac[2][2]),-jac[2][1]/(jac[1][1]*jac[2][2]),1/jac[2][2]}}
  ,jdet = jac[0][0]*jac[1][1]*jac[2][2];
  PetscInt i;

  for (i=0; i<8; i++) {
    const PetscReal *dphir = HexQDeriv[q][i];
    phi[i] = HexQInterp[q][i];
    dphi[i][0] = dphir[0]*ijac[0][0] + dphir[1]*ijac[1][0] + dphir[2]*ijac[2][0];
    dphi[i][1] = dphir[0]*ijac[0][1] + dphir[1]*ijac[1][1] + dphir[2]*ijac[2][1];
    dphi[i][2] = dphir[0]*ijac[0][2] + dphir[1]*ijac[1][2] + dphir[2]*ijac[2][2];
  }
  *jw = 1.0 * jdet;
}

typedef struct _n_THI   *THI;
typedef struct _n_Units *Units;

typedef struct {
  PetscScalar u,v;
} Node;

typedef struct {
  PetscScalar b;                /* bed */
  PetscScalar h;                /* thickness */
  PetscScalar beta2;            /* friction */
} PrmNode;

struct _n_THI {
  MPI_Comm  comm;
  void (*initialize)(THI,PetscReal x,PetscReal y,PrmNode *p);
  PetscInt  nlevels;
  PetscReal Lx,Ly,Lz;           /* Model domain */
  PetscReal alpha;              /* Bed angle */
  Units     units;
  PetscReal dirichlet_scale;
  PetscReal etamin,etamax,cetamin,cetamax;
  struct {
    PetscReal Bd2,eps,exponent;
  } viscosity;
  PetscReal rhog;
  PetscTruth no_slip;
  PetscTruth debug;
};

struct _n_Units {
  /* fundamental */
  PetscReal meter;
  PetscReal kilogram;
  PetscReal second;
  /* derived */
  PetscReal Pascal;
  PetscReal year;
};

static void PrmHexGetZ(const PrmNode pn[],PetscInt k,PetscInt zm,PetscReal zn[])
{
  const PetscScalar zm1 = zm-1,
    znl[8] = {pn[0].b + pn[0].h*(PetscScalar)k/zm1,
              pn[1].b + pn[1].h*(PetscScalar)k/zm1,
              pn[2].b + pn[2].h*(PetscScalar)k/zm1,
              pn[3].b + pn[3].h*(PetscScalar)k/zm1,
              pn[0].b + pn[0].h*(PetscScalar)(k+1)/zm1,
              pn[1].b + pn[1].h*(PetscScalar)(k+1)/zm1,
              pn[2].b + pn[2].h*(PetscScalar)(k+1)/zm1,
              pn[3].b + pn[3].h*(PetscScalar)(k+1)/zm1};
  PetscInt i;
  for (i=0; i<8; i++) zn[i] = PetscRealPart(znl[i]);
}


static void THIInitialize_HOM_A(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units units = thi->units;
  PetscReal s = -x*tan(thi->alpha);
  p->b = s - 1000*units->meter + 500*units->meter * sin(x*2*PETSC_PI/thi->Lx) * sin(y*2*PETSC_PI/thi->Ly);
  p->h = s - p->b;
  p->beta2 = 1e30;
}

static void THIInitialize_HOM_C(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units units = thi->units;
  PetscReal s = -x*tan(thi->alpha);
  p->b = s - 1000*units->meter;
  p->h = s - p->b;
  /* tau_b = beta2 v   is a stress (Pa) */
  p->beta2 = 1000 * (1 + sin(x*2*PETSC_PI/thi->Lx)*sin(y*2*PETSC_PI/thi->Ly)) * units->Pascal * units->year / units->meter;
}

/* This one is just a toy */
static void THIInitialize_HOM_Z(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units units = thi->units;
  PetscReal s = -x*tan(thi->alpha);
  p->b = s - 1000*units->meter + 500*units->meter * sin(x*2*PETSC_PI/thi->Lx);
  p->h = s - p->b;
  p->beta2 = 1e30;
}

static void THIViscosity(THI thi,PetscReal gam,PetscReal *eta,PetscReal *deta)
{
  PetscReal Bd2,eps,exponent;
  if (thi->viscosity.Bd2 == 0) {
    Units units = thi->units;
    const PetscReal
      n = 3.,                                           /* Glen exponent */
      p = 1. + 1./n,                                    /* for Stokes */
      A = 1.e-16 * pow(units->Pascal,-n) / units->year, /* softness parameter (Pa^{-n}/s) */
      B = pow(A,-1./n);                                 /* hardness parameter */
    thi->viscosity.Bd2      = B/2;
    thi->viscosity.exponent = (p-2)/2;
    thi->viscosity.eps      = 0.5*PetscSqr(1e-5 / units->year);
  }
  Bd2      = thi->viscosity.Bd2;
  exponent = thi->viscosity.exponent;
  eps      = thi->viscosity.eps;
  *eta = Bd2 * pow(eps + gam,exponent);
  *deta = exponent * (*eta) / (eps + gam);
}

#undef __FUNCT__  
#define __FUNCT__ "THICreate"
static PetscErrorCode THICreate(MPI_Comm comm,THI *inthi)
{
  THI thi;
  Units units;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *inthi = 0;
  ierr = PetscNew(struct _n_THI,&thi);CHKERRQ(ierr);
  ierr = PetscNew(struct _n_Units,&units);CHKERRQ(ierr);
  thi->comm  = comm;
  thi->units = units;

  units->meter  = 1e-2;
  units->second = 1e-7;
  units->kilogram = 1e-12;
  ierr = PetscOptionsBegin(comm,NULL,"Scaled units options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-units_meter","1 meter in scaled length units","",units->meter,&units->meter,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-units_second","1 second in scaled time units","",units->second,&units->second,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-units_kilogram","1 kilogram in scaled mass units","",units->kilogram,&units->kilogram,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));
  units->year = 31556926. * units->second, /* seconds per year */

  thi->Lx              = 10.e3;
  thi->Ly              = 10.e3;
  thi->Lz              = 1000;
  thi->nlevels         = 1;
  thi->dirichlet_scale = 1;
  thi->debug           = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm,NULL,"Toy Hydrostatic Ice options","");CHKERRQ(ierr);
  {
    char homexp[] = "A";
    PetscReal L;
    PetscTruth flg;
    L = thi->Lx;
    ierr = PetscOptionsReal("-thi_L","Domain size (m)","",L,&L,&flg);CHKERRQ(ierr);
    if (flg) thi->Lx = thi->Ly = L;
    ierr = PetscOptionsReal("-thi_Lx","X Domain size (m)","",thi->Lx,&thi->Lx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_Ly","Y Domain size (m)","",thi->Ly,&thi->Ly,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_Lz","Z Domain size (m)","",thi->Lz,&thi->Lz,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-thi_hom","ISMIP-HOM experiment (A or C)","",homexp,homexp,sizeof(homexp),NULL);CHKERRQ(ierr);
    switch (homexp[0] = toupper(homexp[0])) {
      case 'A':
        thi->initialize = THIInitialize_HOM_A;
        thi->no_slip = PETSC_TRUE;
        thi->alpha = 0.5;
        break;
      case 'C':
        thi->initialize = THIInitialize_HOM_C;
        thi->no_slip = PETSC_FALSE;
        thi->alpha = 0.1;
        break;
      case 'Z':
        thi->initialize = THIInitialize_HOM_Z;
        thi->no_slip = PETSC_TRUE;
        thi->alpha = 5;
        break;
      default:
        SETERRQ1(PETSC_ERR_SUP,"HOM experiment '%c' not implemented",homexp[0]);
    }
    ierr = PetscOptionsReal("-thi_alpha","Bed angle (degrees)","",thi->alpha,&thi->alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_dirichlet_scale","Scale Dirichlet boundary conditions by this factor","",thi->dirichlet_scale,&thi->dirichlet_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-thi_nlevels","Number of levels of refinement","",thi->nlevels,&thi->nlevels,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-thi_debug","Enable debugging output","",thi->debug,&thi->debug,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* dimensionalize */
  thi->Lx     *= units->meter;
  thi->Ly     *= units->meter;
  thi->Lz     *= units->meter;
  thi->alpha  *= PETSC_PI / 180;

  thi->etamin = 1e100;
  thi->etamax = 0;

  {
    PetscReal u = 1000*units->meter/(3e7*units->second),
      gradu = u / (100*units->meter),eta,deta,
      rho = 910 * units->kilogram/pow(units->meter,3),
      grav = 9.81 * units->meter/PetscSqr(units->second),
      driving = rho * grav * tan(thi->alpha) * 1000*units->meter;
    THIViscosity(thi,0.5*gradu*gradu,&eta,&deta);
    thi->rhog = rho * grav;
    if (thi->debug) {
      ierr = PetscPrintf(thi->comm,"Units: meter %8.2g  second %8.2g  kg %8.2g  Pa %8.2g\n",units->meter,units->second,units->kilogram,units->Pascal);CHKERRQ(ierr);
      ierr = PetscPrintf(thi->comm,"Domain (%6.2g,%6.2g,%6.2g), pressure %8.2g, driving stress %8.2g\n",thi->Lx,thi->Ly,thi->Lz,rho*grav*1e3*units->meter,driving);CHKERRQ(ierr);
      ierr = PetscPrintf(thi->comm,"Large velocity 1km/a %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",u,gradu,eta,2*eta*gradu,2*eta*gradu/driving);CHKERRQ(ierr);
      THIViscosity(thi,0.5*PetscSqr(1e-3*gradu),&eta,&deta);
      ierr = PetscPrintf(thi->comm,"Small velocity 1m/a  %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",1e-3*u,1e-3*gradu,eta,2*eta*1e-3*gradu,2*eta*1e-3*gradu/driving);CHKERRQ(ierr);
    }
  }

  *inthi = thi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIDestroy"
static PetscErrorCode THIDestroy(THI thi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(thi->units);CHKERRQ(ierr);
  ierr = PetscFree(thi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIInitializePrm"
static PetscErrorCode THIInitializePrm(THI thi,DA da2prm,Vec prm)
{
  PrmNode **p;
  PetscInt i,j,xs,xm,ys,ym,mx,my;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetGhostCorners(da2prm,&ys,&xs,0,&ym,&xm,0);CHKERRQ(ierr);
  ierr = DAGetInfo(da2prm,0, &my,&mx,0, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2prm,prm,&p);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PetscReal xx = thi->Lx*i/mx,yy = thi->Ly*j/my;
      thi->initialize(thi,xx,yy,&p[i][j]);
    }
  }
  ierr = DAVecRestoreArray(da2prm,prm,&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THISetDMMG"
static PetscErrorCode THISetDMMG(THI thi,DMMG *dmmg)
{
  PetscErrorCode ierr;
  DAStencilType  st;
  PetscInt i,Mx,My,Mz,mx,my,mz,s,nlevels;

  PetscFunctionBegin;
  nlevels = DMMGGetLevels(dmmg);
  if (nlevels != thi->nlevels) SETERRQ(PETSC_ERR_ARG_CORRUPT,"DMMG nlevels does not agree with THI");
  for (i=0; i<nlevels; i++) {
    DA da3 = (DA)dmmg[i]->dm,da2prm;
    Vec X;
    ierr = DAGetInfo(da3,0, &Mz,&My,&Mx, &mz,&my,&mx, 0,&s,0,&st);CHKERRQ(ierr);
    ierr = DACreate2d(thi->comm,DA_XYPERIODIC,st,My,Mx,my,mx,sizeof(PrmNode)/sizeof(PetscScalar),s,0,0,&da2prm);CHKERRQ(ierr);
    ierr = DACreateLocalVector(da2prm,&X);CHKERRQ(ierr);
    {
      PetscReal Lx = thi->Lx / thi->units->meter,Ly = thi->Ly / thi->units->meter,Lz = thi->Lz / thi->units->meter;
      ierr = PetscPrintf(thi->comm,"Level %d domain size (m) %g x %g x %g, num elements %d x %d x %d, size (m) %g x %g x %g\n",i,Lx,Ly,Lz,Mx,My,Mz,Lx/Mx,Ly/My,1000./(Mz-1));CHKERRQ(ierr);
    }
    ierr = THIInitializePrm(thi,da2prm,X);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da3,"DA2Prm",(PetscObject)da2prm);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da3,"DA2Prm_Vec",(PetscObject)X);CHKERRQ(ierr);
    ierr = DADestroy(da2prm);CHKERRQ(ierr);
    ierr = VecDestroy(X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIDAGetPrm"
static PetscErrorCode THIDAGetPrm(DA da,PrmNode ***prm)
{
  PetscErrorCode ierr;
  DA             da2prm;
  Vec            X;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)da,"DA2Prm",(PetscObject*)&da2prm);CHKERRQ(ierr);
  if (!da2prm) SETERRQ(PETSC_ERR_ARG_WRONG,"No DA2Prm composed with given DA");
  ierr = PetscObjectQuery((PetscObject)da,"DA2Prm_Vec",(PetscObject*)&X);CHKERRQ(ierr);
  if (!X) SETERRQ(PETSC_ERR_ARG_WRONG,"No DA2Prm_Vec composed with given DA");
  ierr = DAVecGetArray(da2prm,X,prm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIDARestorePrm"
static PetscErrorCode THIDARestorePrm(DA da,PrmNode ***prm)
{
  PetscErrorCode ierr;
  DA             da2prm;
  Vec            X;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)da,"DA2Prm",(PetscObject*)&da2prm);CHKERRQ(ierr);
  if (!da2prm) SETERRQ(PETSC_ERR_ARG_WRONG,"No DA2Prm composed with given DA");
  ierr = PetscObjectQuery((PetscObject)da,"DA2Prm_Vec",(PetscObject*)&X);CHKERRQ(ierr);
  if (!X) SETERRQ(PETSC_ERR_ARG_WRONG,"No DA2Prm_Vec composed with given DA");
  ierr = DAVecRestoreArray(da2prm,X,prm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIInitial"
static PetscErrorCode THIInitial(DMMG dmmg,Vec X)
{
  THI         thi   = (THI)dmmg->user;
  DA          da    = (DA)dmmg->dm;
  PetscInt    i,j,k,xs,xm,ys,ym,zs,zm,mx,my;
  PetscReal   hx,hy;
  PrmNode     **prm;
  Node        ***x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0, 0,&my,&mx, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = THIDAGetPrm(da,&prm);CHKERRQ(ierr);
  hx = thi->Lx / mx;
  hy = thi->Ly / my;
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      for (k=zs; k<zs+zm; k++) {
        const PetscScalar zm1 = zm-1,
          drivingx = thi->rhog * (prm[i+1][j].b+prm[i+1][j].h - prm[i-1][j].b-prm[i-1][j].h) / (2*hx),
          drivingy = thi->rhog * (prm[i][j+1].b+prm[i][j+1].h - prm[i][j-1].b-prm[i][j-1].h) / (2*hx);
        x[i][j][k].u = 0. * drivingx * prm[i][j].h*(PetscScalar)k/zm1;
        x[i][j][k].v = 0. * drivingy * prm[i][j].h*(PetscScalar)k/zm1;
      }
    }
  }
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = THIDARestorePrm(da,&prm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void PointwiseNonlinearity(THI thi,Node n[],PetscReal phi[],PetscReal dphi[8][3],PetscScalar *u,PetscScalar *v,PetscScalar du[],PetscScalar dv[],PetscReal *eta,PetscReal *deta)
{
  PetscInt l,ll;
  PetscScalar gam;

  du[0] = du[1] = du[2] = 0;
  dv[0] = dv[1] = dv[2] = 0;
  *u = 0;
  *v = 0;
  for (l=0; l<8; l++) {
    *u += phi[l] * n[l].u;
    *v += phi[l] * n[l].v;
    for (ll=0; ll<3; ll++) {
      du[ll] += dphi[l][ll] * n[l].u;
      dv[ll] += dphi[l][ll] * n[l].v;
    }
  }
  gam = Sqr(du[0]) + Sqr(dv[1]) + du[0]*dv[1] + 0.25*Sqr(du[1]+dv[0]) + 0.25*Sqr(du[2]) + 0.25*Sqr(dv[2]);
  THIViscosity(thi,PetscRealPart(gam),eta,deta);
}


#undef __FUNCT__  
#define __FUNCT__ "THIFunctionLocal"
static PetscErrorCode THIFunctionLocal(DALocalInfo *info,Node ***x,Node ***f,THI thi)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l;
  PetscReal      hx,hy,etamin,etamax;
  PrmNode        **prm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;

  etamin = 1e100;
  etamax = 0;

  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  ierr = THIDAGetPrm(info->da,&prm);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm,i,j,pn);
      for (k=0; k<zm-1; k++) {
        PetscInt ls = 0;
        Node n[8],*fn[8];
        PetscReal zn[8],etabase = 0;
        PrmHexGetZ(pn,k,zm,zn);
        HexExtract(x,i,j,k,n);
        HexExtractRef(f,i,j,k,fn);
        if (thi->no_slip && k == 0) {
          for (l=0; l<4; l++) n[l].u = n[l].v = 0;
          /* The first 4 basis functions lie on the bottom layer, so their contribution is exactly 0, hence we can skip them */
          ls = 4;
        }
        for (q=0; q<8; q++) {
          PetscReal dz[3],phi[8],dphi[8][3],jw,eta,deta;
          PetscScalar du[3],dv[3],u,v;
          HexGrad(HexQDeriv[q],zn,dz);
          HexComputeGeometry(q,hx,hy,dz,phi,dphi,&jw);
          PointwiseNonlinearity(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
          if (q == 0) etabase = eta;
          if (eta > etamax)      etamax = eta;
          else if (eta < etamin) etamin = eta;
          for (l=ls; l<8; l++) { /* test functions */
            const PetscReal rhog = thi->rhog,ds[2] = {-tan(thi->alpha),0};
            const PetscReal pp=phi[l],*dp = dphi[l];
            fn[l]->u += dp[0]*jw*eta*(4.*du[0]+2.*dv[1]) + dp[1]*jw*eta*(du[1]+dv[0]) + dp[2]*jw*eta*du[2] + pp*jw*rhog*ds[0];
            fn[l]->v += dp[1]*jw*eta*(2.*du[0]+4.*dv[1]) + dp[0]*jw*eta*(du[1]+dv[0]) + dp[2]*jw*eta*dv[2] + pp*jw*rhog*ds[1];
          }
        }
        if (k == 0) { /* we are on a bottom face */
          if (thi->no_slip) {
            /* Note: Non-Galerkin coarse grid operators are very sensitive to the scaling of Dirichlet boundary
            * conditions.  After shenanigans above, etabase contains the effective viscosity at the closest quadrature
            * point to the bed.  We want the diagonal entry in the Dirichlet condition to have similar magnitude to the
            * diagonal entry corresponding to the adjacent node.  The fundamental scaling of the viscous part is in
            * diagu, diagv below.  This scaling is easy to recognize by considering the finite difference operator after
            * scaling by element size.  The no-slip Dirichlet condition is scaled by this factor, and also in the
            * assembled matrix (see the similar block in THIJacobianLocal).
            */
            const PetscReal hz = PetscRealPart(pn[0].h)/(zm-1.);
            const PetscScalar diagu = 2*etabase*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            fn[0]->u = thi->dirichlet_scale*diagu*n[0].u;
            fn[0]->v = thi->dirichlet_scale*diagv*n[0].v;
          } else {
            for (q=0; q<4; q++) {
              const PetscReal jw = 0.25*hx*hy,*phi = QuadQInterp[q];
              PetscScalar u=0,v=0,beta2=0;
              for (l=0; l<4; l++) {
                u     += phi[l]*n[l].u;
                v     += phi[l]*n[l].v;
                beta2 += phi[l]*pn[l].beta2;
              }
              for (l=0; l<4; l++) {
                const PetscReal pp = phi[l];
                fn[ls+l]->u -= pp*jw*beta2*u;
                fn[ls+l]->v -= pp*jw*beta2*v;
              }
            }
          }
        }
      }
    }
  }

  ierr = THIDARestorePrm(info->da,&prm);CHKERRQ(ierr);

  thi->cetamin = etamin;
  thi->cetamax = etamax;
  if (etamin < thi->etamin) thi->etamin = etamin;
  if (etamax > thi->etamax) thi->etamax = etamax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIJacobianLocal"
static PetscErrorCode THIJacobianLocal(DALocalInfo *info,Node ***x,Mat B,THI thi)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l,ll;
  PetscReal      hx,hy;
  PrmNode        **prm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;

  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  ierr = MatZeroEntries(B);CHKERRQ(ierr);

  ierr = THIDAGetPrm(info->da,&prm);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm,i,j,pn);
      for (k=0; k<zm-1; k++) {
        Node n[8];
        PetscReal zn[8],etabase = 0;
        PetscScalar Ke[8*2][8*2];
        PetscInt ls = 0;

        PrmHexGetZ(pn,k,zm,zn);
        HexExtract(x,i,j,k,n);
        PetscMemzero(Ke,sizeof(Ke));
        if (thi->no_slip && k == 0) {
          for (l=0; l<4; l++) n[l].u = n[l].v = 0;
          ls = 4;
        }
        for (q=0; q<8; q++) {
          PetscReal dz[3],phi[8],dphi[8][3],jw,eta,deta;
          PetscScalar du[3],dv[3],u,v;
          HexGrad(HexQDeriv[q],zn,dz);
          HexComputeGeometry(q,hx,hy,dz,phi,dphi,&jw);
          PointwiseNonlinearity(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
          if (q == 0) etabase = eta;
          for (l=ls; l<8; l++) { /* test functions */
            const PetscReal *dp = dphi[l];
            for (ll=ls; ll<8; ll++) { /* trial functions */
              const PetscReal *dpl = dphi[ll];
              PetscScalar dgdu,dgdv;
              dgdu = 2.*du[0]*dpl[0] + dv[1]*dpl[0] + 0.5*(du[1]+dv[0])*dpl[1] + 0.5*du[2]*dpl[2];
              dgdv = 2.*dv[1]*dpl[1] + du[0]*dpl[1] + 0.5*(du[1]+dv[0])*dpl[0] + 0.5*dv[2]*dpl[2];
              /* Picard part */
              Ke[l*2+0][ll*2+0] += dp[0]*jw*eta*4.*dpl[0] + dp[1]*jw*eta*dpl[1] + dp[2]*jw*eta*dpl[2];
              Ke[l*2+0][ll*2+1] += dp[0]*jw*eta*2.*dpl[1] + dp[1]*jw*eta*dpl[0];
              Ke[l*2+1][ll*2+0] += dp[1]*jw*eta*2.*dpl[0] + dp[0]*jw*eta*dpl[1];
              Ke[l*2+1][ll*2+1] += dp[1]*jw*eta*4.*dpl[1] + dp[0]*jw*eta*dpl[0] + dp[2]*jw*eta*dpl[2];
              /* extra Newton terms */
              Ke[l*2+0][ll*2+0] += dp[0]*jw*deta*dgdu*(4.*du[0]+2.*dv[1]) + dp[1]*jw*deta*dgdu*(du[1]+dv[0]) + dp[2]*jw*deta*dgdu*du[2];
              Ke[l*2+0][ll*2+1] += dp[0]*jw*deta*dgdv*(4.*du[0]+2.*dv[1]) + dp[1]*jw*deta*dgdv*(du[1]+dv[0]) + dp[2]*jw*deta*dgdv*du[2];
              Ke[l*2+1][ll*2+0] += dp[1]*jw*deta*dgdu*(4.*dv[1]+2.*du[0]) + dp[0]*jw*deta*dgdu*(du[1]+dv[0]) + dp[2]*jw*deta*dgdu*dv[2];
              Ke[l*2+1][ll*2+1] += dp[1]*jw*deta*dgdv*(4.*dv[1]+2.*du[0]) + dp[0]*jw*deta*dgdv*(du[1]+dv[0]) + dp[2]*jw*deta*dgdv*dv[2];
            }
          }
        }
        if (k == 0) { /* on a bottom face */
          if (thi->no_slip) {
            const PetscReal hz = PetscRealPart(pn[0].h)/(zm-1);
            const PetscScalar diagu = 2*etabase*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            Ke[0][0] = thi->dirichlet_scale*diagu;
            Ke[1][1] = thi->dirichlet_scale*diagv;
          } else {
            for (q=0; q<4; q++) {
              const PetscReal jw = 0.25*hx*hy,*phi = QuadQInterp[q];
              PetscScalar u=0,v=0,beta2=0;
              for (l=0; l<4; l++) {
                u     += phi[l]*n[l].u;
                v     += phi[l]*n[l].v;
                beta2 += phi[l]*pn[l].beta2;
              }
              for (l=0; l<4; l++) {
                const PetscReal pp = phi[l];
                for (ll=0; ll<4; ll++) {
                  const PetscReal ppl = phi[ll];
                  Ke[l*2+0][ll*2+0] -= pp*jw*beta2*ppl;
                  Ke[l*2+1][ll*2+1] -= pp*jw*beta2*ppl;
                }
              }
            }
          }
        }
        {
          const MatStencil rc[8] = {{i,j,k},{i+1,j,k},{i+1,j+1,k},{i,j+1,k},{i,j,k+1},{i+1,j,k+1},{i+1,j+1,k+1},{i,j+1,k+1}};
          ierr = MatSetValuesBlockedStencil(B,8,rc,8,rc,&Ke[0][0],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = THIDARestorePrm(info->da,&prm);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  if (thi->debug) {
    PetscReal nrm;
    PetscInt  m;
    PetscMPIInt rank;
    ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    ierr = MatGetSize(B,&m,0);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(thi->comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      PetscScalar val0,val2;
      ierr = MatGetValue(B,0,0,&val0);CHKERRQ(ierr);
      ierr = MatGetValue(B,2,2,&val2);CHKERRQ(ierr);
      ierr = PetscPrintf(thi->comm,"Matrix dim %8d  norm %g, (0,0) %8.2g  (2,2) %8.2g, eta [%8.2g,%8.2g]\n",m,nrm,PetscRealPart(val0),PetscRealPart(val2),thi->cetamin,thi->cetamax);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIDAVecView_VTK(THI thi,DA da,Vec X,const char filename[])
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscInt mx,my,mz,i,j,k;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0, &mz,&my,&mx, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(thi->comm,filename,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 3.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Toy Hydrostatic Ice model with periodic boundary conditions\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DATASET STRUCTURED_GRID\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DIMENSIONS %d %d %d\n",mz,my,mx);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"POINTS %d double\n",mx*my*mz);CHKERRQ(ierr);
  for (i=0; i<mx; i++) {
    for (j=0; j<my; j++) {
      for (k=0; k<mz; k++) {
        PrmNode p;
        PetscReal xx = thi->Lx*i/mx,yy = thi->Ly*j/my,zz;
        thi->initialize(thi,xx,yy,&p);
        zz = PetscRealPart(p.b) + PetscRealPart(p.h)*k/(mz-1);
        ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",xx,yy,zz);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\nPOINT_DATA %d\n",mx*my*mz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"VECTORS velocity double\n");CHKERRQ(ierr);
  {
    MPI_Comm comm;
    PetscMPIInt rank,size,tag,nn,nmax;
    PetscInt n;
    PetscScalar *array;

    comm = thi->comm;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = VecGetSize(X,&n);CHKERRQ(ierr);
    nn = PetscMPIIntCast(n);
    ierr = VecGetArray(X,&array);CHKERRQ(ierr);
    ierr = MPI_Reduce(&nn,&nmax,1,MPIU_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
    tag  = ((PetscObject) viewer)->tag;
    if (!rank) {
      PetscScalar *values;
      PetscInt p;
      ierr = PetscMalloc((nmax+1)*sizeof(PetscScalar),&values);CHKERRQ(ierr);
      for(i=0; i<n; i+=2) {
        ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",PetscRealPart(array[i]),PetscRealPart(array[i+1]),0);CHKERRQ(ierr);
      }
      for(p=1; p<size; p++) {
        MPI_Status status;
        ierr = MPI_Recv(values,(PetscMPIInt)n,MPIU_SCALAR,p,tag,comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&nn);CHKERRQ(ierr);
        for(i=0; i<nn; i+=2) {
          ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",PetscRealPart(values[i]),PetscRealPart(values[i+1]),0);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree(values);CHKERRQ(ierr);
    } else {
      ierr = MPI_Send(array,n,MPIU_SCALAR,0,tag,comm);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(X,&array);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  MPI_Comm comm;
  DMMG *dmmg;
  THI thi;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = THICreate(comm,&thi);CHKERRQ(ierr);
  ierr = DMMGCreate(PETSC_COMM_WORLD,thi->nlevels,thi,&dmmg);CHKERRQ(ierr);
  {
    DA da;
    PetscInt M = -3,P = -2;
    ierr = PetscOptionsGetInt(0,"-M",&M,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-P",&P,NULL);CHKERRQ(ierr);
    ierr = DACreate3d(comm,DA_YZPERIODIC,DA_STENCIL_BOX,P,M,M,1,PETSC_DETERMINE,PETSC_DETERMINE,sizeof(Node)/sizeof(PetscScalar),1,0,0,0,&da);CHKERRQ(ierr);
    ierr = DASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
  }
  ierr = PetscOptionsSetValue("-dmmg_form_function_ghost","1");CHKERRQ(ierr); /* Spectacularly ugly */
  ierr = DMMGSetSNESLocal(dmmg,THIFunctionLocal,THIJacobianLocal,0,0);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(DMMGGetB(dmmg),"thi_");CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
  ierr = THISetDMMG(thi,dmmg);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg,THIInitial);CHKERRQ(ierr);
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  {
    PetscInt its;
    ierr = SNESGetIterationNumber(DMMGGetSNES(dmmg),&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
  }
  {
    PetscReal nrm2,min[3],max[3];
    PetscInt i,m;
    PetscScalar *x;
    Vec X = DMMGGetx(dmmg);CHKERRQ(ierr);
    ierr = VecNorm(X,NORM_2,&nrm2);CHKERRQ(ierr);
    ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
    for (i=0; i<m; i+=2) {
      PetscReal u = PetscRealPart(x[i]),v = PetscRealPart(x[i+1]),c = sqrt(u*u+v*v);
      min[0] = PetscMin(u,min[0]);
      min[1] = PetscMin(v,min[1]);
      min[2] = PetscMin(c,min[2]);
      max[0] = PetscMax(u,max[0]);
      max[1] = PetscMax(v,max[1]);
      max[2] = PetscMax(c,max[2]);
    }
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,min,3,MPIU_REAL,MPI_MIN,thi->comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,max,3,MPIU_REAL,MPI_MAX,thi->comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"|X|_2 %g   u in [%g, %g]   v in [%g, %g]   c in [%g, %g] \n",nrm2,min[0],max[0],min[1],max[1],min[2],max[2]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Global eta range [%g, %g], converged range [%g, %g]\n",thi->etamin,thi->etamax,thi->cetamin,thi->cetamax);CHKERRQ(ierr);
  }
  ierr = THIDAVecView_VTK(thi,DMMGGetDA(dmmg),DMMGGetx(dmmg),"thi.vtk");CHKERRQ(ierr);

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = THIDestroy(thi);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
