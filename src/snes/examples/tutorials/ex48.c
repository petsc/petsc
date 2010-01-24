static const char help[] = "Toy hydrostatic ice flow with multigrid in 3D\n\
\n\
Solves the hydrostatic (aka Blatter/Pattyn/First Order) equations for ice sheet flow\n\
using multigrid.  The ice uses a power-law rheology with \"Glen\" exponent 3 (corresponds\n\
to p=4/3 in a p-Laplacian).  The focus is on ISMIP-HOM experiments which assume periodic\n\
boundary conditions in the x- and y-directions.\n\
\n\
Equations are rescaled so that the domain size and solution are O(1), details of this scaling\n\
can be controlled by the options -units_meter, -units_second, and -units_kilogram.\n\
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

In the code, the equations for (u,v) are multiplied through by 1/(rho g) so that residuals are O(1).

The discretization is Q1 finite elements, managed by a DA.  The grid is never distorted in the
map (x,y) plane, but the bed and surface may be bumpy.  This is handled as usual in FEM, through
the Jacobian of the coordinate transformation from a reference element to the physical element.

Since ice-flow is tightly coupled in the z-direction (within columns), the DA is managed
specially so that columns are never distributed, and are always contiguous in memory.
This amounts to reversing the meaning of X,Y,Z compared to the DA's internal interpretation,
and then indexing as vec[i][j][k].  The exotic coarse spaces require 2D DAs which are made to
use compatible domain decomposition relative to the 3D DAs.

There are two compile-time options:

  NO_SSE2:
    If the host supports SSE2, we use integration code that has been vectorized with SSE2
    intrinsics, unless this macro is defined.  The intrinsics speed up integration by about
    30% on my architecture (P8700, gcc-4.5 snapshot).

  COMPUTE_LOWER_TRIANGULAR:
    The element matrices we assemble are lower-triangular so it is not necessary to compute
    all entries explicitly.  If this macro is defined, the lower-triangular entries are
    computed explicitly.

*/

#include <petscdmmg.h>
#include <ctype.h>              /* toupper() */
#include <private/daimpl.h>     /* There is not yet a public interface to manipulate dm->ops */

#if !defined _ISOC99_SOURCE
#  define restrict PETSC_RESTRICT
#endif
#if defined __SSE2__
#  include <emmintrin.h>
#endif

static PetscCookie THI_COOKIE;

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
  {{0,M*H,0}  ,{0,0,0}    ,{P*H,0,0}  ,{M*H,P*H,M},{0,M*L,0}  ,{0,0,0}    ,{P*L,0,0}  ,{M*L,P*L,P}},
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

static void HexComputeGeometry(PetscInt q,PetscReal hx,PetscReal hy,const PetscReal dz[restrict],PetscReal phi[restrict],PetscReal dphi[restrict][3],PetscReal *restrict jw)
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

typedef struct _p_THI   *THI;
typedef struct _n_Units *Units;

typedef struct {
  PetscScalar u,v;
} Node;

typedef struct {
  PetscScalar b;                /* bed */
  PetscScalar h;                /* thickness */
  PetscScalar beta2;            /* friction */
} PrmNode;

typedef enum {THIQSACTION_ZERO,THIQSACTION_KEEP,THIQSACTION_UPDATE} THIQSAction;
typedef enum {THIASSEMBLY_TRIDIAGONAL,THIASSEMBLY_FULL} THIAssemblyMode;

typedef struct {
  PetscScalar eta;
  PetscScalar deta_du2;
  PetscScalar deta_dv2;
} QSNode;

struct _p_THI {
  PETSCHEADER(int);
  void (*initialize)(THI,PetscReal x,PetscReal y,PrmNode *p);
  PetscInt  nlevels;
  PetscInt  zlevels;
  PetscReal Lx,Ly,Lz;           /* Model domain */
  PetscReal alpha;              /* Bed angle */
  Units     units;
  PetscReal dirichlet_scale;
  PetscReal ssa_friction_scale;
  PetscReal etamin,etamax,cetamin,cetamax;
  struct {
    PetscReal Bd2,eps,exponent;
  } viscosity;
  PetscReal rhog;
  PetscTruth no_slip;
  PetscTruth tridiagonal;
  PetscTruth coarse2d;
  PetscTruth verbose;
  MatType mattype;
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
  p->b = s - 1000*units->meter;
  p->h = s - p->b;
  p->beta2 = 1e8 * units->Pascal * units->year / units->meter;
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
#define __FUNCT__ "THIDestroy"
static PetscErrorCode THIDestroy(THI thi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--((PetscObject)thi)->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(thi->units);CHKERRQ(ierr);
  ierr = PetscFree(thi->mattype);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(thi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THICreate"
static PetscErrorCode THICreate(MPI_Comm comm,THI *inthi)
{
  static PetscTruth registered = PETSC_FALSE;
  THI thi;
  Units units;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *inthi = 0;
  if (!registered) {
    ierr = PetscCookieRegister("Toy Hydrostatic Ice",&THI_COOKIE);CHKERRQ(ierr);
    registered = PETSC_TRUE;
  }
  ierr = PetscHeaderCreate(thi,_p_THI,0,THI_COOKIE,-1,"THI",comm,THIDestroy,0);CHKERRQ(ierr);

  ierr = PetscNew(struct _n_Units,&thi->units);CHKERRQ(ierr);
  units = thi->units;
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
  thi->verbose         = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm,NULL,"Toy Hydrostatic Ice options","");CHKERRQ(ierr);
  {
    char homexp[] = "A";
    char mtype[256] = MATSBAIJ;
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
        thi->no_slip = PETSC_FALSE;
        thi->alpha = 0.5;
        break;
      default:
        SETERRQ1(PETSC_ERR_SUP,"HOM experiment '%c' not implemented",homexp[0]);
    }
    ierr = PetscOptionsReal("-thi_alpha","Bed angle (degrees)","",thi->alpha,&thi->alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_dirichlet_scale","Scale Dirichlet boundary conditions by this factor","",thi->dirichlet_scale,&thi->dirichlet_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_ssa_friction_scale","Scale slip boundary conditions by this factor in SSA (2D) assembly","",thi->ssa_friction_scale,&thi->ssa_friction_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-thi_nlevels","Number of levels of refinement","",thi->nlevels,&thi->nlevels,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-thi_coarse2d","Use a 2D coarse space corresponding to SSA","",thi->coarse2d,&thi->coarse2d,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-thi_tridiagonal","Assemble a tridiagonal system (column coupling only) on the finest level","",thi->tridiagonal,&thi->tridiagonal,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-thi_mat_type","Matrix type","MatSetType",MatList,mtype,(char*)mtype,sizeof(mtype),NULL);CHKERRQ(ierr);
    ierr = PetscStrallocpy(mtype,&thi->mattype);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-thi_verbose","Enable verbose output (like matrix sizes and statistics)","",thi->verbose,&thi->verbose,NULL);CHKERRQ(ierr);
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
    if (thi->verbose) {
      ierr = PetscPrintf(((PetscObject)thi)->comm,"Units: meter %8.2g  second %8.2g  kg %8.2g  Pa %8.2g\n",units->meter,units->second,units->kilogram,units->Pascal);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)thi)->comm,"Domain (%6.2g,%6.2g,%6.2g), pressure %8.2g, driving stress %8.2g\n",thi->Lx,thi->Ly,thi->Lz,rho*grav*1e3*units->meter,driving);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)thi)->comm,"Large velocity 1km/a %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",u,gradu,eta,2*eta*gradu,2*eta*gradu/driving);CHKERRQ(ierr);
      THIViscosity(thi,0.5*PetscSqr(1e-3*gradu),&eta,&deta);
      ierr = PetscPrintf(((PetscObject)thi)->comm,"Small velocity 1m/a  %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",1e-3*u,1e-3*gradu,eta,2*eta*1e-3*gradu,2*eta*1e-3*gradu/driving);CHKERRQ(ierr);
    }
  }

  *inthi = thi;
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
  PetscInt i;

  PetscFunctionBegin;
  if (DMMGGetLevels(dmmg) != thi->nlevels) SETERRQ(PETSC_ERR_ARG_CORRUPT,"DMMG nlevels does not agree with THI");
  for (i=0; i<thi->nlevels; i++) {
    PetscInt Mx,My,Mz,mx,my,s,dim;
    DAStencilType  st;
    DA da = (DA)dmmg[i]->dm,da2prm;
    Vec X;
    ierr = DAGetInfo(da,&dim, &Mz,&My,&Mx, 0,&my,&mx, 0,&s,0,&st);CHKERRQ(ierr);
    if (dim == 2) {
      ierr = DAGetInfo(da,&dim, &My,&Mx,0, &my,&mx,0, 0,&s,0,&st);CHKERRQ(ierr);
    }
    ierr = DACreate2d(((PetscObject)thi)->comm,DA_XYPERIODIC,st,My,Mx,my,mx,sizeof(PrmNode)/sizeof(PetscScalar),s,0,0,&da2prm);CHKERRQ(ierr);
    ierr = DACreateLocalVector(da2prm,&X);CHKERRQ(ierr);
    {
      PetscReal Lx = thi->Lx / thi->units->meter,Ly = thi->Ly / thi->units->meter,Lz = thi->Lz / thi->units->meter;
      if (dim == 2) {
        ierr = PetscPrintf(((PetscObject)thi)->comm,"Level %d domain size (m) %8.2g x %8.2g, num elements %3dx%3d (%8d), size (m) %g x %g\n",i,Lx,Ly,Mx,My,Mx*My,Lx/Mx,Ly/My);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(((PetscObject)thi)->comm,"Level %d domain size (m) %8.2g x %8.2g x %8.2g, num elements %3dx%3dx%3d (%8d), size (m) %g x %g x %g\n",i,Lx,Ly,Lz,Mx,My,Mz,Mx*My*Mz,Lx/Mx,Ly/My,1000./(Mz-1));CHKERRQ(ierr);
      }
    }
    ierr = THIInitializePrm(thi,da2prm,X);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da,"DA2Prm",(PetscObject)da2prm);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da,"DA2Prm_Vec",(PetscObject)X);CHKERRQ(ierr);
    ierr = DADestroy(da2prm);CHKERRQ(ierr);
    ierr = VecDestroy(X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIDAGetQS"
static PetscErrorCode THIDAGetQS(DA da,THIQSAction act,QSNode ****qstore)
{
  PetscErrorCode ierr;
  DA daqs;
  Vec X;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)da,"DAQStore",(PetscObject*)&daqs);CHKERRQ(ierr);
  if (!daqs && act == THIQSACTION_KEEP) { /* returns nothing if there is no store in Jacobian assembly */
    *qstore = 0;
    PetscFunctionReturn(0);
  }
  if (!daqs) {
    PetscInt dim,M,N,P,m,n,p,s;
    DAPeriodicType wrap;
    ierr = DAGetInfo(da,&dim,&P,&N,&M,&p,&n,&m,0,&s,&wrap,0);CHKERRQ(ierr);
    if (dim != 3) SETERRQ(PETSC_ERR_ARG_WRONG,"Expected a 3D DA");
    P = 2*(P-1);                /* P was the number of nodes per column, values at two quadrature points per element */
    ierr = DACreate3d(((PetscObject)da)->comm,wrap,DA_STENCIL_BOX,P,N,M,p,n,m,sizeof(QSNode)/sizeof(PetscScalar),s,0,0,0,&daqs);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da,"DAQStore",(PetscObject)daqs);CHKERRQ(ierr);
    ierr = DACreateLocalVector(daqs,&X);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da,"DAQStore_Vec",(PetscObject)X);CHKERRQ(ierr);
    /* give away ownership */
    ierr = DADestroy(daqs);CHKERRQ(ierr);
    ierr = VecDestroy(X);CHKERRQ(ierr);
  }
  ierr = PetscObjectQuery((PetscObject)da,"DAQStore_Vec",(PetscObject*)&X);CHKERRQ(ierr);
  switch (act) {
    case THIQSACTION_ZERO:
      ierr = VecZeroEntries(X);CHKERRQ(ierr);
      break;
    case THIQSACTION_KEEP:
      break;
    case THIQSACTION_UPDATE:
      /* Lazy updating is not yet implemented */
    default: SETERRQ1(PETSC_ERR_ARG_WRONG,"Unexpected action %d",act);
  }
  ierr = DAVecGetArray(daqs,X,qstore);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIDARestoreQS"
static PetscErrorCode THIDARestoreQS(DA da,THIQSAction act,QSNode ****qstore)
{
  PetscErrorCode ierr;
  DA daqs;
  Vec X;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)da,"DAQStore",(PetscObject*)&daqs);CHKERRQ(ierr);
  if (!daqs || !qstore) PetscFunctionReturn(0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)da,"DAQStore_Vec",(PetscObject*)&X);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(daqs,X,qstore);CHKERRQ(ierr);
  switch (act) {
    case THIQSACTION_UPDATE:
      ierr = DALocalToLocalBegin(da,X,ADD_VALUES,X);CHKERRQ(ierr);
      ierr = DALocalToLocalEnd(da,X,ADD_VALUES,X);CHKERRQ(ierr);
      break;
    case THIQSACTION_KEEP:
      break;
    default: SETERRQ1(PETSC_ERR_ARG_WRONG,"Unexpected action %d",act);
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

static void PointwiseNonlinearity(THI thi,const Node n[restrict 8],const PetscReal phi[restrict 3],PetscReal dphi[restrict 8][3],PetscScalar *restrict u,PetscScalar *restrict v,PetscScalar du[restrict 3],PetscScalar dv[restrict 3],PetscReal *eta,PetscReal *deta)
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

static void PointwiseNonlinearity2D(THI thi,Node n[],PetscReal phi[],PetscReal dphi[4][2],PetscScalar *u,PetscScalar *v,PetscScalar du[],PetscScalar dv[],PetscReal *eta,PetscReal *deta)
{
  PetscInt l,ll;
  PetscScalar gam;

  du[0] = du[1] = 0;
  dv[0] = dv[1] = 0;
  *u = 0;
  *v = 0;
  for (l=0; l<4; l++) {
    *u += phi[l] * n[l].u;
    *v += phi[l] * n[l].v;
    for (ll=0; ll<2; ll++) {
      du[ll] += dphi[l][ll] * n[l].u;
      dv[ll] += dphi[l][ll] * n[l].v;
    }
  }
  gam = Sqr(du[0]) + Sqr(dv[1]) + du[0]*dv[1] + 0.25*Sqr(du[1]+dv[0]);
  THIViscosity(thi,PetscRealPart(gam),eta,deta);
}

#undef __FUNCT__  
#define __FUNCT__ "THIFunctionLocal"
static PetscErrorCode THIFunctionLocal(DALocalInfo *info,Node ***x,Node ***f,THI thi)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l;
  PetscReal      hx,hy,etamin,etamax;
  PrmNode        **prm;
  QSNode     ***qstore;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;
  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  etamin = 1e100;
  etamax = 0;

  ierr = THIDAGetPrm(info->da,&prm);CHKERRQ(ierr);
  ierr = THIDAGetQS(info->da,THIQSACTION_ZERO,&qstore);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm,i,j,pn);
      for (k=0; k<zm-1; k++) {
        PetscInt ls = 0;
        Node n[8],*fn[8];
        QSNode *qstoren[8];
        PetscReal zn[8],etabase = 0;
        PrmHexGetZ(pn,k,zm,zn);
        HexExtract(x,i,j,k,n);
        HexExtractRef(f,i,j,k,fn);
        if (qstore) {HexExtractRef(qstore,i,j,2*k,qstoren);}
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
          if (qstore) {
            qstoren[q]->eta      += 0.25*eta;
            qstoren[q]->deta_du2 += 0.25*sqrt(-deta)*du[2]; /* deta is negative */
            qstoren[q]->deta_dv2 += 0.25*sqrt(-deta)*dv[2];
          }
          jw /= thi->rhog;      /* scales residuals to be O(1) */
          if (q == 0) etabase = eta;
          if (eta > etamax)      etamax = eta;
          else if (eta < etamin) etamin = eta;
          for (l=ls; l<8; l++) { /* test functions */
            const PetscReal ds[2] = {-tan(thi->alpha),0};
            const PetscReal pp=phi[l],*dp = dphi[l];
            fn[l]->u += dp[0]*jw*eta*(4.*du[0]+2.*dv[1]) + dp[1]*jw*eta*(du[1]+dv[0]) + dp[2]*jw*eta*du[2] + pp*jw*thi->rhog*ds[0];
            fn[l]->v += dp[1]*jw*eta*(2.*du[0]+4.*dv[1]) + dp[0]*jw*eta*(du[1]+dv[0]) + dp[2]*jw*eta*dv[2] + pp*jw*thi->rhog*ds[1];
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
            const PetscScalar diagu = 2*etabase/thi->rhog*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase/thi->rhog*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            fn[0]->u = thi->dirichlet_scale*diagu*n[0].u;
            fn[0]->v = thi->dirichlet_scale*diagv*n[0].v;
          } else {              /* Integrate over bottom face to apply boundary condition */
            for (q=0; q<4; q++) {
              const PetscReal jw = 0.25*hx*hy/thi->rhog,*phi = QuadQInterp[q];
              PetscScalar u=0,v=0,beta2=0;
              for (l=0; l<4; l++) {
                u     += phi[l]*n[l].u;
                v     += phi[l]*n[l].v;
                beta2 += phi[l]*pn[l].beta2;
              }
              for (l=0; l<4; l++) {
                const PetscReal pp = phi[l];
                fn[ls+l]->u += pp*jw*beta2*u;
                fn[ls+l]->v += pp*jw*beta2*v;
              }
            }
          }
        }
      }
    }
  }

  ierr = THIDARestorePrm(info->da,&prm);CHKERRQ(ierr);
  ierr = THIDARestoreQS(info->da,THIQSACTION_UPDATE,&qstore);CHKERRQ(ierr);

  thi->cetamin = etamin;
  thi->cetamax = etamax;
  if (etamin < thi->etamin) thi->etamin = etamin;
  if (etamax > thi->etamax) thi->etamax = etamax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIMatrixStatistics"
static PetscErrorCode THIMatrixStatistics(THI thi,Mat B,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      nrm;
  PetscInt       m;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
  ierr = MatGetSize(B,&m,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)B)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscScalar val0,val2;
    ierr = MatGetValue(B,0,0,&val0);CHKERRQ(ierr);
    ierr = MatGetValue(B,2,2,&val2);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Matrix dim %8d  norm %8.2e, (0,0) %8.2e  (2,2) %8.2e, eta [%8.2e,%8.2e]\n",m,nrm,PetscRealPart(val0),PetscRealPart(val2),thi->cetamin,thi->cetamax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIJacobianLocal_2D"
static PetscErrorCode THIJacobianLocal_2D(DALocalInfo *info,Node **x,Mat B,THI thi)
{
  PetscInt       xs,ys,xm,ym,i,j,q,l,ll;
  PetscReal      hx,hy;
  PrmNode        **prm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  xs = info->ys;
  ys = info->xs;
  xm = info->ym;
  ym = info->xm;
  hx = thi->Lx / info->my;
  hy = thi->Ly / info->mx;

  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  ierr = THIDAGetPrm(info->da,&prm);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      Node n[4];
      PrmNode pn[4];
      PetscScalar Ke[4*2][4*2];
      QuadExtract(prm,i,j,pn);
      QuadExtract(x,i,j,n);
      PetscMemzero(Ke,sizeof(Ke));
      for (q=0; q<4; q++) {
        PetscReal phi[4],dphi[4][2],jw,eta,deta;
        PetscScalar u,v,du[2],dv[2],h = 0,beta2 = 0;
        for (l=0; l<4; l++) {
          phi[l] = QuadQInterp[q][l];
          dphi[l][0] = QuadQDeriv[q][l][0]*2./hx;
          dphi[l][1] = QuadQDeriv[q][l][1]*2./hy;
          h += phi[l] * pn[l].h;
          beta2 += phi[l] * pn[l].beta2;
        }
        jw = 0.25*hx*hy / thi->rhog; /* rhog is only scaling */
        PointwiseNonlinearity2D(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
        for (l=0; l<4; l++) {
          const PetscReal pp = phi[l],*dp = dphi[l];
          for (ll=0; ll<4; ll++) {
            const PetscReal ppl = phi[ll],*dpl = dphi[ll];
            PetscScalar dgdu,dgdv;
            dgdu = 2.*du[0]*dpl[0] + dv[1]*dpl[0] + 0.5*(du[1]+dv[0])*dpl[1];
            dgdv = 2.*dv[1]*dpl[1] + du[0]*dpl[1] + 0.5*(du[1]+dv[0])*dpl[0];
            /* Picard part */
            Ke[l*2+0][ll*2+0] += dp[0]*jw*eta*4.*dpl[0] + dp[1]*jw*eta*dpl[1] + pp*jw*(beta2/h)*ppl*thi->ssa_friction_scale;
            Ke[l*2+0][ll*2+1] += dp[0]*jw*eta*2.*dpl[1] + dp[1]*jw*eta*dpl[0];
            Ke[l*2+1][ll*2+0] += dp[1]*jw*eta*2.*dpl[0] + dp[0]*jw*eta*dpl[1];
            Ke[l*2+1][ll*2+1] += dp[1]*jw*eta*4.*dpl[1] + dp[0]*jw*eta*dpl[0] + pp*jw*(beta2/h)*ppl*thi->ssa_friction_scale;
            /* extra Newton terms */
            Ke[l*2+0][ll*2+0] += dp[0]*jw*deta*dgdu*(4.*du[0]+2.*dv[1]) + dp[1]*jw*deta*dgdu*(du[1]+dv[0]);
            Ke[l*2+0][ll*2+1] += dp[0]*jw*deta*dgdv*(4.*du[0]+2.*dv[1]) + dp[1]*jw*deta*dgdv*(du[1]+dv[0]);
            Ke[l*2+1][ll*2+0] += dp[1]*jw*deta*dgdu*(4.*dv[1]+2.*du[0]) + dp[0]*jw*deta*dgdu*(du[1]+dv[0]);
            Ke[l*2+1][ll*2+1] += dp[1]*jw*deta*dgdv*(4.*dv[1]+2.*du[0]) + dp[0]*jw*deta*dgdv*(du[1]+dv[0]);
          }
        }
      }
      {
        const MatStencil rc[4] = {{0,i,j},{0,i+1,j},{0,i+1,j+1},{0,i,j+1}};
        ierr = MatSetValuesBlockedStencil(B,4,rc,4,rc,&Ke[0][0],ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = THIDARestorePrm(info->da,&prm);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  if (thi->verbose) {ierr = THIMatrixStatistics(thi,B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIJacobianLocal_3D"
static PetscErrorCode THIJacobianLocal_3D(DALocalInfo *info,Node ***x,Mat B,THI thi,THIAssemblyMode amode)
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
          jw /= thi->rhog;      /* residuals are scaled by this factor */
          if (q == 0) etabase = eta;
          for (l=ls; l<8; l++) { /* test functions */
            const PetscReal *restrict dp = dphi[l];
#if defined __SSE2__ && !defined NO_SSE2
            /* gcc (up to my 4.5 snapshot) is really bad at hoisting intrinsics so we do it manually */
            __m128d
              p4 = _mm_set1_pd(4),p2 = _mm_set1_pd(2),p05 = _mm_set1_pd(0.5),
              p42 = _mm_setr_pd(4,2),p24 = _mm_shuffle_pd(p42,p42,0b01),
              du0 = _mm_set1_pd(du[0]),du1 = _mm_set1_pd(du[1]),du2 = _mm_set1_pd(du[2]),
              dv0 = _mm_set1_pd(dv[0]),dv1 = _mm_set1_pd(dv[1]),dv2 = _mm_set1_pd(dv[2]),
              jweta = _mm_set1_pd(jw*eta),jwdeta = _mm_set1_pd(jw*deta),
              dp0 = _mm_set1_pd(dp[0]),dp1 = _mm_set1_pd(dp[1]),dp2 = _mm_set1_pd(dp[2]),
              dp0jweta = _mm_mul_pd(dp0,jweta),dp1jweta = _mm_mul_pd(dp1,jweta),dp2jweta = _mm_mul_pd(dp2,jweta),
              p4du0p2dv1 = _mm_add_pd(_mm_mul_pd(p4,du0),_mm_mul_pd(p2,dv1)), /* 4 du0 + 2 dv1 */
              p4dv1p2du0 = _mm_add_pd(_mm_mul_pd(p4,dv1),_mm_mul_pd(p2,du0)), /* 4 dv1 + 2 du0 */
              pdu2dv2 = _mm_shuffle_pd(du2,dv2,0b00),                         /* [du2, dv2] */
              du1pdv0 = _mm_add_pd(du1,dv0),                                  /* du1 + dv0 */
              t1 = _mm_mul_pd(dp0,p4du0p2dv1),                                /* dp0 (4 du0 + 2 dv1) */
              t2 = _mm_mul_pd(dp1,p4dv1p2du0);                                /* dp1 (4 dv1 + 2 du0) */

#endif
#if defined COMPUTE_LOWER_TRIANGULAR  /* The element matrices are always symmetric so computing the lower-triangular part is not necessary */
            for (ll=ls; ll<8; ll++) { /* trial functions */
#else
            for (ll=l; ll<8; ll++) {
#endif
              const PetscReal *restrict dpl = dphi[ll];
              if (amode == THIASSEMBLY_TRIDIAGONAL && (l-ll)%4) continue; /* these entries would not be inserted */
#if !(defined __SSE2__ && !defined NO_SSE2)
              /* The analytic Jacobian in nice, easy-to-read form */
              {
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
#else
              /* This SSE2 code is an exact replica of above, but uses explicit packed instructions for some speed
              * benefit.  On my hardware, these intrinsics are 25% faster than above. */
              {
                __m128d
                  keu = _mm_load_pd(&Ke[l*2+0][ll*2+0]),
                  kev = _mm_load_pd(&Ke[l*2+1][ll*2+0]),
                  dpl01 = _mm_loadu_pd(&dpl[0]),dpl10 = _mm_shuffle_pd(dpl01,dpl01,0b01),dpl2 = _mm_set_sd(dpl[2]),
                  t0,t3,pdgduv;
                keu = _mm_add_pd(keu,_mm_add_pd(_mm_mul_pd(_mm_mul_pd(dp0jweta,p42),dpl01),
                                                _mm_add_pd(_mm_mul_pd(dp1jweta,dpl10),
                                                           _mm_mul_pd(dp2jweta,dpl2))));
                kev = _mm_add_pd(kev,_mm_add_pd(_mm_mul_pd(_mm_mul_pd(dp1jweta,p24),dpl01),
                                                _mm_add_pd(_mm_mul_pd(dp0jweta,dpl10),
                                                           _mm_mul_pd(dp2jweta,_mm_shuffle_pd(dpl2,dpl2,0b01)))));
                pdgduv = _mm_mul_pd(p05,_mm_add_pd(_mm_add_pd(_mm_mul_pd(p42,_mm_mul_pd(du0,dpl01)),
                                                              _mm_mul_pd(p24,_mm_mul_pd(dv1,dpl01))),
                                                   _mm_add_pd(_mm_mul_pd(du1pdv0,dpl10),
                                                              _mm_mul_pd(pdu2dv2,_mm_set1_pd(dpl[2]))))); /* [dgdu, dgdv] */
                t0 = _mm_mul_pd(jwdeta,pdgduv);  /* jw deta [dgdu, dgdv] */
                t3 = _mm_mul_pd(t0,du1pdv0);     /* t0 (du1 + dv0) */
                _mm_store_pd(&Ke[l*2+0][ll*2+0],_mm_add_pd(keu,_mm_add_pd(_mm_mul_pd(t1,t0),
                                                                          _mm_add_pd(_mm_mul_pd(dp1,t3),
                                                                                     _mm_mul_pd(t0,_mm_mul_pd(dp2,du2))))));
                _mm_store_pd(&Ke[l*2+1][ll*2+0],_mm_add_pd(kev,_mm_add_pd(_mm_mul_pd(t2,t0),
                                                                          _mm_add_pd(_mm_mul_pd(dp0,t3),
                                                                                     _mm_mul_pd(t0,_mm_mul_pd(dp2,dv2))))));
              }
#endif
            }
          }
        }
        if (k == 0) { /* on a bottom face */
          if (thi->no_slip) {
            const PetscReal hz = PetscRealPart(pn[0].h)/(zm-1);
            const PetscScalar diagu = 2*etabase/thi->rhog*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase/thi->rhog*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            Ke[0][0] = thi->dirichlet_scale*diagu;
            Ke[1][1] = thi->dirichlet_scale*diagv;
          } else {
            for (q=0; q<4; q++) {
              const PetscReal jw = 0.25*hx*hy/thi->rhog,*phi = QuadQInterp[q];
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
                  Ke[l*2+0][ll*2+0] += pp*jw*beta2*ppl;
                  Ke[l*2+1][ll*2+1] += pp*jw*beta2*ppl;
                }
              }
            }
          }
        }
        {
          const MatStencil rc[8] = {{i,j,k},{i+1,j,k},{i+1,j+1,k},{i,j+1,k},{i,j,k+1},{i+1,j,k+1},{i+1,j+1,k+1},{i,j+1,k+1}};
          if (amode == THIASSEMBLY_TRIDIAGONAL) {
            for (l=0; l<4; l++) { /* Copy out each of the blocks, discarding horizontal coupling */
              const PetscInt l4 = l+4;
              const MatStencil rcl[2] = {{rc[l].k,rc[l].j,rc[l].i},{rc[l4].k,rc[l4].j,rc[l4].i}};
#if defined COMPUTE_LOWER_TRIANGULAR
              const PetscScalar Kel[4][4] = {{Ke[2*l+0][2*l+0] ,Ke[2*l+0][2*l+1] ,Ke[2*l+0][2*l4+0] ,Ke[2*l+0][2*l4+1]},
                                             {Ke[2*l+1][2*l+0] ,Ke[2*l+1][2*l+1] ,Ke[2*l+1][2*l4+0] ,Ke[2*l+1][2*l4+1]},
                                             {Ke[2*l4+0][2*l+0],Ke[2*l4+0][2*l+1],Ke[2*l4+0][2*l4+0],Ke[2*l4+0][2*l4+1]},
                                             {Ke[2*l4+1][2*l+0],Ke[2*l4+1][2*l+1],Ke[2*l4+1][2*l4+0],Ke[2*l4+1][2*l4+1]}};
#else
              /* Same as above except for the lower-left block */
              const PetscScalar Kel[4][4] = {{Ke[2*l+0][2*l+0] ,Ke[2*l+0][2*l+1] ,Ke[2*l+0][2*l4+0] ,Ke[2*l+0][2*l4+1]},
                                             {Ke[2*l+1][2*l+0] ,Ke[2*l+1][2*l+1] ,Ke[2*l+1][2*l4+0] ,Ke[2*l+1][2*l4+1]},
                                             {Ke[2*l+0][2*l4+0],Ke[2*l+1][2*l4+0],Ke[2*l4+0][2*l4+0],Ke[2*l4+0][2*l4+1]},
                                             {Ke[2*l+0][2*l4+1],Ke[2*l+1][2*l4+1],Ke[2*l4+1][2*l4+0],Ke[2*l4+1][2*l4+1]}};
#endif
              ierr = MatSetValuesBlockedStencil(B,2,rcl,2,rcl,&Kel[0][0],ADD_VALUES);CHKERRQ(ierr);
            }
          } else {
#if !defined COMPUTE_LOWER_TRIANGULAR /* fill in lower-triangular part, this is really cheap compared to computing the entries */
            for (l=0; l<8; l++) {
              for (ll=l+1; ll<8; ll++) {
                Ke[ll*2+0][l*2+0] = Ke[l*2+0][ll*2+0];
                Ke[ll*2+1][l*2+0] = Ke[l*2+0][ll*2+1];
                Ke[ll*2+0][l*2+1] = Ke[l*2+1][ll*2+0];
                Ke[ll*2+1][l*2+1] = Ke[l*2+1][ll*2+1];
              }
            }
#endif
            ierr = MatSetValuesBlockedStencil(B,8,rc,8,rc,&Ke[0][0],ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = THIDARestorePrm(info->da,&prm);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  if (thi->verbose) {ierr = THIMatrixStatistics(thi,B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIJacobianLocal_3D_Full"
static PetscErrorCode THIJacobianLocal_3D_Full(DALocalInfo *info,Node ***x,Mat B,THI thi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = THIJacobianLocal_3D(info,x,B,thi,THIASSEMBLY_FULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIJacobianLocal_3D_Tridiagonal"
static PetscErrorCode THIJacobianLocal_3D_Tridiagonal(DALocalInfo *info,Node ***x,Mat B,THI thi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = THIJacobianLocal_3D(info,x,B,thi,THIASSEMBLY_TRIDIAGONAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DARefineHierarchy_THI"
static PetscErrorCode DARefineHierarchy_THI(DA dac0,PetscInt nlevels,DA hierarchy[])
{
  PetscErrorCode ierr;
  THI thi;
  PetscInt dim,M,N,m,n,s,dof;
  DA dac,daf;
  DAStencilType  st;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dac0,"THI",(PetscObject*)&thi);CHKERRQ(ierr);
  if (!thi) SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot refine this DA, missing composed THI instance");
  if (nlevels > 1) {
    ierr = DARefineHierarchy(dac0,nlevels-1,hierarchy);CHKERRQ(ierr);
    dac = hierarchy[nlevels-2];
  } else {
    dac = dac0;
  }
  ierr = DAGetInfo(dac,&dim, &N,&M,0, &n,&m,0, &dof,&s,0,&st);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(PETSC_ERR_ARG_WRONG,"This function can only refine 2D DAs");
  /* Creates a 3D DA with the same map-plane layout as the 2D one, with contiguous columns */
  ierr = DACreate3d(((PetscObject)dac)->comm,DA_YZPERIODIC,st,thi->zlevels,N,M,1,n,m,dof,s,NULL,NULL,NULL,&daf);CHKERRQ(ierr);
  daf->ops->getmatrix        = dac->ops->getmatrix;
  daf->ops->getinterpolation = dac->ops->getinterpolation;
  daf->ops->getcoloring      = dac->ops->getcoloring;
  daf->interptype            = dac->interptype;

  ierr = DASetFieldName(daf,0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(daf,1,"y-velocity");CHKERRQ(ierr);
  hierarchy[nlevels-1] = daf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_THI"
static PetscErrorCode DAGetInterpolation_THI(DA dac,DA daf,Mat *A,Vec *scale)
{
  PetscErrorCode ierr;
  PetscTruth flg,isda2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_COOKIE,1);
  PetscValidHeaderSpecific(daf,DM_COOKIE,2);
  PetscValidPointer(A,3);
  if (scale) PetscValidPointer(scale,4);
  ierr = PetscTypeCompare((PetscObject)dac,DA2D,&flg);
  if (!flg) SETERRQ(PETSC_ERR_ARG_WRONG,"Expected coarse DA to be 2D");
  ierr = PetscTypeCompare((PetscObject)daf,DA2D,&isda2);CHKERRQ(ierr);
  if (isda2) {
    /* We are in the 2D problem and use normal DA interpolation */
    ierr = DAGetInterpolation(dac,daf,A,scale);CHKERRQ(ierr);
  } else {
    PetscInt i,j,k,xs,ys,zs,xm,ym,zm,mx,my,mz,rstart,cstart;
    Mat B;

    ierr = DAGetInfo(daf,0, &mz,&my,&mx, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
    ierr = DAGetCorners(daf,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
    if (zs != 0) SETERRQ(1,"unexpected");
    ierr = MatCreate(((PetscObject)daf)->comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,xm*ym*zm,xm*ym,mx*my*mz,mx*my);CHKERRQ(ierr);
    
    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(B,1,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(B,1,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(B,&rstart,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(B,&cstart,NULL);CHKERRQ(ierr);
    for (i=xs; i<xs+xm; i++) {
      for (j=ys; j<ys+ym; j++) {
        for (k=zs; k<zs+zm; k++) {
          PetscInt i2 = i*ym+j,i3 = i2*zm+k;
          PetscScalar val = ((k == 0 || k == mz-1) ? 0.5 : 1.) / (mz-1.); /* Integration using trapezoid rule */
          ierr = MatSetValue(B,cstart+i3,rstart+i2,val,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreateMAIJ(B,sizeof(Node)/sizeof(PetscScalar),A);CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix_THI_Tridiagonal"
static PetscErrorCode DAGetMatrix_THI_Tridiagonal(DA da,const MatType mtype,Mat *J)
{
  PetscErrorCode ierr;
  Mat A;
  PetscInt xm,ym,zm,dim,dof = 2,starts[3],dims[3];
  ISLocalToGlobalMapping ltog,ltogb;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,&dim, 0,0,0, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
  if (dim != 3) SETERRQ(PETSC_ERR_ARG_WRONG,"Expected DA to be 3D");
  ierr = DAGetCorners(da,0,0,0,&zm,&ym,&xm);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMappingBlck(da,&ltogb);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)da)->comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,dof*xm*ym*zm,dof*xm*ym*zm,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,6,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,6,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,dof,3,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,dof,3,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,dof,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,dof,2,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,dof);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(A,ltogb);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);CHKERRQ(ierr);
  ierr = MatSetStencil(A,dim,dims,starts,dof);CHKERRQ(ierr);
  *J = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "THIDAVecView_VTK"
static PetscErrorCode THIDAVecView_VTK(THI thi,DA da,Vec X,const char filename[])
{
  Units units = thi->units;
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscInt mx,my,mz,i,j,k;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0, &mz,&my,&mx, 0,0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(((PetscObject)thi)->comm,filename,&viewer);CHKERRQ(ierr);
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

    comm = ((PetscObject)thi)->comm;
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
        ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",PetscRealPart(array[i])*units->year/units->meter,PetscRealPart(array[i+1])*units->year/units->meter,0);CHKERRQ(ierr);
      }
      for(p=1; p<size; p++) {
        MPI_Status status;
        ierr = MPI_Recv(values,(PetscMPIInt)n,MPIU_SCALAR,p,tag,comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&nn);CHKERRQ(ierr);
        for(i=0; i<nn; i+=2) {
          ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",PetscRealPart(values[i])*units->year/units->meter,PetscRealPart(values[i+1])*units->year/units->meter,0);CHKERRQ(ierr);
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
  MPI_Comm       comm;
  DMMG           *dmmg;
  THI            thi;
  PetscInt       i;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = THICreate(comm,&thi);CHKERRQ(ierr);
  ierr = DMMGCreate(PETSC_COMM_WORLD,thi->nlevels,thi,&dmmg);CHKERRQ(ierr);
  {
    DA da;
    PetscInt M = 3,N = 3,P = 2;
    ierr = PetscOptionsBegin(comm,NULL,"Grid resolution options","");CHKERRQ(ierr);
    {
      ierr = PetscOptionsInt("-M","Number of elements in x-direction on coarse level","",M,&M,NULL);CHKERRQ(ierr);
      N = M;
      ierr = PetscOptionsInt("-N","Number of elements in y-direction on coarse level (if different from M)","",N,&N,NULL);CHKERRQ(ierr);
      if (thi->coarse2d) {
        ierr = PetscOptionsInt("-zlevels","Number of elements in z-direction on fine level","",thi->zlevels,&thi->zlevels,NULL);CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsInt("-P","Number of elements in z-direction on coarse level","",P,&P,NULL);CHKERRQ(ierr);
      }
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (thi->coarse2d) {
      ierr = DACreate2d(comm,DA_XYPERIODIC,DA_STENCIL_BOX,N,M,PETSC_DETERMINE,PETSC_DETERMINE,sizeof(Node)/sizeof(PetscScalar),1,0,0,&da);CHKERRQ(ierr);
      da->ops->refinehierarchy  = DARefineHierarchy_THI;
      da->ops->getinterpolation = DAGetInterpolation_THI;
      ierr = PetscObjectCompose((PetscObject)da,"THI",(PetscObject)thi);CHKERRQ(ierr);
    } else {
      ierr = DACreate3d(comm,DA_YZPERIODIC,DA_STENCIL_BOX,P,N,M,1,PETSC_DETERMINE,PETSC_DETERMINE,sizeof(Node)/sizeof(PetscScalar),1,0,0,0,&da);CHKERRQ(ierr);
    }
    ierr = DASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
  }
  if (thi->tridiagonal) {
    (DMMGGetDA(dmmg))->ops->getmatrix = DAGetMatrix_THI_Tridiagonal;
  }
  ierr = PetscOptionsSetValue("-dmmg_form_function_ghost","1");CHKERRQ(ierr); /* Spectacularly ugly */
  ierr = DMMGSetMatType(dmmg,thi->mattype);CHKERRQ(ierr);
  ierr = DMMGSetSNESLocal(dmmg,THIFunctionLocal,THIJacobianLocal_3D_Full,0,0);CHKERRQ(ierr);
  if (thi->tridiagonal) {
    ierr = DASetLocalJacobian(DMMGGetDA(dmmg),(DALocalFunction1)THIJacobianLocal_3D_Tridiagonal);CHKERRQ(ierr);
  }
  if (thi->coarse2d) {
    for (i=0; i<DMMGGetLevels(dmmg)-1; i++) {
      ierr = DASetLocalJacobian((DA)dmmg[i]->dm,(DALocalFunction1)THIJacobianLocal_2D);CHKERRQ(ierr);
    }
  }
  for (i=0; i<DMMGGetLevels(dmmg); i++) {
    /* This option is only valid for the SBAIJ format.  The matrices we assemble are symmetric, but the SBAIJ assembly
    * functions will complain if we provide lower-triangular entries without setting this option. */
    Mat B = dmmg[i]->B;
    PetscTruth flg1,flg2;
    ierr = PetscTypeCompare((PetscObject)B,MATSEQSBAIJ,&flg1);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)B,MATMPISBAIJ,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) {
      ierr = MatSetOption(B,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
  ierr = THISetDMMG(thi,dmmg);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg,THIInitial);CHKERRQ(ierr);
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  {
    PetscInt its,lits;
    SNESConvergedReason reason;
    ierr = SNESGetIterationNumber(DMMGGetSNES(dmmg),&its);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(DMMGGetSNES(dmmg),&reason);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(DMMGGetSNES(dmmg),&lits);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"%s: Number of Newton iterations = %d, total linear iterations = %d\n",SNESConvergedReasons[reason],its,lits);CHKERRQ(ierr);
  }
  {
    PetscReal nrm2,min[3]={1e100,1e100,1e100},max[3]={-1e100,-1e100,-1e100};
    PetscInt i,j,m;
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
    ierr = MPI_Allreduce(MPI_IN_PLACE,min,3,MPIU_REAL,MPI_MIN,((PetscObject)thi)->comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,max,3,MPIU_REAL,MPI_MAX,((PetscObject)thi)->comm);CHKERRQ(ierr);
    /* Dimensionalize to meters/year */
    nrm2 *= thi->units->year / thi->units->meter;
    for (j=0; j<3; j++) {
      min[j] *= thi->units->year / thi->units->meter;
      max[j] *= thi->units->year / thi->units->meter;
    }
    ierr = PetscPrintf(comm,"|X|_2 %g   u in [%g, %g]   v in [%g, %g]   c in [%g, %g] \n",nrm2,min[0],max[0],min[1],max[1],min[2],max[2]);CHKERRQ(ierr);
    /* These values stay nondimensional */
    ierr = PetscPrintf(comm,"Global eta range [%g, %g], converged range [%g, %g]\n",thi->etamin,thi->etamax,thi->cetamin,thi->cetamax);CHKERRQ(ierr);
  }
  {
    PetscTruth flg;
    char filename[PETSC_MAX_PATH_LEN] = "";
    ierr = PetscOptionsGetString(PETSC_NULL,"-o",filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = THIDAVecView_VTK(thi,DMMGGetDA(dmmg),DMMGGetx(dmmg),filename);CHKERRQ(ierr);
    }
  }

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = THIDestroy(thi);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
