static const char help[] = "Toy hydrostatic ice flow with multigrid in 3D.\n\
\n\
Solves the hydrostatic (aka Blatter/Pattyn/First Order) equations for ice sheet flow\n\
using multigrid.  The ice uses a power-law rheology with \"Glen\" exponent 3 (corresponds\n\
to p=4/3 in a p-Laplacian).  The focus is on ISMIP-HOM experiments which assume periodic\n\
boundary conditions in the x- and y-directions.\n\
\n\
Equations are rescaled so that the domain size and solution are O(1), details of this scaling\n\
can be controlled by the options -units_meter, -units_second, and -units_kilogram.\n\
\n\
A VTK StructuredGrid output file can be written using the option -o filename.vts\n\
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

The discretization is Q1 finite elements, managed by a DMDA.  The grid is never distorted in the
map (x,y) plane, but the bed and surface may be bumpy.  This is handled as usual in FEM, through
the Jacobian of the coordinate transformation from a reference element to the physical element.

Since ice-flow is tightly coupled in the z-direction (within columns), the DMDA is managed
specially so that columns are never distributed, and are always contiguous in memory.
This amounts to reversing the meaning of X,Y,Z compared to the DMDA's internal interpretation,
and then indexing as vec[i][j][k].  The exotic coarse spaces require 2D DMDAs which are made to
use compatible domain decomposition relative to the 3D DMDAs.

*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <ctype.h>              /* toupper() */
#include <petsc/private/petscimpl.h>

#if defined __SSE2__
#  include <emmintrin.h>
#endif

/* The SSE2 kernels are only for PetscScalar=double on architectures that support it */
#define USE_SSE2_KERNELS (!defined NO_SSE2                              \
                          && !defined PETSC_USE_COMPLEX                 \
                          && !defined PETSC_USE_REAL_SINGLE           \
                          && defined __SSE2__)

#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
#  if defined __cplusplus       /* C++ restrict is nonstandard and compilers have inconsistent rules about where it can be used */
#    define restrict
#  else
#    define restrict PETSC_RESTRICT
#  endif
#endif

static PetscClassId THI_CLASSID;

typedef enum {QUAD_GAUSS,QUAD_LOBATTO} QuadratureType;
static const char *QuadratureTypes[] = {"gauss","lobatto","QuadratureType","QUAD_",0};
static const PetscReal HexQWeights[8] = {1,1,1,1,1,1,1,1};
static const PetscReal HexQNodes[]    = {-0.57735026918962573, 0.57735026918962573};
#define G 0.57735026918962573
#define H (0.5*(1.+G))
#define L (0.5*(1.-G))
#define M (-0.5)
#define P (0.5)
/* Special quadrature: Lobatto in horizontal, Gauss in vertical */
static const PetscReal HexQInterp_Lobatto[8][8] = {{H,0,0,0,L,0,0,0},
                                                   {0,H,0,0,0,L,0,0},
                                                   {0,0,H,0,0,0,L,0},
                                                   {0,0,0,H,0,0,0,L},
                                                   {L,0,0,0,H,0,0,0},
                                                   {0,L,0,0,0,H,0,0},
                                                   {0,0,L,0,0,0,H,0},
                                                   {0,0,0,L,0,0,0,H}};
static const PetscReal HexQDeriv_Lobatto[8][8][3] = {
  {{M*H,M*H,M},{P*H,0,0}  ,{0,0,0}    ,{0,P*H,0}  ,{M*L,M*L,P},{P*L,0,0}  ,{0,0,0}    ,{0,P*L,0}  },
  {{M*H,0,0}  ,{P*H,M*H,M},{0,P*H,0}  ,{0,0,0}    ,{M*L,0,0}  ,{P*L,M*L,P},{0,P*L,0}  ,{0,0,0}    },
  {{0,0,0}    ,{0,M*H,0}  ,{P*H,P*H,M},{M*H,0,0}  ,{0,0,0}    ,{0,M*L,0}  ,{P*L,P*L,P},{M*L,0,0}  },
  {{0,M*H,0}  ,{0,0,0}    ,{P*H,0,0}  ,{M*H,P*H,M},{0,M*L,0}  ,{0,0,0}    ,{P*L,0,0}  ,{M*L,P*L,P}},
  {{M*L,M*L,M},{P*L,0,0}  ,{0,0,0}    ,{0,P*L,0}  ,{M*H,M*H,P},{P*H,0,0}  ,{0,0,0}    ,{0,P*H,0}  },
  {{M*L,0,0}  ,{P*L,M*L,M},{0,P*L,0}  ,{0,0,0}    ,{M*H,0,0}  ,{P*H,M*H,P},{0,P*H,0}  ,{0,0,0}    },
  {{0,0,0}    ,{0,M*L,0}  ,{P*L,P*L,M},{M*L,0,0}  ,{0,0,0}    ,{0,M*H,0}  ,{P*H,P*H,P},{M*H,0,0}  },
  {{0,M*L,0}  ,{0,0,0}    ,{P*L,0,0}  ,{M*L,P*L,M},{0,M*H,0}  ,{0,0,0}    ,{P*H,0,0}  ,{M*H,P*H,P}}};
/* Stanndard Gauss */
static const PetscReal HexQInterp_Gauss[8][8] = {{H*H*H,L*H*H,L*L*H,H*L*H, H*H*L,L*H*L,L*L*L,H*L*L},
                                                 {L*H*H,H*H*H,H*L*H,L*L*H, L*H*L,H*H*L,H*L*L,L*L*L},
                                                 {L*L*H,H*L*H,H*H*H,L*H*H, L*L*L,H*L*L,H*H*L,L*H*L},
                                                 {H*L*H,L*L*H,L*H*H,H*H*H, H*L*L,L*L*L,L*H*L,H*H*L},
                                                 {H*H*L,L*H*L,L*L*L,H*L*L, H*H*H,L*H*H,L*L*H,H*L*H},
                                                 {L*H*L,H*H*L,H*L*L,L*L*L, L*H*H,H*H*H,H*L*H,L*L*H},
                                                 {L*L*L,H*L*L,H*H*L,L*H*L, L*L*H,H*L*H,H*H*H,L*H*H},
                                                 {H*L*L,L*L*L,L*H*L,H*H*L, H*L*H,L*L*H,L*H*H,H*H*H}};
static const PetscReal HexQDeriv_Gauss[8][8][3] = {
  {{M*H*H,H*M*H,H*H*M},{P*H*H,L*M*H,L*H*M},{P*L*H,L*P*H,L*L*M},{M*L*H,H*P*H,H*L*M}, {M*H*L,H*M*L,H*H*P},{P*H*L,L*M*L,L*H*P},{P*L*L,L*P*L,L*L*P},{M*L*L,H*P*L,H*L*P}},
  {{M*H*H,L*M*H,L*H*M},{P*H*H,H*M*H,H*H*M},{P*L*H,H*P*H,H*L*M},{M*L*H,L*P*H,L*L*M}, {M*H*L,L*M*L,L*H*P},{P*H*L,H*M*L,H*H*P},{P*L*L,H*P*L,H*L*P},{M*L*L,L*P*L,L*L*P}},
  {{M*L*H,L*M*H,L*L*M},{P*L*H,H*M*H,H*L*M},{P*H*H,H*P*H,H*H*M},{M*H*H,L*P*H,L*H*M}, {M*L*L,L*M*L,L*L*P},{P*L*L,H*M*L,H*L*P},{P*H*L,H*P*L,H*H*P},{M*H*L,L*P*L,L*H*P}},
  {{M*L*H,H*M*H,H*L*M},{P*L*H,L*M*H,L*L*M},{P*H*H,L*P*H,L*H*M},{M*H*H,H*P*H,H*H*M}, {M*L*L,H*M*L,H*L*P},{P*L*L,L*M*L,L*L*P},{P*H*L,L*P*L,L*H*P},{M*H*L,H*P*L,H*H*P}},
  {{M*H*L,H*M*L,H*H*M},{P*H*L,L*M*L,L*H*M},{P*L*L,L*P*L,L*L*M},{M*L*L,H*P*L,H*L*M}, {M*H*H,H*M*H,H*H*P},{P*H*H,L*M*H,L*H*P},{P*L*H,L*P*H,L*L*P},{M*L*H,H*P*H,H*L*P}},
  {{M*H*L,L*M*L,L*H*M},{P*H*L,H*M*L,H*H*M},{P*L*L,H*P*L,H*L*M},{M*L*L,L*P*L,L*L*M}, {M*H*H,L*M*H,L*H*P},{P*H*H,H*M*H,H*H*P},{P*L*H,H*P*H,H*L*P},{M*L*H,L*P*H,L*L*P}},
  {{M*L*L,L*M*L,L*L*M},{P*L*L,H*M*L,H*L*M},{P*H*L,H*P*L,H*H*M},{M*H*L,L*P*L,L*H*M}, {M*L*H,L*M*H,L*L*P},{P*L*H,H*M*H,H*L*P},{P*H*H,H*P*H,H*H*P},{M*H*H,L*P*H,L*H*P}},
  {{M*L*L,H*M*L,H*L*M},{P*L*L,L*M*L,L*L*M},{P*H*L,L*P*L,L*H*M},{M*H*L,H*P*L,H*H*M}, {M*L*H,H*M*H,H*L*P},{P*L*H,L*M*H,L*L*P},{P*H*H,L*P*H,L*H*P},{M*H*H,H*P*H,H*H*P}}};
static const PetscReal (*HexQInterp)[8],(*HexQDeriv)[8][3];
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

#define FieldSize(ntype) ((PetscInt)(sizeof(ntype)/sizeof(PetscScalar)))
#define FieldOffset(ntype,member) ((PetscInt)(offsetof(ntype,member)/sizeof(PetscScalar)))
#define FieldIndex(ntype,i,member) ((PetscInt)((i)*FieldSize(ntype) + FieldOffset(ntype,member)))
#define NODE_SIZE FieldSize(Node)
#define PRMNODE_SIZE FieldSize(PrmNode)

typedef struct {
  PetscReal min,max,cmin,cmax;
} PRange;

struct _p_THI {
  PETSCHEADER(int);
  void      (*initialize)(THI,PetscReal x,PetscReal y,PrmNode *p);
  PetscInt  nlevels;
  PetscInt  zlevels;
  PetscReal Lx,Ly,Lz;           /* Model domain */
  PetscReal alpha;              /* Bed angle */
  Units     units;
  PetscReal dirichlet_scale;
  PetscReal ssa_friction_scale;
  PetscReal inertia;
  PRange    eta;
  PRange    beta2;
  struct {
    PetscReal Bd2,eps,exponent,glen_n;
  } viscosity;
  struct {
    PetscReal irefgam,eps2,exponent;
  } friction;
  struct {
    PetscReal rate,exponent,refvel;
  } erosion;
  PetscReal rhog;
  PetscBool no_slip;
  PetscBool verbose;
  char      *mattype;
  char      *monitor_basename;
  PetscInt  monitor_interval;
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

/* Compute a gradient of all the 2D fields at four quadrature points.  Output for [quadrature_point][direction].field_name */
static PetscErrorCode QuadComputeGrad4(const PetscReal dphi[][4][2],PetscReal hx,PetscReal hy,const PrmNode pn[4],PrmNode dp[4][2])
{
  PetscErrorCode ierr;
  PetscInt       q,i,f;
  const PetscScalar (*restrict pg)[PRMNODE_SIZE] = (const PetscScalar(*)[PRMNODE_SIZE])pn; /* Get generic array pointers to the node */
  PetscScalar (*restrict dpg)[2][PRMNODE_SIZE]   = (PetscScalar(*)[2][PRMNODE_SIZE])dp;

  PetscFunctionBeginUser;
  ierr = PetscArrayzero(dpg,4);CHKERRQ(ierr);
  for (q=0; q<4; q++) {
    for (i=0; i<4; i++) {
      for (f=0; f<PRMNODE_SIZE; f++) {
        dpg[q][0][f] += dphi[q][i][0]/hx * pg[i][f];
        dpg[q][1][f] += dphi[q][i][1]/hy * pg[i][f];
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscReal StaggeredMidpoint2D(PetscScalar a,PetscScalar b,PetscScalar c,PetscScalar d)
{return 0.5*PetscRealPart(0.75*a + 0.75*b + 0.25*c + 0.25*d);}
static inline PetscReal UpwindFlux1D(PetscReal u,PetscReal hL,PetscReal hR)
{return (u > 0) ? hL*u : hR*u;}

#define UpwindFluxXW(x3,x2,h,i,j,k,dj) UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].u,x3[i-1][j][k].u, x3[i-1][j+dj][k].u,x3[i][k+dj][k].u), \
                                                    PetscRealPart(0.75*x2[i-1][j  ].h+0.25*x2[i-1][j+dj].h), PetscRealPart(0.75*x2[i  ][j  ].h+0.25*x2[i  ][j+dj].h))
#define UpwindFluxXE(x3,x2,h,i,j,k,dj) UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].u,x3[i+1][j][k].u, x3[i+1][j+dj][k].u,x3[i][k+dj][k].u), \
                                                    PetscRealPart(0.75*x2[i  ][j  ].h+0.25*x2[i  ][j+dj].h), PetscRealPart(0.75*x2[i+1][j  ].h+0.25*x2[i+1][j+dj].h))
#define UpwindFluxYS(x3,x2,h,i,j,k,di) UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].v,x3[i][j-1][k].v, x3[i+di][j-1][k].v,x3[i+di][j][k].v), \
                                                    PetscRealPart(0.75*x2[i  ][j-1].h+0.25*x2[i+di][j-1].h), PetscRealPart(0.75*x2[i  ][j  ].h+0.25*x2[i+di][j  ].h))
#define UpwindFluxYN(x3,x2,h,i,j,k,di) UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].v,x3[i][j+1][k].v, x3[i+di][j+1][k].v,x3[i+di][j][k].v), \
                                                    PetscRealPart(0.75*x2[i  ][j  ].h+0.25*x2[i+di][j  ].h), PetscRealPart(0.75*x2[i  ][j+1].h+0.25*x2[i+di][j+1].h))

static void PrmNodeGetFaceMeasure(const PrmNode **p,PetscInt i,PetscInt j,PetscScalar h[])
{
  /* West */
  h[0] = StaggeredMidpoint2D(p[i][j].h,p[i-1][j].h,p[i-1][j-1].h,p[i][j-1].h);
  h[1] = StaggeredMidpoint2D(p[i][j].h,p[i-1][j].h,p[i-1][j+1].h,p[i][j+1].h);
  /* East */
  h[2] = StaggeredMidpoint2D(p[i][j].h,p[i+1][j].h,p[i+1][j+1].h,p[i][j+1].h);
  h[3] = StaggeredMidpoint2D(p[i][j].h,p[i+1][j].h,p[i+1][j-1].h,p[i][j-1].h);
  /* South */
  h[4] = StaggeredMidpoint2D(p[i][j].h,p[i][j-1].h,p[i+1][j-1].h,p[i+1][j].h);
  h[5] = StaggeredMidpoint2D(p[i][j].h,p[i][j-1].h,p[i-1][j-1].h,p[i-1][j].h);
  /* North */
  h[6] = StaggeredMidpoint2D(p[i][j].h,p[i][j+1].h,p[i-1][j+1].h,p[i-1][j].h);
  h[7] = StaggeredMidpoint2D(p[i][j].h,p[i][j+1].h,p[i+1][j+1].h,p[i+1][j].h);
}

/* Tests A and C are from the ISMIP-HOM paper (Pattyn et al. 2008) */
static void THIInitialize_HOM_A(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units units = thi->units;
  PetscReal s = -x*PetscSinReal(thi->alpha);
  p->b = s - 1000*units->meter + 500*units->meter * PetscSinReal(x*2*PETSC_PI/thi->Lx) * PetscSinReal(y*2*PETSC_PI/thi->Ly);
  p->h = s - p->b;
  p->beta2 = -1e-10;             /* This value is not used, but it should not be huge because that would change the finite difference step size  */
}

static void THIInitialize_HOM_C(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units units = thi->units;
  PetscReal s = -x*PetscSinReal(thi->alpha);
  p->b = s - 1000*units->meter;
  p->h = s - p->b;
  /* tau_b = beta2 v   is a stress (Pa).
   * This is a big number in our units (it needs to balance the driving force from the surface), so we scale it by 1/rhog, just like the residual. */
  p->beta2 = 1000 * (1 + PetscSinReal(x*2*PETSC_PI/thi->Lx)*PetscSinReal(y*2*PETSC_PI/thi->Ly)) * units->Pascal * units->year / units->meter / thi->rhog;
}

/* These are just toys */

/* From Fred Herman */
static void THIInitialize_HOM_F(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units units = thi->units;
  PetscReal s = -x*PetscSinReal(thi->alpha);
  p->b = s - 1000*units->meter + 100*units->meter * PetscSinReal(x*2*PETSC_PI/thi->Lx);/* * sin(y*2*PETSC_PI/thi->Ly); */
  p->h = s - p->b;
  p->h = (1-(atan((x-thi->Lx/2)/1.)+PETSC_PI/2.)/PETSC_PI)*500*units->meter+1*units->meter;
  s = PetscRealPart(p->b + p->h);
  p->beta2 = -1e-10;
  /*  p->beta2 = 1000 * units->Pascal * units->year / units->meter; */
}

/* Same bed as test A, free slip everywhere except for a discontinuous jump to a circular sticky region in the middle. */
static void THIInitialize_HOM_X(THI thi,PetscReal xx,PetscReal yy,PrmNode *p)
{
  Units units = thi->units;
  PetscReal x = xx*2*PETSC_PI/thi->Lx - PETSC_PI,y = yy*2*PETSC_PI/thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x*x + y*y),s = -x*PetscSinReal(thi->alpha);
  p->b = s - 1000*units->meter + 500*units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h = s - p->b;
  p->beta2 = 1000 * (r < 1 ? 2 : 0) * units->Pascal * units->year / units->meter / thi->rhog;
}

/* Like Z, but with 200 meter cliffs */
static void THIInitialize_HOM_Y(THI thi,PetscReal xx,PetscReal yy,PrmNode *p)
{
  Units units = thi->units;
  PetscReal x = xx*2*PETSC_PI/thi->Lx - PETSC_PI,y = yy*2*PETSC_PI/thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x*x + y*y),s = -x*PetscSinReal(thi->alpha);
  p->b = s - 1000*units->meter + 500*units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  if (PetscRealPart(p->b) > -700*units->meter) p->b += 200*units->meter;
  p->h = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16*r))/PetscSqrtReal(1e-2 + 16*r)*PetscCosReal(x*3/2)*PetscCosReal(y*3/2)) * units->Pascal * units->year / units->meter / thi->rhog;
}

/* Same bed as A, smoothly varying slipperiness, similar to MATLAB's "sombrero" (uncorrelated with bathymetry) */
static void THIInitialize_HOM_Z(THI thi,PetscReal xx,PetscReal yy,PrmNode *p)
{
  Units units = thi->units;
  PetscReal x = xx*2*PETSC_PI/thi->Lx - PETSC_PI,y = yy*2*PETSC_PI/thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x*x + y*y),s = -x*PetscSinReal(thi->alpha);
  p->b = s - 1000*units->meter + 500*units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16*r))/PetscSqrtReal(1e-2 + 16*r)*PetscCosReal(x*3/2)*PetscCosReal(y*3/2)) * units->Pascal * units->year / units->meter / thi->rhog;
}

static void THIFriction(THI thi,PetscReal rbeta2,PetscReal gam,PetscReal *beta2,PetscReal *dbeta2)
{
  if (thi->friction.irefgam == 0) {
    Units units = thi->units;
    thi->friction.irefgam = 1./(0.5*PetscSqr(100 * units->meter / units->year));
    thi->friction.eps2 = 0.5*PetscSqr(1.e-4 / thi->friction.irefgam);
  }
  if (thi->friction.exponent == 0) {
    *beta2  = rbeta2;
    *dbeta2 = 0;
  } else {
    *beta2  = rbeta2 * PetscPowReal(thi->friction.eps2 + gam*thi->friction.irefgam,thi->friction.exponent);
    *dbeta2 = thi->friction.exponent * *beta2 / (thi->friction.eps2 + gam*thi->friction.irefgam) * thi->friction.irefgam;
  }
}

static void THIViscosity(THI thi,PetscReal gam,PetscReal *eta,PetscReal *deta)
{
  PetscReal Bd2,eps,exponent;
  if (thi->viscosity.Bd2 == 0) {
    Units units = thi->units;
    const PetscReal
      n = thi->viscosity.glen_n,                        /* Glen exponent */
      p = 1. + 1./n,                                    /* for Stokes */
      A = 1.e-16 * PetscPowReal(units->Pascal,-n) / units->year, /* softness parameter (Pa^{-n}/s) */
      B = PetscPowReal(A,-1./n);                                 /* hardness parameter */
    thi->viscosity.Bd2      = B/2;
    thi->viscosity.exponent = (p-2)/2;
    thi->viscosity.eps      = 0.5*PetscSqr(1e-5 / units->year);
  }
  Bd2      = thi->viscosity.Bd2;
  exponent = thi->viscosity.exponent;
  eps      = thi->viscosity.eps;
  *eta     = Bd2 * PetscPowReal(eps + gam,exponent);
  *deta    = exponent * (*eta) / (eps + gam);
}

static void THIErosion(THI thi,const Node *vel,PetscScalar *erate,Node *derate)
{
  const PetscScalar magref2 = 1.e-10 + (PetscSqr(vel->u) + PetscSqr(vel->v)) / PetscSqr(thi->erosion.refvel),
                    rate    = -thi->erosion.rate*PetscPowScalar(magref2, 0.5*thi->erosion.exponent);
  if (erate) *erate = rate;
  if (derate) {
    if (thi->erosion.exponent == 1) {
      derate->u = 0;
      derate->v = 0;
    } else {
      derate->u = 0.5*thi->erosion.exponent * rate / magref2 * 2. * vel->u / PetscSqr(thi->erosion.refvel);
      derate->v = 0.5*thi->erosion.exponent * rate / magref2 * 2. * vel->v / PetscSqr(thi->erosion.refvel);
    }
  }
}

static void RangeUpdate(PetscReal *min,PetscReal *max,PetscReal x)
{
  if (x < *min) *min = x;
  if (x > *max) *max = x;
}

static void PRangeClear(PRange *p)
{
  p->cmin = p->min = 1e100;
  p->cmax = p->max = -1e100;
}

static PetscErrorCode PRangeMinMax(PRange *p,PetscReal min,PetscReal max)
{

  PetscFunctionBeginUser;
  p->cmin = min;
  p->cmax = max;
  if (min < p->min) p->min = min;
  if (max > p->max) p->max = max;
  PetscFunctionReturn(0);
}

static PetscErrorCode THIDestroy(THI *thi)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (--((PetscObject)(*thi))->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree((*thi)->units);CHKERRQ(ierr);
  ierr = PetscFree((*thi)->mattype);CHKERRQ(ierr);
  ierr = PetscFree((*thi)->monitor_basename);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(thi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode THICreate(MPI_Comm comm,THI *inthi)
{
  static PetscBool registered = PETSC_FALSE;
  THI              thi;
  Units            units;
  char             monitor_basename[PETSC_MAX_PATH_LEN] = "thi-";
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  *inthi = 0;
  if (!registered) {
    ierr = PetscClassIdRegister("Toy Hydrostatic Ice",&THI_CLASSID);CHKERRQ(ierr);
    registered = PETSC_TRUE;
  }
  ierr = PetscHeaderCreate(thi,THI_CLASSID,"THI","Toy Hydrostatic Ice","THI",comm,THIDestroy,0);CHKERRQ(ierr);

  ierr = PetscNew(&thi->units);CHKERRQ(ierr);

  units           = thi->units;
  units->meter    = 1e-2;
  units->second   = 1e-7;
  units->kilogram = 1e-12;

  ierr = PetscOptionsBegin(comm,NULL,"Scaled units options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-units_meter","1 meter in scaled length units","",units->meter,&units->meter,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-units_second","1 second in scaled time units","",units->second,&units->second,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-units_kilogram","1 kilogram in scaled mass units","",units->kilogram,&units->kilogram,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));
  units->year   = 31556926. * units->second, /* seconds per year */

  thi->Lx              = 10.e3;
  thi->Ly              = 10.e3;
  thi->Lz              = 1000;
  thi->nlevels         = 1;
  thi->dirichlet_scale = 1;
  thi->verbose         = PETSC_FALSE;

  thi->viscosity.glen_n = 3.;
  thi->erosion.rate     = 1e-3; /* m/a */
  thi->erosion.exponent = 1.;
  thi->erosion.refvel   = 1.;   /* m/a */

  ierr = PetscOptionsBegin(comm,NULL,"Toy Hydrostatic Ice options","");CHKERRQ(ierr);
  {
    QuadratureType quad       = QUAD_GAUSS;
    char           homexp[]   = "A";
    char           mtype[256] = MATSBAIJ;
    PetscReal      L,m = 1.0;
    PetscBool      flg;
    L    = thi->Lx;
    ierr = PetscOptionsReal("-thi_L","Domain size (m)","",L,&L,&flg);CHKERRQ(ierr);
    if (flg) thi->Lx = thi->Ly = L;
    ierr = PetscOptionsReal("-thi_Lx","X Domain size (m)","",thi->Lx,&thi->Lx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_Ly","Y Domain size (m)","",thi->Ly,&thi->Ly,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_Lz","Z Domain size (m)","",thi->Lz,&thi->Lz,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-thi_hom","ISMIP-HOM experiment (A or C)","",homexp,homexp,sizeof(homexp),NULL);CHKERRQ(ierr);
    switch (homexp[0] = toupper(homexp[0])) {
    case 'A':
      thi->initialize = THIInitialize_HOM_A;
      thi->no_slip    = PETSC_TRUE;
      thi->alpha      = 0.5;
      break;
    case 'C':
      thi->initialize = THIInitialize_HOM_C;
      thi->no_slip    = PETSC_FALSE;
      thi->alpha      = 0.1;
      break;
    case 'F':
      thi->initialize = THIInitialize_HOM_F;
      thi->no_slip    = PETSC_FALSE;
      thi->alpha      = 0.5;
      break;
    case 'X':
      thi->initialize = THIInitialize_HOM_X;
      thi->no_slip    = PETSC_FALSE;
      thi->alpha      = 0.3;
      break;
    case 'Y':
      thi->initialize = THIInitialize_HOM_Y;
      thi->no_slip    = PETSC_FALSE;
      thi->alpha      = 0.5;
      break;
    case 'Z':
      thi->initialize = THIInitialize_HOM_Z;
      thi->no_slip    = PETSC_FALSE;
      thi->alpha      = 0.5;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HOM experiment '%c' not implemented",homexp[0]);
    }
    ierr = PetscOptionsEnum("-thi_quadrature","Quadrature to use for 3D elements","",QuadratureTypes,(PetscEnum)quad,(PetscEnum*)&quad,NULL);CHKERRQ(ierr);
    switch (quad) {
    case QUAD_GAUSS:
      HexQInterp = HexQInterp_Gauss;
      HexQDeriv  = HexQDeriv_Gauss;
      break;
    case QUAD_LOBATTO:
      HexQInterp = HexQInterp_Lobatto;
      HexQDeriv  = HexQDeriv_Lobatto;
      break;
    }
    ierr = PetscOptionsReal("-thi_alpha","Bed angle (degrees)","",thi->alpha,&thi->alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_viscosity_glen_n","Exponent in Glen flow law, 1=linear, infty=ideal plastic",NULL,thi->viscosity.glen_n,&thi->viscosity.glen_n,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_friction_m","Friction exponent, 0=Coulomb, 1=Navier","",m,&m,NULL);CHKERRQ(ierr);
    thi->friction.exponent = (m-1)/2;
    ierr = PetscOptionsReal("-thi_erosion_rate","Rate of erosion relative to sliding velocity at reference velocity (m/a)",NULL,thi->erosion.rate,&thi->erosion.rate,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_erosion_exponent","Power of sliding velocity appearing in erosion relation",NULL,thi->erosion.exponent,&thi->erosion.exponent,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_erosion_refvel","Reference sliding velocity for erosion (m/a)",NULL,thi->erosion.refvel,&thi->erosion.refvel,NULL);CHKERRQ(ierr);
    thi->erosion.rate   *= units->meter / units->year;
    thi->erosion.refvel *= units->meter / units->year;
    ierr = PetscOptionsReal("-thi_dirichlet_scale","Scale Dirichlet boundary conditions by this factor","",thi->dirichlet_scale,&thi->dirichlet_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_ssa_friction_scale","Scale slip boundary conditions by this factor in SSA (2D) assembly","",thi->ssa_friction_scale,&thi->ssa_friction_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_inertia","Coefficient of accelaration term in velocity system, physical is almost zero",NULL,thi->inertia,&thi->inertia,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-thi_nlevels","Number of levels of refinement","",thi->nlevels,&thi->nlevels,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-thi_mat_type","Matrix type","MatSetType",MatList,mtype,(char*)mtype,sizeof(mtype),NULL);CHKERRQ(ierr);
    ierr = PetscStrallocpy(mtype,&thi->mattype);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-thi_verbose","Enable verbose output (like matrix sizes and statistics)","",thi->verbose,&thi->verbose,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-thi_monitor","Basename to write state files to",NULL,monitor_basename,monitor_basename,sizeof(monitor_basename),&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscStrallocpy(monitor_basename,&thi->monitor_basename);CHKERRQ(ierr);
      thi->monitor_interval = 1;
      ierr = PetscOptionsInt("-thi_monitor_interval","Frequency at which to write state files",NULL,thi->monitor_interval,&thi->monitor_interval,NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* dimensionalize */
  thi->Lx    *= units->meter;
  thi->Ly    *= units->meter;
  thi->Lz    *= units->meter;
  thi->alpha *= PETSC_PI / 180;

  PRangeClear(&thi->eta);
  PRangeClear(&thi->beta2);

  {
    PetscReal u       = 1000*units->meter/(3e7*units->second),
              gradu   = u / (100*units->meter),eta,deta,
              rho     = 910 * units->kilogram/PetscPowRealInt(units->meter,3),
              grav    = 9.81 * units->meter/PetscSqr(units->second),
              driving = rho * grav * PetscSinReal(thi->alpha) * 1000*units->meter;
    THIViscosity(thi,0.5*gradu*gradu,&eta,&deta);
    thi->rhog = rho * grav;
    if (thi->verbose) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Units: meter %8.2g  second %8.2g  kg %8.2g  Pa %8.2g\n",units->meter,units->second,units->kilogram,units->Pascal);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Domain (%6.2g,%6.2g,%6.2g), pressure %8.2g, driving stress %8.2g\n",thi->Lx,thi->Ly,thi->Lz,rho*grav*1e3*units->meter,driving);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Large velocity 1km/a %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",u,gradu,eta,2*eta*gradu,2*eta*gradu/driving);CHKERRQ(ierr);
      THIViscosity(thi,0.5*PetscSqr(1e-3*gradu),&eta,&deta);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Small velocity 1m/a  %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",1e-3*u,1e-3*gradu,eta,2*eta*1e-3*gradu,2*eta*1e-3*gradu/driving);CHKERRQ(ierr);
    }
  }

  *inthi = thi;
  PetscFunctionReturn(0);
}

/* Our problem is periodic, but the domain has a mean slope of alpha so the bed does not line up between the upstream
 * and downstream ends of the domain.  This function fixes the ghost values so that the domain appears truly periodic in
 * the horizontal. */
static PetscErrorCode THIFixGhosts(THI thi,DM da3,DM da2,Vec X3,Vec X2)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PrmNode        **x2;
  PetscInt       i,j;

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(da3,&info);CHKERRQ(ierr);
  /* ierr = VecView(X2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = DMDAVecGetArray(da2,X2,&x2);CHKERRQ(ierr);
  for (i=info.gzs; i<info.gzs+info.gzm; i++) {
    if (i > -1 && i < info.mz) continue;
    for (j=info.gys; j<info.gys+info.gym; j++) {
      x2[i][j].b += PetscSinReal(thi->alpha) * thi->Lx * (i<0 ? 1.0 : -1.0);
    }
  }
  ierr = DMDAVecRestoreArray(da2,X2,&x2);CHKERRQ(ierr);
  /* ierr = VecView(X2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

static PetscErrorCode THIInitializePrm(THI thi,DM da2prm,PrmNode **p)
{
  PetscInt       i,j,xs,xm,ys,ym,mx,my;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMDAGetGhostCorners(da2prm,&ys,&xs,0,&ym,&xm,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da2prm,0, &my,&mx,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PetscReal xx = thi->Lx*i/mx,yy = thi->Ly*j/my;
      thi->initialize(thi,xx,yy,&p[i][j]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIInitial(THI thi,DM pack,Vec X)
{
  DM             da3,da2;
  PetscInt       i,j,k,xs,xm,ys,ym,zs,zm,mx,my;
  PetscReal      hx,hy;
  PrmNode        **prm;
  Node           ***x;
  Vec            X3g,X2g,X2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(pack,&da3,&da2);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(pack,X,&X3g,&X2g);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da2,&X2);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da3,0, 0,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da3,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da3,X3g,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,&prm);CHKERRQ(ierr);

  ierr = THIInitializePrm(thi,da2,prm);CHKERRQ(ierr);

  hx = thi->Lx / mx;
  hy = thi->Ly / my;
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      for (k=zs; k<zs+zm; k++) {
        const PetscScalar zm1      = zm-1,
                          drivingx = thi->rhog * (prm[i+1][j].b+prm[i+1][j].h - prm[i-1][j].b-prm[i-1][j].h) / (2*hx),
                          drivingy = thi->rhog * (prm[i][j+1].b+prm[i][j+1].h - prm[i][j-1].b-prm[i][j-1].h) / (2*hy);
        x[i][j][k].u = 0. * drivingx * prm[i][j].h*(PetscScalar)k/zm1;
        x[i][j][k].v = 0. * drivingy * prm[i][j].h*(PetscScalar)k/zm1;
      }
    }
  }

  ierr = DMDAVecRestoreArray(da3,X3g,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,&prm);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(da2,X2,INSERT_VALUES,X2g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da2,X2,INSERT_VALUES,X2g);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da2,&X2);CHKERRQ(ierr);

  ierr = DMCompositeRestoreAccess(pack,X,&X3g,&X2g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void PointwiseNonlinearity(THI thi,const Node n[restrict 8],const PetscReal phi[restrict 3],PetscReal dphi[restrict 8][3],PetscScalar *restrict u,PetscScalar *restrict v,PetscScalar du[restrict 3],PetscScalar dv[restrict 3],PetscReal *eta,PetscReal *deta)
{
  PetscInt    l,ll;
  PetscScalar gam;

  du[0] = du[1] = du[2] = 0;
  dv[0] = dv[1] = dv[2] = 0;
  *u    = 0;
  *v    = 0;
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

static PetscErrorCode THIFunctionLocal_3D(DMDALocalInfo *info,const Node ***x,const PrmNode **prm,const Node ***xdot,Node ***f,THI thi)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l;
  PetscReal      hx,hy,etamin,etamax,beta2min,beta2max;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;
  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  etamin   = 1e100;
  etamax   = 0;
  beta2min = 1e100;
  beta2max = 0;

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4],dpn[4][2];
      QuadExtract(prm,i,j,pn);
      ierr = QuadComputeGrad4(QuadQDeriv,hx,hy,pn,dpn);CHKERRQ(ierr);
      for (k=0; k<zm-1; k++) {
        PetscInt  ls = 0;
        Node      n[8],ndot[8],*fn[8];
        PetscReal zn[8],etabase = 0;

        PrmHexGetZ(pn,k,zm,zn);
        HexExtract(x,i,j,k,n);
        HexExtract(xdot,i,j,k,ndot);
        HexExtractRef(f,i,j,k,fn);
        if (thi->no_slip && k == 0) {
          for (l=0; l<4; l++) n[l].u = n[l].v = 0;
          /* The first 4 basis functions lie on the bottom layer, so their contribution is exactly 0, hence we can skip them */
          ls = 4;
        }
        for (q=0; q<8; q++) {
          PetscReal   dz[3],phi[8],dphi[8][3],jw,eta,deta;
          PetscScalar du[3],dv[3],u,v,udot=0,vdot=0;
          for (l=ls; l<8; l++) {
            udot += HexQInterp[q][l]*ndot[l].u;
            vdot += HexQInterp[q][l]*ndot[l].v;
          }
          HexGrad(HexQDeriv[q],zn,dz);
          HexComputeGeometry(q,hx,hy,dz,phi,dphi,&jw);
          PointwiseNonlinearity(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
          jw /= thi->rhog;      /* scales residuals to be O(1) */
          if (q == 0) etabase = eta;
          RangeUpdate(&etamin,&etamax,eta);
          for (l=ls; l<8; l++) { /* test functions */
            const PetscScalar ds[2] = {dpn[q%4][0].h+dpn[q%4][0].b, dpn[q%4][1].h+dpn[q%4][1].b};
            const PetscReal   pp    = phi[l],*dp = dphi[l];
            fn[l]->u += dp[0]*jw*eta*(4.*du[0]+2.*dv[1]) + dp[1]*jw*eta*(du[1]+dv[0]) + dp[2]*jw*eta*du[2] + pp*jw*thi->rhog*ds[0];
            fn[l]->v += dp[1]*jw*eta*(2.*du[0]+4.*dv[1]) + dp[0]*jw*eta*(du[1]+dv[0]) + dp[2]*jw*eta*dv[2] + pp*jw*thi->rhog*ds[1];
            fn[l]->u += pp*jw*udot*thi->inertia*pp;
            fn[l]->v += pp*jw*vdot*thi->inertia*pp;
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
            *
            * Note that the residual at this Dirichlet node is linear in the state at this node, but also depends
            * (nonlinearly in general) on the neighboring interior nodes through the local viscosity.  This will make
            * a matrix-free Jacobian have extra entries in the corresponding row.  We assemble only the diagonal part,
            * so the solution will exactly satisfy the boundary condition after the first linear iteration.
            */
            const PetscReal   hz    = PetscRealPart(pn[0].h)/(zm-1.);
            const PetscScalar diagu = 2*etabase/thi->rhog*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase/thi->rhog*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            fn[0]->u = thi->dirichlet_scale*diagu*x[i][j][k].u;
            fn[0]->v = thi->dirichlet_scale*diagv*x[i][j][k].v;
          } else {              /* Integrate over bottom face to apply boundary condition */
            for (q=0; q<4; q++) { /* We remove the explicit scaling of the residual by 1/rhog because beta2 already has that scaling to be O(1) */
              const PetscReal jw = 0.25*hx*hy,*phi = QuadQInterp[q];
              PetscScalar     u  =0,v=0,rbeta2=0;
              PetscReal       beta2,dbeta2;
              for (l=0; l<4; l++) {
                u     += phi[l]*n[l].u;
                v     += phi[l]*n[l].v;
                rbeta2 += phi[l]*pn[l].beta2;
              }
              THIFriction(thi,PetscRealPart(rbeta2),PetscRealPart(u*u+v*v)/2,&beta2,&dbeta2);
              RangeUpdate(&beta2min,&beta2max,beta2);
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

  ierr = PRangeMinMax(&thi->eta,etamin,etamax);CHKERRQ(ierr);
  ierr = PRangeMinMax(&thi->beta2,beta2min,beta2max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode THIFunctionLocal_2D(DMDALocalInfo *info,const Node ***x,const PrmNode **prm,const PrmNode **prmdot,PrmNode **f,THI thi)
{
  PetscInt xs,ys,xm,ym,zm,i,j,k;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PetscScalar div = 0,erate,h[8];
      PrmNodeGetFaceMeasure(prm,i,j,h);
      for (k=0; k<zm; k++) {
        PetscScalar weight = (k==0 || k == zm-1) ? 0.5/(zm-1) : 1.0/(zm-1);
        if (0) {                /* centered flux */
          div += (- weight*h[0] * StaggeredMidpoint2D(x[i][j][k].u,x[i-1][j][k].u, x[i-1][j-1][k].u,x[i][j-1][k].u)
                  - weight*h[1] * StaggeredMidpoint2D(x[i][j][k].u,x[i-1][j][k].u, x[i-1][j+1][k].u,x[i][j+1][k].u)
                  + weight*h[2] * StaggeredMidpoint2D(x[i][j][k].u,x[i+1][j][k].u, x[i+1][j+1][k].u,x[i][j+1][k].u)
                  + weight*h[3] * StaggeredMidpoint2D(x[i][j][k].u,x[i+1][j][k].u, x[i+1][j-1][k].u,x[i][j-1][k].u)
                  - weight*h[4] * StaggeredMidpoint2D(x[i][j][k].v,x[i][j-1][k].v, x[i+1][j-1][k].v,x[i+1][j][k].v)
                  - weight*h[5] * StaggeredMidpoint2D(x[i][j][k].v,x[i][j-1][k].v, x[i-1][j-1][k].v,x[i-1][j][k].v)
                  + weight*h[6] * StaggeredMidpoint2D(x[i][j][k].v,x[i][j+1][k].v, x[i-1][j+1][k].v,x[i-1][j][k].v)
                  + weight*h[7] * StaggeredMidpoint2D(x[i][j][k].v,x[i][j+1][k].v, x[i+1][j+1][k].v,x[i+1][j][k].v));
        } else {                /* Upwind flux */
          div += weight*(-UpwindFluxXW(x,prm,h,i,j,k, 1)
                         -UpwindFluxXW(x,prm,h,i,j,k,-1)
                         +UpwindFluxXE(x,prm,h,i,j,k, 1)
                         +UpwindFluxXE(x,prm,h,i,j,k,-1)
                         -UpwindFluxYS(x,prm,h,i,j,k, 1)
                         -UpwindFluxYS(x,prm,h,i,j,k,-1)
                         +UpwindFluxYN(x,prm,h,i,j,k, 1)
                         +UpwindFluxYN(x,prm,h,i,j,k,-1));
        }
      }
      /* printf("div[%d][%d] %g\n",i,j,div); */
      THIErosion(thi,&x[i][j][0],&erate,NULL);
      f[i][j].b     = prmdot[i][j].b - erate;
      f[i][j].h     = prmdot[i][j].h + div;
      f[i][j].beta2 = prmdot[i][j].beta2;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  THI            thi = (THI)ctx;
  DM             pack,da3,da2;
  Vec            X3,X2,Xdot3,Xdot2,F3,F2,F3g,F2g;
  const Node     ***x3,***xdot3;
  const PrmNode  **x2,**xdot2;
  Node           ***f3;
  PrmNode        **f2;
  DMDALocalInfo  info3;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = DMCompositeGetEntries(pack,&da3,&da2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da3,&info3);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(pack,&X3,&X2);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(pack,&Xdot3,&Xdot2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(pack,X,X3,X2);CHKERRQ(ierr);
  ierr = THIFixGhosts(thi,da3,da2,X3,X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(pack,Xdot,Xdot3,Xdot2);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da3,&F3);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da2,&F2);CHKERRQ(ierr);
  ierr = VecZeroEntries(F3);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da3,X3,&x3);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,&x2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da3,Xdot3,&xdot3);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,Xdot2,&xdot2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da3,F3,&f3);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,F2,&f2);CHKERRQ(ierr);

  ierr = THIFunctionLocal_3D(&info3,x3,x2,xdot3,f3,thi);CHKERRQ(ierr);
  ierr = THIFunctionLocal_2D(&info3,x3,x2,xdot2,f2,thi);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da3,X3,&x3);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,&x2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da3,Xdot3,&xdot3);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,Xdot2,&xdot2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da3,F3,&f3);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,F2,&f2);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(pack,&X3,&X2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(pack,&Xdot3,&Xdot2);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(pack,F,&F3g,&F2g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da3,F3,ADD_VALUES,F3g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da3,F3,ADD_VALUES,F3g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da2,F2,INSERT_VALUES,F2g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da2,F2,INSERT_VALUES,F2g);CHKERRQ(ierr);

  if (thi->verbose) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)thi),&viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"3D_Velocity residual (bs=2):\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = VecView(F3,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"2D_Fields residual (bs=3):\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = VecView(F2,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }

  ierr = DMCompositeRestoreAccess(pack,F,&F3g,&F2g);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da3,&F3);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da2,&F2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode THIMatrixStatistics(THI thi,Mat B,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      nrm;
  PetscInt       m;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
  ierr = MatGetSize(B,&m,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)B),&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    PetscScalar val0,val2;
    ierr = MatGetValue(B,0,0,&val0);CHKERRQ(ierr);
    ierr = MatGetValue(B,2,2,&val2);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Matrix dim %8d  norm %8.2e, (0,0) %8.2e  (2,2) %8.2e, eta [%8.2e,%8.2e] beta2 [%8.2e,%8.2e]\n",m,nrm,PetscRealPart(val0),PetscRealPart(val2),thi->eta.cmin,thi->eta.cmax,thi->beta2.cmin,thi->beta2.cmax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THISurfaceStatistics(DM pack,Vec X,PetscReal *min,PetscReal *max,PetscReal *mean)
{
  PetscErrorCode ierr;
  DM             da3,da2;
  Vec            X3,X2;
  Node           ***x;
  PetscInt       i,j,xs,ys,zs,xm,ym,zm,mx,my,mz;
  PetscReal      umin = 1e100,umax=-1e100;
  PetscScalar    usum =0.0,gusum;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(pack,&da3,&da2);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(pack,X,&X3,&X2);CHKERRQ(ierr);
  *min = *max = *mean = 0;
  ierr = DMDAGetInfo(da3,0, &mz,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da3,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
  PetscCheckFalse(zs != 0 || zm != mz,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unexpected decomposition");
  ierr = DMDAVecGetArray(da3,X3,&x);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PetscReal u = PetscRealPart(x[i][j][zm-1].u);
      RangeUpdate(&umin,&umax,u);
      usum += u;
    }
  }
  ierr = DMDAVecRestoreArray(da3,X3,&x);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(pack,X,&X3,&X2);CHKERRQ(ierr);

  ierr  = MPI_Allreduce(&umin,min,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)da3));CHKERRMPI(ierr);
  ierr  = MPI_Allreduce(&umax,max,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da3));CHKERRMPI(ierr);
  ierr  = MPI_Allreduce(&usum,&gusum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)da3));CHKERRMPI(ierr);
  *mean = PetscRealPart(gusum) / (mx*my);
  PetscFunctionReturn(0);
}

static PetscErrorCode THISolveStatistics(THI thi,TS ts,PetscInt coarsened,const char name[])
{
  MPI_Comm       comm;
  DM             pack;
  Vec            X,X3,X2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)thi,&comm);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(pack,X,&X3,&X2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Solution statistics after solve: %s\n",name);CHKERRQ(ierr);
  {
    PetscInt            its,lits;
    SNESConvergedReason reason;
    SNES                snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"%s: Number of SNES iterations = %d, total linear iterations = %d\n",SNESConvergedReasons[reason],its,lits);CHKERRQ(ierr);
  }
  {
    PetscReal   nrm2,tmin[3]={1e100,1e100,1e100},tmax[3]={-1e100,-1e100,-1e100},min[3],max[3];
    PetscInt    i,j,m;
    PetscScalar *x;
    ierr = VecNorm(X3,NORM_2,&nrm2);CHKERRQ(ierr);
    ierr = VecGetLocalSize(X3,&m);CHKERRQ(ierr);
    ierr = VecGetArray(X3,&x);CHKERRQ(ierr);
    for (i=0; i<m; i+=2) {
      PetscReal u = PetscRealPart(x[i]),v = PetscRealPart(x[i+1]),c = PetscSqrtReal(u*u+v*v);
      tmin[0] = PetscMin(u,tmin[0]);
      tmin[1] = PetscMin(v,tmin[1]);
      tmin[2] = PetscMin(c,tmin[2]);
      tmax[0] = PetscMax(u,tmax[0]);
      tmax[1] = PetscMax(v,tmax[1]);
      tmax[2] = PetscMax(c,tmax[2]);
    }
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = MPI_Allreduce(tmin,min,3,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)thi));CHKERRMPI(ierr);
    ierr = MPI_Allreduce(tmax,max,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)thi));CHKERRMPI(ierr);
    /* Dimensionalize to meters/year */
    nrm2 *= thi->units->year / thi->units->meter;
    for (j=0; j<3; j++) {
      min[j] *= thi->units->year / thi->units->meter;
      max[j] *= thi->units->year / thi->units->meter;
    }
    ierr = PetscPrintf(comm,"|X|_2 %g   u in [%g, %g]   v in [%g, %g]   c in [%g, %g] \n",nrm2,min[0],max[0],min[1],max[1],min[2],max[2]);CHKERRQ(ierr);
    {
      PetscReal umin,umax,umean;
      ierr   = THISurfaceStatistics(pack,X,&umin,&umax,&umean);CHKERRQ(ierr);
      umin  *= thi->units->year / thi->units->meter;
      umax  *= thi->units->year / thi->units->meter;
      umean *= thi->units->year / thi->units->meter;
      ierr   = PetscPrintf(comm,"Surface statistics: u in [%12.6e, %12.6e] mean %12.6e\n",umin,umax,umean);CHKERRQ(ierr);
    }
    /* These values stay nondimensional */
    ierr = PetscPrintf(comm,"Global eta range   [%g, %g], converged range [%g, %g]\n",thi->eta.min,thi->eta.max,thi->eta.cmin,thi->eta.cmax);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Global beta2 range [%g, %g], converged range [%g, %g]\n",thi->beta2.min,thi->beta2.max,thi->beta2.cmin,thi->beta2.cmax);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(pack,X,&X3,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscInt DMDALocalIndex3D(DMDALocalInfo *info,PetscInt i,PetscInt j,PetscInt k)
{return ((i-info->gzs)*info->gym + (j-info->gys))*info->gxm + (k-info->gxs);}
static inline PetscInt DMDALocalIndex2D(DMDALocalInfo *info,PetscInt i,PetscInt j)
{return (i-info->gzs)*info->gym + (j-info->gys);}

static PetscErrorCode THIJacobianLocal_Momentum(DMDALocalInfo *info,const Node ***x,const PrmNode **prm,Mat B,Mat Bcpl,THI thi)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l,ll;
  PetscReal      hx,hy;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;
  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4],dpn[4][2];
      QuadExtract(prm,i,j,pn);
      ierr = QuadComputeGrad4(QuadQDeriv,hx,hy,pn,dpn);CHKERRQ(ierr);
      for (k=0; k<zm-1; k++) {
        Node        n[8];
        PetscReal   zn[8],etabase = 0;
        PetscScalar Ke[8*NODE_SIZE][8*NODE_SIZE],Kcpl[8*NODE_SIZE][4*PRMNODE_SIZE];
        PetscInt    ls = 0;

        PrmHexGetZ(pn,k,zm,zn);
        HexExtract(x,i,j,k,n);
        ierr = PetscMemzero(Ke,sizeof(Ke));CHKERRQ(ierr);
        ierr = PetscMemzero(Kcpl,sizeof(Kcpl));CHKERRQ(ierr);
        if (thi->no_slip && k == 0) {
          for (l=0; l<4; l++) n[l].u = n[l].v = 0;
          ls = 4;
        }
        for (q=0; q<8; q++) {
          PetscReal   dz[3],phi[8],dphi[8][3],jw,eta,deta;
          PetscScalar du[3],dv[3],u,v;
          HexGrad(HexQDeriv[q],zn,dz);
          HexComputeGeometry(q,hx,hy,dz,phi,dphi,&jw);
          PointwiseNonlinearity(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
          jw /= thi->rhog;      /* residuals are scaled by this factor */
          if (q == 0) etabase = eta;
          for (l=ls; l<8; l++) { /* test functions */
            const PetscReal pp=phi[l],*restrict dp = dphi[l];
            for (ll=ls; ll<8; ll++) { /* trial functions */
              const PetscReal *restrict dpl = dphi[ll];
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
              /* inertial part */
              Ke[l*2+0][ll*2+0] += pp*jw*thi->inertia*pp;
              Ke[l*2+1][ll*2+1] += pp*jw*thi->inertia*pp;
            }
            for (ll=0; ll<4; ll++) { /* Trial functions for surface/bed */
              const PetscReal dpl[] = {QuadQDeriv[q%4][ll][0]/hx, QuadQDeriv[q%4][ll][1]/hy}; /* surface = h + b */
              Kcpl[FieldIndex(Node,l,u)][FieldIndex(PrmNode,ll,h)] += pp*jw*thi->rhog*dpl[0];
              Kcpl[FieldIndex(Node,l,u)][FieldIndex(PrmNode,ll,b)] += pp*jw*thi->rhog*dpl[0];
              Kcpl[FieldIndex(Node,l,v)][FieldIndex(PrmNode,ll,h)] += pp*jw*thi->rhog*dpl[1];
              Kcpl[FieldIndex(Node,l,v)][FieldIndex(PrmNode,ll,b)] += pp*jw*thi->rhog*dpl[1];
            }
          }
        }
        if (k == 0) { /* on a bottom face */
          if (thi->no_slip) {
            const PetscReal   hz    = PetscRealPart(pn[0].h)/(zm-1);
            const PetscScalar diagu = 2*etabase/thi->rhog*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase/thi->rhog*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            Ke[0][0] = thi->dirichlet_scale*diagu;
            Ke[0][1] = 0;
            Ke[1][0] = 0;
            Ke[1][1] = thi->dirichlet_scale*diagv;
          } else {
            for (q=0; q<4; q++) { /* We remove the explicit scaling by 1/rhog because beta2 already has that scaling to be O(1) */
              const PetscReal jw = 0.25*hx*hy,*phi = QuadQInterp[q];
              PetscScalar     u  =0,v=0,rbeta2=0;
              PetscReal       beta2,dbeta2;
              for (l=0; l<4; l++) {
                u      += phi[l]*n[l].u;
                v      += phi[l]*n[l].v;
                rbeta2 += phi[l]*pn[l].beta2;
              }
              THIFriction(thi,PetscRealPart(rbeta2),PetscRealPart(u*u+v*v)/2,&beta2,&dbeta2);
              for (l=0; l<4; l++) {
                const PetscReal pp = phi[l];
                for (ll=0; ll<4; ll++) {
                  const PetscReal ppl = phi[ll];
                  Ke[l*2+0][ll*2+0] += pp*jw*beta2*ppl + pp*jw*dbeta2*u*u*ppl;
                  Ke[l*2+0][ll*2+1] +=                   pp*jw*dbeta2*u*v*ppl;
                  Ke[l*2+1][ll*2+0] +=                   pp*jw*dbeta2*v*u*ppl;
                  Ke[l*2+1][ll*2+1] += pp*jw*beta2*ppl + pp*jw*dbeta2*v*v*ppl;
                }
              }
            }
          }
        }
        {
          const PetscInt rc3blocked[8] = {
            DMDALocalIndex3D(info,i+0,j+0,k+0),
            DMDALocalIndex3D(info,i+1,j+0,k+0),
            DMDALocalIndex3D(info,i+1,j+1,k+0),
            DMDALocalIndex3D(info,i+0,j+1,k+0),
            DMDALocalIndex3D(info,i+0,j+0,k+1),
            DMDALocalIndex3D(info,i+1,j+0,k+1),
            DMDALocalIndex3D(info,i+1,j+1,k+1),
            DMDALocalIndex3D(info,i+0,j+1,k+1)
          },col2blocked[PRMNODE_SIZE*4] = {
            DMDALocalIndex2D(info,i+0,j+0),
            DMDALocalIndex2D(info,i+1,j+0),
            DMDALocalIndex2D(info,i+1,j+1),
            DMDALocalIndex2D(info,i+0,j+1)
          };
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
          ierr = MatSetValuesBlockedLocal(B,8,rc3blocked,8,rc3blocked,&Ke[0][0],ADD_VALUES);CHKERRQ(ierr); /* velocity-velocity coupling can use blocked insertion */
          {                     /* The off-diagonal part cannot (yet) */
            PetscInt row3scalar[NODE_SIZE*8],col2scalar[PRMNODE_SIZE*4];
            for (l=0; l<8; l++) for (ll=0; ll<NODE_SIZE; ll++) row3scalar[l*NODE_SIZE+ll] = rc3blocked[l]*NODE_SIZE+ll;
            for (l=0; l<4; l++) for (ll=0; ll<PRMNODE_SIZE; ll++) col2scalar[l*PRMNODE_SIZE+ll] = col2blocked[l]*PRMNODE_SIZE+ll;
            ierr = MatSetValuesLocal(Bcpl,8*NODE_SIZE,row3scalar,4*PRMNODE_SIZE,col2scalar,&Kcpl[0][0],ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobianLocal_2D(DMDALocalInfo *info,const Node ***x3,const PrmNode **x2,const PrmNode **xdot2,PetscReal a,Mat B22,Mat B21,THI thi)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,xm,ym,zm,i,j,k;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;

  PetscCheckFalse(zm > 1024,((PetscObject)info->da)->comm,PETSC_ERR_SUP,"Need to allocate more space");
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      {                         /* Self-coupling */
        const PetscInt    row[]  = {DMDALocalIndex2D(info,i,j)};
        const PetscInt    col[]  = {DMDALocalIndex2D(info,i,j)};
        const PetscScalar vals[] = {
          a,0,0,
          0,a,0,
          0,0,a
        };
        ierr = MatSetValuesBlockedLocal(B22,1,row,1,col,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
      for (k=0; k<zm; k++) {    /* Coupling to velocity problem */
        /* Use a cheaper quadrature than for residual evaluation, because it is much sparser */
        const PetscInt row[]  = {FieldIndex(PrmNode,DMDALocalIndex2D(info,i,j),h)};
        const PetscInt cols[] = {
          FieldIndex(Node,DMDALocalIndex3D(info,i-1,j,k),u),
          FieldIndex(Node,DMDALocalIndex3D(info,i  ,j,k),u),
          FieldIndex(Node,DMDALocalIndex3D(info,i+1,j,k),u),
          FieldIndex(Node,DMDALocalIndex3D(info,i,j-1,k),v),
          FieldIndex(Node,DMDALocalIndex3D(info,i,j  ,k),v),
          FieldIndex(Node,DMDALocalIndex3D(info,i,j+1,k),v)
        };
        const PetscScalar
          w  = (k && k<zm-1) ? 0.5 : 0.25,
          hW = w*(x2[i-1][j  ].h+x2[i  ][j  ].h)/(zm-1.),
          hE = w*(x2[i  ][j  ].h+x2[i+1][j  ].h)/(zm-1.),
          hS = w*(x2[i  ][j-1].h+x2[i  ][j  ].h)/(zm-1.),
          hN = w*(x2[i  ][j  ].h+x2[i  ][j+1].h)/(zm-1.);
        PetscScalar *vals,
                     vals_upwind[] = {((PetscRealPart(x3[i][j][k].u) > 0) ? -hW : 0),
                                      ((PetscRealPart(x3[i][j][k].u) > 0) ? +hE : -hW),
                                      ((PetscRealPart(x3[i][j][k].u) > 0) ?  0  : +hE),
                                      ((PetscRealPart(x3[i][j][k].v) > 0) ? -hS : 0),
                                      ((PetscRealPart(x3[i][j][k].v) > 0) ? +hN : -hS),
                                      ((PetscRealPart(x3[i][j][k].v) > 0) ?  0  : +hN)},
                     vals_centered[] = {-0.5*hW, 0.5*(-hW+hE), 0.5*hE,
                                        -0.5*hS, 0.5*(-hS+hN), 0.5*hN};
        vals = 1 ? vals_upwind : vals_centered;
        if (k == 0) {
          Node derate;
          THIErosion(thi,&x3[i][j][0],NULL,&derate);
          vals[1] -= derate.u;
          vals[4] -= derate.v;
        }
        ierr = MatSetValuesLocal(B21,1,row,6,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  THI            thi = (THI)ctx;
  DM             pack,da3,da2;
  Vec            X3,X2,Xdot2;
  Mat            B11,B12,B21,B22;
  DMDALocalInfo  info3;
  IS             *isloc;
  const Node     ***x3;
  const PrmNode  **x2,**xdot2;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = DMCompositeGetEntries(pack,&da3,&da2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da3,&info3);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(pack,&X3,&X2);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(pack,NULL,&Xdot2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(pack,X,X3,X2);CHKERRQ(ierr);
  ierr = THIFixGhosts(thi,da3,da2,X3,X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(pack,Xdot,NULL,Xdot2);CHKERRQ(ierr);

  ierr = MatZeroEntries(B);CHKERRQ(ierr);

  ierr = DMCompositeGetLocalISs(pack,&isloc);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(B,isloc[0],isloc[0],&B11);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(B,isloc[0],isloc[1],&B12);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(B,isloc[1],isloc[0],&B21);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(B,isloc[1],isloc[1],&B22);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da3,X3,&x3);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,&x2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,Xdot2,&xdot2);CHKERRQ(ierr);

  ierr = THIJacobianLocal_Momentum(&info3,x3,x2,B11,B12,thi);CHKERRQ(ierr);

  /* Need to switch from ADD_VALUES to INSERT_VALUES */
  ierr = MatAssemblyBegin(B,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);

  ierr = THIJacobianLocal_2D(&info3,x3,x2,xdot2,a,B22,B21,thi);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da3,X3,&x3);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,&x2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,Xdot2,&xdot2);CHKERRQ(ierr);

  ierr = MatRestoreLocalSubMatrix(B,isloc[0],isloc[0],&B11);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(B,isloc[0],isloc[1],&B12);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(B,isloc[1],isloc[0],&B21);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(B,isloc[1],isloc[1],&B22);CHKERRQ(ierr);
  ierr = ISDestroy(&isloc[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&isloc[1]);CHKERRQ(ierr);
  ierr = PetscFree(isloc);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(pack,&X3,&X2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(pack,0,&Xdot2);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  if (thi->verbose) {ierr = THIMatrixStatistics(thi,B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* VTK's XML formats are so brain-dead that they can't handle multiple grids in the same file.  Since the communication
 * can be shared between the two grids, we write two files at once, one for velocity (living on a 3D grid defined by
 * h=thickness and b=bed) and another for all properties living on the 2D grid.
 */
static PetscErrorCode THIDAVecView_VTK_XML(THI thi,DM pack,Vec X,const char filename[],const char filename2[])
{
  const PetscInt dof   = NODE_SIZE,dof2 = PRMNODE_SIZE;
  Units          units = thi->units;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscViewer    viewer3,viewer2;
  PetscMPIInt    rank,size,tag,nn,nmax,nn2,nmax2;
  PetscInt       mx,my,mz,r,range[6];
  PetscScalar    *x,*x2;
  DM             da3,da2;
  Vec            X3,X2;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)thi,&comm);CHKERRQ(ierr);
  ierr = DMCompositeGetEntries(pack,&da3,&da2);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(pack,X,&X3,&X2);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da3,0, &mz,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIOpen(comm,filename,&viewer3);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(comm,filename2,&viewer2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer3,"<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer2,"<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer3,"  <StructuredGrid WholeExtent=\"%d %d %d %d %d %d\">\n",0,mz-1,0,my-1,0,mx-1);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer2,"  <StructuredGrid WholeExtent=\"%d %d %d %d %d %d\">\n",0,0,0,my-1,0,mx-1);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da3,range,range+1,range+2,range+3,range+4,range+5);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(range[3]*range[4]*range[5]*dof,&nn);CHKERRQ(ierr);
  ierr = MPI_Reduce(&nn,&nmax,1,MPI_INT,MPI_MAX,0,comm);CHKERRMPI(ierr);
  ierr = PetscMPIIntCast(range[4]*range[5]*dof2,&nn2);CHKERRQ(ierr);
  ierr = MPI_Reduce(&nn2,&nmax2,1,MPI_INT,MPI_MAX,0,comm);CHKERRMPI(ierr);
  tag  = ((PetscObject)viewer3)->tag;
  ierr = VecGetArrayRead(X3,(const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X2,(const PetscScalar**)&x2);CHKERRQ(ierr);
  if (rank == 0) {
    PetscScalar *array,*array2;
    ierr = PetscMalloc2(nmax,&array,nmax2,&array2);CHKERRQ(ierr);
    for (r=0; r<size; r++) {
      PetscInt    i,j,k,f,xs,xm,ys,ym,zs,zm;
      Node        *y3;
      PetscScalar (*y2)[PRMNODE_SIZE];
      MPI_Status status;

      if (r) {
        ierr = MPI_Recv(range,6,MPIU_INT,r,tag,comm,MPI_STATUS_IGNORE);CHKERRMPI(ierr);
      }
      zs = range[0];ys = range[1];xs = range[2];zm = range[3];ym = range[4];xm = range[5];
      PetscCheckFalse(xm*ym*zm*dof > nmax,PETSC_COMM_SELF,PETSC_ERR_PLIB,"should not happen");
      if (r) {
        ierr = MPI_Recv(array,nmax,MPIU_SCALAR,r,tag,comm,&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&nn);CHKERRMPI(ierr);
        PetscCheckFalse(nn != xm*ym*zm*dof,PETSC_COMM_SELF,PETSC_ERR_PLIB,"corrupt da3 send");
        y3   = (Node*)array;
        ierr = MPI_Recv(array2,nmax2,MPIU_SCALAR,r,tag,comm,&status);CHKERRMPI(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&nn2);CHKERRMPI(ierr);
        PetscCheckFalse(nn2 != xm*ym*dof2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"corrupt da2 send");
        y2 = (PetscScalar(*)[PRMNODE_SIZE])array2;
      } else {
        y3 = (Node*)x;
        y2 = (PetscScalar(*)[PRMNODE_SIZE])x2;
      }
      ierr = PetscViewerASCIIPrintf(viewer3,"    <Piece Extent=\"%D %D %D %D %D %D\">\n",zs,zs+zm-1,ys,ys+ym-1,xs,xs+xm-1);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer2,"    <Piece Extent=\"%d %d %D %D %D %D\">\n",0,0,ys,ys+ym-1,xs,xs+xm-1);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer3,"      <Points>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer2,"      <Points>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer3,"        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer2,"        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");CHKERRQ(ierr);
      for (i=xs; i<xs+xm; i++) {
        for (j=ys; j<ys+ym; j++) {
          PetscReal
            xx = thi->Lx*i/mx,
            yy = thi->Ly*j/my,
            b  = PetscRealPart(y2[i*ym+j][FieldOffset(PrmNode,b)]),
            h  = PetscRealPart(y2[i*ym+j][FieldOffset(PrmNode,h)]);
          for (k=zs; k<zs+zm; k++) {
            PetscReal zz = b + h*k/(mz-1.);
            ierr = PetscViewerASCIIPrintf(viewer3,"%f %f %f\n",xx,yy,zz);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer2,"%f %f %f\n",xx,yy,(double)0.0);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer3,"        </DataArray>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer2,"        </DataArray>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer3,"      </Points>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer2,"      </Points>\n");CHKERRQ(ierr);

      {                         /* Velocity and rank (3D) */
        ierr = PetscViewerASCIIPrintf(viewer3,"      <PointData>\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer3,"        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");CHKERRQ(ierr);
        for (i=0; i<nn/dof; i++) {
          ierr = PetscViewerASCIIPrintf(viewer3,"%f %f %f\n",PetscRealPart(y3[i].u)*units->year/units->meter,PetscRealPart(y3[i].v)*units->year/units->meter,0.0);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer3,"        </DataArray>\n");CHKERRQ(ierr);

        ierr = PetscViewerASCIIPrintf(viewer3,"        <DataArray type=\"Int32\" Name=\"rank\" NumberOfComponents=\"1\" format=\"ascii\">\n");CHKERRQ(ierr);
        for (i=0; i<nn; i+=dof) {
          ierr = PetscViewerASCIIPrintf(viewer3,"%D\n",r);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer3,"        </DataArray>\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer3,"      </PointData>\n");CHKERRQ(ierr);
      }

      {                         /* 2D */
        ierr = PetscViewerASCIIPrintf(viewer2,"      <PointData>\n");CHKERRQ(ierr);
        for (f=0; f<PRMNODE_SIZE; f++) {
          const char *fieldname;
          ierr = DMDAGetFieldName(da2,f,&fieldname);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer2,"        <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\">\n",fieldname);CHKERRQ(ierr);
          for (i=0; i<nn2/PRMNODE_SIZE; i++) {
            ierr = PetscViewerASCIIPrintf(viewer2,"%g\n",y2[i][f]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer2,"        </DataArray>\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer2,"      </PointData>\n");CHKERRQ(ierr);
      }

      ierr = PetscViewerASCIIPrintf(viewer3,"    </Piece>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer2,"    </Piece>\n");CHKERRQ(ierr);
    }
    ierr = PetscFree2(array,array2);CHKERRQ(ierr);
  } else {
    ierr = MPI_Send(range,6,MPIU_INT,0,tag,comm);CHKERRMPI(ierr);
    ierr = MPI_Send(x,nn,MPIU_SCALAR,0,tag,comm);CHKERRMPI(ierr);
    ierr = MPI_Send(x2,nn2,MPIU_SCALAR,0,tag,comm);CHKERRMPI(ierr);
  }
  ierr = VecRestoreArrayRead(X3,(const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X2,(const PetscScalar**)&x2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer3,"  </StructuredGrid>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer2,"  </StructuredGrid>\n");CHKERRQ(ierr);

  ierr = DMCompositeRestoreAccess(pack,X,&X3,&X2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer3,"</VTKFile>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer2,"</VTKFile>\n");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer3);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode THITSMonitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  THI            thi = (THI)ctx;
  DM             pack;
  char           filename3[PETSC_MAX_PATH_LEN],filename2[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(0); /* negative one is used to indicate an interpolated solution */
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%3D: t=%g\n",step,(double)t);CHKERRQ(ierr);
  if (thi->monitor_interval && step % thi->monitor_interval) PetscFunctionReturn(0);
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename3,sizeof(filename3),"%s-3d-%03d.vts",thi->monitor_basename,step);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename2,sizeof(filename2),"%s-2d-%03d.vts",thi->monitor_basename,step);CHKERRQ(ierr);
  ierr = THIDAVecView_VTK_XML(thi,pack,X,filename3,filename2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode THICreateDM3d(THI thi,DM *dm3d)
{
  MPI_Comm       comm;
  PetscInt       M    = 3,N = 3,P = 2;
  DM             da;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)thi,&comm);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm,NULL,"Grid resolution options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-M","Number of elements in x-direction on coarse level","",M,&M,NULL);CHKERRQ(ierr);
    N    = M;
    ierr = PetscOptionsInt("-N","Number of elements in y-direction on coarse level (if different from M)","",N,&N,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-P","Number of elements in z-direction on coarse level","",P,&P,NULL);CHKERRQ(ierr);
  }
  ierr  = PetscOptionsEnd();CHKERRQ(ierr);
  ierr  = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,P,N,M,1,PETSC_DETERMINE,PETSC_DETERMINE,sizeof(Node)/sizeof(PetscScalar),1,0,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr  = DMDASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
  ierr  = DMDASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
  *dm3d = da;
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  MPI_Comm       comm;
  DM             pack,da3,da2;
  TS             ts;
  THI            thi;
  Vec            X;
  Mat            B;
  PetscInt       i,steps;
  PetscReal      ftime;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = THICreate(comm,&thi);CHKERRQ(ierr);
  ierr = THICreateDM3d(thi,&da3);CHKERRQ(ierr);
  {
    PetscInt        Mx,My,mx,my,s;
    DMDAStencilType st;
    ierr = DMDAGetInfo(da3,0, 0,&My,&Mx, 0,&my,&mx, 0,&s,0,0,0,&st);CHKERRQ(ierr);
    ierr = DMDACreate2d(PetscObjectComm((PetscObject)thi),DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,st,My,Mx,my,mx,sizeof(PrmNode)/sizeof(PetscScalar),s,0,0,&da2);CHKERRQ(ierr);
    ierr = DMSetUp(da2);CHKERRQ(ierr);
  }

  ierr = PetscObjectSetName((PetscObject)da3,"3D_Velocity");CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da3,"f3d_");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da3,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da3,1,"v");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)da2,"2D_Fields");CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(da2,"f2d_");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,0,"b");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,1,"h");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,2,"beta2");CHKERRQ(ierr);
  ierr = DMCompositeCreate(comm,&pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,da3);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,da2);CHKERRQ(ierr);
  ierr = DMDestroy(&da3);CHKERRQ(ierr);
  ierr = DMDestroy(&da2);CHKERRQ(ierr);
  ierr = DMSetUp(pack);CHKERRQ(ierr);
  ierr = DMCreateMatrix(pack,&B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,"thi_");CHKERRQ(ierr);

  for (i=0; i<thi->nlevels; i++) {
    PetscReal Lx = thi->Lx / thi->units->meter,Ly = thi->Ly / thi->units->meter,Lz = thi->Lz / thi->units->meter;
    PetscInt  Mx,My,Mz;
    ierr = DMCompositeGetEntries(pack,&da3,&da2);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da3,0, &Mz,&My,&Mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Level %D domain size (m) %8.2g x %8.2g x %8.2g, num elements %3d x %3d x %3d (%8d), size (m) %g x %g x %g\n",i,Lx,Ly,Lz,Mx,My,Mz,Mx*My*Mz,Lx/Mx,Ly/My,1000./(Mz-1));CHKERRQ(ierr);
  }

  ierr = DMCreateGlobalVector(pack,&X);CHKERRQ(ierr);
  ierr = THIInitial(thi,pack,X);CHKERRQ(ierr);

  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,pack);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,THITSMonitor,thi,NULL);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,THIFunction,thi);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,B,B,THIJacobian,thi);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,10.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-3);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Steps %D  final time %g\n",steps,(double)ftime);CHKERRQ(ierr);

  if (0) {ierr = THISolveStatistics(thi,ts,0,"Full");CHKERRQ(ierr);}

  {
    PetscBool flg;
    char      filename[PETSC_MAX_PATH_LEN] = "";
    ierr = PetscOptionsGetString(NULL,NULL,"-o",filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = THIDAVecView_VTK_XML(thi,pack,X,filename,NULL);CHKERRQ(ierr);
    }
  }

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = DMDestroy(&pack);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = THIDestroy(&thi);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
