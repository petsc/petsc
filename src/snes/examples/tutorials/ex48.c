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

#if defined(PETSC_APPLE_FRAMEWORK)
#import <PETSc/petscsnes.h>
#import <PETSc/petsc/private/dmdaimpl.h>     /* There is not yet a public interface to manipulate dm->ops */
#else
#include <petscsnes.h>
#include <petsc/private/dmdaimpl.h>     /* There is not yet a public interface to manipulate dm->ops */
#endif
#include <ctype.h>              /* toupper() */

#if defined(__cplusplus)
/*  c++ cannot handle  [_restrict_] notation like C does */
#undef PETSC_RESTRICT
#define PETSC_RESTRICT
#endif

#if defined __SSE2__
#  include <emmintrin.h>
#endif

/* The SSE2 kernels are only for PetscScalar=double on architectures that support it */
#define USE_SSE2_KERNELS (!defined NO_SSE2                              \
                          && !defined PETSC_USE_COMPLEX                 \
                          && !defined PETSC_USE_REAL_SINGLE             \
                          && !defined PETSC_USE_REAL___FLOAT128         \
                          && defined __SSE2__)

static PetscClassId THI_CLASSID;

typedef enum {QUAD_GAUSS,QUAD_LOBATTO} QuadratureType;
static const char      *QuadratureTypes[] = {"gauss","lobatto","QuadratureType","QUAD_",0};
PETSC_UNUSED static const PetscReal HexQWeights[8]     = {1,1,1,1,1,1,1,1};
PETSC_UNUSED static const PetscReal HexQNodes[]        = {-0.57735026918962573, 0.57735026918962573};
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

static void HexComputeGeometry(PetscInt q,PetscReal hx,PetscReal hy,const PetscReal dz[PETSC_RESTRICT],PetscReal phi[PETSC_RESTRICT],PetscReal dphi[PETSC_RESTRICT][3],PetscReal *PETSC_RESTRICT jw)
{
  const PetscReal jac[3][3]  = {{hx/2,0,0}, {0,hy/2,0}, {dz[0],dz[1],dz[2]}};
  const PetscReal ijac[3][3] = {{1/jac[0][0],0,0}, {0,1/jac[1][1],0}, {-jac[2][0]/(jac[0][0]*jac[2][2]),-jac[2][1]/(jac[1][1]*jac[2][2]),1/jac[2][2]}};
  const PetscReal jdet       = jac[0][0]*jac[1][1]*jac[2][2];
  PetscInt        i;

  for (i=0; i<8; i++) {
    const PetscReal *dphir = HexQDeriv[q][i];
    phi[i]     = HexQInterp[q][i];
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

typedef struct {
  PetscReal min,max,cmin,cmax;
} PRange;

typedef enum {THIASSEMBLY_TRIDIAGONAL,THIASSEMBLY_FULL} THIAssemblyMode;

struct _p_THI {
  PETSCHEADER(int);
  void      (*initialize)(THI,PetscReal x,PetscReal y,PrmNode *p);
  PetscInt  zlevels;
  PetscReal Lx,Ly,Lz;           /* Model domain */
  PetscReal alpha;              /* Bed angle */
  Units     units;
  PetscReal dirichlet_scale;
  PetscReal ssa_friction_scale;
  PRange    eta;
  PRange    beta2;
  struct {
    PetscReal Bd2,eps,exponent;
  } viscosity;
  struct {
    PetscReal irefgam,eps2,exponent,refvel,epsvel;
  } friction;
  PetscReal rhog;
  PetscBool no_slip;
  PetscBool tridiagonal;
  PetscBool coarse2d;
  PetscBool verbose;
  MatType   mattype;
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

static PetscErrorCode THIJacobianLocal_3D_Full(DMDALocalInfo*,Node***,Mat,Mat,THI);
static PetscErrorCode THIJacobianLocal_3D_Tridiagonal(DMDALocalInfo*,Node***,Mat,Mat,THI);
static PetscErrorCode THIJacobianLocal_2D(DMDALocalInfo*,Node**,Mat,Mat,THI);

static void PrmHexGetZ(const PrmNode pn[],PetscInt k,PetscInt zm,PetscReal zn[])
{
  const PetscScalar zm1    = zm-1,
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

/* Tests A and C are from the ISMIP-HOM paper (Pattyn et al. 2008) */
static void THIInitialize_HOM_A(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x*PetscSinReal(thi->alpha);

  p->b     = s - 1000*units->meter + 500*units->meter * PetscSinReal(x*2*PETSC_PI/thi->Lx) * PetscSinReal(y*2*PETSC_PI/thi->Ly);
  p->h     = s - p->b;
  p->beta2 = 1e30;
}

static void THIInitialize_HOM_C(THI thi,PetscReal x,PetscReal y,PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x*PetscSinReal(thi->alpha);

  p->b = s - 1000*units->meter;
  p->h = s - p->b;
  /* tau_b = beta2 v   is a stress (Pa) */
  p->beta2 = 1000 * (1 + PetscSinReal(x*2*PETSC_PI/thi->Lx)*PetscSinReal(y*2*PETSC_PI/thi->Ly)) * units->Pascal * units->year / units->meter;
}

/* These are just toys */

/* Same bed as test A, free slip everywhere except for a discontinuous jump to a circular sticky region in the middle. */
static void THIInitialize_HOM_X(THI thi,PetscReal xx,PetscReal yy,PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x     = xx*2*PETSC_PI/thi->Lx - PETSC_PI,y = yy*2*PETSC_PI/thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r     = PetscSqrtReal(x*x + y*y),s = -x*PetscSinReal(thi->alpha);
  p->b     = s - 1000*units->meter + 500*units->meter*PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h     = s - p->b;
  p->beta2 = 1000 * (r < 1 ? 2 : 0) * units->Pascal * units->year / units->meter;
}

/* Like Z, but with 200 meter cliffs */
static void THIInitialize_HOM_Y(THI thi,PetscReal xx,PetscReal yy,PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x     = xx*2*PETSC_PI/thi->Lx - PETSC_PI,y = yy*2*PETSC_PI/thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r     = PetscSqrtReal(x*x + y*y),s = -x*PetscSinReal(thi->alpha);

  p->b = s - 1000*units->meter + 500*units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  if (PetscRealPart(p->b) > -700*units->meter) p->b += 200*units->meter;
  p->h     = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16*r))/PetscSqrtReal(1e-2 + 16*r)*PetscCosReal(x*3/2)*PetscCosReal(y*3/2)) * units->Pascal * units->year / units->meter;
}

/* Same bed as A, smoothly varying slipperiness, similar to MATLAB's "sombrero" (uncorrelated with bathymetry) */
static void THIInitialize_HOM_Z(THI thi,PetscReal xx,PetscReal yy,PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x     = xx*2*PETSC_PI/thi->Lx - PETSC_PI,y = yy*2*PETSC_PI/thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r     = PetscSqrtReal(x*x + y*y),s = -x*PetscSinReal(thi->alpha);

  p->b     = s - 1000*units->meter + 500*units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h     = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16*r))/PetscSqrtReal(1e-2 + 16*r)*PetscCosReal(x*3/2)*PetscCosReal(y*3/2)) * units->Pascal * units->year / units->meter;
}

static void THIFriction(THI thi,PetscReal rbeta2,PetscReal gam,PetscReal *beta2,PetscReal *dbeta2)
{
  if (thi->friction.irefgam == 0) {
    Units units = thi->units;
    thi->friction.irefgam = 1./(0.5*PetscSqr(thi->friction.refvel * units->meter / units->year));
    thi->friction.eps2    = 0.5*PetscSqr(thi->friction.epsvel * units->meter / units->year) * thi->friction.irefgam;
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
      n = 3.,                                           /* Glen exponent */
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

#undef __FUNCT__
#define __FUNCT__ "PRangeMinMax"
static PetscErrorCode PRangeMinMax(PRange *p,PetscReal min,PetscReal max)
{

  PetscFunctionBeginUser;
  p->cmin = min;
  p->cmax = max;
  if (min < p->min) p->min = min;
  if (max > p->max) p->max = max;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIDestroy"
static PetscErrorCode THIDestroy(THI *thi)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!*thi) PetscFunctionReturn(0);
  if (--((PetscObject)(*thi))->refct > 0) {*thi = 0; PetscFunctionReturn(0);}
  ierr = PetscFree((*thi)->units);CHKERRQ(ierr);
  ierr = PetscFree((*thi)->mattype);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(thi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THICreate"
static PetscErrorCode THICreate(MPI_Comm comm,THI *inthi)
{
  static PetscBool registered = PETSC_FALSE;
  THI              thi;
  Units            units;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  *inthi = 0;
  if (!registered) {
    ierr       = PetscClassIdRegister("Toy Hydrostatic Ice",&THI_CLASSID);CHKERRQ(ierr);
    registered = PETSC_TRUE;
  }
  ierr = PetscHeaderCreate(thi,THI_CLASSID,"THI","Toy Hydrostatic Ice","",comm,THIDestroy,0);CHKERRQ(ierr);

  ierr            = PetscNew(&thi->units);CHKERRQ(ierr);
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
  ierr          = PetscOptionsEnd();CHKERRQ(ierr);
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));
  units->year   = 31556926. * units->second, /* seconds per year */

  thi->Lx              = 10.e3;
  thi->Ly              = 10.e3;
  thi->Lz              = 1000;
  thi->dirichlet_scale = 1;
  thi->verbose         = PETSC_FALSE;

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
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"HOM experiment '%c' not implemented",homexp[0]);
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

    thi->friction.refvel = 100.;
    thi->friction.epsvel = 1.;

    ierr = PetscOptionsReal("-thi_friction_refvel","Reference velocity for sliding","",thi->friction.refvel,&thi->friction.refvel,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_friction_epsvel","Regularization velocity for sliding","",thi->friction.epsvel,&thi->friction.epsvel,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_friction_m","Friction exponent, 0=Coulomb, 1=Navier","",m,&m,NULL);CHKERRQ(ierr);

    thi->friction.exponent = (m-1)/2;

    ierr = PetscOptionsReal("-thi_dirichlet_scale","Scale Dirichlet boundary conditions by this factor","",thi->dirichlet_scale,&thi->dirichlet_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-thi_ssa_friction_scale","Scale slip boundary conditions by this factor in SSA (2D) assembly","",thi->ssa_friction_scale,&thi->ssa_friction_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-thi_coarse2d","Use a 2D coarse space corresponding to SSA","",thi->coarse2d,&thi->coarse2d,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-thi_tridiagonal","Assemble a tridiagonal system (column coupling only) on the finest level","",thi->tridiagonal,&thi->tridiagonal,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-thi_mat_type","Matrix type","MatSetType",MatList,mtype,(char*)mtype,sizeof(mtype),NULL);CHKERRQ(ierr);
    ierr = PetscStrallocpy(mtype,(char**)&thi->mattype);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-thi_verbose","Enable verbose output (like matrix sizes and statistics)","",thi->verbose,&thi->verbose,NULL);CHKERRQ(ierr);
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
              rho     = 910 * units->kilogram/PetscPowReal(units->meter,3),
              grav    = 9.81 * units->meter/PetscSqr(units->second),
              driving = rho * grav * PetscSinReal(thi->alpha) * 1000*units->meter;
    THIViscosity(thi,0.5*gradu*gradu,&eta,&deta);
    thi->rhog = rho * grav;
    if (thi->verbose) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Units: meter %8.2g  second %8.2g  kg %8.2g  Pa %8.2g\n",(double)units->meter,(double)units->second,(double)units->kilogram,(double)units->Pascal);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Domain (%6.2g,%6.2g,%6.2g), pressure %8.2g, driving stress %8.2g\n",(double)thi->Lx,(double)thi->Ly,(double)thi->Lz,(double)(rho*grav*1e3*units->meter),(double)driving);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Large velocity 1km/a %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",(double)u,(double)gradu,(double)eta,(double)(2*eta*gradu),(double)(2*eta*gradu/driving));CHKERRQ(ierr);
      THIViscosity(thi,0.5*PetscSqr(1e-3*gradu),&eta,&deta);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Small velocity 1m/a  %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n",(double)(1e-3*u),(double)(1e-3*gradu),(double)eta,(double)(2*eta*1e-3*gradu),(double)(2*eta*1e-3*gradu/driving));CHKERRQ(ierr);
    }
  }

  *inthi = thi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIInitializePrm"
static PetscErrorCode THIInitializePrm(THI thi,DM da2prm,Vec prm)
{
  PrmNode        **p;
  PetscInt       i,j,xs,xm,ys,ym,mx,my;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMDAGetGhostCorners(da2prm,&ys,&xs,0,&ym,&xm,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da2prm,0, &my,&mx,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2prm,prm,&p);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PetscReal xx = thi->Lx*i/mx,yy = thi->Ly*j/my;
      thi->initialize(thi,xx,yy,&p[i][j]);
    }
  }
  ierr = DMDAVecRestoreArray(da2prm,prm,&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THISetUpDM"
static PetscErrorCode THISetUpDM(THI thi,DM dm)
{
  PetscErrorCode  ierr;
  PetscInt        refinelevel,coarsenlevel,level,dim,Mx,My,Mz,mx,my,s;
  DMDAStencilType st;
  DM              da2prm;
  Vec             X;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(dm,&dim, &Mz,&My,&Mx, 0,&my,&mx, 0,&s,0,0,0,&st);CHKERRQ(ierr);
  if (dim == 2) {
    ierr = DMDAGetInfo(dm,&dim, &My,&Mx,0, &my,&mx,0, 0,&s,0,0,0,&st);CHKERRQ(ierr);
  }
  ierr  = DMGetRefineLevel(dm,&refinelevel);CHKERRQ(ierr);
  ierr  = DMGetCoarsenLevel(dm,&coarsenlevel);CHKERRQ(ierr);
  level = refinelevel - coarsenlevel;
  ierr  = DMDACreate2d(PetscObjectComm((PetscObject)thi),DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,st,My,Mx,my,mx,sizeof(PrmNode)/sizeof(PetscScalar),s,0,0,&da2prm);CHKERRQ(ierr);
  ierr  = DMCreateLocalVector(da2prm,&X);CHKERRQ(ierr);
  {
    PetscReal Lx = thi->Lx / thi->units->meter,Ly = thi->Ly / thi->units->meter,Lz = thi->Lz / thi->units->meter;
    if (dim == 2) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Level %D domain size (m) %8.2g x %8.2g, num elements %D x %D (%D), size (m) %g x %g\n",level,(double)Lx,(double)Ly,Mx,My,Mx*My,(double)(Lx/Mx),(double)(Ly/My));CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)thi),"Level %D domain size (m) %8.2g x %8.2g x %8.2g, num elements %D x %D x %D (%D), size (m) %g x %g x %g\n",level,(double)Lx,(double)Ly,(double)Lz,Mx,My,Mz,Mx*My*Mz,(double)(Lx/Mx),(double)(Ly/My),(double)(1000./(Mz-1)));CHKERRQ(ierr);
    }
  }
  ierr = THIInitializePrm(thi,da2prm,X);CHKERRQ(ierr);
  if (thi->tridiagonal) {       /* Reset coarse Jacobian evaluation */
    ierr = DMDASNESSetJacobianLocal(dm,(DMDASNESJacobian)THIJacobianLocal_3D_Full,thi);CHKERRQ(ierr);
  }
  if (thi->coarse2d) {
    ierr = DMDASNESSetJacobianLocal(dm,(DMDASNESJacobian)THIJacobianLocal_2D,thi);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)dm,"DMDA2Prm",(PetscObject)da2prm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dm,"DMDA2Prm_Vec",(PetscObject)X);CHKERRQ(ierr);
  ierr = DMDestroy(&da2prm);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_THI"
static PetscErrorCode DMCoarsenHook_THI(DM dmf,DM dmc,void *ctx)
{
  THI            thi = (THI)ctx;
  PetscErrorCode ierr;
  PetscInt       rlevel,clevel;

  PetscFunctionBeginUser;
  ierr = THISetUpDM(thi,dmc);CHKERRQ(ierr);
  ierr = DMGetRefineLevel(dmc,&rlevel);CHKERRQ(ierr);
  ierr = DMGetCoarsenLevel(dmc,&clevel);CHKERRQ(ierr);
  if (rlevel-clevel == 0) {ierr = DMSetMatType(dmc,MATAIJ);CHKERRQ(ierr);}
  ierr = DMCoarsenHookAdd(dmc,DMCoarsenHook_THI,NULL,thi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHook_THI"
static PetscErrorCode DMRefineHook_THI(DM dmc,DM dmf,void *ctx)
{
  THI            thi = (THI)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = THISetUpDM(thi,dmf);CHKERRQ(ierr);
  ierr = DMSetMatType(dmf,thi->mattype);CHKERRQ(ierr);
  ierr = DMRefineHookAdd(dmf,DMRefineHook_THI,NULL,thi);CHKERRQ(ierr);
  /* With grid sequencing, a formerly-refined DM will later be coarsened by PCSetUp_MG */
  ierr = DMCoarsenHookAdd(dmf,DMCoarsenHook_THI,NULL,thi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIDAGetPrm"
static PetscErrorCode THIDAGetPrm(DM da,PrmNode ***prm)
{
  PetscErrorCode ierr;
  DM             da2prm;
  Vec            X;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)da,"DMDA2Prm",(PetscObject*)&da2prm);CHKERRQ(ierr);
  if (!da2prm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No DMDA2Prm composed with given DMDA");
  ierr = PetscObjectQuery((PetscObject)da,"DMDA2Prm_Vec",(PetscObject*)&X);CHKERRQ(ierr);
  if (!X) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No DMDA2Prm_Vec composed with given DMDA");
  ierr = DMDAVecGetArray(da2prm,X,prm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIDARestorePrm"
static PetscErrorCode THIDARestorePrm(DM da,PrmNode ***prm)
{
  PetscErrorCode ierr;
  DM             da2prm;
  Vec            X;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)da,"DMDA2Prm",(PetscObject*)&da2prm);CHKERRQ(ierr);
  if (!da2prm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No DMDA2Prm composed with given DMDA");
  ierr = PetscObjectQuery((PetscObject)da,"DMDA2Prm_Vec",(PetscObject*)&X);CHKERRQ(ierr);
  if (!X) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No DMDA2Prm_Vec composed with given DMDA");
  ierr = DMDAVecRestoreArray(da2prm,X,prm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIInitial"
static PetscErrorCode THIInitial(SNES snes,Vec X,void *ctx)
{
  THI            thi;
  PetscInt       i,j,k,xs,xm,ys,ym,zs,zm,mx,my;
  PetscReal      hx,hy;
  PrmNode        **prm;
  Node           ***x;
  PetscErrorCode ierr;
  DM             da;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(da,&thi);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, 0,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = THIDAGetPrm(da,&prm);CHKERRQ(ierr);
  hx   = thi->Lx / mx;
  hy   = thi->Ly / my;
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
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = THIDARestorePrm(da,&prm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void PointwiseNonlinearity(THI thi,const Node n[PETSC_RESTRICT],const PetscReal phi[PETSC_RESTRICT],PetscReal dphi[PETSC_RESTRICT][3],PetscScalar *PETSC_RESTRICT u,PetscScalar *PETSC_RESTRICT v,PetscScalar du[PETSC_RESTRICT],PetscScalar dv[PETSC_RESTRICT],PetscReal *eta,PetscReal *deta)
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
  gam = PetscSqr(du[0]) + PetscSqr(dv[1]) + du[0]*dv[1] + 0.25*PetscSqr(du[1]+dv[0]) + 0.25*PetscSqr(du[2]) + 0.25*PetscSqr(dv[2]);
  THIViscosity(thi,PetscRealPart(gam),eta,deta);
}

static void PointwiseNonlinearity2D(THI thi,Node n[],PetscReal phi[],PetscReal dphi[4][2],PetscScalar *u,PetscScalar *v,PetscScalar du[],PetscScalar dv[],PetscReal *eta,PetscReal *deta)
{
  PetscInt    l,ll;
  PetscScalar gam;

  du[0] = du[1] = 0;
  dv[0] = dv[1] = 0;
  *u    = 0;
  *v    = 0;
  for (l=0; l<4; l++) {
    *u += phi[l] * n[l].u;
    *v += phi[l] * n[l].v;
    for (ll=0; ll<2; ll++) {
      du[ll] += dphi[l][ll] * n[l].u;
      dv[ll] += dphi[l][ll] * n[l].v;
    }
  }
  gam = PetscSqr(du[0]) + PetscSqr(dv[1]) + du[0]*dv[1] + 0.25*PetscSqr(du[1]+dv[0]);
  THIViscosity(thi,PetscRealPart(gam),eta,deta);
}

#undef __FUNCT__
#define __FUNCT__ "THIFunctionLocal"
static PetscErrorCode THIFunctionLocal(DMDALocalInfo *info,Node ***x,Node ***f,THI thi)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l;
  PetscReal      hx,hy,etamin,etamax,beta2min,beta2max;
  PrmNode        **prm;
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

  ierr = THIDAGetPrm(info->da,&prm);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm,i,j,pn);
      for (k=0; k<zm-1; k++) {
        PetscInt  ls = 0;
        Node      n[8],*fn[8];
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
          PetscReal   dz[3],phi[8],dphi[8][3],jw,eta,deta;
          PetscScalar du[3],dv[3],u,v;
          HexGrad(HexQDeriv[q],zn,dz);
          HexComputeGeometry(q,hx,hy,dz,phi,dphi,&jw);
          PointwiseNonlinearity(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
          jw /= thi->rhog;      /* scales residuals to be O(1) */
          if (q == 0) etabase = eta;
          RangeUpdate(&etamin,&etamax,eta);
          for (l=ls; l<8; l++) { /* test functions */
            const PetscReal ds[2] = {-PetscSinReal(thi->alpha),0};
            const PetscReal pp    = phi[l],*dp = dphi[l];
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
            for (q=0; q<4; q++) {
              const PetscReal jw = 0.25*hx*hy/thi->rhog,*phi = QuadQInterp[q];
              PetscScalar     u  =0,v=0,rbeta2=0;
              PetscReal       beta2,dbeta2;
              for (l=0; l<4; l++) {
                u      += phi[l]*n[l].u;
                v      += phi[l]*n[l].v;
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

  ierr = THIDARestorePrm(info->da,&prm);CHKERRQ(ierr);

  ierr = PRangeMinMax(&thi->eta,etamin,etamax);CHKERRQ(ierr);
  ierr = PRangeMinMax(&thi->beta2,beta2min,beta2max);CHKERRQ(ierr);
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

  PetscFunctionBeginUser;
  ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
  ierr = MatGetSize(B,&m,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)B),&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscScalar val0,val2;
    ierr = MatGetValue(B,0,0,&val0);CHKERRQ(ierr);
    ierr = MatGetValue(B,2,2,&val2);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Matrix dim %D norm %8.2e (0,0) %8.2e  (2,2) %8.2e %8.2e <= eta <= %8.2e %8.2e <= beta2 <= %8.2e\n",m,(double)nrm,(double)PetscRealPart(val0),(double)PetscRealPart(val2),(double)thi->eta.cmin,(double)thi->eta.cmax,(double)thi->beta2.cmin,(double)thi->beta2.cmax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THISurfaceStatistics"
static PetscErrorCode THISurfaceStatistics(DM da,Vec X,PetscReal *min,PetscReal *max,PetscReal *mean)
{
  PetscErrorCode ierr;
  Node           ***x;
  PetscInt       i,j,xs,ys,zs,xm,ym,zm,mx,my,mz;
  PetscReal      umin = 1e100,umax=-1e100;
  PetscScalar    usum = 0.0,gusum;

  PetscFunctionBeginUser;
  *min = *max = *mean = 0;
  ierr = DMDAGetInfo(da,0, &mz,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
  if (zs != 0 || zm != mz) SETERRQ(PETSC_COMM_SELF,1,"Unexpected decomposition");
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PetscReal u = PetscRealPart(x[i][j][zm-1].u);
      RangeUpdate(&umin,&umax,u);
      usum += u;
    }
  }
  ierr  = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr  = MPI_Allreduce(&umin,min,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  ierr  = MPI_Allreduce(&umax,max,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  ierr  = MPI_Allreduce(&usum,&gusum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  *mean = PetscRealPart(gusum) / (mx*my);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THISolveStatistics"
static PetscErrorCode THISolveStatistics(THI thi,SNES snes,PetscInt coarsened,const char name[])
{
  MPI_Comm       comm;
  Vec            X;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)thi,&comm);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&X);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Solution statistics after solve: %s\n",name);CHKERRQ(ierr);
  {
    PetscInt            its,lits;
    SNESConvergedReason reason;
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"%s: Number of SNES iterations = %D, total linear iterations = %D\n",SNESConvergedReasons[reason],its,lits);CHKERRQ(ierr);
  }
  {
    PetscReal         nrm2,tmin[3]={1e100,1e100,1e100},tmax[3]={-1e100,-1e100,-1e100},min[3],max[3];
    PetscInt          i,j,m;
    const PetscScalar *x;
    ierr = VecNorm(X,NORM_2,&nrm2);CHKERRQ(ierr);
    ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    for (i=0; i<m; i+=2) {
      PetscReal u = PetscRealPart(x[i]),v = PetscRealPart(x[i+1]),c = PetscSqrtReal(u*u+v*v);
      tmin[0] = PetscMin(u,tmin[0]);
      tmin[1] = PetscMin(v,tmin[1]);
      tmin[2] = PetscMin(c,tmin[2]);
      tmax[0] = PetscMax(u,tmax[0]);
      tmax[1] = PetscMax(v,tmax[1]);
      tmax[2] = PetscMax(c,tmax[2]);
    }
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = MPI_Allreduce(tmin,min,3,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)thi));CHKERRQ(ierr);
    ierr = MPI_Allreduce(tmax,max,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)thi));CHKERRQ(ierr);
    /* Dimensionalize to meters/year */
    nrm2 *= thi->units->year / thi->units->meter;
    for (j=0; j<3; j++) {
      min[j] *= thi->units->year / thi->units->meter;
      max[j] *= thi->units->year / thi->units->meter;
    }
    if (min[0] == 0.0) min[0] = 0.0;
    ierr = PetscPrintf(comm,"|X|_2 %g   %g <= u <=  %g   %g <= v <=  %g   %g <= c <=  %g \n",(double)nrm2,(double)min[0],(double)max[0],(double)min[1],(double)max[1],(double)min[2],(double)max[2]);CHKERRQ(ierr);
    {
      PetscReal umin,umax,umean;
      ierr   = THISurfaceStatistics(dm,X,&umin,&umax,&umean);CHKERRQ(ierr);
      umin  *= thi->units->year / thi->units->meter;
      umax  *= thi->units->year / thi->units->meter;
      umean *= thi->units->year / thi->units->meter;
      ierr   = PetscPrintf(comm,"Surface statistics: u in [%12.6e, %12.6e] mean %12.6e\n",(double)umin,(double)umax,(double)umean);CHKERRQ(ierr);
    }
    /* These values stay nondimensional */
    ierr = PetscPrintf(comm,"Global eta range   %g to %g converged range %g to %g\n",(double)thi->eta.min,(double)thi->eta.max,(double)thi->eta.cmin,(double)thi->eta.cmax);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Global beta2 range %g to %g converged range %g to %g\n",(double)thi->beta2.min,(double)thi->beta2.max,(double)thi->beta2.cmin,(double)thi->beta2.cmax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIJacobianLocal_2D"
static PetscErrorCode THIJacobianLocal_2D(DMDALocalInfo *info,Node **x,Mat J,Mat B,THI thi)
{
  PetscInt       xs,ys,xm,ym,i,j,q,l,ll;
  PetscReal      hx,hy;
  PrmNode        **prm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
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
      Node        n[4];
      PrmNode     pn[4];
      PetscScalar Ke[4*2][4*2];
      QuadExtract(prm,i,j,pn);
      QuadExtract(x,i,j,n);
      PetscMemzero(Ke,sizeof(Ke));
      for (q=0; q<4; q++) {
        PetscReal   phi[4],dphi[4][2],jw,eta,deta,beta2,dbeta2;
        PetscScalar u,v,du[2],dv[2],h = 0,rbeta2 = 0;
        for (l=0; l<4; l++) {
          phi[l]     = QuadQInterp[q][l];
          dphi[l][0] = QuadQDeriv[q][l][0]*2./hx;
          dphi[l][1] = QuadQDeriv[q][l][1]*2./hy;
          h         += phi[l] * pn[l].h;
          rbeta2    += phi[l] * pn[l].beta2;
        }
        jw = 0.25*hx*hy / thi->rhog; /* rhog is only scaling */
        PointwiseNonlinearity2D(thi,n,phi,dphi,&u,&v,du,dv,&eta,&deta);
        THIFriction(thi,PetscRealPart(rbeta2),PetscRealPart(u*u+v*v)/2,&beta2,&dbeta2);
        for (l=0; l<4; l++) {
          const PetscReal pp = phi[l],*dp = dphi[l];
          for (ll=0; ll<4; ll++) {
            const PetscReal ppl = phi[ll],*dpl = dphi[ll];
            PetscScalar     dgdu,dgdv;
            dgdu = 2.*du[0]*dpl[0] + dv[1]*dpl[0] + 0.5*(du[1]+dv[0])*dpl[1];
            dgdv = 2.*dv[1]*dpl[1] + du[0]*dpl[1] + 0.5*(du[1]+dv[0])*dpl[0];
            /* Picard part */
            Ke[l*2+0][ll*2+0] += dp[0]*jw*eta*4.*dpl[0] + dp[1]*jw*eta*dpl[1] + pp*jw*(beta2/h)*ppl*thi->ssa_friction_scale;
            Ke[l*2+0][ll*2+1] += dp[0]*jw*eta*2.*dpl[1] + dp[1]*jw*eta*dpl[0];
            Ke[l*2+1][ll*2+0] += dp[1]*jw*eta*2.*dpl[0] + dp[0]*jw*eta*dpl[1];
            Ke[l*2+1][ll*2+1] += dp[1]*jw*eta*4.*dpl[1] + dp[0]*jw*eta*dpl[0] + pp*jw*(beta2/h)*ppl*thi->ssa_friction_scale;
            /* extra Newton terms */
            Ke[l*2+0][ll*2+0] += dp[0]*jw*deta*dgdu*(4.*du[0]+2.*dv[1]) + dp[1]*jw*deta*dgdu*(du[1]+dv[0]) + pp*jw*(dbeta2/h)*u*u*ppl*thi->ssa_friction_scale;
            Ke[l*2+0][ll*2+1] += dp[0]*jw*deta*dgdv*(4.*du[0]+2.*dv[1]) + dp[1]*jw*deta*dgdv*(du[1]+dv[0]) + pp*jw*(dbeta2/h)*u*v*ppl*thi->ssa_friction_scale;
            Ke[l*2+1][ll*2+0] += dp[1]*jw*deta*dgdu*(4.*dv[1]+2.*du[0]) + dp[0]*jw*deta*dgdu*(du[1]+dv[0]) + pp*jw*(dbeta2/h)*v*u*ppl*thi->ssa_friction_scale;
            Ke[l*2+1][ll*2+1] += dp[1]*jw*deta*dgdv*(4.*dv[1]+2.*du[0]) + dp[0]*jw*deta*dgdv*(du[1]+dv[0]) + pp*jw*(dbeta2/h)*v*v*ppl*thi->ssa_friction_scale;
          }
        }
      }
      {
        const MatStencil rc[4] = {{0,i,j,0},{0,i+1,j,0},{0,i+1,j+1,0},{0,i,j+1,0}};
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
static PetscErrorCode THIJacobianLocal_3D(DMDALocalInfo *info,Node ***x,Mat B,THI thi,THIAssemblyMode amode)
{
  PetscInt       xs,ys,xm,ym,zm,i,j,k,q,l,ll;
  PetscReal      hx,hy;
  PrmNode        **prm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;
  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = THIDAGetPrm(info->da,&prm);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    for (j=ys; j<ys+ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm,i,j,pn);
      for (k=0; k<zm-1; k++) {
        Node        n[8];
        PetscReal   zn[8],etabase = 0;
        PetscScalar Ke[8*2][8*2];
        PetscInt    ls = 0;

        PrmHexGetZ(pn,k,zm,zn);
        HexExtract(x,i,j,k,n);
        PetscMemzero(Ke,sizeof(Ke));
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
            const PetscReal *PETSC_RESTRICT dp = dphi[l];
#if USE_SSE2_KERNELS
            /* gcc (up to my 4.5 snapshot) is really bad at hoisting intrinsics so we do it manually */
            __m128d
              p4         = _mm_set1_pd(4),p2 = _mm_set1_pd(2),p05 = _mm_set1_pd(0.5),
              p42        = _mm_setr_pd(4,2),p24 = _mm_shuffle_pd(p42,p42,_MM_SHUFFLE2(0,1)),
              du0        = _mm_set1_pd(du[0]),du1 = _mm_set1_pd(du[1]),du2 = _mm_set1_pd(du[2]),
              dv0        = _mm_set1_pd(dv[0]),dv1 = _mm_set1_pd(dv[1]),dv2 = _mm_set1_pd(dv[2]),
              jweta      = _mm_set1_pd(jw*eta),jwdeta = _mm_set1_pd(jw*deta),
              dp0        = _mm_set1_pd(dp[0]),dp1 = _mm_set1_pd(dp[1]),dp2 = _mm_set1_pd(dp[2]),
              dp0jweta   = _mm_mul_pd(dp0,jweta),dp1jweta = _mm_mul_pd(dp1,jweta),dp2jweta = _mm_mul_pd(dp2,jweta),
              p4du0p2dv1 = _mm_add_pd(_mm_mul_pd(p4,du0),_mm_mul_pd(p2,dv1)), /* 4 du0 + 2 dv1 */
              p4dv1p2du0 = _mm_add_pd(_mm_mul_pd(p4,dv1),_mm_mul_pd(p2,du0)), /* 4 dv1 + 2 du0 */
              pdu2dv2    = _mm_unpacklo_pd(du2,dv2),                          /* [du2, dv2] */
              du1pdv0    = _mm_add_pd(du1,dv0),                               /* du1 + dv0 */
              t1         = _mm_mul_pd(dp0,p4du0p2dv1),                        /* dp0 (4 du0 + 2 dv1) */
              t2         = _mm_mul_pd(dp1,p4dv1p2du0);                        /* dp1 (4 dv1 + 2 du0) */

#endif
#if defined COMPUTE_LOWER_TRIANGULAR  /* The element matrices are always symmetric so computing the lower-triangular part is not necessary */
            for (ll=ls; ll<8; ll++) { /* trial functions */
#else
            for (ll=l; ll<8; ll++) {
#endif
              const PetscReal *PETSC_RESTRICT dpl = dphi[ll];
              if (amode == THIASSEMBLY_TRIDIAGONAL && (l-ll)%4) continue; /* these entries would not be inserted */
#if !USE_SSE2_KERNELS
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
              * benefit.  On my hardware, these intrinsics are almost twice as fast as above, reducing total assembly cost
              * by 25 to 30 percent. */
              {
                __m128d
                  keu   = _mm_loadu_pd(&Ke[l*2+0][ll*2+0]),
                  kev   = _mm_loadu_pd(&Ke[l*2+1][ll*2+0]),
                  dpl01 = _mm_loadu_pd(&dpl[0]),dpl10 = _mm_shuffle_pd(dpl01,dpl01,_MM_SHUFFLE2(0,1)),dpl2 = _mm_set_sd(dpl[2]),
                  t0,t3,pdgduv;
                keu = _mm_add_pd(keu,_mm_add_pd(_mm_mul_pd(_mm_mul_pd(dp0jweta,p42),dpl01),
                                                _mm_add_pd(_mm_mul_pd(dp1jweta,dpl10),
                                                           _mm_mul_pd(dp2jweta,dpl2))));
                kev = _mm_add_pd(kev,_mm_add_pd(_mm_mul_pd(_mm_mul_pd(dp1jweta,p24),dpl01),
                                                _mm_add_pd(_mm_mul_pd(dp0jweta,dpl10),
                                                           _mm_mul_pd(dp2jweta,_mm_shuffle_pd(dpl2,dpl2,_MM_SHUFFLE2(0,1))))));
                pdgduv = _mm_mul_pd(p05,_mm_add_pd(_mm_add_pd(_mm_mul_pd(p42,_mm_mul_pd(du0,dpl01)),
                                                              _mm_mul_pd(p24,_mm_mul_pd(dv1,dpl01))),
                                                   _mm_add_pd(_mm_mul_pd(du1pdv0,dpl10),
                                                              _mm_mul_pd(pdu2dv2,_mm_set1_pd(dpl[2]))))); /* [dgdu, dgdv] */
                t0 = _mm_mul_pd(jwdeta,pdgduv);  /* jw deta [dgdu, dgdv] */
                t3 = _mm_mul_pd(t0,du1pdv0);     /* t0 (du1 + dv0) */
                _mm_storeu_pd(&Ke[l*2+0][ll*2+0],_mm_add_pd(keu,_mm_add_pd(_mm_mul_pd(t1,t0),
                                                                           _mm_add_pd(_mm_mul_pd(dp1,t3),
                                                                                      _mm_mul_pd(t0,_mm_mul_pd(dp2,du2))))));
                _mm_storeu_pd(&Ke[l*2+1][ll*2+0],_mm_add_pd(kev,_mm_add_pd(_mm_mul_pd(t2,t0),
                                                                           _mm_add_pd(_mm_mul_pd(dp0,t3),
                                                                                      _mm_mul_pd(t0,_mm_mul_pd(dp2,dv2))))));
              }
#endif
            }
          }
        }
        if (k == 0) { /* on a bottom face */
          if (thi->no_slip) {
            const PetscReal   hz    = PetscRealPart(pn[0].h)/(zm-1);
            const PetscScalar diagu = 2*etabase/thi->rhog*(hx*hy/hz + hx*hz/hy + 4*hy*hz/hx),diagv = 2*etabase/thi->rhog*(hx*hy/hz + 4*hx*hz/hy + hy*hz/hx);
            Ke[0][0] = thi->dirichlet_scale*diagu;
            Ke[1][1] = thi->dirichlet_scale*diagv;
          } else {
            for (q=0; q<4; q++) {
              const PetscReal jw = 0.25*hx*hy/thi->rhog,*phi = QuadQInterp[q];
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
          const MatStencil rc[8] = {{i,j,k,0},{i+1,j,k,0},{i+1,j+1,k,0},{i,j+1,k,0},{i,j,k+1,0},{i+1,j,k+1,0},{i+1,j+1,k+1,0},{i,j+1,k+1,0}};
          if (amode == THIASSEMBLY_TRIDIAGONAL) {
            for (l=0; l<4; l++) { /* Copy out each of the blocks, discarding horizontal coupling */
              const PetscInt   l4     = l+4;
              const MatStencil rcl[2] = {{rc[l].k,rc[l].j,rc[l].i,0},{rc[l4].k,rc[l4].j,rc[l4].i,0}};
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
static PetscErrorCode THIJacobianLocal_3D_Full(DMDALocalInfo *info,Node ***x,Mat A,Mat B,THI thi)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr  = THIJacobianLocal_3D(info,x,B,thi,THIASSEMBLY_FULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIJacobianLocal_3D_Tridiagonal"
static PetscErrorCode THIJacobianLocal_3D_Tridiagonal(DMDALocalInfo *info,Node ***x,Mat A,Mat B,THI thi)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = THIJacobianLocal_3D(info,x,B,thi,THIASSEMBLY_TRIDIAGONAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHierarchy_THI"
static PetscErrorCode DMRefineHierarchy_THI(DM dac0,PetscInt nlevels,DM hierarchy[])
{
  PetscErrorCode  ierr;
  THI             thi;
  PetscInt        dim,M,N,m,n,s,dof;
  DM              dac,daf;
  DMDAStencilType st;
  DM_DA           *ddf,*ddc;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)dac0,"THI",(PetscObject*)&thi);CHKERRQ(ierr);
  if (!thi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot refine this DMDA, missing composed THI instance");
  if (nlevels > 1) {
    ierr = DMRefineHierarchy(dac0,nlevels-1,hierarchy);CHKERRQ(ierr);
    dac  = hierarchy[nlevels-2];
  } else {
    dac = dac0;
  }
  ierr = DMDAGetInfo(dac,&dim, &N,&M,0, &n,&m,0, &dof,&s,0,0,0,&st);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"This function can only refine 2D DMDAs");

  /* Creates a 3D DMDA with the same map-plane layout as the 2D one, with contiguous columns */
  ierr = DMDACreate3d(PetscObjectComm((PetscObject)dac),DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,st,thi->zlevels,N,M,1,n,m,dof,s,NULL,NULL,NULL,&daf);CHKERRQ(ierr);

  daf->ops->creatematrix        = dac->ops->creatematrix;
  daf->ops->createinterpolation = dac->ops->createinterpolation;
  daf->ops->getcoloring         = dac->ops->getcoloring;
  ddf                           = (DM_DA*)daf->data;
  ddc                           = (DM_DA*)dac->data;
  ddf->interptype               = ddc->interptype;

  ierr = DMDASetFieldName(daf,0,"x-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(daf,1,"y-velocity");CHKERRQ(ierr);

  hierarchy[nlevels-1] = daf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateInterpolation_DA_THI"
static PetscErrorCode DMCreateInterpolation_DA_THI(DM dac,DM daf,Mat *A,Vec *scale)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(A,3);
  if (scale) PetscValidPointer(scale,4);
  ierr = DMDAGetInfo(daf,&dim,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim  == 2) {
    /* We are in the 2D problem and use normal DMDA interpolation */
    ierr = DMCreateInterpolation(dac,daf,A,scale);CHKERRQ(ierr);
  } else {
    PetscInt i,j,k,xs,ys,zs,xm,ym,zm,mx,my,mz,rstart,cstart;
    Mat      B;

    ierr = DMDAGetInfo(daf,0, &mz,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(daf,&zs,&ys,&xs,&zm,&ym,&xm);CHKERRQ(ierr);
    if (zs != 0) SETERRQ(PETSC_COMM_SELF,1,"unexpected");
    ierr = MatCreate(PetscObjectComm((PetscObject)daf),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,xm*ym*zm,xm*ym,mx*my*mz,mx*my);CHKERRQ(ierr);

    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(B,1,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(B,1,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(B,&rstart,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(B,&cstart,NULL);CHKERRQ(ierr);
    for (i=xs; i<xs+xm; i++) {
      for (j=ys; j<ys+ym; j++) {
        for (k=zs; k<zs+zm; k++) {
          PetscInt    i2  = i*ym+j,i3 = i2*zm+k;
          PetscScalar val = ((k == 0 || k == mz-1) ? 0.5 : 1.) / (mz-1.); /* Integration using trapezoid rule */
          ierr = MatSetValue(B,cstart+i3,rstart+i2,val,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreateMAIJ(B,sizeof(Node)/sizeof(PetscScalar),A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_THI_Tridiagonal"
static PetscErrorCode DMCreateMatrix_THI_Tridiagonal(DM da,Mat *J)
{
  PetscErrorCode         ierr;
  Mat                    A;
  PetscInt               xm,ym,zm,dim,dof = 2,starts[3],dims[3];
  ISLocalToGlobalMapping ltog;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,&dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim != 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected DMDA to be 3D");
  ierr = DMDAGetCorners(da,0,0,0,&zm,&ym,&xm);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)da),&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,dof*xm*ym*zm,dof*xm*ym*zm,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,da->mattype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,3*2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3*2,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,2,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,2,3,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,2,2,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,2,2,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,ltog,ltog);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);CHKERRQ(ierr);
  ierr = MatSetStencil(A,dim,dims,starts,dof);CHKERRQ(ierr);
  *J   = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "THIDAVecView_VTK_XML"
static PetscErrorCode THIDAVecView_VTK_XML(THI thi,DM da,Vec X,const char filename[])
{
  const PetscInt    dof   = 2;
  Units             units = thi->units;
  MPI_Comm          comm;
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscMPIInt       rank,size,tag,nn,nmax;
  PetscInt          mx,my,mz,r,range[6];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)thi,&comm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &mz,&my,&mx, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(comm,filename,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  <StructuredGrid WholeExtent=\"%d %D %d %D %d %D\">\n",0,mz-1,0,my-1,0,mx-1);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,range,range+1,range+2,range+3,range+4,range+5);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(range[3]*range[4]*range[5]*dof,&nn);CHKERRQ(ierr);
  ierr = MPI_Reduce(&nn,&nmax,1,MPI_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
  tag  = ((PetscObject) viewer)->tag;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  if (!rank) {
    PetscScalar *array;
    ierr = PetscMalloc1(nmax,&array);CHKERRQ(ierr);
    for (r=0; r<size; r++) {
      PetscInt          i,j,k,xs,xm,ys,ym,zs,zm;
      const PetscScalar *ptr;
      MPI_Status        status;
      if (r) {
        ierr = MPI_Recv(range,6,MPIU_INT,r,tag,comm,MPI_STATUS_IGNORE);CHKERRQ(ierr);
      }
      zs = range[0];ys = range[1];xs = range[2];zm = range[3];ym = range[4];xm = range[5];
      if (xm*ym*zm*dof > nmax) SETERRQ(PETSC_COMM_SELF,1,"should not happen");
      if (r) {
        ierr = MPI_Recv(array,nmax,MPIU_SCALAR,r,tag,comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&nn);CHKERRQ(ierr);
        if (nn != xm*ym*zm*dof) SETERRQ(PETSC_COMM_SELF,1,"should not happen");
        ptr = array;
      } else ptr = x;
      ierr = PetscViewerASCIIPrintf(viewer,"    <Piece Extent=\"%D %D %D %D %D %D\">\n",zs,zs+zm-1,ys,ys+ym-1,xs,xs+xm-1);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"      <Points>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");CHKERRQ(ierr);
      for (i=xs; i<xs+xm; i++) {
        for (j=ys; j<ys+ym; j++) {
          for (k=zs; k<zs+zm; k++) {
            PrmNode   p;
            PetscReal xx = thi->Lx*i/mx,yy = thi->Ly*j/my,zz;
            thi->initialize(thi,xx,yy,&p);
            zz   = PetscRealPart(p.b) + PetscRealPart(p.h)*k/(mz-1);
            ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",(double)xx,(double)yy,(double)zz);CHKERRQ(ierr);
          }
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"        </DataArray>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"      </Points>\n");CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"      <PointData>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");CHKERRQ(ierr);
      for (i=0; i<nn; i+=dof) {
        ierr = PetscViewerASCIIPrintf(viewer,"%f %f %f\n",(double)(PetscRealPart(ptr[i])*units->year/units->meter),(double)(PetscRealPart(ptr[i+1])*units->year/units->meter),0.0);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"        </DataArray>\n");CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"        <DataArray type=\"Int32\" Name=\"rank\" NumberOfComponents=\"1\" format=\"ascii\">\n");CHKERRQ(ierr);
      for (i=0; i<nn; i+=dof) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D\n",r);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"        </DataArray>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"      </PointData>\n");CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"    </Piece>\n");CHKERRQ(ierr);
    }
    ierr = PetscFree(array);CHKERRQ(ierr);
  } else {
    ierr = MPI_Send(range,6,MPIU_INT,0,tag,comm);CHKERRQ(ierr);
    ierr = MPI_Send((PetscScalar*)x,nn,MPIU_SCALAR,0,tag,comm);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  </StructuredGrid>\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"</VTKFile>\n");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  MPI_Comm       comm;
  THI            thi;
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = THICreate(comm,&thi);CHKERRQ(ierr);
  {
    PetscInt M = 3,N = 3,P = 2;
    ierr = PetscOptionsBegin(comm,NULL,"Grid resolution options","");CHKERRQ(ierr);
    {
      ierr = PetscOptionsInt("-M","Number of elements in x-direction on coarse level","",M,&M,NULL);CHKERRQ(ierr);
      N    = M;
      ierr = PetscOptionsInt("-N","Number of elements in y-direction on coarse level (if different from M)","",N,&N,NULL);CHKERRQ(ierr);
      if (thi->coarse2d) {
        ierr = PetscOptionsInt("-zlevels","Number of elements in z-direction on fine level","",thi->zlevels,&thi->zlevels,NULL);CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsInt("-P","Number of elements in z-direction on coarse level","",P,&P,NULL);CHKERRQ(ierr);
      }
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (thi->coarse2d) {
      ierr = DMDACreate2d(comm,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,-N,-M,PETSC_DETERMINE,PETSC_DETERMINE,sizeof(Node)/sizeof(PetscScalar),1,0,0,&da);CHKERRQ(ierr);

      da->ops->refinehierarchy     = DMRefineHierarchy_THI;
      da->ops->createinterpolation = DMCreateInterpolation_DA_THI;

      ierr = PetscObjectCompose((PetscObject)da,"THI",(PetscObject)thi);CHKERRQ(ierr);
    } else {
      ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX,-P,-N,-M,1,PETSC_DETERMINE,PETSC_DETERMINE,sizeof(Node)/sizeof(PetscScalar),1,0,0,0,&da);CHKERRQ(ierr);
    }
    ierr = DMDASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
  }
  ierr = THISetUpDM(thi,da);CHKERRQ(ierr);
  if (thi->tridiagonal) da->ops->creatematrix = DMCreateMatrix_THI_Tridiagonal;

  {                             /* Set the fine level matrix type if -da_refine */
    PetscInt rlevel,clevel;
    ierr = DMGetRefineLevel(da,&rlevel);CHKERRQ(ierr);
    ierr = DMGetCoarsenLevel(da,&clevel);CHKERRQ(ierr);
    if (rlevel - clevel > 0) {ierr = DMSetMatType(da,thi->mattype);CHKERRQ(ierr);}
  }

  ierr = DMDASNESSetFunctionLocal(da,ADD_VALUES,(DMDASNESFunction)THIFunctionLocal,thi);CHKERRQ(ierr);
  if (thi->tridiagonal) {
    ierr = DMDASNESSetJacobianLocal(da,(DMDASNESJacobian)THIJacobianLocal_3D_Tridiagonal,thi);CHKERRQ(ierr);
  } else {
    ierr = DMDASNESSetJacobianLocal(da,(DMDASNESJacobian)THIJacobianLocal_3D_Full,thi);CHKERRQ(ierr);
  }
  ierr = DMCoarsenHookAdd(da,DMCoarsenHook_THI,NULL,thi);CHKERRQ(ierr);
  ierr = DMRefineHookAdd(da,DMRefineHook_THI,NULL,thi);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(da,thi);CHKERRQ(ierr);

  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESSetComputeInitialGuess(snes,THIInitial,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,NULL);CHKERRQ(ierr);

  ierr = THISolveStatistics(thi,snes,0,"Full");CHKERRQ(ierr);

  {
    PetscBool flg;
    char      filename[PETSC_MAX_PATH_LEN] = "";
    ierr = PetscOptionsGetString(NULL,NULL,"-o",filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (flg) {
      Vec X;
      DM  dm;
      ierr = SNESGetSolution(snes,&X);CHKERRQ(ierr);
      ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
      ierr = THIDAVecView_VTK_XML(thi,dm,X,filename);CHKERRQ(ierr);
    }
  }

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = THIDestroy(&thi);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
