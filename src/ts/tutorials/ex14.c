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
#include <ctype.h> /* toupper() */
#include <petsc/private/petscimpl.h>

#if defined __SSE2__
  #include <emmintrin.h>
#endif

/* The SSE2 kernels are only for PetscScalar=double on architectures that support it */
#define USE_SSE2_KERNELS (!defined NO_SSE2 && !defined PETSC_USE_COMPLEX && !defined PETSC_USE_REAL_SINGLE && defined __SSE2__)

#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 199901L
  #if defined __cplusplus /* C++ restrict is nonstandard and compilers have inconsistent rules about where it can be used */
    #define restrict
  #else
    #define restrict PETSC_RESTRICT
  #endif
#endif

static PetscClassId THI_CLASSID;

typedef enum {
  QUAD_GAUSS,
  QUAD_LOBATTO
} QuadratureType;
static const char     *QuadratureTypes[] = {"gauss", "lobatto", "QuadratureType", "QUAD_", 0};
static const PetscReal HexQWeights[8]    = {1, 1, 1, 1, 1, 1, 1, 1};
static const PetscReal HexQNodes[]       = {-0.57735026918962573, 0.57735026918962573};
#define G 0.57735026918962573
#define H (0.5 * (1. + G))
#define L (0.5 * (1. - G))
#define M (-0.5)
#define P (0.5)
/* Special quadrature: Lobatto in horizontal, Gauss in vertical */
static const PetscReal HexQInterp_Lobatto[8][8] = {
  {H, 0, 0, 0, L, 0, 0, 0},
  {0, H, 0, 0, 0, L, 0, 0},
  {0, 0, H, 0, 0, 0, L, 0},
  {0, 0, 0, H, 0, 0, 0, L},
  {L, 0, 0, 0, H, 0, 0, 0},
  {0, L, 0, 0, 0, H, 0, 0},
  {0, 0, L, 0, 0, 0, H, 0},
  {0, 0, 0, L, 0, 0, 0, H}
};
static const PetscReal HexQDeriv_Lobatto[8][8][3] = {
  {{M * H, M *H, M}, {P * H, 0, 0},    {0, 0, 0},        {0, P *H, 0},     {M * L, M *L, P}, {P * L, 0, 0},    {0, 0, 0},        {0, P *L, 0}    },
  {{M * H, 0, 0},    {P * H, M *H, M}, {0, P *H, 0},     {0, 0, 0},        {M * L, 0, 0},    {P * L, M *L, P}, {0, P *L, 0},     {0, 0, 0}       },
  {{0, 0, 0},        {0, M *H, 0},     {P * H, P *H, M}, {M * H, 0, 0},    {0, 0, 0},        {0, M *L, 0},     {P * L, P *L, P}, {M * L, 0, 0}   },
  {{0, M *H, 0},     {0, 0, 0},        {P * H, 0, 0},    {M * H, P *H, M}, {0, M *L, 0},     {0, 0, 0},        {P * L, 0, 0},    {M * L, P *L, P}},
  {{M * L, M *L, M}, {P * L, 0, 0},    {0, 0, 0},        {0, P *L, 0},     {M * H, M *H, P}, {P * H, 0, 0},    {0, 0, 0},        {0, P *H, 0}    },
  {{M * L, 0, 0},    {P * L, M *L, M}, {0, P *L, 0},     {0, 0, 0},        {M * H, 0, 0},    {P * H, M *H, P}, {0, P *H, 0},     {0, 0, 0}       },
  {{0, 0, 0},        {0, M *L, 0},     {P * L, P *L, M}, {M * L, 0, 0},    {0, 0, 0},        {0, M *H, 0},     {P * H, P *H, P}, {M * H, 0, 0}   },
  {{0, M *L, 0},     {0, 0, 0},        {P * L, 0, 0},    {M * L, P *L, M}, {0, M *H, 0},     {0, 0, 0},        {P * H, 0, 0},    {M * H, P *H, P}}
};
/* Stanndard Gauss */
static const PetscReal HexQInterp_Gauss[8][8] = {
  {H * H * H, L *H *H, L *L *H, H *L *H, H *H *L, L *H *L, L *L *L, H *L *L},
  {L * H * H, H *H *H, H *L *H, L *L *H, L *H *L, H *H *L, H *L *L, L *L *L},
  {L * L * H, H *L *H, H *H *H, L *H *H, L *L *L, H *L *L, H *H *L, L *H *L},
  {H * L * H, L *L *H, L *H *H, H *H *H, H *L *L, L *L *L, L *H *L, H *H *L},
  {H * H * L, L *H *L, L *L *L, H *L *L, H *H *H, L *H *H, L *L *H, H *L *H},
  {L * H * L, H *H *L, H *L *L, L *L *L, L *H *H, H *H *H, H *L *H, L *L *H},
  {L * L * L, H *L *L, H *H *L, L *H *L, L *L *H, H *L *H, H *H *H, L *H *H},
  {H * L * L, L *L *L, L *H *L, H *H *L, H *L *H, L *L *H, L *H *H, H *H *H}
};
static const PetscReal HexQDeriv_Gauss[8][8][3] = {
  {{M * H * H, H *M *H, H *H *M}, {P * H * H, L *M *H, L *H *M}, {P * L * H, L *P *H, L *L *M}, {M * L * H, H *P *H, H *L *M}, {M * H * L, H *M *L, H *H *P}, {P * H * L, L *M *L, L *H *P}, {P * L * L, L *P *L, L *L *P}, {M * L * L, H *P *L, H *L *P}},
  {{M * H * H, L *M *H, L *H *M}, {P * H * H, H *M *H, H *H *M}, {P * L * H, H *P *H, H *L *M}, {M * L * H, L *P *H, L *L *M}, {M * H * L, L *M *L, L *H *P}, {P * H * L, H *M *L, H *H *P}, {P * L * L, H *P *L, H *L *P}, {M * L * L, L *P *L, L *L *P}},
  {{M * L * H, L *M *H, L *L *M}, {P * L * H, H *M *H, H *L *M}, {P * H * H, H *P *H, H *H *M}, {M * H * H, L *P *H, L *H *M}, {M * L * L, L *M *L, L *L *P}, {P * L * L, H *M *L, H *L *P}, {P * H * L, H *P *L, H *H *P}, {M * H * L, L *P *L, L *H *P}},
  {{M * L * H, H *M *H, H *L *M}, {P * L * H, L *M *H, L *L *M}, {P * H * H, L *P *H, L *H *M}, {M * H * H, H *P *H, H *H *M}, {M * L * L, H *M *L, H *L *P}, {P * L * L, L *M *L, L *L *P}, {P * H * L, L *P *L, L *H *P}, {M * H * L, H *P *L, H *H *P}},
  {{M * H * L, H *M *L, H *H *M}, {P * H * L, L *M *L, L *H *M}, {P * L * L, L *P *L, L *L *M}, {M * L * L, H *P *L, H *L *M}, {M * H * H, H *M *H, H *H *P}, {P * H * H, L *M *H, L *H *P}, {P * L * H, L *P *H, L *L *P}, {M * L * H, H *P *H, H *L *P}},
  {{M * H * L, L *M *L, L *H *M}, {P * H * L, H *M *L, H *H *M}, {P * L * L, H *P *L, H *L *M}, {M * L * L, L *P *L, L *L *M}, {M * H * H, L *M *H, L *H *P}, {P * H * H, H *M *H, H *H *P}, {P * L * H, H *P *H, H *L *P}, {M * L * H, L *P *H, L *L *P}},
  {{M * L * L, L *M *L, L *L *M}, {P * L * L, H *M *L, H *L *M}, {P * H * L, H *P *L, H *H *M}, {M * H * L, L *P *L, L *H *M}, {M * L * H, L *M *H, L *L *P}, {P * L * H, H *M *H, H *L *P}, {P * H * H, H *P *H, H *H *P}, {M * H * H, L *P *H, L *H *P}},
  {{M * L * L, H *M *L, H *L *M}, {P * L * L, L *M *L, L *L *M}, {P * H * L, L *P *L, L *H *M}, {M * H * L, H *P *L, H *H *M}, {M * L * H, H *M *H, H *L *P}, {P * L * H, L *M *H, L *L *P}, {P * H * H, L *P *H, L *H *P}, {M * H * H, H *P *H, H *H *P}}
};
static const PetscReal (*HexQInterp)[8], (*HexQDeriv)[8][3];
/* Standard 2x2 Gauss quadrature for the bottom layer. */
static const PetscReal QuadQInterp[4][4] = {
  {H * H, L *H, L *L, H *L},
  {L * H, H *H, H *L, L *L},
  {L * L, H *L, H *H, L *H},
  {H * L, L *L, L *H, H *H}
};
static const PetscReal QuadQDeriv[4][4][2] = {
  {{M * H, M *H}, {P * H, M *L}, {P * L, P *L}, {M * L, P *H}},
  {{M * H, M *L}, {P * H, M *H}, {P * L, P *H}, {M * L, P *L}},
  {{M * L, M *L}, {P * L, M *H}, {P * H, P *H}, {M * H, P *L}},
  {{M * L, M *H}, {P * L, M *L}, {P * H, P *L}, {M * H, P *H}}
};
#undef G
#undef H
#undef L
#undef M
#undef P

#define HexExtract(x, i, j, k, n) \
  do { \
    (n)[0] = (x)[i][j][k]; \
    (n)[1] = (x)[i + 1][j][k]; \
    (n)[2] = (x)[i + 1][j + 1][k]; \
    (n)[3] = (x)[i][j + 1][k]; \
    (n)[4] = (x)[i][j][k + 1]; \
    (n)[5] = (x)[i + 1][j][k + 1]; \
    (n)[6] = (x)[i + 1][j + 1][k + 1]; \
    (n)[7] = (x)[i][j + 1][k + 1]; \
  } while (0)

#define HexExtractRef(x, i, j, k, n) \
  do { \
    (n)[0] = &(x)[i][j][k]; \
    (n)[1] = &(x)[i + 1][j][k]; \
    (n)[2] = &(x)[i + 1][j + 1][k]; \
    (n)[3] = &(x)[i][j + 1][k]; \
    (n)[4] = &(x)[i][j][k + 1]; \
    (n)[5] = &(x)[i + 1][j][k + 1]; \
    (n)[6] = &(x)[i + 1][j + 1][k + 1]; \
    (n)[7] = &(x)[i][j + 1][k + 1]; \
  } while (0)

#define QuadExtract(x, i, j, n) \
  do { \
    (n)[0] = (x)[i][j]; \
    (n)[1] = (x)[i + 1][j]; \
    (n)[2] = (x)[i + 1][j + 1]; \
    (n)[3] = (x)[i][j + 1]; \
  } while (0)

static PetscScalar Sqr(PetscScalar a)
{
  return a * a;
}

static void HexGrad(const PetscReal dphi[][3], const PetscReal zn[], PetscReal dz[])
{
  PetscInt i;
  dz[0] = dz[1] = dz[2] = 0;
  for (i = 0; i < 8; i++) {
    dz[0] += dphi[i][0] * zn[i];
    dz[1] += dphi[i][1] * zn[i];
    dz[2] += dphi[i][2] * zn[i];
  }
}

static void HexComputeGeometry(PetscInt q, PetscReal hx, PetscReal hy, const PetscReal dz[restrict], PetscReal phi[restrict], PetscReal dphi[restrict][3], PetscReal *restrict jw)
{
  const PetscReal jac[3][3] =
    {
      {hx / 2, 0,      0    },
      {0,      hy / 2, 0    },
      {dz[0],  dz[1],  dz[2]}
  },
                  ijac[3][3] = {{1 / jac[0][0], 0, 0}, {0, 1 / jac[1][1], 0}, {-jac[2][0] / (jac[0][0] * jac[2][2]), -jac[2][1] / (jac[1][1] * jac[2][2]), 1 / jac[2][2]}}, jdet = jac[0][0] * jac[1][1] * jac[2][2];
  PetscInt i;

  for (i = 0; i < 8; i++) {
    const PetscReal *dphir = HexQDeriv[q][i];
    phi[i]                 = HexQInterp[q][i];
    dphi[i][0]             = dphir[0] * ijac[0][0] + dphir[1] * ijac[1][0] + dphir[2] * ijac[2][0];
    dphi[i][1]             = dphir[0] * ijac[0][1] + dphir[1] * ijac[1][1] + dphir[2] * ijac[2][1];
    dphi[i][2]             = dphir[0] * ijac[0][2] + dphir[1] * ijac[1][2] + dphir[2] * ijac[2][2];
  }
  *jw = 1.0 * jdet;
}

typedef struct _p_THI   *THI;
typedef struct _n_Units *Units;

typedef struct {
  PetscScalar u, v;
} Node;

typedef struct {
  PetscScalar b;     /* bed */
  PetscScalar h;     /* thickness */
  PetscScalar beta2; /* friction */
} PrmNode;

#define FieldSize(ntype)             ((PetscInt)(sizeof(ntype) / sizeof(PetscScalar)))
#define FieldOffset(ntype, member)   ((PetscInt)(offsetof(ntype, member) / sizeof(PetscScalar)))
#define FieldIndex(ntype, i, member) ((PetscInt)((i)*FieldSize(ntype) + FieldOffset(ntype, member)))
#define NODE_SIZE                    FieldSize(Node)
#define PRMNODE_SIZE                 FieldSize(PrmNode)

typedef struct {
  PetscReal min, max, cmin, cmax;
} PRange;

struct _p_THI {
  PETSCHEADER(int);
  void (*initialize)(THI, PetscReal x, PetscReal y, PrmNode *p);
  PetscInt  nlevels;
  PetscInt  zlevels;
  PetscReal Lx, Ly, Lz; /* Model domain */
  PetscReal alpha;      /* Bed angle */
  Units     units;
  PetscReal dirichlet_scale;
  PetscReal ssa_friction_scale;
  PetscReal inertia;
  PRange    eta;
  PRange    beta2;
  struct {
    PetscReal Bd2, eps, exponent, glen_n;
  } viscosity;
  struct {
    PetscReal irefgam, eps2, exponent;
  } friction;
  struct {
    PetscReal rate, exponent, refvel;
  } erosion;
  PetscReal rhog;
  PetscBool no_slip;
  PetscBool verbose;
  char     *mattype;
  char     *monitor_basename;
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

static void PrmHexGetZ(const PrmNode pn[], PetscInt k, PetscInt zm, PetscReal zn[])
{
  const PetscScalar zm1 = zm - 1, znl[8] = {pn[0].b + pn[0].h * (PetscScalar)k / zm1,       pn[1].b + pn[1].h * (PetscScalar)k / zm1,       pn[2].b + pn[2].h * (PetscScalar)k / zm1,       pn[3].b + pn[3].h * (PetscScalar)k / zm1,
                                            pn[0].b + pn[0].h * (PetscScalar)(k + 1) / zm1, pn[1].b + pn[1].h * (PetscScalar)(k + 1) / zm1, pn[2].b + pn[2].h * (PetscScalar)(k + 1) / zm1, pn[3].b + pn[3].h * (PetscScalar)(k + 1) / zm1};
  PetscInt          i;
  for (i = 0; i < 8; i++) zn[i] = PetscRealPart(znl[i]);
}

/* Compute a gradient of all the 2D fields at four quadrature points.  Output for [quadrature_point][direction].field_name */
static PetscErrorCode QuadComputeGrad4(const PetscReal dphi[][4][2], PetscReal hx, PetscReal hy, const PrmNode pn[4], PrmNode dp[4][2])
{
  PetscInt q, i, f;
  const PetscScalar(*restrict pg)[PRMNODE_SIZE] = (const PetscScalar(*)[PRMNODE_SIZE])pn; /* Get generic array pointers to the node */
  PetscScalar(*restrict dpg)[2][PRMNODE_SIZE]   = (PetscScalar(*)[2][PRMNODE_SIZE])dp;

  PetscFunctionBeginUser;
  PetscCall(PetscArrayzero(dpg, 4));
  for (q = 0; q < 4; q++) {
    for (i = 0; i < 4; i++) {
      for (f = 0; f < PRMNODE_SIZE; f++) {
        dpg[q][0][f] += dphi[q][i][0] / hx * pg[i][f];
        dpg[q][1][f] += dphi[q][i][1] / hy * pg[i][f];
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscReal StaggeredMidpoint2D(PetscScalar a, PetscScalar b, PetscScalar c, PetscScalar d)
{
  return 0.5 * PetscRealPart(0.75 * a + 0.75 * b + 0.25 * c + 0.25 * d);
}
static inline PetscReal UpwindFlux1D(PetscReal u, PetscReal hL, PetscReal hR)
{
  return (u > 0) ? hL * u : hR * u;
}

#define UpwindFluxXW(x3, x2, h, i, j, k, dj) \
  UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].u, x3[i - 1][j][k].u, x3[i - 1][j + dj][k].u, x3[i][k + dj][k].u), PetscRealPart(0.75 * x2[i - 1][j].h + 0.25 * x2[i - 1][j + dj].h), PetscRealPart(0.75 * x2[i][j].h + 0.25 * x2[i][j + dj].h))
#define UpwindFluxXE(x3, x2, h, i, j, k, dj) \
  UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].u, x3[i + 1][j][k].u, x3[i + 1][j + dj][k].u, x3[i][k + dj][k].u), PetscRealPart(0.75 * x2[i][j].h + 0.25 * x2[i][j + dj].h), PetscRealPart(0.75 * x2[i + 1][j].h + 0.25 * x2[i + 1][j + dj].h))
#define UpwindFluxYS(x3, x2, h, i, j, k, di) \
  UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].v, x3[i][j - 1][k].v, x3[i + di][j - 1][k].v, x3[i + di][j][k].v), PetscRealPart(0.75 * x2[i][j - 1].h + 0.25 * x2[i + di][j - 1].h), PetscRealPart(0.75 * x2[i][j].h + 0.25 * x2[i + di][j].h))
#define UpwindFluxYN(x3, x2, h, i, j, k, di) \
  UpwindFlux1D(StaggeredMidpoint2D(x3[i][j][k].v, x3[i][j + 1][k].v, x3[i + di][j + 1][k].v, x3[i + di][j][k].v), PetscRealPart(0.75 * x2[i][j].h + 0.25 * x2[i + di][j].h), PetscRealPart(0.75 * x2[i][j + 1].h + 0.25 * x2[i + di][j + 1].h))

static void PrmNodeGetFaceMeasure(const PrmNode **p, PetscInt i, PetscInt j, PetscScalar h[])
{
  /* West */
  h[0] = StaggeredMidpoint2D(p[i][j].h, p[i - 1][j].h, p[i - 1][j - 1].h, p[i][j - 1].h);
  h[1] = StaggeredMidpoint2D(p[i][j].h, p[i - 1][j].h, p[i - 1][j + 1].h, p[i][j + 1].h);
  /* East */
  h[2] = StaggeredMidpoint2D(p[i][j].h, p[i + 1][j].h, p[i + 1][j + 1].h, p[i][j + 1].h);
  h[3] = StaggeredMidpoint2D(p[i][j].h, p[i + 1][j].h, p[i + 1][j - 1].h, p[i][j - 1].h);
  /* South */
  h[4] = StaggeredMidpoint2D(p[i][j].h, p[i][j - 1].h, p[i + 1][j - 1].h, p[i + 1][j].h);
  h[5] = StaggeredMidpoint2D(p[i][j].h, p[i][j - 1].h, p[i - 1][j - 1].h, p[i - 1][j].h);
  /* North */
  h[6] = StaggeredMidpoint2D(p[i][j].h, p[i][j + 1].h, p[i - 1][j + 1].h, p[i - 1][j].h);
  h[7] = StaggeredMidpoint2D(p[i][j].h, p[i][j + 1].h, p[i + 1][j + 1].h, p[i + 1][j].h);
}

/* Tests A and C are from the ISMIP-HOM paper (Pattyn et al. 2008) */
static void THIInitialize_HOM_A(THI thi, PetscReal x, PetscReal y, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x * PetscSinReal(thi->alpha);
  p->b            = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x * 2 * PETSC_PI / thi->Lx) * PetscSinReal(y * 2 * PETSC_PI / thi->Ly);
  p->h            = s - p->b;
  p->beta2        = -1e-10; /* This value is not used, but it should not be huge because that would change the finite difference step size  */
}

static void THIInitialize_HOM_C(THI thi, PetscReal x, PetscReal y, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x * PetscSinReal(thi->alpha);
  p->b            = s - 1000 * units->meter;
  p->h            = s - p->b;
  /* tau_b = beta2 v   is a stress (Pa).
   * This is a big number in our units (it needs to balance the driving force from the surface), so we scale it by 1/rhog, just like the residual. */
  p->beta2 = 1000 * (1 + PetscSinReal(x * 2 * PETSC_PI / thi->Lx) * PetscSinReal(y * 2 * PETSC_PI / thi->Ly)) * units->Pascal * units->year / units->meter / thi->rhog;
}

/* These are just toys */

/* From Fred Herman */
static void THIInitialize_HOM_F(THI thi, PetscReal x, PetscReal y, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x * PetscSinReal(thi->alpha);
  p->b            = s - 1000 * units->meter + 100 * units->meter * PetscSinReal(x * 2 * PETSC_PI / thi->Lx); /* * sin(y*2*PETSC_PI/thi->Ly); */
  p->h            = s - p->b;
  p->h            = (1 - (atan((x - thi->Lx / 2) / 1.) + PETSC_PI / 2.) / PETSC_PI) * 500 * units->meter + 1 * units->meter;
  s               = PetscRealPart(p->b + p->h);
  p->beta2        = -1e-10;
  /*  p->beta2 = 1000 * units->Pascal * units->year / units->meter; */
}

/* Same bed as test A, free slip everywhere except for a discontinuous jump to a circular sticky region in the middle. */
static void THIInitialize_HOM_X(THI thi, PetscReal xx, PetscReal yy, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x = xx * 2 * PETSC_PI / thi->Lx - PETSC_PI, y = yy * 2 * PETSC_PI / thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x * x + y * y), s = -x * PetscSinReal(thi->alpha);
  p->b     = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h     = s - p->b;
  p->beta2 = 1000 * (r < 1 ? 2 : 0) * units->Pascal * units->year / units->meter / thi->rhog;
}

/* Like Z, but with 200 meter cliffs */
static void THIInitialize_HOM_Y(THI thi, PetscReal xx, PetscReal yy, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x = xx * 2 * PETSC_PI / thi->Lx - PETSC_PI, y = yy * 2 * PETSC_PI / thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x * x + y * y), s = -x * PetscSinReal(thi->alpha);
  p->b = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  if (PetscRealPart(p->b) > -700 * units->meter) p->b += 200 * units->meter;
  p->h     = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16 * r)) / PetscSqrtReal(1e-2 + 16 * r) * PetscCosReal(x * 3 / 2) * PetscCosReal(y * 3 / 2)) * units->Pascal * units->year / units->meter / thi->rhog;
}

/* Same bed as A, smoothly varying slipperiness, similar to MATLAB's "sombrero" (uncorrelated with bathymetry) */
static void THIInitialize_HOM_Z(THI thi, PetscReal xx, PetscReal yy, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x = xx * 2 * PETSC_PI / thi->Lx - PETSC_PI, y = yy * 2 * PETSC_PI / thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x * x + y * y), s = -x * PetscSinReal(thi->alpha);
  p->b     = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h     = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16 * r)) / PetscSqrtReal(1e-2 + 16 * r) * PetscCosReal(x * 3 / 2) * PetscCosReal(y * 3 / 2)) * units->Pascal * units->year / units->meter / thi->rhog;
}

static void THIFriction(THI thi, PetscReal rbeta2, PetscReal gam, PetscReal *beta2, PetscReal *dbeta2)
{
  if (thi->friction.irefgam == 0) {
    Units units           = thi->units;
    thi->friction.irefgam = 1. / (0.5 * PetscSqr(100 * units->meter / units->year));
    thi->friction.eps2    = 0.5 * PetscSqr(1.e-4 / thi->friction.irefgam);
  }
  if (thi->friction.exponent == 0) {
    *beta2  = rbeta2;
    *dbeta2 = 0;
  } else {
    *beta2  = rbeta2 * PetscPowReal(thi->friction.eps2 + gam * thi->friction.irefgam, thi->friction.exponent);
    *dbeta2 = thi->friction.exponent * *beta2 / (thi->friction.eps2 + gam * thi->friction.irefgam) * thi->friction.irefgam;
  }
}

static void THIViscosity(THI thi, PetscReal gam, PetscReal *eta, PetscReal *deta)
{
  PetscReal Bd2, eps, exponent;
  if (thi->viscosity.Bd2 == 0) {
    Units           units   = thi->units;
    const PetscReal n       = thi->viscosity.glen_n,                                  /* Glen exponent */
      p                     = 1. + 1. / n,                                            /* for Stokes */
      A                     = 1.e-16 * PetscPowReal(units->Pascal, -n) / units->year, /* softness parameter (Pa^{-n}/s) */
      B                     = PetscPowReal(A, -1. / n);                               /* hardness parameter */
    thi->viscosity.Bd2      = B / 2;
    thi->viscosity.exponent = (p - 2) / 2;
    thi->viscosity.eps      = 0.5 * PetscSqr(1e-5 / units->year);
  }
  Bd2      = thi->viscosity.Bd2;
  exponent = thi->viscosity.exponent;
  eps      = thi->viscosity.eps;
  *eta     = Bd2 * PetscPowReal(eps + gam, exponent);
  *deta    = exponent * (*eta) / (eps + gam);
}

static void THIErosion(THI thi, const Node *vel, PetscScalar *erate, Node *derate)
{
  const PetscScalar magref2 = 1.e-10 + (PetscSqr(vel->u) + PetscSqr(vel->v)) / PetscSqr(thi->erosion.refvel), rate = -thi->erosion.rate * PetscPowScalar(magref2, 0.5 * thi->erosion.exponent);
  if (erate) *erate = rate;
  if (derate) {
    if (thi->erosion.exponent == 1) {
      derate->u = 0;
      derate->v = 0;
    } else {
      derate->u = 0.5 * thi->erosion.exponent * rate / magref2 * 2. * vel->u / PetscSqr(thi->erosion.refvel);
      derate->v = 0.5 * thi->erosion.exponent * rate / magref2 * 2. * vel->v / PetscSqr(thi->erosion.refvel);
    }
  }
}

static void RangeUpdate(PetscReal *min, PetscReal *max, PetscReal x)
{
  if (x < *min) *min = x;
  if (x > *max) *max = x;
}

static void PRangeClear(PRange *p)
{
  p->cmin = p->min = 1e100;
  p->cmax = p->max = -1e100;
}

static PetscErrorCode PRangeMinMax(PRange *p, PetscReal min, PetscReal max)
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
  PetscFunctionBeginUser;
  if (--((PetscObject)(*thi))->refct > 0) PetscFunctionReturn(0);
  PetscCall(PetscFree((*thi)->units));
  PetscCall(PetscFree((*thi)->mattype));
  PetscCall(PetscFree((*thi)->monitor_basename));
  PetscCall(PetscHeaderDestroy(thi));
  PetscFunctionReturn(0);
}

static PetscErrorCode THICreate(MPI_Comm comm, THI *inthi)
{
  static PetscBool registered = PETSC_FALSE;
  THI              thi;
  Units            units;
  char             monitor_basename[PETSC_MAX_PATH_LEN] = "thi-";
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  *inthi = 0;
  if (!registered) {
    PetscCall(PetscClassIdRegister("Toy Hydrostatic Ice", &THI_CLASSID));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscHeaderCreate(thi, THI_CLASSID, "THI", "Toy Hydrostatic Ice", "THI", comm, THIDestroy, 0));

  PetscCall(PetscNew(&thi->units));

  units           = thi->units;
  units->meter    = 1e-2;
  units->second   = 1e-7;
  units->kilogram = 1e-12;

  PetscOptionsBegin(comm, NULL, "Scaled units options", "");
  {
    PetscCall(PetscOptionsReal("-units_meter", "1 meter in scaled length units", "", units->meter, &units->meter, NULL));
    PetscCall(PetscOptionsReal("-units_second", "1 second in scaled time units", "", units->second, &units->second, NULL));
    PetscCall(PetscOptionsReal("-units_kilogram", "1 kilogram in scaled mass units", "", units->kilogram, &units->kilogram, NULL));
  }
  PetscOptionsEnd();
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));
  units->year   = 31556926. * units->second, /* seconds per year */

    thi->Lx            = 10.e3;
  thi->Ly              = 10.e3;
  thi->Lz              = 1000;
  thi->nlevels         = 1;
  thi->dirichlet_scale = 1;
  thi->verbose         = PETSC_FALSE;

  thi->viscosity.glen_n = 3.;
  thi->erosion.rate     = 1e-3; /* m/a */
  thi->erosion.exponent = 1.;
  thi->erosion.refvel   = 1.; /* m/a */

  PetscOptionsBegin(comm, NULL, "Toy Hydrostatic Ice options", "");
  {
    QuadratureType quad       = QUAD_GAUSS;
    char           homexp[]   = "A";
    char           mtype[256] = MATSBAIJ;
    PetscReal      L, m = 1.0;
    PetscBool      flg;
    L = thi->Lx;
    PetscCall(PetscOptionsReal("-thi_L", "Domain size (m)", "", L, &L, &flg));
    if (flg) thi->Lx = thi->Ly = L;
    PetscCall(PetscOptionsReal("-thi_Lx", "X Domain size (m)", "", thi->Lx, &thi->Lx, NULL));
    PetscCall(PetscOptionsReal("-thi_Ly", "Y Domain size (m)", "", thi->Ly, &thi->Ly, NULL));
    PetscCall(PetscOptionsReal("-thi_Lz", "Z Domain size (m)", "", thi->Lz, &thi->Lz, NULL));
    PetscCall(PetscOptionsString("-thi_hom", "ISMIP-HOM experiment (A or C)", "", homexp, homexp, sizeof(homexp), NULL));
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
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "HOM experiment '%c' not implemented", homexp[0]);
    }
    PetscCall(PetscOptionsEnum("-thi_quadrature", "Quadrature to use for 3D elements", "", QuadratureTypes, (PetscEnum)quad, (PetscEnum *)&quad, NULL));
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
    PetscCall(PetscOptionsReal("-thi_alpha", "Bed angle (degrees)", "", thi->alpha, &thi->alpha, NULL));
    PetscCall(PetscOptionsReal("-thi_viscosity_glen_n", "Exponent in Glen flow law, 1=linear, infty=ideal plastic", NULL, thi->viscosity.glen_n, &thi->viscosity.glen_n, NULL));
    PetscCall(PetscOptionsReal("-thi_friction_m", "Friction exponent, 0=Coulomb, 1=Navier", "", m, &m, NULL));
    thi->friction.exponent = (m - 1) / 2;
    PetscCall(PetscOptionsReal("-thi_erosion_rate", "Rate of erosion relative to sliding velocity at reference velocity (m/a)", NULL, thi->erosion.rate, &thi->erosion.rate, NULL));
    PetscCall(PetscOptionsReal("-thi_erosion_exponent", "Power of sliding velocity appearing in erosion relation", NULL, thi->erosion.exponent, &thi->erosion.exponent, NULL));
    PetscCall(PetscOptionsReal("-thi_erosion_refvel", "Reference sliding velocity for erosion (m/a)", NULL, thi->erosion.refvel, &thi->erosion.refvel, NULL));
    thi->erosion.rate *= units->meter / units->year;
    thi->erosion.refvel *= units->meter / units->year;
    PetscCall(PetscOptionsReal("-thi_dirichlet_scale", "Scale Dirichlet boundary conditions by this factor", "", thi->dirichlet_scale, &thi->dirichlet_scale, NULL));
    PetscCall(PetscOptionsReal("-thi_ssa_friction_scale", "Scale slip boundary conditions by this factor in SSA (2D) assembly", "", thi->ssa_friction_scale, &thi->ssa_friction_scale, NULL));
    PetscCall(PetscOptionsReal("-thi_inertia", "Coefficient of accelaration term in velocity system, physical is almost zero", NULL, thi->inertia, &thi->inertia, NULL));
    PetscCall(PetscOptionsInt("-thi_nlevels", "Number of levels of refinement", "", thi->nlevels, &thi->nlevels, NULL));
    PetscCall(PetscOptionsFList("-thi_mat_type", "Matrix type", "MatSetType", MatList, mtype, (char *)mtype, sizeof(mtype), NULL));
    PetscCall(PetscStrallocpy(mtype, &thi->mattype));
    PetscCall(PetscOptionsBool("-thi_verbose", "Enable verbose output (like matrix sizes and statistics)", "", thi->verbose, &thi->verbose, NULL));
    PetscCall(PetscOptionsString("-thi_monitor", "Basename to write state files to", NULL, monitor_basename, monitor_basename, sizeof(monitor_basename), &flg));
    if (flg) {
      PetscCall(PetscStrallocpy(monitor_basename, &thi->monitor_basename));
      thi->monitor_interval = 1;
      PetscCall(PetscOptionsInt("-thi_monitor_interval", "Frequency at which to write state files", NULL, thi->monitor_interval, &thi->monitor_interval, NULL));
    }
  }
  PetscOptionsEnd();

  /* dimensionalize */
  thi->Lx *= units->meter;
  thi->Ly *= units->meter;
  thi->Lz *= units->meter;
  thi->alpha *= PETSC_PI / 180;

  PRangeClear(&thi->eta);
  PRangeClear(&thi->beta2);

  {
    PetscReal u = 1000 * units->meter / (3e7 * units->second), gradu = u / (100 * units->meter), eta, deta, rho = 910 * units->kilogram / PetscPowRealInt(units->meter, 3), grav = 9.81 * units->meter / PetscSqr(units->second),
              driving = rho * grav * PetscSinReal(thi->alpha) * 1000 * units->meter;
    THIViscosity(thi, 0.5 * gradu * gradu, &eta, &deta);
    thi->rhog = rho * grav;
    if (thi->verbose) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Units: meter %8.2g  second %8.2g  kg %8.2g  Pa %8.2g\n", (double)units->meter, (double)units->second, (double)units->kilogram, (double)units->Pascal));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Domain (%6.2g,%6.2g,%6.2g), pressure %8.2g, driving stress %8.2g\n", (double)thi->Lx, (double)thi->Ly, (double)thi->Lz, (double)(rho * grav * 1e3 * units->meter), (double)driving));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Large velocity 1km/a %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n", (double)u, (double)gradu, (double)eta, (double)(2 * eta * gradu, 2 * eta * gradu / driving)));
      THIViscosity(thi, 0.5 * PetscSqr(1e-3 * gradu), &eta, &deta);
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Small velocity 1m/a  %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n", (double)(1e-3 * u), (double)(1e-3 * gradu), (double)eta, (double)(2 * eta * 1e-3 * gradu, 2 * eta * 1e-3 * gradu / driving)));
    }
  }

  *inthi = thi;
  PetscFunctionReturn(0);
}

/* Our problem is periodic, but the domain has a mean slope of alpha so the bed does not line up between the upstream
 * and downstream ends of the domain.  This function fixes the ghost values so that the domain appears truly periodic in
 * the horizontal. */
static PetscErrorCode THIFixGhosts(THI thi, DM da3, DM da2, Vec X3, Vec X2)
{
  DMDALocalInfo info;
  PrmNode     **x2;
  PetscInt      i, j;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da3, &info));
  /* PetscCall(VecView(X2,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(DMDAVecGetArray(da2, X2, &x2));
  for (i = info.gzs; i < info.gzs + info.gzm; i++) {
    if (i > -1 && i < info.mz) continue;
    for (j = info.gys; j < info.gys + info.gym; j++) x2[i][j].b += PetscSinReal(thi->alpha) * thi->Lx * (i < 0 ? 1.0 : -1.0);
  }
  PetscCall(DMDAVecRestoreArray(da2, X2, &x2));
  /* PetscCall(VecView(X2,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscFunctionReturn(0);
}

static PetscErrorCode THIInitializePrm(THI thi, DM da2prm, PrmNode **p)
{
  PetscInt i, j, xs, xm, ys, ym, mx, my;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetGhostCorners(da2prm, &ys, &xs, 0, &ym, &xm, 0));
  PetscCall(DMDAGetInfo(da2prm, 0, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscReal xx = thi->Lx * i / mx, yy = thi->Ly * j / my;
      thi->initialize(thi, xx, yy, &p[i][j]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIInitial(THI thi, DM pack, Vec X)
{
  DM        da3, da2;
  PetscInt  i, j, k, xs, xm, ys, ym, zs, zm, mx, my;
  PetscReal hx, hy;
  PrmNode **prm;
  Node   ***x;
  Vec       X3g, X2g, X2;

  PetscFunctionBeginUser;
  PetscCall(DMCompositeGetEntries(pack, &da3, &da2));
  PetscCall(DMCompositeGetAccess(pack, X, &X3g, &X2g));
  PetscCall(DMGetLocalVector(da2, &X2));

  PetscCall(DMDAGetInfo(da3, 0, 0, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da3, &zs, &ys, &xs, &zm, &ym, &xm));
  PetscCall(DMDAVecGetArray(da3, X3g, &x));
  PetscCall(DMDAVecGetArray(da2, X2, &prm));

  PetscCall(THIInitializePrm(thi, da2, prm));

  hx = thi->Lx / mx;
  hy = thi->Ly / my;
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      for (k = zs; k < zs + zm; k++) {
        const PetscScalar zm1 = zm - 1, drivingx = thi->rhog * (prm[i + 1][j].b + prm[i + 1][j].h - prm[i - 1][j].b - prm[i - 1][j].h) / (2 * hx), drivingy = thi->rhog * (prm[i][j + 1].b + prm[i][j + 1].h - prm[i][j - 1].b - prm[i][j - 1].h) / (2 * hy);
        x[i][j][k].u = 0. * drivingx * prm[i][j].h * (PetscScalar)k / zm1;
        x[i][j][k].v = 0. * drivingy * prm[i][j].h * (PetscScalar)k / zm1;
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da3, X3g, &x));
  PetscCall(DMDAVecRestoreArray(da2, X2, &prm));

  PetscCall(DMLocalToGlobalBegin(da2, X2, INSERT_VALUES, X2g));
  PetscCall(DMLocalToGlobalEnd(da2, X2, INSERT_VALUES, X2g));
  PetscCall(DMRestoreLocalVector(da2, &X2));

  PetscCall(DMCompositeRestoreAccess(pack, X, &X3g, &X2g));
  PetscFunctionReturn(0);
}

static void PointwiseNonlinearity(THI thi, const Node n[restrict 8], const PetscReal phi[restrict 3], PetscReal dphi[restrict 8][3], PetscScalar *restrict u, PetscScalar *restrict v, PetscScalar du[restrict 3], PetscScalar dv[restrict 3], PetscReal *eta, PetscReal *deta)
{
  PetscInt    l, ll;
  PetscScalar gam;

  du[0] = du[1] = du[2] = 0;
  dv[0] = dv[1] = dv[2] = 0;
  *u                    = 0;
  *v                    = 0;
  for (l = 0; l < 8; l++) {
    *u += phi[l] * n[l].u;
    *v += phi[l] * n[l].v;
    for (ll = 0; ll < 3; ll++) {
      du[ll] += dphi[l][ll] * n[l].u;
      dv[ll] += dphi[l][ll] * n[l].v;
    }
  }
  gam = Sqr(du[0]) + Sqr(dv[1]) + du[0] * dv[1] + 0.25 * Sqr(du[1] + dv[0]) + 0.25 * Sqr(du[2]) + 0.25 * Sqr(dv[2]);
  THIViscosity(thi, PetscRealPart(gam), eta, deta);
}

static PetscErrorCode THIFunctionLocal_3D(DMDALocalInfo *info, const Node ***x, const PrmNode **prm, const Node ***xdot, Node ***f, THI thi)
{
  PetscInt  xs, ys, xm, ym, zm, i, j, k, q, l;
  PetscReal hx, hy, etamin, etamax, beta2min, beta2max;

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

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PrmNode pn[4], dpn[4][2];
      QuadExtract(prm, i, j, pn);
      PetscCall(QuadComputeGrad4(QuadQDeriv, hx, hy, pn, dpn));
      for (k = 0; k < zm - 1; k++) {
        PetscInt  ls = 0;
        Node      n[8], ndot[8], *fn[8];
        PetscReal zn[8], etabase = 0;

        PrmHexGetZ(pn, k, zm, zn);
        HexExtract(x, i, j, k, n);
        HexExtract(xdot, i, j, k, ndot);
        HexExtractRef(f, i, j, k, fn);
        if (thi->no_slip && k == 0) {
          for (l = 0; l < 4; l++) n[l].u = n[l].v = 0;
          /* The first 4 basis functions lie on the bottom layer, so their contribution is exactly 0, hence we can skip them */
          ls = 4;
        }
        for (q = 0; q < 8; q++) {
          PetscReal   dz[3], phi[8], dphi[8][3], jw, eta, deta;
          PetscScalar du[3], dv[3], u, v, udot = 0, vdot = 0;
          for (l = ls; l < 8; l++) {
            udot += HexQInterp[q][l] * ndot[l].u;
            vdot += HexQInterp[q][l] * ndot[l].v;
          }
          HexGrad(HexQDeriv[q], zn, dz);
          HexComputeGeometry(q, hx, hy, dz, phi, dphi, &jw);
          PointwiseNonlinearity(thi, n, phi, dphi, &u, &v, du, dv, &eta, &deta);
          jw /= thi->rhog; /* scales residuals to be O(1) */
          if (q == 0) etabase = eta;
          RangeUpdate(&etamin, &etamax, eta);
          for (l = ls; l < 8; l++) { /* test functions */
            const PetscScalar ds[2] = {dpn[q % 4][0].h + dpn[q % 4][0].b, dpn[q % 4][1].h + dpn[q % 4][1].b};
            const PetscReal   pp = phi[l], *dp = dphi[l];
            fn[l]->u += dp[0] * jw * eta * (4. * du[0] + 2. * dv[1]) + dp[1] * jw * eta * (du[1] + dv[0]) + dp[2] * jw * eta * du[2] + pp * jw * thi->rhog * ds[0];
            fn[l]->v += dp[1] * jw * eta * (2. * du[0] + 4. * dv[1]) + dp[0] * jw * eta * (du[1] + dv[0]) + dp[2] * jw * eta * dv[2] + pp * jw * thi->rhog * ds[1];
            fn[l]->u += pp * jw * udot * thi->inertia * pp;
            fn[l]->v += pp * jw * vdot * thi->inertia * pp;
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
            const PetscReal   hz    = PetscRealPart(pn[0].h) / (zm - 1.);
            const PetscScalar diagu = 2 * etabase / thi->rhog * (hx * hy / hz + hx * hz / hy + 4 * hy * hz / hx), diagv = 2 * etabase / thi->rhog * (hx * hy / hz + 4 * hx * hz / hy + hy * hz / hx);
            fn[0]->u = thi->dirichlet_scale * diagu * x[i][j][k].u;
            fn[0]->v = thi->dirichlet_scale * diagv * x[i][j][k].v;
          } else {                    /* Integrate over bottom face to apply boundary condition */
            for (q = 0; q < 4; q++) { /* We remove the explicit scaling of the residual by 1/rhog because beta2 already has that scaling to be O(1) */
              const PetscReal jw = 0.25 * hx * hy, *phi = QuadQInterp[q];
              PetscScalar     u = 0, v = 0, rbeta2 = 0;
              PetscReal       beta2, dbeta2;
              for (l = 0; l < 4; l++) {
                u += phi[l] * n[l].u;
                v += phi[l] * n[l].v;
                rbeta2 += phi[l] * pn[l].beta2;
              }
              THIFriction(thi, PetscRealPart(rbeta2), PetscRealPart(u * u + v * v) / 2, &beta2, &dbeta2);
              RangeUpdate(&beta2min, &beta2max, beta2);
              for (l = 0; l < 4; l++) {
                const PetscReal pp = phi[l];
                fn[ls + l]->u += pp * jw * beta2 * u;
                fn[ls + l]->v += pp * jw * beta2 * v;
              }
            }
          }
        }
      }
    }
  }

  PetscCall(PRangeMinMax(&thi->eta, etamin, etamax));
  PetscCall(PRangeMinMax(&thi->beta2, beta2min, beta2max));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIFunctionLocal_2D(DMDALocalInfo *info, const Node ***x, const PrmNode **prm, const PrmNode **prmdot, PrmNode **f, THI thi)
{
  PetscInt xs, ys, xm, ym, zm, i, j, k;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscScalar div = 0, erate, h[8];
      PrmNodeGetFaceMeasure(prm, i, j, h);
      for (k = 0; k < zm; k++) {
        PetscScalar weight = (k == 0 || k == zm - 1) ? 0.5 / (zm - 1) : 1.0 / (zm - 1);
        if (0) { /* centered flux */
          div += (-weight * h[0] * StaggeredMidpoint2D(x[i][j][k].u, x[i - 1][j][k].u, x[i - 1][j - 1][k].u, x[i][j - 1][k].u) - weight * h[1] * StaggeredMidpoint2D(x[i][j][k].u, x[i - 1][j][k].u, x[i - 1][j + 1][k].u, x[i][j + 1][k].u) +
                  weight * h[2] * StaggeredMidpoint2D(x[i][j][k].u, x[i + 1][j][k].u, x[i + 1][j + 1][k].u, x[i][j + 1][k].u) + weight * h[3] * StaggeredMidpoint2D(x[i][j][k].u, x[i + 1][j][k].u, x[i + 1][j - 1][k].u, x[i][j - 1][k].u) -
                  weight * h[4] * StaggeredMidpoint2D(x[i][j][k].v, x[i][j - 1][k].v, x[i + 1][j - 1][k].v, x[i + 1][j][k].v) - weight * h[5] * StaggeredMidpoint2D(x[i][j][k].v, x[i][j - 1][k].v, x[i - 1][j - 1][k].v, x[i - 1][j][k].v) +
                  weight * h[6] * StaggeredMidpoint2D(x[i][j][k].v, x[i][j + 1][k].v, x[i - 1][j + 1][k].v, x[i - 1][j][k].v) + weight * h[7] * StaggeredMidpoint2D(x[i][j][k].v, x[i][j + 1][k].v, x[i + 1][j + 1][k].v, x[i + 1][j][k].v));
        } else { /* Upwind flux */
          div += weight * (-UpwindFluxXW(x, prm, h, i, j, k, 1) - UpwindFluxXW(x, prm, h, i, j, k, -1) + UpwindFluxXE(x, prm, h, i, j, k, 1) + UpwindFluxXE(x, prm, h, i, j, k, -1) - UpwindFluxYS(x, prm, h, i, j, k, 1) - UpwindFluxYS(x, prm, h, i, j, k, -1) + UpwindFluxYN(x, prm, h, i, j, k, 1) + UpwindFluxYN(x, prm, h, i, j, k, -1));
        }
      }
      /* printf("div[%d][%d] %g\n",i,j,div); */
      THIErosion(thi, &x[i][j][0], &erate, NULL);
      f[i][j].b     = prmdot[i][j].b - erate;
      f[i][j].h     = prmdot[i][j].h + div;
      f[i][j].beta2 = prmdot[i][j].beta2;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  THI             thi = (THI)ctx;
  DM              pack, da3, da2;
  Vec             X3, X2, Xdot3, Xdot2, F3, F2, F3g, F2g;
  const Node   ***x3, ***xdot3;
  const PrmNode **x2, **xdot2;
  Node         ***f3;
  PrmNode       **f2;
  DMDALocalInfo   info3;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMCompositeGetEntries(pack, &da3, &da2));
  PetscCall(DMDAGetLocalInfo(da3, &info3));
  PetscCall(DMCompositeGetLocalVectors(pack, &X3, &X2));
  PetscCall(DMCompositeGetLocalVectors(pack, &Xdot3, &Xdot2));
  PetscCall(DMCompositeScatter(pack, X, X3, X2));
  PetscCall(THIFixGhosts(thi, da3, da2, X3, X2));
  PetscCall(DMCompositeScatter(pack, Xdot, Xdot3, Xdot2));

  PetscCall(DMGetLocalVector(da3, &F3));
  PetscCall(DMGetLocalVector(da2, &F2));
  PetscCall(VecZeroEntries(F3));

  PetscCall(DMDAVecGetArray(da3, X3, &x3));
  PetscCall(DMDAVecGetArray(da2, X2, &x2));
  PetscCall(DMDAVecGetArray(da3, Xdot3, &xdot3));
  PetscCall(DMDAVecGetArray(da2, Xdot2, &xdot2));
  PetscCall(DMDAVecGetArray(da3, F3, &f3));
  PetscCall(DMDAVecGetArray(da2, F2, &f2));

  PetscCall(THIFunctionLocal_3D(&info3, x3, x2, xdot3, f3, thi));
  PetscCall(THIFunctionLocal_2D(&info3, x3, x2, xdot2, f2, thi));

  PetscCall(DMDAVecRestoreArray(da3, X3, &x3));
  PetscCall(DMDAVecRestoreArray(da2, X2, &x2));
  PetscCall(DMDAVecRestoreArray(da3, Xdot3, &xdot3));
  PetscCall(DMDAVecRestoreArray(da2, Xdot2, &xdot2));
  PetscCall(DMDAVecRestoreArray(da3, F3, &f3));
  PetscCall(DMDAVecRestoreArray(da2, F2, &f2));

  PetscCall(DMCompositeRestoreLocalVectors(pack, &X3, &X2));
  PetscCall(DMCompositeRestoreLocalVectors(pack, &Xdot3, &Xdot2));

  PetscCall(VecZeroEntries(F));
  PetscCall(DMCompositeGetAccess(pack, F, &F3g, &F2g));
  PetscCall(DMLocalToGlobalBegin(da3, F3, ADD_VALUES, F3g));
  PetscCall(DMLocalToGlobalEnd(da3, F3, ADD_VALUES, F3g));
  PetscCall(DMLocalToGlobalBegin(da2, F2, INSERT_VALUES, F2g));
  PetscCall(DMLocalToGlobalEnd(da2, F2, INSERT_VALUES, F2g));

  if (thi->verbose) {
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)thi), &viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "3D_Velocity residual (bs=2):\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(VecView(F3, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "2D_Fields residual (bs=3):\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(VecView(F2, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }

  PetscCall(DMCompositeRestoreAccess(pack, F, &F3g, &F2g));

  PetscCall(DMRestoreLocalVector(da3, &F3));
  PetscCall(DMRestoreLocalVector(da2, &F2));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIMatrixStatistics(THI thi, Mat B, PetscViewer viewer)
{
  PetscReal   nrm;
  PetscInt    m;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(MatNorm(B, NORM_FROBENIUS, &nrm));
  PetscCall(MatGetSize(B, &m, 0));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)B), &rank));
  if (rank == 0) {
    PetscScalar val0, val2;
    PetscCall(MatGetValue(B, 0, 0, &val0));
    PetscCall(MatGetValue(B, 2, 2, &val2));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Matrix dim %8" PetscInt_FMT "  norm %8.2e, (0,0) %8.2e  (2,2) %8.2e, eta [%8.2e,%8.2e] beta2 [%8.2e,%8.2e]\n", m, (double)nrm, (double)PetscRealPart(val0), (double)PetscRealPart(val2), (double)thi->eta.cmin,
                                     (double)thi->eta.cmax, (double)thi->beta2.cmin, (double)thi->beta2.cmax));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THISurfaceStatistics(DM pack, Vec X, PetscReal *min, PetscReal *max, PetscReal *mean)
{
  DM          da3, da2;
  Vec         X3, X2;
  Node     ***x;
  PetscInt    i, j, xs, ys, zs, xm, ym, zm, mx, my, mz;
  PetscReal   umin = 1e100, umax = -1e100;
  PetscScalar usum = 0.0, gusum;

  PetscFunctionBeginUser;
  PetscCall(DMCompositeGetEntries(pack, &da3, &da2));
  PetscCall(DMCompositeGetAccess(pack, X, &X3, &X2));
  *min = *max = *mean = 0;
  PetscCall(DMDAGetInfo(da3, 0, &mz, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da3, &zs, &ys, &xs, &zm, &ym, &xm));
  PetscCheck(zs == 0 && zm == mz, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected decomposition");
  PetscCall(DMDAVecGetArray(da3, X3, &x));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscReal u = PetscRealPart(x[i][j][zm - 1].u);
      RangeUpdate(&umin, &umax, u);
      usum += u;
    }
  }
  PetscCall(DMDAVecRestoreArray(da3, X3, &x));
  PetscCall(DMCompositeRestoreAccess(pack, X, &X3, &X2));

  PetscCallMPI(MPI_Allreduce(&umin, min, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)da3)));
  PetscCallMPI(MPI_Allreduce(&umax, max, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)da3)));
  PetscCallMPI(MPI_Allreduce(&usum, &gusum, 1, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)da3)));
  *mean = PetscRealPart(gusum) / (mx * my);
  PetscFunctionReturn(0);
}

static PetscErrorCode THISolveStatistics(THI thi, TS ts, PetscInt coarsened, const char name[])
{
  MPI_Comm comm;
  DM       pack;
  Vec      X, X3, X2;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)thi, &comm));
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(TSGetSolution(ts, &X));
  PetscCall(DMCompositeGetAccess(pack, X, &X3, &X2));
  PetscCall(PetscPrintf(comm, "Solution statistics after solve: %s\n", name));
  {
    PetscInt            its, lits;
    SNESConvergedReason reason;
    SNES                snes;
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetIterationNumber(snes, &its));
    PetscCall(SNESGetConvergedReason(snes, &reason));
    PetscCall(SNESGetLinearSolveIterations(snes, &lits));
    PetscCall(PetscPrintf(comm, "%s: Number of SNES iterations = %" PetscInt_FMT ", total linear iterations = %" PetscInt_FMT "\n", SNESConvergedReasons[reason], its, lits));
  }
  {
    PetscReal    nrm2, tmin[3] = {1e100, 1e100, 1e100}, tmax[3] = {-1e100, -1e100, -1e100}, min[3], max[3];
    PetscInt     i, j, m;
    PetscScalar *x;
    PetscCall(VecNorm(X3, NORM_2, &nrm2));
    PetscCall(VecGetLocalSize(X3, &m));
    PetscCall(VecGetArray(X3, &x));
    for (i = 0; i < m; i += 2) {
      PetscReal u = PetscRealPart(x[i]), v = PetscRealPart(x[i + 1]), c = PetscSqrtReal(u * u + v * v);
      tmin[0] = PetscMin(u, tmin[0]);
      tmin[1] = PetscMin(v, tmin[1]);
      tmin[2] = PetscMin(c, tmin[2]);
      tmax[0] = PetscMax(u, tmax[0]);
      tmax[1] = PetscMax(v, tmax[1]);
      tmax[2] = PetscMax(c, tmax[2]);
    }
    PetscCall(VecRestoreArray(X, &x));
    PetscCallMPI(MPI_Allreduce(tmin, min, 3, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)thi)));
    PetscCallMPI(MPI_Allreduce(tmax, max, 3, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)thi)));
    /* Dimensionalize to meters/year */
    nrm2 *= thi->units->year / thi->units->meter;
    for (j = 0; j < 3; j++) {
      min[j] *= thi->units->year / thi->units->meter;
      max[j] *= thi->units->year / thi->units->meter;
    }
    PetscCall(PetscPrintf(comm, "|X|_2 %g   u in [%g, %g]   v in [%g, %g]   c in [%g, %g] \n", (double)nrm2, (double)min[0], (double)max[0], (double)min[1], (double)max[1], (double)min[2], (double)max[2]));
    {
      PetscReal umin, umax, umean;
      PetscCall(THISurfaceStatistics(pack, X, &umin, &umax, &umean));
      umin *= thi->units->year / thi->units->meter;
      umax *= thi->units->year / thi->units->meter;
      umean *= thi->units->year / thi->units->meter;
      PetscCall(PetscPrintf(comm, "Surface statistics: u in [%12.6e, %12.6e] mean %12.6e\n", (double)umin, (double)umax, (double)umean));
    }
    /* These values stay nondimensional */
    PetscCall(PetscPrintf(comm, "Global eta range   [%g, %g], converged range [%g, %g]\n", (double)thi->eta.min, (double)thi->eta.max, (double)thi->eta.cmin, (double)thi->eta.cmax));
    PetscCall(PetscPrintf(comm, "Global beta2 range [%g, %g], converged range [%g, %g]\n", (double)thi->beta2.min, (double)thi->beta2.max, (double)thi->beta2.cmin, (double)thi->beta2.cmax));
  }
  PetscCall(PetscPrintf(comm, "\n"));
  PetscCall(DMCompositeRestoreAccess(pack, X, &X3, &X2));
  PetscFunctionReturn(0);
}

static inline PetscInt DMDALocalIndex3D(DMDALocalInfo *info, PetscInt i, PetscInt j, PetscInt k)
{
  return ((i - info->gzs) * info->gym + (j - info->gys)) * info->gxm + (k - info->gxs);
}
static inline PetscInt DMDALocalIndex2D(DMDALocalInfo *info, PetscInt i, PetscInt j)
{
  return (i - info->gzs) * info->gym + (j - info->gys);
}

static PetscErrorCode THIJacobianLocal_Momentum(DMDALocalInfo *info, const Node ***x, const PrmNode **prm, Mat B, Mat Bcpl, THI thi)
{
  PetscInt  xs, ys, xm, ym, zm, i, j, k, q, l, ll;
  PetscReal hx, hy;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;
  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PrmNode pn[4], dpn[4][2];
      QuadExtract(prm, i, j, pn);
      PetscCall(QuadComputeGrad4(QuadQDeriv, hx, hy, pn, dpn));
      for (k = 0; k < zm - 1; k++) {
        Node        n[8];
        PetscReal   zn[8], etabase = 0;
        PetscScalar Ke[8 * NODE_SIZE][8 * NODE_SIZE], Kcpl[8 * NODE_SIZE][4 * PRMNODE_SIZE];
        PetscInt    ls = 0;

        PrmHexGetZ(pn, k, zm, zn);
        HexExtract(x, i, j, k, n);
        PetscCall(PetscMemzero(Ke, sizeof(Ke)));
        PetscCall(PetscMemzero(Kcpl, sizeof(Kcpl)));
        if (thi->no_slip && k == 0) {
          for (l = 0; l < 4; l++) n[l].u = n[l].v = 0;
          ls = 4;
        }
        for (q = 0; q < 8; q++) {
          PetscReal   dz[3], phi[8], dphi[8][3], jw, eta, deta;
          PetscScalar du[3], dv[3], u, v;
          HexGrad(HexQDeriv[q], zn, dz);
          HexComputeGeometry(q, hx, hy, dz, phi, dphi, &jw);
          PointwiseNonlinearity(thi, n, phi, dphi, &u, &v, du, dv, &eta, &deta);
          jw /= thi->rhog; /* residuals are scaled by this factor */
          if (q == 0) etabase = eta;
          for (l = ls; l < 8; l++) { /* test functions */
            const PetscReal pp = phi[l], *restrict dp = dphi[l];
            for (ll = ls; ll < 8; ll++) { /* trial functions */
              const PetscReal *restrict dpl = dphi[ll];
              PetscScalar dgdu, dgdv;
              dgdu = 2. * du[0] * dpl[0] + dv[1] * dpl[0] + 0.5 * (du[1] + dv[0]) * dpl[1] + 0.5 * du[2] * dpl[2];
              dgdv = 2. * dv[1] * dpl[1] + du[0] * dpl[1] + 0.5 * (du[1] + dv[0]) * dpl[0] + 0.5 * dv[2] * dpl[2];
              /* Picard part */
              Ke[l * 2 + 0][ll * 2 + 0] += dp[0] * jw * eta * 4. * dpl[0] + dp[1] * jw * eta * dpl[1] + dp[2] * jw * eta * dpl[2];
              Ke[l * 2 + 0][ll * 2 + 1] += dp[0] * jw * eta * 2. * dpl[1] + dp[1] * jw * eta * dpl[0];
              Ke[l * 2 + 1][ll * 2 + 0] += dp[1] * jw * eta * 2. * dpl[0] + dp[0] * jw * eta * dpl[1];
              Ke[l * 2 + 1][ll * 2 + 1] += dp[1] * jw * eta * 4. * dpl[1] + dp[0] * jw * eta * dpl[0] + dp[2] * jw * eta * dpl[2];
              /* extra Newton terms */
              Ke[l * 2 + 0][ll * 2 + 0] += dp[0] * jw * deta * dgdu * (4. * du[0] + 2. * dv[1]) + dp[1] * jw * deta * dgdu * (du[1] + dv[0]) + dp[2] * jw * deta * dgdu * du[2];
              Ke[l * 2 + 0][ll * 2 + 1] += dp[0] * jw * deta * dgdv * (4. * du[0] + 2. * dv[1]) + dp[1] * jw * deta * dgdv * (du[1] + dv[0]) + dp[2] * jw * deta * dgdv * du[2];
              Ke[l * 2 + 1][ll * 2 + 0] += dp[1] * jw * deta * dgdu * (4. * dv[1] + 2. * du[0]) + dp[0] * jw * deta * dgdu * (du[1] + dv[0]) + dp[2] * jw * deta * dgdu * dv[2];
              Ke[l * 2 + 1][ll * 2 + 1] += dp[1] * jw * deta * dgdv * (4. * dv[1] + 2. * du[0]) + dp[0] * jw * deta * dgdv * (du[1] + dv[0]) + dp[2] * jw * deta * dgdv * dv[2];
              /* inertial part */
              Ke[l * 2 + 0][ll * 2 + 0] += pp * jw * thi->inertia * pp;
              Ke[l * 2 + 1][ll * 2 + 1] += pp * jw * thi->inertia * pp;
            }
            for (ll = 0; ll < 4; ll++) {                                                              /* Trial functions for surface/bed */
              const PetscReal dpl[] = {QuadQDeriv[q % 4][ll][0] / hx, QuadQDeriv[q % 4][ll][1] / hy}; /* surface = h + b */
              Kcpl[FieldIndex(Node, l, u)][FieldIndex(PrmNode, ll, h)] += pp * jw * thi->rhog * dpl[0];
              Kcpl[FieldIndex(Node, l, u)][FieldIndex(PrmNode, ll, b)] += pp * jw * thi->rhog * dpl[0];
              Kcpl[FieldIndex(Node, l, v)][FieldIndex(PrmNode, ll, h)] += pp * jw * thi->rhog * dpl[1];
              Kcpl[FieldIndex(Node, l, v)][FieldIndex(PrmNode, ll, b)] += pp * jw * thi->rhog * dpl[1];
            }
          }
        }
        if (k == 0) { /* on a bottom face */
          if (thi->no_slip) {
            const PetscReal   hz    = PetscRealPart(pn[0].h) / (zm - 1);
            const PetscScalar diagu = 2 * etabase / thi->rhog * (hx * hy / hz + hx * hz / hy + 4 * hy * hz / hx), diagv = 2 * etabase / thi->rhog * (hx * hy / hz + 4 * hx * hz / hy + hy * hz / hx);
            Ke[0][0] = thi->dirichlet_scale * diagu;
            Ke[0][1] = 0;
            Ke[1][0] = 0;
            Ke[1][1] = thi->dirichlet_scale * diagv;
          } else {
            for (q = 0; q < 4; q++) { /* We remove the explicit scaling by 1/rhog because beta2 already has that scaling to be O(1) */
              const PetscReal jw = 0.25 * hx * hy, *phi = QuadQInterp[q];
              PetscScalar     u = 0, v = 0, rbeta2 = 0;
              PetscReal       beta2, dbeta2;
              for (l = 0; l < 4; l++) {
                u += phi[l] * n[l].u;
                v += phi[l] * n[l].v;
                rbeta2 += phi[l] * pn[l].beta2;
              }
              THIFriction(thi, PetscRealPart(rbeta2), PetscRealPart(u * u + v * v) / 2, &beta2, &dbeta2);
              for (l = 0; l < 4; l++) {
                const PetscReal pp = phi[l];
                for (ll = 0; ll < 4; ll++) {
                  const PetscReal ppl = phi[ll];
                  Ke[l * 2 + 0][ll * 2 + 0] += pp * jw * beta2 * ppl + pp * jw * dbeta2 * u * u * ppl;
                  Ke[l * 2 + 0][ll * 2 + 1] += pp * jw * dbeta2 * u * v * ppl;
                  Ke[l * 2 + 1][ll * 2 + 0] += pp * jw * dbeta2 * v * u * ppl;
                  Ke[l * 2 + 1][ll * 2 + 1] += pp * jw * beta2 * ppl + pp * jw * dbeta2 * v * v * ppl;
                }
              }
            }
          }
        }
        {
          const PetscInt rc3blocked[8]                 = {DMDALocalIndex3D(info, i + 0, j + 0, k + 0), DMDALocalIndex3D(info, i + 1, j + 0, k + 0), DMDALocalIndex3D(info, i + 1, j + 1, k + 0), DMDALocalIndex3D(info, i + 0, j + 1, k + 0),
                                                          DMDALocalIndex3D(info, i + 0, j + 0, k + 1), DMDALocalIndex3D(info, i + 1, j + 0, k + 1), DMDALocalIndex3D(info, i + 1, j + 1, k + 1), DMDALocalIndex3D(info, i + 0, j + 1, k + 1)},
                         col2blocked[PRMNODE_SIZE * 4] = {DMDALocalIndex2D(info, i + 0, j + 0), DMDALocalIndex2D(info, i + 1, j + 0), DMDALocalIndex2D(info, i + 1, j + 1), DMDALocalIndex2D(info, i + 0, j + 1)};
#if !defined COMPUTE_LOWER_TRIANGULAR /* fill in lower-triangular part, this is really cheap compared to computing the entries */
          for (l = 0; l < 8; l++) {
            for (ll = l + 1; ll < 8; ll++) {
              Ke[ll * 2 + 0][l * 2 + 0] = Ke[l * 2 + 0][ll * 2 + 0];
              Ke[ll * 2 + 1][l * 2 + 0] = Ke[l * 2 + 0][ll * 2 + 1];
              Ke[ll * 2 + 0][l * 2 + 1] = Ke[l * 2 + 1][ll * 2 + 0];
              Ke[ll * 2 + 1][l * 2 + 1] = Ke[l * 2 + 1][ll * 2 + 1];
            }
          }
#endif
          PetscCall(MatSetValuesBlockedLocal(B, 8, rc3blocked, 8, rc3blocked, &Ke[0][0], ADD_VALUES)); /* velocity-velocity coupling can use blocked insertion */
          {                                                                                            /* The off-diagonal part cannot (yet) */
            PetscInt row3scalar[NODE_SIZE * 8], col2scalar[PRMNODE_SIZE * 4];
            for (l = 0; l < 8; l++)
              for (ll = 0; ll < NODE_SIZE; ll++) row3scalar[l * NODE_SIZE + ll] = rc3blocked[l] * NODE_SIZE + ll;
            for (l = 0; l < 4; l++)
              for (ll = 0; ll < PRMNODE_SIZE; ll++) col2scalar[l * PRMNODE_SIZE + ll] = col2blocked[l] * PRMNODE_SIZE + ll;
            PetscCall(MatSetValuesLocal(Bcpl, 8 * NODE_SIZE, row3scalar, 4 * PRMNODE_SIZE, col2scalar, &Kcpl[0][0], ADD_VALUES));
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobianLocal_2D(DMDALocalInfo *info, const Node ***x3, const PrmNode **x2, const PrmNode **xdot2, PetscReal a, Mat B22, Mat B21, THI thi)
{
  PetscInt xs, ys, xm, ym, zm, i, j, k;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;

  PetscCheck(zm <= 1024, ((PetscObject)info->da)->comm, PETSC_ERR_SUP, "Need to allocate more space");
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      { /* Self-coupling */
        const PetscInt    row[]  = {DMDALocalIndex2D(info, i, j)};
        const PetscInt    col[]  = {DMDALocalIndex2D(info, i, j)};
        const PetscScalar vals[] = {a, 0, 0, 0, a, 0, 0, 0, a};
        PetscCall(MatSetValuesBlockedLocal(B22, 1, row, 1, col, vals, INSERT_VALUES));
      }
      for (k = 0; k < zm; k++) { /* Coupling to velocity problem */
        /* Use a cheaper quadrature than for residual evaluation, because it is much sparser */
        const PetscInt    row[]  = {FieldIndex(PrmNode, DMDALocalIndex2D(info, i, j), h)};
        const PetscInt    cols[] = {FieldIndex(Node, DMDALocalIndex3D(info, i - 1, j, k), u), FieldIndex(Node, DMDALocalIndex3D(info, i, j, k), u), FieldIndex(Node, DMDALocalIndex3D(info, i + 1, j, k), u),
                                    FieldIndex(Node, DMDALocalIndex3D(info, i, j - 1, k), v), FieldIndex(Node, DMDALocalIndex3D(info, i, j, k), v), FieldIndex(Node, DMDALocalIndex3D(info, i, j + 1, k), v)};
        const PetscScalar w = (k && k < zm - 1) ? 0.5 : 0.25, hW = w * (x2[i - 1][j].h + x2[i][j].h) / (zm - 1.), hE = w * (x2[i][j].h + x2[i + 1][j].h) / (zm - 1.), hS = w * (x2[i][j - 1].h + x2[i][j].h) / (zm - 1.),
                          hN = w * (x2[i][j].h + x2[i][j + 1].h) / (zm - 1.);
        PetscScalar *vals, vals_upwind[] = {((PetscRealPart(x3[i][j][k].u) > 0) ? -hW : 0), ((PetscRealPart(x3[i][j][k].u) > 0) ? +hE : -hW), ((PetscRealPart(x3[i][j][k].u) > 0) ? 0 : +hE),
                                            ((PetscRealPart(x3[i][j][k].v) > 0) ? -hS : 0), ((PetscRealPart(x3[i][j][k].v) > 0) ? +hN : -hS), ((PetscRealPart(x3[i][j][k].v) > 0) ? 0 : +hN)},
                           vals_centered[] = {-0.5 * hW, 0.5 * (-hW + hE), 0.5 * hE, -0.5 * hS, 0.5 * (-hS + hN), 0.5 * hN};
        vals                               = 1 ? vals_upwind : vals_centered;
        if (k == 0) {
          Node derate;
          THIErosion(thi, &x3[i][j][0], NULL, &derate);
          vals[1] -= derate.u;
          vals[4] -= derate.v;
        }
        PetscCall(MatSetValuesLocal(B21, 1, row, 6, cols, vals, INSERT_VALUES));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, void *ctx)
{
  THI             thi = (THI)ctx;
  DM              pack, da3, da2;
  Vec             X3, X2, Xdot2;
  Mat             B11, B12, B21, B22;
  DMDALocalInfo   info3;
  IS             *isloc;
  const Node   ***x3;
  const PrmNode **x2, **xdot2;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMCompositeGetEntries(pack, &da3, &da2));
  PetscCall(DMDAGetLocalInfo(da3, &info3));
  PetscCall(DMCompositeGetLocalVectors(pack, &X3, &X2));
  PetscCall(DMCompositeGetLocalVectors(pack, NULL, &Xdot2));
  PetscCall(DMCompositeScatter(pack, X, X3, X2));
  PetscCall(THIFixGhosts(thi, da3, da2, X3, X2));
  PetscCall(DMCompositeScatter(pack, Xdot, NULL, Xdot2));

  PetscCall(MatZeroEntries(B));

  PetscCall(DMCompositeGetLocalISs(pack, &isloc));
  PetscCall(MatGetLocalSubMatrix(B, isloc[0], isloc[0], &B11));
  PetscCall(MatGetLocalSubMatrix(B, isloc[0], isloc[1], &B12));
  PetscCall(MatGetLocalSubMatrix(B, isloc[1], isloc[0], &B21));
  PetscCall(MatGetLocalSubMatrix(B, isloc[1], isloc[1], &B22));

  PetscCall(DMDAVecGetArray(da3, X3, &x3));
  PetscCall(DMDAVecGetArray(da2, X2, &x2));
  PetscCall(DMDAVecGetArray(da2, Xdot2, &xdot2));

  PetscCall(THIJacobianLocal_Momentum(&info3, x3, x2, B11, B12, thi));

  /* Need to switch from ADD_VALUES to INSERT_VALUES */
  PetscCall(MatAssemblyBegin(B, MAT_FLUSH_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FLUSH_ASSEMBLY));

  PetscCall(THIJacobianLocal_2D(&info3, x3, x2, xdot2, a, B22, B21, thi));

  PetscCall(DMDAVecRestoreArray(da3, X3, &x3));
  PetscCall(DMDAVecRestoreArray(da2, X2, &x2));
  PetscCall(DMDAVecRestoreArray(da2, Xdot2, &xdot2));

  PetscCall(MatRestoreLocalSubMatrix(B, isloc[0], isloc[0], &B11));
  PetscCall(MatRestoreLocalSubMatrix(B, isloc[0], isloc[1], &B12));
  PetscCall(MatRestoreLocalSubMatrix(B, isloc[1], isloc[0], &B21));
  PetscCall(MatRestoreLocalSubMatrix(B, isloc[1], isloc[1], &B22));
  PetscCall(ISDestroy(&isloc[0]));
  PetscCall(ISDestroy(&isloc[1]));
  PetscCall(PetscFree(isloc));

  PetscCall(DMCompositeRestoreLocalVectors(pack, &X3, &X2));
  PetscCall(DMCompositeRestoreLocalVectors(pack, 0, &Xdot2));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  if (thi->verbose) PetscCall(THIMatrixStatistics(thi, B, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

/* VTK's XML formats are so brain-dead that they can't handle multiple grids in the same file.  Since the communication
 * can be shared between the two grids, we write two files at once, one for velocity (living on a 3D grid defined by
 * h=thickness and b=bed) and another for all properties living on the 2D grid.
 */
static PetscErrorCode THIDAVecView_VTK_XML(THI thi, DM pack, Vec X, const char filename[], const char filename2[])
{
  const PetscInt dof = NODE_SIZE, dof2 = PRMNODE_SIZE;
  Units          units = thi->units;
  MPI_Comm       comm;
  PetscViewer    viewer3, viewer2;
  PetscMPIInt    rank, size, tag, nn, nmax, nn2, nmax2;
  PetscInt       mx, my, mz, r, range[6];
  PetscScalar   *x, *x2;
  DM             da3, da2;
  Vec            X3, X2;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)thi, &comm));
  PetscCall(DMCompositeGetEntries(pack, &da3, &da2));
  PetscCall(DMCompositeGetAccess(pack, X, &X3, &X2));
  PetscCall(DMDAGetInfo(da3, 0, &mz, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerASCIIOpen(comm, filename, &viewer3));
  PetscCall(PetscViewerASCIIOpen(comm, filename2, &viewer2));
  PetscCall(PetscViewerASCIIPrintf(viewer3, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer2, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer3, "  <StructuredGrid WholeExtent=\"%d %" PetscInt_FMT " %d %" PetscInt_FMT " %d %" PetscInt_FMT "\">\n", 0, mz - 1, 0, my - 1, 0, mx - 1));
  PetscCall(PetscViewerASCIIPrintf(viewer2, "  <StructuredGrid WholeExtent=\"%d %d %d %" PetscInt_FMT " %d %" PetscInt_FMT "\">\n", 0, 0, 0, my - 1, 0, mx - 1));

  PetscCall(DMDAGetCorners(da3, range, range + 1, range + 2, range + 3, range + 4, range + 5));
  PetscCall(PetscMPIIntCast(range[3] * range[4] * range[5] * dof, &nn));
  PetscCallMPI(MPI_Reduce(&nn, &nmax, 1, MPI_INT, MPI_MAX, 0, comm));
  PetscCall(PetscMPIIntCast(range[4] * range[5] * dof2, &nn2));
  PetscCallMPI(MPI_Reduce(&nn2, &nmax2, 1, MPI_INT, MPI_MAX, 0, comm));
  tag = ((PetscObject)viewer3)->tag;
  PetscCall(VecGetArrayRead(X3, (const PetscScalar **)&x));
  PetscCall(VecGetArrayRead(X2, (const PetscScalar **)&x2));
  if (rank == 0) {
    PetscScalar *array, *array2;
    PetscCall(PetscMalloc2(nmax, &array, nmax2, &array2));
    for (r = 0; r < size; r++) {
      PetscInt i, j, k, f, xs, xm, ys, ym, zs, zm;
      Node    *y3;
      PetscScalar(*y2)[PRMNODE_SIZE];
      MPI_Status status;

      if (r) PetscCallMPI(MPI_Recv(range, 6, MPIU_INT, r, tag, comm, MPI_STATUS_IGNORE));
      zs = range[0];
      ys = range[1];
      xs = range[2];
      zm = range[3];
      ym = range[4];
      xm = range[5];
      PetscCheck(xm * ym * zm * dof <= nmax, PETSC_COMM_SELF, PETSC_ERR_PLIB, "should not happen");
      if (r) {
        PetscCallMPI(MPI_Recv(array, nmax, MPIU_SCALAR, r, tag, comm, &status));
        PetscCallMPI(MPI_Get_count(&status, MPIU_SCALAR, &nn));
        PetscCheck(nn == xm * ym * zm * dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "corrupt da3 send");
        y3 = (Node *)array;
        PetscCallMPI(MPI_Recv(array2, nmax2, MPIU_SCALAR, r, tag, comm, &status));
        PetscCallMPI(MPI_Get_count(&status, MPIU_SCALAR, &nn2));
        PetscCheck(nn2 == xm * ym * dof2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "corrupt da2 send");
        y2 = (PetscScalar(*)[PRMNODE_SIZE])array2;
      } else {
        y3 = (Node *)x;
        y2 = (PetscScalar(*)[PRMNODE_SIZE])x2;
      }
      PetscCall(PetscViewerASCIIPrintf(viewer3, "    <Piece Extent=\"%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\">\n", zs, zs + zm - 1, ys, ys + ym - 1, xs, xs + xm - 1));
      PetscCall(PetscViewerASCIIPrintf(viewer2, "    <Piece Extent=\"%d %d %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\">\n", 0, 0, ys, ys + ym - 1, xs, xs + xm - 1));

      PetscCall(PetscViewerASCIIPrintf(viewer3, "      <Points>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer2, "      <Points>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer3, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer2, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n"));
      for (i = xs; i < xs + xm; i++) {
        for (j = ys; j < ys + ym; j++) {
          PetscReal xx = thi->Lx * i / mx, yy = thi->Ly * j / my, b = PetscRealPart(y2[i * ym + j][FieldOffset(PrmNode, b)]), h = PetscRealPart(y2[i * ym + j][FieldOffset(PrmNode, h)]);
          for (k = zs; k < zs + zm; k++) {
            PetscReal zz = b + h * k / (mz - 1.);
            PetscCall(PetscViewerASCIIPrintf(viewer3, "%f %f %f\n", (double)xx, (double)yy, (double)zz));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer2, "%f %f %f\n", (double)xx, (double)yy, (double)0.0));
        }
      }
      PetscCall(PetscViewerASCIIPrintf(viewer3, "        </DataArray>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer2, "        </DataArray>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer3, "      </Points>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer2, "      </Points>\n"));

      { /* Velocity and rank (3D) */
        PetscCall(PetscViewerASCIIPrintf(viewer3, "      <PointData>\n"));
        PetscCall(PetscViewerASCIIPrintf(viewer3, "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n"));
        for (i = 0; i < nn / dof; i++) PetscCall(PetscViewerASCIIPrintf(viewer3, "%f %f %f\n", (double)(PetscRealPart(y3[i].u) * units->year / units->meter), (double)(PetscRealPart(y3[i].v) * units->year / units->meter), 0.0));
        PetscCall(PetscViewerASCIIPrintf(viewer3, "        </DataArray>\n"));

        PetscCall(PetscViewerASCIIPrintf(viewer3, "        <DataArray type=\"Int32\" Name=\"rank\" NumberOfComponents=\"1\" format=\"ascii\">\n"));
        for (i = 0; i < nn; i += dof) PetscCall(PetscViewerASCIIPrintf(viewer3, "%" PetscInt_FMT "\n", r));
        PetscCall(PetscViewerASCIIPrintf(viewer3, "        </DataArray>\n"));
        PetscCall(PetscViewerASCIIPrintf(viewer3, "      </PointData>\n"));
      }

      { /* 2D */
        PetscCall(PetscViewerASCIIPrintf(viewer2, "      <PointData>\n"));
        for (f = 0; f < PRMNODE_SIZE; f++) {
          const char *fieldname;
          PetscCall(DMDAGetFieldName(da2, f, &fieldname));
          PetscCall(PetscViewerASCIIPrintf(viewer2, "        <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\">\n", fieldname));
          for (i = 0; i < nn2 / PRMNODE_SIZE; i++) PetscCall(PetscViewerASCIIPrintf(viewer2, "%g\n", (double)y2[i][f]));
          PetscCall(PetscViewerASCIIPrintf(viewer2, "        </DataArray>\n"));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer2, "      </PointData>\n"));
      }

      PetscCall(PetscViewerASCIIPrintf(viewer3, "    </Piece>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer2, "    </Piece>\n"));
    }
    PetscCall(PetscFree2(array, array2));
  } else {
    PetscCallMPI(MPI_Send(range, 6, MPIU_INT, 0, tag, comm));
    PetscCallMPI(MPI_Send(x, nn, MPIU_SCALAR, 0, tag, comm));
    PetscCallMPI(MPI_Send(x2, nn2, MPIU_SCALAR, 0, tag, comm));
  }
  PetscCall(VecRestoreArrayRead(X3, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayRead(X2, (const PetscScalar **)&x2));
  PetscCall(PetscViewerASCIIPrintf(viewer3, "  </StructuredGrid>\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer2, "  </StructuredGrid>\n"));

  PetscCall(DMCompositeRestoreAccess(pack, X, &X3, &X2));
  PetscCall(PetscViewerASCIIPrintf(viewer3, "</VTKFile>\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer2, "</VTKFile>\n"));
  PetscCall(PetscViewerDestroy(&viewer3));
  PetscCall(PetscViewerDestroy(&viewer2));
  PetscFunctionReturn(0);
}

static PetscErrorCode THITSMonitor(TS ts, PetscInt step, PetscReal t, Vec X, void *ctx)
{
  THI  thi = (THI)ctx;
  DM   pack;
  char filename3[PETSC_MAX_PATH_LEN], filename2[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(0); /* negative one is used to indicate an interpolated solution */
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "%3" PetscInt_FMT ": t=%g\n", step, (double)t));
  if (thi->monitor_interval && step % thi->monitor_interval) PetscFunctionReturn(0);
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(PetscSNPrintf(filename3, sizeof(filename3), "%s-3d-%03" PetscInt_FMT ".vts", thi->monitor_basename, step));
  PetscCall(PetscSNPrintf(filename2, sizeof(filename2), "%s-2d-%03" PetscInt_FMT ".vts", thi->monitor_basename, step));
  PetscCall(THIDAVecView_VTK_XML(thi, pack, X, filename3, filename2));
  PetscFunctionReturn(0);
}

static PetscErrorCode THICreateDM3d(THI thi, DM *dm3d)
{
  MPI_Comm comm;
  PetscInt M = 3, N = 3, P = 2;
  DM       da;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)thi, &comm));
  PetscOptionsBegin(comm, NULL, "Grid resolution options", "");
  {
    PetscCall(PetscOptionsInt("-M", "Number of elements in x-direction on coarse level", "", M, &M, NULL));
    N = M;
    PetscCall(PetscOptionsInt("-N", "Number of elements in y-direction on coarse level (if different from M)", "", N, &N, NULL));
    PetscCall(PetscOptionsInt("-P", "Number of elements in z-direction on coarse level", "", P, &P, NULL));
  }
  PetscOptionsEnd();
  PetscCall(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, P, N, M, 1, PETSC_DETERMINE, PETSC_DETERMINE, sizeof(Node) / sizeof(PetscScalar), 1, 0, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "x-velocity"));
  PetscCall(DMDASetFieldName(da, 1, "y-velocity"));
  *dm3d = da;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MPI_Comm  comm;
  DM        pack, da3, da2;
  TS        ts;
  THI       thi;
  Vec       X;
  Mat       B;
  PetscInt  i, steps;
  PetscReal ftime;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(THICreate(comm, &thi));
  PetscCall(THICreateDM3d(thi, &da3));
  {
    PetscInt        Mx, My, mx, my, s;
    DMDAStencilType st;
    PetscCall(DMDAGetInfo(da3, 0, 0, &My, &Mx, 0, &my, &mx, 0, &s, 0, 0, 0, &st));
    PetscCall(DMDACreate2d(PetscObjectComm((PetscObject)thi), DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, st, My, Mx, my, mx, sizeof(PrmNode) / sizeof(PetscScalar), s, 0, 0, &da2));
    PetscCall(DMSetUp(da2));
  }

  PetscCall(PetscObjectSetName((PetscObject)da3, "3D_Velocity"));
  PetscCall(DMSetOptionsPrefix(da3, "f3d_"));
  PetscCall(DMDASetFieldName(da3, 0, "u"));
  PetscCall(DMDASetFieldName(da3, 1, "v"));
  PetscCall(PetscObjectSetName((PetscObject)da2, "2D_Fields"));
  PetscCall(DMSetOptionsPrefix(da2, "f2d_"));
  PetscCall(DMDASetFieldName(da2, 0, "b"));
  PetscCall(DMDASetFieldName(da2, 1, "h"));
  PetscCall(DMDASetFieldName(da2, 2, "beta2"));
  PetscCall(DMCompositeCreate(comm, &pack));
  PetscCall(DMCompositeAddDM(pack, da3));
  PetscCall(DMCompositeAddDM(pack, da2));
  PetscCall(DMDestroy(&da3));
  PetscCall(DMDestroy(&da2));
  PetscCall(DMSetUp(pack));
  PetscCall(DMCreateMatrix(pack, &B));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOptionsPrefix(B, "thi_"));

  for (i = 0; i < thi->nlevels; i++) {
    PetscReal Lx = thi->Lx / thi->units->meter, Ly = thi->Ly / thi->units->meter, Lz = thi->Lz / thi->units->meter;
    PetscInt  Mx, My, Mz;
    PetscCall(DMCompositeGetEntries(pack, &da3, &da2));
    PetscCall(DMDAGetInfo(da3, 0, &Mz, &My, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Level %" PetscInt_FMT " domain size (m) %8.2g x %8.2g x %8.2g, num elements %3d x %3d x %3d (%8d), size (m) %g x %g x %g\n", i, Lx, Ly, Lz, Mx, My, Mz, Mx * My * Mz, Lx / Mx, Ly / My, 1000. / (Mz - 1)));
  }

  PetscCall(DMCreateGlobalVector(pack, &X));
  PetscCall(THIInitial(thi, pack, X));

  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSMonitorSet(ts, THITSMonitor, thi, NULL));
  PetscCall(TSSetType(ts, TSTHETA));
  PetscCall(TSSetIFunction(ts, NULL, THIFunction, thi));
  PetscCall(TSSetIJacobian(ts, B, B, THIJacobian, thi));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSSetTimeStep(ts, 1e-3));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Steps %" PetscInt_FMT "  final time %g\n", steps, (double)ftime));

  if (0) PetscCall(THISolveStatistics(thi, ts, 0, "Full"));

  {
    PetscBool flg;
    char      filename[PETSC_MAX_PATH_LEN] = "";
    PetscCall(PetscOptionsGetString(NULL, NULL, "-o", filename, sizeof(filename), &flg));
    if (flg) PetscCall(THIDAVecView_VTK_XML(thi, pack, X, filename, NULL));
  }

  PetscCall(VecDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(DMDestroy(&pack));
  PetscCall(TSDestroy(&ts));
  PetscCall(THIDestroy(&thi));
  PetscCall(PetscFinalize());
  return 0;
}
