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
  #import <PETSc/petsc/private/dmdaimpl.h> /* There is not yet a public interface to manipulate dm->ops */
#else

  #include <petscsnes.h>
  #include <petsc/private/dmdaimpl.h> /* There is not yet a public interface to manipulate dm->ops */
#endif
#include <ctype.h> /* toupper() */

#if defined(__cplusplus) || defined(PETSC_HAVE_WINDOWS_COMPILERS) || defined(__PGI)
  /*  c++ cannot handle  [_restrict_] notation like C does */
  #undef PETSC_RESTRICT
  #define PETSC_RESTRICT
#endif

#if defined __SSE2__
  #include <emmintrin.h>
#endif

/* The SSE2 kernels are only for PetscScalar=double on architectures that support it */
#if !defined NO_SSE2 && !defined PETSC_USE_COMPLEX && !defined PETSC_USE_REAL_SINGLE && !defined PETSC_USE_REAL___FLOAT128 && !defined PETSC_USE_REAL___FP16 && defined __SSE2__
  #define USE_SSE2_KERNELS 1
#else
  #define USE_SSE2_KERNELS 0
#endif

static PetscClassId THI_CLASSID;

typedef enum {
  QUAD_GAUSS,
  QUAD_LOBATTO
} QuadratureType;
static const char                  *QuadratureTypes[] = {"gauss", "lobatto", "QuadratureType", "QUAD_", 0};
PETSC_UNUSED static const PetscReal HexQWeights[8]    = {1, 1, 1, 1, 1, 1, 1, 1};
PETSC_UNUSED static const PetscReal HexQNodes[]       = {-0.57735026918962573, 0.57735026918962573};
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

static void HexComputeGeometry(PetscInt q, PetscReal hx, PetscReal hy, const PetscReal dz[PETSC_RESTRICT], PetscReal phi[PETSC_RESTRICT], PetscReal dphi[PETSC_RESTRICT][3], PetscReal *PETSC_RESTRICT jw)
{
  const PetscReal jac[3][3] = {
    {hx / 2, 0,      0    },
    {0,      hy / 2, 0    },
    {dz[0],  dz[1],  dz[2]}
  };
  const PetscReal ijac[3][3] = {
    {1 / jac[0][0],                        0,                                    0            },
    {0,                                    1 / jac[1][1],                        0            },
    {-jac[2][0] / (jac[0][0] * jac[2][2]), -jac[2][1] / (jac[1][1] * jac[2][2]), 1 / jac[2][2]}
  };
  const PetscReal jdet = jac[0][0] * jac[1][1] * jac[2][2];
  PetscInt        i;

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

typedef struct {
  PetscReal min, max, cmin, cmax;
} PRange;

typedef enum {
  THIASSEMBLY_TRIDIAGONAL,
  THIASSEMBLY_FULL
} THIAssemblyMode;

struct _p_THI {
  PETSCHEADER(int);
  void (*initialize)(THI, PetscReal x, PetscReal y, PrmNode *p);
  PetscInt  zlevels;
  PetscReal Lx, Ly, Lz; /* Model domain */
  PetscReal alpha;      /* Bed angle */
  Units     units;
  PetscReal dirichlet_scale;
  PetscReal ssa_friction_scale;
  PRange    eta;
  PRange    beta2;
  struct {
    PetscReal Bd2, eps, exponent;
  } viscosity;
  struct {
    PetscReal irefgam, eps2, exponent, refvel, epsvel;
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

static PetscErrorCode THIJacobianLocal_3D_Full(DMDALocalInfo *, Node ***, Mat, Mat, THI);
static PetscErrorCode THIJacobianLocal_3D_Tridiagonal(DMDALocalInfo *, Node ***, Mat, Mat, THI);
static PetscErrorCode THIJacobianLocal_2D(DMDALocalInfo *, Node **, Mat, Mat, THI);

static void PrmHexGetZ(const PrmNode pn[], PetscInt k, PetscInt zm, PetscReal zn[])
{
  const PetscScalar zm1 = zm - 1, znl[8] = {pn[0].b + pn[0].h * (PetscScalar)k / zm1,       pn[1].b + pn[1].h * (PetscScalar)k / zm1,       pn[2].b + pn[2].h * (PetscScalar)k / zm1,       pn[3].b + pn[3].h * (PetscScalar)k / zm1,
                                            pn[0].b + pn[0].h * (PetscScalar)(k + 1) / zm1, pn[1].b + pn[1].h * (PetscScalar)(k + 1) / zm1, pn[2].b + pn[2].h * (PetscScalar)(k + 1) / zm1, pn[3].b + pn[3].h * (PetscScalar)(k + 1) / zm1};
  PetscInt          i;
  for (i = 0; i < 8; i++) zn[i] = PetscRealPart(znl[i]);
}

/* Tests A and C are from the ISMIP-HOM paper (Pattyn et al. 2008) */
static void THIInitialize_HOM_A(THI thi, PetscReal x, PetscReal y, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x * PetscSinReal(thi->alpha);

  p->b     = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x * 2 * PETSC_PI / thi->Lx) * PetscSinReal(y * 2 * PETSC_PI / thi->Ly);
  p->h     = s - p->b;
  p->beta2 = 1e30;
}

static void THIInitialize_HOM_C(THI thi, PetscReal x, PetscReal y, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal s     = -x * PetscSinReal(thi->alpha);

  p->b = s - 1000 * units->meter;
  p->h = s - p->b;
  /* tau_b = beta2 v   is a stress (Pa) */
  p->beta2 = 1000 * (1 + PetscSinReal(x * 2 * PETSC_PI / thi->Lx) * PetscSinReal(y * 2 * PETSC_PI / thi->Ly)) * units->Pascal * units->year / units->meter;
}

/* These are just toys */

/* Same bed as test A, free slip everywhere except for a discontinuous jump to a circular sticky region in the middle. */
static void THIInitialize_HOM_X(THI thi, PetscReal xx, PetscReal yy, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x = xx * 2 * PETSC_PI / thi->Lx - PETSC_PI, y = yy * 2 * PETSC_PI / thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x * x + y * y), s = -x * PetscSinReal(thi->alpha);
  p->b     = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h     = s - p->b;
  p->beta2 = 1000 * (r < 1 ? 2 : 0) * units->Pascal * units->year / units->meter;
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
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16 * r)) / PetscSqrtReal(1e-2 + 16 * r) * PetscCosReal(x * 3 / 2) * PetscCosReal(y * 3 / 2)) * units->Pascal * units->year / units->meter;
}

/* Same bed as A, smoothly varying slipperiness, similar to MATLAB's "sombrero" (uncorrelated with bathymetry) */
static void THIInitialize_HOM_Z(THI thi, PetscReal xx, PetscReal yy, PrmNode *p)
{
  Units     units = thi->units;
  PetscReal x = xx * 2 * PETSC_PI / thi->Lx - PETSC_PI, y = yy * 2 * PETSC_PI / thi->Ly - PETSC_PI; /* [-pi,pi] */
  PetscReal r = PetscSqrtReal(x * x + y * y), s = -x * PetscSinReal(thi->alpha);

  p->b     = s - 1000 * units->meter + 500 * units->meter * PetscSinReal(x + PETSC_PI) * PetscSinReal(y + PETSC_PI);
  p->h     = s - p->b;
  p->beta2 = 1000 * (1. + PetscSinReal(PetscSqrtReal(16 * r)) / PetscSqrtReal(1e-2 + 16 * r) * PetscCosReal(x * 3 / 2) * PetscCosReal(y * 3 / 2)) * units->Pascal * units->year / units->meter;
}

static void THIFriction(THI thi, PetscReal rbeta2, PetscReal gam, PetscReal *beta2, PetscReal *dbeta2)
{
  if (thi->friction.irefgam == 0) {
    Units units           = thi->units;
    thi->friction.irefgam = 1. / (0.5 * PetscSqr(thi->friction.refvel * units->meter / units->year));
    thi->friction.eps2    = 0.5 * PetscSqr(thi->friction.epsvel * units->meter / units->year) * thi->friction.irefgam;
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
    const PetscReal n       = 3.,                                                     /* Glen exponent */
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
  if (!*thi) PetscFunctionReturn(0);
  if (--((PetscObject)(*thi))->refct > 0) {
    *thi = 0;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscFree((*thi)->units));
  PetscCall(PetscFree((*thi)->mattype));
  PetscCall(PetscHeaderDestroy(thi));
  PetscFunctionReturn(0);
}

static PetscErrorCode THICreate(MPI_Comm comm, THI *inthi)
{
  static PetscBool registered = PETSC_FALSE;
  THI              thi;
  Units            units;

  PetscFunctionBeginUser;
  *inthi = 0;
  if (!registered) {
    PetscCall(PetscClassIdRegister("Toy Hydrostatic Ice", &THI_CLASSID));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscHeaderCreate(thi, THI_CLASSID, "THI", "Toy Hydrostatic Ice", "", comm, THIDestroy, 0));

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
  units->year   = 31556926. * units->second; /* seconds per year */

  thi->Lx              = 10.e3;
  thi->Ly              = 10.e3;
  thi->Lz              = 1000;
  thi->dirichlet_scale = 1;
  thi->verbose         = PETSC_FALSE;

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

    thi->friction.refvel = 100.;
    thi->friction.epsvel = 1.;

    PetscCall(PetscOptionsReal("-thi_friction_refvel", "Reference velocity for sliding", "", thi->friction.refvel, &thi->friction.refvel, NULL));
    PetscCall(PetscOptionsReal("-thi_friction_epsvel", "Regularization velocity for sliding", "", thi->friction.epsvel, &thi->friction.epsvel, NULL));
    PetscCall(PetscOptionsReal("-thi_friction_m", "Friction exponent, 0=Coulomb, 1=Navier", "", m, &m, NULL));

    thi->friction.exponent = (m - 1) / 2;

    PetscCall(PetscOptionsReal("-thi_dirichlet_scale", "Scale Dirichlet boundary conditions by this factor", "", thi->dirichlet_scale, &thi->dirichlet_scale, NULL));
    PetscCall(PetscOptionsReal("-thi_ssa_friction_scale", "Scale slip boundary conditions by this factor in SSA (2D) assembly", "", thi->ssa_friction_scale, &thi->ssa_friction_scale, NULL));
    PetscCall(PetscOptionsBool("-thi_coarse2d", "Use a 2D coarse space corresponding to SSA", "", thi->coarse2d, &thi->coarse2d, NULL));
    PetscCall(PetscOptionsBool("-thi_tridiagonal", "Assemble a tridiagonal system (column coupling only) on the finest level", "", thi->tridiagonal, &thi->tridiagonal, NULL));
    PetscCall(PetscOptionsFList("-thi_mat_type", "Matrix type", "MatSetType", MatList, mtype, (char *)mtype, sizeof(mtype), NULL));
    PetscCall(PetscStrallocpy(mtype, (char **)&thi->mattype));
    PetscCall(PetscOptionsBool("-thi_verbose", "Enable verbose output (like matrix sizes and statistics)", "", thi->verbose, &thi->verbose, NULL));
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
    PetscReal u = 1000 * units->meter / (3e7 * units->second), gradu = u / (100 * units->meter), eta, deta, rho = 910 * units->kilogram / PetscPowReal(units->meter, 3), grav = 9.81 * units->meter / PetscSqr(units->second),
              driving = rho * grav * PetscSinReal(thi->alpha) * 1000 * units->meter;
    THIViscosity(thi, 0.5 * gradu * gradu, &eta, &deta);
    thi->rhog = rho * grav;
    if (thi->verbose) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Units: meter %8.2g  second %8.2g  kg %8.2g  Pa %8.2g\n", (double)units->meter, (double)units->second, (double)units->kilogram, (double)units->Pascal));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Domain (%6.2g,%6.2g,%6.2g), pressure %8.2g, driving stress %8.2g\n", (double)thi->Lx, (double)thi->Ly, (double)thi->Lz, (double)(rho * grav * 1e3 * units->meter), (double)driving));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Large velocity 1km/a %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n", (double)u, (double)gradu, (double)eta, (double)(2 * eta * gradu), (double)(2 * eta * gradu / driving)));
      THIViscosity(thi, 0.5 * PetscSqr(1e-3 * gradu), &eta, &deta);
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Small velocity 1m/a  %8.2g, velocity gradient %8.2g, eta %8.2g, stress %8.2g, ratio %8.2g\n", (double)(1e-3 * u), (double)(1e-3 * gradu), (double)eta, (double)(2 * eta * 1e-3 * gradu), (double)(2 * eta * 1e-3 * gradu / driving)));
    }
  }

  *inthi = thi;
  PetscFunctionReturn(0);
}

static PetscErrorCode THIInitializePrm(THI thi, DM da2prm, Vec prm)
{
  PrmNode **p;
  PetscInt  i, j, xs, xm, ys, ym, mx, my;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetGhostCorners(da2prm, &ys, &xs, 0, &ym, &xm, 0));
  PetscCall(DMDAGetInfo(da2prm, 0, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAVecGetArray(da2prm, prm, &p));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscReal xx = thi->Lx * i / mx, yy = thi->Ly * j / my;
      thi->initialize(thi, xx, yy, &p[i][j]);
    }
  }
  PetscCall(DMDAVecRestoreArray(da2prm, prm, &p));
  PetscFunctionReturn(0);
}

static PetscErrorCode THISetUpDM(THI thi, DM dm)
{
  PetscInt        refinelevel, coarsenlevel, level, dim, Mx, My, Mz, mx, my, s;
  DMDAStencilType st;
  DM              da2prm;
  Vec             X;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(dm, &dim, &Mz, &My, &Mx, 0, &my, &mx, 0, &s, 0, 0, 0, &st));
  if (dim == 2) PetscCall(DMDAGetInfo(dm, &dim, &My, &Mx, 0, &my, &mx, 0, 0, &s, 0, 0, 0, &st));
  PetscCall(DMGetRefineLevel(dm, &refinelevel));
  PetscCall(DMGetCoarsenLevel(dm, &coarsenlevel));
  level = refinelevel - coarsenlevel;
  PetscCall(DMDACreate2d(PetscObjectComm((PetscObject)thi), DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, st, My, Mx, my, mx, sizeof(PrmNode) / sizeof(PetscScalar), s, 0, 0, &da2prm));
  PetscCall(DMSetUp(da2prm));
  PetscCall(DMCreateLocalVector(da2prm, &X));
  {
    PetscReal Lx = thi->Lx / thi->units->meter, Ly = thi->Ly / thi->units->meter, Lz = thi->Lz / thi->units->meter;
    if (dim == 2) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Level %" PetscInt_FMT " domain size (m) %8.2g x %8.2g, num elements %" PetscInt_FMT " x %" PetscInt_FMT " (%" PetscInt_FMT "), size (m) %g x %g\n", level, (double)Lx, (double)Ly, Mx, My, Mx * My, (double)(Lx / Mx), (double)(Ly / My)));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)thi), "Level %" PetscInt_FMT " domain size (m) %8.2g x %8.2g x %8.2g, num elements %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT " (%" PetscInt_FMT "), size (m) %g x %g x %g\n", level, (double)Lx, (double)Ly, (double)Lz, Mx, My, Mz, Mx * My * Mz, (double)(Lx / Mx), (double)(Ly / My), (double)(1000. / (Mz - 1))));
    }
  }
  PetscCall(THIInitializePrm(thi, da2prm, X));
  if (thi->tridiagonal) { /* Reset coarse Jacobian evaluation */
    PetscCall(DMDASNESSetJacobianLocal(dm, (DMDASNESJacobian)THIJacobianLocal_3D_Full, thi));
  }
  if (thi->coarse2d) PetscCall(DMDASNESSetJacobianLocal(dm, (DMDASNESJacobian)THIJacobianLocal_2D, thi));
  PetscCall(PetscObjectCompose((PetscObject)dm, "DMDA2Prm", (PetscObject)da2prm));
  PetscCall(PetscObjectCompose((PetscObject)dm, "DMDA2Prm_Vec", (PetscObject)X));
  PetscCall(DMDestroy(&da2prm));
  PetscCall(VecDestroy(&X));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_THI(DM dmf, DM dmc, void *ctx)
{
  THI      thi = (THI)ctx;
  PetscInt rlevel, clevel;

  PetscFunctionBeginUser;
  PetscCall(THISetUpDM(thi, dmc));
  PetscCall(DMGetRefineLevel(dmc, &rlevel));
  PetscCall(DMGetCoarsenLevel(dmc, &clevel));
  if (rlevel - clevel == 0) PetscCall(DMSetMatType(dmc, MATAIJ));
  PetscCall(DMCoarsenHookAdd(dmc, DMCoarsenHook_THI, NULL, thi));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefineHook_THI(DM dmc, DM dmf, void *ctx)
{
  THI thi = (THI)ctx;

  PetscFunctionBeginUser;
  PetscCall(THISetUpDM(thi, dmf));
  PetscCall(DMSetMatType(dmf, thi->mattype));
  PetscCall(DMRefineHookAdd(dmf, DMRefineHook_THI, NULL, thi));
  /* With grid sequencing, a formerly-refined DM will later be coarsened by PCSetUp_MG */
  PetscCall(DMCoarsenHookAdd(dmf, DMCoarsenHook_THI, NULL, thi));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIDAGetPrm(DM da, PrmNode ***prm)
{
  DM  da2prm;
  Vec X;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)da, "DMDA2Prm", (PetscObject *)&da2prm));
  PetscCheck(da2prm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No DMDA2Prm composed with given DMDA");
  PetscCall(PetscObjectQuery((PetscObject)da, "DMDA2Prm_Vec", (PetscObject *)&X));
  PetscCheck(X, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No DMDA2Prm_Vec composed with given DMDA");
  PetscCall(DMDAVecGetArray(da2prm, X, prm));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIDARestorePrm(DM da, PrmNode ***prm)
{
  DM  da2prm;
  Vec X;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)da, "DMDA2Prm", (PetscObject *)&da2prm));
  PetscCheck(da2prm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No DMDA2Prm composed with given DMDA");
  PetscCall(PetscObjectQuery((PetscObject)da, "DMDA2Prm_Vec", (PetscObject *)&X));
  PetscCheck(X, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No DMDA2Prm_Vec composed with given DMDA");
  PetscCall(DMDAVecRestoreArray(da2prm, X, prm));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIInitial(SNES snes, Vec X, void *ctx)
{
  THI       thi;
  PetscInt  i, j, k, xs, xm, ys, ym, zs, zm, mx, my;
  PetscReal hx, hy;
  PrmNode **prm;
  Node   ***x;
  DM        da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetApplicationContext(da, &thi));
  PetscCall(DMDAGetInfo(da, 0, 0, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &zs, &ys, &xs, &zm, &ym, &xm));
  PetscCall(DMDAVecGetArray(da, X, &x));
  PetscCall(THIDAGetPrm(da, &prm));
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
  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscCall(THIDARestorePrm(da, &prm));
  PetscFunctionReturn(0);
}

static void PointwiseNonlinearity(THI thi, const Node n[PETSC_RESTRICT], const PetscReal phi[PETSC_RESTRICT], PetscReal dphi[PETSC_RESTRICT][3], PetscScalar *PETSC_RESTRICT u, PetscScalar *PETSC_RESTRICT v, PetscScalar du[PETSC_RESTRICT], PetscScalar dv[PETSC_RESTRICT], PetscReal *eta, PetscReal *deta)
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
  gam = PetscSqr(du[0]) + PetscSqr(dv[1]) + du[0] * dv[1] + 0.25 * PetscSqr(du[1] + dv[0]) + 0.25 * PetscSqr(du[2]) + 0.25 * PetscSqr(dv[2]);
  THIViscosity(thi, PetscRealPart(gam), eta, deta);
}

static void PointwiseNonlinearity2D(THI thi, Node n[], PetscReal phi[], PetscReal dphi[4][2], PetscScalar *u, PetscScalar *v, PetscScalar du[], PetscScalar dv[], PetscReal *eta, PetscReal *deta)
{
  PetscInt    l, ll;
  PetscScalar gam;

  du[0] = du[1] = 0;
  dv[0] = dv[1] = 0;
  *u            = 0;
  *v            = 0;
  for (l = 0; l < 4; l++) {
    *u += phi[l] * n[l].u;
    *v += phi[l] * n[l].v;
    for (ll = 0; ll < 2; ll++) {
      du[ll] += dphi[l][ll] * n[l].u;
      dv[ll] += dphi[l][ll] * n[l].v;
    }
  }
  gam = PetscSqr(du[0]) + PetscSqr(dv[1]) + du[0] * dv[1] + 0.25 * PetscSqr(du[1] + dv[0]);
  THIViscosity(thi, PetscRealPart(gam), eta, deta);
}

static PetscErrorCode THIFunctionLocal(DMDALocalInfo *info, Node ***x, Node ***f, THI thi)
{
  PetscInt  xs, ys, xm, ym, zm, i, j, k, q, l;
  PetscReal hx, hy, etamin, etamax, beta2min, beta2max;
  PrmNode **prm;

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

  PetscCall(THIDAGetPrm(info->da, &prm));

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm, i, j, pn);
      for (k = 0; k < zm - 1; k++) {
        PetscInt  ls = 0;
        Node      n[8], *fn[8];
        PetscReal zn[8], etabase = 0;
        PrmHexGetZ(pn, k, zm, zn);
        HexExtract(x, i, j, k, n);
        HexExtractRef(f, i, j, k, fn);
        if (thi->no_slip && k == 0) {
          for (l = 0; l < 4; l++) n[l].u = n[l].v = 0;
          /* The first 4 basis functions lie on the bottom layer, so their contribution is exactly 0, hence we can skip them */
          ls = 4;
        }
        for (q = 0; q < 8; q++) {
          PetscReal   dz[3], phi[8], dphi[8][3], jw, eta, deta;
          PetscScalar du[3], dv[3], u, v;
          HexGrad(HexQDeriv[q], zn, dz);
          HexComputeGeometry(q, hx, hy, dz, phi, dphi, &jw);
          PointwiseNonlinearity(thi, n, phi, dphi, &u, &v, du, dv, &eta, &deta);
          jw /= thi->rhog; /* scales residuals to be O(1) */
          if (q == 0) etabase = eta;
          RangeUpdate(&etamin, &etamax, eta);
          for (l = ls; l < 8; l++) { /* test functions */
            const PetscReal ds[2] = {-PetscSinReal(thi->alpha), 0};
            const PetscReal pp = phi[l], *dp = dphi[l];
            fn[l]->u += dp[0] * jw * eta * (4. * du[0] + 2. * dv[1]) + dp[1] * jw * eta * (du[1] + dv[0]) + dp[2] * jw * eta * du[2] + pp * jw * thi->rhog * ds[0];
            fn[l]->v += dp[1] * jw * eta * (2. * du[0] + 4. * dv[1]) + dp[0] * jw * eta * (du[1] + dv[0]) + dp[2] * jw * eta * dv[2] + pp * jw * thi->rhog * ds[1];
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
          } else { /* Integrate over bottom face to apply boundary condition */
            for (q = 0; q < 4; q++) {
              const PetscReal jw = 0.25 * hx * hy / thi->rhog, *phi = QuadQInterp[q];
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

  PetscCall(THIDARestorePrm(info->da, &prm));

  PetscCall(PRangeMinMax(&thi->eta, etamin, etamax));
  PetscCall(PRangeMinMax(&thi->beta2, beta2min, beta2max));
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "Matrix dim %" PetscInt_FMT " norm %8.2e (0,0) %8.2e  (2,2) %8.2e %8.2e <= eta <= %8.2e %8.2e <= beta2 <= %8.2e\n", m, (double)nrm, (double)PetscRealPart(val0), (double)PetscRealPart(val2),
                                     (double)thi->eta.cmin, (double)thi->eta.cmax, (double)thi->beta2.cmin, (double)thi->beta2.cmax));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THISurfaceStatistics(DM da, Vec X, PetscReal *min, PetscReal *max, PetscReal *mean)
{
  Node     ***x;
  PetscInt    i, j, xs, ys, zs, xm, ym, zm, mx, my, mz;
  PetscReal   umin = 1e100, umax = -1e100;
  PetscScalar usum = 0.0, gusum;

  PetscFunctionBeginUser;
  *min = *max = *mean = 0;
  PetscCall(DMDAGetInfo(da, 0, &mz, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &zs, &ys, &xs, &zm, &ym, &xm));
  PetscCheck(zs == 0 && zm == mz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected decomposition");
  PetscCall(DMDAVecGetArray(da, X, &x));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscReal u = PetscRealPart(x[i][j][zm - 1].u);
      RangeUpdate(&umin, &umax, u);
      usum += u;
    }
  }
  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscCallMPI(MPI_Allreduce(&umin, min, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)da)));
  PetscCallMPI(MPI_Allreduce(&umax, max, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)da)));
  PetscCallMPI(MPI_Allreduce(&usum, &gusum, 1, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)da)));
  *mean = PetscRealPart(gusum) / (mx * my);
  PetscFunctionReturn(0);
}

static PetscErrorCode THISolveStatistics(THI thi, SNES snes, PetscInt coarsened, const char name[])
{
  MPI_Comm comm;
  Vec      X;
  DM       dm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)thi, &comm));
  PetscCall(SNESGetSolution(snes, &X));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(PetscPrintf(comm, "Solution statistics after solve: %s\n", name));
  {
    PetscInt            its, lits;
    SNESConvergedReason reason;
    PetscCall(SNESGetIterationNumber(snes, &its));
    PetscCall(SNESGetConvergedReason(snes, &reason));
    PetscCall(SNESGetLinearSolveIterations(snes, &lits));
    PetscCall(PetscPrintf(comm, "%s: Number of SNES iterations = %" PetscInt_FMT ", total linear iterations = %" PetscInt_FMT "\n", SNESConvergedReasons[reason], its, lits));
  }
  {
    PetscReal          nrm2, tmin[3] = {1e100, 1e100, 1e100}, tmax[3] = {-1e100, -1e100, -1e100}, min[3], max[3];
    PetscInt           i, j, m;
    const PetscScalar *x;
    PetscCall(VecNorm(X, NORM_2, &nrm2));
    PetscCall(VecGetLocalSize(X, &m));
    PetscCall(VecGetArrayRead(X, &x));
    for (i = 0; i < m; i += 2) {
      PetscReal u = PetscRealPart(x[i]), v = PetscRealPart(x[i + 1]), c = PetscSqrtReal(u * u + v * v);
      tmin[0] = PetscMin(u, tmin[0]);
      tmin[1] = PetscMin(v, tmin[1]);
      tmin[2] = PetscMin(c, tmin[2]);
      tmax[0] = PetscMax(u, tmax[0]);
      tmax[1] = PetscMax(v, tmax[1]);
      tmax[2] = PetscMax(c, tmax[2]);
    }
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCallMPI(MPI_Allreduce(tmin, min, 3, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)thi)));
    PetscCallMPI(MPI_Allreduce(tmax, max, 3, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)thi)));
    /* Dimensionalize to meters/year */
    nrm2 *= thi->units->year / thi->units->meter;
    for (j = 0; j < 3; j++) {
      min[j] *= thi->units->year / thi->units->meter;
      max[j] *= thi->units->year / thi->units->meter;
    }
    if (min[0] == 0.0) min[0] = 0.0;
    PetscCall(PetscPrintf(comm, "|X|_2 %g   %g <= u <=  %g   %g <= v <=  %g   %g <= c <=  %g \n", (double)nrm2, (double)min[0], (double)max[0], (double)min[1], (double)max[1], (double)min[2], (double)max[2]));
    {
      PetscReal umin, umax, umean;
      PetscCall(THISurfaceStatistics(dm, X, &umin, &umax, &umean));
      umin *= thi->units->year / thi->units->meter;
      umax *= thi->units->year / thi->units->meter;
      umean *= thi->units->year / thi->units->meter;
      PetscCall(PetscPrintf(comm, "Surface statistics: u in [%12.6e, %12.6e] mean %12.6e\n", (double)umin, (double)umax, (double)umean));
    }
    /* These values stay nondimensional */
    PetscCall(PetscPrintf(comm, "Global eta range   %g to %g converged range %g to %g\n", (double)thi->eta.min, (double)thi->eta.max, (double)thi->eta.cmin, (double)thi->eta.cmax));
    PetscCall(PetscPrintf(comm, "Global beta2 range %g to %g converged range %g to %g\n", (double)thi->beta2.min, (double)thi->beta2.max, (double)thi->beta2.cmin, (double)thi->beta2.cmax));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobianLocal_2D(DMDALocalInfo *info, Node **x, Mat J, Mat B, THI thi)
{
  PetscInt  xs, ys, xm, ym, i, j, q, l, ll;
  PetscReal hx, hy;
  PrmNode **prm;

  PetscFunctionBeginUser;
  xs = info->ys;
  ys = info->xs;
  xm = info->ym;
  ym = info->xm;
  hx = thi->Lx / info->my;
  hy = thi->Ly / info->mx;

  PetscCall(MatZeroEntries(B));
  PetscCall(THIDAGetPrm(info->da, &prm));

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      Node        n[4];
      PrmNode     pn[4];
      PetscScalar Ke[4 * 2][4 * 2];
      QuadExtract(prm, i, j, pn);
      QuadExtract(x, i, j, n);
      PetscCall(PetscMemzero(Ke, sizeof(Ke)));
      for (q = 0; q < 4; q++) {
        PetscReal   phi[4], dphi[4][2], jw, eta, deta, beta2, dbeta2;
        PetscScalar u, v, du[2], dv[2], h = 0, rbeta2 = 0;
        for (l = 0; l < 4; l++) {
          phi[l]     = QuadQInterp[q][l];
          dphi[l][0] = QuadQDeriv[q][l][0] * 2. / hx;
          dphi[l][1] = QuadQDeriv[q][l][1] * 2. / hy;
          h += phi[l] * pn[l].h;
          rbeta2 += phi[l] * pn[l].beta2;
        }
        jw = 0.25 * hx * hy / thi->rhog; /* rhog is only scaling */
        PointwiseNonlinearity2D(thi, n, phi, dphi, &u, &v, du, dv, &eta, &deta);
        THIFriction(thi, PetscRealPart(rbeta2), PetscRealPart(u * u + v * v) / 2, &beta2, &dbeta2);
        for (l = 0; l < 4; l++) {
          const PetscReal pp = phi[l], *dp = dphi[l];
          for (ll = 0; ll < 4; ll++) {
            const PetscReal ppl = phi[ll], *dpl = dphi[ll];
            PetscScalar     dgdu, dgdv;
            dgdu = 2. * du[0] * dpl[0] + dv[1] * dpl[0] + 0.5 * (du[1] + dv[0]) * dpl[1];
            dgdv = 2. * dv[1] * dpl[1] + du[0] * dpl[1] + 0.5 * (du[1] + dv[0]) * dpl[0];
            /* Picard part */
            Ke[l * 2 + 0][ll * 2 + 0] += dp[0] * jw * eta * 4. * dpl[0] + dp[1] * jw * eta * dpl[1] + pp * jw * (beta2 / h) * ppl * thi->ssa_friction_scale;
            Ke[l * 2 + 0][ll * 2 + 1] += dp[0] * jw * eta * 2. * dpl[1] + dp[1] * jw * eta * dpl[0];
            Ke[l * 2 + 1][ll * 2 + 0] += dp[1] * jw * eta * 2. * dpl[0] + dp[0] * jw * eta * dpl[1];
            Ke[l * 2 + 1][ll * 2 + 1] += dp[1] * jw * eta * 4. * dpl[1] + dp[0] * jw * eta * dpl[0] + pp * jw * (beta2 / h) * ppl * thi->ssa_friction_scale;
            /* extra Newton terms */
            Ke[l * 2 + 0][ll * 2 + 0] += dp[0] * jw * deta * dgdu * (4. * du[0] + 2. * dv[1]) + dp[1] * jw * deta * dgdu * (du[1] + dv[0]) + pp * jw * (dbeta2 / h) * u * u * ppl * thi->ssa_friction_scale;
            Ke[l * 2 + 0][ll * 2 + 1] += dp[0] * jw * deta * dgdv * (4. * du[0] + 2. * dv[1]) + dp[1] * jw * deta * dgdv * (du[1] + dv[0]) + pp * jw * (dbeta2 / h) * u * v * ppl * thi->ssa_friction_scale;
            Ke[l * 2 + 1][ll * 2 + 0] += dp[1] * jw * deta * dgdu * (4. * dv[1] + 2. * du[0]) + dp[0] * jw * deta * dgdu * (du[1] + dv[0]) + pp * jw * (dbeta2 / h) * v * u * ppl * thi->ssa_friction_scale;
            Ke[l * 2 + 1][ll * 2 + 1] += dp[1] * jw * deta * dgdv * (4. * dv[1] + 2. * du[0]) + dp[0] * jw * deta * dgdv * (du[1] + dv[0]) + pp * jw * (dbeta2 / h) * v * v * ppl * thi->ssa_friction_scale;
          }
        }
      }
      {
        const MatStencil rc[4] = {
          {0, i,     j,     0},
          {0, i + 1, j,     0},
          {0, i + 1, j + 1, 0},
          {0, i,     j + 1, 0}
        };
        PetscCall(MatSetValuesBlockedStencil(B, 4, rc, 4, rc, &Ke[0][0], ADD_VALUES));
      }
    }
  }
  PetscCall(THIDARestorePrm(info->da, &prm));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE));
  if (thi->verbose) PetscCall(THIMatrixStatistics(thi, B, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobianLocal_3D(DMDALocalInfo *info, Node ***x, Mat B, THI thi, THIAssemblyMode amode)
{
  PetscInt  xs, ys, xm, ym, zm, i, j, k, q, l, ll;
  PetscReal hx, hy;
  PrmNode **prm;

  PetscFunctionBeginUser;
  xs = info->zs;
  ys = info->ys;
  xm = info->zm;
  ym = info->ym;
  zm = info->xm;
  hx = thi->Lx / info->mz;
  hy = thi->Ly / info->my;

  PetscCall(MatZeroEntries(B));
  PetscCall(MatSetOption(B, MAT_SUBSET_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(THIDAGetPrm(info->da, &prm));

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PrmNode pn[4];
      QuadExtract(prm, i, j, pn);
      for (k = 0; k < zm - 1; k++) {
        Node        n[8];
        PetscReal   zn[8], etabase = 0;
        PetscScalar Ke[8 * 2][8 * 2];
        PetscInt    ls = 0;

        PrmHexGetZ(pn, k, zm, zn);
        HexExtract(x, i, j, k, n);
        PetscCall(PetscMemzero(Ke, sizeof(Ke)));
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
            const PetscReal *PETSC_RESTRICT dp = dphi[l];
#if USE_SSE2_KERNELS
            /* gcc (up to my 4.5 snapshot) is really bad at hoisting intrinsics so we do it manually */
            __m128d p4 = _mm_set1_pd(4), p2 = _mm_set1_pd(2), p05 = _mm_set1_pd(0.5), p42 = _mm_setr_pd(4, 2), p24 = _mm_shuffle_pd(p42, p42, _MM_SHUFFLE2(0, 1)), du0 = _mm_set1_pd(du[0]), du1 = _mm_set1_pd(du[1]), du2 = _mm_set1_pd(du[2]), dv0 = _mm_set1_pd(dv[0]), dv1 = _mm_set1_pd(dv[1]), dv2 = _mm_set1_pd(dv[2]), jweta = _mm_set1_pd(jw * eta), jwdeta = _mm_set1_pd(jw * deta), dp0 = _mm_set1_pd(dp[0]), dp1 = _mm_set1_pd(dp[1]), dp2 = _mm_set1_pd(dp[2]), dp0jweta = _mm_mul_pd(dp0, jweta), dp1jweta = _mm_mul_pd(dp1, jweta), dp2jweta = _mm_mul_pd(dp2, jweta), p4du0p2dv1 = _mm_add_pd(_mm_mul_pd(p4, du0), _mm_mul_pd(p2, dv1)), /* 4 du0 + 2 dv1 */
              p4dv1p2du0 = _mm_add_pd(_mm_mul_pd(p4, dv1), _mm_mul_pd(p2, du0)), /* 4 dv1 + 2 du0 */
              pdu2dv2    = _mm_unpacklo_pd(du2, dv2),                            /* [du2, dv2] */
              du1pdv0    = _mm_add_pd(du1, dv0),                                 /* du1 + dv0 */
              t1         = _mm_mul_pd(dp0, p4du0p2dv1),                          /* dp0 (4 du0 + 2 dv1) */
              t2         = _mm_mul_pd(dp1, p4dv1p2du0);                          /* dp1 (4 dv1 + 2 du0) */

#endif
#if defined COMPUTE_LOWER_TRIANGULAR      /* The element matrices are always symmetric so computing the lower-triangular part is not necessary */
            for (ll = ls; ll < 8; ll++) { /* trial functions */
#else
            for (ll = l; ll < 8; ll++) {
#endif
              const PetscReal *PETSC_RESTRICT dpl = dphi[ll];
              if (amode == THIASSEMBLY_TRIDIAGONAL && (l - ll) % 4) continue; /* these entries would not be inserted */
#if !USE_SSE2_KERNELS
              /* The analytic Jacobian in nice, easy-to-read form */
              {
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
              }
#else
              /* This SSE2 code is an exact replica of above, but uses explicit packed instructions for some speed
              * benefit.  On my hardware, these intrinsics are almost twice as fast as above, reducing total assembly cost
              * by 25 to 30 percent. */
              {
                __m128d keu = _mm_loadu_pd(&Ke[l * 2 + 0][ll * 2 + 0]), kev = _mm_loadu_pd(&Ke[l * 2 + 1][ll * 2 + 0]), dpl01 = _mm_loadu_pd(&dpl[0]), dpl10 = _mm_shuffle_pd(dpl01, dpl01, _MM_SHUFFLE2(0, 1)), dpl2 = _mm_set_sd(dpl[2]), t0, t3, pdgduv;
                keu    = _mm_add_pd(keu, _mm_add_pd(_mm_mul_pd(_mm_mul_pd(dp0jweta, p42), dpl01), _mm_add_pd(_mm_mul_pd(dp1jweta, dpl10), _mm_mul_pd(dp2jweta, dpl2))));
                kev    = _mm_add_pd(kev, _mm_add_pd(_mm_mul_pd(_mm_mul_pd(dp1jweta, p24), dpl01), _mm_add_pd(_mm_mul_pd(dp0jweta, dpl10), _mm_mul_pd(dp2jweta, _mm_shuffle_pd(dpl2, dpl2, _MM_SHUFFLE2(0, 1))))));
                pdgduv = _mm_mul_pd(p05, _mm_add_pd(_mm_add_pd(_mm_mul_pd(p42, _mm_mul_pd(du0, dpl01)), _mm_mul_pd(p24, _mm_mul_pd(dv1, dpl01))), _mm_add_pd(_mm_mul_pd(du1pdv0, dpl10), _mm_mul_pd(pdu2dv2, _mm_set1_pd(dpl[2]))))); /* [dgdu, dgdv] */
                t0     = _mm_mul_pd(jwdeta, pdgduv); /* jw deta [dgdu, dgdv] */
                t3     = _mm_mul_pd(t0, du1pdv0);    /* t0 (du1 + dv0) */
                _mm_storeu_pd(&Ke[l * 2 + 0][ll * 2 + 0], _mm_add_pd(keu, _mm_add_pd(_mm_mul_pd(t1, t0), _mm_add_pd(_mm_mul_pd(dp1, t3), _mm_mul_pd(t0, _mm_mul_pd(dp2, du2))))));
                _mm_storeu_pd(&Ke[l * 2 + 1][ll * 2 + 0], _mm_add_pd(kev, _mm_add_pd(_mm_mul_pd(t2, t0), _mm_add_pd(_mm_mul_pd(dp0, t3), _mm_mul_pd(t0, _mm_mul_pd(dp2, dv2))))));
              }
#endif
            }
          }
        }
        if (k == 0) { /* on a bottom face */
          if (thi->no_slip) {
            const PetscReal   hz    = PetscRealPart(pn[0].h) / (zm - 1);
            const PetscScalar diagu = 2 * etabase / thi->rhog * (hx * hy / hz + hx * hz / hy + 4 * hy * hz / hx), diagv = 2 * etabase / thi->rhog * (hx * hy / hz + 4 * hx * hz / hy + hy * hz / hx);
            Ke[0][0] = thi->dirichlet_scale * diagu;
            Ke[1][1] = thi->dirichlet_scale * diagv;
          } else {
            for (q = 0; q < 4; q++) {
              const PetscReal jw = 0.25 * hx * hy / thi->rhog, *phi = QuadQInterp[q];
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
          const MatStencil rc[8] = {
            {i,     j,     k,     0},
            {i + 1, j,     k,     0},
            {i + 1, j + 1, k,     0},
            {i,     j + 1, k,     0},
            {i,     j,     k + 1, 0},
            {i + 1, j,     k + 1, 0},
            {i + 1, j + 1, k + 1, 0},
            {i,     j + 1, k + 1, 0}
          };
          if (amode == THIASSEMBLY_TRIDIAGONAL) {
            for (l = 0; l < 4; l++) { /* Copy out each of the blocks, discarding horizontal coupling */
              const PetscInt   l4     = l + 4;
              const MatStencil rcl[2] = {
                {rc[l].k,  rc[l].j,  rc[l].i,  0},
                {rc[l4].k, rc[l4].j, rc[l4].i, 0}
              };
#if defined COMPUTE_LOWER_TRIANGULAR
              const PetscScalar Kel[4][4] = {
                {Ke[2 * l + 0][2 * l + 0],  Ke[2 * l + 0][2 * l + 1],  Ke[2 * l + 0][2 * l4 + 0],  Ke[2 * l + 0][2 * l4 + 1] },
                {Ke[2 * l + 1][2 * l + 0],  Ke[2 * l + 1][2 * l + 1],  Ke[2 * l + 1][2 * l4 + 0],  Ke[2 * l + 1][2 * l4 + 1] },
                {Ke[2 * l4 + 0][2 * l + 0], Ke[2 * l4 + 0][2 * l + 1], Ke[2 * l4 + 0][2 * l4 + 0], Ke[2 * l4 + 0][2 * l4 + 1]},
                {Ke[2 * l4 + 1][2 * l + 0], Ke[2 * l4 + 1][2 * l + 1], Ke[2 * l4 + 1][2 * l4 + 0], Ke[2 * l4 + 1][2 * l4 + 1]}
              };
#else
              /* Same as above except for the lower-left block */
              const PetscScalar Kel[4][4] = {
                {Ke[2 * l + 0][2 * l + 0],  Ke[2 * l + 0][2 * l + 1],  Ke[2 * l + 0][2 * l4 + 0],  Ke[2 * l + 0][2 * l4 + 1] },
                {Ke[2 * l + 1][2 * l + 0],  Ke[2 * l + 1][2 * l + 1],  Ke[2 * l + 1][2 * l4 + 0],  Ke[2 * l + 1][2 * l4 + 1] },
                {Ke[2 * l + 0][2 * l4 + 0], Ke[2 * l + 1][2 * l4 + 0], Ke[2 * l4 + 0][2 * l4 + 0], Ke[2 * l4 + 0][2 * l4 + 1]},
                {Ke[2 * l + 0][2 * l4 + 1], Ke[2 * l + 1][2 * l4 + 1], Ke[2 * l4 + 1][2 * l4 + 0], Ke[2 * l4 + 1][2 * l4 + 1]}
              };
#endif
              PetscCall(MatSetValuesBlockedStencil(B, 2, rcl, 2, rcl, &Kel[0][0], ADD_VALUES));
            }
          } else {
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
            PetscCall(MatSetValuesBlockedStencil(B, 8, rc, 8, rc, &Ke[0][0], ADD_VALUES));
          }
        }
      }
    }
  }
  PetscCall(THIDARestorePrm(info->da, &prm));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE));
  if (thi->verbose) PetscCall(THIMatrixStatistics(thi, B, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobianLocal_3D_Full(DMDALocalInfo *info, Node ***x, Mat A, Mat B, THI thi)
{
  PetscFunctionBeginUser;
  PetscCall(THIJacobianLocal_3D(info, x, B, thi, THIASSEMBLY_FULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode THIJacobianLocal_3D_Tridiagonal(DMDALocalInfo *info, Node ***x, Mat A, Mat B, THI thi)
{
  PetscFunctionBeginUser;
  PetscCall(THIJacobianLocal_3D(info, x, B, thi, THIASSEMBLY_TRIDIAGONAL));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefineHierarchy_THI(DM dac0, PetscInt nlevels, DM hierarchy[])
{
  THI             thi;
  PetscInt        dim, M, N, m, n, s, dof;
  DM              dac, daf;
  DMDAStencilType st;
  DM_DA          *ddf, *ddc;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)dac0, "THI", (PetscObject *)&thi));
  PetscCheck(thi, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot refine this DMDA, missing composed THI instance");
  if (nlevels > 1) {
    PetscCall(DMRefineHierarchy(dac0, nlevels - 1, hierarchy));
    dac = hierarchy[nlevels - 2];
  } else {
    dac = dac0;
  }
  PetscCall(DMDAGetInfo(dac, &dim, &N, &M, 0, &n, &m, 0, &dof, &s, 0, 0, 0, &st));
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "This function can only refine 2D DMDAs");

  /* Creates a 3D DMDA with the same map-plane layout as the 2D one, with contiguous columns */
  PetscCall(DMDACreate3d(PetscObjectComm((PetscObject)dac), DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, st, thi->zlevels, N, M, 1, n, m, dof, s, NULL, NULL, NULL, &daf));
  PetscCall(DMSetUp(daf));

  daf->ops->creatematrix        = dac->ops->creatematrix;
  daf->ops->createinterpolation = dac->ops->createinterpolation;
  daf->ops->getcoloring         = dac->ops->getcoloring;
  ddf                           = (DM_DA *)daf->data;
  ddc                           = (DM_DA *)dac->data;
  ddf->interptype               = ddc->interptype;

  PetscCall(DMDASetFieldName(daf, 0, "x-velocity"));
  PetscCall(DMDASetFieldName(daf, 1, "y-velocity"));

  hierarchy[nlevels - 1] = daf;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateInterpolation_DA_THI(DM dac, DM daf, Mat *A, Vec *scale)
{
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dac, DM_CLASSID, 1);
  PetscValidHeaderSpecific(daf, DM_CLASSID, 2);
  PetscValidPointer(A, 3);
  if (scale) PetscValidPointer(scale, 4);
  PetscCall(DMDAGetInfo(daf, &dim, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  if (dim == 2) {
    /* We are in the 2D problem and use normal DMDA interpolation */
    PetscCall(DMCreateInterpolation(dac, daf, A, scale));
  } else {
    PetscInt i, j, k, xs, ys, zs, xm, ym, zm, mx, my, mz, rstart, cstart;
    Mat      B;

    PetscCall(DMDAGetInfo(daf, 0, &mz, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    PetscCall(DMDAGetCorners(daf, &zs, &ys, &xs, &zm, &ym, &xm));
    PetscCheck(!zs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "unexpected");
    PetscCall(MatCreate(PetscObjectComm((PetscObject)daf), &B));
    PetscCall(MatSetSizes(B, xm * ym * zm, xm * ym, mx * my * mz, mx * my));

    PetscCall(MatSetType(B, MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(B, 1, NULL));
    PetscCall(MatMPIAIJSetPreallocation(B, 1, NULL, 0, NULL));
    PetscCall(MatGetOwnershipRange(B, &rstart, NULL));
    PetscCall(MatGetOwnershipRangeColumn(B, &cstart, NULL));
    for (i = xs; i < xs + xm; i++) {
      for (j = ys; j < ys + ym; j++) {
        for (k = zs; k < zs + zm; k++) {
          PetscInt    i2 = i * ym + j, i3 = i2 * zm + k;
          PetscScalar val = ((k == 0 || k == mz - 1) ? 0.5 : 1.) / (mz - 1.); /* Integration using trapezoid rule */
          PetscCall(MatSetValue(B, cstart + i3, rstart + i2, val, INSERT_VALUES));
        }
      }
    }
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreateMAIJ(B, sizeof(Node) / sizeof(PetscScalar), A));
    PetscCall(MatDestroy(&B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_THI_Tridiagonal(DM da, Mat *J)
{
  Mat                    A;
  PetscInt               xm, ym, zm, dim, dof = 2, starts[3], dims[3];
  ISLocalToGlobalMapping ltog;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, &dim, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCheck(dim == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected DMDA to be 3D");
  PetscCall(DMDAGetCorners(da, 0, 0, 0, &zm, &ym, &xm));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)da), &A));
  PetscCall(MatSetSizes(A, dof * xm * ym * zm, dof * xm * ym * zm, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, da->mattype));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3 * 2, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3 * 2, NULL, 0, NULL));
  PetscCall(MatSeqBAIJSetPreallocation(A, 2, 3, NULL));
  PetscCall(MatMPIBAIJSetPreallocation(A, 2, 3, NULL, 0, NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(A, 2, 2, NULL));
  PetscCall(MatMPISBAIJSetPreallocation(A, 2, 2, NULL, 0, NULL));
  PetscCall(MatSetLocalToGlobalMapping(A, ltog, ltog));
  PetscCall(DMDAGetGhostCorners(da, &starts[0], &starts[1], &starts[2], &dims[0], &dims[1], &dims[2]));
  PetscCall(MatSetStencil(A, dim, dims, starts, dof));
  *J = A;
  PetscFunctionReturn(0);
}

static PetscErrorCode THIDAVecView_VTK_XML(THI thi, DM da, Vec X, const char filename[])
{
  const PetscInt     dof   = 2;
  Units              units = thi->units;
  MPI_Comm           comm;
  PetscViewer        viewer;
  PetscMPIInt        rank, size, tag, nn, nmax;
  PetscInt           mx, my, mz, r, range[6];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)thi, &comm));
  PetscCall(DMDAGetInfo(da, 0, &mz, &my, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerASCIIOpen(comm, filename, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  <StructuredGrid WholeExtent=\"%d %" PetscInt_FMT " %d %" PetscInt_FMT " %d %" PetscInt_FMT "\">\n", 0, mz - 1, 0, my - 1, 0, mx - 1));

  PetscCall(DMDAGetCorners(da, range, range + 1, range + 2, range + 3, range + 4, range + 5));
  PetscCall(PetscMPIIntCast(range[3] * range[4] * range[5] * dof, &nn));
  PetscCallMPI(MPI_Reduce(&nn, &nmax, 1, MPI_INT, MPI_MAX, 0, comm));
  tag = ((PetscObject)viewer)->tag;
  PetscCall(VecGetArrayRead(X, &x));
  if (rank == 0) {
    PetscScalar *array;
    PetscCall(PetscMalloc1(nmax, &array));
    for (r = 0; r < size; r++) {
      PetscInt           i, j, k, xs, xm, ys, ym, zs, zm;
      const PetscScalar *ptr;
      MPI_Status         status;
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
        PetscCheck(nn == xm * ym * zm * dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "should not happen");
        ptr = array;
      } else ptr = x;
      PetscCall(PetscViewerASCIIPrintf(viewer, "    <Piece Extent=\"%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\">\n", zs, zs + zm - 1, ys, ys + ym - 1, xs, xs + xm - 1));

      PetscCall(PetscViewerASCIIPrintf(viewer, "      <Points>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n"));
      for (i = xs; i < xs + xm; i++) {
        for (j = ys; j < ys + ym; j++) {
          for (k = zs; k < zs + zm; k++) {
            PrmNode   p;
            PetscReal xx = thi->Lx * i / mx, yy = thi->Ly * j / my, zz;
            thi->initialize(thi, xx, yy, &p);
            zz = PetscRealPart(p.b) + PetscRealPart(p.h) * k / (mz - 1);
            PetscCall(PetscViewerASCIIPrintf(viewer, "%f %f %f\n", (double)xx, (double)yy, (double)zz));
          }
        }
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "        </DataArray>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "      </Points>\n"));

      PetscCall(PetscViewerASCIIPrintf(viewer, "      <PointData>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n"));
      for (i = 0; i < nn; i += dof) PetscCall(PetscViewerASCIIPrintf(viewer, "%f %f %f\n", (double)(PetscRealPart(ptr[i]) * units->year / units->meter), (double)(PetscRealPart(ptr[i + 1]) * units->year / units->meter), 0.0));
      PetscCall(PetscViewerASCIIPrintf(viewer, "        </DataArray>\n"));

      PetscCall(PetscViewerASCIIPrintf(viewer, "        <DataArray type=\"Int32\" Name=\"rank\" NumberOfComponents=\"1\" format=\"ascii\">\n"));
      for (i = 0; i < nn; i += dof) PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", r));
      PetscCall(PetscViewerASCIIPrintf(viewer, "        </DataArray>\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "      </PointData>\n"));

      PetscCall(PetscViewerASCIIPrintf(viewer, "    </Piece>\n"));
    }
    PetscCall(PetscFree(array));
  } else {
    PetscCallMPI(MPI_Send(range, 6, MPIU_INT, 0, tag, comm));
    PetscCallMPI(MPI_Send((PetscScalar *)x, nn, MPIU_SCALAR, 0, tag, comm));
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  </StructuredGrid>\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "</VTKFile>\n"));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MPI_Comm comm;
  THI      thi;
  DM       da;
  SNES     snes;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(THICreate(comm, &thi));
  {
    PetscInt M = 3, N = 3, P = 2;
    PetscOptionsBegin(comm, NULL, "Grid resolution options", "");
    {
      PetscCall(PetscOptionsInt("-M", "Number of elements in x-direction on coarse level", "", M, &M, NULL));
      N = M;
      PetscCall(PetscOptionsInt("-N", "Number of elements in y-direction on coarse level (if different from M)", "", N, &N, NULL));
      if (thi->coarse2d) {
        PetscCall(PetscOptionsInt("-zlevels", "Number of elements in z-direction on fine level", "", thi->zlevels, &thi->zlevels, NULL));
      } else {
        PetscCall(PetscOptionsInt("-P", "Number of elements in z-direction on coarse level", "", P, &P, NULL));
      }
    }
    PetscOptionsEnd();
    if (thi->coarse2d) {
      PetscCall(DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, N, M, PETSC_DETERMINE, PETSC_DETERMINE, sizeof(Node) / sizeof(PetscScalar), 1, 0, 0, &da));
      PetscCall(DMSetFromOptions(da));
      PetscCall(DMSetUp(da));
      da->ops->refinehierarchy     = DMRefineHierarchy_THI;
      da->ops->createinterpolation = DMCreateInterpolation_DA_THI;

      PetscCall(PetscObjectCompose((PetscObject)da, "THI", (PetscObject)thi));
    } else {
      PetscCall(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, P, N, M, 1, PETSC_DETERMINE, PETSC_DETERMINE, sizeof(Node) / sizeof(PetscScalar), 1, 0, 0, 0, &da));
      PetscCall(DMSetFromOptions(da));
      PetscCall(DMSetUp(da));
    }
    PetscCall(DMDASetFieldName(da, 0, "x-velocity"));
    PetscCall(DMDASetFieldName(da, 1, "y-velocity"));
  }
  PetscCall(THISetUpDM(thi, da));
  if (thi->tridiagonal) da->ops->creatematrix = DMCreateMatrix_THI_Tridiagonal;

  { /* Set the fine level matrix type if -da_refine */
    PetscInt rlevel, clevel;
    PetscCall(DMGetRefineLevel(da, &rlevel));
    PetscCall(DMGetCoarsenLevel(da, &clevel));
    if (rlevel - clevel > 0) PetscCall(DMSetMatType(da, thi->mattype));
  }

  PetscCall(DMDASNESSetFunctionLocal(da, ADD_VALUES, (DMDASNESFunction)THIFunctionLocal, thi));
  if (thi->tridiagonal) {
    PetscCall(DMDASNESSetJacobianLocal(da, (DMDASNESJacobian)THIJacobianLocal_3D_Tridiagonal, thi));
  } else {
    PetscCall(DMDASNESSetJacobianLocal(da, (DMDASNESJacobian)THIJacobianLocal_3D_Full, thi));
  }
  PetscCall(DMCoarsenHookAdd(da, DMCoarsenHook_THI, NULL, thi));
  PetscCall(DMRefineHookAdd(da, DMRefineHook_THI, NULL, thi));

  PetscCall(DMSetApplicationContext(da, thi));

  PetscCall(SNESCreate(comm, &snes));
  PetscCall(SNESSetDM(snes, da));
  PetscCall(DMDestroy(&da));
  PetscCall(SNESSetComputeInitialGuess(snes, THIInitial, NULL));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes, NULL, NULL));

  PetscCall(THISolveStatistics(thi, snes, 0, "Full"));

  {
    PetscBool flg;
    char      filename[PETSC_MAX_PATH_LEN] = "";
    PetscCall(PetscOptionsGetString(NULL, NULL, "-o", filename, sizeof(filename), &flg));
    if (flg) {
      Vec X;
      DM  dm;
      PetscCall(SNESGetSolution(snes, &X));
      PetscCall(SNESGetDM(snes, &dm));
      PetscCall(THIDAVecView_VTK_XML(thi, dm, X, filename));
    }
  }

  PetscCall(DMDestroy(&da));
  PetscCall(SNESDestroy(&snes));
  PetscCall(THIDestroy(&thi));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !single

   test:
      args: -M 6 -P 4 -da_refine 1 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type sbaij -ksp_type fgmres -pc_type mg -pc_mg_type full -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type icc

   test:
      suffix: 2
      nsize: 2
      args: -M 6 -P 4 -thi_hom z -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type sbaij -ksp_type fgmres -pc_type mg -pc_mg_type full -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type asm -mg_levels_pc_asm_blocks 6 -mg_levels_0_pc_type redundant -snes_grid_sequence 1 -mat_partitioning_type current -ksp_atol -1

   test:
      suffix: 3
      nsize: 3
      args: -M 7 -P 4 -thi_hom z -da_refine 1 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type baij -ksp_type fgmres -pc_type mg -pc_mg_type full -mg_levels_pc_asm_type restrict -mg_levels_pc_type asm -mg_levels_pc_asm_blocks 9 -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mat_partitioning_type current

   test:
      suffix: 4
      nsize: 6
      args: -M 4 -P 2 -da_refine_hierarchy_x 1,1,3 -da_refine_hierarchy_y 2,2,1 -da_refine_hierarchy_z 2,2,1 -snes_grid_sequence 3 -ksp_converged_reason -ksp_type fgmres -ksp_rtol 1e-2 -pc_type mg -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi -mg_levels_1_sub_pc_type cholesky -pc_mg_type multiplicative -snes_converged_reason -snes_stol 1e-12 -thi_L 80e3 -thi_alpha 0.05 -thi_friction_m 1 -thi_hom x -snes_view -mg_levels_0_pc_type redundant -mg_levels_0_ksp_type preonly -ksp_atol -1

   test:
      suffix: 5
      nsize: 6
      args: -M 12 -P 5 -snes_monitor_short -ksp_converged_reason -pc_type asm -pc_asm_type restrict -dm_mat_type {{aij baij sbaij}}

TEST*/
