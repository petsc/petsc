#include <petscdm.h>
#include <petscdmceed.h>

#ifdef __CUDACC_RTC__
  #define PETSC_HAVE_LIBCEED
// Define PETSc types to be equal to Ceed types
typedef CeedInt PetscInt;
typedef CeedScalar PetscReal;
typedef CeedScalar PetscScalar;
typedef CeedInt PetscErrorCode;
  // Define things we are missing from PETSc headers
  #undef PETSC_SUCCESS
  #define PETSC_SUCCESS   0
  #define PETSC_COMM_SELF MPI_COMM_SELF
  #undef PetscFunctionBeginUser
  #define PetscFunctionBeginUser
  #undef PetscFunctionReturn
  #define PetscFunctionReturn(x) return x
  #undef PetscCall
  #define PetscCall(a)              a
  #define PetscFunctionReturnVoid() return
  //   Math definitions
  #undef PetscSqrtReal
  #define PetscSqrtReal(x) sqrt(x)
  #undef PetscSqrtScalar
  #define PetscSqrtScalar(x) sqrt(x)
  #undef PetscSqr
  #define PetscSqr(x)          (x * x)
  #define PetscSqrReal(x)      (x * x)
  #define PetscAbsReal(x)      abs(x)
  #define PetscAbsScalar(x)    abs(x)
  #define PetscMax(x, y)       x > y ? x : y
  #define PetscMin(x, y)       x < y ? x : y
  #define PetscRealPart(a)     a
  #define PetscPowScalar(a, b) pow(a, b)
#endif

#define DIM 2 /* Geometric dimension */

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

struct FieldDescription {
  const char *name;
  PetscInt    dof;
};

struct _n_Physics {
  void (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);
  PetscInt                       dof;      /* number of degrees of freedom per cell */
  PetscReal                      maxspeed; /* kludge to pick initial time step, need to add monitoring and step control */
  void                          *data;
  PetscInt                       nfields;
  const struct FieldDescription *field_desc;
};

typedef struct {
  PetscReal gravity;
  struct {
    PetscInt Height;
    PetscInt Speed;
    PetscInt Energy;
  } functional;
} Physics_SW;

typedef struct {
  PetscReal h;
  PetscReal uh[DIM];
} SWNode;
typedef union
{
  SWNode    swnode;
  PetscReal vals[DIM + 1];
} SWNodeUnion;

typedef enum {
  EULER_IV_SHOCK,
  EULER_SS_SHOCK,
  EULER_SHOCK_TUBE,
  EULER_LINEAR_WAVE
} EulerType;

typedef struct {
  PetscReal gamma;
  PetscReal rhoR;
  PetscReal amach;
  PetscReal itana;
  EulerType type;
  struct {
    PetscInt Density;
    PetscInt Momentum;
    PetscInt Energy;
    PetscInt Pressure;
    PetscInt Speed;
  } monitor;
} Physics_Euler;

typedef struct {
  PetscReal r;
  PetscReal ru[DIM];
  PetscReal E;
} EulerNode;
typedef union
{
  EulerNode eulernode;
  PetscReal vals[DIM + 2];
} EulerNodeUnion;

static inline PetscReal Dot2Real(const PetscReal *x, const PetscReal *y)
{
  return x[0] * y[0] + x[1] * y[1];
}
static inline PetscReal Norm2Real(const PetscReal *x)
{
  return PetscSqrtReal(PetscAbsReal(Dot2Real(x, x)));
}
static inline void Normalize2Real(PetscReal *x)
{
  PetscReal a = 1. / Norm2Real(x);
  x[0] *= a;
  x[1] *= a;
}
static inline void Scale2Real(PetscReal a, const PetscReal *x, PetscReal *y)
{
  y[0] = a * x[0];
  y[1] = a * x[1];
}

static inline PetscReal DotDIMReal(const PetscReal *x, const PetscReal *y)
{
  PetscInt  i;
  PetscReal prod = 0.0;

  for (i = 0; i < DIM; i++) prod += x[i] * y[i];
  return prod;
}
static inline PetscReal NormDIM(const PetscReal *x)
{
  return PetscSqrtReal(PetscAbsReal(DotDIMReal(x, x)));
}
static inline void Waxpy2Real(PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w)
{
  w[0] = a * x[0] + y[0];
  w[1] = a * x[1] + y[1];
}

/*
 * h_t + div(uh) = 0
 * (uh)_t + div (u\otimes uh + g h^2 / 2 I) = 0
 *
 * */
static PetscErrorCode SWFlux(Physics phys, const PetscReal *n, const SWNode *x, SWNode *f)
{
  Physics_SW *sw = (Physics_SW *)phys->data;
  PetscReal   uhn, u[DIM];
  PetscInt    i;

  PetscFunctionBeginUser;
  Scale2Real(1. / x->h, x->uh, u);
  uhn  = x->uh[0] * n[0] + x->uh[1] * n[1];
  f->h = uhn;
  for (i = 0; i < DIM; i++) f->uh[i] = u[i] * uhn + sw->gravity * PetscSqr(x->h) * n[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void PhysicsRiemann_SW_Rusanov(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  Physics_SW *sw = (Physics_SW *)phys->data;
  PetscReal   cL, cR, speed;
  PetscReal   nn[DIM];
#if !defined(PETSC_USE_COMPLEX)
  const SWNode *uL = (const SWNode *)xL, *uR = (const SWNode *)xR;
#else
  SWNodeUnion   uLreal, uRreal;
  const SWNode *uL = &uLreal.swnode;
  const SWNode *uR = &uRreal.swnode;
#endif
  SWNodeUnion    fL, fR;
  PetscInt       i;
  PetscReal      zero = 0.;
  PetscErrorCode ierr;

#if defined(PETSC_USE_COMPLEX)
  uLreal.swnode.h = 0;
  uRreal.swnode.h = 0;
  for (i = 0; i < 1 + dim; i++) uLreal.vals[i] = PetscRealPart(xL[i]);
  for (i = 0; i < 1 + dim; i++) uRreal.vals[i] = PetscRealPart(xR[i]);
#endif

  if (uL->h < 0 || uR->h < 0) {
    // reconstructed thickness is negative
    PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    for (i = 0; i < 1 + dim; ++i) flux[i] = zero / zero;
    PetscCallVoid(PetscFPTrapPop());
    return;
  }

  nn[0] = n[0];
  nn[1] = n[1];
  Normalize2Real(nn);
  ierr = SWFlux(phys, nn, uL, &fL.swnode);
  if (ierr) {
    PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    for (i = 0; i < 1 + dim; ++i) fL.vals[i] = zero / zero;
    PetscCallVoid(PetscFPTrapPop());
  }
  ierr = SWFlux(phys, nn, uR, &fR.swnode);
  if (ierr) {
    PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    for (i = 0; i < 1 + dim; ++i) fR.vals[i] = zero / zero;
    PetscCallVoid(PetscFPTrapPop());
  }
  cL    = PetscSqrtReal(sw->gravity * uL->h);
  cR    = PetscSqrtReal(sw->gravity * uR->h); /* gravity wave speed */
  speed = PetscMax(PetscAbsReal(Dot2Real(uL->uh, nn) / uL->h) + cL, PetscAbsReal(Dot2Real(uR->uh, nn) / uR->h) + cR);
  for (i = 0; i < 1 + dim; i++) flux[i] = (0.5 * (fL.vals[i] + fR.vals[i]) + 0.5 * speed * (xL[i] - xR[i])) * Norm2Real(n);
#if 0
  PetscPrintf(PETSC_COMM_SELF, "Rusanov Flux (%g)\n", sw->gravity);
  for (PetscInt j = 0; j < 3; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", flux[j]);
#endif
}

#ifdef PETSC_HAVE_LIBCEED
CEED_QFUNCTION(PhysicsRiemann_SW_Rusanov_CEED)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[])
{
  const CeedScalar *xL = in[0], *xR = in[1], *geom = in[2];
  CeedScalar       *cL = out[0], *cR = out[1];
  const Physics_SW *sw = (Physics_SW *)ctx;
  struct _n_Physics phys;
  #if 0
  const CeedScalar *info = in[3];
  #endif

  phys.data = (void *)sw;
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
  {
    const CeedScalar qL[3] = {xL[i + Q * 0], xL[i + Q * 1], xL[i + Q * 2]};
    const CeedScalar qR[3] = {xR[i + Q * 0], xR[i + Q * 1], xR[i + Q * 2]};
    const CeedScalar n[2]  = {geom[i + Q * 0], geom[i + Q * 1]};
    CeedScalar       flux[3];

  #if 0
    PetscPrintf(PETSC_COMM_SELF, "Face %d Normal\n", (int)info[i + Q * 0]);
    for (CeedInt j = 0; j < DIM; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", n[j]);
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: left state\n", (int)info[i + Q * 1]);
    for (CeedInt j = 0; j < DIM + 1; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", qL[j]);
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: right state\n", (int)info[i + Q * 2]);
    for (CeedInt j = 0; j < DIM + 1; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", qR[j]);
  #endif
    PhysicsRiemann_SW_Rusanov(DIM, DIM + 1, NULL, n, qL, qR, 0, NULL, flux, &phys);
    for (CeedInt j = 0; j < 3; ++j) {
      cL[i + Q * j] = -flux[j] / geom[i + Q * 2];
      cR[i + Q * j] = flux[j] / geom[i + Q * 3];
    }
  #if 0
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: left flux\n", (int)info[i + Q * 1]);
    for (CeedInt j = 0; j < DIM + 1; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g | (%g)\n", cL[i + Q * j], geom[i + Q * 2]);
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: right flux\n", (int)info[i + Q * 2]);
    for (CeedInt j = 0; j < DIM + 1; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g | (%g)\n", cR[i + Q * j], geom[i + Q * 3]);
  #endif
  }
  return CEED_ERROR_SUCCESS;
}
#endif

static void PhysicsRiemann_SW_HLL(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  Physics_SW *sw = (Physics_SW *)phys->data;
  PetscReal   aL, aR;
  PetscReal   nn[DIM];
#if !defined(PETSC_USE_COMPLEX)
  const SWNode *uL = (const SWNode *)xL, *uR = (const SWNode *)xR;
#else
  SWNodeUnion   uLreal, uRreal;
  const SWNode *uL = &uLreal.swnode;
  const SWNode *uR = &uRreal.swnode;
#endif
  SWNodeUnion    fL, fR;
  PetscInt       i;
  PetscReal      zero = 0.;
  PetscErrorCode ierr;

#if defined(PETSC_USE_COMPLEX)
  uLreal.swnode.h = 0;
  uRreal.swnode.h = 0;
  for (i = 0; i < 1 + dim; i++) uLreal.vals[i] = PetscRealPart(xL[i]);
  for (i = 0; i < 1 + dim; i++) uRreal.vals[i] = PetscRealPart(xR[i]);
#endif
  if (uL->h <= 0 || uR->h <= 0) {
    PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    for (i = 0; i < 1 + dim; i++) flux[i] = zero;
    PetscCallVoid(PetscFPTrapPop());
    return;
  } /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative"); */
  nn[0] = n[0];
  nn[1] = n[1];
  Normalize2Real(nn);
  ierr = SWFlux(phys, nn, uL, &fL.swnode);
  if (ierr) {
    PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    for (i = 0; i < 1 + dim; ++i) fL.vals[i] = zero / zero;
    PetscCallVoid(PetscFPTrapPop());
  }
  ierr = SWFlux(phys, nn, uR, &fR.swnode);
  if (ierr) {
    PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    for (i = 0; i < 1 + dim; ++i) fR.vals[i] = zero / zero;
    PetscCallVoid(PetscFPTrapPop());
  }
  /* gravity wave speed */
  aL = PetscSqrtReal(sw->gravity * uL->h);
  aR = PetscSqrtReal(sw->gravity * uR->h);
  // Defining u_tilda and v_tilda as u and v
  PetscReal u_L, u_R;
  u_L = Dot2Real(uL->uh, nn) / uL->h;
  u_R = Dot2Real(uR->uh, nn) / uR->h;
  PetscReal sL, sR;
  sL = PetscMin(u_L - aL, u_R - aR);
  sR = PetscMax(u_L + aL, u_R + aR);
  if (sL > zero) {
    for (i = 0; i < dim + 1; i++) flux[i] = fL.vals[i] * Norm2Real(n);
  } else if (sR < zero) {
    for (i = 0; i < dim + 1; i++) flux[i] = fR.vals[i] * Norm2Real(n);
  } else {
    for (i = 0; i < dim + 1; i++) flux[i] = ((sR * fL.vals[i] - sL * fR.vals[i] + sR * sL * (xR[i] - xL[i])) / (sR - sL)) * Norm2Real(n);
  }
}

static PetscErrorCode Pressure_PG(const PetscReal gamma, const EulerNode *x, PetscReal *p)
{
  PetscReal ru2;

  PetscFunctionBeginUser;
  ru2  = DotDIMReal(x->ru, x->ru);
  (*p) = (x->E - 0.5 * ru2 / x->r) * (gamma - 1.0); /* (E - rho V^2/2)(gamma-1) = e rho (gamma-1) */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SpeedOfSound_PG(const PetscReal gamma, const EulerNode *x, PetscReal *c)
{
  PetscReal p;

  PetscFunctionBeginUser;
  PetscCall(Pressure_PG(gamma, x, &p));
  /* gamma = heat capacity ratio */
  (*c) = PetscSqrtReal(gamma * p / x->r);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * x = (rho,rho*(u_1),...,rho*e)^T
 * x_t+div(f_1(x))+...+div(f_DIM(x)) = 0
 *
 * f_i(x) = u_i*x+(0,0,...,p,...,p*u_i)^T
 *
 */
static PetscErrorCode EulerFlux(Physics phys, const PetscReal *n, const EulerNode *x, EulerNode *f)
{
  Physics_Euler *eu = (Physics_Euler *)phys->data;
  PetscReal      nu, p;
  PetscInt       i;

  PetscFunctionBeginUser;
  PetscCall(Pressure_PG(eu->gamma, x, &p));
  nu   = DotDIMReal(x->ru, n);
  f->r = nu;                                                     /* A rho u */
  nu /= x->r;                                                    /* A u */
  for (i = 0; i < DIM; i++) f->ru[i] = nu * x->ru[i] + n[i] * p; /* r u^2 + p */
  f->E = nu * (x->E + p);                                        /* u(e+p) */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Godunov fluxs */
static PetscScalar cvmgp_(PetscScalar *a, PetscScalar *b, PetscScalar *test)
{
  /* System generated locals */
  PetscScalar ret_val;

  if (PetscRealPart(*test) > 0.) goto L10;
  ret_val = *b;
  return ret_val;
L10:
  ret_val = *a;
  return ret_val;
} /* cvmgp_ */

static PetscScalar cvmgm_(PetscScalar *a, PetscScalar *b, PetscScalar *test)
{
  /* System generated locals */
  PetscScalar ret_val;

  if (PetscRealPart(*test) < 0.) goto L10;
  ret_val = *b;
  return ret_val;
L10:
  ret_val = *a;
  return ret_val;
} /* cvmgm_ */

static int riem1mdt(PetscScalar *gaml, PetscScalar *gamr, PetscScalar *rl, PetscScalar *pl, PetscScalar *uxl, PetscScalar *rr, PetscScalar *pr, PetscScalar *uxr, PetscScalar *rstarl, PetscScalar *rstarr, PetscScalar *pstar, PetscScalar *ustar)
{
  /* Initialized data */

  static PetscScalar smallp = 1e-8;

  /* System generated locals */
  int         i__1;
  PetscScalar d__1, d__2;

  /* Local variables */
  static int         i0;
  static PetscScalar cl, cr, wl, zl, wr, zr, pst, durl, skpr1, skpr2;
  static int         iwave;
  static PetscScalar gascl4, gascr4, cstarl, dpstar, cstarr;
  /* static PetscScalar csqrl, csqrr, gascl1, gascl2, gascl3, gascr1, gascr2, gascr3; */
  static int         iterno;
  static PetscScalar ustarl, ustarr, rarepr1, rarepr2;

  /* gascl1 = *gaml - 1.; */
  /* gascl2 = (*gaml + 1.) * .5; */
  /* gascl3 = gascl2 / *gaml; */
  gascl4 = 1. / (*gaml - 1.);

  /* gascr1 = *gamr - 1.; */
  /* gascr2 = (*gamr + 1.) * .5; */
  /* gascr3 = gascr2 / *gamr; */
  gascr4 = 1. / (*gamr - 1.);
  iterno = 10;
  /*        find pstar: */
  cl = PetscSqrtScalar(*gaml * *pl / *rl);
  cr = PetscSqrtScalar(*gamr * *pr / *rr);
  wl = *rl * cl;
  wr = *rr * cr;
  /* csqrl = wl * wl; */
  /* csqrr = wr * wr; */
  *pstar  = (wl * *pr + wr * *pl) / (wl + wr);
  *pstar  = PetscMax(PetscRealPart(*pstar), PetscRealPart(smallp));
  pst     = *pl / *pr;
  skpr1   = cr * (pst - 1.) * PetscSqrtScalar(2. / (*gamr * (*gamr - 1. + (*gamr + 1.) * pst)));
  d__1    = (*gamr - 1.) / (*gamr * 2.);
  rarepr2 = gascr4 * 2. * cr * (1. - PetscPowScalar(pst, d__1));
  pst     = *pr / *pl;
  skpr2   = cl * (pst - 1.) * PetscSqrtScalar(2. / (*gaml * (*gaml - 1. + (*gaml + 1.) * pst)));
  d__1    = (*gaml - 1.) / (*gaml * 2.);
  rarepr1 = gascl4 * 2. * cl * (1. - PetscPowScalar(pst, d__1));
  durl    = *uxr - *uxl;
  if (PetscRealPart(*pr) < PetscRealPart(*pl)) {
    if (PetscRealPart(durl) >= PetscRealPart(rarepr1)) {
      iwave = 100;
    } else if (PetscRealPart(durl) <= PetscRealPart(-skpr1)) {
      iwave = 300;
    } else {
      iwave = 400;
    }
  } else {
    if (PetscRealPart(durl) >= PetscRealPart(rarepr2)) {
      iwave = 100;
    } else if (PetscRealPart(durl) <= PetscRealPart(-skpr2)) {
      iwave = 300;
    } else {
      iwave = 200;
    }
  }
  if (iwave == 100) {
    /*     1-wave: rarefaction wave, 3-wave: rarefaction wave */
    /*     case (100) */
    i__1 = iterno;
    for (i0 = 1; i0 <= i__1; ++i0) {
      d__1    = *pstar / *pl;
      d__2    = 1. / *gaml;
      *rstarl = *rl * PetscPowScalar(d__1, d__2);
      cstarl  = PetscSqrtScalar(*gaml * *pstar / *rstarl);
      ustarl  = *uxl - gascl4 * 2. * (cstarl - cl);
      zl      = *rstarl * cstarl;
      d__1    = *pstar / *pr;
      d__2    = 1. / *gamr;
      *rstarr = *rr * PetscPowScalar(d__1, d__2);
      cstarr  = PetscSqrtScalar(*gamr * *pstar / *rstarr);
      ustarr  = *uxr + gascr4 * 2. * (cstarr - cr);
      zr      = *rstarr * cstarr;
      dpstar  = zl * zr * (ustarr - ustarl) / (zl + zr);
      *pstar -= dpstar;
      *pstar = PetscMax(PetscRealPart(*pstar), PetscRealPart(smallp));
      if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
        break;
#endif
      }
    }
    /*     1-wave: shock wave, 3-wave: rarefaction wave */
  } else if (iwave == 200) {
    /*     case (200) */
    i__1 = iterno;
    for (i0 = 1; i0 <= i__1; ++i0) {
      pst     = *pstar / *pl;
      ustarl  = *uxl - (pst - 1.) * cl * PetscSqrtScalar(2. / (*gaml * (*gaml - 1. + (*gaml + 1.) * pst)));
      zl      = *pl / cl * PetscSqrtScalar(*gaml * 2. * (*gaml - 1. + (*gaml + 1.) * pst)) * (*gaml - 1. + (*gaml + 1.) * pst) / (*gaml * 3. - 1. + (*gaml + 1.) * pst);
      d__1    = *pstar / *pr;
      d__2    = 1. / *gamr;
      *rstarr = *rr * PetscPowScalar(d__1, d__2);
      cstarr  = PetscSqrtScalar(*gamr * *pstar / *rstarr);
      zr      = *rstarr * cstarr;
      ustarr  = *uxr + gascr4 * 2. * (cstarr - cr);
      dpstar  = zl * zr * (ustarr - ustarl) / (zl + zr);
      *pstar -= dpstar;
      *pstar = PetscMax(PetscRealPart(*pstar), PetscRealPart(smallp));
      if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
        break;
#endif
      }
    }
    /*     1-wave: shock wave, 3-wave: shock */
  } else if (iwave == 300) {
    /*     case (300) */
    i__1 = iterno;
    for (i0 = 1; i0 <= i__1; ++i0) {
      pst    = *pstar / *pl;
      ustarl = *uxl - (pst - 1.) * cl * PetscSqrtScalar(2. / (*gaml * (*gaml - 1. + (*gaml + 1.) * pst)));
      zl     = *pl / cl * PetscSqrtScalar(*gaml * 2. * (*gaml - 1. + (*gaml + 1.) * pst)) * (*gaml - 1. + (*gaml + 1.) * pst) / (*gaml * 3. - 1. + (*gaml + 1.) * pst);
      pst    = *pstar / *pr;
      ustarr = *uxr + (pst - 1.) * cr * PetscSqrtScalar(2. / (*gamr * (*gamr - 1. + (*gamr + 1.) * pst)));
      zr     = *pr / cr * PetscSqrtScalar(*gamr * 2. * (*gamr - 1. + (*gamr + 1.) * pst)) * (*gamr - 1. + (*gamr + 1.) * pst) / (*gamr * 3. - 1. + (*gamr + 1.) * pst);
      dpstar = zl * zr * (ustarr - ustarl) / (zl + zr);
      *pstar -= dpstar;
      *pstar = PetscMax(PetscRealPart(*pstar), PetscRealPart(smallp));
      if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
        break;
#endif
      }
    }
    /*     1-wave: rarefaction wave, 3-wave: shock */
  } else if (iwave == 400) {
    /*     case (400) */
    i__1 = iterno;
    for (i0 = 1; i0 <= i__1; ++i0) {
      d__1    = *pstar / *pl;
      d__2    = 1. / *gaml;
      *rstarl = *rl * PetscPowScalar(d__1, d__2);
      cstarl  = PetscSqrtScalar(*gaml * *pstar / *rstarl);
      ustarl  = *uxl - gascl4 * 2. * (cstarl - cl);
      zl      = *rstarl * cstarl;
      pst     = *pstar / *pr;
      ustarr  = *uxr + (pst - 1.) * cr * PetscSqrtScalar(2. / (*gamr * (*gamr - 1. + (*gamr + 1.) * pst)));
      zr      = *pr / cr * PetscSqrtScalar(*gamr * 2. * (*gamr - 1. + (*gamr + 1.) * pst)) * (*gamr - 1. + (*gamr + 1.) * pst) / (*gamr * 3. - 1. + (*gamr + 1.) * pst);
      dpstar  = zl * zr * (ustarr - ustarl) / (zl + zr);
      *pstar -= dpstar;
      *pstar = PetscMax(PetscRealPart(*pstar), PetscRealPart(smallp));
      if (PetscAbsScalar(dpstar) / PetscRealPart(*pstar) <= 1e-8) {
#if 0
              break;
#endif
      }
    }
  }

  *ustar = (zl * ustarr + zr * ustarl) / (zl + zr);
  if (PetscRealPart(*pstar) > PetscRealPart(*pl)) {
    pst     = *pstar / *pl;
    *rstarl = ((*gaml + 1.) * pst + *gaml - 1.) / ((*gaml - 1.) * pst + *gaml + 1.) * *rl;
  }
  if (PetscRealPart(*pstar) > PetscRealPart(*pr)) {
    pst     = *pstar / *pr;
    *rstarr = ((*gamr + 1.) * pst + *gamr - 1.) / ((*gamr - 1.) * pst + *gamr + 1.) * *rr;
  }
  return iwave;
}

static PetscScalar sign(PetscScalar x)
{
  if (PetscRealPart(x) > 0) return 1.0;
  if (PetscRealPart(x) < 0) return -1.0;
  return 0.0;
}
/*        Riemann Solver */
/* -------------------------------------------------------------------- */
static int riemannsolver(PetscScalar *xcen, PetscScalar *xp, PetscScalar *dtt, PetscScalar *rl, PetscScalar *uxl, PetscScalar *pl, PetscScalar *utl, PetscScalar *ubl, PetscScalar *gaml, PetscScalar *rho1l, PetscScalar *rr, PetscScalar *uxr, PetscScalar *pr, PetscScalar *utr, PetscScalar *ubr, PetscScalar *gamr, PetscScalar *rho1r, PetscScalar *rx, PetscScalar *uxm, PetscScalar *px, PetscScalar *utx, PetscScalar *ubx, PetscScalar *gam, PetscScalar *rho1)
{
  /* System generated locals */
  PetscScalar d__1, d__2;

  /* Local variables */
  static PetscScalar s, c0, p0, r0, u0, w0, x0, x2, ri, cx, sgn0, wsp0, gasc1, gasc2, gasc3, gasc4;
  static PetscScalar cstar, pstar, rstar, ustar, xstar, wspst, ushock, streng, rstarl, rstarr, rstars;
  int                iwave;

  if (*rl == *rr && *pr == *pl && *uxl == *uxr && *gaml == *gamr) {
    *rx  = *rl;
    *px  = *pl;
    *uxm = *uxl;
    *gam = *gaml;
    x2   = *xcen + *uxm * *dtt;

    if (PetscRealPart(*xp) >= PetscRealPart(x2)) {
      *utx  = *utr;
      *ubx  = *ubr;
      *rho1 = *rho1r;
    } else {
      *utx  = *utl;
      *ubx  = *ubl;
      *rho1 = *rho1l;
    }
    return 0;
  }
  iwave = riem1mdt(gaml, gamr, rl, pl, uxl, rr, pr, uxr, &rstarl, &rstarr, &pstar, &ustar);

  x2   = *xcen + ustar * *dtt;
  d__1 = *xp - x2;
  sgn0 = sign(d__1);
  /*            x is in 3-wave if sgn0 = 1 */
  /*            x is in 1-wave if sgn0 = -1 */
  r0     = cvmgm_(rl, rr, &sgn0);
  p0     = cvmgm_(pl, pr, &sgn0);
  u0     = cvmgm_(uxl, uxr, &sgn0);
  *gam   = cvmgm_(gaml, gamr, &sgn0);
  gasc1  = *gam - 1.;
  gasc2  = (*gam + 1.) * .5;
  gasc3  = gasc2 / *gam;
  gasc4  = 1. / (*gam - 1.);
  c0     = PetscSqrtScalar(*gam * p0 / r0);
  streng = pstar - p0;
  w0     = *gam * r0 * p0 * (gasc3 * streng / p0 + 1.);
  rstars = r0 / (1. - r0 * streng / w0);
  d__1   = p0 / pstar;
  d__2   = -1. / *gam;
  rstarr = r0 * PetscPowScalar(d__1, d__2);
  rstar  = cvmgm_(&rstarr, &rstars, &streng);
  w0     = PetscSqrtScalar(w0);
  cstar  = PetscSqrtScalar(*gam * pstar / rstar);
  wsp0   = u0 + sgn0 * c0;
  wspst  = ustar + sgn0 * cstar;
  ushock = ustar + sgn0 * w0 / rstar;
  wspst  = cvmgp_(&ushock, &wspst, &streng);
  wsp0   = cvmgp_(&ushock, &wsp0, &streng);
  x0     = *xcen + wsp0 * *dtt;
  xstar  = *xcen + wspst * *dtt;
  /*           using gas formula to evaluate rarefaction wave */
  /*            ri : reiman invariant */
  ri   = u0 - sgn0 * 2. * gasc4 * c0;
  cx   = sgn0 * .5 * gasc1 / gasc2 * ((*xp - *xcen) / *dtt - ri);
  *uxm = ri + sgn0 * 2. * gasc4 * cx;
  s    = p0 / PetscPowScalar(r0, *gam);
  d__1 = cx * cx / (*gam * s);
  *rx  = PetscPowScalar(d__1, gasc4);
  *px  = cx * cx * *rx / *gam;
  d__1 = sgn0 * (x0 - *xp);
  *rx  = cvmgp_(rx, &r0, &d__1);
  d__1 = sgn0 * (x0 - *xp);
  *px  = cvmgp_(px, &p0, &d__1);
  d__1 = sgn0 * (x0 - *xp);
  *uxm = cvmgp_(uxm, &u0, &d__1);
  d__1 = sgn0 * (xstar - *xp);
  *rx  = cvmgm_(rx, &rstar, &d__1);
  d__1 = sgn0 * (xstar - *xp);
  *px  = cvmgm_(px, &pstar, &d__1);
  d__1 = sgn0 * (xstar - *xp);
  *uxm = cvmgm_(uxm, &ustar, &d__1);
  if (PetscRealPart(*xp) >= PetscRealPart(x2)) {
    *utx  = *utr;
    *ubx  = *ubr;
    *rho1 = *rho1r;
  } else {
    *utx  = *utl;
    *ubx  = *ubl;
    *rho1 = *rho1l;
  }
  return iwave;
}

static int godunovflux(const PetscScalar *ul, const PetscScalar *ur, PetscScalar *flux, const PetscReal *nn, int ndim, PetscReal gamma)
{
  /* System generated locals */
  int         i__1, iwave;
  PetscScalar d__1, d__2, d__3;

  /* Local variables */
  static int         k;
  static PetscScalar bn[3], fn, ft, tg[3], pl, rl, pm, pr, rr, xp, ubl, ubm, ubr, dtt, unm, tmp, utl, utm, uxl, utr, uxr, gaml, gamm, gamr, xcen, rhom, rho1l, rho1m, rho1r;

  /* Function Body */
  xcen = 0.;
  xp   = 0.;
  i__1 = ndim;
  for (k = 1; k <= i__1; ++k) {
    tg[k - 1] = 0.;
    bn[k - 1] = 0.;
  }
  dtt = 1.;
  if (ndim == 3) {
    if (nn[0] == 0. && nn[1] == 0.) {
      tg[0] = 1.;
    } else {
      tg[0] = -nn[1];
      tg[1] = nn[0];
    }
    /*           tmp=dsqrt(tg(1)**2+tg(2)**2) */
    /*           tg=tg/tmp */
    bn[0] = -nn[2] * tg[1];
    bn[1] = nn[2] * tg[0];
    bn[2] = nn[0] * tg[1] - nn[1] * tg[0];
    /* Computing 2nd power */
    d__1 = bn[0];
    /* Computing 2nd power */
    d__2 = bn[1];
    /* Computing 2nd power */
    d__3 = bn[2];
    tmp  = PetscSqrtScalar(d__1 * d__1 + d__2 * d__2 + d__3 * d__3);
    i__1 = ndim;
    for (k = 1; k <= i__1; ++k) bn[k - 1] /= tmp;
  } else if (ndim == 2) {
    tg[0] = -nn[1];
    tg[1] = nn[0];
    /*           tmp=dsqrt(tg(1)**2+tg(2)**2) */
    /*           tg=tg/tmp */
    bn[0] = 0.;
    bn[1] = 0.;
    bn[2] = 1.;
  }
  rl   = ul[0];
  rr   = ur[0];
  uxl  = 0.;
  uxr  = 0.;
  utl  = 0.;
  utr  = 0.;
  ubl  = 0.;
  ubr  = 0.;
  i__1 = ndim;
  for (k = 1; k <= i__1; ++k) {
    uxl += ul[k] * nn[k - 1];
    uxr += ur[k] * nn[k - 1];
    utl += ul[k] * tg[k - 1];
    utr += ur[k] * tg[k - 1];
    ubl += ul[k] * bn[k - 1];
    ubr += ur[k] * bn[k - 1];
  }
  uxl /= rl;
  uxr /= rr;
  utl /= rl;
  utr /= rr;
  ubl /= rl;
  ubr /= rr;

  gaml = gamma;
  gamr = gamma;
  /* Computing 2nd power */
  d__1 = uxl;
  /* Computing 2nd power */
  d__2 = utl;
  /* Computing 2nd power */
  d__3 = ubl;
  pl   = (gamma - 1.) * (ul[ndim + 1] - rl * .5 * (d__1 * d__1 + d__2 * d__2 + d__3 * d__3));
  /* Computing 2nd power */
  d__1 = uxr;
  /* Computing 2nd power */
  d__2 = utr;
  /* Computing 2nd power */
  d__3  = ubr;
  pr    = (gamma - 1.) * (ur[ndim + 1] - rr * .5 * (d__1 * d__1 + d__2 * d__2 + d__3 * d__3));
  rho1l = rl;
  rho1r = rr;

  iwave = riemannsolver(&xcen, &xp, &dtt, &rl, &uxl, &pl, &utl, &ubl, &gaml, &rho1l, &rr, &uxr, &pr, &utr, &ubr, &gamr, &rho1r, &rhom, &unm, &pm, &utm, &ubm, &gamm, &rho1m);

  flux[0] = rhom * unm;
  fn      = rhom * unm * unm + pm;
  ft      = rhom * unm * utm;
  /*           flux(2)=fn*nn(1)+ft*nn(2) */
  /*           flux(3)=fn*tg(1)+ft*tg(2) */
  flux[1] = fn * nn[0] + ft * tg[0];
  flux[2] = fn * nn[1] + ft * tg[1];
  /*           flux(2)=rhom*unm*(unm)+pm */
  /*           flux(3)=rhom*(unm)*utm */
  if (ndim == 3) flux[3] = rhom * unm * ubm;
  flux[ndim + 1] = (rhom * .5 * (unm * unm + utm * utm + ubm * ubm) + gamm / (gamm - 1.) * pm) * unm;
  return iwave;
} /* godunovflux_ */

/* PetscReal* => EulerNode* conversion */
static void PhysicsRiemann_Euler_Godunov(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  Physics_Euler  *eu    = (Physics_Euler *)phys->data;
  const PetscReal gamma = eu->gamma;
  PetscReal       zero  = 0.;
  PetscReal       cL, cR, speed, velL, velR, nn[DIM], s2;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  for (i = 0, s2 = 0.; i < DIM; i++) {
    nn[i] = n[i];
    s2 += nn[i] * nn[i];
  }
  s2 = PetscSqrtReal(s2); /* |n|_2 = sum(n^2)^1/2 */
  for (i = 0.; i < DIM; i++) nn[i] /= s2;
  if (0) { /* Rusanov */
    const EulerNode *uL = (const EulerNode *)xL, *uR = (const EulerNode *)xR;
    EulerNodeUnion   fL, fR;
    ierr = EulerFlux(phys, nn, uL, &fL.eulernode);
    if (ierr) {
      PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      for (i = 0; i < 2 + dim; i++) fL.vals[i] = zero / zero;
      PetscCallVoid(PetscFPTrapPop());
    }
    ierr = EulerFlux(phys, nn, uR, &fR.eulernode);
    if (ierr) {
      PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      for (i = 0; i < 2 + dim; i++) fR.vals[i] = zero / zero;
      PetscCallVoid(PetscFPTrapPop());
    }
    ierr = SpeedOfSound_PG(gamma, uL, &cL);
    if (ierr) {
      PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      cL = zero / zero;
      PetscCallVoid(PetscFPTrapPop());
    }
    ierr = SpeedOfSound_PG(gamma, uR, &cR);
    if (ierr) {
      PetscCallVoid(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      cR = zero / zero;
      PetscCallVoid(PetscFPTrapPop());
    }
    velL  = DotDIMReal(uL->ru, nn) / uL->r;
    velR  = DotDIMReal(uR->ru, nn) / uR->r;
    speed = PetscMax(velR + cR, velL + cL);
    for (i = 0; i < 2 + dim; i++) flux[i] = 0.5 * ((fL.vals[i] + fR.vals[i]) + speed * (xL[i] - xR[i])) * s2;
  } else {
    /* int iwave =  */
    godunovflux(xL, xR, flux, nn, DIM, gamma);
    for (i = 0; i < 2 + dim; i++) flux[i] *= s2;
  }
  PetscFunctionReturnVoid();
}

#ifdef PETSC_HAVE_LIBCEED
CEED_QFUNCTION(PhysicsRiemann_Euler_Godunov_CEED)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[])
{
  const CeedScalar    *xL = in[0], *xR = in[1], *geom = in[2];
  CeedScalar          *cL = out[0], *cR = out[1];
  const Physics_Euler *eu = (Physics_Euler *)ctx;
  struct _n_Physics    phys;

  phys.data = (void *)eu;
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
  {
    const CeedScalar qL[DIM + 2] = {xL[i + Q * 0], xL[i + Q * 1], xL[i + Q * 2], xL[i + Q * 3]};
    const CeedScalar qR[DIM + 2] = {xR[i + Q * 0], xR[i + Q * 1], xR[i + Q * 2], xR[i + Q * 3]};
    const CeedScalar n[DIM]      = {geom[i + Q * 0], geom[i + Q * 1]};
    CeedScalar       flux[DIM + 2];

  #if 0
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Normal\n", 0);
    for (CeedInt j = 0; j < DIM; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", n[j]);
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: left state\n", 0);
    for (CeedInt j = 0; j < DIM + 2; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", qL[j]);
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: right state\n", 0);
    for (CeedInt j = 0; j < DIM + 2; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", qR[j]);
  #endif
    PhysicsRiemann_Euler_Godunov(DIM, DIM + 2, NULL, n, qL, qR, 0, NULL, flux, &phys);
    for (CeedInt j = 0; j < DIM + 2; ++j) {
      cL[i + Q * j] = -flux[j] / geom[i + Q * 2];
      cR[i + Q * j] = flux[j] / geom[i + Q * 3];
    }
  #if 0
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: left flux\n", 0);
    for (CeedInt j = 0; j < DIM + 2; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g | (%g)\n", cL[i + Q * j], geom[i + Q * 2]);
    PetscPrintf(PETSC_COMM_SELF, "Cell %d Element Residual: right flux\n", 0);
    for (CeedInt j = 0; j < DIM + 2; ++j) PetscPrintf(PETSC_COMM_SELF, "  | %g | (%g)\n", cR[i + Q * j], geom[i + Q * 3]);
  #endif
  }
  return CEED_ERROR_SUCCESS;
}
#endif
