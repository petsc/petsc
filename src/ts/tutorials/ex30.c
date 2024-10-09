static char help[] = "Biological network from https://link.springer.com/article/10.1007/s42967-023-00297-3\n\n\n";

#include <petscts.h>
#include <petscsf.h>
#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscdmforest.h>
#include <petscviewerhdf5.h>
#include <petscds.h>

/*
    Here we solve the system of PDEs on \Omega \in R^2:

    * dC/dt - D^2 \Delta C - c^2 \nabla p \cross \nabla p + \alpha sqrt(||C||^2_F + eps)^(\gamma-2) C = 0
    * - \nabla \cdot ((r + C) \nabla p) = S

    where:
      C = symmetric 2x2 conductivity tensor
      p = potential
      S = source

    with natural boundary conditions on \partial\Omega:
      \nabla C \cdot n  = 0
      \nabla ((r + C)\nabla p) \cdot n  = 0

    Parameters:
      D = diffusion constant
      c = activation parameter
      \alpha = metabolic coefficient
      \gamma = metabolic exponent
      r, eps are regularization parameters

    We use Lagrange elements for C_ij and P.
*/

typedef enum _fieldidx {
  C_FIELD_ID = 0,
  P_FIELD_ID,
  NUM_FIELDS
} FieldIdx;

typedef enum _constantidx {
  R_ID = 0,
  EPS_ID,
  ALPHA_ID,
  GAMMA_ID,
  D_ID,
  C2_ID,
  FIXC_ID,
  NUM_CONSTANTS
} ConstantIdx;

PetscLogStage SetupStage, SolveStage;

#define NORM2C(c00, c01, c11) PetscSqr(c00) + 2 * PetscSqr(c01) + PetscSqr(c11)

/* residual for C when tested against basis functions */
static void C_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal   c2       = PetscRealPart(constants[C2_ID]);
  const PetscReal   alpha    = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma    = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal   eps      = PetscRealPart(constants[EPS_ID]);
  const PetscScalar gradp[]  = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};
  const PetscScalar crossp[] = {gradp[0] * gradp[0], gradp[0] * gradp[1], gradp[1] * gradp[1]};
  const PetscScalar C00      = u[uOff[C_FIELD_ID]];
  const PetscScalar C01      = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C11      = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar norm     = NORM2C(C00, C01, C11) + eps;
  const PetscScalar nexp     = (gamma - 2.0) / 2.0;
  const PetscScalar fnorm    = PetscPowScalar(norm, nexp);

  for (PetscInt k = 0; k < 3; k++) f0[k] = u_t[uOff[C_FIELD_ID] + k] - c2 * crossp[k] + alpha * fnorm * u[uOff[C_FIELD_ID] + k];
}

/* Jacobian for C against C basis functions */
static void JC_0_c0c0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   alpha  = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma  = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal   eps    = PetscRealPart(constants[EPS_ID]);
  const PetscScalar C00    = u[uOff[C_FIELD_ID]];
  const PetscScalar C01    = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C11    = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar norm   = NORM2C(C00, C01, C11) + eps;
  const PetscScalar nexp   = (gamma - 2.0) / 2.0;
  const PetscScalar fnorm  = PetscPowScalar(norm, nexp);
  const PetscScalar dfnorm = nexp * PetscPowScalar(norm, nexp - 1.0);
  const PetscScalar dC[]   = {2 * C00, 4 * C01, 2 * C11};

  for (PetscInt k = 0; k < 3; k++) {
    for (PetscInt j = 0; j < 3; j++) J[k * 3 + j] = alpha * dfnorm * dC[j] * u[uOff[C_FIELD_ID] + k];
    J[k * 3 + k] += alpha * fnorm + u_tShift;
  }
}

/* Jacobian for C against C basis functions and gradients of P basis functions */
static void JC_0_c0p1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   c2      = PetscRealPart(constants[C2_ID]);
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  J[0] = -c2 * 2 * gradp[0];
  J[1] = 0.0;
  J[2] = -c2 * gradp[1];
  J[3] = -c2 * gradp[0];
  J[4] = 0.0;
  J[5] = -c2 * 2 * gradp[1];
}

/* residual for C when tested against gradients of basis functions */
static void C_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt d = 0; d < 2; d++) f1[k * 2 + d] = PetscSqr(D) * u_x[uOff_x[C_FIELD_ID] + k * 2 + d];
}

/* Jacobian for C against gradients of C basis functions */
static void JC_1_c1c1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt d = 0; d < 2; d++) J[k * (3 + 1) * 2 * 2 + d * 2 + d] = PetscSqr(D);
}

/* residual for P when tested against basis functions.
   The source term always comes from the auxiliary vec because it needs to have zero mean */
static void P_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar S = a[aOff[P_FIELD_ID]];

  f0[0] = -S;
}

static inline void QuadraticRoots(PetscReal a, PetscReal b, PetscReal c, PetscReal x[2]);

/* compute shift to make C positive definite */
static inline PetscReal FIX_C(PetscScalar C00, PetscScalar C01, PetscScalar C11)
{
#if !PetscDefined(USE_COMPLEX)
  PetscReal eigs[2], s = 0.0;

  QuadraticRoots(1, -(C00 + C11), C00 * C11 - PetscSqr(C01), eigs);
  if (eigs[0] < 0 || eigs[1] < 0) s = -PetscMin(eigs[0], eigs[1]) + PETSC_SMALL;
  return s;
#else
  return 0.0;
#endif
}

/* residual for P when tested against gradients of basis functions */
static void P_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal   r       = PetscRealPart(constants[R_ID]);
  const PetscScalar C00     = u[uOff[C_FIELD_ID]] + r;
  const PetscScalar C01     = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10     = C01;
  const PetscScalar C11     = u[uOff[C_FIELD_ID] + 2] + r;
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};
  const PetscBool   fix_c   = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 1.0);
  const PetscScalar s       = fix_c ? FIX_C(C00, C01, C11) : 0.0;

  f1[0] = (C00 + s) * gradp[0] + C01 * gradp[1];
  f1[1] = C10 * gradp[0] + (C11 + s) * gradp[1];
}

/* Same as above for the P-only subproblem for initial conditions: the conductivity values come from the auxiliary vec */
static void P_1_aux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal   r       = PetscRealPart(constants[R_ID]);
  const PetscScalar C00     = a[aOff[C_FIELD_ID]] + r;
  const PetscScalar C01     = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar C10     = C01;
  const PetscScalar C11     = a[aOff[C_FIELD_ID] + 2] + r;
  const PetscScalar gradp[] = {u_x[uOff_x[0]], u_x[uOff_x[0] + 1]};
  const PetscBool   fix_c   = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 1.0);
  const PetscScalar s       = fix_c ? FIX_C(C00, C01, C11) : 0.0;

  f1[0] = (C00 + s) * gradp[0] + C01 * gradp[1];
  f1[1] = C10 * gradp[0] + (C11 + s) * gradp[1];
}

/* Jacobian for P against gradients of P basis functions */
static void JP_1_p1p1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   r     = PetscRealPart(constants[R_ID]);
  const PetscScalar C00   = u[uOff[C_FIELD_ID]] + r;
  const PetscScalar C01   = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10   = C01;
  const PetscScalar C11   = u[uOff[C_FIELD_ID] + 2] + r;
  const PetscBool   fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 0.0);
  const PetscScalar s     = fix_c ? FIX_C(C00, C01, C11) : 0.0;

  J[0] = C00 + s;
  J[1] = C01;
  J[2] = C10;
  J[3] = C11 + s;
}

/* Same as above for the P-only subproblem for initial conditions */
static void JP_1_p1p1_aux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   r     = PetscRealPart(constants[R_ID]);
  const PetscScalar C00   = a[aOff[C_FIELD_ID]] + r;
  const PetscScalar C01   = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar C10   = C01;
  const PetscScalar C11   = a[aOff[C_FIELD_ID] + 2] + r;
  const PetscBool   fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 0.0);
  const PetscScalar s     = fix_c ? FIX_C(C00, C01, C11) : 0.0;

  J[0] = C00 + s;
  J[1] = C01;
  J[2] = C10;
  J[3] = C11 + s;
}

/* Jacobian for P against gradients of P basis functions and C basis functions */
static void JP_1_p1c0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  J[0] = gradp[0];
  J[1] = 0;
  J[2] = gradp[1];
  J[3] = gradp[0];
  J[4] = 0;
  J[5] = gradp[1];
}

/* the source term S(x) = exp(-500*||x - x0||^2) */
static PetscErrorCode source_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscReal *x0 = (PetscReal *)ctx;
  PetscReal  n  = 0;

  for (PetscInt d = 0; d < dim; ++d) n += (x[d] - x0[d]) * (x[d] - x0[d]);
  u[0] = PetscExpReal(-500 * n);
  return PETSC_SUCCESS;
}

static PetscErrorCode source_1(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscScalar     ut[1];
  const PetscReal x0[] = {0.25, 0.25};
  const PetscReal x1[] = {0.75, 0.75};

  PetscCall(source_0(dim, time, x, Nf, ut, (void *)x0));
  PetscCall(source_0(dim, time, x, Nf, u, (void *)x1));
  u[0] += ut[0];
  return PETSC_SUCCESS;
}

/* functionals to be integrated: average -> \int_\Omega u dx */
static void average(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = u[uOff[P_FIELD_ID]];
}

/* stable implementation of roots of a*x^2 + b*x + c = 0 */
static inline void QuadraticRoots(PetscReal a, PetscReal b, PetscReal c, PetscReal x[2])
{
  PetscReal delta = PetscMax(b * b - 4 * a * c, 0); /* eigenvalues symmetric matrix */
  PetscReal temp  = -0.5 * (b + PetscCopysignReal(1.0, b) * PetscSqrtReal(delta));

  x[0] = temp / a;
  x[1] = c / temp;
}

/* functionals to be integrated: energy -> D^2/2 * ||\nabla C||^2 + c^2\nabla p * (r + C) * \nabla p + \alpha/ \gamma * ||C||^\gamma */
static void energy(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  const PetscReal   D         = PetscRealPart(constants[D_ID]);
  const PetscReal   c2        = PetscRealPart(constants[C2_ID]);
  const PetscReal   r         = PetscRealPart(constants[R_ID]);
  const PetscReal   alpha     = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma     = PetscRealPart(constants[GAMMA_ID]);
  const PetscScalar C00       = u[uOff[C_FIELD_ID]];
  const PetscScalar C01       = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10       = C01;
  const PetscScalar C11       = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar gradp[]   = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};
  const PetscScalar gradC00[] = {u_x[uOff_x[C_FIELD_ID] + 0], u_x[uOff_x[C_FIELD_ID] + 1]};
  const PetscScalar gradC01[] = {u_x[uOff_x[C_FIELD_ID] + 2], u_x[uOff_x[C_FIELD_ID] + 3]};
  const PetscScalar gradC11[] = {u_x[uOff_x[C_FIELD_ID] + 4], u_x[uOff_x[C_FIELD_ID] + 5]};
  const PetscScalar normC     = NORM2C(C00, C01, C11);
  const PetscScalar normgradC = NORM2C(gradC00[0], gradC01[0], gradC11[0]) + NORM2C(gradC00[1], gradC01[1], gradC11[1]);
  const PetscScalar nexp      = gamma / 2.0;

  const PetscScalar t0 = PetscSqr(D) / 2.0 * normgradC;
  const PetscScalar t1 = c2 * (gradp[0] * ((C00 + r) * gradp[0] + C01 * gradp[1]) + gradp[1] * (C10 * gradp[0] + (C11 + r) * gradp[1]));
  const PetscScalar t2 = alpha / gamma * PetscPowScalar(normC, nexp);

  obj[0] = t0 + t1 + t2;
}

/* functionals to be integrated: ellipticity_fail -> 0 means C+r is elliptic at quadrature point, otherwise it returns the absolute value of the most negative eigenvalue */
static void ellipticity_fail(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  const PetscReal r   = PetscRealPart(constants[R_ID]);
  const PetscReal C00 = PetscRealPart(u[uOff[C_FIELD_ID]] + r);
  const PetscReal C01 = PetscRealPart(u[uOff[C_FIELD_ID] + 1]);
  const PetscReal C11 = PetscRealPart(u[uOff[C_FIELD_ID] + 2] + r);

  PetscReal eigs[2];
  QuadraticRoots(1, -(C00 + C11), C00 * C11 - PetscSqr(C01), eigs);
  if (eigs[0] < 0 || eigs[1] < 0) obj[0] = -PetscMin(eigs[0], eigs[1]);
  else obj[0] = 0.0;
}

/* initial conditions for C: eq. 16 */
static PetscErrorCode initial_conditions_C_0(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1;
  u[1] = 0;
  u[2] = 1;
  return PETSC_SUCCESS;
}

/* initial conditions for C: eq. 17 */
static PetscErrorCode initial_conditions_C_1(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal x = xx[0];
  const PetscReal y = xx[1];

  u[0] = (2 - PetscAbsReal(x + y)) * PetscExpReal(-10 * PetscAbsReal(x - y));
  u[1] = 0;
  u[2] = (2 - PetscAbsReal(x + y)) * PetscExpReal(-10 * PetscAbsReal(x - y));
  return PETSC_SUCCESS;
}

/* initial conditions for C: eq. 18 */
static PetscErrorCode initial_conditions_C_2(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0;
  u[1] = 0;
  u[2] = 0;
  return PETSC_SUCCESS;
}

/* functionals to be sampled: C * \grad p */
static void flux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscScalar C00     = u[uOff[C_FIELD_ID]];
  const PetscScalar C01     = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10     = C01;
  const PetscScalar C11     = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  f[0] = C00 * gradp[0] + C01 * gradp[1];
  f[1] = C10 * gradp[0] + C11 * gradp[1];
}

/* functionals to be sampled: ||C|| */
static void normc(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscScalar C00 = u[uOff[C_FIELD_ID]];
  const PetscScalar C01 = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C11 = u[uOff[C_FIELD_ID] + 2];

  f[0] = PetscSqrtScalar(NORM2C(C00, C01, C11));
}

/* functionals to be sampled: zero */
static void zero(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  f[0] = 0.0;
}

/* functions to be sampled: zero function */
static PetscErrorCode zerof(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt d = 0; d < Nc; ++d) u[d] = 0.0;
  return PETSC_SUCCESS;
}

/* functions to be sampled: constant function */
static PetscErrorCode constantf(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < Nc; ++d) u[d] = 1.0;
  return PETSC_SUCCESS;
}

/* application context: customizable parameters */
typedef struct {
  PetscReal r;
  PetscReal eps;
  PetscReal alpha;
  PetscReal gamma;
  PetscReal D;
  PetscReal c;
  PetscInt  ic_num;
  PetscInt  source_num;
  PetscReal x0[2];
  PetscBool lump;
  PetscBool amr;
  PetscBool load;
  char      load_filename[PETSC_MAX_PATH_LEN];
  PetscBool save;
  char      save_filename[PETSC_MAX_PATH_LEN];
  PetscInt  save_every;
  PetscBool test_restart;
  PetscBool ellipticity;
  PetscInt  fix_c;
} AppCtx;

/* process command line options */
static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscInt dim = PETSC_STATIC_ARRAY_LENGTH(options->x0);

  PetscFunctionBeginUser;
  options->r            = 1.e-1;
  options->eps          = 1.e-3;
  options->alpha        = 0.75;
  options->gamma        = 0.75;
  options->c            = 5;
  options->D            = 1.e-2;
  options->ic_num       = 0;
  options->source_num   = 0;
  options->x0[0]        = 0.25;
  options->x0[1]        = 0.25;
  options->lump         = PETSC_FALSE;
  options->amr          = PETSC_FALSE;
  options->load         = PETSC_FALSE;
  options->save         = PETSC_FALSE;
  options->save_every   = -1;
  options->test_restart = PETSC_FALSE;
  options->ellipticity  = PETSC_FALSE;
  options->fix_c        = 1; /* 1 means only Jac, 2 means function and Jac */

  PetscOptionsBegin(PETSC_COMM_WORLD, "", __FILE__, "DMPLEX");
  PetscCall(PetscOptionsReal("-alpha", "alpha", __FILE__, options->alpha, &options->alpha, NULL));
  PetscCall(PetscOptionsReal("-gamma", "gamma", __FILE__, options->gamma, &options->gamma, NULL));
  PetscCall(PetscOptionsReal("-c", "c", __FILE__, options->c, &options->c, NULL));
  PetscCall(PetscOptionsReal("-d", "D", __FILE__, options->D, &options->D, NULL));
  PetscCall(PetscOptionsReal("-eps", "eps", __FILE__, options->eps, &options->eps, NULL));
  PetscCall(PetscOptionsReal("-r", "r", __FILE__, options->r, &options->r, NULL));
  PetscCall(PetscOptionsRealArray("-x0", "x0", __FILE__, options->x0, &dim, NULL));
  PetscCall(PetscOptionsInt("-ic_num", "ic_num", __FILE__, options->ic_num, &options->ic_num, NULL));
  PetscCall(PetscOptionsInt("-source_num", "source_num", __FILE__, options->source_num, &options->source_num, NULL));
  PetscCall(PetscOptionsBool("-lump", "use mass lumping", __FILE__, options->lump, &options->lump, NULL));
  PetscCall(PetscOptionsInt("-fix_c", "shift conductivity to always be positive semi-definite", __FILE__, options->fix_c, &options->fix_c, NULL));
  PetscCall(PetscOptionsBool("-amr", "use adaptive mesh refinement", __FILE__, options->amr, &options->amr, NULL));
  PetscCall(PetscOptionsBool("-test_restart", "test restarting files", __FILE__, options->test_restart, &options->test_restart, NULL));
  if (!options->test_restart) {
    PetscCall(PetscOptionsString("-load", "filename with data to be loaded for restarting", __FILE__, options->load_filename, options->load_filename, PETSC_MAX_PATH_LEN, &options->load));
    PetscCall(PetscOptionsString("-save", "filename with data to be saved for restarting", __FILE__, options->save_filename, options->save_filename, PETSC_MAX_PATH_LEN, &options->save));
    if (options->save) PetscCall(PetscOptionsInt("-save_every", "save every n timestep (-1 saves only the last)", __FILE__, options->save_every, &options->save_every, NULL));
  }
  PetscCall(PetscOptionsBool("-monitor_ellipticity", "Dump locations of ellipticity violation", __FILE__, options->ellipticity, &options->ellipticity, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SaveToFile(DM dm, Vec u, const char *filename)
{
#if defined(PETSC_HAVE_HDF5)
  PetscViewerFormat format = PETSC_VIEWER_HDF5_PETSC;
  PetscViewer       viewer;
  DM                cdm       = dm;
  PetscInt          numlevels = 0;

  PetscFunctionBeginUser;
  while (cdm) {
    numlevels++;
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  /* Cannot be set programmatically */
  PetscCall(PetscOptionsInsertString(NULL, "-dm_plex_view_hdf5_storage_version 3.0.0"));
  PetscCall(PetscViewerHDF5Open(PetscObjectComm((PetscObject)dm), filename, FILE_MODE_WRITE, &viewer));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "numlevels", PETSC_INT, &numlevels));
  PetscCall(PetscViewerPushFormat(viewer, format));
  for (PetscInt level = numlevels - 1; level >= 0; level--) {
    PetscInt    cc, rr;
    PetscBool   isRegular, isUniform;
    const char *dmname;
    char        groupname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(groupname, sizeof(groupname), "level_%" PetscInt_FMT, level));
    PetscCall(PetscViewerHDF5PushGroup(viewer, groupname));
    PetscCall(PetscObjectGetName((PetscObject)dm, &dmname));
    PetscCall(DMGetCoarsenLevel(dm, &cc));
    PetscCall(DMGetRefineLevel(dm, &rr));
    PetscCall(DMPlexGetRegularRefinement(dm, &isRegular));
    PetscCall(DMPlexGetRefinementUniform(dm, &isUniform));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "meshname", PETSC_STRING, dmname));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "refinelevel", PETSC_INT, &rr));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "coarsenlevel", PETSC_INT, &cc));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "refRegular", PETSC_BOOL, &isRegular));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "refUniform", PETSC_BOOL, &isUniform));
    PetscCall(DMPlexTopologyView(dm, viewer));
    PetscCall(DMPlexLabelsView(dm, viewer));
    PetscCall(DMPlexCoordinatesView(dm, viewer));
    PetscCall(DMPlexSectionView(dm, viewer, NULL));
    if (level == numlevels - 1) {
      PetscCall(PetscObjectSetName((PetscObject)u, "solution_"));
      PetscCall(DMPlexGlobalVectorView(dm, viewer, NULL, u));
    }
    if (level) {
      PetscInt        cStart, cEnd, ccStart, ccEnd, cpStart;
      DMPolytopeType  ct;
      DMPlexTransform tr;
      DM              sdm;
      PetscScalar    *array;
      PetscSection    section;
      Vec             map;
      IS              gis;
      const PetscInt *gidx;

      PetscCall(DMGetCoarseDM(dm, &cdm));
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
      PetscCall(PetscSectionSetChart(section, cStart, cEnd));
      for (PetscInt c = cStart; c < cEnd; c++) PetscCall(PetscSectionSetDof(section, c, 1));
      PetscCall(PetscSectionSetUp(section));

      PetscCall(DMClone(dm, &sdm));
      PetscCall(PetscObjectSetName((PetscObject)sdm, "pdm"));
      PetscCall(PetscObjectSetName((PetscObject)section, "pdm_section"));
      PetscCall(DMSetLocalSection(sdm, section));
      PetscCall(PetscSectionDestroy(&section));

      PetscCall(DMGetLocalVector(sdm, &map));
      PetscCall(PetscObjectSetName((PetscObject)map, "pdm_map"));
      PetscCall(VecGetArray(map, &array));
      PetscCall(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
      PetscCall(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
      PetscCall(DMPlexTransformSetDM(tr, cdm));
      PetscCall(DMPlexTransformSetFromOptions(tr));
      PetscCall(DMPlexTransformSetUp(tr));
      PetscCall(DMPlexGetHeightStratum(cdm, 0, &ccStart, &ccEnd));
      PetscCall(DMPlexGetChart(cdm, &cpStart, NULL));
      PetscCall(DMPlexCreatePointNumbering(cdm, &gis));
      PetscCall(ISGetIndices(gis, &gidx));
      for (PetscInt c = ccStart; c < ccEnd; c++) {
        PetscInt       *rsize, *rcone, *rornt, Nt;
        DMPolytopeType *rct;
        PetscInt        gnum = gidx[c - cpStart] >= 0 ? gidx[c - cpStart] : -(gidx[c - cpStart] + 1);

        PetscCall(DMPlexGetCellType(cdm, c, &ct));
        PetscCall(DMPlexTransformCellTransform(tr, ct, c, NULL, &Nt, &rct, &rsize, &rcone, &rornt));
        for (PetscInt r = 0; r < rsize[Nt - 1]; ++r) {
          PetscInt pNew;

          PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[Nt - 1], c, r, &pNew));
          array[pNew - cStart] = gnum;
        }
      }
      PetscCall(ISRestoreIndices(gis, &gidx));
      PetscCall(ISDestroy(&gis));
      PetscCall(VecRestoreArray(map, &array));
      PetscCall(DMPlexTransformDestroy(&tr));
      PetscCall(DMPlexSectionView(dm, viewer, sdm));
      PetscCall(DMPlexLocalVectorView(dm, viewer, sdm, map));
      PetscCall(DMRestoreLocalVector(sdm, &map));
      PetscCall(DMDestroy(&sdm));
    }
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    PetscCall(DMGetCoarseDM(dm, &dm));
  }
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Needs HDF5 support. Please reconfigure using --download-hdf5");
#endif
}

static PetscErrorCode LoadFromFile(MPI_Comm comm, const char *filename, DM *odm)
{
#if defined(PETSC_HAVE_HDF5)
  PetscViewerFormat format = PETSC_VIEWER_HDF5_PETSC;
  PetscViewer       viewer;
  DM                dm, cdm = NULL;
  PetscSF           sfXC      = NULL;
  PetscInt          numlevels = -1;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "numlevels", PETSC_INT, NULL, &numlevels));
  PetscCall(PetscViewerPushFormat(viewer, format));
  for (PetscInt level = 0; level < numlevels; level++) {
    char             groupname[PETSC_MAX_PATH_LEN], *dmname;
    PetscSF          sfXB, sfBC, sfG;
    PetscPartitioner part;
    PetscInt         rr, cc;
    PetscBool        isRegular, isUniform;

    PetscCall(DMCreate(comm, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(PetscSNPrintf(groupname, sizeof(groupname), "level_%" PetscInt_FMT, level));
    PetscCall(PetscViewerHDF5PushGroup(viewer, groupname));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "meshname", PETSC_STRING, NULL, &dmname));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "refinelevel", PETSC_INT, NULL, &rr));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "coarsenlevel", PETSC_INT, NULL, &cc));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "refRegular", PETSC_BOOL, NULL, &isRegular));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "refUniform", PETSC_BOOL, NULL, &isUniform));
    PetscCall(PetscObjectSetName((PetscObject)dm, dmname));
    PetscCall(DMPlexTopologyLoad(dm, viewer, &sfXB));
    PetscCall(DMPlexLabelsLoad(dm, viewer, sfXB));
    PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfXB));
    PetscCall(DMPlexGetPartitioner(dm, &part));
    if (!level) { /* partition the coarse level only */
      PetscCall(PetscPartitionerSetFromOptions(part));
    } else { /* propagate partitioning information from coarser to finer level */
      DM           sdm;
      Vec          map;
      PetscSF      sf;
      PetscLayout  clayout;
      PetscScalar *array;
      PetscInt    *cranks_leaf, *cranks_root, *npoints, *points, *ranks, *starts, *gidxs;
      PetscInt     nparts, cStart, cEnd, nr, ccStart, ccEnd, cpStart, cpEnd;
      PetscMPIInt  size, rank;

      PetscCall(DMClone(dm, &sdm));
      PetscCall(PetscObjectSetName((PetscObject)sdm, "pdm"));
      PetscCall(DMPlexSectionLoad(dm, viewer, sdm, sfXB, NULL, &sf));
      PetscCall(DMGetLocalVector(sdm, &map));
      PetscCall(PetscObjectSetName((PetscObject)map, "pdm_map"));
      PetscCall(DMPlexLocalVectorLoad(dm, viewer, sdm, sf, map));

      PetscCallMPI(MPI_Comm_size(comm, &size));
      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      nparts = size;
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      PetscCall(DMPlexGetHeightStratum(cdm, 0, &ccStart, &ccEnd));
      PetscCall(DMPlexGetChart(cdm, &cpStart, &cpEnd));
      PetscCall(PetscCalloc1(nparts, &npoints));
      PetscCall(PetscMalloc4(cEnd - cStart, &points, cEnd - cStart, &ranks, nparts + 1, &starts, cEnd - cStart, &gidxs));
      PetscCall(PetscSFGetGraph(sfXC, &nr, NULL, NULL, NULL));
      PetscCall(PetscMalloc2(cpEnd - cpStart, &cranks_leaf, nr, &cranks_root));
      for (PetscInt c = 0; c < cpEnd - cpStart; c++) cranks_leaf[c] = rank;
      PetscCall(PetscSFReduceBegin(sfXC, MPIU_INT, cranks_leaf, cranks_root, MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(sfXC, MPIU_INT, cranks_leaf, cranks_root, MPI_REPLACE));

      PetscCall(VecGetArray(map, &array));
      for (PetscInt c = 0; c < cEnd - cStart; c++) gidxs[c] = (PetscInt)PetscRealPart(array[c]);
      PetscCall(VecRestoreArray(map, &array));

      PetscCall(PetscLayoutCreate(comm, &clayout));
      PetscCall(PetscLayoutSetLocalSize(clayout, nr));
      PetscCall(PetscSFSetGraphLayout(sf, clayout, cEnd - cStart, NULL, PETSC_OWN_POINTER, gidxs));
      PetscCall(PetscLayoutDestroy(&clayout));

      PetscCall(PetscSFBcastBegin(sf, MPIU_INT, cranks_root, ranks, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf, MPIU_INT, cranks_root, ranks, MPI_REPLACE));
      PetscCall(PetscSFDestroy(&sf));
      PetscCall(PetscFree2(cranks_leaf, cranks_root));
      for (PetscInt c = 0; c < cEnd - cStart; c++) npoints[ranks[c]]++;

      starts[0] = 0;
      for (PetscInt c = 0; c < nparts; c++) starts[c + 1] = starts[c] + npoints[c];
      for (PetscInt c = 0; c < cEnd - cStart; c++) points[starts[ranks[c]]++] = c;
      PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERSHELL));
      PetscCall(PetscPartitionerShellSetPartition(part, nparts, npoints, points));
      PetscCall(PetscFree(npoints));
      PetscCall(PetscFree4(points, ranks, starts, gidxs));
      PetscCall(DMRestoreLocalVector(sdm, &map));
      PetscCall(DMDestroy(&sdm));
    }
    PetscCall(PetscSFDestroy(&sfXC));
    PetscCall(DMPlexDistribute(dm, 0, &sfBC, odm));
    if (*odm) {
      PetscCall(DMDestroy(&dm));
      dm   = *odm;
      *odm = NULL;
      PetscCall(PetscObjectSetName((PetscObject)dm, dmname));
    }
    if (sfBC) PetscCall(PetscSFCompose(sfXB, sfBC, &sfXC));
    else {
      PetscCall(PetscObjectReference((PetscObject)sfXB));
      sfXC = sfXB;
    }
    PetscCall(PetscSFDestroy(&sfXB));
    PetscCall(PetscSFDestroy(&sfBC));
    PetscCall(DMSetCoarsenLevel(dm, cc));
    PetscCall(DMSetRefineLevel(dm, rr));
    PetscCall(DMPlexSetRegularRefinement(dm, isRegular));
    PetscCall(DMPlexSetRefinementUniform(dm, isUniform));
    PetscCall(DMPlexSectionLoad(dm, viewer, NULL, sfXC, &sfG, NULL));
    if (level == numlevels - 1) {
      Vec u;

      PetscCall(DMGetNamedGlobalVector(dm, "solution_", &u));
      PetscCall(PetscObjectSetName((PetscObject)u, "solution_"));
      PetscCall(DMPlexGlobalVectorLoad(dm, viewer, NULL, sfG, u));
      PetscCall(DMRestoreNamedGlobalVector(dm, "solution_", &u));
    }
    PetscCall(PetscFree(dmname));
    PetscCall(PetscSFDestroy(&sfG));
    PetscCall(DMSetCoarseDM(dm, cdm));
    PetscCall(DMDestroy(&cdm));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    cdm = dm;
  }
  *odm = dm;
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSFDestroy(&sfXC));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Needs HDF5 support. Please reconfigure using --download-hdf5");
#endif
}

/* Project source function and make it zero-mean */
static PetscErrorCode ProjectSource(DM dm, PetscReal time, AppCtx *ctx)
{
  PetscInt    id = C_FIELD_ID;
  DM          dmAux;
  Vec         u, lu;
  IS          is;
  void       *ctxs[NUM_FIELDS];
  PetscScalar vals[NUM_FIELDS];
  PetscDS     ds;
  PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

  PetscFunctionBeginUser;
  switch (ctx->source_num) {
  case 0:
    funcs[P_FIELD_ID] = source_0;
    ctxs[P_FIELD_ID]  = ctx->x0;
    break;
  case 1:
    funcs[P_FIELD_ID] = source_1;
    ctxs[P_FIELD_ID]  = NULL;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unknown source");
  }
  funcs[C_FIELD_ID] = zerof;
  ctxs[C_FIELD_ID]  = NULL;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMProjectFunction(dm, time, funcs, ctxs, INSERT_ALL_VALUES, u));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, average));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals, NULL));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));
  PetscCall(VecShift(u, -vals[P_FIELD_ID]));
  PetscCall(DMCreateSubDM(dm, 1, &id, &is, NULL));
  PetscCall(VecISSet(u, is, 0));
  PetscCall(ISDestroy(&is));

  /* Attach source vector as auxiliary vector:
     Use a different DM to break ref cycles */
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMCopyDisc(dm, dmAux));
  PetscCall(DMCreateLocalVector(dmAux, &lu));
  PetscCall(DMDestroy(&dmAux));
  PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, lu));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, lu));
  PetscCall(VecViewFromOptions(lu, NULL, "-aux_view"));
  PetscCall(VecDestroy(&lu));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* callback for the creation of the potential null space */
static PetscErrorCode CreatePotentialNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
  Vec vec;
  PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zerof};

  PetscFunctionBeginUser;
  funcs[nfield] = constantf;
  PetscCall(DMCreateGlobalVector(dm, &vec));
  PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec));
  PetscCall(VecNormalize(vec, NULL));
  PetscCall(PetscObjectSetName((PetscObject)vec, "Potential Null Space"));
  PetscCall(VecViewFromOptions(vec, NULL, "-potential_nullspace_view"));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullSpace));
  /* break ref cycles */
  PetscCall(VecSetDM(vec, NULL));
  PetscCall(VecDestroy(&vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetLumpedMass(DM dm, PetscBool local, Vec *lumped_mass)
{
  PetscBool has;

  PetscFunctionBeginUser;
  if (local) {
    PetscCall(DMHasNamedLocalVector(dm, "lumped_mass", &has));
    PetscCall(DMGetNamedLocalVector(dm, "lumped_mass", lumped_mass));
  } else {
    PetscCall(DMHasNamedGlobalVector(dm, "lumped_mass", &has));
    PetscCall(DMGetNamedGlobalVector(dm, "lumped_mass", lumped_mass));
  }
  if (!has) {
    Vec w;
    IS  is;

    PetscCall(PetscObjectQuery((PetscObject)dm, "IS potential", (PetscObject *)&is));
    if (!is) {
      PetscInt fields[NUM_FIELDS] = {C_FIELD_ID, P_FIELD_ID};

      PetscCall(DMCreateSubDM(dm, NUM_FIELDS - 1, fields + 1, &is, NULL));
      PetscCall(PetscObjectCompose((PetscObject)dm, "IS potential", (PetscObject)is));
      PetscCall(PetscObjectDereference((PetscObject)is));
    }
    if (local) {
      Vec w2, wg;

      PetscCall(DMCreateMassMatrixLumped(dm, &w, NULL));
      PetscCall(DMGetGlobalVector(dm, &wg));
      PetscCall(DMGetLocalVector(dm, &w2));
      PetscCall(VecSet(w2, 0.0));
      PetscCall(VecSet(wg, 1.0));
      PetscCall(VecISSet(wg, is, 0.0));
      PetscCall(DMGlobalToLocal(dm, wg, INSERT_VALUES, w2));
      PetscCall(VecPointwiseMult(w, w, w2));
      PetscCall(DMRestoreGlobalVector(dm, &wg));
      PetscCall(DMRestoreLocalVector(dm, &w2));
    } else {
      PetscCall(DMCreateMassMatrixLumped(dm, NULL, &w));
      PetscCall(VecISSet(w, is, 0.0));
    }
    PetscCall(VecCopy(w, *lumped_mass));
    PetscCall(VecDestroy(&w));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRestoreLumpedMass(DM dm, PetscBool local, Vec *lumped_mass)
{
  PetscFunctionBeginUser;
  if (local) PetscCall(DMRestoreNamedLocalVector(dm, "lumped_mass", lumped_mass));
  else PetscCall(DMRestoreNamedGlobalVector(dm, "lumped_mass", lumped_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* callbacks for lumped mass matrix residual and Jacobian */
static PetscErrorCode DMPlexTSComputeIFunctionFEM_Lumped(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *user)
{
  Vec work, local_lumped_mass;

  PetscFunctionBeginUser;
  PetscCall(DMGetLumpedMass(dm, PETSC_TRUE, &local_lumped_mass));
  PetscCall(DMGetLocalVector(dm, &work));
  PetscCall(VecSet(work, 0.0));
  PetscCall(DMPlexTSComputeIFunctionFEM(dm, time, locX, work, locF, user));
  PetscCall(VecPointwiseMult(work, locX_t, local_lumped_mass));
  PetscCall(VecAXPY(locF, 1.0, work));
  PetscCall(DMRestoreLocalVector(dm, &work));
  PetscCall(DMRestoreLumpedMass(dm, PETSC_TRUE, &local_lumped_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTSComputeIJacobianFEM_Lumped(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  Vec lumped_mass, work;

  PetscFunctionBeginUser;
  // XXX CHECK DIRK
  PetscCall(DMGetLumpedMass(dm, PETSC_FALSE, &lumped_mass));
  PetscCall(DMPlexTSComputeIJacobianFEM(dm, time, locX, locX_t, 0.0, Jac, JacP, user));
  PetscCall(DMGetGlobalVector(dm, &work));
  PetscCall(VecAXPBY(work, X_tShift, 0.0, lumped_mass));
  PetscCall(MatDiagonalSet(JacP, work, ADD_VALUES));
  PetscCall(DMRestoreGlobalVector(dm, &work));
  PetscCall(DMRestoreLumpedMass(dm, PETSC_FALSE, &lumped_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* customize residuals and Jacobians */
static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS     ds;
  PetscInt    cdim, dim;
  PetscScalar constants[NUM_CONSTANTS];

  PetscFunctionBeginUser;
  constants[R_ID]     = ctx->r;
  constants[EPS_ID]   = ctx->eps;
  constants[ALPHA_ID] = ctx->alpha;
  constants[GAMMA_ID] = ctx->gamma;
  constants[D_ID]     = ctx->D;
  constants[C2_ID]    = ctx->c * ctx->c;
  constants[FIXC_ID]  = ctx->fix_c;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(dim == 2 && cdim == 2, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only for 2D meshes");
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  PetscCall(PetscDSSetImplicit(ds, C_FIELD_ID, PETSC_TRUE));
  PetscCall(PetscDSSetImplicit(ds, P_FIELD_ID, PETSC_TRUE));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));
  PetscCall(PetscDSSetResidual(ds, C_FIELD_ID, C_0, C_1));
  PetscCall(PetscDSSetResidual(ds, P_FIELD_ID, P_0, P_1));
  PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, C_FIELD_ID, JC_0_c0c0, NULL, NULL, JC_1_c1c1));
  PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, P_FIELD_ID, NULL, JC_0_c0p1, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, C_FIELD_ID, NULL, NULL, JP_1_p1c0, NULL));
  PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, P_FIELD_ID, NULL, NULL, NULL, JP_1_p1p1));

  /* Attach potential nullspace */
  PetscCall(DMSetNullSpaceConstructor(dm, P_FIELD_ID, CreatePotentialNullSpace));

  /* Attach source function as auxiliary vector */
  PetscCall(ProjectSource(dm, 0, ctx));

  /* Add callbacks */
  if (ctx->lump) {
    PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM_Lumped, NULL));
    PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM_Lumped, NULL));
  } else {
    PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL));
    PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL));
  }
  /* This is not really needed because we use Neumann boundaries */
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* setup discrete spaces and residuals */
static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM           plex, cdm = dm;
  PetscFE      feC, feP;
  PetscBool    simplex;
  PetscInt     dim;
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm);
  MatNullSpace nsp;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &simplex, 1, MPIU_BOOL, MPI_LOR, comm));
  PetscCall(DMDestroy(&plex));

  /* We model Cij with Cij = Cji -> dim*(dim+1)/2 components */
  PetscCall(PetscFECreateDefault(comm, dim, (dim * (dim + 1)) / 2, simplex, "c_", -1, &feC));
  PetscCall(PetscObjectSetName((PetscObject)feC, "conductivity"));
  PetscCall(PetscFECreateDefault(comm, dim, 1, simplex, "p_", -1, &feP));
  PetscCall(PetscObjectSetName((PetscObject)feP, "potential"));
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nsp));
  PetscCall(PetscObjectCompose((PetscObject)feP, "nullspace", (PetscObject)nsp));
  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(PetscFECopyQuadrature(feP, feC));

  PetscCall(DMSetNumFields(dm, 2));
  PetscCall(DMSetField(dm, C_FIELD_ID, NULL, (PetscObject)feC));
  PetscCall(DMSetField(dm, P_FIELD_ID, NULL, (PetscObject)feP));
  PetscCall(PetscFEDestroy(&feC));
  PetscCall(PetscFEDestroy(&feP));
  PetscCall(DMCreateDS(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(SetupProblem(cdm, ctx));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create mesh by command line options */
static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  if (ctx->load) {
    PetscInt  refine = 0;
    PetscBool isHierarchy;
    DM       *dms;
    char      typeName[256];
    PetscBool flg;

    PetscCall(LoadFromFile(comm, ctx->load_filename, dm));
    PetscOptionsBegin(comm, "", "Additional mesh options", "DMPLEX");
    PetscCall(PetscOptionsFList("-dm_mat_type", "Matrix type used for created matrices", "DMSetMatType", MatList, MATAIJ, typeName, sizeof(typeName), &flg));
    if (flg) PetscCall(DMSetMatType(*dm, typeName));
    PetscCall(PetscOptionsBoundedInt("-dm_refine", "The number of uniform refinements", "DMCreate", refine, &refine, NULL, 0));
    PetscCall(PetscOptionsBoundedInt("-dm_refine_hierarchy", "The number of uniform refinements", "DMCreate", refine, &refine, &isHierarchy, 0));
    PetscOptionsEnd();
    if (refine) {
      PetscCall(SetupDiscretization(*dm, ctx));
      PetscCall(DMPlexSetRefinementUniform(*dm, PETSC_TRUE));
    }
    PetscCall(PetscCalloc1(refine, &dms));
    if (isHierarchy) PetscCall(DMRefineHierarchy(*dm, refine, dms));
    for (PetscInt r = 0; r < refine; r++) {
      Mat M;
      DM  dmr = dms[r];
      Vec u, ur;

      if (!isHierarchy) {
        PetscCall(DMRefine(*dm, PetscObjectComm((PetscObject)*dm), &dmr));
        PetscCall(DMSetCoarseDM(dmr, *dm));
      }
      PetscCall(DMCreateInterpolation(*dm, dmr, &M, NULL));
      PetscCall(DMGetNamedGlobalVector(*dm, "solution_", &u));
      PetscCall(DMGetNamedGlobalVector(dmr, "solution_", &ur));
      PetscCall(MatInterpolate(M, u, ur));
      PetscCall(DMRestoreNamedGlobalVector(*dm, "solution_", &u));
      PetscCall(DMRestoreNamedGlobalVector(dmr, "solution_", &ur));
      PetscCall(MatDestroy(&M));
      if (!isHierarchy) PetscCall(DMSetCoarseDM(dmr, NULL));
      PetscCall(DMDestroy(dm));
      *dm = dmr;
    }
    if (refine && !isHierarchy) PetscCall(DMSetRefineLevel(*dm, 0));
    PetscCall(PetscFree(dms));
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(DMLocalizeCoordinates(*dm));
    {
      char      convType[256];
      PetscBool flg;
      PetscOptionsBegin(comm, "", "Additional mesh options", "DMPLEX");
      PetscCall(PetscOptionsFList("-dm_plex_convert_type", "Convert DMPlex to another format", __FILE__, DMList, DMPLEX, convType, 256, &flg));
      PetscOptionsEnd();
      if (flg) {
        DM dmConv;
        PetscCall(DMConvert(*dm, convType, &dmConv));
        if (dmConv) {
          PetscCall(DMDestroy(dm));
          *dm = dmConv;
          PetscCall(DMSetFromOptions(*dm));
          PetscCall(DMSetUp(*dm));
        }
      }
    }
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Make potential field zero mean */
static PetscErrorCode ZeroMeanPotential(DM dm, Vec u)
{
  PetscScalar vals[NUM_FIELDS];
  PetscDS     ds;
  IS          is;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)dm, "IS potential", (PetscObject *)&is));
  PetscCheck(is, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing potential IS");
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, average));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals, NULL));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));
  PetscCall(VecISShift(u, is, -vals[P_FIELD_ID]));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute initial conditions and exclude potential from local truncation error
   Since we are solving a DAE, once the initial conditions for the differential
   variables are set, we need to compute the corresponding value for the
   algebraic variables. We do so by creating a subDM for the potential only
   and solve a static problem with SNES */
static PetscErrorCode SetInitialConditionsAndTolerances(TS ts, PetscInt nv, Vec vecs[], PetscBool valid)
{
  DM         dm;
  Vec        tu, u, p, lsource, subaux, vatol, vrtol;
  PetscReal  t, atol, rtol;
  PetscInt   fields[NUM_FIELDS] = {C_FIELD_ID, P_FIELD_ID};
  IS         isp;
  DM         dmp;
  VecScatter sctp = NULL;
  PetscDS    ds;
  SNES       snes;
  KSP        ksp;
  PC         pc;
  AppCtx    *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetApplicationContext(ts, &ctx));
  if (valid) {
    PetscCall(DMCreateSubDM(dm, NUM_FIELDS - 1, fields + 1, &isp, NULL));
    PetscCall(PetscObjectCompose((PetscObject)dm, "IS potential", (PetscObject)isp));
    PetscCall(DMCreateGlobalVector(dm, &vatol));
    PetscCall(DMCreateGlobalVector(dm, &vrtol));
    PetscCall(TSGetTolerances(ts, &atol, NULL, &rtol, NULL));
    PetscCall(VecSet(vatol, atol));
    PetscCall(VecISSet(vatol, isp, -1));
    PetscCall(VecSet(vrtol, rtol));
    PetscCall(VecISSet(vrtol, isp, -1));
    PetscCall(TSSetTolerances(ts, atol, vatol, rtol, vrtol));
    PetscCall(VecDestroy(&vatol));
    PetscCall(VecDestroy(&vrtol));
    PetscCall(ISDestroy(&isp));
    for (PetscInt i = 0; i < nv; i++) { PetscCall(ZeroMeanPotential(dm, vecs[i])); }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMCreateSubDM(dm, NUM_FIELDS - 1, fields + 1, &isp, &dmp));
  PetscCall(PetscObjectCompose((PetscObject)dm, "IS potential", (PetscObject)isp));
  PetscCall(DMSetMatType(dmp, MATAIJ));
  PetscCall(DMGetDS(dmp, &ds));
  //PetscCall(PetscDSSetResidual(ds, 0, P_0, P_1_aux));
  PetscCall(PetscDSSetResidual(ds, 0, 0, P_1_aux));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, JP_1_p1p1_aux));
  PetscCall(DMPlexSetSNESLocalFEM(dmp, PETSC_FALSE, NULL));

  PetscCall(DMCreateGlobalVector(dmp, &p));

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dmp), &snes));
  PetscCall(SNESSetDM(snes, dmp));
  PetscCall(SNESSetOptionsPrefix(snes, "initial_"));
  PetscCall(SNESSetErrorIfNotConverged(snes, PETSC_TRUE));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPFGMRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCGAMG));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetUp(snes));

  /* Loop over input vectors and compute corresponding potential */
  for (PetscInt i = 0; i < nv; i++) {
    PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

    u = vecs[i];
    if (!valid) { /* Assumes entries in u are not valid */
      PetscCall(TSGetTime(ts, &t));
      switch (ctx->ic_num) {
      case 0:
        funcs[C_FIELD_ID] = initial_conditions_C_0;
        break;
      case 1:
        funcs[C_FIELD_ID] = initial_conditions_C_1;
        break;
      case 2:
        funcs[C_FIELD_ID] = initial_conditions_C_2;
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Unknown IC");
      }
      funcs[P_FIELD_ID] = zerof;
      PetscCall(DMProjectFunction(dm, t, funcs, NULL, INSERT_ALL_VALUES, u));
    }

    /* pass conductivity and source information via auxiliary data */
    PetscCall(DMGetGlobalVector(dm, &tu));
    PetscCall(VecCopy(u, tu));
    PetscCall(VecISSet(tu, isp, 0.0));
    PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &lsource));
    PetscCall(DMCreateLocalVector(dm, &subaux));
    PetscCall(DMGlobalToLocal(dm, tu, INSERT_VALUES, subaux));
    PetscCall(DMRestoreGlobalVector(dm, &tu));
    PetscCall(VecAXPY(subaux, 1.0, lsource));
    PetscCall(VecViewFromOptions(subaux, NULL, "-initial_aux_view"));
    PetscCall(DMSetAuxiliaryVec(dmp, NULL, 0, 0, subaux));
    PetscCall(VecDestroy(&subaux));

    /* solve the subproblem */
    if (!sctp) PetscCall(VecScatterCreate(u, isp, p, NULL, &sctp));
    PetscCall(VecScatterBegin(sctp, u, p, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sctp, u, p, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(SNESSolve(snes, NULL, p));

    /* scatter from potential only to full space */
    PetscCall(VecScatterBegin(sctp, p, u, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(sctp, p, u, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(ZeroMeanPotential(dm, u));
  }
  PetscCall(VecDestroy(&p));
  PetscCall(DMDestroy(&dmp));
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecScatterDestroy(&sctp));

  /* exclude potential from computation of the LTE */
  PetscCall(DMCreateGlobalVector(dm, &vatol));
  PetscCall(DMCreateGlobalVector(dm, &vrtol));
  PetscCall(TSGetTolerances(ts, &atol, NULL, &rtol, NULL));
  PetscCall(VecSet(vatol, atol));
  PetscCall(VecISSet(vatol, isp, -1));
  PetscCall(VecSet(vrtol, rtol));
  PetscCall(VecISSet(vrtol, isp, -1));
  PetscCall(TSSetTolerances(ts, atol, vatol, rtol, vrtol));
  PetscCall(VecDestroy(&vatol));
  PetscCall(VecDestroy(&vrtol));
  PetscCall(ISDestroy(&isp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* mesh adaption context */
typedef struct {
  VecTagger refineTag;
  DMLabel   adaptLabel;
  PetscInt  cnt;
} AdaptCtx;

static PetscErrorCode ResizeSetUp(TS ts, PetscInt nstep, PetscReal time, Vec u, PetscBool *resize, void *vctx)
{
  AdaptCtx *ctx = (AdaptCtx *)vctx;
  Vec       ellVecCells, ellVecCellsF;
  DM        dm, plex;
  PetscDS   ds;
  PetscReal norm;
  PetscInt  cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd));
  PetscCall(DMDestroy(&plex));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ts), NUM_FIELDS * (cEnd - cStart), PETSC_DECIDE, &ellVecCellsF));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ts), cEnd - cStart, PETSC_DECIDE, &ellVecCells));
  PetscCall(VecSetBlockSize(ellVecCellsF, NUM_FIELDS));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, ellipticity_fail));
  PetscCall(DMPlexComputeCellwiseIntegralFEM(dm, u, ellVecCellsF, NULL));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
  PetscCall(VecStrideGather(ellVecCellsF, C_FIELD_ID, ellVecCells, INSERT_VALUES));
  PetscCall(VecDestroy(&ellVecCellsF));
  PetscCall(VecNorm(ellVecCells, NORM_1, &norm));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "STEP %d norm %g\n", (int)nstep, (double)norm));
  if (norm && !ctx->cnt) {
    IS refineIS;

    *resize = PETSC_TRUE;
    if (!ctx->refineTag) {
      VecTaggerBox refineBox;
      refineBox.min = PETSC_MACHINE_EPSILON;
      refineBox.max = PETSC_MAX_REAL;

      PetscCall(VecTaggerCreate(PETSC_COMM_SELF, &ctx->refineTag));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ctx->refineTag, "refine_"));
      PetscCall(VecTaggerSetType(ctx->refineTag, VECTAGGERABSOLUTE));
      PetscCall(VecTaggerAbsoluteSetBox(ctx->refineTag, &refineBox));
      PetscCall(VecTaggerSetFromOptions(ctx->refineTag));
      PetscCall(VecTaggerSetUp(ctx->refineTag));
      PetscCall(PetscObjectViewFromOptions((PetscObject)ctx->refineTag, NULL, "-tag_view"));
    }
    PetscCall(DMLabelDestroy(&ctx->adaptLabel));
    PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)ts), "adapt", &ctx->adaptLabel));
    PetscCall(VecTaggerComputeIS(ctx->refineTag, ellVecCells, &refineIS, NULL));
    PetscCall(DMLabelSetStratumIS(ctx->adaptLabel, DM_ADAPT_REFINE, refineIS));
    PetscCall(ISDestroy(&refineIS));
#if 0
    void (*funcs[NUM_FIELDS])(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);
    Vec ellVec;

    funcs[P_FIELD_ID] = ellipticity_fail;
    funcs[C_FIELD_ID] = NULL;

    PetscCall(DMGetGlobalVector(dm, &ellVec));
    PetscCall(DMProjectField(dm, 0, u, funcs, INSERT_VALUES, ellVec));
    PetscCall(VecViewFromOptions(ellVec,NULL,"-view_amr_ell"));
    PetscCall(DMRestoreGlobalVector(dm, &ellVec));
#endif
    ctx->cnt++;
  } else {
    ctx->cnt = 0;
  }
  PetscCall(VecDestroy(&ellVecCells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ResizeTransfer(TS ts, PetscInt nv, Vec vecsin[], Vec vecsout[], void *vctx)
{
  AdaptCtx *actx = (AdaptCtx *)vctx;
  AppCtx   *ctx;
  DM        dm, adm;
  PetscReal time;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCheck(actx->adaptLabel, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE, "Missing adaptLabel");
  PetscCall(DMAdaptLabel(dm, actx->adaptLabel, &adm));
  PetscCheck(adm, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE, "Missing adapted DM");
  PetscCall(TSGetTime(ts, &time));
  PetscCall(DMLabelDestroy(&actx->adaptLabel));
  for (PetscInt i = 0; i < nv; i++) {
    PetscCall(DMCreateGlobalVector(adm, &vecsout[i]));
    PetscCall(DMForestTransferVec(dm, vecsin[i], adm, vecsout[i], PETSC_TRUE, time));
  }
  PetscCall(DMForestSetAdaptivityForest(adm, NULL));
  PetscCall(DMSetCoarseDM(adm, NULL));
  PetscCall(DMSetLocalSection(adm, NULL));
  PetscCall(TSSetDM(ts, adm));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetApplicationContext(ts, &ctx));
  PetscCall(DMSetNullSpaceConstructor(adm, P_FIELD_ID, CreatePotentialNullSpace));
  PetscCall(ProjectSource(adm, time, ctx));
  PetscCall(SetInitialConditionsAndTolerances(ts, nv, vecsout, PETSC_TRUE));
  PetscCall(DMDestroy(&adm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OutputVTK(DM dm, const char *filename, PetscViewer *viewer)
{
  PetscFunctionBeginUser;
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)dm), viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERVTK));
  PetscCall(PetscViewerFileSetName(*viewer, filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Monitor relevant functionals */
static PetscErrorCode Monitor(TS ts, PetscInt stepnum, PetscReal time, Vec u, void *vctx)
{
  PetscScalar vals[2 * NUM_FIELDS];
  DM          dm;
  PetscDS     ds;
  AppCtx     *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetApplicationContext(ts, &ctx));
  PetscCall(DMGetDS(dm, &ds));

  /* monitor energy and potential average */
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, average));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals, NULL));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));

  /* monitor ellipticity_fail */
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, ellipticity_fail));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals + NUM_FIELDS, NULL));
  if (ctx->ellipticity) {
    void (*funcs[NUM_FIELDS])(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);
    Vec         ellVec;
    PetscViewer viewer;
    char        filename[PETSC_MAX_PATH_LEN];

    funcs[P_FIELD_ID] = ellipticity_fail;
    funcs[C_FIELD_ID] = NULL;

    PetscCall(DMGetGlobalVector(dm, &ellVec));
    PetscCall(DMProjectField(dm, 0, u, funcs, INSERT_VALUES, ellVec));
    PetscCall(PetscSNPrintf(filename, sizeof filename, "ellipticity_fail-%03" PetscInt_FMT ".vtu", stepnum));
    PetscCall(OutputVTK(dm, filename, &viewer));
    PetscCall(VecView(ellVec, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(DMRestoreGlobalVector(dm, &ellVec));
  }
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "%4" PetscInt_FMT " TS: time %g, energy %g, intp %g, ell %g\n", stepnum, (double)time, (double)PetscRealPart(vals[C_FIELD_ID]), (double)PetscRealPart(vals[P_FIELD_ID]), (double)PetscRealPart(vals[NUM_FIELDS + C_FIELD_ID])));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Save restart information */
static PetscErrorCode MonitorSave(TS ts, PetscInt steps, PetscReal time, Vec u, void *vctx)
{
  DM                dm;
  AppCtx           *ctx        = (AppCtx *)vctx;
  PetscInt          save_every = ctx->save_every;
  TSConvergedReason reason;

  PetscFunctionBeginUser;
  if (!ctx->save) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetConvergedReason(ts, &reason));
  if ((save_every > 0 && steps % save_every == 0) || (save_every == -1 && reason) || save_every < -1) PetscCall(SaveToFile(dm, u, ctx->save_filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Make potential zero mean after SNES solve */
static PetscErrorCode PostStage(TS ts, PetscReal stagetime, PetscInt stageindex, Vec *Y)
{
  DM       dm;
  Vec      u = Y[stageindex];
  SNES     snes;
  PetscInt nits, lits, stepnum;
  AppCtx  *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(ZeroMeanPotential(dm, u));

  PetscCall(TSGetApplicationContext(ts, &ctx));
  if (ctx->test_restart) PetscFunctionReturn(PETSC_SUCCESS);

  /* monitor linear and nonlinear iterations */
  PetscCall(TSGetStepNumber(ts, &stepnum));
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetIterationNumber(snes, &nits));
  PetscCall(SNESGetLinearSolveIterations(snes, &lits));

  /* if function evals in TSDIRK are zero in the first stage, it is FSAL */
  if (stageindex == 0) {
    PetscBool dirk;
    PetscInt  nf;

    PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSDIRK, &dirk));
    PetscCall(SNESGetNumberFunctionEvals(snes, &nf));
    if (dirk && nf == 0) nits = 0;
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "         step %" PetscInt_FMT " stage %" PetscInt_FMT " nonlinear its %" PetscInt_FMT ", linear its %" PetscInt_FMT "\n", stepnum, stageindex, nits, lits));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecViewFlux(Vec u, const char *opts)
{
  Vec       fluxVec;
  DM        dmFlux, dm, plex;
  PetscInt  dim;
  PetscFE   feC, feFluxC, feNormC;
  PetscBool simplex, has;

  void (*funcs[])(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]) = {normc, flux};

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsHasName(NULL, NULL, opts, &has));
  if (!has) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetDM(u, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetField(dm, C_FIELD_ID, NULL, (PetscObject *)&feC));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, dim, simplex, "flux_", -1, &feFluxC));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, "normc_", -1, &feNormC));
  PetscCall(PetscFECopyQuadrature(feC, feFluxC));
  PetscCall(PetscFECopyQuadrature(feC, feNormC));
  PetscCall(DMClone(dm, &dmFlux));
  PetscCall(DMSetNumFields(dmFlux, 1));
  PetscCall(DMSetField(dmFlux, 0, NULL, (PetscObject)feNormC));
  /* paraview segfaults! */
  //PetscCall(DMSetField(dmFlux, 1, NULL, (PetscObject)feFluxC));
  PetscCall(DMCreateDS(dmFlux));
  PetscCall(PetscFEDestroy(&feFluxC));
  PetscCall(PetscFEDestroy(&feNormC));

  PetscCall(DMGetGlobalVector(dmFlux, &fluxVec));
  PetscCall(DMProjectField(dmFlux, 0.0, u, funcs, INSERT_VALUES, fluxVec));
  PetscCall(VecViewFromOptions(fluxVec, NULL, opts));
  PetscCall(DMRestoreGlobalVector(dmFlux, &fluxVec));
  PetscCall(DMDestroy(&dmFlux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Run(MPI_Comm comm, AppCtx *ctx)
{
  DM        dm;
  TS        ts;
  Vec       u;
  AdaptCtx *actx;
  PetscBool flg;

  PetscFunctionBeginUser;
  if (ctx->test_restart) {
    PetscViewer subviewer;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, comm, &subviewer));
    if (ctx->load) PetscCall(PetscViewerASCIIPrintf(subviewer, "rank %d loading from %s\n", rank, ctx->load_filename));
    if (ctx->save) PetscCall(PetscViewerASCIIPrintf(subviewer, "rank %d saving to %s\n", rank, ctx->save_filename));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, comm, &subviewer));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  } else {
    PetscCall(PetscPrintf(comm, "----------------------------\n"));
    PetscCall(PetscPrintf(comm, "Simulation parameters:\n"));
    PetscCall(PetscPrintf(comm, "  r    : %g\n", (double)ctx->r));
    PetscCall(PetscPrintf(comm, "  eps  : %g\n", (double)ctx->eps));
    PetscCall(PetscPrintf(comm, "  alpha: %g\n", (double)ctx->alpha));
    PetscCall(PetscPrintf(comm, "  gamma: %g\n", (double)ctx->gamma));
    PetscCall(PetscPrintf(comm, "  D    : %g\n", (double)ctx->D));
    PetscCall(PetscPrintf(comm, "  c    : %g\n", (double)ctx->c));
    if (ctx->load) PetscCall(PetscPrintf(comm, "  load : %s\n", ctx->load_filename));
    else PetscCall(PetscPrintf(comm, "  IC   : %" PetscInt_FMT "\n", ctx->ic_num));
    PetscCall(PetscPrintf(comm, "  S    : %" PetscInt_FMT "\n", ctx->source_num));
    PetscCall(PetscPrintf(comm, "  x0   : (%g,%g)\n", (double)ctx->x0[0], (double)ctx->x0[1]));
    PetscCall(PetscPrintf(comm, "----------------------------\n"));
  }

  if (!ctx->test_restart) PetscCall(PetscLogStagePush(SetupStage));
  PetscCall(CreateMesh(comm, &dm, ctx));
  PetscCall(SetupDiscretization(dm, ctx));

  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetApplicationContext(ts, ctx));

  PetscCall(TSSetDM(ts, dm));
  if (ctx->test_restart) {
    PetscViewer subviewer;
    PetscMPIInt rank;
    PetscInt    level;

    PetscCall(DMGetRefineLevel(dm, &level));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, comm, &subviewer));
    PetscCall(PetscViewerASCIIPrintf(subviewer, "rank %d DM refinement level %" PetscInt_FMT "\n", rank, level));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, comm, &subviewer));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  }

  if (ctx->test_restart) PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  if (!ctx->test_restart) PetscCall(TSMonitorSet(ts, Monitor, NULL, NULL));
  PetscCall(TSMonitorSet(ts, MonitorSave, ctx, NULL));
  PetscCall(PetscNew(&actx));
  if (ctx->amr) PetscCall(TSSetResize(ts, PETSC_TRUE, ResizeSetUp, ResizeTransfer, actx));
  PetscCall(TSSetPostStage(ts, PostStage));
  PetscCall(TSSetMaxSNESFailures(ts, -1));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution_"));
  PetscCall(DMHasNamedGlobalVector(dm, "solution_", &flg));
  if (flg) {
    Vec ru;

    PetscCall(DMGetNamedGlobalVector(dm, "solution_", &ru));
    PetscCall(VecCopy(ru, u));
    PetscCall(DMRestoreNamedGlobalVector(dm, "solution_", &ru));
  }
  PetscCall(SetInitialConditionsAndTolerances(ts, 1, &u, PETSC_FALSE));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));
  if (!ctx->test_restart) PetscCall(PetscLogStagePop());

  if (!ctx->test_restart) PetscCall(PetscLogStagePush(SolveStage));
  PetscCall(TSSolve(ts, NULL));
  if (!ctx->test_restart) PetscCall(PetscLogStagePop());

  PetscCall(TSGetSolution(ts, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-final_view"));
  PetscCall(VecViewFlux(u, "-final_flux_view"));

  PetscCall(TSDestroy(&ts));
  PetscCall(VecTaggerDestroy(&actx->refineTag));
  PetscCall(PetscFree(actx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(&ctx));
  PetscCall(PetscLogStageRegister("Setup", &SetupStage));
  PetscCall(PetscLogStageRegister("Solve", &SolveStage));
  if (ctx.test_restart) { /* Test sequences of save and loads */
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    /* test saving */
    ctx.load = PETSC_FALSE;
    ctx.save = PETSC_TRUE;
    /* sequential save */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test sequential save\n"));
    PetscCall(PetscSNPrintf(ctx.save_filename, sizeof(ctx.save_filename), "test_ex30_seq_%d.h5", rank));
    PetscCall(Run(PETSC_COMM_SELF, &ctx));
    /* parallel save */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test parallel save\n"));
    PetscCall(PetscSNPrintf(ctx.save_filename, sizeof(ctx.save_filename), "test_ex30_par.h5"));
    PetscCall(Run(PETSC_COMM_WORLD, &ctx));

    /* test loading */
    ctx.load = PETSC_TRUE;
    ctx.save = PETSC_FALSE;
    /* sequential and parallel runs from sequential save */
    PetscCall(PetscSNPrintf(ctx.load_filename, sizeof(ctx.load_filename), "test_ex30_seq_0.h5"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test sequential load from sequential save\n"));
    PetscCall(Run(PETSC_COMM_SELF, &ctx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test parallel load from sequential save\n"));
    PetscCall(Run(PETSC_COMM_WORLD, &ctx));
    /* sequential and parallel runs from parallel save */
    PetscCall(PetscSNPrintf(ctx.load_filename, sizeof(ctx.load_filename), "test_ex30_par.h5"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test sequential load from parallel save\n"));
    PetscCall(Run(PETSC_COMM_SELF, &ctx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test parallel load from parallel save\n"));
    PetscCall(Run(PETSC_COMM_WORLD, &ctx));
  } else { /* Run the simulation */
    PetscCall(Run(PETSC_COMM_WORLD, &ctx));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_box_faces 3,3 -ksp_type preonly -pc_type svd -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 1 -initial_snes_test_jacobian -snes_test_jacobian -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none

    test:
      suffix: 0
      nsize: {{1 2}}
      args: -dm_refine 1 -lump {{0 1}}

    test:
      suffix: 0_dirk
      nsize: {{1 2}}
      args: -dm_refine 1 -ts_type dirk

    test:
      suffix: 0_dirk_mg
      nsize: {{1 2}}
      args: -dm_refine_hierarchy 1 -ts_type dirk -pc_type mg -mg_levels_pc_type bjacobi -mg_levels_sub_pc_factor_levels 2 -mg_levels_sub_pc_factor_mat_ordering_type rcm -mg_levels_sub_pc_factor_reuse_ordering -mg_coarse_pc_type svd -lump {{0 1}}

    test:
      suffix: 0_dirk_fieldsplit
      nsize: {{1 2}}
      args: -dm_refine 1 -ts_type dirk -pc_type fieldsplit -pc_fieldsplit_type multiplicative -lump {{0 1}}

    test:
      requires: p4est
      suffix: 0_p4est
      nsize: {{1 2}}
      args: -dm_refine 1 -dm_plex_convert_type p4est -lump 0

    test:
      suffix: 0_periodic
      nsize: {{1 2}}
      args: -dm_plex_box_bd periodic,periodic -dm_refine_pre 1 -lump {{0 1}}

    test:
      requires: p4est
      suffix: 0_p4est_periodic
      nsize: {{1 2}}
      args: -dm_plex_box_bd periodic,periodic -dm_refine_pre 1 -dm_plex_convert_type p4est -lump 0

    test:
      requires: p4est
      suffix: 0_p4est_mg
      nsize: {{1 2}}
      args: -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_plex_convert_type p4est -pc_type mg -mg_coarse_pc_type svd -mg_levels_pc_type svd -lump 0

  testset:
    requires: hdf5
    args: -test_restart -dm_plex_box_faces 3,3 -ksp_type preonly -pc_type mg -mg_levels_pc_type svd -c_petscspace_degree 1 -p_petscspace_degree 1 -petscpartitioner_type simple -test_restart

    test:
      requires: !single
      suffix: restart
      nsize: {{1 2}separate output}
      args: -dm_refine_hierarchy {{0 1}separate output} -dm_plex_simplex 0

    test:
      requires: triangle
      suffix: restart_simplex
      nsize: {{1 2}separate output}
      args: -dm_refine_hierarchy {{0 1}separate output} -dm_plex_simplex 1

    test:
      requires: !single
      suffix: restart_refonly
      nsize: {{1 2}separate output}
      args: -dm_refine 1 -dm_plex_simplex 0

    test:
      requires: triangle
      suffix: restart_simplex_refonly
      nsize: {{1 2}separate output}
      args: -dm_refine 1 -dm_plex_simplex 1

TEST*/
