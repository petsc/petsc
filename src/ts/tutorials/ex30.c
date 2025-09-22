static char help[] = "Biological network from https://link.springer.com/article/10.1007/s42967-023-00297-3\n\n\n";

#include <petscts.h>
#include <petscsf.h>
#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscdmforest.h>
#include <petscviewerhdf5.h>
#include <petscds.h>

/*
    Here we solve the system of PDEs on \Omega \in R^d:

    * dC/dt - D^2 \Delta C - \nabla p \cross \nabla p + \alpha sqrt(||C||^2_F + eps)^(\gamma-2) C = 0
    * - \nabla \cdot ((r + C) \nabla p) = S

    where:
      C = symmetric dxd conductivity tensor
      p = potential
      S = source

    with natural boundary conditions on \partial\Omega:
      \nabla C \cdot n  = 0
      \nabla ((r + C)\nabla p) \cdot n  = 0

    Parameters:
      D = diffusion constant
      \alpha = metabolic coefficient
      \gamma = metabolic exponent
      r, eps are regularization parameters

    We use Lagrange elements for C_ij and P.
    Equations are rescaled to obtain a symmetric Jacobian.
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
  FIXC_ID,
  SPLIT_ID,
  NUM_CONSTANTS
} ConstantIdx;

PetscLogStage SetupStage, SolveStage;

#define NORM2C(c00, c01, c11)                   (PetscSqr(c00) + 2 * PetscSqr(c01) + PetscSqr(c11))
#define NORM2C_3d(c00, c01, c02, c11, c12, c22) (PetscSqr(c00) + 2 * PetscSqr(c01) + 2 * PetscSqr(c02) + PetscSqr(c11) + 2 * PetscSqr(c12) + PetscSqr(c22))

/* Eigenvalues real 3x3 symmetric matrix https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7153 */
#if !PetscDefined(USE_COMPLEX)
static inline void Eigenvalues_Sym3x3(PetscReal a11, PetscReal a12, PetscReal a13, PetscReal a22, PetscReal a23, PetscReal a33, PetscReal x[3])
{
  const PetscBool td = (PetscBool)(a13 == 0 && a23 == 0);
  if (td) { /* two-dimensional case */
    PetscReal a12_2 = PetscSqr(a12);
    PetscReal delta = PetscSqr(a11 - a22) + 4 * a12_2;
    PetscReal b     = -(a11 + a22);
    PetscReal c     = a11 * a22 - a12_2;
    PetscReal temp  = -0.5 * (b + PetscCopysignReal(1.0, b) * PetscSqrtReal(delta));

    x[0] = temp;
    x[1] = c / temp;
    x[2] = a33;
  } else {
    const PetscReal I1  = a11 + a22 + a33;
    const PetscReal J2  = (PetscSqr(a11 - a22) + PetscSqr(a22 - a33) + PetscSqr(a33 - a11)) / 6 + PetscSqr(a12) + PetscSqr(a23) + PetscSqr(a13);
    const PetscReal s   = PetscSqrtReal(J2 / 3);
    const PetscReal tol = PETSC_MACHINE_EPSILON;

    if (s < tol) {
      x[0] = x[1] = x[2] = 0.0;
    } else {
      const PetscReal S[] = {a11 - I1 / 3, a12, a13, a22 - I1 / 3, a23, a33 - I1 / 3};

      /* T = S^2 */
      PetscReal T[6];
      T[0] = S[0] * S[0] + S[1] * S[1] + S[2] * S[2];
      T[1] = S[0] * S[1] + S[1] * S[3] + S[2] * S[4];
      T[2] = S[0] * S[2] + S[1] * S[4] + S[2] * S[5];
      T[3] = S[1] * S[1] + S[3] * S[3] + S[4] * S[4];
      T[4] = S[1] * S[2] + S[3] * S[4] + S[4] * S[5];
      T[5] = S[2] * S[2] + S[4] * S[4] + S[5] * S[5];

      T[0] = T[0] - 2.0 * J2 / 3.0;
      T[3] = T[3] - 2.0 * J2 / 3.0;
      T[5] = T[5] - 2.0 * J2 / 3.0;

      const PetscReal aa = NORM2C_3d(T[0] - s * S[0], T[1] - s * S[1], T[2] - s * S[2], T[3] - s * S[3], T[4] - s * S[4], T[5] - s * S[5]);
      const PetscReal bb = NORM2C_3d(T[0] + s * S[0], T[1] + s * S[1], T[2] + s * S[2], T[3] + s * S[3], T[4] + s * S[4], T[5] + s * S[5]);
      const PetscReal d  = PetscSqrtReal(aa / bb);
      const PetscBool sj = (PetscBool)(1.0 - d > 0.0);

      if (PetscAbsReal(1 - d) < tol) {
        x[0] = -PetscSqrtReal(J2);
        x[1] = 0.0;
        x[2] = PetscSqrtReal(J2);
      } else {
        const PetscReal sjn = sj ? 1.0 : -1.0;
        //const PetscReal atanarg = sj ? d : 1.0 / d;
        //const PetscReal alpha   = 2.0 * PetscAtanReal(atanarg) / 3.0;
        const PetscReal atanval = d > tol ? 2.0 * PetscAtanReal(sj ? d : 1.0 / d) : (sj ? 0.0 : PETSC_PI);
        const PetscReal alpha   = atanval / 3.0;
        const PetscReal cd      = s * PetscCosReal(alpha) * sjn;
        const PetscReal sd      = PetscSqrtReal(J2) * PetscSinReal(alpha);

        x[0] = 2 * cd;
        x[1] = -cd + sd;
        x[2] = -cd - sd;
      }
    }
    x[0] += I1 / 3.0;
    x[1] += I1 / 3.0;
    x[2] += I1 / 3.0;
  }
}
#endif

/* compute shift to make C positive definite */
static inline PetscReal FIX_C_3d(PetscScalar C00, PetscScalar C01, PetscScalar C02, PetscScalar C11, PetscScalar C12, PetscScalar C22)
{
#if !PetscDefined(USE_COMPLEX)
  PetscReal eigs[3], s = 0.0;
  PetscBool twod = (PetscBool)(C02 == 0 && C12 == 0 && C22 == 0);
  Eigenvalues_Sym3x3(C00, C01, C02, C11, C12, C22, eigs);
  if (twod) eigs[2] = 1.0;
  if (eigs[0] <= 0 || eigs[1] <= 0 || eigs[2] <= 0) s = -PetscMin(eigs[0], PetscMin(eigs[1], eigs[2])) + PETSC_SMALL;
  return s;
#else
  return 0.0;
#endif
}

static inline PetscReal FIX_C(PetscScalar C00, PetscScalar C01, PetscScalar C11)
{
  return FIX_C_3d(C00, C01, 0, C11, 0, 0);
}

/* residual for C when tested against basis functions */
static void C_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal    alpha    = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal    gamma    = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal    eps      = PetscRealPart(constants[EPS_ID]);
  const PetscBool    split    = (PetscBool)(PetscRealPart(constants[SPLIT_ID]) != 0.0);
  const PetscScalar *gradp    = split ? a_x + aOff_x[P_FIELD_ID] : u_x + uOff_x[P_FIELD_ID];
  const PetscScalar  outerp[] = {gradp[0] * gradp[0], gradp[0] * gradp[1], gradp[1] * gradp[1]};
  const PetscScalar  C00      = split ? a[aOff[C_FIELD_ID]] : u[uOff[C_FIELD_ID]];
  const PetscScalar  C01      = split ? a[aOff[C_FIELD_ID] + 1] : u[uOff[C_FIELD_ID] + 1];
  const PetscScalar  C11      = split ? a[aOff[C_FIELD_ID] + 2] : u[uOff[C_FIELD_ID] + 2];
  const PetscScalar  norm     = NORM2C(C00, C01, C11) + eps;
  const PetscScalar  nexp     = (gamma - 2.0) / 2.0;
  const PetscScalar  fnorm    = PetscPowScalar(norm, nexp);
  const PetscScalar  eqss[]   = {0.5, 1., 0.5}; /* equations rescaling for a symmetric Jacobian */

  for (PetscInt k = 0; k < 3; k++) f0[k] = eqss[k] * (u_t[uOff[C_FIELD_ID] + k] - outerp[k] + alpha * fnorm * u[uOff[C_FIELD_ID] + k]);
}

/* 3D version */
static void C_0_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal    alpha    = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal    gamma    = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal    eps      = PetscRealPart(constants[EPS_ID]);
  const PetscBool    split    = (PetscBool)(PetscRealPart(constants[SPLIT_ID]) != 0.0);
  const PetscScalar *gradp    = split ? a_x + aOff_x[P_FIELD_ID] : u_x + uOff_x[P_FIELD_ID];
  const PetscScalar  outerp[] = {gradp[0] * gradp[0], gradp[0] * gradp[1], gradp[0] * gradp[2], gradp[1] * gradp[1], gradp[1] * gradp[2], gradp[2] * gradp[2]};
  const PetscScalar  C00      = split ? a[aOff[C_FIELD_ID]] : u[uOff[C_FIELD_ID]];
  const PetscScalar  C01      = split ? a[aOff[C_FIELD_ID] + 1] : u[uOff[C_FIELD_ID] + 1];
  const PetscScalar  C02      = split ? a[aOff[C_FIELD_ID] + 2] : u[uOff[C_FIELD_ID] + 2];
  const PetscScalar  C11      = split ? a[aOff[C_FIELD_ID] + 3] : u[uOff[C_FIELD_ID] + 3];
  const PetscScalar  C12      = split ? a[aOff[C_FIELD_ID] + 4] : u[uOff[C_FIELD_ID] + 4];
  const PetscScalar  C22      = split ? a[aOff[C_FIELD_ID] + 5] : u[uOff[C_FIELD_ID] + 5];
  const PetscScalar  norm     = NORM2C_3d(C00, C01, C02, C11, C12, C22) + eps;
  const PetscScalar  nexp     = (gamma - 2.0) / 2.0;
  const PetscScalar  fnorm    = PetscPowScalar(norm, nexp);
  const PetscScalar  eqss[]   = {0.5, 1., 1., 0.5, 1., 0.5};

  for (PetscInt k = 0; k < 6; k++) f0[k] = eqss[k] * (u_t[uOff[C_FIELD_ID] + k] - outerp[k] + alpha * fnorm * u[uOff[C_FIELD_ID] + k]);
}

/* Jacobian for C against C basis functions */
static void JC_0_c0c0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   alpha  = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma  = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal   eps    = PetscRealPart(constants[EPS_ID]);
  const PetscBool   split  = (PetscBool)(PetscRealPart(constants[SPLIT_ID]) != 0.0);
  const PetscScalar C00    = split ? a[aOff[C_FIELD_ID]] : u[uOff[C_FIELD_ID]];
  const PetscScalar C01    = split ? a[aOff[C_FIELD_ID] + 1] : u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C11    = split ? a[aOff[C_FIELD_ID] + 2] : u[uOff[C_FIELD_ID] + 2];
  const PetscScalar norm   = NORM2C(C00, C01, C11) + eps;
  const PetscScalar nexp   = (gamma - 2.0) / 2.0;
  const PetscScalar fnorm  = PetscPowScalar(norm, nexp);
  const PetscScalar dfnorm = nexp * PetscPowScalar(norm, nexp - 1.0);
  const PetscScalar dC[]   = {2 * C00, 4 * C01, 2 * C11};
  const PetscScalar eqss[] = {0.5, 1., 0.5};

  for (PetscInt k = 0; k < 3; k++) {
    if (!split) {
      for (PetscInt j = 0; j < 3; j++) J[k * 3 + j] = eqss[k] * (alpha * dfnorm * dC[j] * u[uOff[C_FIELD_ID] + k]);
    }
    J[k * 3 + k] += eqss[k] * (alpha * fnorm + u_tShift);
  }
}

static void JC_0_c0c0_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   alpha  = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma  = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal   eps    = PetscRealPart(constants[EPS_ID]);
  const PetscBool   split  = (PetscBool)(PetscRealPart(constants[SPLIT_ID]) != 0.0);
  const PetscScalar C00    = split ? a[aOff[C_FIELD_ID]] : u[uOff[C_FIELD_ID]];
  const PetscScalar C01    = split ? a[aOff[C_FIELD_ID] + 1] : u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C02    = split ? a[aOff[C_FIELD_ID] + 2] : u[uOff[C_FIELD_ID] + 2];
  const PetscScalar C11    = split ? a[aOff[C_FIELD_ID] + 3] : u[uOff[C_FIELD_ID] + 3];
  const PetscScalar C12    = split ? a[aOff[C_FIELD_ID] + 4] : u[uOff[C_FIELD_ID] + 4];
  const PetscScalar C22    = split ? a[aOff[C_FIELD_ID] + 5] : u[uOff[C_FIELD_ID] + 5];
  const PetscScalar norm   = NORM2C_3d(C00, C01, C02, C11, C12, C22) + eps;
  const PetscScalar nexp   = (gamma - 2.0) / 2.0;
  const PetscScalar fnorm  = PetscPowScalar(norm, nexp);
  const PetscScalar dfnorm = nexp * PetscPowScalar(norm, nexp - 1.0);
  const PetscScalar dC[]   = {2 * C00, 4 * C01, 4 * C02, 2 * C11, 4 * C12, 2 * C22};
  const PetscScalar eqss[] = {0.5, 1., 1., 0.5, 1., 0.5};

  for (PetscInt k = 0; k < 6; k++) {
    if (!split) {
      for (PetscInt j = 0; j < 6; j++) J[k * 6 + j] = eqss[k] * (alpha * dfnorm * dC[j] * u[uOff[C_FIELD_ID] + k]);
    }
    J[k * 6 + k] += eqss[k] * (alpha * fnorm + u_tShift);
  }
}

/* Jacobian for C against C basis functions and gradients of P basis functions */
static void JC_0_c0p1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscScalar *gradp  = u_x + uOff_x[P_FIELD_ID];
  const PetscScalar  eqss[] = {0.5, 1., 0.5};

  J[0] = -2 * gradp[0] * eqss[0];
  J[1] = 0.0;

  J[2] = -gradp[1] * eqss[1];
  J[3] = -gradp[0] * eqss[1];

  J[4] = 0.0;
  J[5] = -2 * gradp[1] * eqss[2];
}

static void JC_0_c0p1_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscScalar *gradp  = u_x + uOff_x[P_FIELD_ID];
  const PetscScalar  eqss[] = {0.5, 1., 1., 0.5, 1., 0.5};

  J[0] = -2 * gradp[0] * eqss[0];
  J[1] = 0.0;
  J[2] = 0.0;

  J[3] = -gradp[1] * eqss[1];
  J[4] = -gradp[0] * eqss[1];
  J[5] = 0.0;

  J[6] = -gradp[2] * eqss[2];
  J[7] = 0.0;
  J[8] = -gradp[0] * eqss[2];

  J[9]  = 0.0;
  J[10] = -2 * gradp[1] * eqss[3];
  J[11] = 0.0;

  J[12] = 0.0;
  J[13] = -gradp[2] * eqss[4];
  J[14] = -gradp[1] * eqss[4];

  J[15] = 0.0;
  J[16] = 0.0;
  J[17] = -2 * gradp[2] * eqss[5];
}

/* residual for C when tested against gradients of basis functions */
static void C_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt d = 0; d < 2; d++) f1[k * 2 + d] = PetscSqr(D) * u_x[uOff_x[C_FIELD_ID] + k * 2 + d];
}

static void C_1_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 6; k++)
    for (PetscInt d = 0; d < 3; d++) f1[k * 3 + d] = PetscSqr(D) * u_x[uOff_x[C_FIELD_ID] + k * 3 + d];
}

/* Jacobian for C against gradients of C basis functions */
static void JC_1_c1c1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt d = 0; d < 2; d++) J[k * (3 + 1) * 2 * 2 + d * 2 + d] = PetscSqr(D);
}

static void JC_1_c1c1_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 6; k++)
    for (PetscInt d = 0; d < 3; d++) J[k * (6 + 1) * 3 * 3 + d * 3 + d] = PetscSqr(D);
}

/* residual for P when tested against basis functions.
   The source term always comes from the auxiliary data because it must be zero mean (algebraically) */
static void P_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar S = a[aOff[NUM_FIELDS]];

  f0[0] = S;
}

/* residual for P when tested against basis functions for the initial condition problem
   here we don't impose symmetry, and we thus flip the sign of the source function */
static void P_0_aux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar S = a[aOff[NUM_FIELDS]];

  f0[0] = -S;
}

/* residual for P when tested against gradients of basis functions */
static void P_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal    r     = PetscRealPart(constants[R_ID]);
  const PetscScalar  C00   = u[uOff[C_FIELD_ID]] + r;
  const PetscScalar  C01   = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar  C10   = C01;
  const PetscScalar  C11   = u[uOff[C_FIELD_ID] + 2] + r;
  const PetscScalar *gradp = u_x + uOff_x[P_FIELD_ID];
  const PetscBool    fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 1.0);
  const PetscScalar  s     = fix_c ? FIX_C(C00, C01, C11) : 0.0;

  f1[0] = -((C00 + s) * gradp[0] + C01 * gradp[1]);
  f1[1] = -(C10 * gradp[0] + (C11 + s) * gradp[1]);
}

static void P_1_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal    r     = PetscRealPart(constants[R_ID]);
  const PetscScalar  C00   = u[uOff[C_FIELD_ID]] + r;
  const PetscScalar  C01   = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar  C02   = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar  C10   = C01;
  const PetscScalar  C11   = u[uOff[C_FIELD_ID] + 3] + r;
  const PetscScalar  C12   = u[uOff[C_FIELD_ID] + 4];
  const PetscScalar  C20   = C02;
  const PetscScalar  C21   = C12;
  const PetscScalar  C22   = u[uOff[C_FIELD_ID] + 5] + r;
  const PetscScalar *gradp = u_x + uOff_x[P_FIELD_ID];
  const PetscBool    fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 1.0);
  const PetscScalar  s     = fix_c ? FIX_C_3d(C00, C01, C02, C11, C12, C22) : 0.0;

  f1[0] = -((C00 + s) * gradp[0] + C01 * gradp[1] + C02 * gradp[2]);
  f1[1] = -(C10 * gradp[0] + (C11 + s) * gradp[1] + C12 * gradp[2]);
  f1[2] = -(C20 * gradp[0] + C21 * gradp[1] + (C22 + s) * gradp[2]);
}

/* Same as above except that the conductivity values come from the auxiliary vec */
static void P_1_aux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal    r     = PetscRealPart(constants[R_ID]);
  const PetscScalar  C00   = a[aOff[C_FIELD_ID]] + r;
  const PetscScalar  C01   = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar  C10   = C01;
  const PetscScalar  C11   = a[aOff[C_FIELD_ID] + 2] + r;
  const PetscScalar *gradp = u_x + uOff_x[Nf > 1 ? P_FIELD_ID : 0];
  const PetscBool    fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 1.0);
  const PetscScalar  s     = fix_c ? FIX_C(C00, C01, C11) : 0.0;

  f1[0] = (C00 + s) * gradp[0] + C01 * gradp[1];
  f1[1] = C10 * gradp[0] + (C11 + s) * gradp[1];
}

static void P_1_aux_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal    r     = PetscRealPart(constants[R_ID]);
  const PetscScalar  C00   = a[aOff[C_FIELD_ID]] + r;
  const PetscScalar  C01   = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar  C02   = a[aOff[C_FIELD_ID] + 2];
  const PetscScalar  C10   = C01;
  const PetscScalar  C11   = a[aOff[C_FIELD_ID] + 3] + r;
  const PetscScalar  C12   = a[aOff[C_FIELD_ID] + 4];
  const PetscScalar  C20   = C02;
  const PetscScalar  C21   = C12;
  const PetscScalar  C22   = a[aOff[C_FIELD_ID] + 5] + r;
  const PetscScalar *gradp = u_x + uOff_x[Nf > 1 ? P_FIELD_ID : 0];
  const PetscBool    fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 1.0);
  const PetscScalar  s     = fix_c ? FIX_C_3d(C00, C01, C02, C11, C12, C22) : 0.0;

  f1[0] = (C00 + s) * gradp[0] + C01 * gradp[1] + C02 * gradp[2];
  f1[1] = C10 * gradp[0] + (C11 + s) * gradp[1] + C12 * gradp[2];
  f1[2] = C20 * gradp[0] + C21 * gradp[1] + (C22 + s) * gradp[2];
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

  J[0] = -(C00 + s);
  J[1] = -C01;
  J[2] = -C10;
  J[3] = -(C11 + s);
}

static void JP_1_p1p1_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   r     = PetscRealPart(constants[R_ID]);
  const PetscScalar C00   = u[uOff[C_FIELD_ID]] + r;
  const PetscScalar C01   = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C02   = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar C10   = C01;
  const PetscScalar C11   = u[uOff[C_FIELD_ID] + 3] + r;
  const PetscScalar C12   = u[uOff[C_FIELD_ID] + 4];
  const PetscScalar C20   = C02;
  const PetscScalar C21   = C12;
  const PetscScalar C22   = u[uOff[C_FIELD_ID] + 5] + r;
  const PetscBool   fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 0.0);
  const PetscScalar s     = fix_c ? FIX_C_3d(C00, C01, C02, C11, C12, C22) : 0.0;

  J[0] = -(C00 + s);
  J[1] = -C01;
  J[2] = -C02;
  J[3] = -C10;
  J[4] = -(C11 + s);
  J[5] = -C12;
  J[6] = -C20;
  J[7] = -C21;
  J[8] = -(C22 + s);
}

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

static void JP_1_p1p1_aux_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   r     = PetscRealPart(constants[R_ID]);
  const PetscScalar C00   = a[aOff[C_FIELD_ID]] + r;
  const PetscScalar C01   = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar C02   = a[aOff[C_FIELD_ID] + 2];
  const PetscScalar C10   = C01;
  const PetscScalar C11   = a[aOff[C_FIELD_ID] + 3] + r;
  const PetscScalar C12   = a[aOff[C_FIELD_ID] + 4];
  const PetscScalar C20   = C02;
  const PetscScalar C21   = C12;
  const PetscScalar C22   = a[aOff[C_FIELD_ID] + 5] + r;
  const PetscBool   fix_c = (PetscBool)(PetscRealPart(constants[FIXC_ID]) > 0.0);
  const PetscScalar s     = fix_c ? FIX_C_3d(C00, C01, C02, C11, C12, C22) : 0.0;

  J[0] = C00 + s;
  J[1] = C01;
  J[2] = C02;
  J[3] = C10;
  J[4] = C11 + s;
  J[5] = C12;
  J[6] = C20;
  J[7] = C21;
  J[8] = C22 + s;
}

/* Jacobian for P against gradients of P basis functions and C basis functions */
static void JP_1_p1c0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscScalar *gradp = u_x + uOff_x[P_FIELD_ID];

  J[0] = -gradp[0];
  J[1] = 0;

  J[2] = -gradp[1];
  J[3] = -gradp[0];

  J[4] = 0;
  J[5] = -gradp[1];
}

static void JP_1_p1c0_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscScalar *gradp = u_x + uOff_x[P_FIELD_ID];

  J[0] = -gradp[0];
  J[1] = 0;
  J[2] = 0;

  J[3] = -gradp[1];
  J[4] = -gradp[0];
  J[5] = 0;

  J[6] = -gradp[2];
  J[7] = 0;
  J[8] = -gradp[0];

  J[9]  = 0;
  J[10] = -gradp[1];
  J[11] = 0;

  J[12] = 0;
  J[13] = -gradp[2];
  J[14] = -gradp[1];

  J[15] = 0;
  J[16] = 0;
  J[17] = -gradp[2];
}

/* a collection of gaussian, Dirac-like, source term S(x) = scale * cos(2*pi*(frequency*t + phase)) * exp(-w*||x - x0||^2) */
typedef struct {
  PetscInt   n;
  PetscReal *x0;
  PetscReal *w;
  PetscReal *k;
  PetscReal *p;
  PetscReal *r;
} MultiSourceCtx;

typedef struct {
  PetscReal x0[3];
  PetscReal w;
  PetscReal k;
  PetscReal p;
  PetscReal r;
} SingleSourceCtx;

static PetscErrorCode gaussian(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  SingleSourceCtx *s  = (SingleSourceCtx *)ctx;
  const PetscReal *x0 = s->x0;
  const PetscReal  w  = s->w;
  const PetscReal  k  = s->k; /* frequency */
  const PetscReal  p  = s->p; /* phase */
  const PetscReal  r  = s->r; /* scale */
  PetscReal        n  = 0;

  for (PetscInt d = 0; d < dim; ++d) n += (x[d] - x0[d]) * (x[d] - x0[d]);
  u[0] = r * PetscCosReal(2 * PETSC_PI * (k * time + p)) * PetscExpReal(-w * n);
  return PETSC_SUCCESS;
}

static PetscErrorCode source(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  MultiSourceCtx *s = (MultiSourceCtx *)ctx;

  u[0] = 0.0;
  for (PetscInt i = 0; i < s->n; i++) {
    SingleSourceCtx sctx;
    PetscScalar     ut[1];

    sctx.x0[0] = s->x0[dim * i];
    sctx.x0[1] = s->x0[dim * i + 1];
    sctx.x0[2] = dim > 2 ? s->x0[dim * i + 2] : 0.0;
    sctx.w     = s->w[i];
    sctx.k     = s->k[i];
    sctx.p     = s->p[i];
    sctx.r     = s->r[i];

    PetscCall(gaussian(dim, time, x, Nf, ut, &sctx));

    u[0] += ut[0];
  }
  return PETSC_SUCCESS;
}

/* functionals to be integrated: average -> \int_\Omega u dx */
static void average(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  const PetscInt fid = (PetscInt)PetscRealPart(constants[numConstants]);
  obj[0]             = u[uOff[fid]];
}

/* functionals to be integrated: volume -> \int_\Omega dx */
static void volume(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = 1;
}

/* functionals to be integrated: energy -> D^2/2 * ||\nabla C||^2 + c^2\nabla p * (r + C) * \nabla p + \alpha/ \gamma * ||C||^\gamma */
static void energy(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  const PetscReal D     = PetscRealPart(constants[D_ID]);
  const PetscReal r     = PetscRealPart(constants[R_ID]);
  const PetscReal alpha = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal gamma = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal eps   = PetscRealPart(constants[EPS_ID]);

  if (dim == 2) {
    const PetscScalar  C00       = u[uOff[C_FIELD_ID]];
    const PetscScalar  C01       = u[uOff[C_FIELD_ID] + 1];
    const PetscScalar  C10       = C01;
    const PetscScalar  C11       = u[uOff[C_FIELD_ID] + 2];
    const PetscScalar *gradp     = u_x + uOff_x[P_FIELD_ID];
    const PetscScalar *gradC00   = u_x + uOff_x[C_FIELD_ID];
    const PetscScalar *gradC01   = u_x + uOff_x[C_FIELD_ID] + 2;
    const PetscScalar *gradC11   = u_x + uOff_x[C_FIELD_ID] + 4;
    const PetscScalar  normC     = NORM2C(C00, C01, C11) + eps;
    const PetscScalar  normgradC = NORM2C(gradC00[0], gradC01[0], gradC11[0]) + NORM2C(gradC00[1], gradC01[1], gradC11[1]);
    const PetscScalar  nexp      = gamma / 2.0;

    const PetscScalar t0 = PetscSqr(D) / 2.0 * normgradC;
    const PetscScalar t1 = gradp[0] * ((C00 + r) * gradp[0] + C01 * gradp[1]) + gradp[1] * (C10 * gradp[0] + (C11 + r) * gradp[1]);
    const PetscScalar t2 = alpha / gamma * PetscPowScalar(normC, nexp);

    obj[0] = t0 + t1 + t2;
  } else {
    const PetscScalar  C00     = u[uOff[C_FIELD_ID]];
    const PetscScalar  C01     = u[uOff[C_FIELD_ID] + 1];
    const PetscScalar  C02     = u[uOff[C_FIELD_ID] + 2];
    const PetscScalar  C10     = C01;
    const PetscScalar  C11     = u[uOff[C_FIELD_ID] + 3];
    const PetscScalar  C12     = u[uOff[C_FIELD_ID] + 4];
    const PetscScalar  C20     = C02;
    const PetscScalar  C21     = C12;
    const PetscScalar  C22     = u[uOff[C_FIELD_ID] + 5];
    const PetscScalar *gradp   = u_x + uOff_x[P_FIELD_ID];
    const PetscScalar *gradC00 = u_x + uOff_x[C_FIELD_ID];
    const PetscScalar *gradC01 = u_x + uOff_x[C_FIELD_ID] + 3;
    const PetscScalar *gradC02 = u_x + uOff_x[C_FIELD_ID] + 6;
    const PetscScalar *gradC11 = u_x + uOff_x[C_FIELD_ID] + 9;
    const PetscScalar *gradC12 = u_x + uOff_x[C_FIELD_ID] + 12;
    const PetscScalar *gradC22 = u_x + uOff_x[C_FIELD_ID] + 15;
    const PetscScalar  normC   = NORM2C_3d(C00, C01, C02, C11, C12, C22) + eps;
    const PetscScalar normgradC = NORM2C_3d(gradC00[0], gradC01[0], gradC02[0], gradC11[0], gradC12[0], gradC22[0]) + NORM2C_3d(gradC00[1], gradC01[1], gradC02[1], gradC11[1], gradC12[1], gradC22[1]) + NORM2C_3d(gradC00[2], gradC01[2], gradC02[2], gradC11[2], gradC12[2], gradC22[2]);
    const PetscScalar nexp = gamma / 2.0;

    const PetscScalar t0 = PetscSqr(D) / 2.0 * normgradC;
    const PetscScalar t1 = gradp[0] * ((C00 + r) * gradp[0] + C01 * gradp[1] + C02 * gradp[2]) + gradp[1] * (C10 * gradp[0] + (C11 + r) * gradp[1] + C12 * gradp[2]) + gradp[2] * (C20 * gradp[0] + C21 * gradp[1] + (C22 + r) * gradp[2]);
    const PetscScalar t2 = alpha / gamma * PetscPowScalar(normC, nexp);

    obj[0] = t0 + t1 + t2;
  }
}

/* functionals to be integrated: ellipticity_fail_private -> see below */
static void ellipticity_fail_private(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[], PetscBool add_reg)
{
#if !PetscDefined(USE_COMPLEX)
  PetscReal       eigs[3];
  PetscScalar     C00, C01, C02 = 0.0, C11, C12 = 0.0, C22 = 0.0;
  const PetscReal r = add_reg ? PetscRealPart(constants[R_ID]) : 0.0;

  if (dim == 2) {
    C00 = u[uOff[C_FIELD_ID]] + r;
    C01 = u[uOff[C_FIELD_ID] + 1];
    C11 = u[uOff[C_FIELD_ID] + 2] + r;
  } else {
    C00 = u[uOff[C_FIELD_ID]] + r;
    C01 = u[uOff[C_FIELD_ID] + 1];
    C02 = u[uOff[C_FIELD_ID] + 2];
    C11 = u[uOff[C_FIELD_ID] + 3] + r;
    C12 = u[uOff[C_FIELD_ID] + 4];
    C22 = u[uOff[C_FIELD_ID] + 5] + r;
  }
  Eigenvalues_Sym3x3(C00, C01, C02, C11, C12, C22, eigs);
  if (eigs[0] < 0 || eigs[1] < 0 || eigs[2] < 0) obj[0] = -PetscMin(eigs[0], PetscMin(eigs[1], eigs[2]));
  else obj[0] = 0.0;
#else
  obj[0] = 0.0;
#endif
}

/* functionals to be integrated: ellipticity_fail -> 0 means C is elliptic at quadrature point, otherwise it returns 1 */
static void ellipticity_fail(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  ellipticity_fail_private(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, x, numConstants, constants, obj, PETSC_FALSE);
  if (PetscAbsScalar(obj[0]) > 0.0) obj[0] = 1.0;
}

/* functionals to be integrated: jacobian_fail -> 0 means C + r is elliptic at quadrature point, otherwise it returns the absolute value of the most negative eigenvalue */
static void jacobian_fail(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  ellipticity_fail_private(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, x, numConstants, constants, obj, PETSC_TRUE);
}

/* initial conditions for C: eq. 16 */
static PetscErrorCode initial_conditions_C_0(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  if (dim == 2) {
    u[0] = 1;
    u[1] = 0;
    u[2] = 1;
  } else {
    u[0] = 1;
    u[1] = 0;
    u[2] = 0;
    u[3] = 1;
    u[4] = 0;
    u[5] = 1;
  }
  return PETSC_SUCCESS;
}

/* initial conditions for C: eq. 17 */
static PetscErrorCode initial_conditions_C_1(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal x = xx[0];
  const PetscReal y = xx[1];

  if (dim == 3) return PETSC_ERR_SUP;
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
  if (dim == 3) {
    u[3] = 0;
    u[4] = 0;
    u[5] = 0;
  }
  return PETSC_SUCCESS;
}

/* random initial conditions for the diagonal of C */
static PetscErrorCode initial_conditions_C_random(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscScalar vals[3];
  PetscRandom r = (PetscRandom)ctx;

  PetscCall(PetscRandomGetValues(r, dim, vals));
  if (dim == 2) {
    u[0] = vals[0];
    u[1] = 0;
    u[2] = vals[1];
  } else {
    u[0] = vals[0];
    u[1] = 0;
    u[2] = 0;
    u[3] = vals[1];
    u[4] = 0;
    u[5] = vals[2];
  }
  return PETSC_SUCCESS;
}

/* functionals to be sampled: zero */
static void zero(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  f[0] = 0.0;
}

/* functionals to be sampled: - (C + r) * \grad p */
static void flux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscReal    r     = PetscRealPart(constants[R_ID]);
  const PetscScalar *gradp = u_x + uOff_x[P_FIELD_ID];

  if (dim == 2) {
    const PetscScalar C00 = u[uOff[C_FIELD_ID]] + r;
    const PetscScalar C01 = u[uOff[C_FIELD_ID] + 1];
    const PetscScalar C10 = C01;
    const PetscScalar C11 = u[uOff[C_FIELD_ID] + 2] + r;

    f[0] = -C00 * gradp[0] - C01 * gradp[1];
    f[1] = -C10 * gradp[0] - C11 * gradp[1];
  } else {
    const PetscScalar C00 = u[uOff[C_FIELD_ID]] + r;
    const PetscScalar C01 = u[uOff[C_FIELD_ID] + 1];
    const PetscScalar C02 = u[uOff[C_FIELD_ID] + 2];
    const PetscScalar C10 = C01;
    const PetscScalar C11 = u[uOff[C_FIELD_ID] + 3] + r;
    const PetscScalar C12 = u[uOff[C_FIELD_ID] + 4];
    const PetscScalar C20 = C02;
    const PetscScalar C21 = C12;
    const PetscScalar C22 = u[uOff[C_FIELD_ID] + 5] + r;

    f[0] = -C00 * gradp[0] - C01 * gradp[1] - C02 * gradp[2];
    f[1] = -C10 * gradp[0] - C11 * gradp[1] - C12 * gradp[2];
    f[2] = -C20 * gradp[0] - C21 * gradp[1] - C22 * gradp[2];
  }
}

/* functionals to be sampled: ||C|| */
static void normc(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  if (dim == 2) {
    const PetscScalar C00 = u[uOff[C_FIELD_ID]];
    const PetscScalar C01 = u[uOff[C_FIELD_ID] + 1];
    const PetscScalar C11 = u[uOff[C_FIELD_ID] + 2];

    f[0] = PetscSqrtReal(PetscRealPart(NORM2C(C00, C01, C11)));
  } else {
    const PetscScalar C00 = u[uOff[C_FIELD_ID]];
    const PetscScalar C01 = u[uOff[C_FIELD_ID] + 1];
    const PetscScalar C02 = u[uOff[C_FIELD_ID] + 2];
    const PetscScalar C11 = u[uOff[C_FIELD_ID] + 3];
    const PetscScalar C12 = u[uOff[C_FIELD_ID] + 4];
    const PetscScalar C22 = u[uOff[C_FIELD_ID] + 5];

    f[0] = PetscSqrtReal(PetscRealPart(NORM2C_3d(C00, C01, C02, C11, C12, C22)));
  }
}

/* functionals to be sampled: eigenvalues of C */
static void eigsc(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
#if !PetscDefined(USE_COMPLEX)
  PetscReal   eigs[3];
  PetscScalar C00, C01, C02 = 0.0, C11, C12 = 0.0, C22 = 0.0;
  if (dim == 2) {
    C00 = u[uOff[C_FIELD_ID]];
    C01 = u[uOff[C_FIELD_ID] + 1];
    C11 = u[uOff[C_FIELD_ID] + 2];
  } else {
    C00 = u[uOff[C_FIELD_ID]];
    C01 = u[uOff[C_FIELD_ID] + 1];
    C02 = u[uOff[C_FIELD_ID] + 2];
    C11 = u[uOff[C_FIELD_ID] + 3];
    C12 = u[uOff[C_FIELD_ID] + 4];
    C22 = u[uOff[C_FIELD_ID] + 5];
  }
  Eigenvalues_Sym3x3(C00, C01, C02, C11, C12, C22, eigs);
  PetscCallVoid(PetscSortReal(dim, eigs));
  for (PetscInt d = 0; d < dim; d++) f[d] = eigs[d];
#else
  for (PetscInt d = 0; d < dim; d++) f[d] = 0;
#endif
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
  for (PetscInt d = 0; d < Nc; ++d) u[d] = 1.0;
  return PETSC_SUCCESS;
}

/* application context: customizable parameters */
typedef struct {
  PetscInt              dim;
  PetscBool             simplex;
  PetscReal             r;
  PetscReal             eps;
  PetscReal             alpha;
  PetscReal             gamma;
  PetscReal             D;
  PetscReal             domain_volume;
  PetscInt              ic_num;
  PetscBool             split;
  PetscBool             lump;
  PetscBool             amr;
  PetscBool             load;
  char                  load_filename[PETSC_MAX_PATH_LEN];
  PetscBool             save;
  char                  save_filename[PETSC_MAX_PATH_LEN];
  PetscInt              save_every;
  PetscBool             test_restart;
  PetscInt              fix_c;
  MultiSourceCtx       *source_ctx;
  DM                    view_dm;
  TSMonitorVTKCtx       view_vtk_ctx;
  PetscViewerAndFormat *view_hdf5_ctx;
  PetscInt              diagnostic_num;
  PetscReal             view_times[64];
  PetscInt              view_times_n, view_times_k;
  PetscReal             function_domain_error_tol;
  VecScatter            subsct[NUM_FIELDS];
  Vec                   subvec[NUM_FIELDS];
  PetscBool             monitor_norms;
  PetscBool             exclude_potential_lte;

  /* hack: need some more plumbing in the library */
  SNES snes;
} AppCtx;

/* process command line options */
#include <petsc/private/tsimpl.h> /* To access TSMonitorVTKCtx */
static PetscErrorCode ProcessOptions(AppCtx *options)
{
  char      vtkmonfilename[PETSC_MAX_PATH_LEN];
  char      hdf5monfilename[PETSC_MAX_PATH_LEN];
  PetscInt  tmp;
  PetscBool flg;

  PetscFunctionBeginUser;
  options->dim                       = 2;
  options->r                         = 1.e-1;
  options->eps                       = 1.e-3;
  options->alpha                     = 0.75;
  options->gamma                     = 0.75;
  options->D                         = 1.e-2;
  options->ic_num                    = 0;
  options->split                     = PETSC_FALSE;
  options->lump                      = PETSC_FALSE;
  options->amr                       = PETSC_FALSE;
  options->load                      = PETSC_FALSE;
  options->save                      = PETSC_FALSE;
  options->save_every                = -1;
  options->test_restart              = PETSC_FALSE;
  options->fix_c                     = 1; /* 1 means only Jac, 2 means function and Jac, < 0 means raise SNESFunctionDomainError when C+r is not posdef */
  options->view_vtk_ctx              = NULL;
  options->view_hdf5_ctx             = NULL;
  options->view_dm                   = NULL;
  options->diagnostic_num            = 1;
  options->function_domain_error_tol = -1;
  options->monitor_norms             = PETSC_FALSE;
  options->exclude_potential_lte     = PETSC_FALSE;
  for (PetscInt i = 0; i < NUM_FIELDS; i++) {
    options->subsct[i] = NULL;
    options->subvec[i] = NULL;
  }
  for (PetscInt i = 0; i < 64; i++) options->view_times[i] = PETSC_MAX_REAL;

  PetscOptionsBegin(PETSC_COMM_WORLD, "", __FILE__, "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "space dimension", __FILE__, options->dim, &options->dim, NULL));
  PetscCall(PetscOptionsReal("-alpha", "alpha", __FILE__, options->alpha, &options->alpha, NULL));
  PetscCall(PetscOptionsReal("-gamma", "gamma", __FILE__, options->gamma, &options->gamma, NULL));
  PetscCall(PetscOptionsReal("-d", "D", __FILE__, options->D, &options->D, NULL));
  PetscCall(PetscOptionsReal("-eps", "eps", __FILE__, options->eps, &options->eps, NULL));
  PetscCall(PetscOptionsReal("-r", "r", __FILE__, options->r, &options->r, NULL));
  PetscCall(PetscOptionsInt("-ic_num", "ic_num", __FILE__, options->ic_num, &options->ic_num, NULL));
  PetscCall(PetscOptionsBool("-split", "Operator splitting", __FILE__, options->split, &options->split, NULL));
  PetscCall(PetscOptionsBool("-lump", "use mass lumping", __FILE__, options->lump, &options->lump, NULL));
  PetscCall(PetscOptionsInt("-fix_c", "Fix conductivity: shift to always be positive semi-definite or raise domain error", __FILE__, options->fix_c, &options->fix_c, NULL));
  PetscCall(PetscOptionsBool("-amr", "use adaptive mesh refinement", __FILE__, options->amr, &options->amr, NULL));
  PetscCall(PetscOptionsReal("-domain_error_tol", "domain error tolerance", __FILE__, options->function_domain_error_tol, &options->function_domain_error_tol, NULL));
  PetscCall(PetscOptionsBool("-test_restart", "test restarting files", __FILE__, options->test_restart, &options->test_restart, NULL));
  if (!options->test_restart) {
    PetscCall(PetscOptionsString("-load", "filename with data to be loaded for restarting", __FILE__, options->load_filename, options->load_filename, PETSC_MAX_PATH_LEN, &options->load));
    PetscCall(PetscOptionsString("-save", "filename with data to be saved for restarting", __FILE__, options->save_filename, options->save_filename, PETSC_MAX_PATH_LEN, &options->save));
    if (options->save) PetscCall(PetscOptionsInt("-save_every", "save every n timestep (-1 saves only the last)", __FILE__, options->save_every, &options->save_every, NULL));
  }
  PetscCall(PetscOptionsBool("-exclude_potential_lte", "exclude potential from LTE", __FILE__, options->exclude_potential_lte, &options->exclude_potential_lte, NULL));
  options->view_times_k = 0;
  options->view_times_n = 0;
  PetscCall(PetscOptionsRealArray("-monitor_times", "Save at specific times", NULL, options->view_times, (tmp = 64, &tmp), &flg));
  if (flg) options->view_times_n = tmp;

  PetscCall(PetscOptionsString("-monitor_vtk", "Dump VTK file for diagnostic", "TSMonitorSolutionVTK", NULL, vtkmonfilename, sizeof(vtkmonfilename), &flg));
  if (flg) {
    PetscCall(TSMonitorSolutionVTKCtxCreate(vtkmonfilename, &options->view_vtk_ctx));
    PetscCall(PetscOptionsInt("-monitor_vtk_interval", "Save every interval time steps", NULL, options->view_vtk_ctx->interval, &options->view_vtk_ctx->interval, NULL));
  }
  PetscCall(PetscOptionsString("-monitor_hdf5", "Dump HDF5 file for diagnostic", "TSMonitorSolution", NULL, hdf5monfilename, sizeof(hdf5monfilename), &flg));
  PetscCall(PetscOptionsInt("-monitor_diagnostic_num", "Number of diagnostics to be computed", __FILE__, options->diagnostic_num, &options->diagnostic_num, NULL));

  if (flg) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewer viewer;

    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, hdf5monfilename, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerAndFormatCreate(viewer, PETSC_VIEWER_HDF5_VIZ, &options->view_hdf5_ctx));
    options->view_hdf5_ctx->view_interval = 1;
    PetscCall(PetscOptionsInt("-monitor_hdf5_interval", "Save every interval time steps", NULL, options->view_hdf5_ctx->view_interval, &options->view_hdf5_ctx->view_interval, NULL));
    PetscCall(PetscViewerDestroy(&viewer));
#else
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Needs HDF5 support. Please reconfigure using --download-hdf5");
#endif
  }
  PetscCall(PetscOptionsBool("-monitor_norms", "Monitor separate SNES norms", __FILE__, options->monitor_norms, &options->monitor_norms, NULL));

  /* source options */
  PetscCall(PetscNew(&options->source_ctx));
  options->source_ctx->n = 1;

  PetscCall(PetscOptionsInt("-source_num", "number of sources", __FILE__, options->source_ctx->n, &options->source_ctx->n, NULL));
  tmp = options->source_ctx->n;
  PetscCall(PetscMalloc5(options->dim * tmp, &options->source_ctx->x0, tmp, &options->source_ctx->w, tmp, &options->source_ctx->k, tmp, &options->source_ctx->p, tmp, &options->source_ctx->r));
  for (PetscInt i = 0; i < options->source_ctx->n; i++) {
    for (PetscInt d = 0; d < options->dim; d++) options->source_ctx->x0[options->dim * i + d] = 0.25;
    options->source_ctx->w[i] = 500;
    options->source_ctx->k[i] = 0;
    options->source_ctx->p[i] = 0;
    options->source_ctx->r[i] = 1;
  }
  tmp = options->dim * options->source_ctx->n;
  PetscCall(PetscOptionsRealArray("-source_x0", "source location", __FILE__, options->source_ctx->x0, &tmp, NULL));
  tmp = options->source_ctx->n;
  PetscCall(PetscOptionsRealArray("-source_w", "source factor", __FILE__, options->source_ctx->w, &tmp, NULL));
  tmp = options->source_ctx->n;
  PetscCall(PetscOptionsRealArray("-source_k", "source frequency", __FILE__, options->source_ctx->k, &tmp, NULL));
  tmp = options->source_ctx->n;
  PetscCall(PetscOptionsRealArray("-source_p", "source phase", __FILE__, options->source_ctx->p, &tmp, NULL));
  tmp = options->source_ctx->n;
  PetscCall(PetscOptionsRealArray("-source_r", "source scaling", __FILE__, options->source_ctx->r, &tmp, NULL));
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

/*
   Setup AuxDM:
     - project source function and make it zero-mean
     - sample frozen fields for operator splitting
*/
static PetscErrorCode ProjectAuxDM(DM dm, PetscReal time, Vec u, AppCtx *ctx)
{
  DM          dmAux;
  Vec         la, a;
  void       *ctxs[NUM_FIELDS + 1];
  PetscScalar vals[NUM_FIELDS + 1];
  VecScatter  sctAux;
  PetscDS     ds;
  PetscErrorCode (*funcs[NUM_FIELDS + 1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

  PetscFunctionBeginUser;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &la));
  if (!la) {
    PetscFE  field;
    PetscInt fields[NUM_FIELDS];
    IS       is;
    Vec      tu, ta;
    PetscInt dim;

    PetscCall(DMClone(dm, &dmAux));
    PetscCall(DMSetNumFields(dmAux, NUM_FIELDS + 1));
    for (PetscInt i = 0; i < NUM_FIELDS; i++) {
      PetscCall(DMGetField(dm, i, NULL, (PetscObject *)&field));
      PetscCall(DMSetField(dmAux, i, NULL, (PetscObject)field));
      fields[i] = i;
    }
    /* PetscFEDuplicate? */
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, ctx->simplex, "p_", -1, &field));
    PetscCall(PetscObjectSetName((PetscObject)field, "source"));
    PetscCall(DMSetField(dmAux, NUM_FIELDS, NULL, (PetscObject)field));
    PetscCall(PetscFEDestroy(&field));
    PetscCall(DMCreateDS(dmAux));
    PetscCall(DMCreateSubDM(dmAux, NUM_FIELDS, fields, &is, NULL));
    PetscCall(DMGetGlobalVector(dm, &tu));
    PetscCall(DMGetGlobalVector(dmAux, &ta));
    PetscCall(VecScatterCreate(tu, NULL, ta, is, &sctAux));
    PetscCall(DMRestoreGlobalVector(dm, &tu));
    PetscCall(DMRestoreGlobalVector(dmAux, &ta));
    PetscCall(PetscObjectCompose((PetscObject)dmAux, "scatterAux", (PetscObject)sctAux));
    PetscCall(VecScatterDestroy(&sctAux));
    PetscCall(ISDestroy(&is));
    PetscCall(DMCreateLocalVector(dmAux, &la));
    PetscCall(PetscObjectSetName((PetscObject)la, "auxiliary_"));
    PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, la));
    PetscCall(DMDestroy(&dmAux));
    PetscCall(VecDestroy(&la));
  }
  if (time == PETSC_MIN_REAL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &la));
  PetscCall(VecGetDM(la, &dmAux));
  PetscCall(DMGetDS(dmAux, &ds));
  PetscCall(DMGetGlobalVector(dmAux, &a));
  funcs[C_FIELD_ID] = zerof;
  ctxs[C_FIELD_ID]  = NULL;
  funcs[P_FIELD_ID] = zerof;
  ctxs[P_FIELD_ID]  = NULL;
  funcs[NUM_FIELDS] = source;
  ctxs[NUM_FIELDS]  = ctx->source_ctx;
  PetscCall(DMProjectFunction(dmAux, time, funcs, ctxs, INSERT_ALL_VALUES, a));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, zero));
  PetscCall(PetscDSSetObjective(ds, NUM_FIELDS, average));
  PetscCall(DMPlexComputeIntegralFEM(dmAux, a, vals, NULL));
  PetscCall(VecShift(a, -vals[NUM_FIELDS] / ctx->domain_volume));
  if (u) {
    PetscCall(PetscObjectQuery((PetscObject)dmAux, "scatterAux", (PetscObject *)&sctAux));
    PetscCheck(sctAux, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing scatterAux");
    PetscCall(VecScatterBegin(sctAux, u, a, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sctAux, u, a, INSERT_VALUES, SCATTER_FORWARD));
  }
  PetscCall(DMGlobalToLocal(dmAux, a, INSERT_VALUES, la));
  PetscCall(VecViewFromOptions(la, NULL, "-aux_view"));
  PetscCall(DMRestoreGlobalVector(dmAux, &a));
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
    PetscCheck(is, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing potential IS");
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

/* callbacks for residual and Jacobian */
static PetscErrorCode DMPlexTSComputeIFunctionFEM_Private(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *user)
{
  Vec     work, local_lumped_mass;
  AppCtx *ctx = (AppCtx *)user;

  PetscFunctionBeginUser;
  if (ctx->fix_c < 0) {
    PetscReal vals[NUM_FIELDS];
    PetscDS   ds;

    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, jacobian_fail));
    PetscCall(DMPlexSNESComputeObjectiveFEM(dm, locX, vals, NULL));
    PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
    if (vals[C_FIELD_ID]) PetscCall(SNESSetFunctionDomainError(ctx->snes));
  }
  if (ctx->lump) {
    PetscCall(DMGetLumpedMass(dm, PETSC_TRUE, &local_lumped_mass));
    PetscCall(DMGetLocalVector(dm, &work));
    PetscCall(VecSet(work, 0.0));
    PetscCall(DMPlexTSComputeIFunctionFEM(dm, time, locX, work, locF, user));
    PetscCall(VecPointwiseMult(work, locX_t, local_lumped_mass));
    PetscCall(VecAXPY(locF, 1.0, work));
    PetscCall(DMRestoreLocalVector(dm, &work));
    PetscCall(DMRestoreLumpedMass(dm, PETSC_TRUE, &local_lumped_mass));
  } else {
    PetscCall(DMPlexTSComputeIFunctionFEM(dm, time, locX, locX_t, locF, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTSComputeIJacobianFEM_Private(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  Vec     lumped_mass, work;
  AppCtx *ctx = (AppCtx *)user;

  PetscFunctionBeginUser;
  if (ctx->lump) {
    PetscCall(DMGetLumpedMass(dm, PETSC_FALSE, &lumped_mass));
    PetscCall(DMPlexTSComputeIJacobianFEM(dm, time, locX, locX_t, 0.0, Jac, JacP, user));
    PetscCall(DMGetGlobalVector(dm, &work));
    PetscCall(VecAXPBY(work, X_tShift, 0.0, lumped_mass));
    PetscCall(MatDiagonalSet(JacP, work, ADD_VALUES));
    PetscCall(DMRestoreGlobalVector(dm, &work));
    PetscCall(DMRestoreLumpedMass(dm, PETSC_FALSE, &lumped_mass));
  } else {
    PetscCall(DMPlexTSComputeIJacobianFEM(dm, time, locX, locX_t, X_tShift, Jac, JacP, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* customize residuals and Jacobians */
static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS     ds;
  PetscInt    cdim, dim;
  PetscScalar constants[NUM_CONSTANTS];
  PetscScalar vals[NUM_FIELDS];
  PetscInt    fields[NUM_FIELDS] = {C_FIELD_ID, P_FIELD_ID};
  Vec         dummy;
  IS          is;

  PetscFunctionBeginUser;
  constants[R_ID]     = ctx->r;
  constants[EPS_ID]   = ctx->eps;
  constants[ALPHA_ID] = ctx->alpha;
  constants[GAMMA_ID] = ctx->gamma;
  constants[D_ID]     = ctx->D;
  constants[FIXC_ID]  = ctx->fix_c;
  constants[SPLIT_ID] = ctx->split;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(dim == 2 || dim == 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Topological dimension must be 2 or 3");
  PetscCheck(dim == ctx->dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Topological dimension mismatch: expected %" PetscInt_FMT ", found %" PetscInt_FMT, dim, ctx->dim);
  PetscCheck(cdim == ctx->dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Geometrical dimension mismatch: expected %" PetscInt_FMT ", found %" PetscInt_FMT, cdim, ctx->dim);
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  PetscCall(PetscDSSetImplicit(ds, C_FIELD_ID, PETSC_TRUE));
  PetscCall(PetscDSSetImplicit(ds, P_FIELD_ID, PETSC_TRUE));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));
  if (ctx->dim == 2) {
    PetscCall(PetscDSSetResidual(ds, C_FIELD_ID, C_0, C_1));
    PetscCall(PetscDSSetResidual(ds, P_FIELD_ID, P_0, P_1));
    PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, C_FIELD_ID, JC_0_c0c0, NULL, NULL, ctx->D > 0 ? JC_1_c1c1 : NULL));
    if (!ctx->split) { /* we solve a block diagonal system to mimic operator splitting, jacobians will not be correct */
      PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, P_FIELD_ID, NULL, JC_0_c0p1, NULL, NULL));
      PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, C_FIELD_ID, NULL, NULL, JP_1_p1c0, NULL));
    }
    PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, P_FIELD_ID, NULL, NULL, NULL, JP_1_p1p1));
  } else {
    PetscCall(PetscDSSetResidual(ds, C_FIELD_ID, C_0_3d, C_1_3d));
    PetscCall(PetscDSSetResidual(ds, P_FIELD_ID, P_0, P_1_3d));
    PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, C_FIELD_ID, JC_0_c0c0_3d, NULL, NULL, ctx->D > 0 ? JC_1_c1c1_3d : NULL));
    if (!ctx->split) {
      PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, P_FIELD_ID, NULL, JC_0_c0p1_3d, NULL, NULL));
      PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, C_FIELD_ID, NULL, NULL, JP_1_p1c0_3d, NULL));
    }
    PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, P_FIELD_ID, NULL, NULL, NULL, JP_1_p1p1_3d));
  }
  /* Attach potential nullspace */
  PetscCall(DMSetNullSpaceConstructor(dm, P_FIELD_ID, CreatePotentialNullSpace));

  /* Compute domain volume */
  PetscCall(DMGetGlobalVector(dm, &dummy));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, volume));
  PetscCall(DMPlexComputeIntegralFEM(dm, dummy, vals, NULL));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, zero));
  PetscCall(DMRestoreGlobalVector(dm, &dummy));
  ctx->domain_volume = PetscRealPart(vals[P_FIELD_ID]);

  /* Index sets for potential and conductivities */
  PetscCall(DMCreateSubDM(dm, 1, fields, &is, NULL));
  PetscCall(PetscObjectCompose((PetscObject)dm, "IS conductivity", (PetscObject)is));
  PetscCall(PetscObjectSetName((PetscObject)is, "C"));
  PetscCall(ISViewFromOptions(is, NULL, "-is_conductivity_view"));
  PetscCall(ISDestroy(&is));
  PetscCall(DMCreateSubDM(dm, 1, fields + 1, &is, NULL));
  PetscCall(PetscObjectSetName((PetscObject)is, "P"));
  PetscCall(PetscObjectCompose((PetscObject)dm, "IS potential", (PetscObject)is));
  PetscCall(ISViewFromOptions(is, NULL, "-is_potential_view"));
  PetscCall(ISDestroy(&is));

  /* Create auxiliary DM */
  PetscCall(ProjectAuxDM(dm, PETSC_MIN_REAL, NULL, ctx));

  /* Add callbacks */
  PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM_Private, ctx));
  PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM_Private, ctx));
  /* DMPlexTSComputeBoundary is not needed because we use natural boundary conditions */
  PetscCall(DMTSSetBoundaryLocal(dm, NULL, NULL));

  /* handle nullspace in residual (move it to TSComputeIFunction_DMLocal?) */
  {
    MatNullSpace nullsp;

    PetscCall(CreatePotentialNullSpace(dm, P_FIELD_ID, P_FIELD_ID, &nullsp));
    PetscCall(PetscObjectCompose((PetscObject)dm, "__dmtsnullspace", (PetscObject)nullsp));
    PetscCall(MatNullSpaceDestroy(&nullsp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* setup discrete spaces and residuals */
static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM       cdm = dm;
  PetscFE  feC, feP;
  PetscInt dim;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  /* We model Cij with Cij = Cji -> dim*(dim+1)/2 components */
  PetscCall(PetscFECreateDefault(comm, dim, (dim * (dim + 1)) / 2, ctx->simplex, "c_", -1, &feC));
  PetscCall(PetscObjectSetName((PetscObject)feC, "conductivity"));
  PetscCall(PetscFECreateDefault(comm, dim, 1, ctx->simplex, "p_", -1, &feP));
  PetscCall(PetscObjectSetName((PetscObject)feP, "potential"));
  PetscCall(PetscFECopyQuadrature(feP, feC));
  PetscCall(PetscFEViewFromOptions(feP, NULL, "-view_fe"));
  PetscCall(PetscFEViewFromOptions(feC, NULL, "-view_fe"));

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
  DM plex;

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
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "ref_"));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, NULL));

  PetscCall(DMConvert(*dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &ctx->simplex));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &ctx->simplex, 1, MPI_C_BOOL, MPI_LOR, comm));
  PetscCall(DMDestroy(&plex));

  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Make potential field zero mean */
static PetscErrorCode ZeroMeanPotential(DM dm, Vec u, PetscScalar domain_volume)
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
  PetscCall(VecISShift(u, is, -vals[P_FIELD_ID] / domain_volume));
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
  Vec        u, p, subaux, vatol, vrtol;
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
  PetscCall(PetscObjectQuery((PetscObject)dm, "IS potential", (PetscObject *)&isp));
  PetscCheck(isp, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing potential IS");
  if (valid) {
    if (ctx->exclude_potential_lte) {
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
    }
    for (PetscInt i = 0; i < nv; i++) PetscCall(ZeroMeanPotential(dm, vecs[i], ctx->domain_volume));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMCreateSubDM(dm, 1, fields + 1, NULL, &dmp));
  PetscCall(DMSetMatType(dmp, MATAIJ));
  PetscCall(DMGetDS(dmp, &ds));
  if (ctx->dim == 2) {
    PetscCall(PetscDSSetResidual(ds, 0, P_0_aux, P_1_aux));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, JP_1_p1p1_aux));
  } else {
    PetscCall(PetscDSSetResidual(ds, 0, P_0_aux, P_1_aux_3d));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, JP_1_p1p1_aux_3d));
  }
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
    u = vecs[i];
    if (!valid) { /* Assumes entries in u are not valid */
      PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
      void       *ctxs[NUM_FIELDS] = {NULL};
      PetscRandom r                = NULL;

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
      case 3:
        funcs[C_FIELD_ID] = initial_conditions_C_random;
        PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)ts), &r));
        PetscCall(PetscRandomSetOptionsPrefix(r, "ic_"));
        PetscCall(PetscRandomSetFromOptions(r));
        ctxs[C_FIELD_ID] = r;
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Unknown IC");
      }
      funcs[P_FIELD_ID] = zerof;
      PetscCall(DMProjectFunction(dm, t, funcs, ctxs, INSERT_ALL_VALUES, u));
      PetscCall(ProjectAuxDM(dm, t, u, ctx));
      PetscCall(PetscRandomDestroy(&r));
    }

    /* pass conductivity information via auxiliary data */
    PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &subaux));
    PetscCall(DMSetAuxiliaryVec(dmp, NULL, 0, 0, subaux));

    /* solve the subproblem */
    if (!sctp) PetscCall(VecScatterCreate(u, isp, p, NULL, &sctp));
    PetscCall(VecScatterBegin(sctp, u, p, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sctp, u, p, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(SNESSolve(snes, NULL, p));

    /* scatter from potential only to full space */
    PetscCall(VecScatterBegin(sctp, p, u, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(sctp, p, u, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(ZeroMeanPotential(dm, u, ctx->domain_volume));
  }
  PetscCall(VecDestroy(&p));
  PetscCall(DMDestroy(&dmp));
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecScatterDestroy(&sctp));

  /* exclude potential from computation of the LTE */
  if (ctx->exclude_potential_lte) {
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
  }
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
  PetscInt  fields[NUM_FIELDS] = {C_FIELD_ID, P_FIELD_ID};
  IS        is;

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
  PetscCall(DMCreateSubDM(adm, 1, fields, &is, NULL));
  PetscCall(PetscObjectCompose((PetscObject)adm, "IS conductivity", (PetscObject)is));
  PetscCall(ISDestroy(&is));
  PetscCall(DMCreateSubDM(adm, 1, fields + 1, &is, NULL));
  PetscCall(PetscObjectCompose((PetscObject)adm, "IS potential", (PetscObject)is));
  PetscCall(ISDestroy(&is));
  PetscCall(ProjectAuxDM(adm, time, NULL, ctx));
  {
    MatNullSpace nullsp;

    PetscCall(CreatePotentialNullSpace(adm, P_FIELD_ID, P_FIELD_ID, &nullsp));
    PetscCall(PetscObjectCompose((PetscObject)adm, "__dmtsnullspace", (PetscObject)nullsp));
    PetscCall(MatNullSpaceDestroy(&nullsp));
  }
  PetscCall(SetInitialConditionsAndTolerances(ts, nv, vecsout, PETSC_TRUE));
  PetscCall(DMDestroy(&ctx->view_dm));
  for (PetscInt i = 0; i < NUM_FIELDS; i++) {
    PetscCall(VecScatterDestroy(&ctx->subsct[i]));
    PetscCall(VecDestroy(&ctx->subvec[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeDiagnostic(Vec u, AppCtx *ctx, Vec *out)
{
  DM       dm;
  PetscInt dim;
  PetscFE  feFluxC, feNormC, feEigsC;

  void (*funcs[])(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]) = {normc, eigsc, flux};

  PetscFunctionBeginUser;
  if (!ctx->view_dm) {
    PetscFE  feP;
    PetscInt nf = PetscMax(PetscMin(ctx->diagnostic_num, 3), 1);

    PetscCall(VecGetDM(u, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, ctx->simplex, "normc_", -1, &feNormC));
    PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, dim, ctx->simplex, "eigsc_", -1, &feEigsC));
    PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, dim, ctx->simplex, "flux_", -1, &feFluxC));
    PetscCall(DMGetField(dm, P_FIELD_ID, NULL, (PetscObject *)&feP));
    PetscCall(PetscFECopyQuadrature(feP, feNormC));
    PetscCall(PetscFECopyQuadrature(feP, feEigsC));
    PetscCall(PetscFECopyQuadrature(feP, feFluxC));
    PetscCall(PetscObjectSetName((PetscObject)feNormC, "normC"));
    PetscCall(PetscObjectSetName((PetscObject)feEigsC, "eigsC"));
    PetscCall(PetscObjectSetName((PetscObject)feFluxC, "flux"));

    PetscCall(DMClone(dm, &ctx->view_dm));
    PetscCall(DMSetNumFields(ctx->view_dm, nf));
    PetscCall(DMSetField(ctx->view_dm, 0, NULL, (PetscObject)feNormC));
    if (nf > 1) PetscCall(DMSetField(ctx->view_dm, 1, NULL, (PetscObject)feEigsC));
    if (nf > 2) PetscCall(DMSetField(ctx->view_dm, 2, NULL, (PetscObject)feFluxC));
    PetscCall(DMCreateDS(ctx->view_dm));
    PetscCall(PetscFEDestroy(&feFluxC));
    PetscCall(PetscFEDestroy(&feNormC));
    PetscCall(PetscFEDestroy(&feEigsC));
  }
  PetscCall(DMCreateGlobalVector(ctx->view_dm, out));
  PetscCall(DMProjectField(ctx->view_dm, 0.0, u, funcs, INSERT_VALUES, *out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MakeScatterAndVec(Vec X, IS is, Vec *Y, VecScatter *sct)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)X), n, PETSC_DECIDE, Y));
  PetscCall(VecScatterCreate(X, is, *Y, NULL, sct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FunctionDomainError(TS ts, PetscReal time, Vec X, PetscBool *accept)
{
  AppCtx     *ctx;
  PetscScalar vals[NUM_FIELDS];
  DM          dm;
  PetscDS     ds;

  PetscFunctionBeginUser;
  *accept = PETSC_TRUE;
  PetscCall(TSGetApplicationContext(ts, &ctx));
  if (ctx->function_domain_error_tol < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, ellipticity_fail));
  PetscCall(DMPlexComputeIntegralFEM(dm, X, vals, NULL));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
  if (PetscAbsScalar(vals[C_FIELD_ID]) > ctx->function_domain_error_tol) *accept = PETSC_FALSE;
  if (!*accept) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "Domain error value %g > %g\n", (double)PetscAbsScalar(vals[C_FIELD_ID]), (double)ctx->function_domain_error_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Monitor relevant functionals */
static PetscErrorCode Monitor(TS ts, PetscInt stepnum, PetscReal time, Vec u, void *vctx)
{
  PetscScalar vals[2 * NUM_FIELDS];
  DM          dm;
  PetscDS     ds;
  AppCtx     *ctx;
  PetscBool   need_hdf5, need_vtk;

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
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
  vals[C_FIELD_ID] /= ctx->domain_volume;

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "%4" PetscInt_FMT " TS: time %g, energy %g, intp %g, ell %g\n", stepnum, (double)time, (double)PetscRealPart(vals[C_FIELD_ID]), (double)PetscRealPart(vals[P_FIELD_ID]), (double)PetscRealPart(vals[NUM_FIELDS + C_FIELD_ID])));

  /* monitor diagnostic */
  need_hdf5 = (PetscBool)(ctx->view_hdf5_ctx && ((ctx->view_hdf5_ctx->view_interval > 0 && !(stepnum % ctx->view_hdf5_ctx->view_interval)) || (ctx->view_hdf5_ctx->view_interval && ts->reason)));
  need_vtk  = (PetscBool)(ctx->view_vtk_ctx && ((ctx->view_vtk_ctx->interval > 0 && !(stepnum % ctx->view_vtk_ctx->interval)) || (ctx->view_vtk_ctx->interval && ts->reason)));
  if (ctx->view_times_k < ctx->view_times_n && time >= ctx->view_times[ctx->view_times_k] && time < ctx->view_times[ctx->view_times_k + 1]) {
    if (ctx->view_hdf5_ctx) need_hdf5 = PETSC_TRUE;
    if (ctx->view_vtk_ctx) need_vtk = PETSC_TRUE;
    ctx->view_times_k++;
  }
  if (need_vtk || need_hdf5) {
    Vec       diagnostic;
    PetscBool dump_dm = (PetscBool)(!!ctx->view_dm);

    PetscCall(ComputeDiagnostic(u, ctx, &diagnostic));
    if (need_vtk) {
      PetscCall(PetscObjectSetName((PetscObject)diagnostic, ""));
      PetscCall(TSMonitorSolutionVTK(ts, stepnum, time, diagnostic, ctx->view_vtk_ctx));
    }
    if (need_hdf5) {
      DM       vdm;
      PetscInt seqnum;

      PetscCall(VecGetDM(diagnostic, &vdm));
      if (!dump_dm) {
        PetscCall(PetscViewerPushFormat(ctx->view_hdf5_ctx->viewer, ctx->view_hdf5_ctx->format));
        PetscCall(DMView(vdm, ctx->view_hdf5_ctx->viewer));
        PetscCall(PetscViewerPopFormat(ctx->view_hdf5_ctx->viewer));
      }
      PetscCall(DMGetOutputSequenceNumber(vdm, &seqnum, NULL));
      PetscCall(DMSetOutputSequenceNumber(vdm, seqnum + 1, time));
      PetscCall(PetscObjectSetName((PetscObject)diagnostic, "diagnostic"));
      PetscCall(PetscViewerPushFormat(ctx->view_hdf5_ctx->viewer, ctx->view_hdf5_ctx->format));
      PetscCall(VecView(diagnostic, ctx->view_hdf5_ctx->viewer));
      if (ctx->diagnostic_num > 3 || ctx->diagnostic_num < 0) {
        PetscCall(DMSetOutputSequenceNumber(dm, seqnum + 1, time));
        PetscCall(VecView(u, ctx->view_hdf5_ctx->viewer));
      }
      PetscCall(PetscViewerPopFormat(ctx->view_hdf5_ctx->viewer));
    }
    PetscCall(VecDestroy(&diagnostic));
  }
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

/* Resample source if time dependent */
static PetscErrorCode PreStage(TS ts, PetscReal stagetime)
{
  AppCtx   *ctx;
  PetscBool resample, ismatis;
  Mat       A, P;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &ctx));
  /* in case we need to call SNESSetFunctionDomainError */
  PetscCall(TSGetSNES(ts, &ctx->snes));

  resample = ctx->split ? PETSC_TRUE : PETSC_FALSE;
  for (PetscInt i = 0; i < ctx->source_ctx->n; i++) {
    if (ctx->source_ctx->k[i] != 0.0) {
      resample = PETSC_TRUE;
      break;
    }
  }
  if (resample) {
    DM  dm;
    Vec u = NULL;

    PetscCall(TSGetDM(ts, &dm));
    /* In case of a multistage method, we always use the frozen values at the previous time-step */
    if (ctx->split) PetscCall(TSGetSolution(ts, &u));
    PetscCall(ProjectAuxDM(dm, stagetime, u, ctx));
  }

  /* element matrices are sparse, ignore zero entries */
  PetscCall(TSGetIJacobian(ts, &A, &P, NULL, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATIS, &ismatis));
  if (!ismatis) PetscCall(MatSetOption(P, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));

  /* Set symmetric flag */
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(P, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(MatSetOption(P, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
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
  PetscCall(TSGetApplicationContext(ts, &ctx));

  PetscCall(ZeroMeanPotential(dm, u, ctx->domain_volume));

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

PetscErrorCode MonitorNorms(SNES snes, PetscInt its, PetscReal f, void *vctx)
{
  AppCtx   *ctx = (AppCtx *)vctx;
  Vec       F;
  DM        dm;
  PetscReal subnorm[NUM_FIELDS];

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(SNESGetFunction(snes, &F, NULL, NULL));
  if (!ctx->subsct[C_FIELD_ID]) {
    IS is;

    PetscCall(PetscObjectQuery((PetscObject)dm, "IS conductivity", (PetscObject *)&is));
    PetscCheck(is, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing conductivity IS");
    PetscCall(MakeScatterAndVec(F, is, &ctx->subvec[C_FIELD_ID], &ctx->subsct[C_FIELD_ID]));
  }
  if (!ctx->subsct[P_FIELD_ID]) {
    IS is;

    PetscCall(PetscObjectQuery((PetscObject)dm, "IS potential", (PetscObject *)&is));
    PetscCheck(is, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing potential IS");
    PetscCall(MakeScatterAndVec(F, is, &ctx->subvec[P_FIELD_ID], &ctx->subsct[P_FIELD_ID]));
  }
  PetscCall(VecScatterBegin(ctx->subsct[C_FIELD_ID], F, ctx->subvec[C_FIELD_ID], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->subsct[C_FIELD_ID], F, ctx->subvec[C_FIELD_ID], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(ctx->subsct[P_FIELD_ID], F, ctx->subvec[P_FIELD_ID], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->subsct[P_FIELD_ID], F, ctx->subvec[P_FIELD_ID], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecNorm(ctx->subvec[C_FIELD_ID], NORM_2, &subnorm[C_FIELD_ID]));
  PetscCall(VecNorm(ctx->subvec[P_FIELD_ID], NORM_2, &subnorm[P_FIELD_ID]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "    %3" PetscInt_FMT " SNES Function norms %14.12e, %14.12e\n", its, (double)subnorm[C_FIELD_ID], (double)subnorm[P_FIELD_ID]));
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
    PetscCall(PetscPrintf(comm, "  dim  : %" PetscInt_FMT "\n", ctx->dim));
    PetscCall(PetscPrintf(comm, "  r    : %g\n", (double)ctx->r));
    PetscCall(PetscPrintf(comm, "  eps  : %g\n", (double)ctx->eps));
    PetscCall(PetscPrintf(comm, "  alpha: %g\n", (double)ctx->alpha));
    PetscCall(PetscPrintf(comm, "  gamma: %g\n", (double)ctx->gamma));
    PetscCall(PetscPrintf(comm, "  D    : %g\n", (double)ctx->D));
    if (ctx->load) PetscCall(PetscPrintf(comm, "  load : %s\n", ctx->load_filename));
    else PetscCall(PetscPrintf(comm, "  IC   : %" PetscInt_FMT "\n", ctx->ic_num));
    PetscCall(PetscPrintf(comm, "  snum : %" PetscInt_FMT "\n", ctx->source_ctx->n));
    for (PetscInt i = 0; i < ctx->source_ctx->n; i++) {
      const PetscReal *x0 = ctx->source_ctx->x0 + ctx->dim * i;
      const PetscReal  w  = ctx->source_ctx->w[i];
      const PetscReal  k  = ctx->source_ctx->k[i];
      const PetscReal  p  = ctx->source_ctx->p[i];
      const PetscReal  r  = ctx->source_ctx->r[i];

      if (ctx->dim == 2) {
        PetscCall(PetscPrintf(comm, "  x0   : (%g,%g)\n", (double)x0[0], (double)x0[1]));
      } else {
        PetscCall(PetscPrintf(comm, "  x0   : (%g,%g,%g)\n", (double)x0[0], (double)x0[1], (double)x0[2]));
      }
      PetscCall(PetscPrintf(comm, "  scals: (%g,%g,%g,%g)\n", (double)w, (double)k, (double)p, (double)r));
    }
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
  PetscCall(TSSetPreStage(ts, PreStage));
  PetscCall(TSSetPostStage(ts, PostStage));
  PetscCall(TSSetMaxSNESFailures(ts, -1));
  PetscCall(TSSetFunctionDomainError(ts, FunctionDomainError));
  PetscCall(TSSetFromOptions(ts));
  if (ctx->monitor_norms) {
    SNES snes;

    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESMonitorSet(snes, MonitorNorms, ctx, NULL));
  }

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution_"));
  PetscCall(DMHasNamedGlobalVector(dm, "solution_", &flg));
  if (flg) { /* load from restart file */
    Vec ru;

    PetscCall(DMGetNamedGlobalVector(dm, "solution_", &ru));
    PetscCall(VecCopy(ru, u));
    PetscCall(DMRestoreNamedGlobalVector(dm, "solution_", &ru));
  }
  PetscCall(SetInitialConditionsAndTolerances(ts, 1, &u, flg));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));
  if (!ctx->test_restart) PetscCall(PetscLogStagePop());

  if (!ctx->test_restart) PetscCall(PetscLogStagePush(SolveStage));
  PetscCall(TSSolve(ts, NULL));
  if (!ctx->test_restart) PetscCall(PetscLogStagePop());
  if (ctx->view_vtk_ctx) PetscCall(TSMonitorSolutionVTKDestroy(&ctx->view_vtk_ctx));
  if (ctx->view_hdf5_ctx) PetscCall(PetscViewerAndFormatDestroy(&ctx->view_hdf5_ctx));
  PetscCall(DMDestroy(&ctx->view_dm));
  for (PetscInt i = 0; i < NUM_FIELDS; i++) {
    PetscCall(VecScatterDestroy(&ctx->subsct[i]));
    PetscCall(VecDestroy(&ctx->subvec[i]));
  }

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
  PetscCall(PetscFree5(ctx.source_ctx->x0, ctx.source_ctx->w, ctx.source_ctx->k, ctx.source_ctx->p, ctx.source_ctx->r));
  PetscCall(PetscFree(ctx.source_ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_box_faces 3,3 -ksp_type preonly -pc_type svd -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 1 -initial_snes_test_jacobian -snes_test_jacobian -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -ic_num 3

    test:
      suffix: 0
      nsize: {{1 2}}
      args: -dm_refine 1 -lump {{0 1}} -exclude_potential_lte

    test:
      suffix: 0_split
      nsize: {{1 2}}
      args: -dm_refine 1 -split

    test:
      suffix: 0_3d
      nsize: {{1 2}}
      args: -dm_plex_box_faces 3,3,3 -dim 3 -dm_plex_dim 3 -lump {{0 1}}

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

  test:
    suffix: annulus
    requires: exodusii
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -ksp_type preonly -pc_type none -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 2 -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -source_num 2 -source_k 1,2

  test:
    suffix: hdf5_diagnostic
    requires: hdf5 exodusii
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -ksp_type preonly -pc_type none -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 2 -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -source_num 2 -source_k 1,2 -monitor_hdf5 diagnostic.h5 -ic_num 3

  test:
    suffix: vtk_diagnostic
    requires: exodusii
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -ksp_type preonly -pc_type none -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 2 -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -source_num 2 -source_k 1,2 -monitor_vtk 'diagnostic-%03d.vtu' -ic_num 3

  test:
    suffix: full_cdisc
    args: -dm_plex_box_faces 3,3 -c_petscspace_degree 0 -p_petscspace_degree 1 -ts_max_steps 1 -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -ic_num 0 -dm_refine 1 -ts_type beuler -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_precondition selfp -fieldsplit_conductivity_pc_type pbjacobi -fieldsplit_potential_mat_schur_complement_ainv_type blockdiag -fieldsplit_potential_ksp_type preonly -fieldsplit_potential_pc_type svd

  test:
    suffix: full_cdisc_split
    args: -dm_plex_box_faces 3,3 -c_petscspace_degree 0 -p_petscspace_degree 1 -ts_max_steps 1 -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -ic_num 0 -dm_refine 1 -ts_type beuler -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_conductivity_pc_type pbjacobi -fieldsplit_potential_pc_type gamg -split -monitor_norms

  test:
    suffix: full_cdisc_minres
    args: -dm_plex_box_faces 3,3 -c_petscspace_degree 0 -p_petscspace_degree 1 -ts_max_steps 1 -petscpartitioner_type simple -dm_plex_simplex 0 -ts_adapt_type none -ic_num 0 -dm_refine 1 -ts_type beuler -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type diag -pc_fieldsplit_schur_precondition selfp -fieldsplit_conductivity_pc_type pbjacobi -fieldsplit_potential_mat_schur_complement_ainv_type blockdiag -fieldsplit_potential_ksp_type preonly -fieldsplit_potential_pc_type svd -ksp_type minres

TEST*/
