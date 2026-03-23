static char help[] = "Test for DMPlexMetricIntersection using two constant 3x3 metrics.\n\n";

#include <petscdmplex.h>
#include <petscblaslapack.h>
#include <petscmath.h>

/* Euler z-x-z (extrinsic) rotation same as used in DMPlexCreateBasisRotation
   R = Rz(alpha) * Rx(beta) * Rz(gamma) */

static void RotationZ(PetscScalar a, PetscScalar R[3][3])
{
  PetscScalar c = PetscCosScalar(a), s = PetscSinScalar(a);
  R[0][0] = c;
  R[0][1] = -s;
  R[0][2] = 0.;
  R[1][0] = s;
  R[1][1] = c;
  R[1][2] = 0.;
  R[2][0] = 0.;
  R[2][1] = 0.;
  R[2][2] = 1.;
}

static void RotationX(PetscScalar b, PetscScalar R[3][3])
{
  PetscScalar c = PetscCosScalar(b), s = PetscSinScalar(b);
  R[0][0] = 1.;
  R[0][1] = 0.;
  R[0][2] = 0.;
  R[1][0] = 0.;
  R[1][1] = c;
  R[1][2] = -s;
  R[2][0] = 0.;
  R[2][1] = s;
  R[2][2] = c;
}

static void MatMult3(const PetscScalar A[3][3], const PetscScalar B[3][3], PetscScalar C[3][3])
{
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      C[i][j] = 0.0;
      for (PetscInt k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
    }
}

static void EulerZXZ(PetscScalar a, PetscScalar b, PetscScalar g, PetscScalar Q[3][3])
{
  PetscScalar Rz1[3][3], Rx[3][3], Rz2[3][3], T[3][3];
  RotationZ(a, Rz1);
  RotationX(b, Rx);
  RotationZ(g, Rz2);
  MatMult3(Rz1, Rx, T);
  MatMult3(T, Rz2, Q);
}

/* Build metric M = Q^T D Q given Euler and eigenvalues */
static void BuildMetric(PetscScalar a, PetscScalar b, PetscScalar g, const PetscScalar lam[3], PetscScalar M[3][3])
{
  PetscScalar Q[3][3], QT[3][3], DQ[3][3];
  EulerZXZ(a, b, g, Q);
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) QT[i][j] = Q[j][i];
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) DQ[i][j] = lam[i] * Q[i][j];
  MatMult3(QT, DQ, M);
}

/* Write a constant 3x3 metric into the metric Vec */
static PetscErrorCode SetConstantMetricVertices(DM dm, Vec metric, const PetscScalar M[3][3])
{
  PetscFunctionBeginUser;
  PetscInt     vStart, vEnd;
  PetscSection sec;
  Vec          loc;
  PetscScalar *lptr;

  PetscCall(DMGetLocalSection(dm, &sec));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));

  PetscCall(DMGetLocalVector(dm, &loc));
  PetscCall(VecZeroEntries(loc));
  PetscCall(VecGetArray(loc, &lptr));

  for (PetscInt p = vStart; p < vEnd; ++p) {
    PetscInt dof, off;
    PetscCall(PetscSectionGetDof(sec, p, &dof));
    PetscCall(PetscSectionGetOffset(sec, p, &off));
    if (dof == 9) {
      PetscInt t = 0;
      for (PetscInt i = 0; i < 3; i++)
        for (PetscInt j = 0; j < 3; j++) lptr[off + t++] = M[i][j];
    } else if (dof > 0) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected dof count %" PetscInt_FMT, dof);
    }
  }

  PetscCall(VecRestoreArray(loc, &lptr));
  PetscCall(DMLocalToGlobal(dm, loc, INSERT_VALUES, metric));
  PetscCall(DMRestoreLocalVector(dm, &loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Loewner (semidefinite) inequality check via eigenvalues of A-B, using LAPACK syev
   Loewner inequality: A >= B iff A-B is positive semi-definite */
static PetscBool LoewnerGE(const PetscScalar A[3][3], const PetscScalar B[3][3], PetscScalar tol)
{
  /* Build C = A - B in flat buffer */
  PetscScalar C[9];
  {
    PetscInt t = 0;
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt j = 0; j < 3; j++) C[t++] = A[i][j] - B[i][j];
  }

  PetscScalar  w[3];    /* eigenvalues */
  PetscScalar  work[9]; /* minimal workspace for 3x3 */
  PetscBLASInt n = 3, lda = 3, lwork = 9, info;

  /* jobz='N' (eigenvalues only), uplo='L' (lower-triangular part).
     Column-major vs row-major does not matter for symmetric matrices. */
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("N", "L", &n, C, &lda, w, work, &lwork, &info));
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKsyev failed in LoewnerGE info=%" PetscInt_FMT, info);

  /* Smallest eigenvalue >= -tol? */
  return (w[0] >= -tol) ? PETSC_TRUE : PETSC_FALSE;
}

int main(int argc, char **argv)
{
  DM          dm;
  Vec         m1, m2, mcap;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* 1) Tiny 3D simplex mesh */
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 3, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_FALSE, &dm));
  PetscCall(DMSetUp(dm));

  /* Allocate metric vectors with the correct layout for this DM */
  PetscCall(DMPlexMetricCreate(dm, 0, &m1));
  PetscCall(DMPlexMetricCreate(dm, 0, &m2));
  PetscCall(DMPlexMetricCreate(dm, 0, &mcap));

  /* 2) Two constant metrics M1, M2 via Q^T D Q (aligned eigenbasis) */
  PetscScalar a = 0.3, b = 0.7, g = 1.1; /* Euler angles (z-x-z extrinsic) */
  PetscScalar lam1[3] = {1.0, 4.0, 9.0};
  PetscScalar lam2[3] = {2.0, 3.0, 16.0};

  PetscScalar M1mat[3][3], M2mat[3][3], Mcap_expected[3][3];
  BuildMetric(a, b, g, lam1, M1mat);
  BuildMetric(a, b, g, lam2, M2mat);

  /* Expected intersection for aligned eigenbasis: diag(max(lam1, lam2)) */
  PetscScalar lamcap[3] = {PetscMax(lam1[0], lam2[0]), PetscMax(lam1[1], lam2[1]), PetscMax(lam1[2], lam2[2])};
  BuildMetric(a, b, g, lamcap, Mcap_expected);

  PetscCall(VecZeroEntries(m1));
  PetscCall(VecZeroEntries(m2));
  PetscCall(SetConstantMetricVertices(dm, m1, M1mat));
  PetscCall(SetConstantMetricVertices(dm, m2, M2mat));

  /* 3) Intersect */
  Vec metrics[2] = {m1, m2};
  PetscCall(DMPlexMetricIntersection(dm, 2, metrics, mcap));

  /* 4) Verify Loewner order and aligned-eigenbasis equality */
  PetscInt vStart, vEnd;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  const PetscScalar *arr;
  PetscCall(VecGetArrayRead(mcap, &arr));
  PetscSection sec;
  PetscCall(DMGetLocalSection(dm, &sec));

  for (PetscInt p = vStart; p < vEnd; ++p) {
    PetscInt dof, off;
    PetscCall(PetscSectionGetDof(sec, p, &dof));
    PetscCall(PetscSectionGetOffset(sec, p, &off));
    if (dof == 9) {
      /* Inline read: directly from arr into Mread */
      PetscScalar Mread[3][3];
      for (PetscInt i = 0, t = 0; i < 3; i++)
        for (PetscInt j = 0; j < 3; j++) Mread[i][j] = arr[off + t++];

      /* Loewner checks */
      PetscBool ge1 = LoewnerGE(Mread, M1mat, 1e-10);
      PetscBool ge2 = LoewnerGE(Mread, M2mat, 1e-10);
      PetscCheck(ge1 && ge2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Intersection not >= inputs in Loewner order");

      /* Exact match for aligned eigenbasis */
      PetscScalar maxdiff = 0.0;
      for (PetscInt i = 0; i < 3; i++)
        for (PetscInt j = 0; j < 3; j++) {
          PetscScalar d = fabs(Mread[i][j] - Mcap_expected[i][j]);
          if (d > maxdiff) maxdiff = d;
        }
      PetscCheck(maxdiff < 1e-12, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Aligned-eigenbasis formula failed at vertex (maxdiff=%.3e)", maxdiff);
    }
  }
  PetscCall(VecRestoreArrayRead(mcap, &arr));

  if (!rank) PetscCall(PetscPrintf(PETSC_COMM_SELF, "OK: Intersection passed (3x3 @ vertices, aligned case).\n"));

  /* misaligned subtest: change Q of M2 */
  {
    PetscScalar a2 = a + 0.2, b2 = b - 0.1, g2 = g + 0.3;
    BuildMetric(a2, b2, g2, lam2, M2mat);
    PetscCall(VecZeroEntries(m2));
    PetscCall(SetConstantMetricVertices(dm, m2, M2mat));

    PetscCall(DMPlexMetricIntersection(dm, 2, metrics, mcap));

    const PetscScalar *ar2;
    PetscCall(VecGetArrayRead(mcap, &ar2));
    for (PetscInt p = vStart; p < vEnd; ++p) {
      PetscInt dof, off;
      PetscCall(PetscSectionGetDof(sec, p, &dof));
      PetscCall(PetscSectionGetOffset(sec, p, &off));
      if (dof == 9) {
        PetscScalar Mread[3][3];
        for (PetscInt i = 0, t = 0; i < 3; i++)
          for (PetscInt j = 0; j < 3; j++) Mread[i][j] = ar2[off + t++];

        PetscBool ge1 = LoewnerGE(Mread, M1mat, 1e-10);
        PetscBool ge2 = LoewnerGE(Mread, M2mat, 1e-10);
        PetscCheck(ge1 && ge2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Intersection not >= inputs (misaligned)");
      }
    }
    PetscCall(VecRestoreArrayRead(mcap, &ar2));
    if (!rank) PetscCall(PetscPrintf(PETSC_COMM_SELF, "OK: Intersection passed Loewner checks (misaligned Q).\n"));
  }

  PetscCall(VecDestroy(&m1));
  PetscCall(VecDestroy(&m2));
  PetscCall(VecDestroy(&mcap));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: !complex
  test:
    requires: ctetgen

TEST*/
