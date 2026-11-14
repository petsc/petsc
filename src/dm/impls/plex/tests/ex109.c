static char help[] = "Tests boundary integrals on every face of a hexahedral box mesh.\n\n";

/*
For each Face Sets value, the boundary load

  L(w) = int_face g w dS

is assembled with DMPlexComputeBdResidualSingle(), and compared with a 10x10 Gauss rule on the face.
Here

  g = 1 + x + 2y + 3z + 4xyz
  w = (1+x)(1+y)(1+z),

so the element quadrature integrates g w exactly and every face should agree to rounding error.

The option -face_orientation o acts with orientation o on every boundary face, to test rotated and reflected faces.

The option -check_jacobian instead assembles the boundary residual int_face grad u . grad v dS for u = w, and its Jacobian
with DMPlexComputeBdJacobianSingle(). The residual is linear in u, so it must equal the Jacobian times u. This check needs no
exact integral, so it also runs on meshes with non-affine cells.
*/

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscInt  k;   // FEM space degree
  PetscInt  o;   // Orientation for faces
  PetscBool jac; // Check the boundary residual against its Jacobian
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->k   = 1;
  options->o   = 0;
  options->jac = PETSC_FALSE;
  PetscOptionsBegin(comm, "", "Boundary integral options", "DMPLEX");
  PetscCall(PetscOptionsInt("-k", "Polynomial degree of the Lagrange element", __FILE__, options->k, &options->k, NULL));
  PetscCall(PetscOptionsInt("-face_orientation", "Orientation applied to every boundary face", __FILE__, options->o, &options->o, NULL));
  PetscCall(PetscOptionsBool("-check_jacobian", "Check the boundary residual against its Jacobian", __FILE__, options->jac, &options->jac, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscInt  dim;
  PetscBool simplex;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscCall(DMGetDimension(*dm, &dim));
  PetscCall(DMPlexIsSimplex(*dm, &simplex));
  PetscCheck(dim == 3 && simplex == PETSC_FALSE, comm, PETSC_ERR_SUP, "This test requires a 3D hexahedral box mesh");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OrientFaces(DM dm, DMLabel label, PetscInt o)
{
  PetscInt fStart, fEnd;

  PetscFunctionBeginUser;
  if (!o) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (PetscInt f = fStart; f < fEnd; ++f) {
    PetscInt val;

    PetscCall(DMLabelGetValue(label, f, &val));
    if (val >= 0) PetscCall(DMPlexOrientPoint(dm, f, o));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Quadrature of order k + 2 integrates g w exactly on every face
static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, ctx->k, ctx->k + 2, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal g_func(const PetscReal x[])
{
  return 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2] + 4.0 * x[0] * x[1] * x[2];
}

static PetscErrorCode w_func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  u[0] = (1.0 + x[0]) * (1.0 + x[1]) * (1.0 + x[2]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_bd(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = g_func(x);
}

static void f1_bd_grad(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_bd_identity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

// DMPlexCreateBoxMesh() face values 1..6 are z = lower, z = upper, y = lower, y = upper, x = upper, x = lower
static const PetscInt    faceAxis[6]  = {2, 2, 1, 1, 0, 0};
static const PetscBool   faceUpper[6] = {PETSC_FALSE, PETSC_TRUE, PETSC_FALSE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE};
static const char *const faceName[6]  = {"z = lower", "z = upper", "y = lower", "y = upper", "x = upper", "x = lower"};

// Integrate g w over the face where coordinate axis is fixed at value s, using a 10x10 Gauss rule
static PetscErrorCode ReferenceFaceIntegral(const PetscReal lower[], const PetscReal upper[], PetscInt axis, PetscReal s, PetscReal *val)
{
  const PetscInt nq = 10, a0 = (axis + 1) % 3, a1 = (axis + 2) % 3;
  PetscReal      qx[10], qw[10], sum              = 0.0;

  PetscFunctionBeginUser;
  *val = 0.;
  PetscCall(PetscDTGaussQuadrature(nq, -1.0, 1.0, qx, qw));
  for (PetscInt i = 0; i < nq; ++i) {
    for (PetscInt j = 0; j < nq; ++j) {
      PetscReal   x[3];
      PetscScalar w;

      x[axis] = s;
      x[a0]   = lower[a0] + 0.5 * (qx[i] + 1.0) * (upper[a0] - lower[a0]);
      x[a1]   = lower[a1] + 0.5 * (qx[j] + 1.0) * (upper[a1] - lower[a1]);
      PetscCall(w_func(3, 0.0, x, 1, &w, NULL));
      sum += qw[i] * qw[j] * g_func(x) * PetscRealPart(w);
    }
  }
  *val = sum * 0.25 * (upper[a0] - lower[a0]) * (upper[a1] - lower[a1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Integrate g against the test functions on each face, take the dot product with W, and compare with the exact integral of g w
static PetscErrorCode CheckFaceIntegrals(DM dm, DMLabel label, Vec W)
{
  PetscWeakForm wf;
  Vec           locX, locF, F;
  PetscReal     lower[3], upper[3];

  PetscFunctionBeginUser;
  PetscCall(DMGetBoundingBox(dm, lower, upper));
  PetscCall(PetscWeakFormCreate(PETSC_COMM_SELF, &wf));
  for (PetscInt v = 1; v <= 6; ++v) PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, v, 0, 0, 0, f0_bd, 0, NULL));
  PetscCall(VecDuplicate(W, &F));
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(VecZeroEntries(locX));
  for (PetscInt v = 1; v <= 6; ++v) {
    const PetscInt axis = faceAxis[v - 1];
    PetscFormKey   key;
    PetscScalar    L;
    PetscReal      ref;

    key.label = label;
    key.value = v;
    key.field = 0;
    key.part  = 0;
    PetscCall(VecZeroEntries(locF));
    PetscCall(DMPlexComputeBdResidualSingle(dm, wf, key, locX, NULL, 0.0, locF));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobal(dm, locF, ADD_VALUES, F));
    PetscCall(VecDot(F, W, &L));
    PetscCall(ReferenceFaceIntegral(lower, upper, axis, faceUpper[v - 1] == PETSC_TRUE ? upper[axis] : lower[axis], &ref));
    if (PetscAbsReal(PetscRealPart(L) - ref) < PETSC_SMALL * PetscAbsReal(ref)) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Face %" PetscInt_FMT " (%s): boundary integral is correct\n", v, faceName[v - 1]));
    else
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Face %" PetscInt_FMT " (%s): boundary integral %g differs from exact %g, relative error %.2e\n", v, faceName[v - 1], (double)PetscRealPart(L), (double)ref, (double)(PetscAbsReal(PetscRealPart(L) - ref) / PetscAbsReal(ref))));
  }
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(DMRestoreLocalVector(dm, &locF));
  PetscCall(VecDestroy(&F));
  PetscCall(PetscWeakFormDestroy(&wf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compare the boundary residual of int_face grad u . grad v dS at u = U with its Jacobian times U
static PetscErrorCode CheckBdJacobian(DM dm, DMLabel label, Vec U)
{
  const PetscInt values[6] = {1, 2, 3, 4, 5, 6};
  PetscWeakForm  wf;
  Mat            J;
  Vec            locU, locF, F, JU;
  PetscReal      norm, err;

  PetscFunctionBeginUser;
  PetscCall(PetscWeakFormCreate(PETSC_COMM_SELF, &wf));
  for (PetscInt v = 1; v <= 6; ++v) {
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, v, 0, 0, 0, NULL, 0, f1_bd_grad));
    PetscCall(PetscWeakFormSetIndexBdJacobian(wf, label, v, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_bd_identity));
  }
  PetscCall(DMGetLocalVector(dm, &locU));
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(DMGlobalToLocal(dm, U, INSERT_VALUES, locU));
  PetscCall(VecZeroEntries(locF));
  for (PetscInt v = 1; v <= 6; ++v) {
    PetscFormKey key;

    key.label = label;
    key.value = v;
    key.field = 0;
    key.part  = 0;
    PetscCall(DMPlexComputeBdResidualSingle(dm, wf, key, locU, NULL, 0.0, locF));
  }
  PetscCall(VecDuplicate(U, &F));
  PetscCall(VecDuplicate(U, &JU));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMLocalToGlobal(dm, locF, ADD_VALUES, F));
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(DMPlexComputeBdJacobianSingle(dm, wf, label, 6, values, 0, locU, NULL, 0.0, 0.0, J, J));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMult(J, U, JU));
  PetscCall(VecNorm(F, NORM_2, &norm));
  PetscCall(VecAXPY(JU, -1.0, F));
  PetscCall(VecNorm(JU, NORM_2, &err));
  if (err < PETSC_SMALL * norm) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Boundary residual matches its Jacobian\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Boundary residual differs from its Jacobian, relative error %.2e\n", (double)(err / norm)));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&JU));
  PetscCall(VecDestroy(&F));
  PetscCall(DMRestoreLocalVector(dm, &locU));
  PetscCall(DMRestoreLocalVector(dm, &locF));
  PetscCall(PetscWeakFormDestroy(&wf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM      dm;
  DMLabel label;
  Vec     W;
  AppCtx  ctx;
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {w_func};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SetupDiscretization(dm, &ctx));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCall(OrientFaces(dm, label, ctx.o));
  PetscCall(DMCreateGlobalVector(dm, &W));
  // First project w(x) into the finite element space as W. This works because it is in Q1
  PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, W));
  if (ctx.jac == PETSC_TRUE) PetscCall(CheckBdJacobian(dm, label, W));
  else PetscCall(CheckFaceIntegrals(dm, label, W));
  PetscCall(VecDestroy(&W));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_upper 1,2,1

    test:
      suffix: q1
      args: -dm_plex_box_faces 2,2,2 -k 1

    test:
      suffix: q2
      args: -dm_plex_box_faces 2,2,2 -k 2

    test:
      suffix: q2_orient
      args: -dm_plex_box_faces 2,2,2 -k 2 -face_orientation {{-4 -3 -2 -1 1 2 3}separate output}
      output_file: output/ex109_q2.out

  test:
    suffix: q2_flare_jacobian
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 1,0,0 -dm_plex_box_upper 2,2,1 -dm_plex_box_faces 1,1,1 -dm_coord_remap -dm_coord_map flare -k 2 -check_jacobian -face_orientation {{-4 -3 -2 -1 0 1 2 3}separate output}
    output_file: output/ex109_q2_flare_jacobian.out

TEST*/
