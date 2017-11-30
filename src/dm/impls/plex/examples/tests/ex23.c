static char help[] = "Test for function and field projection\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscInt  dim;         /* The topological mesh dimension */
  PetscBool cellSimplex; /* Flag for simplices */
} AppCtx;

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = (x[0] + x[1])*Nc + c;
  return 0;
}

static void linear_vector(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f[d] = u[uOff[0]+d];
}

static void linear_scalar(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  f[0] = u[uOff[1]];
}

static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex23.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cellSimplex", "Flag for simplices", "ex23.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim      = user->dim;
  PetscBool      simplex  = user->cellSimplex;
  PetscInt       cells[3] = {1, 1, 1};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateBoxMesh(comm, dim, simplex, cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  {
    DM ddm = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &ddm);CHKERRQ(ierr);
    if (ddm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = ddm;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  const PetscInt  dim     = user->dim;
  const PetscBool simplex = user->cellSimplex;
  PetscFE         fe;
  PetscDS         prob;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, dim, simplex, "velocity_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "velocity");CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "pressure_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "pressure");CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFunctionProjection(DM dm, AppCtx *user)
{
  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  Vec               x, lx;
  PetscInt          Nf, f;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf, &funcs);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) funcs[f] = linear;
  ierr = DMGetGlobalVector(dm, &x);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, x);CHKERRQ(ierr);
  ierr = VecViewFromOptions(x, NULL, "-func_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &x);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lx);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_VALUES, lx);CHKERRQ(ierr);
  ierr = VecViewFromOptions(lx, NULL, "-local_func_view");CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lx);CHKERRQ(ierr);
  ierr = PetscFree(funcs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldProjection(DM dm, AppCtx *user)
{
  PetscErrorCode (**afuncs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  void           (**funcs)(PetscInt, PetscInt, PetscInt,
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  Vec               lx, la;
  PetscInt          Nf, f;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf, &funcs, Nf, &afuncs);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) afuncs[f]  = linear;
  funcs[0] = linear_vector;
  funcs[1] = linear_scalar;
  ierr = DMGetLocalVector(dm, &la);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, 0.0, afuncs, NULL, INSERT_VALUES, la);CHKERRQ(ierr);
  ierr = VecViewFromOptions(la, NULL, "-local_input_view");CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lx);CHKERRQ(ierr);
  ierr = DMProjectFieldLocal(dm, 0.0, la, funcs, INSERT_VALUES, lx);CHKERRQ(ierr);
  ierr = VecViewFromOptions(lx, NULL, "-local_field_view");CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &la);CHKERRQ(ierr);
  ierr = PetscFree2(funcs, afuncs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(&user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = TestFunctionProjection(dm, &user);CHKERRQ(ierr);
  ierr = TestFieldProjection(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dim 2 -func_view -local_func_view -local_input_view -local_field_view
  test:
    suffix: 1
    requires: triangle
    args: -dim 2 -velocity_petscspace_order 1 -velocity_petscfe_default_quadrature_order 2 -pressure_petscspace_order 2 -pressure_petscfe_default_quadrature_order 2 -func_view -local_func_view -local_input_view -local_field_view

TEST*/
