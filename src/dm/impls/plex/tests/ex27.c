static char help[] = "Test section ordering for FEM discretizations\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim      = 1;
  options->faces[0] = 1;
  options->faces[1] = 1;
  options->faces[2] = 1;
  options->simplex  = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Plex ex27", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex27.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise tensor product cells", "ex27.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  dim  = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex27.c", options->faces, &dim, NULL);CHKERRQ(ierr);
  if (dim) options->dim = dim;
  ierr = PetscOptionsEnd();

  if (options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %d must in [0, 4)", options->dim);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *options, DM *dm)
{
  PetscInt       dim     = options->dim;
  PetscInt      *faces   = options->faces;
  PetscBool      simplex = options->simplex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Serial Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestLocalDofOrder(DM dm, AppCtx *ctx)
{
  PetscFE        fe[3];
  PetscSection   s;
  PetscInt       dim, Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, dim, ctx->simplex, "field0_", -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1,   ctx->simplex, "field1_", -1, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1,   ctx->simplex, "field2_", -1, &fe[2]);CHKERRQ(ierr);

  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 2, NULL, (PetscObject) fe[2]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) s, NULL, "-dof_view");CHKERRQ(ierr);

  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&fe[f]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &ctx, &dm);CHKERRQ(ierr);
  ierr = TestLocalDofOrder(dm, &ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: tri_pm
    requires: triangle
    args: -dim 2 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -dm_view -dof_view

  test:
    suffix: quad_pm
    requires:
    args: -dim 2 -simplex 0 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -dm_view -dof_view

  test:
    suffix: tri_fm
    requires: triangle
    args: -dim 2 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -petscsection_point_major 0 -dm_view -dof_view

  test:
    suffix: quad_fm
    requires:
    args: -dim 2 -simplex 0 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -petscsection_point_major 0 -dm_view -dof_view

TEST*/
