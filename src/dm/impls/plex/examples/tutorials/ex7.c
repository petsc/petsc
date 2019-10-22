static char help[] = "Create a Plex sphere from quads and create a P1 section\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;     /* Topological problem dimension */
  PetscBool simplex; /* Mesh with simplices */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Sphere Mesh Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "Problem dimension", "ex7.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices, or tensor product cells", "ex7.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode ProjectToUnitSphere(DM dm)
{
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       Nv, v, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &Nv);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &dim);CHKERRQ(ierr);
  Nv  /= dim;
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < Nv; ++v) {
    PetscReal r = 0.0;

    for (d = 0; d < dim; ++d) r += PetscSqr(PetscRealPart(coords[v*dim+d]));
    r = PetscSqrtReal(r);
    for (d = 0; d < dim; ++d) coords[v*dim+d] /= r;
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupSection(DM dm)
{
  PetscSection   s;
  PetscInt       vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &s);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(s, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(s, 0, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(s, v, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(s, v, 0, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, ctx.dim, ctx.simplex, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Sphere");CHKERRQ(ierr);
  /* Distribute mesh over processes */
  {
     DM dmDist = NULL;
     PetscPartitioner part;
     ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
     ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
     ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
     if (dmDist) {
       ierr = DMDestroy(&dm);CHKERRQ(ierr);
       dm  = dmDist;
     }
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = ProjectToUnitSphere(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = SetupSection(dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 2);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_quad
    requires: !__float128
    args: -dm_view

  test:
    suffix: 2d_quad_parallel
    requires: !__float128
    args: -dm_view -petscpartitioner_type simple
    nsize: 2

  test:
    suffix: 2d_tri
    requires: !__float128
    args: -simplex -dm_view

  test:
    suffix: 2d_tri_parallel
    requires: !__float128
    args: -simplex -dm_view -petscpartitioner_type simple
    nsize: 2

  test:
    suffix: 3d_tri
    requires: !__float128
    args: -dim 3 -simplex -dm_view

  test:
    suffix: 3d_tri_parallel
    requires: !__float128
    args: -dim 3 -simplex -dm_view -petscpartitioner_type simple
    nsize: 2

TEST*/
