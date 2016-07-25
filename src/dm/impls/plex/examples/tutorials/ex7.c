static char help[] = "Create a Plex sphere from quads and use a DMDA ordering in each quad\n\n";

#include <petscdmplex.h>

#undef __FUNCT__
#define __FUNCT__ "ProjectToSphere"
static PetscErrorCode ProjectToSphere(DM dm)
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

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
static PetscErrorCode SetupSection(DM dm)
{
  PetscSection   s;
  PetscInt       vStart, vEnd, v, dim, d;
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
  ierr = DMSetDefaultSection(dm, s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = DMPlexCreateQuadSphereMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = ProjectToSphere(dm);CHKERRQ(ierr);
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
