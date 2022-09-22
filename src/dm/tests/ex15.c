
static char help[] = "Tests DMDA interpolation.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscInt       M1 = 3, M2, dof = 1, s = 1, ratio = 2, dim = 1;
  DM             da_c, da_f;
  Vec            v_c, v_f;
  Mat            Interp;
  PetscScalar    one = 1.0;
  PetscBool      pt;
  DMBoundaryType bx = DM_BOUNDARY_NONE, by = DM_BOUNDARY_NONE, bz = DM_BOUNDARY_NONE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M1, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_width", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ratio", &ratio, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof", &dof, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-periodic", (PetscBool *)&pt, NULL));

  if (pt) {
    if (dim > 0) bx = DM_BOUNDARY_PERIODIC;
    if (dim > 1) by = DM_BOUNDARY_PERIODIC;
    if (dim > 2) bz = DM_BOUNDARY_PERIODIC;
  }
  if (bx == DM_BOUNDARY_NONE) {
    M2 = ratio * (M1 - 1) + 1;
  } else {
    M2 = ratio * M1;
  }

  /* Set up the array */
  if (dim == 1) {
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, bx, M1, dof, s, NULL, &da_c));
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, bx, M2, dof, s, NULL, &da_f));
  } else if (dim == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bx, by, DMDA_STENCIL_BOX, M1, M1, PETSC_DECIDE, PETSC_DECIDE, dof, s, NULL, NULL, &da_c));
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bx, by, DMDA_STENCIL_BOX, M2, M2, PETSC_DECIDE, PETSC_DECIDE, dof, s, NULL, NULL, &da_f));
  } else if (dim == 3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX, M1, M1, M1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, s, NULL, NULL, NULL, &da_c));
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX, M2, M2, M2, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, s, NULL, NULL, NULL, &da_f));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "dim must be 1,2, or 3");
  PetscCall(DMSetFromOptions(da_c));
  PetscCall(DMSetUp(da_c));
  PetscCall(DMSetFromOptions(da_f));
  PetscCall(DMSetUp(da_f));

  PetscCall(DMCreateGlobalVector(da_c, &v_c));
  PetscCall(DMCreateGlobalVector(da_f, &v_f));

  PetscCall(VecSet(v_c, one));
  PetscCall(DMCreateInterpolation(da_c, da_f, &Interp, NULL));
  PetscCall(MatMult(Interp, v_c, v_f));
  PetscCall(VecView(v_f, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatMultTranspose(Interp, v_f, v_c));
  PetscCall(VecView(v_c, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&Interp));
  PetscCall(VecDestroy(&v_c));
  PetscCall(DMDestroy(&da_c));
  PetscCall(VecDestroy(&v_f));
  PetscCall(DMDestroy(&da_f));
  PetscCall(PetscFinalize());
  return 0;
}
