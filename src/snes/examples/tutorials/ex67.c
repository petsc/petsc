static char help[] = "Simple test for using advanced discretizations with DMDA\n\n\n";

#include <petscdmda.h>
#include <petscsnes.h>

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscInt      debug;             /* The debugging level */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;

  ierr = PetscOptionsBegin(comm, "", "DMDA Test Problem Options", "DMDA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(dim) {
  case 2:
    ierr = DMDACreate2d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, PETSC_DECIDE, PETSC_DECIDE, 2, 1, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, -4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 2, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not create mesh for dimension %d", dim);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMDASetFieldName(*dm, 0, "velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(*dm, 1, "pressure");CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user) {
  PetscInt       dim             = user->dim;
  PetscInt       numComp[2]      = {dim, 1};
  PetscInt       numVertexDof[2] = {0, 0};
  PetscInt       numCellDof[2]   = {0, 1};
  PetscInt       numFaceDof[6];
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(d = 0; d < dim; ++d) {
    numFaceDof[0*dim + d] = 1;
    numFaceDof[1*dim + d] = 0;
  }
  ierr = DMDACreateSection(dm, numComp, numVertexDof, numFaceDof, numCellDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  Vec            u,r;                  /* solution, residual vectors */
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  ierr = SetupSection(user.dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = VecView(u, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
