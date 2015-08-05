static char help[] = "Simple MOAB example\n\n";

#include <petscdmmoab.h>

typedef struct {
  DM            dm;                /* DM implementation using the MOAB interface */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  int dim, nlevels;
  char filename[PETSC_MAX_PATH_LEN];
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscStrcpy(options->filename, "");CHKERRQ(ierr);
  options->dim = 3;
  options->nlevels = 0;
  ierr = PetscOptionsBegin(comm, "", "MOAB example options", "DMMOAB");CHKERRQ(ierr);
  ierr = PetscOptionsString("-file", "The file containing the mesh", "ex4.cxx", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.cxx", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-levels", "Number of levels in the hierarchy", "ex4.cxx", options->nlevels, &options->nlevels, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  moab::Interface *iface=NULL;
  moab::Range range;
  moab::ErrorCode merr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);

  // load file and get entities of requested or max dimension
  if (strlen(user->filename) > 0) {
    PetscPrintf(comm, "Loading mesh from file: %s and creating a DM object.\n",user->filename);
    ierr = DMMoabLoadFromFile(comm, user->dim, 1, user->filename, "", &user->dm);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Please specify a mesh file.\n");
  }
  
  ierr = DMMoabGetInterface(*dm, &iface);CHKERRQ(ierr);
  if (-1 == user->dim) {
    moab::Range tmp_range;
    merr = iface->get_entities_by_handle(0, tmp_range);MBERRNM(merr);
    if (tmp_range.empty()) {
      MBERRNM(moab::MB_FAILURE);
    }
    user->dim = iface->dimension_from_handle(*tmp_range.rbegin());
  }
  merr = iface->get_entities_by_dimension(0, user->dim, range);MBERRNM(merr);
  ierr = DMMoabSetLocalVertices(*dm, &range);CHKERRQ(ierr);

  ierr = DMSetUp(*dm);CHKERRQ(ierr);

  // create the DMMoab object and initialize its data
  ierr = PetscObjectSetName((PetscObject) *dm, "MOAB mesh");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);

  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr); /* create the MOAB dm and the mesh */

  if (user.nlevels) { /* If user requested a refined mesh, generate nlevels of UMR hierarchy */
    ierr = DMMoabGenerateHierarchy(user.dm,user.nlevels,NULL);CHKERRQ(ierr);
  }

  std::cout << "Writing out file to: ex4.h5m." << std::endl;
  ierr = DMMoabOutput(user.dm, "ex4.h5m", NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  std::cout << "Destroyed DMMoab." << std::endl;
  ierr = PetscFinalize();
  return 0;
}

