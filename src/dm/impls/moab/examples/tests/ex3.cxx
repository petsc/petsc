static char help[] = "Create a box mesh with DMMoab and test defining a tag on the mesh\n\n";

#include <petscdmmoab.h>

typedef struct {
  DM            dm;                /* DM implementation using the MOAB interface */
  PetscBool     debug;             /* The debugging level */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt      dim;                            /* The topological mesh dimension */
  PetscInt      nele;                           /* Elements in each dimension */
  PetscBool     simplex;                        /* Use simplex elements */
  PetscInt      nlevels;                        /* Number of levels in mesh hierarchy */
  PetscInt      nghost;                        /* Number of ghost layers in the mesh */
  char          input_file[PETSC_MAX_PATH_LEN];   /* Import mesh from file */
  char          output_file[PETSC_MAX_PATH_LEN];   /* Output mesh file name */
  PetscBool     write_output;                        /* Write output mesh and data to file */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug             = PETSC_FALSE;
  options->nlevels           = 1;
  options->nghost            = 1;
  options->dim               = 2;
  options->nele              = 5;
  options->simplex           = PETSC_FALSE;
  options->write_output      = PETSC_FALSE;
  options->input_file[0]     = '\0';
  ierr = PetscStrcpy(options->output_file,"ex3.h5m");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Uniform Mesh Refinement Options", "DMMOAB");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-debug", "Enable debug messages", "ex2.cxx", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex3.cxx", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The number of elements in each dimension", "ex3.cxx", options->nele, &options->nele, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-levels", "Number of levels in the hierarchy", "ex3.cxx", options->nlevels, &options->nlevels, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ghost", "Number of ghost layers in the mesh", "ex3.cxx", options->nghost, &options->nghost, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Create simplices instead of tensor product elements", "ex3.cxx", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-input", "The input mesh file", "ex3.cxx", options->input_file, options->input_file, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-io", "Write out the mesh and solution that is defined on it (Default H5M format)", "ex3.cxx", options->output_file, options->output_file, PETSC_MAX_PATH_LEN, &options->write_output);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user)
{
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(user->input_file, &len);CHKERRQ(ierr);
  if (len) {
    if (user->debug) PetscPrintf(comm, "Loading mesh from file: %s and creating the coarse level DM object.\n",user->input_file);
    ierr = DMMoabLoadFromFile(comm, user->dim, user->nghost, user->input_file, "", &user->dm);CHKERRQ(ierr);
  }
  else {
    if (user->debug) {
      PetscPrintf(comm, "Creating a %D-dimensional structured %s mesh of %Dx%Dx%D in memory and creating a DM object.\n",user->dim,(user->simplex?"simplex":"regular"),user->nele,user->nele,user->nele);
    }
    ierr = DMMoabCreateBoxMesh(comm, user->dim, user->simplex, NULL, user->nele, user->nghost, &user->dm);CHKERRQ(ierr);
  }

  ierr     = PetscObjectSetName((PetscObject)user->dm, "Coarse Mesh");CHKERRQ(ierr);
  ierr     = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  MPI_Comm       comm;
  PetscInt       i;
  Mat            R;
  DM            *dmhierarchy;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user);CHKERRQ(ierr);

  ierr = DMSetFromOptions(user.dm);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(user.dm);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(DM)*(user.nlevels+1),&dmhierarchy);
  for (i=0; i<=user.nlevels; i++) dmhierarchy[i] = NULL;

  // coarsest grid = 0
  // finest grid = nlevels
  dmhierarchy[0] = user.dm;
  PetscObjectReference((PetscObject)user.dm);

  if (user.nlevels) {
    if (user.debug) PetscPrintf(comm, "Generate the MOAB mesh hierarchy with %D levels.\n", user.nlevels);
    ierr = DMMoabGenerateHierarchy(user.dm,user.nlevels,PETSC_NULL);CHKERRQ(ierr);

    PetscBool usehierarchy=PETSC_FALSE;
    if (usehierarchy) {
      ierr = DMRefineHierarchy(user.dm,user.nlevels,&dmhierarchy[1]);CHKERRQ(ierr);
    }
    else {
      if (user.debug) {
        PetscPrintf(PETSC_COMM_WORLD, "Level %D\n", 0);
        ierr = DMView(user.dm, 0);CHKERRQ(ierr);
      }
      for (i=1; i<=user.nlevels; i++) {
        if (user.debug) PetscPrintf(PETSC_COMM_WORLD, "Level %D\n", i);
        ierr = DMRefine(dmhierarchy[i-1],MPI_COMM_NULL,&dmhierarchy[i]);CHKERRQ(ierr);
        ierr = DMCreateInterpolation(dmhierarchy[i-1],dmhierarchy[i],&R,NULL);CHKERRQ(ierr);
        if (user.debug) {
          ierr = DMView(dmhierarchy[i], 0);CHKERRQ(ierr);
          ierr = MatView(R,0);CHKERRQ(ierr);
        }
        /* Solvers could now set operator "R" to the multigrid PC object for level i 
            PCMGSetInterpolation(pc,i,R)
        */
        ierr = MatDestroy(&R);CHKERRQ(ierr);
      }
    }
  }

  if (user.write_output) {
    if (user.debug) PetscPrintf(comm, "Output mesh hierarchy to file: %s.\n",user.output_file);
    ierr = DMMoabOutput(dmhierarchy[user.nlevels],(const char*)user.output_file,"");CHKERRQ(ierr);
  }

  for (i=0; i<=user.nlevels; i++) {
    ierr = DMDestroy(&dmhierarchy[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dmhierarchy);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
