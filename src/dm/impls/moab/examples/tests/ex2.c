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
  char          input_file[PETSC_MAX_PATH_LEN];   /* Import mesh from file */
  char          output_file[PETSC_MAX_PATH_LEN];   /* Output mesh file name */
  PetscBool     write_output;                        /* Write output mesh and data to file */
  PetscInt      nfields;         /* Number of fields */
  char          *fieldnames[128]; /* Name of a defined field on the mesh */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  options->debug             = PETSC_FALSE;
  options->dim               = 2;
  options->nele              = 5;
  options->nfields           = 256;
  options->simplex           = PETSC_FALSE;
  options->write_output      = PETSC_FALSE;
  options->input_file[0]     = '\0';
  ierr = PetscStrcpy(options->output_file,"ex2.h5m");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMMOAB");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-debug", "Enable debug messages", "ex2.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Create simplices instead of tensor product elements", "ex2.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The number of elements in each dimension", "ex2.c", options->nele, &options->nele, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-meshfile", "The input mesh file", "ex2.c", options->input_file, options->input_file, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-out", "Write out the mesh and solution that is defined on it (Default H5M format)", "ex2.c", options->output_file, options->output_file, PETSC_MAX_PATH_LEN, &options->write_output);CHKERRQ(ierr);
  ierr = PetscOptionsStringArray("-fields", "The list of names of the field variables", "ex2.c", options->fieldnames,&options->nfields, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  if (options->debug) PetscPrintf(comm, "Total number of fields: %D.\n",options->nfields);
  if (!flg) { /* if no field names were given by user, assign a default */
    options->nfields = 1;
    ierr = PetscStrallocpy("TestEX2Var",&options->fieldnames[0]);CHKERRQ(ierr);
  }

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       i;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(user->input_file, &len);CHKERRQ(ierr);
  if (len) {
    if (user->debug) PetscPrintf(comm, "Loading mesh from file: %s and creating a DM object.\n",user->input_file);
    ierr = DMMoabLoadFromFile(comm, user->dim, user->input_file, "", dm);CHKERRQ(ierr);
  } else {
    if (user->debug) {
      PetscPrintf(comm, "Creating a %D-dimensional structured mesh of %Dx%Dx%D in memory and creating a DM object.\n",user->dim,user->nele,user->nele,user->nele);
      PetscPrintf(comm, "Using simplex ? %D\n", user->simplex);
    }
    ierr = DMMoabCreateBoxMesh(comm, user->dim, user->simplex, NULL, user->nele, 1, dm);CHKERRQ(ierr);
  }

  if (user->debug) {
    PetscPrintf(comm, "Setting field names to DM: \n");
    for (i=0; i<user->nfields; i++)
      PetscPrintf(comm, "\t Field{0} = %s.\n",user->fieldnames[i]);
  }
  ierr     = DMMoabSetFieldNames(*dm, user->nfields, (const char**)user->fieldnames);CHKERRQ(ierr);
  ierr     = DMSetUp(*dm);CHKERRQ(ierr);
  ierr     = PetscObjectSetName((PetscObject) *dm, "Structured Mesh");CHKERRQ(ierr);
  ierr     = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr     = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscRandom    rctx;
  Vec            solution;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &user.dm);CHKERRQ(ierr);

  if (user.debug) PetscPrintf(comm, "Creating a global vector defined on DM and setting random data.\n");
  ierr = DMCreateGlobalVector(user.dm, &solution);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetType(rctx,PETSCRAND48);CHKERRQ(ierr);
  ierr = VecSetRandom(solution,rctx);CHKERRQ(ierr);

  if (user.write_output) {
    ierr = DMMoabSetGlobalFieldVector(user.dm, solution);CHKERRQ(ierr);
    if (user.debug) PetscPrintf(comm, "Output mesh and associated field data to file: %s.\n",user.output_file);
    ierr = DMMoabOutput(user.dm,(const char*)user.output_file,"");CHKERRQ(ierr);
  }
  ierr = DMView(user.dm,0);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&solution);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
