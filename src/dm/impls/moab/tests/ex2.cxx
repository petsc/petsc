static char help[] = "Create a box mesh with DMMoab and test defining a tag on the mesh\n\n";

#include <petscdmmoab.h>

typedef struct {
  DM            dm;    /* DM implementation using the MOAB interface */
  PetscBool     debug; /* The debugging level */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt  dim;     /* The topological mesh dimension */
  PetscInt  nele;    /* Elements in each dimension */
  PetscBool simplex; /* Use simplex elements */
  PetscBool interlace;
  char      input_file[PETSC_MAX_PATH_LEN];  /* Import mesh from file */
  char      output_file[PETSC_MAX_PATH_LEN]; /* Output mesh file name */
  PetscBool write_output;                    /* Write output mesh and data to file */
  PetscInt  nfields;                         /* Number of fields */
  char     *fieldnames[PETSC_MAX_PATH_LEN];  /* Name of a defined field on the mesh */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool flg;

  PetscFunctionBegin;
  options->debug         = PETSC_FALSE;
  options->dim           = 2;
  options->nele          = 5;
  options->nfields       = 256;
  options->simplex       = PETSC_FALSE;
  options->write_output  = PETSC_FALSE;
  options->interlace     = PETSC_FALSE;
  options->input_file[0] = '\0';
  PetscCall(PetscStrncpy(options->output_file, "ex2.h5m", sizeof(options->output_file)));

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMMOAB");
  PetscCall(PetscOptionsBool("-debug", "Enable debug messages", "ex2.cxx", options->debug, &options->debug, NULL));
  PetscCall(PetscOptionsBool("-interlace", "Use interlaced arrangement for the field data", "ex2.cxx", options->interlace, &options->interlace, NULL));
  PetscCall(PetscOptionsBool("-simplex", "Create simplices instead of tensor product elements", "ex2.cxx", options->simplex, &options->simplex, NULL));
  PetscCall(PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex2.cxx", options->dim, &options->dim, NULL, 1, 3));
  PetscCall(PetscOptionsBoundedInt("-n", "The number of elements in each dimension", "ex2.cxx", options->nele, &options->nele, NULL, 1));
  PetscCall(PetscOptionsString("-meshfile", "The input mesh file", "ex2.cxx", options->input_file, options->input_file, sizeof(options->input_file), NULL));
  PetscCall(PetscOptionsString("-io", "Write out the mesh and solution that is defined on it (Default H5M format)", "ex2.cxx", options->output_file, options->output_file, sizeof(options->output_file), &options->write_output));
  PetscCall(PetscOptionsStringArray("-fields", "The list of names of the field variables", "ex2.cxx", options->fieldnames, &options->nfields, &flg));
  PetscOptionsEnd();

  if (options->debug) PetscCall(PetscPrintf(comm, "Total number of fields: %" PetscInt_FMT ".\n", options->nfields));
  if (!flg) { /* if no field names were given by user, assign a default */
    options->nfields = 1;
    PetscCall(PetscStrallocpy("TestEX2Var", &options->fieldnames[0]));
  }

  PetscCall(PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user)
{
  PetscInt    i;
  size_t      len;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(user->createMeshEvent, 0, 0, 0, 0));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscStrlen(user->input_file, &len));
  if (len) {
    if (user->debug) PetscCall(PetscPrintf(comm, "Loading mesh from file: %s and creating a DM object.\n", user->input_file));
    PetscCall(DMMoabLoadFromFile(comm, user->dim, 1, user->input_file, "", &user->dm));
  } else {
    if (user->debug) {
      PetscCall(PetscPrintf(comm, "Creating a %" PetscInt_FMT "-dimensional structured %s mesh of %" PetscInt_FMT "x%" PetscInt_FMT "x%" PetscInt_FMT " in memory and creating a DM object.\n", user->dim, (user->simplex ? "simplex" : "regular"), user->nele,
                            user->nele, user->nele));
    }
    PetscCall(DMMoabCreateBoxMesh(comm, user->dim, user->simplex, NULL, user->nele, 1, &user->dm));
  }

  if (user->debug) {
    PetscCall(PetscPrintf(comm, "Setting field names to DM: \n"));
    for (i = 0; i < user->nfields; i++) PetscCall(PetscPrintf(comm, "\t Field{%" PetscInt_FMT "} = %s.\n", i, user->fieldnames[i]));
  }
  PetscCall(DMMoabSetFieldNames(user->dm, user->nfields, (const char **)user->fieldnames));
  PetscCall(PetscObjectSetName((PetscObject)user->dm, "Structured Mesh"));
  PetscCall(PetscLogEventEnd(user->createMeshEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      user; /* user-defined work context */
  PetscRandom rctx;
  Vec         solution;
  Mat         system;
  MPI_Comm    comm;
  PetscInt    i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &user));

  /* set block size */
  PetscCall(DMMoabSetBlockSize(user.dm, (user.interlace ? user.nfields : 1)));
  PetscCall(DMSetMatType(user.dm, MATAIJ));

  PetscCall(DMSetFromOptions(user.dm));

  /* SetUp the data structures for DMMOAB */
  PetscCall(DMSetUp(user.dm));

  if (user.debug) PetscCall(PetscPrintf(comm, "Creating a global vector defined on DM and setting random data.\n"));
  PetscCall(DMCreateGlobalVector(user.dm, &solution));
  PetscCall(PetscRandomCreate(comm, &rctx));
  PetscCall(VecSetRandom(solution, rctx));

  /* test if matrix allocation for the prescribed matrix type is done correctly */
  if (user.debug) PetscCall(PetscPrintf(comm, "Creating a global matrix defined on DM with the right block structure.\n"));
  PetscCall(DMCreateMatrix(user.dm, &system));

  if (user.write_output) {
    PetscCall(DMMoabSetGlobalFieldVector(user.dm, solution));
    if (user.debug) PetscCall(PetscPrintf(comm, "Output mesh and associated field data to file: %s.\n", user.output_file));
    PetscCall(DMMoabOutput(user.dm, (const char *)user.output_file, ""));
  }

  if (user.nfields) {
    for (i = 0; i < user.nfields; i++) PetscCall(PetscFree(user.fieldnames[i]));
  }
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDestroy(&solution));
  PetscCall(MatDestroy(&system));
  PetscCall(DMDestroy(&user.dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: moab

   test:
     args: -debug -fields v1,v2,v3

TEST*/
