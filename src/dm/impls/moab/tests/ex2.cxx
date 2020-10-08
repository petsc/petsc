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
  PetscBool     interlace;
  char          input_file[PETSC_MAX_PATH_LEN];   /* Import mesh from file */
  char          output_file[PETSC_MAX_PATH_LEN];   /* Output mesh file name */
  PetscBool     write_output;                        /* Write output mesh and data to file */
  PetscInt      nfields;         /* Number of fields */
  char          *fieldnames[PETSC_MAX_PATH_LEN]; /* Name of a defined field on the mesh */
} AppCtx;

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
  options->interlace         = PETSC_FALSE;
  options->input_file[0]     = '\0';
  ierr = PetscStrcpy(options->output_file,"ex2.h5m");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMMOAB");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-debug", "Enable debug messages", "ex2.cxx", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interlace", "Use interlaced arrangement for the field data", "ex2.cxx", options->interlace, &options->interlace, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Create simplices instead of tensor product elements", "ex2.cxx", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex2.cxx", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-n", "The number of elements in each dimension", "ex2.cxx", options->nele, &options->nele, NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsString("-meshfile", "The input mesh file", "ex2.cxx", options->input_file, options->input_file, sizeof(options->input_file), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-io", "Write out the mesh and solution that is defined on it (Default H5M format)", "ex2.cxx", options->output_file, options->output_file, sizeof(options->output_file), &options->write_output);CHKERRQ(ierr);
  ierr = PetscOptionsStringArray("-fields", "The list of names of the field variables", "ex2.cxx", options->fieldnames,&options->nfields, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  if (options->debug) PetscPrintf(comm, "Total number of fields: %D.\n",options->nfields);
  if (!flg) { /* if no field names were given by user, assign a default */
    options->nfields = 1;
    ierr = PetscStrallocpy("TestEX2Var",&options->fieldnames[0]);CHKERRQ(ierr);
  }

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user)
{
  PetscInt       i;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscStrlen(user->input_file, &len);CHKERRQ(ierr);
  if (len) {
    if (user->debug) PetscPrintf(comm, "Loading mesh from file: %s and creating a DM object.\n",user->input_file);
    ierr = DMMoabLoadFromFile(comm, user->dim, 1, user->input_file, "", &user->dm);CHKERRQ(ierr);
  }
  else {
    if (user->debug) {
      PetscPrintf(comm, "Creating a %D-dimensional structured %s mesh of %Dx%Dx%D in memory and creating a DM object.\n",user->dim,(user->simplex?"simplex":"regular"),user->nele,user->nele,user->nele);
    }
    ierr = DMMoabCreateBoxMesh(comm, user->dim, user->simplex, NULL, user->nele, 1, &user->dm);CHKERRQ(ierr);
  }

  if (user->debug) {
    PetscPrintf(comm, "Setting field names to DM: \n");
    for (i=0; i<user->nfields; i++)
      PetscPrintf(comm, "\t Field{%D} = %s.\n",i,user->fieldnames[i]);
  }
  ierr     = DMMoabSetFieldNames(user->dm, user->nfields, (const char**)user->fieldnames);CHKERRQ(ierr);
  ierr     = PetscObjectSetName((PetscObject)user->dm, "Structured Mesh");CHKERRQ(ierr);
  ierr     = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscRandom    rctx;
  Vec            solution;
  Mat            system;
  MPI_Comm       comm;
  PetscInt       i;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user);CHKERRQ(ierr);

  /* set block size */
  ierr = DMMoabSetBlockSize(user.dm, (user.interlace?user.nfields:1));CHKERRQ(ierr);
  ierr = DMSetMatType(user.dm,MATAIJ);CHKERRQ(ierr);

  ierr = DMSetFromOptions(user.dm);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(user.dm);CHKERRQ(ierr);

  if (user.debug) PetscPrintf(comm, "Creating a global vector defined on DM and setting random data.\n");
  ierr = DMCreateGlobalVector(user.dm,&solution);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(solution,rctx);CHKERRQ(ierr);

  /* test if matrix allocation for the prescribed matrix type is done correctly */
  if (user.debug) PetscPrintf(comm, "Creating a global matrix defined on DM with the right block structure.\n");
  ierr = DMCreateMatrix(user.dm,&system);CHKERRQ(ierr);

  if (user.write_output) {
    ierr = DMMoabSetGlobalFieldVector(user.dm, solution);CHKERRQ(ierr);
    if (user.debug) PetscPrintf(comm, "Output mesh and associated field data to file: %s.\n",user.output_file);
    ierr = DMMoabOutput(user.dm,(const char*)user.output_file,"");CHKERRQ(ierr);
  }

  if (user.nfields) {
    for (i=0; i<user.nfields; i++) {
      ierr = PetscFree(user.fieldnames[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&solution);CHKERRQ(ierr);
  ierr = MatDestroy(&system);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: moab

   test:
     args: -debug -fields v1,v2,v3

TEST*/
