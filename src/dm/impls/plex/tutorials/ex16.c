#include "petscsf.h"
static char help[] = "Simple demonstration of CGNS parallel load-save including data\n\n";
// As this is a tutorial that is intended to be an easy starting point feel free to make new
// example files that extend this but please keep this one simple.
// In subsequent examples we will also provide tools to generate an arbitrary size initial
// CGNS file to support performance benchmarking.

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#define EX "ex16.c"

typedef struct {
  char infile[PETSC_MAX_PATH_LEN]; /* Input mesh filename */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->infile[0] = '\0';
  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsString("-infile", "The input CGNS file", EX, options->infile, options->infile, sizeof(options->infile), NULL));
  PetscOptionsEnd();
  PetscCheck(options->infile[0], comm, PETSC_ERR_USER_INPUT, "-infile needs to be specified");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create DM from CGNS file and setup PetscFE to VecLoad solution from that file
PetscErrorCode ReadCGNSDM(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscInt degree;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateFromFile(comm, filename, "ex16_plex", PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  { // Get degree of the natural section
    PetscFE        fe_natural;
    PetscDualSpace dual_space_natural;

    PetscCall(DMGetField(*dm, 0, NULL, (PetscObject *)&fe_natural));
    PetscCall(PetscFEGetDualSpace(fe_natural, &dual_space_natural));
    PetscCall(PetscDualSpaceGetOrder(dual_space_natural, &degree));
    PetscCall(DMClearFields(*dm));
    PetscCall(DMSetLocalSection(*dm, NULL));
  }

  { // Setup fe to load in the initial condition data
    PetscFE        fe;
    PetscInt       dim, cStart, cEnd;
    PetscInt       ctInt, mincti, maxcti;
    DMPolytopeType dm_polytope, cti;

    PetscCall(DMGetDimension(*dm, &dim));
    // Limiting to single topology in this simple example
    PetscCall(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetCellType(*dm, cStart, &dm_polytope));
    for (PetscInt i = cStart + 1; i < cEnd; i++) {
      PetscCall(DMPlexGetCellType(*dm, i, &cti));
      PetscCheck(cti == dm_polytope, comm, PETSC_ERR_RETURN, "Multi-topology not yet supported in this example!");
    }
    ctInt = cti;
    PetscCallMPI(MPIU_Allreduce(&ctInt, &maxcti, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPIU_Allreduce(&ctInt, &mincti, 1, MPIU_INT, MPI_MIN, comm));
    PetscCheck(mincti == maxcti, comm, PETSC_ERR_RETURN, "Multi-topology not yet supported in this example!");
    PetscCall(PetscPrintf(comm, "Mesh confirmed to be single topology degree %" PetscInt_FMT " %s\n", degree, DMPolytopeTypes[cti]));
    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 5, dm_polytope, degree, PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "FE for VecLoad"));
    PetscCall(DMAddField(*dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(*dm));
    PetscCall(PetscFEDestroy(&fe));
  }

  // Set section component names, used when writing out CGNS files
  PetscSection section;
  PetscCall(DMGetLocalSection(*dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "Pressure"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "VelocityX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "VelocityY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "VelocityZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "Temperature"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      user;
  MPI_Comm    comm;
  const char *infilename;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  infilename = user.infile;

  DM          dm;
  Vec         V;
  PetscViewer viewer;
  const char *name;
  PetscReal   time;
  PetscBool   set;
  comm = PETSC_COMM_WORLD;

  // Load DM from CGNS file
  PetscCall(ReadCGNSDM(comm, infilename, &dm));
  PetscCall(DMSetOptionsPrefix(dm, "loaded_"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  // Load solution from CGNS file
  PetscCall(PetscViewerCGNSOpen(comm, infilename, FILE_MODE_READ, &viewer));
  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(PetscViewerCGNSSetSolutionIndex(viewer, 1));
  PetscCall(PetscViewerCGNSGetSolutionName(viewer, &name));
  PetscCall(PetscViewerCGNSGetSolutionTime(viewer, &time, &set));
  PetscCall(PetscPrintf(comm, "Solution Name: %s, and time %g\n", name, (double)time));
  PetscCall(VecLoad(V, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // Write loaded solution (e.g. example in TEST below is to CGNS file)
  PetscCall(VecViewFromOptions(V, NULL, "-vec_view"));

  PetscCall(DMRestoreGlobalVector(dm, &V));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: cgns
  testset:
    suffix: cgns
    requires: !complex
    nsize: 4
    args: -infile ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns
    args: -dm_plex_cgns_parallel -loaded_dm_view
    test:
      suffix: simple
      args: -vec_view cgns:2x2x2_Q3Vecview.cgns
TEST*/
