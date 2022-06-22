static char help[] = "Tests PetscSectionView()/Load() with HDF5.\n\n";

#include <petscdmshell.h>
#include <petscdmplex.h>
#include <petscsection.h>
#include <petscsf.h>
#include <petsclayouthdf5.h>

/* Save/Load abstract sections

=====================
 Save on 2 processes
=====================

section:
                         0   1   2   3
  rank 0: Dof (Field 0)  2   3   5   7
          Dof (Field 1)  1   0   0   0

                         0   1   2
  rank 1: Dof (Field 0)  7   5  11 <- DoF 7 is constrained
          Dof (Field 1)  0   0   2

sf:
  [0] 3 <- (1, 0)
  [1] 1 <- (0, 2)

global section (includesConstraints = PETSC_FALSE):
                         0   1   2   3
  rank 0: Dof (Field 0)  2   3   5  -8
          Off (Field 0)  0   3   6  -12
          Dof (Field 1)  1   0   0  -1
          Off (Field 1)  2   6  11  -19

                         0   1   2
  rank 1: Dof (Field 0)  7  -6  11
          Off (Field 0) 11  -7  18
          Dof (Field 1)  0  -1   2
          Off (Field 1) 18 -12  28

global section (includesConstraints = PETSC_TRUE):
                         0   1   2   3
  rank 0: Dof (Field 0)  2   3   5  -8
          Off (Field 0)  0   3   6  -12
          Dof (Field 1)  1   0   0  -1
          Off (Field 1)  2   6  11  -19

                         0   1   2
  rank 1: Dof (Field 0)  7  -6  11
          Off (Field 0) 11  -7  18
          Dof (Field 1)  0  -1   2
          Off (Field 1) 18 -12  29

=====================
 Load on 3 Processes
=====================

(Set chartSize = 4, 0, 1 for rank 0, 1, 2, respectively)

global section (includesConstraints = PETSC_FALSE):

  rank 0: Dof (Field 0)  2   3   5   7
          Off (Field 0)  0   3   6  11
          Dof (Field 1)  1   0   0   0
          Off (Field 1)  2   6  11  18

  rank 1: Dof (Field 0)
          Dof (Field 1)

  rank 2: Dof (Field 0) 11
          Off (Field 0) 18
          Dof (Field 1)  2
          Off (Field 1) 28

global section (includesConstraints = PETSC_TRUE):

  rank 0: Dof (Field 0)  2   3   5   7
          Off (Field 0)  0   3   6  11
          Dof (Field 1)  1   0   0   0
          Off (Field 1)  2   6  11  18

  rank 1: Dof (Field 0)
          Dof (Field 1)

  rank 2: Dof (Field 0) 11
          Off (Field 0) 18
          Dof (Field 1)  2
          Off (Field 1) 29
*/

typedef struct {
  char      fname[PETSC_MAX_PATH_LEN]; /* Output mesh filename */
  PetscBool includes_constraints;      /* Flag for if global section is to include constrained DoFs or not */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->fname[0] = '\0';
  options->includes_constraints = PETSC_TRUE;
  PetscOptionsBegin(comm, "", "PetscSectionView()/Load() in HDF5 Test Options", "DMPLEX");
  PetscCall(PetscOptionsString("-fname", "The output file", "ex5.c", options->fname, options->fname, sizeof(options->fname), NULL));
  PetscCall(PetscOptionsBool("-includes_constraints", "Flag for if global section is to include constrained DoFs or not", "ex5.c", options->includes_constraints, &options->includes_constraints, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm        comm;
  PetscMPIInt     size, rank, mycolor;
  AppCtx          user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size >= 3,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");

  /* Save */
  mycolor = (PetscMPIInt)(rank >= 2);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    PetscSection  section, gsection;
    PetscSF       sf;
    PetscInt      nroots = -1, nleaves = -1, *ilocal;
    PetscSFNode  *iremote;
    PetscViewer   viewer;

    /* Create section */
    PetscCall(PetscSectionCreate(comm, &section));
    PetscCall(PetscSectionSetNumFields(section, 2));
    switch (rank) {
    case 0:
      PetscCall(PetscSectionSetChart(section, 0, 4));
      PetscCall(PetscSectionSetDof(section, 0, 3));
      PetscCall(PetscSectionSetDof(section, 1, 3));
      PetscCall(PetscSectionSetDof(section, 2, 5));
      PetscCall(PetscSectionSetDof(section, 3, 7));
      PetscCall(PetscSectionSetFieldDof(section, 0, 0, 2));
      PetscCall(PetscSectionSetFieldDof(section, 1, 0, 3));
      PetscCall(PetscSectionSetFieldDof(section, 2, 0, 5));
      PetscCall(PetscSectionSetFieldDof(section, 3, 0, 7));
      PetscCall(PetscSectionSetFieldDof(section, 0, 1, 1));
      break;
    case 1:
      PetscCall(PetscSectionSetChart(section, 0, 3));
      PetscCall(PetscSectionSetDof(section, 0, 7));
      PetscCall(PetscSectionSetDof(section, 1, 5));
      PetscCall(PetscSectionSetDof(section, 2, 13));
      PetscCall(PetscSectionSetConstraintDof(section, 2, 1));
      PetscCall(PetscSectionSetFieldDof(section, 0, 0, 7));
      PetscCall(PetscSectionSetFieldDof(section, 1, 0, 5));
      PetscCall(PetscSectionSetFieldDof(section, 2, 0, 11));
      PetscCall(PetscSectionSetFieldDof(section, 2, 1, 2));
      PetscCall(PetscSectionSetFieldConstraintDof(section, 2, 0, 1));
      break;
    }
    PetscCall(PetscSectionSetUp(section));
    if (rank == 1)
    {
      const PetscInt indices[] = {7};
      const PetscInt indices0[] = {7};

      PetscCall(PetscSectionSetConstraintIndices(section, 2, indices));
      PetscCall(PetscSectionSetFieldConstraintIndices(section, 2, 0, indices0));
    }
    /* Create sf */
    switch (rank) {
    case 0:
      nroots = 4;
      nleaves = 1;
      PetscCall(PetscMalloc1(nleaves, &ilocal));
      PetscCall(PetscMalloc1(nleaves, &iremote));
      ilocal[0] = 3;
      iremote[0].rank = 1;
      iremote[0].index = 0;
      break;
    case 1:
      nroots = 3;
      nleaves = 1;
      PetscCall(PetscMalloc1(nleaves, &ilocal));
      PetscCall(PetscMalloc1(nleaves, &iremote));
      ilocal[0] = 1;
      iremote[0].rank = 0;
      iremote[0].index = 2;
      break;
    }
    PetscCall(PetscSFCreate(comm, &sf));
    PetscCall(PetscSFSetGraph(sf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    /* Create global section*/
    PetscCall(PetscSectionCreateGlobalSection(section, sf, user.includes_constraints, PETSC_FALSE, &gsection));
    PetscCall(PetscSFDestroy(&sf));
    /* View */
    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscSectionView(gsection, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscObjectSetName((PetscObject)section, "Save: local section"));
    PetscCall(PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscObjectSetName((PetscObject)gsection, "Save: global section"));
    PetscCall(PetscSectionView(gsection, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscSectionDestroy(&gsection));
    PetscCall(PetscSectionDestroy(&section));
  }
  PetscCallMPI(MPI_Comm_free(&comm));

  /* Load */
  mycolor = (PetscMPIInt)(rank >= 3);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    PetscSection  section;
    PetscInt      chartSize = -1;
    PetscViewer   viewer;

    PetscCall(PetscSectionCreate(comm, &section));
    switch (rank) {
    case 0:
      chartSize = 4;
      break;
    case 1:
      chartSize = 0;
      break;
    case 2:
      chartSize = 1;
      break;
    }
    PetscCall(PetscSectionSetChart(section, 0, chartSize));
    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer));
    PetscCall(PetscSectionLoad(section, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscObjectSetName((PetscObject)section, "Load: section"));
    PetscCall(PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscSectionDestroy(&section));
  }
  PetscCallMPI(MPI_Comm_free(&comm));

  /* Finalize */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hdf5
    requires: !complex
  testset:
    nsize: 4
    test:
      suffix: 0
      args: -fname ex5_dump.h5 -includes_constraints 0
    test:
      suffix: 1
      args: -fname ex5_dump.h5 -includes_constraints 1

TEST*/
