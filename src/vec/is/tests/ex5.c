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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  options->fname[0] = '\0';
  options->includes_constraints = PETSC_TRUE;
  ierr = PetscOptionsBegin(comm, "", "PetscSectionView()/Load() in HDF5 Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-fname", "The output file", "ex5.c", options->fname, options->fname, sizeof(options->fname), NULL));
  CHKERRQ(PetscOptionsBool("-includes_constraints", "Flag for if global section is to include constrained DoFs or not", "ex5.c", options->includes_constraints, &options->includes_constraints, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm        comm;
  PetscMPIInt     size, rank, mycolor;
  AppCtx          user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheckFalse(size < 3,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");

  /* Save */
  mycolor = (PetscMPIInt)(rank >= 2);
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    PetscSection  section, gsection;
    PetscSF       sf;
    PetscInt      nroots = -1, nleaves = -1, *ilocal;
    PetscSFNode  *iremote;
    PetscViewer   viewer;

    /* Create section */
    CHKERRQ(PetscSectionCreate(comm, &section));
    CHKERRQ(PetscSectionSetNumFields(section, 2));
    switch (rank) {
    case 0:
      CHKERRQ(PetscSectionSetChart(section, 0, 4));
      CHKERRQ(PetscSectionSetDof(section, 0, 3));
      CHKERRQ(PetscSectionSetDof(section, 1, 3));
      CHKERRQ(PetscSectionSetDof(section, 2, 5));
      CHKERRQ(PetscSectionSetDof(section, 3, 7));
      CHKERRQ(PetscSectionSetFieldDof(section, 0, 0, 2));
      CHKERRQ(PetscSectionSetFieldDof(section, 1, 0, 3));
      CHKERRQ(PetscSectionSetFieldDof(section, 2, 0, 5));
      CHKERRQ(PetscSectionSetFieldDof(section, 3, 0, 7));
      CHKERRQ(PetscSectionSetFieldDof(section, 0, 1, 1));
      break;
    case 1:
      CHKERRQ(PetscSectionSetChart(section, 0, 3));
      CHKERRQ(PetscSectionSetDof(section, 0, 7));
      CHKERRQ(PetscSectionSetDof(section, 1, 5));
      CHKERRQ(PetscSectionSetDof(section, 2, 13));
      CHKERRQ(PetscSectionSetConstraintDof(section, 2, 1));
      CHKERRQ(PetscSectionSetFieldDof(section, 0, 0, 7));
      CHKERRQ(PetscSectionSetFieldDof(section, 1, 0, 5));
      CHKERRQ(PetscSectionSetFieldDof(section, 2, 0, 11));
      CHKERRQ(PetscSectionSetFieldDof(section, 2, 1, 2));
      CHKERRQ(PetscSectionSetFieldConstraintDof(section, 2, 0, 1));
      break;
    }
    CHKERRQ(PetscSectionSetUp(section));
    if (rank == 1)
    {
      const PetscInt indices[] = {7};
      const PetscInt indices0[] = {7};

      CHKERRQ(PetscSectionSetConstraintIndices(section, 2, indices));
      CHKERRQ(PetscSectionSetFieldConstraintIndices(section, 2, 0, indices0));
    }
    /* Create sf */
    switch (rank) {
    case 0:
      nroots = 4;
      nleaves = 1;
      CHKERRQ(PetscMalloc1(nleaves, &ilocal));
      CHKERRQ(PetscMalloc1(nleaves, &iremote));
      ilocal[0] = 3;
      iremote[0].rank = 1;
      iremote[0].index = 0;
      break;
    case 1:
      nroots = 3;
      nleaves = 1;
      CHKERRQ(PetscMalloc1(nleaves, &ilocal));
      CHKERRQ(PetscMalloc1(nleaves, &iremote));
      ilocal[0] = 1;
      iremote[0].rank = 0;
      iremote[0].index = 2;
      break;
    }
    CHKERRQ(PetscSFCreate(comm, &sf));
    CHKERRQ(PetscSFSetGraph(sf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    /* Create global section*/
    CHKERRQ(PetscSectionCreateGlobalSection(section, sf, user.includes_constraints, PETSC_FALSE, &gsection));
    CHKERRQ(PetscSFDestroy(&sf));
    /* View */
    CHKERRQ(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer));
    CHKERRQ(PetscSectionView(gsection, viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)section, "Save: local section"));
    CHKERRQ(PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm)));
    CHKERRQ(PetscObjectSetName((PetscObject)gsection, "Save: global section"));
    CHKERRQ(PetscSectionView(gsection, PETSC_VIEWER_STDOUT_(comm)));
    CHKERRQ(PetscSectionDestroy(&gsection));
    CHKERRQ(PetscSectionDestroy(&section));
  }
  CHKERRMPI(MPI_Comm_free(&comm));

  /* Load */
  mycolor = (PetscMPIInt)(rank >= 3);
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    PetscSection  section;
    PetscInt      chartSize = -1;
    PetscViewer   viewer;

    CHKERRQ(PetscSectionCreate(comm, &section));
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
    CHKERRQ(PetscSectionSetChart(section, 0, chartSize));
    CHKERRQ(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer));
    CHKERRQ(PetscSectionLoad(section, viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)section, "Load: section"));
    CHKERRQ(PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm)));
    CHKERRQ(PetscSectionDestroy(&section));
  }
  CHKERRMPI(MPI_Comm_free(&comm));

  /* Finalize */
  CHKERRQ(PetscFinalize());
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
