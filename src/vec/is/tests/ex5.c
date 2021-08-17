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
  ierr = PetscOptionsString("-fname", "The output file", "ex5.c", options->fname, options->fname, sizeof(options->fname), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-includes_constraints", "Flag for if global section is to include constrained DoFs or not", "ex5.c", options->includes_constraints, &options->includes_constraints, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm        comm;
  PetscMPIInt     size, rank, mycolor;
  AppCtx          user;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  if (size < 3) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");

  /* Save */
  mycolor = (PetscMPIInt)(rank >= 2);
  ierr = MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm);CHKERRMPI(ierr);
  if (mycolor == 0) {
    PetscSection  section, gsection;
    PetscSF       sf;
    PetscInt      nroots = -1, nleaves = -1, *ilocal;
    PetscSFNode  *iremote;
    PetscViewer   viewer;

    /* Create section */
    ierr = PetscSectionCreate(comm, &section);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(section, 2);CHKERRQ(ierr);
    switch (rank) {
    case 0:
      ierr = PetscSectionSetChart(section, 0, 4);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 0, 3);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 1, 3);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 2, 5);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 3, 7);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 0, 0, 2);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 1, 0, 3);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 2, 0, 5);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 3, 0, 7);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 0, 1, 1);CHKERRQ(ierr);
      break;
    case 1:
      ierr = PetscSectionSetChart(section, 0, 3);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 0, 7);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 1, 5);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, 2, 13);CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintDof(section, 2, 1);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 0, 0, 7);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 1, 0, 5);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 2, 0, 11);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(section, 2, 1, 2);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldConstraintDof(section, 2, 0, 1);CHKERRQ(ierr);
      break;
    }
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    if (rank == 1)
    {
      const PetscInt indices[] = {7};
      const PetscInt indices0[] = {7};

      ierr = PetscSectionSetConstraintIndices(section, 2, indices);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldConstraintIndices(section, 2, 0, indices0);CHKERRQ(ierr);
    }
    /* Create sf */
    switch (rank) {
    case 0:
      nroots = 4;
      nleaves = 1;
      ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
      ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
      ilocal[0] = 3;
      iremote[0].rank = 1;
      iremote[0].index = 0;
      break;
    case 1:
      nroots = 3;
      nleaves = 1;
      ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
      ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
      ilocal[0] = 1;
      iremote[0].rank = 0;
      iremote[0].index = 2;
      break;
    }
    ierr = PetscSFCreate(comm, &sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    /* Create global section*/
    ierr = PetscSectionCreateGlobalSection(section, sf, user.includes_constraints, PETSC_FALSE, &gsection);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    /* View */
    ierr = PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    ierr = PetscSectionView(gsection, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)section, "Save: local section");CHKERRQ(ierr);
    ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)gsection, "Save: global section");CHKERRQ(ierr);
    ierr = PetscSectionView(gsection, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&gsection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_free(&comm);CHKERRMPI(ierr);

  /* Load */
  mycolor = (PetscMPIInt)(rank >= 3);
  ierr = MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm);CHKERRMPI(ierr);
  if (mycolor == 0) {
    PetscSection  section;
    PetscInt      chartSize = -1;
    PetscViewer   viewer;

    ierr = PetscSectionCreate(comm, &section);CHKERRQ(ierr);
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
    ierr = PetscSectionSetChart(section, 0, chartSize);CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
    ierr = PetscSectionLoad(section, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)section, "Load: section");CHKERRQ(ierr);
    ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_free(&comm);CHKERRMPI(ierr);

  /* Finalize */
  ierr = PetscFinalize();
  return ierr;
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
