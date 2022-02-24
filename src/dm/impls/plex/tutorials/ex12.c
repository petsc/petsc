static char help[] = "Tests save/load of plex/section/vec in HDF5.\n\n";

#include <petscdmshell.h>
#include <petscdmplex.h>
#include <petscsection.h>
#include <petscsf.h>
#include <petsclayouthdf5.h>

/* A six-element mesh

=====================
 Save on 2 processes
=====================

exampleDMPlex: Local numbering:

             7---17--8---18--9--19--(12)(24)(13)
             |       |       |       |       |
  rank 0:   20   0  21   1  22   2  (25) (3)(26)
             |       |       |       |       |
             4---14--5---15--6--16--(10)(23)(11)

                           (13)(25)--8--17---9--18--10--19--11
                             |       |       |       |       |
  rank 1:                  (26) (3) 20   0   21  1  22   2  23
                             |       |       |       |       |
                           (12)(24)--4--14---5--15---6--16---7

exampleDMPlex: globalPointNumbering:

             9--23--10--24--11--25--16--32--17--33--18--34--19
             |       |       |       |       |       |       |
            26   0  27   1  28   2  35   3  36   4  37   5  38
             |       |       |       |       |       |       |
             6--20---7--21---8--22--12--29--13--30--14--31--15

exampleSectionDM:
  - includesConstraints = TRUE for local section (default)
  - includesConstraints = FALSE for global section (default)

exampleSectionDM: Dofs (Field 0):

             0---0---0---0---0---0---2---0---0---0---0---0---0
             |       |       |       |       |       |       |
             0   0   0   0   0   0   0   2   0   0   0   0   0
             |       |       |       |       |       |       |
             0---0---0---0---0---0---0---0---0---0---0---0---0

exampleSectionDM: Dofs (Field 1):      constrained
                                      /
             0---0---0---0---0---0---1---0---0---0---0---0---0
             |       |       |       |       |       |       |
             0   0   0   0   0   0   2   0   0   1   0   0   0
             |       |       |       |       |       |       |
             0---0---0---0---0---0---0---0---0---0---0---0---0

exampleSectionDM: Offsets (total) in global section:

             0---0---0---0---0---0---3---5---5---5---5---5---5
             |       |       |       |       |       |       |
             0   0   0   0   0   0   5   0   7   2   7   3   7
             |       |       |       |       |       |       |
             0---0---0---0---0---0---3---5---3---5---3---5---3

exampleVec: Values (Field 0):          (1.3, 1.4)
                                      /
             +-------+-------+-------*-------+-------+-------+
             |       |       |       |       |       |       |
             |       |       |       |   * (1.0, 1.1)|       |
             |       |       |       |       |       |       |
             +-------+-------+-------+-------+-------+-------+

exampleVec: Values (Field 1):          (1.5,) constrained
                                      /
             +-------+-------+-------*-------+-------+-------+
             |       |       |       |       |       |       |
             |       |    (1.6, 1.7) *       |   * (1.2,)    |
             |       |       |       |       |       |       |
             +-------+-------+-------+-------+-------+-------+

exampleVec: as global vector

  rank 0: []
  rakn 1: [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7]

=====================
 Load on 3 Processes
=====================

exampleDMPlex: Loaded/Distributed:

             5--13---6--14--(8)(18)(10)
             |       |       |       |
  rank 0:   15   0   16  1  (19)(2)(20)
             |       |       |       |
             3--11---4--12--(7)(17)-(9)

                    (9)(21)--5--15---7--18-(12)(24)(13)
                     |       |       |       |       |
  rank 1:          (22) (2) 16   0  19   1 (25) (3)(26)
                     |       |       |       |       |
                    (8)(20)--4--14---6--17-(10)(23)(11)

                               +-> (10)(19)--6--13---7--14---8
                       permute |     |       |       |       |
  rank 2:                      +-> (20) (2) 15   0  16   1  17
                                     |       |       |       |
                                    (9)(18)--3--11---4--12---5

exampleSectionDM:
  - includesConstraints = TRUE for local section (default)
  - includesConstraints = FALSE for global section (default)

exampleVec: as local vector:

  rank 0: [1.3, 1.4, 1.5, 1.6, 1.7]
  rank 1: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
  rank 2: [1.2, 1.0, 1.1, 1.6, 1.7, 1.3, 1.4, 1.5]

exampleVec: as global vector:

  rank 0: []
  rank 1: [1.0, 1.1, 1.3, 1.4, 1.6, 1.7]
  rank 2: [1.2]

*/

typedef struct {
  char       fname[PETSC_MAX_PATH_LEN]; /* Output mesh filename */
  PetscBool  shell;                     /* Use DMShell to wrap sections */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  options->fname[0] = '\0';
  ierr = PetscOptionsBegin(comm, "", "DMPlex View/Load Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-fname", "The output mesh file", "ex12.c", options->fname, options->fname, sizeof(options->fname), &flg));
  CHKERRQ(PetscOptionsBool("-shell", "Use DMShell to wrap sections", "ex12.c", options->shell, &options->shell, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm           comm;
  PetscMPIInt        size, rank, mycolor;
  const char         exampleDMPlexName[]    = "exampleDMPlex";
  const char         exampleSectionDMName[] = "exampleSectionDM";
  const char         exampleVecName[]       = "exampleVec";
  PetscScalar        constraintValue        = 1.5;
  PetscViewerFormat  format                 = PETSC_VIEWER_HDF5_PETSC;
  AppCtx             user;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheckFalse(size < 3,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");

  /* Save */
  mycolor = (PetscMPIInt)(rank >= 2);
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM           dm;
    PetscViewer  viewer;

    CHKERRQ(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer));
    /* Save exampleDMPlex */
    {
      DM              pdm;
      const PetscInt  faces[2] = {6, 1};
      PetscSF         sf;
      PetscInt        overlap = 1;

      CHKERRQ(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
      CHKERRQ(DMPlexDistribute(dm, overlap, &sf, &pdm));
      if (pdm) {
        CHKERRQ(DMDestroy(&dm));
        dm = pdm;
      }
      CHKERRQ(PetscSFDestroy(&sf));
      CHKERRQ(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      CHKERRQ(PetscViewerPushFormat(viewer, format));
      CHKERRQ(DMPlexTopologyView(dm, viewer));
      CHKERRQ(DMPlexLabelsView(dm, viewer));
      CHKERRQ(PetscViewerPopFormat(viewer));
    }
    /* Save coordinates */
    CHKERRQ(PetscViewerPushFormat(viewer, format));
    CHKERRQ(DMPlexCoordinatesView(dm, viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    /* Save exampleVec */
    {
      PetscInt      pStart = -1, pEnd = -1;
      DM            sdm;
      PetscSection  section, gsection;
      PetscBool     includesConstraints = PETSC_FALSE;
      Vec           vec;
      PetscScalar  *array = NULL;

      /* Create section */
      CHKERRQ(PetscSectionCreate(comm, &section));
      CHKERRQ(PetscSectionSetNumFields(section, 2));
      CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
      CHKERRQ(PetscSectionSetChart(section, pStart, pEnd));
      switch (rank) {
      case 0:
        CHKERRQ(PetscSectionSetDof(section, 3, 2));
        CHKERRQ(PetscSectionSetDof(section, 12, 3));
        CHKERRQ(PetscSectionSetDof(section, 25, 2));
        CHKERRQ(PetscSectionSetConstraintDof(section, 12, 1));
        CHKERRQ(PetscSectionSetFieldDof(section, 3, 0, 2));
        CHKERRQ(PetscSectionSetFieldDof(section, 12, 0, 2));
        CHKERRQ(PetscSectionSetFieldDof(section, 12, 1, 1));
        CHKERRQ(PetscSectionSetFieldDof(section, 25, 1, 2));
        CHKERRQ(PetscSectionSetFieldConstraintDof(section, 12, 1, 1));
        break;
      case 1:
        CHKERRQ(PetscSectionSetDof(section, 0, 2));
        CHKERRQ(PetscSectionSetDof(section, 1, 1));
        CHKERRQ(PetscSectionSetDof(section, 8, 3));
        CHKERRQ(PetscSectionSetDof(section, 20, 2));
        CHKERRQ(PetscSectionSetConstraintDof(section, 8, 1));
        CHKERRQ(PetscSectionSetFieldDof(section, 0, 0, 2));
        CHKERRQ(PetscSectionSetFieldDof(section, 8, 0, 2));
        CHKERRQ(PetscSectionSetFieldDof(section, 1, 1, 1));
        CHKERRQ(PetscSectionSetFieldDof(section, 8, 1, 1));
        CHKERRQ(PetscSectionSetFieldDof(section, 20, 1, 2));
        CHKERRQ(PetscSectionSetFieldConstraintDof(section, 8, 1, 1));
        break;
      }
      CHKERRQ(PetscSectionSetUp(section));
      {
        const PetscInt indices[] = {2};
        const PetscInt indices1[] = {0};

        switch (rank) {
        case 0:
          CHKERRQ(PetscSectionSetConstraintIndices(section, 12, indices));
          CHKERRQ(PetscSectionSetFieldConstraintIndices(section, 12, 1, indices1));
          break;
        case 1:
          CHKERRQ(PetscSectionSetConstraintIndices(section, 8, indices));
          CHKERRQ(PetscSectionSetFieldConstraintIndices(section, 8, 1, indices1));
          break;
        }
      }
      if (user.shell) {
        PetscSF  sf;

        CHKERRQ(DMShellCreate(comm, &sdm));
        CHKERRQ(DMGetPointSF(dm, &sf));
        CHKERRQ(DMSetPointSF(sdm, sf));
      }
      else {
        CHKERRQ(DMClone(dm, &sdm));
      }
      CHKERRQ(PetscObjectSetName((PetscObject)sdm, exampleSectionDMName));
      CHKERRQ(DMSetLocalSection(sdm, section));
      CHKERRQ(PetscSectionDestroy(&section));
      CHKERRQ(DMPlexSectionView(dm, viewer, sdm));
      /* Create global vector */
      CHKERRQ(DMGetGlobalSection(sdm, &gsection));
      CHKERRQ(PetscSectionGetIncludesConstraints(gsection, &includesConstraints));
      if (user.shell) {
        PetscInt  n = -1;

        CHKERRQ(VecCreate(comm, &vec));
        if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(gsection, &n));
        else CHKERRQ(PetscSectionGetConstrainedStorageSize(gsection, &n));
        CHKERRQ(VecSetSizes(vec, n, PETSC_DECIDE));
        CHKERRQ(VecSetUp(vec));
      } else {
        CHKERRQ(DMGetGlobalVector(sdm, &vec));
      }
      CHKERRQ(PetscObjectSetName((PetscObject)vec, exampleVecName));
      CHKERRQ(VecGetArrayWrite(vec, &array));
      if (includesConstraints) {
        switch (rank) {
        case 0:
          break;
        case 1:
          array[0] = 1.0;
          array[1] = 1.1;
          array[2] = 1.2;
          array[3] = 1.3;
          array[4] = 1.4;
          array[5] = 1.5;
          array[6] = 1.6;
          array[7] = 1.7;
          break;
        }
      } else {
        switch (rank) {
        case 0:
          break;
        case 1:
          array[0] = 1.0;
          array[1] = 1.1;
          array[2] = 1.2;
          array[3] = 1.3;
          array[4] = 1.4;
          array[5] = 1.6;
          array[6] = 1.7;
          break;
        }
      }
      CHKERRQ(VecRestoreArrayWrite(vec, &array));
      CHKERRQ(DMPlexGlobalVectorView(dm, viewer, sdm, vec));
      if (user.shell) {
        CHKERRQ(VecDestroy(&vec));
      } else {
        CHKERRQ(DMRestoreGlobalVector(sdm, &vec));
      }
      CHKERRQ(DMDestroy(&sdm));
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(DMDestroy(&dm));
  }
  CHKERRMPI(MPI_Comm_free(&comm));
  /* Load */
  mycolor = (PetscMPIInt)(rank >= 3);
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM           dm;
    PetscSF      sfXC;
    PetscViewer  viewer;

    CHKERRQ(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer));
    /* Load exampleDMPlex */
    {
      PetscSF  sfXB, sfBC;

      CHKERRQ(DMCreate(comm, &dm));
      CHKERRQ(DMSetType(dm, DMPLEX));
      CHKERRQ(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      /* sfXB: X -> B                         */
      /* X: set of globalPointNumbers, [0, N) */
      /* B: loaded naive in-memory plex       */
      CHKERRQ(PetscViewerPushFormat(viewer, format));
      CHKERRQ(DMPlexTopologyLoad(dm, viewer, &sfXB));
      CHKERRQ(PetscViewerPopFormat(viewer));
      CHKERRQ(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      {
        DM               distributedDM;
        PetscInt         overlap = 1;
        PetscPartitioner part;

        CHKERRQ(DMPlexGetPartitioner(dm, &part));
        CHKERRQ(PetscPartitionerSetFromOptions(part));
        /* sfBC: B -> C                    */
        /* B: loaded naive in-memory plex  */
        /* C: redistributed good in-memory */
        CHKERRQ(DMPlexDistribute(dm, overlap, &sfBC, &distributedDM));
        if (distributedDM) {
          CHKERRQ(DMDestroy(&dm));
          dm = distributedDM;
        }
        CHKERRQ(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      }
      /* sfXC: X -> C */
      CHKERRQ(PetscSFCompose(sfXB, sfBC, &sfXC));
      CHKERRQ(PetscSFDestroy(&sfXB));
      CHKERRQ(PetscSFDestroy(&sfBC));
    }
    /* Load labels */
    CHKERRQ(PetscViewerPushFormat(viewer, format));
    CHKERRQ(DMPlexLabelsLoad(dm, viewer, sfXC));
    CHKERRQ(PetscViewerPopFormat(viewer));
    /* Load coordinates */
    CHKERRQ(PetscViewerPushFormat(viewer, format));
    CHKERRQ(DMPlexCoordinatesLoad(dm, viewer, sfXC));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)dm, "Load: DM (with coordinates)"));
    CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
    CHKERRQ(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
    /* Load exampleVec */
    {
      DM            sdm;
      PetscSection  section, gsection;
      IS            perm;
      PetscBool     includesConstraints = PETSC_FALSE;
      Vec           vec;
      PetscSF       lsf, gsf;

      if (user.shell) {
        PetscSF  sf;

        CHKERRQ(DMShellCreate(comm, &sdm));
        CHKERRQ(DMGetPointSF(dm, &sf));
        CHKERRQ(DMSetPointSF(sdm, sf));
      } else {
        CHKERRQ(DMClone(dm, &sdm));
      }
      CHKERRQ(PetscObjectSetName((PetscObject)sdm, exampleSectionDMName));
      CHKERRQ(PetscSectionCreate(comm, &section));
      {
        PetscInt      pStart = -1, pEnd = -1, p = -1;
        PetscInt     *pinds = NULL;

        CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
        CHKERRQ(PetscMalloc1(pEnd - pStart, &pinds));
        for (p = 0; p < pEnd - pStart; ++p) pinds[p] = p;
        if (rank == 2) {pinds[10] = 20; pinds[20] = 10;}
        CHKERRQ(ISCreateGeneral(comm, pEnd - pStart, pinds, PETSC_OWN_POINTER, &perm));
      }
      CHKERRQ(PetscSectionSetPermutation(section, perm));
      CHKERRQ(ISDestroy(&perm));
      CHKERRQ(DMSetLocalSection(sdm, section));
      CHKERRQ(PetscSectionDestroy(&section));
      CHKERRQ(DMPlexSectionLoad(dm, viewer, sdm, sfXC, &gsf, &lsf));
      /* Load as local vector */
      CHKERRQ(DMGetLocalSection(sdm, &section));
      CHKERRQ(PetscObjectSetName((PetscObject)section, "Load: local section"));
      CHKERRQ(PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm)));
      CHKERRQ(PetscSectionGetIncludesConstraints(section, &includesConstraints));
      if (user.shell) {
        PetscInt  m = -1;

        CHKERRQ(VecCreate(comm, &vec));
        if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(section, &m));
        else CHKERRQ(PetscSectionGetConstrainedStorageSize(section, &m));
        CHKERRQ(VecSetSizes(vec, m, PETSC_DECIDE));
        CHKERRQ(VecSetUp(vec));
      } else {
        CHKERRQ(DMGetLocalVector(sdm, &vec));
      }
      CHKERRQ(PetscObjectSetName((PetscObject)vec, exampleVecName));
      CHKERRQ(VecSet(vec, constraintValue));
      CHKERRQ(DMPlexLocalVectorLoad(dm, viewer, sdm, lsf, vec));
      CHKERRQ(PetscSFDestroy(&lsf));
      if (user.shell) {
        CHKERRQ(VecView(vec, PETSC_VIEWER_STDOUT_(comm)));
        CHKERRQ(VecDestroy(&vec));
      } else {
        CHKERRQ(DMRestoreLocalVector(sdm, &vec));
      }
      /* Load as global vector */
      CHKERRQ(DMGetGlobalSection(sdm, &gsection));
      CHKERRQ(PetscObjectSetName((PetscObject)gsection, "Load: global section"));
      CHKERRQ(PetscSectionView(gsection, PETSC_VIEWER_STDOUT_(comm)));
      CHKERRQ(PetscSectionGetIncludesConstraints(gsection, &includesConstraints));
      if (user.shell) {
        PetscInt  m = -1;

        CHKERRQ(VecCreate(comm, &vec));
        if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(gsection, &m));
        else CHKERRQ(PetscSectionGetConstrainedStorageSize(gsection, &m));
        CHKERRQ(VecSetSizes(vec, m, PETSC_DECIDE));
        CHKERRQ(VecSetUp(vec));
      } else {
        CHKERRQ(DMGetGlobalVector(sdm, &vec));
      }
      CHKERRQ(PetscObjectSetName((PetscObject)vec, exampleVecName));
      CHKERRQ(DMPlexGlobalVectorLoad(dm, viewer, sdm, gsf, vec));
      CHKERRQ(PetscSFDestroy(&gsf));
      CHKERRQ(VecView(vec, PETSC_VIEWER_STDOUT_(comm)));
      if (user.shell) {
        CHKERRQ(VecDestroy(&vec));
      } else {
        CHKERRQ(DMRestoreGlobalVector(sdm, &vec));
      }
      CHKERRQ(DMDestroy(&sdm));
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(PetscSFDestroy(&sfXC));
    CHKERRQ(DMDestroy(&dm));
  }
  CHKERRMPI(MPI_Comm_free(&comm));

  /* Finalize */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: hdf5
  testset:
    suffix: 0
    requires: !complex
    nsize: 4
    args: -fname ex12_dump.h5 -shell {{True False}separate output} -dm_view ascii::ascii_info_detail
    args: -dm_plex_view_hdf5_storage_version 2.0.0
    test:
      suffix: parmetis
      requires: parmetis
      args: -petscpartitioner_type parmetis
    test:
      suffix: ptscotch
      requires: ptscotch
      args: -petscpartitioner_type ptscotch

TEST*/
