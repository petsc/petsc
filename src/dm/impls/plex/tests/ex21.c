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
  ierr = PetscOptionsBegin(comm, "", "DMPlex View/Load Test Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsString("-fname", "The output mesh file", "ex12.c", options->fname, options->fname, sizeof(options->fname), &flg));
  PetscCall(PetscOptionsBool("-shell", "Use DMShell to wrap sections", "ex12.c", options->shell, &options->shell, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
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

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheckFalse(size < 3,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");

  /* Save */
  mycolor = (PetscMPIInt)(rank >= 2);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM           dm;
    PetscViewer  viewer;

    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer));
    /* Save exampleDMPlex */
    {
      DM              pdm;
      const PetscInt  faces[2] = {6, 1};
      PetscSF         sf;
      PetscInt        overlap = 1;

      PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
      PetscCall(DMPlexDistribute(dm, overlap, &sf, &pdm));
      if (pdm) {
        PetscCall(DMDestroy(&dm));
        dm = pdm;
      }
      PetscCall(PetscSFDestroy(&sf));
      PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(DMPlexTopologyView(dm, viewer));
      PetscCall(DMPlexLabelsView(dm, viewer));
      PetscCall(PetscViewerPopFormat(viewer));
    }
    /* Save coordinates */
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(DMPlexCoordinatesView(dm, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    /* Save exampleVec */
    {
      PetscInt      pStart = -1, pEnd = -1;
      DM            sdm;
      PetscSection  section, gsection;
      PetscBool     includesConstraints = PETSC_FALSE;
      Vec           vec;
      PetscScalar  *array = NULL;

      /* Create section */
      PetscCall(PetscSectionCreate(comm, &section));
      PetscCall(PetscSectionSetNumFields(section, 2));
      PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
      PetscCall(PetscSectionSetChart(section, pStart, pEnd));
      switch (rank) {
      case 0:
        PetscCall(PetscSectionSetDof(section, 3, 2));
        PetscCall(PetscSectionSetDof(section, 12, 3));
        PetscCall(PetscSectionSetDof(section, 25, 2));
        PetscCall(PetscSectionSetConstraintDof(section, 12, 1));
        PetscCall(PetscSectionSetFieldDof(section, 3, 0, 2));
        PetscCall(PetscSectionSetFieldDof(section, 12, 0, 2));
        PetscCall(PetscSectionSetFieldDof(section, 12, 1, 1));
        PetscCall(PetscSectionSetFieldDof(section, 25, 1, 2));
        PetscCall(PetscSectionSetFieldConstraintDof(section, 12, 1, 1));
        break;
      case 1:
        PetscCall(PetscSectionSetDof(section, 0, 2));
        PetscCall(PetscSectionSetDof(section, 1, 1));
        PetscCall(PetscSectionSetDof(section, 8, 3));
        PetscCall(PetscSectionSetDof(section, 20, 2));
        PetscCall(PetscSectionSetConstraintDof(section, 8, 1));
        PetscCall(PetscSectionSetFieldDof(section, 0, 0, 2));
        PetscCall(PetscSectionSetFieldDof(section, 8, 0, 2));
        PetscCall(PetscSectionSetFieldDof(section, 1, 1, 1));
        PetscCall(PetscSectionSetFieldDof(section, 8, 1, 1));
        PetscCall(PetscSectionSetFieldDof(section, 20, 1, 2));
        PetscCall(PetscSectionSetFieldConstraintDof(section, 8, 1, 1));
        break;
      }
      PetscCall(PetscSectionSetUp(section));
      {
        const PetscInt indices[] = {2};
        const PetscInt indices1[] = {0};

        switch (rank) {
        case 0:
          PetscCall(PetscSectionSetConstraintIndices(section, 12, indices));
          PetscCall(PetscSectionSetFieldConstraintIndices(section, 12, 1, indices1));
          break;
        case 1:
          PetscCall(PetscSectionSetConstraintIndices(section, 8, indices));
          PetscCall(PetscSectionSetFieldConstraintIndices(section, 8, 1, indices1));
          break;
        }
      }
      if (user.shell) {
        PetscSF  sf;

        PetscCall(DMShellCreate(comm, &sdm));
        PetscCall(DMGetPointSF(dm, &sf));
        PetscCall(DMSetPointSF(sdm, sf));
      }
      else {
        PetscCall(DMClone(dm, &sdm));
      }
      PetscCall(PetscObjectSetName((PetscObject)sdm, exampleSectionDMName));
      PetscCall(DMSetLocalSection(sdm, section));
      PetscCall(PetscSectionDestroy(&section));
      PetscCall(DMPlexSectionView(dm, viewer, sdm));
      /* Create global vector */
      PetscCall(DMGetGlobalSection(sdm, &gsection));
      PetscCall(PetscSectionGetIncludesConstraints(gsection, &includesConstraints));
      if (user.shell) {
        PetscInt  n = -1;

        PetscCall(VecCreate(comm, &vec));
        if (includesConstraints) PetscCall(PetscSectionGetStorageSize(gsection, &n));
        else PetscCall(PetscSectionGetConstrainedStorageSize(gsection, &n));
        PetscCall(VecSetSizes(vec, n, PETSC_DECIDE));
        PetscCall(VecSetUp(vec));
      } else {
        PetscCall(DMGetGlobalVector(sdm, &vec));
      }
      PetscCall(PetscObjectSetName((PetscObject)vec, exampleVecName));
      PetscCall(VecGetArrayWrite(vec, &array));
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
      PetscCall(VecRestoreArrayWrite(vec, &array));
      PetscCall(DMPlexGlobalVectorView(dm, viewer, sdm, vec));
      if (user.shell) {
        PetscCall(VecDestroy(&vec));
      } else {
        PetscCall(DMRestoreGlobalVector(sdm, &vec));
      }
      PetscCall(DMDestroy(&sdm));
    }
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(DMDestroy(&dm));
  }
  PetscCallMPI(MPI_Comm_free(&comm));
  /* Load */
  mycolor = (PetscMPIInt)(rank >= 3);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM           dm;
    PetscSF      sfXC;
    PetscViewer  viewer;

    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer));
    /* Load exampleDMPlex */
    {
      PetscSF  sfXB, sfBC;

      PetscCall(DMCreate(comm, &dm));
      PetscCall(DMSetType(dm, DMPLEX));
      PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      /* sfXB: X -> B                         */
      /* X: set of globalPointNumbers, [0, N) */
      /* B: loaded naive in-memory plex       */
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(DMPlexTopologyLoad(dm, viewer, &sfXB));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      {
        DM               distributedDM;
        PetscInt         overlap = 1;
        PetscPartitioner part;

        PetscCall(DMPlexGetPartitioner(dm, &part));
        PetscCall(PetscPartitionerSetFromOptions(part));
        /* sfBC: B -> C                    */
        /* B: loaded naive in-memory plex  */
        /* C: redistributed good in-memory */
        PetscCall(DMPlexDistribute(dm, overlap, &sfBC, &distributedDM));
        if (distributedDM) {
          PetscCall(DMDestroy(&dm));
          dm = distributedDM;
        }
        PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      }
      /* sfXC: X -> C */
      PetscCall(PetscSFCompose(sfXB, sfBC, &sfXC));
      PetscCall(PetscSFDestroy(&sfXB));
      PetscCall(PetscSFDestroy(&sfBC));
    }
    /* Load labels */
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(DMPlexLabelsLoad(dm, viewer, sfXC));
    PetscCall(PetscViewerPopFormat(viewer));
    /* Load coordinates */
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfXC));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscObjectSetName((PetscObject)dm, "Load: DM (with coordinates)"));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
    PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
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

        PetscCall(DMShellCreate(comm, &sdm));
        PetscCall(DMGetPointSF(dm, &sf));
        PetscCall(DMSetPointSF(sdm, sf));
      } else {
        PetscCall(DMClone(dm, &sdm));
      }
      PetscCall(PetscObjectSetName((PetscObject)sdm, exampleSectionDMName));
      PetscCall(PetscSectionCreate(comm, &section));
      {
        PetscInt      pStart = -1, pEnd = -1, p = -1;
        PetscInt     *pinds = NULL;

        PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
        PetscCall(PetscMalloc1(pEnd - pStart, &pinds));
        for (p = 0; p < pEnd - pStart; ++p) pinds[p] = p;
        if (rank == 2) {pinds[10] = 20; pinds[20] = 10;}
        PetscCall(ISCreateGeneral(comm, pEnd - pStart, pinds, PETSC_OWN_POINTER, &perm));
      }
      PetscCall(PetscSectionSetPermutation(section, perm));
      PetscCall(ISDestroy(&perm));
      PetscCall(DMSetLocalSection(sdm, section));
      PetscCall(PetscSectionDestroy(&section));
      PetscCall(DMPlexSectionLoad(dm, viewer, sdm, sfXC, &gsf, &lsf));
      /* Load as local vector */
      PetscCall(DMGetLocalSection(sdm, &section));
      PetscCall(PetscObjectSetName((PetscObject)section, "Load: local section"));
      PetscCall(PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm)));
      PetscCall(PetscSectionGetIncludesConstraints(section, &includesConstraints));
      if (user.shell) {
        PetscInt  m = -1;

        PetscCall(VecCreate(comm, &vec));
        if (includesConstraints) PetscCall(PetscSectionGetStorageSize(section, &m));
        else PetscCall(PetscSectionGetConstrainedStorageSize(section, &m));
        PetscCall(VecSetSizes(vec, m, PETSC_DECIDE));
        PetscCall(VecSetUp(vec));
      } else {
        PetscCall(DMGetLocalVector(sdm, &vec));
      }
      PetscCall(PetscObjectSetName((PetscObject)vec, exampleVecName));
      PetscCall(VecSet(vec, constraintValue));
      PetscCall(DMPlexLocalVectorLoad(dm, viewer, sdm, lsf, vec));
      PetscCall(PetscSFDestroy(&lsf));
      if (user.shell) {
        PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_(comm)));
        PetscCall(VecDestroy(&vec));
      } else {
        PetscCall(DMRestoreLocalVector(sdm, &vec));
      }
      /* Load as global vector */
      PetscCall(DMGetGlobalSection(sdm, &gsection));
      PetscCall(PetscObjectSetName((PetscObject)gsection, "Load: global section"));
      PetscCall(PetscSectionView(gsection, PETSC_VIEWER_STDOUT_(comm)));
      PetscCall(PetscSectionGetIncludesConstraints(gsection, &includesConstraints));
      if (user.shell) {
        PetscInt  m = -1;

        PetscCall(VecCreate(comm, &vec));
        if (includesConstraints) PetscCall(PetscSectionGetStorageSize(gsection, &m));
        else PetscCall(PetscSectionGetConstrainedStorageSize(gsection, &m));
        PetscCall(VecSetSizes(vec, m, PETSC_DECIDE));
        PetscCall(VecSetUp(vec));
      } else {
        PetscCall(DMGetGlobalVector(sdm, &vec));
      }
      PetscCall(PetscObjectSetName((PetscObject)vec, exampleVecName));
      PetscCall(DMPlexGlobalVectorLoad(dm, viewer, sdm, gsf, vec));
      PetscCall(PetscSFDestroy(&gsf));
      PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_(comm)));
      if (user.shell) {
        PetscCall(VecDestroy(&vec));
      } else {
        PetscCall(DMRestoreGlobalVector(sdm, &vec));
      }
      PetscCall(DMDestroy(&sdm));
    }
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscSFDestroy(&sfXC));
    PetscCall(DMDestroy(&dm));
  }
  PetscCallMPI(MPI_Comm_free(&comm));

  /* Finalize */
  PetscCall(PetscFinalize());
  return 0;
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
