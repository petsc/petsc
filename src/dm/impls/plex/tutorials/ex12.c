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
  ierr = PetscOptionsString("-fname", "The output mesh file", "ex12.c", options->fname, options->fname, sizeof(options->fname), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-shell", "Use DMShell to wrap sections", "ex12.c", options->shell, &options->shell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  if (size < 3) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");

  /* Save */
  mycolor = (PetscMPIInt)(rank >= 2);
  ierr = MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm);CHKERRMPI(ierr);
  if (mycolor == 0) {
    DM           dm;
    PetscViewer  viewer;

    ierr = PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    /* Save exampleDMPlex */
    {
      DM              pdm;
      const PetscInt  faces[2] = {6, 1};
      PetscSF         sf;
      PetscInt        overlap = 1;

      ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm);CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm, overlap, &sf, &pdm);CHKERRQ(ierr);
      if (pdm) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm = pdm;
      }
      ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dm, exampleDMPlexName);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
      ierr = DMPlexTopologyView(dm, viewer);CHKERRQ(ierr);
      ierr = DMPlexLabelsView(dm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    /* Save coordinates */
    /* The following block is to replace DMPlexCoordinatesView(). */
    {
      DM         cdm;
      Vec        coords, newcoords;
      PetscInt   m = -1, M = -1, bs = -1;
      PetscReal  lengthScale = -1;

      ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)cdm, "coordinateDM");CHKERRQ(ierr);
      ierr = DMPlexSectionView(dm, viewer, cdm);CHKERRQ(ierr);
      ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
      ierr = VecCreate(PetscObjectComm((PetscObject)coords), &newcoords);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)newcoords, "coordinates");CHKERRQ(ierr);
      ierr = VecGetSize(coords, &M);CHKERRQ(ierr);
      ierr = VecGetLocalSize(coords, &m);CHKERRQ(ierr);
      ierr = VecSetSizes(newcoords, m, M);CHKERRQ(ierr);
      ierr = VecGetBlockSize(coords, &bs);CHKERRQ(ierr);
      ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);
      ierr = VecSetType(newcoords,VECSTANDARD);CHKERRQ(ierr);
      ierr = VecCopy(coords, newcoords);CHKERRQ(ierr);
      ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
      ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
      ierr = DMPlexGlobalVectorView(dm, viewer, cdm, newcoords);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
    }
    /* Save exampleVec */
    {
      PetscInt      pStart = -1, pEnd = -1;
      DM            sdm;
      PetscSection  section, gsection;
      PetscBool     includesConstraints = PETSC_FALSE;
      Vec           vec;
      PetscScalar  *array = NULL;

      /* Create section */
      ierr = PetscSectionCreate(comm, &section);CHKERRQ(ierr);
      ierr = PetscSectionSetNumFields(section, 2);CHKERRQ(ierr);
      ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
      switch (rank) {
      case 0:
        ierr = PetscSectionSetDof(section, 3, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 12, 3);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 25, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetConstraintDof(section, 12, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 3, 0, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 12, 0, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 12, 1, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 25, 1, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldConstraintDof(section, 12, 1, 1);CHKERRQ(ierr);
        break;
      case 1:
        ierr = PetscSectionSetDof(section, 0, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 1, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 8, 3);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 20, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetConstraintDof(section, 8, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 0, 0, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 8, 0, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 1, 1, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 8, 1, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(section, 20, 1, 2);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldConstraintDof(section, 8, 1, 1);CHKERRQ(ierr);
        break;
      }
      ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
      {
        const PetscInt indices[] = {2};
        const PetscInt indices1[] = {0};

        switch (rank) {
        case 0:
          ierr = PetscSectionSetConstraintIndices(section, 12, indices);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldConstraintIndices(section, 12, 1, indices1);CHKERRQ(ierr);
          break;
        case 1:
          ierr = PetscSectionSetConstraintIndices(section, 8, indices);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldConstraintIndices(section, 8, 1, indices1);CHKERRQ(ierr);
          break;
        }
      }
      if (user.shell) {
        PetscSF  sf;

        ierr = DMShellCreate(comm, &sdm);CHKERRQ(ierr);
        ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
        ierr = DMSetPointSF(sdm, sf);CHKERRQ(ierr);
      }
      else {
        ierr = DMClone(dm, &sdm);CHKERRQ(ierr);
      }
      ierr = PetscObjectSetName((PetscObject)sdm, exampleSectionDMName);CHKERRQ(ierr);
      ierr = DMSetLocalSection(sdm, section);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
      ierr = DMPlexSectionView(dm, viewer, sdm);CHKERRQ(ierr);
      /* Create global vector */
      ierr = DMGetGlobalSection(sdm, &gsection);CHKERRQ(ierr);
      ierr = PetscSectionGetIncludesConstraints(gsection, &includesConstraints);CHKERRQ(ierr);
      if (user.shell) {
        PetscInt  n = -1;

        ierr = VecCreate(comm, &vec);CHKERRQ(ierr);
        if (includesConstraints) {ierr = PetscSectionGetStorageSize(gsection, &n);CHKERRQ(ierr);}
        else {ierr = PetscSectionGetConstrainedStorageSize(gsection, &n);CHKERRQ(ierr);}
        ierr = VecSetSizes(vec, n, PETSC_DECIDE);CHKERRQ(ierr);
        ierr = VecSetUp(vec);CHKERRQ(ierr);
      } else {
        ierr = DMGetGlobalVector(sdm, &vec);CHKERRQ(ierr);
      }
      ierr = PetscObjectSetName((PetscObject)vec, exampleVecName);CHKERRQ(ierr);
      ierr = VecGetArrayWrite(vec, &array);CHKERRQ(ierr);
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
      ierr = VecRestoreArrayWrite(vec, &array);CHKERRQ(ierr);
      ierr = DMPlexGlobalVectorView(dm, viewer, sdm, vec);CHKERRQ(ierr);
      if (user.shell) {
        ierr = VecDestroy(&vec);CHKERRQ(ierr);
      } else {
        ierr = DMRestoreGlobalVector(sdm, &vec);CHKERRQ(ierr);
      }
      ierr = DMDestroy(&sdm);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_free(&comm);CHKERRMPI(ierr);
  /* Load */
  mycolor = (PetscMPIInt)(rank >= 3);
  ierr = MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm);CHKERRMPI(ierr);
  if (mycolor == 0) {
    DM           dm;
    PetscSF      sfXC;
    PetscViewer  viewer;

    ierr = PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
    /* Load exampleDMPlex */
    {
      PetscSF  sfXB, sfBC;

      ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
      ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dm, exampleDMPlexName);CHKERRQ(ierr);
      /* sfXB: X -> B                         */
      /* X: set of globalPointNumbers, [0, N) */
      /* B: loaded naive in-memory plex       */
      ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
      ierr = DMPlexTopologyLoad(dm, viewer, &sfXB);CHKERRQ(ierr);
      ierr = DMPlexLabelsLoad(dm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dm, exampleDMPlexName);CHKERRQ(ierr);
      {
        DM               distributedDM;
        PetscInt         overlap = 1;
        PetscPartitioner part;

        ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
        ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
        /* sfBC: B -> C                    */
        /* B: loaded naive in-memory plex  */
        /* C: redistributed good in-memory */
        ierr = DMPlexDistribute(dm, overlap, &sfBC, &distributedDM);CHKERRQ(ierr);
        if (distributedDM) {
          ierr = DMDestroy(&dm);CHKERRQ(ierr);
          dm = distributedDM;
        }
        ierr = PetscObjectSetName((PetscObject)dm, exampleDMPlexName);CHKERRQ(ierr);
      }
      /* sfXC: X -> C */
      ierr = PetscSFCompose(sfXB, sfBC, &sfXC);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfXB);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfBC);CHKERRQ(ierr);
    }
    /* Load coordinates */
    /* The following block is to replace DMPlexCoordinatesLoad() */
    {
      DM            cdm;
      PetscSection  coordSection;
      Vec           coords;
      PetscInt      m = -1;
      PetscReal     lengthScale = -1;
      PetscSF       lsf, gsf;

      ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)cdm, "coordinateDM");CHKERRQ(ierr);
      /* lsf: on-disk data -> in-memory local vector associated with cdm's local section */
      /* gsf: on-disk data -> in-memory global vector associated with cdm's global section */
      ierr = DMPlexSectionLoad(dm, viewer, cdm, sfXC, &gsf, &lsf);CHKERRQ(ierr);
      ierr = VecCreate(comm, &coords);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)coords, "coordinates");CHKERRQ(ierr);
      ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(coordSection, &m);CHKERRQ(ierr);
      ierr = VecSetSizes(coords, m, PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetUp(coords);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
      ierr = DMPlexLocalVectorLoad(dm, viewer, cdm, lsf, coords);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
      ierr = VecScale(coords, 1.0/lengthScale);CHKERRQ(ierr);
      ierr = DMSetCoordinatesLocal(dm, coords);CHKERRQ(ierr);
      ierr = VecDestroy(&coords);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&lsf);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&gsf);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dm, "Load: DM (with coordinates)");CHKERRQ(ierr);
      ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dm, exampleDMPlexName);CHKERRQ(ierr);
    }
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

        ierr = DMShellCreate(comm, &sdm);CHKERRQ(ierr);
        ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
        ierr = DMSetPointSF(sdm, sf);CHKERRQ(ierr);
      } else {
        ierr = DMClone(dm, &sdm);CHKERRQ(ierr);
      }
      ierr = PetscObjectSetName((PetscObject)sdm, exampleSectionDMName);CHKERRQ(ierr);
      ierr = PetscSectionCreate(comm, &section);CHKERRQ(ierr);
      {
        PetscInt      pStart = -1, pEnd = -1, p = -1;
        PetscInt     *pinds = NULL;

        ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
        ierr = PetscMalloc1(pEnd - pStart, &pinds);CHKERRQ(ierr);
        for (p = 0; p < pEnd - pStart; ++p) pinds[p] = p;
        if (rank == 2) {pinds[10] = 20; pinds[20] = 10;}
        ierr = ISCreateGeneral(comm, pEnd - pStart, pinds, PETSC_OWN_POINTER, &perm);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetPermutation(section, perm);CHKERRQ(ierr);
      ierr = ISDestroy(&perm);CHKERRQ(ierr);
      ierr = DMSetLocalSection(sdm, section);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
      ierr = DMPlexSectionLoad(dm, viewer, sdm, sfXC, &gsf, &lsf);CHKERRQ(ierr);
      /* Load as local vector */
      ierr = DMGetLocalSection(sdm, &section);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)section, "Load: local section");CHKERRQ(ierr);
      ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
      ierr = PetscSectionGetIncludesConstraints(section, &includesConstraints);CHKERRQ(ierr);
      if (user.shell) {
        PetscInt  m = -1;

        ierr = VecCreate(comm, &vec);CHKERRQ(ierr);
        if (includesConstraints) {ierr = PetscSectionGetStorageSize(section, &m);CHKERRQ(ierr);}
        else {ierr = PetscSectionGetConstrainedStorageSize(section, &m);CHKERRQ(ierr);}
        ierr = VecSetSizes(vec, m, PETSC_DECIDE);CHKERRQ(ierr);
        ierr = VecSetUp(vec);CHKERRQ(ierr);
      } else {
        ierr = DMGetLocalVector(sdm, &vec);CHKERRQ(ierr);
      }
      ierr = PetscObjectSetName((PetscObject)vec, exampleVecName);CHKERRQ(ierr);
      ierr = VecSet(vec, constraintValue);CHKERRQ(ierr);
      ierr = DMPlexLocalVectorLoad(dm, viewer, sdm, lsf, vec);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&lsf);CHKERRQ(ierr);
      if (user.shell) {
        ierr = VecView(vec, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
        ierr = VecDestroy(&vec);CHKERRQ(ierr);
      } else {
        ierr = DMRestoreLocalVector(sdm, &vec);CHKERRQ(ierr);
      }
      /* Load as global vector */
      ierr = DMGetGlobalSection(sdm, &gsection);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)gsection, "Load: global section");CHKERRQ(ierr);
      ierr = PetscSectionView(gsection, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
      ierr = PetscSectionGetIncludesConstraints(gsection, &includesConstraints);CHKERRQ(ierr);
      if (user.shell) {
        PetscInt  m = -1;

        ierr = VecCreate(comm, &vec);CHKERRQ(ierr);
        if (includesConstraints) {ierr = PetscSectionGetStorageSize(gsection, &m);CHKERRQ(ierr);}
        else {ierr = PetscSectionGetConstrainedStorageSize(gsection, &m);CHKERRQ(ierr);}
        ierr = VecSetSizes(vec, m, PETSC_DECIDE);CHKERRQ(ierr);
        ierr = VecSetUp(vec);CHKERRQ(ierr);
      } else {
        ierr = DMGetGlobalVector(sdm, &vec);CHKERRQ(ierr);
      }
      ierr = PetscObjectSetName((PetscObject)vec, exampleVecName);CHKERRQ(ierr);
      ierr = DMPlexGlobalVectorLoad(dm, viewer, sdm, gsf, vec);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&gsf);CHKERRQ(ierr);
      ierr = VecView(vec, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
      if (user.shell) {
        ierr = VecDestroy(&vec);CHKERRQ(ierr);
      } else {
        ierr = DMRestoreGlobalVector(sdm, &vec);CHKERRQ(ierr);
      }
      ierr = DMDestroy(&sdm);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfXC);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_free(&comm);CHKERRMPI(ierr);

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
    test:
      suffix: parmetis
      requires: parmetis
      args: -petscpartitioner_type parmetis
    test:
      suffix: ptscotch
      requires: ptscotch
      args: -petscpartitioner_type ptscotch

TEST*/
