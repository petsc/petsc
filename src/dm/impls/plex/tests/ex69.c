static char help[] = "Tests for creation of cohesive meshes by transforms\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>

PETSC_EXTERN char tri_2_cv[];
char              tri_2_cv[] = "\
2 4 6 3 1\n\
0 2 1\n\
1 2 3\n\
4 1 5\n\
4 0 1\n\
-1.0  0.0 0.0  1\n\
 0.0  1.0 0.0 -1\n\
 0.0 -1.0 0.0  1\n\
 1.0  0.0 0.0 -1\n\
-2.0  1.0 0.0  1\n\
-1.0  2.0 0.0 -1";

/* List of test meshes

Test tri_0: triangle

 4-10--5      8-16--7-14--4
 |\  1 |      |\     \  1 |
 | \   |      | \     \   |
 6  8  9  ->  9 12  2  11 13
 |   \ |      |   \     \ |
 | 0  \|      | 0  \     \|
 2--7--3      3-10--6-15--5

Test tri_1: triangle, not tensor

 4-10--5      8-10--7-16--4
 |\  1 |      |\     \  1 |
 | \   |      | \     \   |
 6  8  9  -> 11 14  2  13 15
 |   \ |      |   \     \ |
 | 0  \|      | 0  \     \|
 2--7--3      3-12--6--9--5

Test tri_2: 4 triangles, non-oriented surface

           9
          / \
         /   \
       17  2  16
       /       \
      /         \
     8-----15----5
      \         /|\
       \       / | \
       18  3  12 |  14
         \   /   |   \
          \ /    |    \
           4  0 11  1  7
            \    |    /
             \   |   /
             10  |  13
               \ | /
                \|/
                 6
  becomes
           8
          / \
         /   \
        /     \
      25   2   24
      /         \
     /           \
   13-----18------9
28  |     5    26/ \
   14----19----10   \
     \         /|   |\
      \       / |   | \
      21  3  20 |   |  23
        \   /   |   |   \
         \ /    |   |    \
          6  0 17 4 16 1  7
           \    |   |    /
            \   |   |   /
            15  |   |  22
              \ |   | /
               \|   |/
               12---11
                 27

Test tri_3: tri_2, in parallel

           6
          / \
         /   \
        /     \
      12   1   11
      /         \
     /           \
    5-----10------2
                   \
    5-----9-----3   2
     \         /|   |\
      \       / |   | \
      10  1  8  |   |  9
        \   /   |   |   \
         \ /    |   |    \
          2  0  7   7  0  4
           \    |   |    /
            \   |   |   /
             6  |   |  8
              \ |   | /
               \|   |/
                4   3
  becomes
                 11
                / \
               /   \
              /     \
            19   1   18
            /         \
           /           \
          8-----14------4
        22 \     3       |
            9------15    |\
                    \    | \
    9------14-----5  \  20 |
  20\    3     18/ \  \/   |
   10----15-----6   |  5   |
     \         /|   |  |   |\
      \       / |   |  |   | \
      17  1 16  |   |  |   |  17
        \   /   | 2 |  | 2 |   \
         \ /    |   |  |   |    \
          4  0  13 12  13  12 0 10
           \    |   |  |   |    /
            \   |   |  |   |   /
            11  |   |  |   |  16
              \ |   |  |   | /
               \|   |  |   |/
                8---7  7---6
                 19      21

Test quad_0: quadrilateral

 5-10--6-11--7       5-12-10-20--9-14--6
 |     |     |       |     |     |     |
12  0 13  1  14 --> 15  0 18  2 17  1  16
 |     |     |       |     |     |     |
 2--8--3--9--4       3-11--8-19--7-13--4

Test quad_1: quadrilateral, not tensor

 5-10--6-11--7       5-14-10-12--9-16--6
 |     |     |       |     |     |     |
12  0 13  1  14 --> 17  0 20  2 19  1  18
 |     |     |       |     |     |     |
 2--8--3--9--4       3-13--8-11--7-15--4

Test quad_2: quadrilateral, 2 processes

 3--6--4  3--6--4       3--9--7-14--6   5-14--4--9--7
 |     |  |     |       |     |     |   |     |     |
 7  0  8  7  0  8  --> 10  0 12  1 11  12  1 11  0  10
 |     |  |     |       |     |     |   |     |     |
 1--5--2  1--5--2       2--8--5-13--4   3-13--2--8--6

Test quad_3: quadrilateral, 4 processes, non-oriented surface

 3--6--4  3--6--4      3--9--7-14--6   5-14--4--9--7
 |     |  |     |      |     |     |   |     |     |
 7  0  8  7  0  8     10  0  12 1  11 12  1 11  0  10
 |     |  |     |      |     |     |   |     |     |
 1--5--2  1--5--2      2--8--5-13--4   3-13--2--8--6
                   -->
 3--6--4  3--6--4      3--9--7-14--6   5-14--4--9--7
 |     |  |     |      |     |     |   |     |     |
 7  0  8  7  0  8     10  0  12 1  11 12  1 11  0  10
 |     |  |     |      |     |     |   |     |     |
 1--5--2  1--5--2      2--8--5-13--4   3-13--2--8--6

Test quad_4: embedded fault

14-24-15-25-16-26--17
 |     |     |     |
28  3 30  4 32  5  34
 |     |     |     |
10-21-11-22-12-23--13
 |     |     |     |
27  0 29  1 31  2  33
 |     |     |     |
 6-18--7-19--8-20--9

becomes

 13-26-14-27-15-28--16
  |     |     |     |
 30  3 32  4 39  5  40
  |     |     |     |
 12-25-17-36-19-38--21
        |     |     |
       41  6 42  7  43
        |     |     |
 12-25-17-35-18-37--20
  |     |     |     |
 29  0 31  1 33  2  34
  |     |     |     |
  8-22--9-23-10-24--11

Test quad_5: two faults

14-24-15-25-16-26--17
 |     |     |     |
28  3 30  4 32  5  34
 |     |     |     |
10-21-11-22-12-23--13
 |     |     |     |
27  0 29  1 31  2  33
 |     |     |     |
 6-18--7-19--8-20--9

becomes

12-26-13-27-14-28--15
 |     |     |     |
37  4 31  3 33  5  40
 |     |     |     |
17-36-18-25-19-39--21
 |     |     |     |
43  6  44   41  7  42
 |     |     |     |
16-35-18-25-19-38--20
 |     |     |     |
29  0 30  1 32  2  34
 |     |     |     |
 8-22--9-23-10-24--11

Test quad_6: T-junction

14-24-15-25-16-26--17
 |     |     |     |
28  3 30  4 32  5  34
 |     |     |     |
10-21-11-22-12-23--13
 |     |     |     |
27  0 29  1 31  2  33
 |     |     |     |
 6-18--7-19--8-20--9

becomes

 13-26-14-27-15-28--16
  |     |     |     |
 30  3 32  4 39  5  40
  |     |     |     |
 12-25-17-36-19-38--21
        |     |     |
       41  6 42  7  43
        |     |     |
 12-25-17-35-18-37--20
  |     |     |     |
 29  0 31  1 33  2  34
  |     |     |     |
  8-22--9-23-10-24--11

becomes

 14-28-15-41-21-44--20-29-16
  |     |     |     |     |
 31  3 33  5 43  8 42  4  40
  |     |     |     |     |
 13-27-17-37-23-46--23-39-19
        |     |     |     |
       47  6 48    48  7  49
        |     |     |     |
 13-27-17-36-22-45--22-38-18
  |     |     |     |     |
 30  0 32  1 34    34  2  35
  |     |     |     |     |
  9-24-10-25-11-----11-26-12

List of future tests:
- Detect and error on intersecting faults
- 3D
*/

typedef struct {
  PetscInt testNum; // The mesh to test
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->testNum = 0;

  PetscOptionsBegin(comm, "", "Cohesive Meshing Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-test_num", "The particular mesh to test", "ex5.c", options->testNum, &options->testNum, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateQuadMesh1(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const PetscInt faces[2] = {1, 1};
  PetscReal      lower[2], upper[2];
  DMLabel        label;
  PetscMPIInt    rank;
  void          *get_tmp;
  PetscInt64    *cidx;
  PetscMPIInt    flg;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // Create serial mesh
  lower[0] = (PetscReal)(rank % 2);
  lower[1] = (PetscReal)(rank / 2);
  upper[0] = (PetscReal)(rank % 2) + 1.;
  upper[1] = (PetscReal)(rank / 2) + 1.;
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, 2, PETSC_FALSE, faces, lower, upper, NULL, PETSC_TRUE, dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "box"));
  // Flip edges to make fault non-oriented
  switch (rank) {
  case 2:
    PetscCall(DMPlexOrientPoint(*dm, 8, -1));
    break;
  case 3:
    PetscCall(DMPlexOrientPoint(*dm, 7, -1));
    break;
  default:
    break;
  }
  // Need this so that all procs create the cell types
  PetscCall(DMPlexGetCellTypeLabel(*dm, &label));
  // Replace comm in object (copied from PetscHeaderCreate/Destroy())
  PetscCall(PetscCommDestroy(&(*dm)->hdr.comm));
  PetscCall(PetscCommDuplicate(comm, &(*dm)->hdr.comm, &(*dm)->hdr.tag));
  PetscCallMPI(MPI_Comm_get_attr((*dm)->hdr.comm, Petsc_CreationIdx_keyval, &get_tmp, &flg));
  PetscCheck(flg, (*dm)->hdr.comm, PETSC_ERR_ARG_CORRUPT, "MPI_Comm does not have an object creation index");
  cidx            = (PetscInt64 *)get_tmp;
  (*dm)->hdr.cidx = (*cidx)++;
  // Create new pointSF
  {
    PetscSF      sf;
    PetscInt    *local  = NULL;
    PetscSFNode *remote = NULL;
    PetscInt     Nl;

    PetscCall(PetscSFCreate(comm, &sf));
    switch (rank) {
    case 0:
      Nl = 5;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 2;
      remote[0].index = 1;
      remote[0].rank  = 1;
      local[1]        = 3;
      remote[1].index = 1;
      remote[1].rank  = 2;
      local[2]        = 4;
      remote[2].index = 1;
      remote[2].rank  = 3;
      local[3]        = 6;
      remote[3].index = 5;
      remote[3].rank  = 2;
      local[4]        = 8;
      remote[4].index = 7;
      remote[4].rank  = 1;
      break;
    case 1:
      Nl = 3;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 3;
      remote[0].index = 1;
      remote[0].rank  = 3;
      local[1]        = 4;
      remote[1].index = 2;
      remote[1].rank  = 3;
      local[2]        = 6;
      remote[2].index = 5;
      remote[2].rank  = 3;
      break;
    case 2:
      Nl = 3;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 2;
      remote[0].index = 1;
      remote[0].rank  = 3;
      local[1]        = 4;
      remote[1].index = 3;
      remote[1].rank  = 3;
      local[2]        = 8;
      remote[2].index = 7;
      remote[2].rank  = 3;
      break;
    case 3:
      Nl = 0;
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "This example only supports 4 ranks");
    }
    PetscCall(PetscSFSetGraph(sf, 9, Nl, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
    PetscCall(DMSetPointSF(*dm, sf));
    PetscCall(PetscSFDestroy(&sf));
  }
  // Create fault label
  PetscCall(DMCreateLabel(*dm, "fault"));
  PetscCall(DMGetLabel(*dm, "fault", &label));
  switch (rank) {
  case 0:
  case 2:
    PetscCall(DMLabelSetValue(label, 8, 1));
    PetscCall(DMLabelSetValue(label, 2, 0));
    PetscCall(DMLabelSetValue(label, 4, 0));
    break;
  case 1:
  case 3:
    PetscCall(DMLabelSetValue(label, 7, 1));
    PetscCall(DMLabelSetValue(label, 1, 0));
    PetscCall(DMLabelSetValue(label, 3, 0));
    break;
  default:
    break;
  }
  PetscCall(DMPlexOrientLabel(*dm, label));
  PetscCall(DMPlexLabelCohesiveComplete(*dm, label, NULL, 1, PETSC_FALSE, PETSC_FALSE, NULL));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  if (user->testNum) {
    PetscCall(CreateQuadMesh1(comm, user, dm));
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
  }
  PetscCall(DMSetFromOptions(*dm));
  {
    const char *prefix;

    // We cannot redistribute with cohesive cells in the SF
    PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)*dm, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "f0_"));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "f1_"));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, prefix));
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: triangle
    args: -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail

    test:
      suffix: tri_0
      args: -dm_plex_box_faces 1,1 -dm_plex_cohesive_label_fault 8
    test:
      suffix: tri_1
      args: -dm_plex_box_faces 1,1 -dm_plex_cohesive_label_fault 8 \
              -dm_plex_transform_extrude_use_tensor 0
    test:
      suffix: tri_2
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15
    test:
      suffix: tri_3
      nsize: 2
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15 \
              -petscpartitioner_type shell -petscpartitioner_shell_sizes 2,2 \
              -petscpartitioner_shell_points 0,3,1,2

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,1 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_cohesive_label_fault 13 \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail

    test:
      suffix: quad_0
    test:
      suffix: quad_1
      args: -dm_plex_transform_extrude_use_tensor 0
    test:
      suffix: quad_2
      nsize: 2
      args: -petscpartitioner_type simple

  test:
    suffix: quad_3
    nsize: 4
    args: -test_num 1 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -orientation_view -orientation_view_synchronized

  test:
    suffix: quad_4
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_cohesive_label_fault 22,23 \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail

  test:
    suffix: quad_5
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
            -dm_plex_cohesive_label_fault0 21 \
            -dm_plex_cohesive_label_fault1 23 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
          -dm_view ::ascii_info_detail

  test:
    suffix: quad_6
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
            -dm_plex_cohesive_label_fault0 22,23 \
            -dm_plex_cohesive_label_fault1 32 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
          -dm_view ::ascii_info_detail

TEST*/
