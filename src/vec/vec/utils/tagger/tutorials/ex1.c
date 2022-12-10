
static char help[] = "VecTagger interface routines.\n\n";

#include <petscvec.h>

static PetscErrorCode ISGetBlockGlobalIS(IS is, Vec vec, PetscInt bs, IS *isBlockGlobal)
{
  const PetscInt *idxin;
  PetscInt       *idxout, i, n, rstart;
  PetscLayout     map;

  PetscFunctionBegin;

  PetscCall(VecGetLayout(vec, &map));
  rstart = map->rstart / bs;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(PetscMalloc1(n, &idxout));
  PetscCall(ISGetIndices(is, &idxin));
  for (i = 0; i < n; i++) idxout[i] = rstart + idxin[i];
  PetscCall(ISRestoreIndices(is, &idxin));
  PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)vec), bs, n, idxout, PETSC_OWN_POINTER, isBlockGlobal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec           vec, tagged, untagged;
  VecScatter    taggedScatter, untaggedScatter;
  PetscInt      bs;
  PetscInt      n, nloc, nint, i, j, k, localStart, localEnd, ntagged, nuntagged;
  MPI_Comm      comm;
  VecTagger     tagger;
  PetscScalar  *array;
  PetscRandom   rand;
  VecTaggerBox *defaultBox;
  VecTaggerBox *boxes;
  IS            is, isBlockGlobal, isComp;
  PetscBool     listed;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  n    = 10.;
  bs   = 1;
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, "", "VecTagger Test Options", "Vec");
  PetscCall(PetscOptionsInt("-bs", "The block size of the vector", "ex1.c", bs, &bs, NULL));
  PetscCall(PetscOptionsInt("-n", "The size of the vector (in blocks)", "ex1.c", n, &n, NULL));
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(VecCreate(comm, &vec));
  PetscCall(PetscObjectSetName((PetscObject)vec, "Vec to Tag"));
  PetscCall(VecSetBlockSize(vec, bs));
  PetscCall(VecSetSizes(vec, PETSC_DECIDE, n));
  PetscCall(VecSetUp(vec));
  PetscCall(VecGetLocalSize(vec, &nloc));
  PetscCall(VecGetArray(vec, &array));
  for (i = 0; i < nloc; i++) {
    PetscScalar val;

    PetscCall(PetscRandomGetValue(rand, &val));
    array[i] = val;
  }
  PetscCall(VecRestoreArray(vec, &array));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecViewFromOptions(vec, NULL, "-vec_view"));

  PetscCall(VecTaggerCreate(comm, &tagger));
  PetscCall(VecTaggerSetBlockSize(tagger, bs));
  PetscCall(VecTaggerSetType(tagger, VECTAGGERABSOLUTE));
  PetscCall(PetscMalloc1(bs, &defaultBox));
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    defaultBox[i].min = 0.1;
    defaultBox[i].max = 1.5;
#else
    defaultBox[i].min = PetscCMPLX(0.1, 0.1);
    defaultBox[i].max = PetscCMPLX(1.5, 1.5);
#endif
  }
  PetscCall(VecTaggerAbsoluteSetBox(tagger, defaultBox));
  PetscCall(PetscFree(defaultBox));
  PetscCall(VecTaggerSetFromOptions(tagger));
  PetscCall(VecTaggerSetUp(tagger));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tagger, NULL, "-vec_tagger_view"));
  PetscCall(VecTaggerGetBlockSize(tagger, &bs));

  PetscCall(VecTaggerComputeBoxes(tagger, vec, &nint, &boxes, &listed));
  if (listed) {
    PetscViewer viewer = NULL;

    PetscCall(PetscOptionsGetViewer(comm, NULL, NULL, "-vec_tagger_boxes_view", &viewer, NULL, NULL));
    if (viewer) {
      PetscBool iascii;

      PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
      if (iascii) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Num boxes: %" PetscInt_FMT "\n", nint));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        for (i = 0, k = 0; i < nint; i++) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ": ", i));
          for (j = 0; j < bs; j++, k++) {
            if (j) PetscCall(PetscViewerASCIIPrintf(viewer, " x "));
#if !defined(PETSC_USE_COMPLEX)
            PetscCall(PetscViewerASCIIPrintf(viewer, "[%g,%g]", (double)boxes[k].min, (double)boxes[k].max));
#else
            PetscCall(PetscViewerASCIIPrintf(viewer, "[%g+%gi,%g+%gi]", (double)PetscRealPart(boxes[k].min), (double)PetscImaginaryPart(boxes[k].min), (double)PetscRealPart(boxes[k].max), (double)PetscImaginaryPart(boxes[k].max)));
#endif
          }
          PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
        }
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
    }
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscFree(boxes));
  }

  PetscCall(VecTaggerComputeIS(tagger, vec, &is, &listed));
  PetscCall(ISGetBlockGlobalIS(is, vec, bs, &isBlockGlobal));
  PetscCall(PetscObjectSetName((PetscObject)isBlockGlobal, "Tagged IS (block global)"));
  PetscCall(ISViewFromOptions(isBlockGlobal, NULL, "-tagged_is_view"));

  PetscCall(VecGetOwnershipRange(vec, &localStart, &localEnd));
  PetscCall(ISComplement(isBlockGlobal, localStart, localEnd, &isComp));
  PetscCall(PetscObjectSetName((PetscObject)isComp, "Untagged IS (global)"));
  PetscCall(ISViewFromOptions(isComp, NULL, "-untagged_is_view"));

  PetscCall(ISGetLocalSize(isBlockGlobal, &ntagged));
  PetscCall(ISGetLocalSize(isComp, &nuntagged));

  PetscCall(VecCreate(comm, &tagged));
  PetscCall(PetscObjectSetName((PetscObject)tagged, "Tagged selection"));
  PetscCall(VecSetSizes(tagged, ntagged, PETSC_DETERMINE));
  PetscCall(VecSetUp(tagged));

  PetscCall(VecCreate(comm, &untagged));
  PetscCall(PetscObjectSetName((PetscObject)untagged, "Untagged selection"));
  PetscCall(VecSetSizes(untagged, nuntagged, PETSC_DETERMINE));
  PetscCall(VecSetUp(untagged));

  PetscCall(VecScatterCreate(vec, isBlockGlobal, tagged, NULL, &taggedScatter));
  PetscCall(VecScatterCreate(vec, isComp, untagged, NULL, &untaggedScatter));

  PetscCall(VecScatterBegin(taggedScatter, vec, tagged, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(taggedScatter, vec, tagged, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(untaggedScatter, vec, untagged, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(untaggedScatter, vec, untagged, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecViewFromOptions(tagged, NULL, "-tagged_vec_view"));
  PetscCall(VecViewFromOptions(untagged, NULL, "-untagged_vec_view"));

  PetscCall(VecScatterDestroy(&untaggedScatter));
  PetscCall(VecScatterDestroy(&taggedScatter));

  PetscCall(VecDestroy(&untagged));
  PetscCall(VecDestroy(&tagged));
  PetscCall(ISDestroy(&isComp));
  PetscCall(ISDestroy(&isBlockGlobal));
  PetscCall(ISDestroy(&is));

  PetscCall(VecTaggerDestroy(&tagger));
  PetscCall(VecDestroy(&vec));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: !complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view

  test:
    suffix: 1
    requires: !complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view

  test:
    suffix: 2
    requires: !complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -bs 2

  test:
    suffix: 3
    requires: !complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_block_size 2 -vec_tagger_box 0.1,1.5,0.1,1.5

  test:
    suffix: 4
    requires: !complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_invert

  test:
    suffix: 5
    requires: !complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type relative -vec_tagger_box 0.25,0.75

  test:
    suffix: 6
    requires: !complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type cdf -vec_tagger_box 0.25,0.75

  test:
    suffix: 7
    requires: !complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type cdf -vec_tagger_box 0.25,0.75 -vec_tagger_cdf_method iterative -vec_tagger_cdf_max_it 10

  test:
    suffix: 8
    requires: !complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type or -vec_tagger_num_subs 2 -sub_0_vec_tagger_type absolute -sub_0_vec_tagger_box 0.0,0.25 -sub_1_vec_tagger_type relative -sub_1_vec_tagger_box 0.75,inf
    filter: sed -e s~Inf~inf~g

  test:
    suffix: 9
    requires: !complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type and -vec_tagger_num_subs 2 -sub_0_vec_tagger_type absolute -sub_0_vec_tagger_box -inf,0.5 -sub_1_vec_tagger_type relative -sub_1_vec_tagger_box 0.25,0.75
    filter: sed -e s~Inf~inf~g

  test:
    suffix: 10
    requires: complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view

  test:
    suffix: 11
    requires: complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view

  test:
    suffix: 12
    requires: complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -bs 2

  test:
    suffix: 13
    requires: complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_block_size 2 -vec_tagger_box 0.1+0.1i,1.5+1.5i,0.1+0.1i,1.5+1.5i

  test:
    suffix: 14
    requires: complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_invert

  test:
    suffix: 15
    requires: complex
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type relative -vec_tagger_box 0.25+0.25i,0.75+0.75i

  test:
    suffix: 16
    requires: complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type cdf -vec_tagger_box 0.25+0.25i,0.75+0.75i

  test:
    suffix: 17
    requires: complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type cdf -vec_tagger_box 0.25+0.25i,0.75+0.75i -vec_tagger_cdf_method iterative -vec_tagger_cdf_max_it 10

  test:
    suffix: 18
    requires: complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type or -vec_tagger_num_subs 2 -sub_0_vec_tagger_type absolute -sub_0_vec_tagger_box 0.0+0.0i,0.25+0.25i -sub_1_vec_tagger_type relative -sub_1_vec_tagger_box 0.75+0.75i,inf+infi
    filter: sed -e s~Inf~inf~g

  test:
    suffix: 19
    requires: complex
    nsize: 3
    args: -n 12 -vec_view -vec_tagger_view -vec_tagger_boxes_view -tagged_is_view -untagged_is_view -tagged_vec_view -untagged_vec_view -vec_tagger_type and -vec_tagger_num_subs 2 -sub_0_vec_tagger_type absolute -sub_0_vec_tagger_box -inf-infi,0.75+0.75i -sub_1_vec_tagger_type relative -sub_1_vec_tagger_box 0.25+0.25i,1.+1.i
    filter: sed -e s~Inf~inf~g

TEST*/
