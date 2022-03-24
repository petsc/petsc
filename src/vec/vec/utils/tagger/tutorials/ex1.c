
static char help[] = "VecTagger interface routines.\n\n";

#include <petscvec.h>

static PetscErrorCode ISGetBlockGlobalIS(IS is, Vec vec, PetscInt bs, IS *isBlockGlobal)
{
  const PetscInt *idxin;
  PetscInt       *idxout, i, n, rstart;
  PetscLayout    map;

  PetscFunctionBegin;

  CHKERRQ(VecGetLayout(vec,&map));
  rstart = map->rstart / bs;
  CHKERRQ(ISGetLocalSize(is, &n));
  CHKERRQ(PetscMalloc1(n, &idxout));
  CHKERRQ(ISGetIndices(is, &idxin));
  for (i = 0; i < n; i++) idxout[i] = rstart + idxin[i];
  CHKERRQ(ISRestoreIndices(is, &idxin));
  CHKERRQ(ISCreateBlock(PetscObjectComm((PetscObject)vec),bs,n,idxout,PETSC_OWN_POINTER,isBlockGlobal));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  Vec            vec, tagged, untagged;
  VecScatter     taggedScatter, untaggedScatter;
  PetscInt       bs;
  PetscInt       n, nloc, nint, i, j, k, localStart, localEnd, ntagged, nuntagged;
  MPI_Comm       comm;
  VecTagger      tagger;
  PetscScalar    *array;
  PetscRandom    rand;
  VecTaggerBox   *defaultBox;
  VecTaggerBox   *boxes;
  IS             is, isBlockGlobal, isComp;
  PetscErrorCode ierr;
  PetscBool      listed;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  n    = 10.;
  bs   = 1;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "" , "VecTagger Test Options", "Vec");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-bs","The block size of the vector","ex1.c",bs,&bs,NULL));
  CHKERRQ(PetscOptionsInt("-n","The size of the vector (in blocks)","ex1.c",n,&n,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscRandomCreate(comm,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));

  CHKERRQ(VecCreate(comm,&vec));
  CHKERRQ(PetscObjectSetName((PetscObject)vec,"Vec to Tag"));
  CHKERRQ(VecSetBlockSize(vec,bs));
  CHKERRQ(VecSetSizes(vec,PETSC_DECIDE,n));
  CHKERRQ(VecSetUp(vec));
  CHKERRQ(VecGetLocalSize(vec,&nloc));
  CHKERRQ(VecGetArray(vec,&array));
  for (i = 0; i < nloc; i++) {
    PetscScalar val;

    CHKERRQ(PetscRandomGetValue(rand,&val));
    array[i] = val;
  }
  CHKERRQ(VecRestoreArray(vec,&array));
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(VecViewFromOptions(vec,NULL,"-vec_view"));

  CHKERRQ(VecTaggerCreate(comm,&tagger));
  CHKERRQ(VecTaggerSetBlockSize(tagger,bs));
  CHKERRQ(VecTaggerSetType(tagger,VECTAGGERABSOLUTE));
  CHKERRQ(PetscMalloc1(bs,&defaultBox));
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    defaultBox[i].min = 0.1;
    defaultBox[i].max = 1.5;
#else
    defaultBox[i].min = PetscCMPLX(0.1,0.1);
    defaultBox[i].max = PetscCMPLX(1.5,1.5);
#endif
  }
  CHKERRQ(VecTaggerAbsoluteSetBox(tagger,defaultBox));
  CHKERRQ(PetscFree(defaultBox));
  CHKERRQ(VecTaggerSetFromOptions(tagger));
  CHKERRQ(VecTaggerSetUp(tagger));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)tagger,NULL,"-vec_tagger_view"));
  CHKERRQ(VecTaggerGetBlockSize(tagger,&bs));

  CHKERRQ(VecTaggerComputeBoxes(tagger,vec,&nint,&boxes,&listed));
  if (listed) {
    PetscViewer viewer = NULL;

    CHKERRQ(PetscOptionsGetViewer(comm,NULL,NULL,"-vec_tagger_boxes_view",&viewer,NULL,NULL));
    if (viewer) {
      PetscBool iascii;

      CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
      if (iascii) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"Num boxes: %" PetscInt_FMT "\n",nint));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        for (i = 0, k = 0; i < nint; i++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT ": ",i));
          for (j = 0; j < bs; j++, k++) {
            if (j) CHKERRQ(PetscViewerASCIIPrintf(viewer," x "));
#if !defined(PETSC_USE_COMPLEX)
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"[%g,%g]",(double)boxes[k].min,(double)boxes[k].max));
#else
            CHKERRQ(PetscViewerASCIIPrintf(viewer,"[%g+%gi,%g+%gi]",(double)PetscRealPart(boxes[k].min),(double)PetscImaginaryPart(boxes[k].min),(double)PetscRealPart(boxes[k].max),(double)PetscImaginaryPart(boxes[k].max)));
#endif
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
        }
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
      }
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(PetscFree(boxes));
  }

  CHKERRQ(VecTaggerComputeIS(tagger,vec,&is,&listed));
  CHKERRQ(ISGetBlockGlobalIS(is,vec,bs,&isBlockGlobal));
  CHKERRQ(PetscObjectSetName((PetscObject)isBlockGlobal,"Tagged IS (block global)"));
  CHKERRQ(ISViewFromOptions(isBlockGlobal,NULL,"-tagged_is_view"));

  CHKERRQ(VecGetOwnershipRange(vec,&localStart,&localEnd));
  CHKERRQ(ISComplement(isBlockGlobal,localStart,localEnd,&isComp));
  CHKERRQ(PetscObjectSetName((PetscObject)isComp,"Untagged IS (global)"));
  CHKERRQ(ISViewFromOptions(isComp,NULL,"-untagged_is_view"));

  CHKERRQ(ISGetLocalSize(isBlockGlobal,&ntagged));
  CHKERRQ(ISGetLocalSize(isComp,&nuntagged));

  CHKERRQ(VecCreate(comm,&tagged));
  CHKERRQ(PetscObjectSetName((PetscObject)tagged,"Tagged selection"));
  CHKERRQ(VecSetSizes(tagged,ntagged,PETSC_DETERMINE));
  CHKERRQ(VecSetUp(tagged));

  CHKERRQ(VecCreate(comm,&untagged));
  CHKERRQ(PetscObjectSetName((PetscObject)untagged,"Untagged selection"));
  CHKERRQ(VecSetSizes(untagged,nuntagged,PETSC_DETERMINE));
  CHKERRQ(VecSetUp(untagged));

  CHKERRQ(VecScatterCreate(vec,isBlockGlobal,tagged,NULL,&taggedScatter));
  CHKERRQ(VecScatterCreate(vec,isComp,untagged,NULL,&untaggedScatter));

  CHKERRQ(VecScatterBegin(taggedScatter,vec,tagged,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(taggedScatter,vec,tagged,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(untaggedScatter,vec,untagged,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(untaggedScatter,vec,untagged,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecViewFromOptions(tagged,NULL,"-tagged_vec_view"));
  CHKERRQ(VecViewFromOptions(untagged,NULL,"-untagged_vec_view"));

  CHKERRQ(VecScatterDestroy(&untaggedScatter));
  CHKERRQ(VecScatterDestroy(&taggedScatter));

  CHKERRQ(VecDestroy(&untagged));
  CHKERRQ(VecDestroy(&tagged));
  CHKERRQ(ISDestroy(&isComp));
  CHKERRQ(ISDestroy(&isBlockGlobal));
  CHKERRQ(ISDestroy(&is));

  CHKERRQ(VecTaggerDestroy(&tagger));
  CHKERRQ(VecDestroy(&vec));
  CHKERRQ(PetscFinalize());
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
