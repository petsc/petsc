
static char help[] = "VecTagger interface routines.\n\n";

#include <petscvec.h>

static PetscErrorCode ISGetBlockGlobalIS(IS is, Vec vec, PetscInt bs, IS *isBlockGlobal)
{
  const PetscInt *idxin;
  PetscInt       *idxout, i, n, rstart;
  PetscLayout    map;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecGetLayout(vec,&map);CHKERRQ(ierr);
  rstart = map->rstart / bs;
  ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &idxout);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &idxin);CHKERRQ(ierr);
  for (i = 0; i < n; i++) idxout[i] = rstart + idxin[i];
  ierr = ISRestoreIndices(is, &idxin);CHKERRQ(ierr);
  ierr = ISCreateBlock(PetscObjectComm((PetscObject)vec),bs,n,idxout,PETSC_OWN_POINTER,isBlockGlobal);CHKERRQ(ierr);
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

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  n    = 10.;
  bs   = 1;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "" , "VecTagger Test Options", "Vec");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-bs","The block size of the vector","ex1.c",bs,&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","The size of the vector (in blocks)","ex1.c",n,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = VecCreate(comm,&vec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec,"Vec to Tag");CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
  ierr = VecSetSizes(vec,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetUp(vec);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nloc);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  for (i = 0; i < nloc; i++) {
    PetscScalar val;

    ierr = PetscRandomGetValue(rand,&val);CHKERRQ(ierr);
    array[i] = val;
  }
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec,NULL,"-vec_view");CHKERRQ(ierr);

  ierr = VecTaggerCreate(comm,&tagger);CHKERRQ(ierr);
  ierr = VecTaggerSetBlockSize(tagger,bs);CHKERRQ(ierr);
  ierr = VecTaggerSetType(tagger,VECTAGGERABSOLUTE);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&defaultBox);CHKERRQ(ierr);
  for (i = 0; i < bs; i++) {
#if !defined(PETSC_USE_COMPLEX)
    defaultBox[i].min = 0.1;
    defaultBox[i].max = 1.5;
#else
    defaultBox[i].min = PetscCMPLX(0.1,0.1);
    defaultBox[i].max = PetscCMPLX(1.5,1.5);
#endif
  }
  ierr = VecTaggerAbsoluteSetBox(tagger,defaultBox);CHKERRQ(ierr);
  ierr = PetscFree(defaultBox);CHKERRQ(ierr);
  ierr = VecTaggerSetFromOptions(tagger);CHKERRQ(ierr);
  ierr = VecTaggerSetUp(tagger);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)tagger,NULL,"-vec_tagger_view");CHKERRQ(ierr);
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);

  ierr = VecTaggerComputeBoxes(tagger,vec,&nint,&boxes,&listed);CHKERRQ(ierr);
  if (listed) {
    PetscViewer viewer = NULL;

    ierr = PetscOptionsGetViewer(comm,NULL,NULL,"-vec_tagger_boxes_view",&viewer,NULL,NULL);CHKERRQ(ierr);
    if (viewer) {
      PetscBool iascii;

      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
      if (iascii) {
        ierr = PetscViewerASCIIPrintf(viewer,"Num boxes: %D\n",nint);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        for (i = 0, k = 0; i < nint; i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%D: ",i);CHKERRQ(ierr);
          for (j = 0; j < bs; j++, k++) {
            if (j) {ierr = PetscViewerASCIIPrintf(viewer," x ");CHKERRQ(ierr);}
#if !defined(PETSC_USE_COMPLEX)
            ierr = PetscViewerASCIIPrintf(viewer,"[%g,%g]",(double)boxes[k].min,(double)boxes[k].max);CHKERRQ(ierr);
#else
            ierr = PetscViewerASCIIPrintf(viewer,"[%g+%gi,%g+%gi]",(double)PetscRealPart(boxes[k].min),(double)PetscImaginaryPart(boxes[k].min),(double)PetscRealPart(boxes[k].max),(double)PetscImaginaryPart(boxes[k].max));CHKERRQ(ierr);
#endif
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscFree(boxes);CHKERRQ(ierr);
  }

  ierr = VecTaggerComputeIS(tagger,vec,&is,&listed);CHKERRQ(ierr);
  ierr = ISGetBlockGlobalIS(is,vec,bs,&isBlockGlobal);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)isBlockGlobal,"Tagged IS (block global)");CHKERRQ(ierr);
  ierr = ISViewFromOptions(isBlockGlobal,NULL,"-tagged_is_view");CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(vec,&localStart,&localEnd);CHKERRQ(ierr);
  ierr = ISComplement(isBlockGlobal,localStart,localEnd,&isComp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)isComp,"Untagged IS (global)");CHKERRQ(ierr);
  ierr = ISViewFromOptions(isComp,NULL,"-untagged_is_view");CHKERRQ(ierr);

  ierr = ISGetLocalSize(isBlockGlobal,&ntagged);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isComp,&nuntagged);CHKERRQ(ierr);

  ierr = VecCreate(comm,&tagged);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)tagged,"Tagged selection");CHKERRQ(ierr);
  ierr = VecSetSizes(tagged,ntagged,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(tagged);CHKERRQ(ierr);

  ierr = VecCreate(comm,&untagged);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)untagged,"Untagged selection");CHKERRQ(ierr);
  ierr = VecSetSizes(untagged,nuntagged,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(untagged);CHKERRQ(ierr);

  ierr = VecScatterCreate(vec,isBlockGlobal,tagged,NULL,&taggedScatter);CHKERRQ(ierr);
  ierr = VecScatterCreate(vec,isComp,untagged,NULL,&untaggedScatter);CHKERRQ(ierr);

  ierr = VecScatterBegin(taggedScatter,vec,tagged,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(taggedScatter,vec,tagged,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(untaggedScatter,vec,untagged,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(untaggedScatter,vec,untagged,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecViewFromOptions(tagged,NULL,"-tagged_vec_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(untagged,NULL,"-untagged_vec_view");CHKERRQ(ierr);

  ierr = VecScatterDestroy(&untaggedScatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&taggedScatter);CHKERRQ(ierr);

  ierr = VecDestroy(&untagged);CHKERRQ(ierr);
  ierr = VecDestroy(&tagged);CHKERRQ(ierr);
  ierr = ISDestroy(&isComp);CHKERRQ(ierr);
  ierr = ISDestroy(&isBlockGlobal);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = VecTaggerDestroy(&tagger);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
