
static char help[] = "VecTagger interface routines.\n\n";

/*T
   Processors: n
T*/

#include <petscis.h>
#include <petscvec.h>

static PetscErrorCode ISGetBlockGlobalIS(IS is, Vec vec, PetscInt bs, IS *isBlockGlobal)
{
  PetscLayout    layout;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLayout(vec, &layout);CHKERRQ(ierr);
  if (bs == 1) {
    if (layout->mapping) {
      ierr = ISLocalToGlobalMappingApplyIS(layout->mapping,is,isBlockGlobal);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
      *isBlockGlobal = is;
    }
  } else {
    const PetscInt *idxin;
    PetscInt       *idxout, i, n, rstart = layout->rstart / bs;

    ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &idxout);CHKERRQ(ierr);
    ierr = ISGetIndices(is, &idxin);CHKERRQ(ierr);
    for (i = 0; i < n; i++) idxout[i] = rstart + idxin[i];
    ierr = ISRestoreIndices(is, &idxin);CHKERRQ(ierr);
    ierr = ISCreateBlock(PetscObjectComm((PetscObject)vec),bs,n,idxout,PETSC_OWN_POINTER,isBlockGlobal);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  Vec            vec, tagged, untagged;
  VecScatter     taggedScatter, untaggedScatter;
  PetscInt       bs;
  PetscInt       n, N, nloc, nint, i, j, k, localStart, localEnd, ntagged, nuntagged;
  MPI_Comm       comm;
  VecTagger      tagger;
  PetscScalar    *array;
  PetscRandom    rand;
  PetscScalar    (*defaultInterval)[2];
  PetscScalar    (*intervals)[2];
  IS             is, isBlockGlobal, isComp;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  n    = 10.;
  bs   = 1;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "" , "VecTagger Test Options", "Vec");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-bs","The block size of the vector","ex1.c",bs,&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","The size of the vector (in blocks)","ex1.c",n,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  N    = n * bs;
  ierr = VecCreate(comm,&vec);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
  ierr = VecSetSizes(vec,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetUp(vec);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

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
  ierr = PetscMalloc1(bs,&defaultInterval);CHKERRQ(ierr);
  for (i = 0; i < bs; i++) {
    defaultInterval[i][0] = 0.1;
    defaultInterval[i][1] = 1.5;
  }
  ierr = VecTaggerAbsoluteSetInterval(tagger,defaultInterval);CHKERRQ(ierr);
  ierr = PetscFree(defaultInterval);CHKERRQ(ierr);
  ierr = VecTaggerSetFromOptions(tagger);CHKERRQ(ierr);
  ierr = VecTaggerSetUp(tagger);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)tagger,NULL,"-vec_tagger_view");CHKERRQ(ierr);
  ierr = VecTaggerGetBlockSize(tagger,&bs);CHKERRQ(ierr);

  ierr = VecTaggerComputeIntervals(tagger,vec,&nint,&intervals);
  if (ierr && ierr != PETSC_ERR_SUP) CHKERRQ(ierr);
  else {
    PetscViewer viewer = NULL;

    ierr = PetscOptionsGetViewer(comm,NULL,"-vec_tagger_intervals_view",&viewer,NULL,NULL);CHKERRQ(ierr);
    if (viewer) {
      PetscBool iascii;

      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
      if (iascii) {
        ierr = PetscViewerASCIIPrintf(viewer,"Num intervals: %D\n",nint);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        for (i = 0, k = 0; i < nint; i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%D: ",i);CHKERRQ(ierr);
          for (j = 0; j < bs; j++, k++) {
            if (j) {ierr = PetscViewerASCIIPrintf(viewer," x ");CHKERRQ(ierr);}
#if !defined(PETSC_USE_COMPLEX)
            ierr = PetscViewerASCIIPrintf(viewer,"[%g,%g]",(double)intervals[k][0],(double)intervals[k][1]);CHKERRQ(ierr);
#endif
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscFree(intervals);CHKERRQ(ierr);
  }

  ierr = VecTaggerComputeIS(tagger,vec,&is);CHKERRQ(ierr);
  ierr = ISGetBlockGlobalIS(is,vec,bs,&isBlockGlobal);CHKERRQ(ierr);
  ierr = ISViewFromOptions(isBlockGlobal,NULL,"-tagged_is_view");CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(vec,&localStart,&localEnd);CHKERRQ(ierr);
  ierr = ISComplement(isBlockGlobal,localStart,localEnd,&isComp);CHKERRQ(ierr);
  ierr = ISViewFromOptions(isComp,NULL,"-untagged_is_view");CHKERRQ(ierr);

  ierr = ISGetLocalSize(isBlockGlobal,&ntagged);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isComp,&nuntagged);CHKERRQ(ierr);

  ierr = VecCreate(comm,&tagged);CHKERRQ(ierr);
  ierr = VecSetSizes(tagged,ntagged,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(tagged);CHKERRQ(ierr);

  ierr = VecCreate(comm,&untagged);CHKERRQ(ierr);
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
