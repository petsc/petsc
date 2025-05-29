static char help[] = "Tests PetscLogView() called from different places.\n\n";

#include <petscsys.h>
#include <petscvec.h>

int main(int argc, char **args)
{
  Vec         vec;
  PetscViewer viewer[2];
  PetscBool   flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, viewer));
  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, viewer + 1));
  PetscCall(PetscViewerDestroy(viewer + 1));
  PetscCall(PetscViewerSetType(viewer[0], PETSCVIEWERASCII));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-with_vec", &flg, NULL));
  if (flg) {
    PetscCall(VecCreate(PETSC_COMM_SELF, &vec));
    PetscCall(VecDestroy(&vec));
  }
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-with_options", &flg, NULL));
  if (!flg) {
    PetscCall(PetscLogView(viewer[0]));
    PetscCall(PetscLogView(viewer[0]));
    PetscCall(PetscLogView(viewer[0]));
  } else {
    PetscCall(PetscLogViewFromOptions());
    PetscCall(PetscLogViewFromOptions());
    PetscCall(PetscLogViewFromOptions());
  }
  PetscCall(PetscViewerDestroy(viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: defined(PETSC_USE_LOG)
    nsize: 1
    temporaries: default.log flamegraph.log
    filter: grep Viewer | uniq
    test:
      args: -log_view ascii,:flamegraph.log:ascii_flamegraph,:default.log -with_vec {{false true}shared output}
  # test:
  #   TODO: broken (wrong count when PetscLogViewFromOptions() is called by the user)
  #   args: -log_view ascii,:flamegraph.log:ascii_flamegraph,:default.log -with_vec {{false true}shared output} -with_options

 TEST*/
