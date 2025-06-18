static char help[] = "Tests PetscLogView() called with no PetscGlobalArgc and PetscGlobalArgs.\n\n";

#include <petscsys.h>
#include <petscvec.h>

int main(int argc, char **args)
{
  Vec         vec;
  PetscBool   flg = PETSC_FALSE;
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(NULL, NULL, NULL, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  for (PetscInt i = 1; i < argc; ++i) {
    PetscCall(PetscStrcmp(args[i], "foo", &flg));
    if (flg) break;
  }
  if (flg) {
    PetscCall(VecCreate(PETSC_COMM_SELF, &vec));
    PetscCall(VecDestroy(&vec));
  }
  PetscCall(PetscLogView(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: defined(PETSC_USE_LOG)
    nsize: 1
    filter: grep -E "^              (Vector|Viewer)"
    test:
      suffix: 1
      output_file: output/ex81_1.out
    test:
      suffix: 2
      args: foo
      output_file: output/ex81_2.out

 TEST*/
