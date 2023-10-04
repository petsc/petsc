const char help[] = "Test PetscOptionsGetViewers()";

#include <petscviewer.h>

#define N_MAX 5

int main(int argc, char **argv)
{
  PetscInt          n_max = N_MAX;
  PetscViewer       viewers[N_MAX];
  PetscViewerFormat formats[N_MAX];

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetViewers(PETSC_COMM_WORLD, NULL, NULL, "-test_view", &n_max, viewers, formats, NULL));
  for (PetscInt i = 0; i < n_max; i++) {
    PetscCall(PetscViewerPushFormat(viewers[i], formats[i]));
    PetscCall(PetscViewerASCIIPrintf(viewers[i], "This is viewer %d\n", (int)i));
    PetscCall(PetscViewerPopFormat(viewers[i]));
    PetscCall(PetscViewerDestroy(&viewers[i]));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -test_view ascii:viewer0.txt,ascii:viewer1.txt
    filter: cat viewer0.txt viewer1.txt

  test:
    suffix: 1
    args: -test_view ,,,ascii:viewer3.txt,

  test:
    suffix: 2
    args: -test_view ,,,ascii:viewer3.txt,
    filter: cat viewer3.txt

TEST*/
