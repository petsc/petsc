static char help[] = "Test parallel ruotines for GLVis\n\n";

#include <petscdmshell.h>
#include <petsc/private/glvisvecimpl.h>

PetscErrorCode VecView_Shell(Vec v, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscBool         isglvis, isascii;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isglvis) {
    DM dm;

    PetscCall(VecGetDM(v, &dm));
    /* DMView() cannot be tested, as DMView_Shell defaults to VecView */
    if (!dm) PetscFunctionReturn(0);
    PetscCall(VecView_GLVis(v, viewer));
  } else if (isascii) {
    const char *name;
    PetscInt    n;

    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(PetscObjectGetName((PetscObject)v, &name));
    if (!PetscGlobalRank) PetscCall(PetscViewerASCIIPrintf(viewer, "Hello from rank 0 -> vector name %s, size %" PetscInt_FMT "\n", name, n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUpGLVisViewer_Shell(PetscObject odm, PetscViewer viewer)
{
  DM          dm = (DM)odm;
  Vec         V;
  PetscInt    dim      = 2;
  const char *fec_type = {"testme"};

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(dm, &V));
  PetscCall(PetscObjectSetName((PetscObject)V, "sample"));
  PetscCall(PetscViewerGLVisSetFields(viewer, 1, &fec_type, &dim, NULL, (PetscObject *)&V, NULL, NULL));
  PetscCall(VecDestroy(&V));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM  dm;
  Vec v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMShellCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMSetUpGLVisViewer_C", DMSetUpGLVisViewer_Shell));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DECIDE, &v));
  PetscCall(PetscObjectSetName((PetscObject)v, "seed"));
  PetscCall(VecSetOperation(v, VECOP_VIEW, (void (*)(void))VecView_Shell));
  PetscCall(DMShellSetGlobalVector(dm, v));
  PetscCall(VecDestroy(&v));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetGlobalVector(dm, &v));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));
  PetscCall(DMRestoreGlobalVector(dm, &v));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMSetUpGLVisViewer_C", NULL));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: glvis_par
    nsize: {{1 2}}
    args: -dm_view glvis: -vec_view glvis:
    output_file: output/ex8_glvis.out

TEST*/
