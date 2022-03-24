static char help[] = "Test parallel ruotines for GLVis\n\n";

#include <petscdmshell.h>
#include <petsc/private/glvisvecimpl.h>

PetscErrorCode VecView_Shell(Vec v, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscBool         isglvis,isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isglvis) {
    DM dm;

    CHKERRQ(VecGetDM(v,&dm));
    /* DMView() cannot be tested, as DMView_Shell defaults to VecView */
    if (!dm) PetscFunctionReturn(0);
    CHKERRQ(VecView_GLVis(v,viewer));
  } else if (isascii) {
    const char* name;
    PetscInt    n;

    CHKERRQ(VecGetLocalSize(v,&n));
    CHKERRQ(PetscObjectGetName((PetscObject)v,&name));
    if (!PetscGlobalRank) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Hello from rank 0 -> vector name %s, size %D\n",name,n));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUpGLVisViewer_Shell(PetscObject odm, PetscViewer viewer)
{
  DM             dm = (DM)odm;
  Vec            V;
  PetscInt       dim = 2;
  const char     *fec_type = { "testme" };

  PetscFunctionBegin;
  CHKERRQ(DMCreateGlobalVector(dm,&V));
  CHKERRQ(PetscObjectSetName((PetscObject)V,"sample"));
  CHKERRQ(PetscViewerGLVisSetFields(viewer,1,&fec_type,&dim,NULL,(PetscObject*)&V,NULL,NULL));
  CHKERRQ(VecDestroy(&V));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            v;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(DMShellCreate(PETSC_COMM_WORLD,&dm));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Shell));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DECIDE,&v));
  CHKERRQ(PetscObjectSetName((PetscObject)v,"seed"));
  CHKERRQ(VecSetOperation(v,VECOP_VIEW,(void (*)(void))VecView_Shell));
  CHKERRQ(DMShellSetGlobalVector(dm,v));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(DMViewFromOptions(dm,NULL,"-dm_view"));
  CHKERRQ(DMGetGlobalVector(dm,&v));
  CHKERRQ(VecViewFromOptions(v,NULL,"-vec_view"));
  CHKERRQ(DMRestoreGlobalVector(dm,&v));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: glvis_par
    nsize: {{1 2}}
    args: -dm_view glvis: -vec_view glvis:
    output_file: output/ex8_glvis.out

TEST*/
