static char help[] = "Test parallel ruotines for GLVis\n\n";

#include <petscdmshell.h>
#include <petsc/private/glvisvecimpl.h>

PetscErrorCode VecView_Shell(Vec v, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscBool         isglvis,isascii;

  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isglvis) {
    DM dm;

    ierr = VecGetDM(v,&dm);CHKERRQ(ierr);
    /* DMView() cannot be tested, as DMView_Shell defaults to VecView */
    if (!dm) PetscFunctionReturn(0);
    ierr = VecView_GLVis(v,viewer);CHKERRQ(ierr);
  } else if (isascii) {
    const char* name;
    PetscInt    n;

    ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)v,&name);CHKERRQ(ierr);
    if (!PetscGlobalRank) {
      ierr = PetscViewerASCIIPrintf(viewer,"Hello from rank 0 -> vector name %s, size %D\n",name,n);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUpGLVisViewer_Shell(PetscObject odm, PetscViewer viewer)
{
  DM             dm = (DM)odm;
  Vec            V;
  PetscInt       dim = 2;
  char           *fec_type[] = { "testme" };
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dm,&V);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)V,"sample");CHKERRQ(ierr);
  ierr = PetscViewerGLVisSetFields(viewer,1,(const char**)fec_type,&dim,NULL,(PetscObject*)&V,NULL,NULL);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            v;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = DMShellCreate(PETSC_COMM_WORLD,&dm);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Shell);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DECIDE,&v);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)v,"seed");CHKERRQ(ierr);
  ierr = VecSetOperation(v,VECOP_VIEW,(void (*)(void))VecView_Shell);CHKERRQ(ierr);
  ierr = DMShellSetGlobalVector(dm,v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&v);CHKERRQ(ierr);
  ierr = VecViewFromOptions(v,NULL,"-vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&v);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: glvis_par
    nsize: {{1 2}}
    args: -dm_view glvis: -vec_view glvis:
    output_file: output/ex8_glvis.out

TEST*/

