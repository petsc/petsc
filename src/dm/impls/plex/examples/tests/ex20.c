
const char help[] = "Test DMPlex implementation of DMAdaptLabel().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm, dmAdapt;
  DMLabel        adaptLabel;
  PetscInt       dim, nfaces, cStart, cEnd;
  PetscBool      interpolate;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  dim         = 2;
  nfaces      = 3;
  interpolate = PETSC_TRUE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex20",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","domain dimension",NULL,dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nfaces","number of faces per dimension",NULL,nfaces,&nfaces,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,nfaces,interpolate,&dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dm,"Pre Adaptation Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-pre_adapt_dm_view");CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMLabelCreate("adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(adaptLabel,DM_ADAPT_COARSEN);CHKERRQ(ierr);
  if (cEnd > cStart) {ierr = DMLabelSetValue(adaptLabel,cStart,DM_ADAPT_REFINE);CHKERRQ(ierr);}
  ierr = DMAdaptLabel(dm,adaptLabel,&dmAdapt);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dmAdapt,"Post Adaptation Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmAdapt,NULL,"-post_adapt_dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dmAdapt);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d
    requires: triangle
    args: -dim 2 -pre_adapt_dm_view ascii::ascii_info_detail -post_adapt_dm_view ascii::ascii_info_detail
  test:
    suffix: 3d_tetgen
    requires: tetgen complex
    args: -dim 3 -pre_adapt_dm_view ascii::ascii_info_detail -post_adapt_dm_view ascii::ascii_info_detail
  test:
    suffix: 3d_ctetgen
    requires: ctetgen !complex
    args: -dim 3 -pre_adapt_dm_view ascii::ascii_info_detail -post_adapt_dm_view ascii::ascii_info_detail

TEST*/
