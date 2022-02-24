static char help[] = "Test DMCreateFieldDecomposition_Stag()\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  DM             *sub_dms;
  PetscInt       dim,n_fields;
  IS             *fields;
  char           **field_names;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));

  switch (dim) {
    case 1:
      CHKERRQ(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm));
      break;
    case 2:
      CHKERRQ(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
      break;
    case 3:
      CHKERRQ(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
  }

  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(DMCreateFieldDecomposition(dm,&n_fields,&field_names,&fields,&sub_dms));
  for (PetscInt i=0; i<n_fields; ++i) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %s\n", i, field_names[i]);
    CHKERRQ(ISView(fields[i],PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(DMView(sub_dms[i],PETSC_VIEWER_STDOUT_WORLD));
  }

  for (PetscInt i=0; i<n_fields; ++i) {
    ierr = PetscFree(field_names[i]);
    CHKERRQ(ISDestroy(&fields[i]));
    CHKERRQ(DMDestroy(&sub_dms[i]));
  }
  CHKERRQ(PetscFree(fields));
  CHKERRQ(PetscFree(field_names));
  CHKERRQ(PetscFree(sub_dms));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -dim {{1 2 3}separate output}

TEST*/
