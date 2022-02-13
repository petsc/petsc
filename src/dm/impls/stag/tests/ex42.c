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
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);

  switch (dim) {
    case 1:
      ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
  }

  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMView(dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMCreateFieldDecomposition(dm,&n_fields,&field_names,&fields,&sub_dms);CHKERRQ(ierr);
  for (PetscInt i=0; i<n_fields; ++i) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %s\n", i, field_names[i]);
    ierr = ISView(fields[i],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMView(sub_dms[i],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  for (PetscInt i=0; i<n_fields; ++i) {
    ierr = PetscFree(field_names[i]);
    ierr = ISDestroy(&fields[i]);CHKERRQ(ierr);
    ierr = DMDestroy(&sub_dms[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(fields);CHKERRQ(ierr);
  ierr = PetscFree(field_names);CHKERRQ(ierr);
  ierr = PetscFree(sub_dms);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -dim {{1 2 3}separate output}

TEST*/
