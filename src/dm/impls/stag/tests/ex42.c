static char help[] = "Test DMCreateFieldDecomposition_Stag()\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM         dm;
  DM        *sub_dms;
  PetscInt   dim,n_fields;
  IS        *fields;
  char     **field_names;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));

  switch (dim) {
    case 1:
      PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm));
      break;
    case 2:
      PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
      break;
    case 3:
      PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %" PetscInt_FMT,dim);
  }

  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMCreateFieldDecomposition(dm,&n_fields,&field_names,&fields,&sub_dms));
  for (PetscInt i=0; i<n_fields; ++i) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %s\n", i, field_names[i]));
    PetscCall(ISView(fields[i],PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMView(sub_dms[i],PETSC_VIEWER_STDOUT_WORLD));
  }

  for (PetscInt i=0; i<n_fields; ++i) {
    PetscCall(PetscFree(field_names[i]));
    PetscCall(ISDestroy(&fields[i]));
    PetscCall(DMDestroy(&sub_dms[i]));
  }
  PetscCall(PetscFree(fields));
  PetscCall(PetscFree(field_names));
  PetscCall(PetscFree(sub_dms));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args: -dim {{1 2 3}separate output}

TEST*/
