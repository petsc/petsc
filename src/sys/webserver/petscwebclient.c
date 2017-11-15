
static char help[] = "Tests publishing and receiving back values.\n\n";

#include <petscsys.h>
#include <petscviewerams.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AMS_Memory     amem;
  AMS_Comm       acomm;
  PetscViewer    viewer;
  PetscInt       anint[2]    = {9,10};
  PetscReal      areal[3]    = {10.0,11.1,12.2};
  PetscBool      abool[4]    = {PETSC_FALSE,PETSC_TRUE,PETSC_FALSE,PETSC_TRUE};
  char           *astring[2];

  PetscInitialize(&argc,&argv,(char*)0,help);

  viewer = PETSC_VIEWER_AMS_WORLD;
  ierr = PetscViewerAMSGetAMSComm(viewer,&acomm);CHKERRQ(ierr);
  PetscStackCallAMS(AMS_Memory_create,(acomm,"My Memory",&amem));

  astring[0] = (char*) "astring0";
  astring[1] = (char*) "astring1";
  PetscStackCallAMS(AMS_Memory_take_access,(amem));
  PetscStackCallAMS(AMS_Memory_add_field,(amem,"anint",anint,2,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF));
  PetscStackCallAMS(AMS_Memory_add_field,(amem,"areal",areal,3,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF));
  PetscStackCallAMS(AMS_Memory_add_field,(amem,"abool",abool,4,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF));
  PetscStackCallAMS(AMS_Memory_add_field,(amem,"astring",astring,2,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF));
  PetscStackCallAMS(AMS_Memory_publish,(amem));
  PetscStackCallAMS(AMS_Memory_grant_access,(amem));
  ierr = PetscSleep(20);CHKERRQ(ierr);

  PetscStackCallAMS(AMS_Memory_take_access,(amem));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"anint %d\n",(int)anint[0]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"areal %g\n",(double)areal[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"abool %d\n",(int)abool[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"astring %s\n",astring[1]);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
