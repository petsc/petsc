
static char help[] = "Tests publishing and receiving back values.\n\n";

#include <petscsys.h>
#include <petscviewerams.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AMS_Memory     amem;
  PetscViewer    viewer;
  PetscInt       anint[2]    = {9,10};
  PetscReal      areal[3]    = {10.0,11.1,12.2};
  PetscBool      abool[4]    = {PETSC_FALSE,PETSC_TRUE,PETSC_FALSE,PETSC_TRUE};
  char           *astring[2];

  PetscInitialize(&argc,&argv,(char*)0,help);

  viewer = PETSC_VIEWER_AMS_WORLD;
  PetscStackCallAMS(AMS_Memory_Create,("MyMemory",&amem));

  astring[0] = (char*) "astring0";
  astring[1] = (char*) "astring1";
  PetscStackCallAMS(AMS_New_Field,(amem,"anint",anint,2,AMS_WRITE,AMS_INT));
  PetscStackCallAMS(AMS_New_Field,(amem,"areal",areal,3,AMS_WRITE,AMS_DOUBLE));
  PetscStackCallAMS(AMS_New_Field,(amem,"abool",abool,4,AMS_WRITE,AMS_BOOLEAN));
  PetscStackCallAMS(AMS_New_Field,(amem,"astring",astring,2,AMS_WRITE,AMS_STRING));
  ierr = PetscSleep(20);CHKERRQ(ierr);

  PetscStackCallAMS(AMS_Lock_Memory,(amem));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"anint %d\n",(int)anint[0]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"areal %g\n",(double)areal[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"abool %d\n",(int)abool[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"astring %s\n",astring[1]);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
