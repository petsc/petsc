
static char help[] = "Tests publishing and receiving back values.\n\n";

#include <petscsys.h>
#include <petscviewersaws.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  PetscInt       anint[2]    = {9,10};
  PetscReal      areal[3]    = {10.0,11.1,12.2};
  PetscBool      abool[4]    = {PETSC_FALSE,PETSC_TRUE,PETSC_FALSE,PETSC_TRUE};
  char           *astring[2];

  PetscInitialize(&argc,&argv,(char*)0,help);

  viewer = PETSC_VIEWER_SAWS_WORLD;

  astring[0] = (char*) "astring0";
  astring[1] = (char*) "astring1";
  PetscStackCallSAWs(SAWs_Register,("/anint",anint,2,SAWs_WRITE,SAWs_INT));
  PetscStackCallSAWs(SAWs_Register,("/areal",areal,3,SAWs_WRITE,SAWs_DOUBLE));
  PetscStackCallSAWs(SAWs_Register,("/abool",abool,4,SAWs_WRITE,SAWs_BOOLEAN));
  PetscStackCallSAWs(SAWs_Register,("/astring",astring,2,SAWs_WRITE,SAWs_STRING));
  ierr = PetscSleep(20);CHKERRQ(ierr);

  PetscStackCallSAWs(SAWs_Lock,());
  ierr = PetscPrintf(PETSC_COMM_WORLD,"anint %d\n",(int)anint[0]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"areal %g\n",(double)areal[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"abool %d\n",(int)abool[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"astring %s\n",astring[1]);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
