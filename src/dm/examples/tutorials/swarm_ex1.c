
static char help[] = "Tests DMSwarm\n\n";

#include <petscdm.h>
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DM dms;
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,20,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}
