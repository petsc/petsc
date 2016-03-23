
static char help[] = "Tests DMSwarm\n\n";

#include <petscdm.h>
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DM dms;
  PetscErrorCode ierr;
  Vec x;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,20,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"viscosity",NULL,NULL,&array);CHKERRQ(ierr);
    array[0] = 11.1;
    array[3] = 33.3;
    ierr = DMSwarmRestoreField(dms,"viscosity",NULL,NULL,&array);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  
  ierr = DMSwarmVectorDefineField(dms,"strain");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dms,&x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}
