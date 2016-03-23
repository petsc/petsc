
static char help[] = "Tests DMSwarm\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DM dms;
  PetscErrorCode ierr;
  Vec x;
  PetscMPIInt rank,commsize;
  PetscInt p;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms,"strain",1,PETSC_REAL);CHKERRQ(ierr);
  
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  
  ierr = DMSwarmSetLocalSizes(dms,5+rank,4);CHKERRQ(ierr);
  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"viscosity",NULL,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<5+rank; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    ierr = DMSwarmRestoreField(dms,"viscosity",NULL,NULL,(void**)&array);CHKERRQ(ierr);
  }

  {
    PetscReal *array;
    ierr = DMSwarmGetField(dms,"strain",NULL,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<5+rank; p++) {
      array[p] = 2.0e-2 + p*0.001 + rank*1.0;
    }
    ierr = DMSwarmRestoreField(dms,"strain",NULL,NULL,(void**)&array);CHKERRQ(ierr);
  }

  ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  //ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  
  ierr = DMSwarmVectorDefineField(dms,"strain");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dms,&x);CHKERRQ(ierr);
  //ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  
  {
    PetscInt *rankval;
    PetscInt npoints[2],npoints_orig[2];
    
    ierr = DMSwarmGetLocalSize(dms,&npoints_orig[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints_orig[1]);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    if ((rank == 0) && (commsize > 1)) {
      rankval[0] = 1;
      rankval[3] = 1;
    }
    if (rank == 3) {
      rankval[2] = 1;
    }
    ierr = DMSwarmRestoreField(dms,"DMSwarm_rank",NULL,NULL,(void**)&rankval);CHKERRQ(ierr);
    
    ierr = DMSwarmMigrate(dms,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dms,&npoints[0]);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(dms,&npoints[1]);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_SELF,"rank[%d] before(%D,%D) after(%D,%D)\n",rank,npoints_orig[0],npoints_orig[1],npoints[0],npoints[1]);
  }
  
  {
    ierr = DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x);CHKERRQ(ierr);
  }
  {
    ierr = DMSwarmCreateGlobalVectorFromField(dms,"strain",&x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(dms,"strain",&x);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}
