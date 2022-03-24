
static char help[] = "Tests DMSwarm\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

/*
 Checks for variable blocksize
*/
PetscErrorCode ex2_1(void)
{
  DM             dms;
  Vec            x;
  PetscMPIInt    rank;
  PetscInt       p,bs,nlocal;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dms));
  CHKERRQ(DMSetType(dms,DMSWARM));
  CHKERRQ(DMSwarmInitializeFieldRegister(dms));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(dms,"strain",3,PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(dms));
  CHKERRQ(DMSwarmSetLocalSizes(dms,5+rank,4));
  CHKERRQ(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMSwarmGetLocalSize(dms,&nlocal));

  {
    PetscReal *array;
    CHKERRQ(DMSwarmGetField(dms,"viscosity",&bs,NULL,(void**)&array));
    for (p=0; p<nlocal; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"viscosity",&bs,NULL,(void**)&array));
  }

  {
    PetscReal *array;
    CHKERRQ(DMSwarmGetField(dms,"strain",&bs,NULL,(void**)&array));
    for (p=0; p<nlocal; p++) {
      array[bs*p+0] = 2.0e-2 + p*0.001 + rank*1.0;
      array[bs*p+1] = 2.0e-2 + p*0.002 + rank*1.0;
      array[bs*p+2] = 2.0e-2 + p*0.003 + rank*1.0;
    }
    CHKERRQ(DMSwarmRestoreField(dms,"strain",&bs,NULL,(void**)&array));
  }

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(dms,"strain",&x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dms,"strain",&x));

  CHKERRQ(DMSwarmVectorDefineField(dms,"strain"));
  CHKERRQ(DMCreateGlobalVector(dms,&x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(ex2_1());
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: !complex double
      nsize: 3
      filter: grep -v atomic
      filter_output: grep -v atomic

TEST*/
