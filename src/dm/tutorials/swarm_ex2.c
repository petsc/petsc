
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
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms));
  PetscCall(DMSetType(dms,DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject) dms, "Particles"));
  PetscCall(DMSwarmInitializeFieldRegister(dms));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"viscosity",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms,"strain",3,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms));
  PetscCall(DMSwarmSetLocalSizes(dms,5+rank,4));
  PetscCall(DMView(dms,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSwarmGetLocalSize(dms,&nlocal));

  {
    PetscReal *array;
    PetscCall(DMSwarmGetField(dms,"viscosity",&bs,NULL,(void**)&array));
    for (p=0; p<nlocal; p++) {
      array[p] = 11.1 + p*0.1 + rank*100.0;
    }
    PetscCall(DMSwarmRestoreField(dms,"viscosity",&bs,NULL,(void**)&array));
  }

  {
    PetscReal *array;
    PetscCall(DMSwarmGetField(dms,"strain",&bs,NULL,(void**)&array));
    for (p=0; p<nlocal; p++) {
      array[bs*p+0] = 2.0e-2 + p*0.001 + rank*1.0;
      array[bs*p+1] = 2.0e-2 + p*0.002 + rank*1.0;
      array[bs*p+2] = 2.0e-2 + p*0.003 + rank*1.0;
    }
    PetscCall(DMSwarmRestoreField(dms,"strain",&bs,NULL,(void**)&array));
  }

  PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"viscosity",&x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"viscosity",&x));

  PetscCall(DMSwarmCreateGlobalVectorFromField(dms,"strain",&x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dms,"strain",&x));

  PetscCall(DMSwarmVectorDefineField(dms,"strain"));
  PetscCall(DMCreateGlobalVector(dms,&x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dms));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(ex2_1());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: !complex double
      nsize: 3
      filter: grep -v atomic
      filter_output: grep -v atomic

TEST*/
