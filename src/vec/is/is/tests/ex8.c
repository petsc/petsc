
static char help[] = "Tests ISLocalToGlobalMappingSetBlockSize.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt               indices[] = {0,1,2,3,-1,-1,-1,-1,4,5,6,7};
  PetscInt               indices2[] = {0,1,2,3,4,5,-1,-1,-1,-1,-1,-1,6,7,8,9,10,11};
  ISLocalToGlobalMapping map;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,12,indices,PETSC_COPY_VALUES,&map));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,2));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,4));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,2));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,1));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,18,indices2,PETSC_COPY_VALUES,&map));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,3));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,6));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,3));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,1));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,5,2,indices2,PETSC_COPY_VALUES,&map));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(map,2));
  PetscCall(ISLocalToGlobalMappingView(map,NULL));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
