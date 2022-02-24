
static char help[] = "Tests ISLocalToGlobalMappingSetBlockSize.\n\n";

/*T
    Concepts: local to global mappings
    Concepts: global to local mappings

    Description:  Creates a map with a general set of indices and then change them to blocks of integers.
T*/

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               indices[] = {0,1,2,3,-1,-1,-1,-1,4,5,6,7};
  PetscInt               indices2[] = {0,1,2,3,4,5,-1,-1,-1,-1,-1,-1,6,7,8,9,10,11};
  ISLocalToGlobalMapping map;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,12,indices,PETSC_COPY_VALUES,&map));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,2));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,4));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,2));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,1));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&map));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,18,indices2,PETSC_COPY_VALUES,&map));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,3));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,6));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,3));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,1));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&map));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,5,2,indices2,PETSC_COPY_VALUES,&map));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(map,2));
  CHKERRQ(ISLocalToGlobalMappingView(map,NULL));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&map));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
