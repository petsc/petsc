
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
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,12,indices,PETSC_COPY_VALUES,&map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,2);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,4);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,2);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,18,indices2,PETSC_COPY_VALUES,&map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,3);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,6);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,3);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,5,2,indices2,PETSC_COPY_VALUES,&map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map,2);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(map,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
