
static char help[] = "Tests ISLocalToGlobalMappingSetBlockSize.\n\n";

/*T
    Concepts: local to global mappings
    Concepts: global to local mappings

    Description:  Creates a map with a general set of indices and then change them to blocks of integers.
T*/

#include <petscis.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               indices[] = {0,1,2,3,-1,-1,-1,-1,8,9,10,11};
  ISLocalToGlobalMapping map1,map2;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,12,indices,PETSC_COPY_VALUES,&map1);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,12,indices,PETSC_COPY_VALUES,&map2);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map1,2);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(map2,4);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map1);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


