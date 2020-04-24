
static char help[] = "Demonstrates using ISLocalToGlobalMappings with block size.\n\n";

/*T
    Concepts: local to global mappings
    Concepts: global to local mappings

    Description:  Creates an index set based on blocks of integers. Views that index set
    and then destroys it.
T*/

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               i,n = 4,indices[] = {0,3,9,12},m = 2,input[] = {0,2};
  PetscInt               output[2],inglobals[13],outlocals[13];
  ISLocalToGlobalMapping mapping;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*
      Create a local to global mapping. Each processor independently
     creates a mapping
  */
  ierr = PetscIntView(n,indices,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,2,n,indices,PETSC_COPY_VALUES,&mapping);CHKERRQ(ierr);

  /*
     Map a set of local indices to their global values
  */
  ierr = PetscIntView(m,input,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mapping,m,input,output);CHKERRQ(ierr);
  ierr = PetscIntView(m,output,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Map some global indices to local, retaining the ones without a local index by -1
  */
  for (i=0; i<13; i++) inglobals[i] = i;
  ierr = PetscIntView(13,inglobals,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(mapping,IS_GTOLM_MASK,13,inglobals,NULL,outlocals);CHKERRQ(ierr);
  ierr = PetscIntView(13,outlocals,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Map some block global indices to local, dropping the ones without a local index.
  */
  ierr = PetscIntView(13,inglobals,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(mapping,IS_GTOLM_DROP,13,inglobals,&m,outlocals);CHKERRQ(ierr);
  ierr = PetscIntView(m,outlocals,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Free the space used by the local to global mapping
  */
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);


  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
