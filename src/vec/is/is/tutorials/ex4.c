
static char help[] = "Demonstrates using ISLocalToGlobalMappings.\n\n";

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
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,n,indices,PETSC_COPY_VALUES,&mapping));
  CHKERRQ(ISLocalToGlobalMappingSetFromOptions(mapping));

  /*
     Map a set of local indices to their global values
  */
  CHKERRQ(ISLocalToGlobalMappingApply(mapping,m,input,output));
  CHKERRQ(PetscIntView(m,output,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Map some global indices to local, retaining the ones without a local index by -1
  */
  for (i=0; i<13; i++) inglobals[i] = i;
  CHKERRQ(ISGlobalToLocalMappingApply(mapping,IS_GTOLM_MASK,13,inglobals,NULL,outlocals));
  CHKERRQ(PetscIntView(13,outlocals,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Map some global indices to local, dropping the ones without a local index.
  */
  CHKERRQ(ISGlobalToLocalMappingApply(mapping,IS_GTOLM_DROP,13,inglobals,&m,outlocals));
  CHKERRQ(PetscIntView(m,outlocals,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISLocalToGlobalMappingView(mapping,PETSC_VIEWER_STDOUT_WORLD));
  /*
     Free the space used by the local to global mapping
  */
  CHKERRQ(ISLocalToGlobalMappingDestroy(&mapping));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -islocaltoglobalmapping_type hash

TEST*/
