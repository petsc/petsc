
static char help[] = "Demonstrates using ISLocalToGlobalMappings with block size.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt               i,n = 4,indices[] = {0,3,9,12},m = 2,input[] = {0,2};
  PetscInt               output[2],inglobals[13],outlocals[13];
  ISLocalToGlobalMapping mapping;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /*
      Create a local to global mapping. Each processor independently
     creates a mapping
  */
  PetscCall(PetscIntView(n,indices,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,2,n,indices,PETSC_COPY_VALUES,&mapping));

  /*
     Map a set of local indices to their global values
  */
  PetscCall(PetscIntView(m,input,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISLocalToGlobalMappingApply(mapping,m,input,output));
  PetscCall(PetscIntView(m,output,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Map some global indices to local, retaining the ones without a local index by -1
  */
  for (i=0; i<13; i++) inglobals[i] = i;
  PetscCall(PetscIntView(13,inglobals,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISGlobalToLocalMappingApply(mapping,IS_GTOLM_MASK,13,inglobals,NULL,outlocals));
  PetscCall(PetscIntView(13,outlocals,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Map some block global indices to local, dropping the ones without a local index.
  */
  PetscCall(PetscIntView(13,inglobals,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISGlobalToLocalMappingApplyBlock(mapping,IS_GTOLM_DROP,13,inglobals,&m,outlocals));
  PetscCall(PetscIntView(m,outlocals,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free the space used by the local to global mapping
  */
  PetscCall(ISLocalToGlobalMappingDestroy(&mapping));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
