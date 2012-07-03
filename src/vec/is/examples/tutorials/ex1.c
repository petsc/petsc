
static char help[] = "Creating a general index set.\n\n";

/*T
    Concepts: index sets^manipulating a general index set;
    Concepts: index sets^creating general;
    Concepts: IS^creating a general index set;

    Description: Creates an index set based on a set of integers. Views that index set
    and then destroys it.
    
T*/
 
/*
    Include petscis.h so we can use PETSc IS objects. Note that this automatically 
  includes petscsys.h.
*/
#include <petscis.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       *indices,n;
  const PetscInt *nindices;
  PetscMPIInt    rank;
  IS             is;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*
     Create an index set with 5 entries. Each processor creates
   its own index set with its own list of integers.
  */
  ierr = PetscMalloc(5*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  indices[0] = rank + 1; 
  indices[1] = rank + 2; 
  indices[2] = rank + 3; 
  indices[3] = rank + 4; 
  indices[4] = rank + 5; 
  ierr = ISCreateGeneral(PETSC_COMM_SELF,5,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  /*
     Note that ISCreateGeneral() has made a copy of the indices
     so we may (and generally should) free indices[]
  */
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /*
     Print the index set to stdout
  */
  ierr = ISView(is,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /*
     Get the number of indices in the set 
  */
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);

  /*
     Get the indices in the index set
  */
  ierr = ISGetIndices(is,&nindices);CHKERRQ(ierr);
  /*
     Now any code that needs access to the list of integers
   has access to it here through indices[].
   */
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] First index %D\n",rank,nindices[0]);CHKERRQ(ierr);

  /*
     Once we no longer need access to the indices they should 
     returned to the system 
  */
  ierr = ISRestoreIndices(is,&nindices);CHKERRQ(ierr);

  /*
     One should destroy any PETSc object once one is completely
    done with it.
  */
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
