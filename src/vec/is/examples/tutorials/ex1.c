#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.13 1999/06/30 23:50:12 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates creating a general index set.\n\n";

/*T
    Concepts: Index sets^Manipulating a general index set;
    Routines: ISCreateGeneral(); ISDestroy(); ISView()
    Routines: ISGetIndices(); ISRestoreIndices()
    
    Comment: Creates an index set based on a set of integers. Views that index set
  and then destroys it.
T*/
 
/*
    Include is.h so we can use PETSc IS objects. Note that this automatically 
  includes petsc.h.
*/
#include "is.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int      ierr,*indices,rank,n;
  IS       is;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /*
     Create an index set with 5 entries. Each processor creates
   its own index set with its own list of integers.
  */
  indices = (int *) PetscMalloc( 5*sizeof(int) );CHKPTRA(indices);
  indices[0] = rank + 1; 
  indices[1] = rank + 2; 
  indices[2] = rank + 3; 
  indices[3] = rank + 4; 
  indices[4] = rank + 5; 
  ierr = ISCreateGeneral(PETSC_COMM_SELF,5,indices,&is);CHKERRA(ierr);
  /*
     Note that ISCreateGeneral() has made a copy of the indices
     so we may (and generally should) free indices[]
  */
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /*
     Print the index set to stdout
  */
  ierr = ISView(is,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  /*
     Get the number of indices in the set 
  */
  ierr = ISGetSize(is,&n);CHKERRA(ierr);

  /*
     Get the indices in the index set
  */
  ierr = ISGetIndices(is,&indices);CHKERRA(ierr);
  /*
     Now any code that needs access to the list of integers
   has access to it here through indices[].
   */
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] First index %d\n",rank,indices[0]);CHKERRA(ierr);

  /*
     Once we no longer need access to the indices they should 
     returned to the system 
  */
  ierr = ISRestoreIndices(is,&indices);CHKERRA(ierr);

  /*
     One should destroy any PETSc object once one is completely
    done with it.
  */
  ierr = ISDestroy(is);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
