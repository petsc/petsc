#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.1 1996/08/14 02:57:35 bsmith Exp bsmith $";
#endif

static char help[] = "Demonstrates creating a general index set.\n\n";

/*T

    Concepts: Index set, indices
    Routines: ISCreateSeq(), ISDestroy(), ISView()
    
     Creates an index set based on a set of integers. Views that index set
  and then destroys it.
T*/
 
#include "petsc.h"
#include "is.h"
#include "sys.h"
#include <math.h>

int main(int argc,char **argv)
{
  int      ierr,*indices,rank,n;
  IS       is;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /*
     Create an index set with 5 entries. Each processor creates
   its own index set with its own list of integers.
  */
  indices = (int *) PetscMalloc( 5*sizeof(int) ); CHKPTRA(indices);
  indices[0] = rank + 1; 
  indices[1] = rank + 2; 
  indices[2] = rank + 3; 
  indices[3] = rank + 4; 
  indices[4] = rank + 5; 
  ierr = ISCreateSeq(MPI_COMM_SELF,5,indices,&is); CHKERRA(ierr);
  /*
     Note that ISCreateSeq() has made a copy of the indices
     so we may (and generally should) free indices[]
  */
  PetscFree(indices);

  /*
     Print the index set to stdout
  */
  ierr = ISView(is,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  /*
     Get the number of indices in the set 
  */
  ierr = ISGetSize(is,&n); CHKERRA(ierr);

  /*
     Get the indices in the index set
  */
  ierr = ISGetIndices(is,&indices); CHKERRA(ierr);
  /*
     No any code that needs access to the list of integers
   has access to it here through indices[].
   */

  /*
     Once we no longer need access to the indices they should 
     returned to the system 
  */
  ierr = ISRestoreIndices(is,&indices); CHKERRA(ierr);
  
  /*
     One should destroy any PETSc object once one is completely
    done with it.
  */
  ierr = ISDestroy(is); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
