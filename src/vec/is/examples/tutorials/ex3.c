
/*      "$Id: ex3.c,v 1.9 1999/05/04 20:30:20 balay Exp bsmith $"; */

static char help[] = "Demonstrates creating a blocked index set.\n\n";

/*T
    Concepts: Index sets^Manipulating a block index set;
    Routines: ISCreateBlock(); ISDestroy(); ISView()
    Routines: ISGetIndices(); ISRestoreIndices(); ISBlockGetSize()
    Routines: ISBlockGetBlockSize(); ISBlockGetIndices(); ISBlockRestoreIndices()
    Routines: ISBlock()

    Comment:  Creates an index set based on blocks of integers. Views that index set
    and then destroys it.
T*/

#include "is.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        i, n = 4, ierr,  inputindices[] = {0,3,9,12}, bs = 3,issize,*indices;
  IS         set;
  PetscTruth isblock;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);
      
  /*
    Create a block index set. The index set has 4 blocks each of size 3.
    The indices are {0,1,2,3,4,5,9,10,11,12,13,14}
    Note each processor is generating its own index set 
    (in this case they are all identical)
  */
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,n,inputindices,&set);CHKERRA(ierr);
  ierr = ISView(set,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  /*
    Extract indices from set.
  */
  ierr = ISGetSize(set,&issize);CHKERRA(ierr);
  ierr = ISGetIndices(set,&indices);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Printing indices directly\n");CHKERRA(ierr);
  for (i=0; i<issize; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%d\n",indices[i]);CHKERRA(ierr);
  }
  ierr = ISRestoreIndices(set,&indices);CHKERRA(ierr);

  /*
    Extract the block indices. This returns one index per block.
  */
  ierr = ISBlockGetIndices(set,&indices);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Printing block indices directly\n");CHKERRA(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%d\n",indices[i]);CHKERRA(ierr);
  }
  ierr = ISBlockRestoreIndices(set,&indices);CHKERRA(ierr);

  /*
    Check if this is really a block index set
  */
  ierr = ISBlock(set,&isblock);CHKERRA(ierr);
  if (isblock != PETSC_TRUE) SETERRA(1,0,"Index set is not blocked!");

  /*
    Determine the block size of the index set
  */
  ierr = ISBlockGetBlockSize(set,&bs);CHKERRA(ierr);
  if (bs != 3) SETERRA(1,0,"Block size is not 3!");

  /*
    Get the number of blocks
  */
  ierr = ISBlockGetSize(set,&n);CHKERRA(ierr);
  if (n != 4) SETERRA(1,0,"Number of blocks not 4!");

  ierr = ISDestroy(set);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


