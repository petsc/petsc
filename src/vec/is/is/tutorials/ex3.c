
static char help[] = "Demonstrates creating a blocked index set.\n\n";

/*T
    Concepts: index sets^creating a block index set;
    Concepts: IS^creating a block index set;

    Description:  Creates an index set based on blocks of integers. Views that index set
    and then destroys it.
T*/

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 4, inputindices[] = {0,1,3,4},bs = 3,issize;
  const PetscInt *indices;
  IS             set;
  PetscBool      isblock;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*
    Create a block index set. The index set has 4 blocks each of size 3.
    The indices are {0,1,2,3,4,5,9,10,11,12,13,14}
    Note each processor is generating its own index set
    (in this case they are all identical)
  */
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,n,inputindices,PETSC_COPY_VALUES,&set);CHKERRQ(ierr);
  ierr = ISView(set,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /*
    Extract indices from set.
  */
  ierr = ISGetLocalSize(set,&issize);CHKERRQ(ierr);
  ierr = ISGetIndices(set,&indices);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Printing indices directly\n");CHKERRQ(ierr);
  for (i=0; i<issize; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "\n",indices[i]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(set,&indices);CHKERRQ(ierr);

  /*
    Extract the block indices. This returns one index per block.
  */
  ierr = ISBlockGetIndices(set,&indices);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Printing block indices directly\n");CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "\n",indices[i]);CHKERRQ(ierr);
  }
  ierr = ISBlockRestoreIndices(set,&indices);CHKERRQ(ierr);

  /*
    Check if this is really a block index set
  */
  ierr = PetscObjectTypeCompare((PetscObject)set,ISBLOCK,&isblock);CHKERRQ(ierr);
  PetscAssertFalse(!isblock,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Index set is not blocked!");

  /*
    Determine the block size of the index set
  */
  ierr = ISGetBlockSize(set,&bs);CHKERRQ(ierr);
  PetscAssertFalse(bs != 3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Block size is not 3!");

  /*
    Get the number of blocks
  */
  ierr = ISBlockGetLocalSize(set,&n);CHKERRQ(ierr);
  PetscAssertFalse(n != 4,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of blocks not 4!");

  ierr = ISDestroy(&set);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
