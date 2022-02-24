
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
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,n,inputindices,PETSC_COPY_VALUES,&set));
  CHKERRQ(ISView(set,PETSC_VIEWER_STDOUT_SELF));

  /*
    Extract indices from set.
  */
  CHKERRQ(ISGetLocalSize(set,&issize));
  CHKERRQ(ISGetIndices(set,&indices));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Printing indices directly\n"));
  for (i=0; i<issize; i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "\n",indices[i]));
  }
  CHKERRQ(ISRestoreIndices(set,&indices));

  /*
    Extract the block indices. This returns one index per block.
  */
  CHKERRQ(ISBlockGetIndices(set,&indices));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Printing block indices directly\n"));
  for (i=0; i<n; i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "\n",indices[i]));
  }
  CHKERRQ(ISBlockRestoreIndices(set,&indices));

  /*
    Check if this is really a block index set
  */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)set,ISBLOCK,&isblock));
  PetscCheckFalse(!isblock,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Index set is not blocked!");

  /*
    Determine the block size of the index set
  */
  CHKERRQ(ISGetBlockSize(set,&bs));
  PetscCheckFalse(bs != 3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Block size is not 3!");

  /*
    Get the number of blocks
  */
  CHKERRQ(ISBlockGetLocalSize(set,&n));
  PetscCheckFalse(n != 4,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of blocks not 4!");

  CHKERRQ(ISDestroy(&set));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
