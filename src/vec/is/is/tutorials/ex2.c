
static char help[] = "Demonstrates creating a stride index set.\n\n";

/*T
    Concepts: index sets^creating a stride index set;
    Concepts: stride^creating a stride index set;
    Concepts: IS^creating a stride index set;

    Description: Creates an index set based on a stride. Views that index set
    and then destroys it.
T*/

/*
  Include petscis.h so we can use PETSc IS objects. Note that this automatically
  includes petscsys.h.
*/

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n,first,step;
  IS             set;
  const PetscInt *indices;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  n     = 10;
  first = 3;
  step  = 2;

  /*
    Create stride index set, starting at 3 with a stride of 2
    Note each processor is generating its own index set
    (in this case they are all identical)
  */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,first,step,&set);CHKERRQ(ierr);
  ierr = ISView(set,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /*
    Extract indices from set.
  */
  ierr = ISGetIndices(set,&indices);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing indices directly\n");CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "\n",indices[i]);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(set,&indices);CHKERRQ(ierr);

  /*
      Determine information on stride
  */
  ierr = ISStrideGetInfo(set,&first,&step);CHKERRQ(ierr);
  PetscAssertFalse(first != 3 || step != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Stride info not correct!");
  ierr = ISDestroy(&set);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
