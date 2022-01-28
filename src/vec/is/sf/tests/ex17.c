static const char help[] = "Test PetscSF with MPI large count (more than 2 billion elements in messages)\n\n";

#include <petscsys.h>
#include <petscsf.h>

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  PetscSF           sf;
  PetscInt          i,nroots,nleaves;
  PetscInt          n = (1ULL<<31) + 1024; /* a little over 2G elements */
  PetscSFNode       *iremote = NULL;
  PetscMPIInt       rank,size;
  char              *rootdata=NULL,*leafdata=NULL;
  Vec               x,y;
  VecScatter        vscat;
  PetscInt          rstart,rend;
  IS                ix;
  const PetscScalar *xv;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"The test can only run with two MPI ranks");

  /* Test PetscSF */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);

  if (!rank) {
    nroots  = n;
    nleaves = 0;
  } else {
    nroots  = 0;
    nleaves = n;
    ierr    = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) {
      iremote[i].rank  = 0;
      iremote[i].index = i;
    }
  }
  ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscMalloc2(nroots,&rootdata,nleaves,&leafdata);CHKERRQ(ierr);
  if (!rank) {
    memset(rootdata,11,nroots);
    rootdata[nroots-1] = 12; /* Use a different value at the end */
  }

  ierr = PetscSFBcastBegin(sf,MPI_SIGNED_CHAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr); /* rank 0->1, bcast rootdata to leafdata */
  ierr = PetscSFBcastEnd(sf,MPI_SIGNED_CHAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPI_SIGNED_CHAR,leafdata,rootdata,MPI_SUM);CHKERRQ(ierr); /* rank 1->0, add leafdata to rootdata */
  ierr = PetscSFReduceEnd(sf,MPI_SIGNED_CHAR,leafdata,rootdata,MPI_SUM);CHKERRQ(ierr);
  if (!rank) {
    PetscAssertFalse(rootdata[0] != 22 || rootdata[nroots-1] != 24,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF: wrong results");
  }

  ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscFree(iremote);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test VecScatter */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(x,rank==0? n : 64,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetSizes(y,rank==0? 64 : n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,rend-rstart,rstart,1,&ix);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,ix,y,ix,&vscat);CHKERRQ(ierr);

  ierr = VecSet(x,3.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x,&xv);CHKERRQ(ierr);
  PetscAssertFalse(xv[0] != 6.0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"VecScatter: wrong results");
  ierr = VecRestoreArrayRead(x,&xv);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
  ierr = ISDestroy(&ix);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/**TEST
   test:
     requires: defined(PETSC_HAVE_MPI_LARGE_COUNT) defined(PETSC_USE_64BIT_INDICES)
     TODO: need a machine with big memory (~150GB) to run the test
     nsize: 2
     args: -sf_type {{basic neighbor}}

TEST**/

