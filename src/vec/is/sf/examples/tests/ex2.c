static const char help[] = "Test embedded sf with one leaf data item connnected to multiple roots\n\n";

#include <petscsf.h>

int main(int argc,char **argv)
{
  PetscSF        sf,newsf;
  PetscInt       i,nroots,nleaves,ilocal[2],leafdata,rootdata[2],nselected,selected,errors=0;
  PetscSFNode    iremote[2];
  PetscMPIInt    myrank,next,nproc;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&myrank);CHKERRQ(ierr);

  /* Create an SF that each process has two roots and two leaves. The two leaves are connected to the two
     roots on next process. The special thing is each process only has one data item in its leaf data buffer.
   */
  nroots    = 2;
  nleaves   = 2;
  ilocal[0] = 0; /* One leaf data item serves two leaves */
  ilocal[1] = 0;
  next      = (myrank+1)%nproc;
  for (i=0; i<nleaves; i++) {
    iremote[i].rank  = next;
    iremote[i].index = i;
  }

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);

  /* Create an embedded sf by only selecting the first root on each process */
  nselected = 1;
  selected  = 0;
  ierr = PetscSFCreateEmbeddedSF(sf,nselected,&selected,&newsf);CHKERRQ(ierr);

  /* Do reduce */
  leafdata    = 1;
  rootdata[0] = rootdata[1] = 0;
  ierr = PetscSFReduceBegin(newsf,MPIU_INT,&leafdata,rootdata,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(newsf,MPIU_INT,&leafdata,rootdata,MPIU_REPLACE);CHKERRQ(ierr);

  /* Check rootdata */
  if (rootdata[0] != 1 || rootdata[1] != 0) errors = 1;
  ierr = MPI_Allreduce(MPI_IN_PLACE,&errors,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (errors) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Unexpected rootdata on processors\n");CHKERRQ(ierr);}

  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&newsf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
TEST*/
