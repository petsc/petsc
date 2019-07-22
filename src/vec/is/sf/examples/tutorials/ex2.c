static const char help[] = "Test overlapped communication on a single star forest (PetscSF)\n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscInt    ierr;
  PetscSF     sf;
  Vec         A,B,Aout,Bout;
  PetscScalar *bufA,*bufB,*bufAout,*bufBout;
  PetscInt    a,b,aout,bout,*bufa,*bufb,*bufaout,*bufbout;
  PetscMPIInt rank,size;
  PetscInt    i,*ilocal,nroots,nleaves;
  PetscSFNode *iremote;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  if (size != 2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for two MPI processes\n");

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);

  nleaves = 2;
  nroots = 1;
  ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);

  for (i = 0; i<nleaves; i++) {
    ilocal[i] = i;
  }

  ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
  if (rank == 0) {
    iremote[0].rank = 0;
    iremote[0].index = 0;
    iremote[1].rank = 1;
    iremote[1].index = 0;
  } else {
    iremote[0].rank = 1;
    iremote[0].index = 0;
    iremote[1].rank = 0;
    iremote[1].index = 0;
  }
  ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = VecSetSizes(A,2,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(A);CHKERRQ(ierr);
  ierr = VecSetUp(A);CHKERRQ(ierr);

  ierr = VecDuplicate(A,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(A,&Aout);CHKERRQ(ierr);
  ierr = VecDuplicate(A,&Bout);CHKERRQ(ierr);
  ierr = VecGetArray(A,&bufA);CHKERRQ(ierr);
  ierr = VecGetArray(B,&bufB);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    bufA[i] = (PetscScalar)rank;
    bufB[i] = (PetscScalar)(rank) + 10.0;
  }
  ierr = VecRestoreArray(A,&bufA);CHKERRQ(ierr);
  ierr = VecRestoreArray(B,&bufB);CHKERRQ(ierr);

  ierr = VecGetArrayRead(A,(const PetscScalar**)&bufA);CHKERRQ(ierr);
  ierr = VecGetArrayRead(B,(const PetscScalar**)&bufB);CHKERRQ(ierr);
  ierr = VecGetArray(Aout,&bufAout);CHKERRQ(ierr);
  ierr = VecGetArray(Bout,&bufBout);CHKERRQ(ierr);

  /* Testing overlapped PetscSFBcast with different rootdata and leafdata */
  ierr = PetscSFBcastBegin(sf,MPIU_SCALAR,bufA,bufAout);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_SCALAR,bufB,bufBout);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_SCALAR,bufA,bufAout);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_SCALAR,bufB,bufBout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(A,(const PetscScalar**)&bufA);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(B,(const PetscScalar**)&bufB);CHKERRQ(ierr);
  ierr = VecRestoreArray(Aout,&bufAout);CHKERRQ(ierr);
  ierr = VecRestoreArray(Bout,&bufBout);CHKERRQ(ierr);

  ierr = VecView(Aout,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(Bout,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&Aout);CHKERRQ(ierr);
  ierr = VecDestroy(&Bout);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Another very simple graph: rank 0 has one root, zero leaves; rank 1 has zero roots, one leave.
     Zero roots or leaves will result in NULL rootdata or leafdata. Therefore, one can not use that
     as key to identify pending communications.
   */
  if (!rank) {
    nroots  = 1;
    nleaves = 0;
  } else {
    nroots  = 0;
    nleaves = 1;
  }

  ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
  if (rank) {
    ilocal[0]        = 0;
    iremote[0].rank  = 0;
    iremote[0].index = 0;
  }

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);

  if (!rank) {
    a       = 1;
    b       = 10;
    bufa    = &a;
    bufb    = &b;
    bufaout = NULL;
    bufbout = NULL;
  } else {
    bufa    = NULL;
    bufb    = NULL;
    bufaout = &aout;
    bufbout = &bout;
  }

  /* Do Bcast to challenge PetscSFBcast if it uses rootdata to identify pending communications.
                    Rank 0             Rank 1
      rootdata:   bufa=XXX (1)       bufa=NULL
                  bufb=YYY (10)      bufb=NULL

      leafdata:   bufaout=NULL       bufaout=WWW
                  bufbout=NULL       bufbout=ZZZ
   */
  ierr = PetscSFBcastBegin(sf,MPIU_INT,bufa,bufaout);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,bufb,bufbout);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_INT,bufa,bufaout);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_INT,bufb,bufbout);CHKERRQ(ierr);
  if (rank) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"On rank 1, aout=%D, bout=%D\n",aout,bout);CHKERRQ(ierr);} /* On rank 1, aout=1, bout=10 */
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL);CHKERRQ(ierr);

  /* Do Reduce to challenge PetscSFReduce if it uses leafdata to identify pending communications.
                    Rank 0             Rank 1
      rootdata:   bufa=XXX (1)       bufa=NULL
                  bufb=YYY (10)      bufb=NULL

      leafdata:   bufaout=NULL       bufaout=WWW (1)
                  bufbout=NULL       bufbout=ZZZ (10)
   */
  ierr = PetscSFReduceBegin(sf,MPIU_INT,bufaout,bufa,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,bufbout,bufb,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,bufaout,bufa,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,bufbout,bufb,MPI_SUM);CHKERRQ(ierr);
  if (!rank) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"On rank 0, a   =%D, b   =%D\n",a,b);CHKERRQ(ierr);} /* On rank 0, a=2, b=20 */
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL);CHKERRQ(ierr);

/*  A difficult case PETSc can not handle correctly. So comment out. */
#if 0
  /*  Note on rank 0, the following overlapped Bcast have the same rootdata (bufa) and leafdata (NULL).
      It challenges PetscSF if it uses (rootdata, leafdata) as key to identify pending communications.
                    Rank 0             Rank 1
      rootdata:   bufa=XXX (2)       bufa=NULL

      leafdata:   bufaout=NULL       bufaout=WWW (1)
                  bufbout=NULL       bufbout=ZZZ (10)
   */
  ierr = PetscSFBcastBegin(sf,MPIU_INT,bufa,bufaout);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,bufa,bufbout);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_INT,bufa,bufaout);CHKERRQ(ierr); /* Buggy PetscSF could match a wrong PetscSFBcastBegin */
  ierr = PetscSFBcastEnd  (sf,MPIU_INT,bufa,bufbout);CHKERRQ(ierr);
  if (rank) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"On rank 1, aout=%D, bout=%D\n",aout,bout);CHKERRQ(ierr);} /* On rank 1, aout=2, bout=2 */

  /* The above Bcasts may successfully populate leafdata with correct values. But communinication contexts (i.e., the links,
     each with a unique MPI tag) may have retired in different order in the ->avail list. Processes doing the following
     PetscSFReduce will get tag-unmatched links from the ->avail list, resulting in dead lock.
   */
  ierr = PetscSFReduceBegin(sf,MPIU_INT,bufaout,bufa,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,bufaout,bufa,MPI_SUM);CHKERRQ(ierr);
  if (!rank) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"On rank 0, a   =%D, b   =%D\n",a,b);CHKERRQ(ierr);} /* On rank 0, a=4, b=20 */
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL);CHKERRQ(ierr);
#endif

  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      suffix: basic
      nsize: 2
      args: -sf_type basic

   test:
      suffix: window
      nsize: 2
      args: -sf_type window
      requires: define(PETSC_HAVE_MPI_WIN_CREATE)

TEST*/
