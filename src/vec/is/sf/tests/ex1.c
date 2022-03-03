static const char help[] = "Test star forest communication (PetscSF)\n\n";

/*T
    Description: This example creates empty star forests to test the API.
T*/

#include <petscsf.h>
#include <petsc/private/sfimpl.h>

static PetscErrorCode CheckGraphNotSet(PetscSF sf)
{
  PetscInt          nroots,nleaves;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;

  PetscFunctionBegin;
  PetscCheck(!sf->graphset,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  CHKERRQ(PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote));
  PetscCheckFalse(nroots  >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheckFalse(nleaves >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheck(!ilocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheck(!iremote,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheckFalse(sf->minleaf != PETSC_MAX_INT,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not PETSC_MAX_INT");
  PetscCheckFalse(sf->maxleaf != PETSC_MIN_INT,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not PETSC_MIN_INT");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckGraphEmpty(PetscSF sf)
{
  PetscInt          nroots,nleaves;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscInt          minleaf,maxleaf;

  PetscFunctionBegin;
  CHKERRQ(PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote));
  PetscCheck(!nroots,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCheck(!nleaves,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCheck(!ilocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCheck(!iremote,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  CHKERRQ(PetscSFGetLeafRange(sf,&minleaf,&maxleaf));
  PetscCheckFalse(minleaf !=  0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not 0");
  PetscCheckFalse(maxleaf != -1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF maximum leaf is not -1");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckRanksNotSet(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCheckFalse(sf->nranks != -1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks are set");
  PetscCheckFalse(sf->ranks  != NULL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks are set");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckRanksEmpty(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCheckFalse(sf->nranks != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks not empty");
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscSF        sf,sfDup,sfInv,sfEmbed,sfA,sfB,sfBA;
  const PetscInt *degree;
  PetscErrorCode ierr;
  char           sftype[64] = PETSCSFBASIC;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-user_sf_type",sftype,sizeof(sftype),NULL));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFReset(sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFReset(sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(CheckGraphEmpty(sf));
  CHKERRQ(PetscSFReset(sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(CheckGraphEmpty(sf));
  CHKERRQ(PetscSFReset(sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test setup */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(CheckRanksNotSet(sf));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(CheckRanksNotSet(sf));
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(CheckRanksEmpty(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test setup then reset */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(PetscSFReset(sf));
  CHKERRQ(CheckRanksNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test view (no graph set, no type set) */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFView(sf,NULL));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test set graph then view (no type set) */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFView(sf,NULL));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test set type then view (no graph set) */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFView(sf,NULL));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test set type then graph then view */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFView(sf,NULL));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test set graph then type */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(CheckGraphEmpty(sf));
  CHKERRQ(PetscSFReset(sf));
  CHKERRQ(CheckGraphNotSet(sf));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test Bcast (we call setfromoptions) */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFBcastBegin(sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd  (sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  CHKERRQ(PetscSFDestroy(&sf));

  /* From now on we also call SetFromOptions */

  /* Test Reduce */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFReduceBegin(sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  CHKERRQ(PetscSFReduceEnd  (sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  CHKERRQ(PetscSFReduceBegin(sf,MPI_INT,NULL,NULL,MPI_SUM));
  CHKERRQ(PetscSFReduceEnd  (sf,MPI_INT,NULL,NULL,MPI_SUM));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test FetchAndOp */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFFetchAndOpBegin(sf,MPI_INT,NULL,NULL,NULL,MPI_SUM));
  CHKERRQ(PetscSFFetchAndOpEnd  (sf,MPI_INT,NULL,NULL,NULL,MPI_SUM));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test ComputeDegree */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFComputeDegreeBegin(sf,&degree));
  CHKERRQ(PetscSFComputeDegreeEnd(sf,&degree));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test PetscSFDuplicate() */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFDuplicate(sf,PETSCSF_DUPLICATE_GRAPH,&sfDup));
  CHKERRQ(CheckGraphEmpty(sfDup));
  CHKERRQ(PetscSFDestroy(&sfDup));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test PetscSFCreateInverseSF() */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFCreateInverseSF(sf,&sfInv));
  CHKERRQ(CheckGraphEmpty(sfInv));
  CHKERRQ(PetscSFDestroy(&sfInv));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test PetscSFCreateEmbeddedRootSF() */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFCreateEmbeddedRootSF(sf,0,NULL,&sfEmbed));
  CHKERRQ(CheckGraphEmpty(sfEmbed));
  CHKERRQ(PetscSFDestroy(&sfEmbed));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test PetscSFCreateEmbeddedLeafSF() */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetType(sf,sftype));
  CHKERRQ(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFCreateEmbeddedLeafSF(sf,0,NULL,&sfEmbed));
  CHKERRQ(CheckGraphEmpty(sfEmbed));
  CHKERRQ(PetscSFDestroy(&sfEmbed));
  CHKERRQ(PetscSFDestroy(&sf));

  /* Test PetscSFCompose() */
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sfA));
  CHKERRQ(PetscSFSetType(sfA,sftype));
  CHKERRQ(PetscSFSetGraph(sfA,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sfB));
  CHKERRQ(PetscSFSetType(sfB,sftype));
  CHKERRQ(PetscSFSetGraph(sfB,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  CHKERRQ(PetscSFCompose(sfA,sfB,&sfBA));
  CHKERRQ(CheckGraphEmpty(sfBA));
  CHKERRQ(PetscSFDestroy(&sfBA));
  CHKERRQ(PetscSFDestroy(&sfA));
  CHKERRQ(PetscSFDestroy(&sfB));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: basic_1
      nsize: 1

   test:
      suffix: basic_2
      nsize: 2

   test:
      suffix: basic_3
      nsize: 3

   test:
      suffix: window
      args: -user_sf_type window -sf_type window -sf_window_flavor {{create dynamic allocate}} -sf_window_sync {{fence active lock}}
      nsize: {{1 2 3}separate output}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   # The nightly test suite with MPICH uses ch3:sock, which is broken when winsize == 0 in some of the processes
   test:
      suffix: window_shared
      args: -user_sf_type window -sf_type window -sf_window_flavor shared -sf_window_sync {{fence active lock}}
      nsize: {{1 2 3}separate output}
      requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) !defined(PETSC_HAVE_MPICH_NUMVERSION) defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

TEST*/
