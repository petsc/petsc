static const char help[] = "Test star forest communication (PetscSF)\n\n";

#include <petscsf.h>
#include <petsc/private/sfimpl.h>

static PetscErrorCode CheckGraphNotSet(PetscSF sf)
{
  PetscInt          nroots,nleaves;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;

  PetscFunctionBegin;
  PetscCheck(!sf->graphset,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCall(PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote));
  PetscCheck(nroots  < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheck(nleaves < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheck(!ilocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheck(!iremote,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  PetscCheck(sf->minleaf == PETSC_MAX_INT,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not PETSC_MAX_INT");
  PetscCheck(sf->maxleaf == PETSC_MIN_INT,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not PETSC_MIN_INT");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckGraphEmpty(PetscSF sf)
{
  PetscInt          nroots,nleaves;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscInt          minleaf,maxleaf;

  PetscFunctionBegin;
  PetscCall(PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote));
  PetscCheck(!nroots,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCheck(!nleaves,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCheck(!ilocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCheck(!iremote,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  PetscCall(PetscSFGetLeafRange(sf,&minleaf,&maxleaf));
  PetscCheck(minleaf ==  0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not 0");
  PetscCheck(maxleaf == -1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF maximum leaf is not -1");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckRanksNotSet(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCheck(sf->nranks == -1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks are set");
  PetscCheck(sf->ranks  == NULL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks are set");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckRanksEmpty(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCheck(sf->nranks == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks not empty");
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscSF        sf,sfDup,sfInv,sfEmbed,sfA,sfB,sfBA;
  const PetscInt *degree;
  char           sftype[64] = PETSCSFBASIC;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-user_sf_type",sftype,sizeof(sftype),NULL));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFReset(sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFReset(sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(CheckGraphEmpty(sf));
  PetscCall(PetscSFReset(sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(CheckGraphEmpty(sf));
  PetscCall(PetscSFReset(sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  /* Test setup */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(CheckRanksNotSet(sf));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(CheckRanksNotSet(sf));
  PetscCall(PetscSFSetUp(sf));
  PetscCall(CheckRanksEmpty(sf));
  PetscCall(PetscSFDestroy(&sf));

  /* Test setup then reset */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFReset(sf));
  PetscCall(CheckRanksNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  /* Test view (no graph set, no type set) */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFView(sf,NULL));
  PetscCall(PetscSFDestroy(&sf));

  /* Test set graph then view (no type set) */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFView(sf,NULL));
  PetscCall(PetscSFDestroy(&sf));

  /* Test set type then view (no graph set) */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFView(sf,NULL));
  PetscCall(PetscSFDestroy(&sf));

  /* Test set type then graph then view */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFView(sf,NULL));
  PetscCall(PetscSFDestroy(&sf));

  /* Test set graph then type */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(CheckGraphEmpty(sf));
  PetscCall(PetscSFReset(sf));
  PetscCall(CheckGraphNotSet(sf));
  PetscCall(PetscSFDestroy(&sf));

  /* Test Bcast (we call setfromoptions) */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFBcastBegin(sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd  (sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));

  /* From now on we also call SetFromOptions */

  /* Test Reduce */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFReduceBegin(sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  PetscCall(PetscSFReduceEnd  (sf,MPI_INT,NULL,NULL,MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(sf,MPI_INT,NULL,NULL,MPI_SUM));
  PetscCall(PetscSFReduceEnd  (sf,MPI_INT,NULL,NULL,MPI_SUM));
  PetscCall(PetscSFDestroy(&sf));

  /* Test FetchAndOp */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFFetchAndOpBegin(sf,MPI_INT,NULL,NULL,NULL,MPI_SUM));
  PetscCall(PetscSFFetchAndOpEnd  (sf,MPI_INT,NULL,NULL,NULL,MPI_SUM));
  PetscCall(PetscSFDestroy(&sf));

  /* Test ComputeDegree */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFComputeDegreeBegin(sf,&degree));
  PetscCall(PetscSFComputeDegreeEnd(sf,&degree));
  PetscCall(PetscSFDestroy(&sf));

  /* Test PetscSFDuplicate() */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFDuplicate(sf,PETSCSF_DUPLICATE_GRAPH,&sfDup));
  PetscCall(CheckGraphEmpty(sfDup));
  PetscCall(PetscSFDestroy(&sfDup));
  PetscCall(PetscSFDestroy(&sf));

  /* Test PetscSFCreateInverseSF() */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFCreateInverseSF(sf,&sfInv));
  PetscCall(CheckGraphEmpty(sfInv));
  PetscCall(PetscSFDestroy(&sfInv));
  PetscCall(PetscSFDestroy(&sf));

  /* Test PetscSFCreateEmbeddedRootSF() */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFCreateEmbeddedRootSF(sf,0,NULL,&sfEmbed));
  PetscCall(CheckGraphEmpty(sfEmbed));
  PetscCall(PetscSFDestroy(&sfEmbed));
  PetscCall(PetscSFDestroy(&sf));

  /* Test PetscSFCreateEmbeddedLeafSF() */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetType(sf,sftype));
  PetscCall(PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFCreateEmbeddedLeafSF(sf,0,NULL,&sfEmbed));
  PetscCall(CheckGraphEmpty(sfEmbed));
  PetscCall(PetscSFDestroy(&sfEmbed));
  PetscCall(PetscSFDestroy(&sf));

  /* Test PetscSFCompose() */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sfA));
  PetscCall(PetscSFSetType(sfA,sftype));
  PetscCall(PetscSFSetGraph(sfA,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sfB));
  PetscCall(PetscSFSetType(sfB,sftype));
  PetscCall(PetscSFSetGraph(sfB,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER));
  PetscCall(PetscSFCompose(sfA,sfB,&sfBA));
  PetscCall(CheckGraphEmpty(sfBA));
  PetscCall(PetscSFDestroy(&sfBA));
  PetscCall(PetscSFDestroy(&sfA));
  PetscCall(PetscSFDestroy(&sfB));

  PetscCall(PetscFinalize());
  return 0;
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
