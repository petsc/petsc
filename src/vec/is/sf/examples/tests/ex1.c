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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (sf->graphset) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
  if (nroots  >= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  if (nleaves >= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  if (ilocal)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  if (iremote) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is set");
  if (sf->minleaf != PETSC_MAX_INT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not PETSC_MAX_INT");
  if (sf->maxleaf != PETSC_MIN_INT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not PETSC_MIN_INT");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckGraphEmpty(PetscSF sf)
{
  PetscInt          nroots,nleaves;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscInt          minleaf,maxleaf;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
  if (nroots)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  if (nleaves) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  if (ilocal)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  if (iremote) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF graph is not empty");
  ierr = PetscSFGetLeafRange(sf,&minleaf,&maxleaf);CHKERRQ(ierr);
  if (minleaf !=  0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF minimum leaf is not 0");
  if (maxleaf != -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF maximum leaf is not -1");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckRanksNotSet(PetscSF sf)
{
  PetscFunctionBegin;
  if (sf->nranks != -1)   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks are set");
  if (sf->ranks  != NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks are set");
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckRanksEmpty(PetscSF sf)
{
  PetscFunctionBegin;
  if (sf->nranks != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"SF ranks not empty");
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscSF        sf,sfDup,sfInv,sfEmbed,sfA,sfB,sfBA;
  const PetscInt *degree;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sf);CHKERRQ(ierr);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sf);CHKERRQ(ierr);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test setup */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = CheckRanksNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = CheckRanksNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = CheckRanksEmpty(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test setup then reset */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = CheckRanksNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test view (no graph set, no type set) */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFView(sf,NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test set graph then view (no type set) */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFView(sf,NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test set type then view (no graph set) */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFView(sf,NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test set type then graph then view */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFView(sf,NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test set graph then type */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sf);CHKERRQ(ierr);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = CheckGraphNotSet(sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test Bcast */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPI_INT,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPI_INT,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test Reduce */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPI_INT,NULL,NULL,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPI_INT,NULL,NULL,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPI_INT,NULL,NULL,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPI_INT,NULL,NULL,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test FetchAndOp */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpBegin(sf,MPI_INT,NULL,NULL,NULL,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd  (sf,MPI_INT,NULL,NULL,NULL,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test ComputeDegree */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_COPY_VALUES,NULL,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test PetscSFDuplicate() */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER);CHKERRQ(ierr);
  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_GRAPH,&sfDup);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sfDup);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfDup);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test PetscSFCreateInverseSF() */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER);CHKERRQ(ierr);
  ierr = PetscSFCreateInverseSF(sf,&sfInv);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sfInv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfInv);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test PetscSFCreateEmbeddedSF() */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER);CHKERRQ(ierr);
  ierr = PetscSFCreateEmbeddedSF(sf,0,NULL,&sfEmbed);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sfEmbed);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfEmbed);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test PetscSFCreateEmbeddedLeafSF() */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER);CHKERRQ(ierr);
  ierr = PetscSFCreateEmbeddedLeafSF(sf,0,NULL,&sfEmbed);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sfEmbed);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfEmbed);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* Test PetscSFCompose() */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sfA);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfA,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER);CHKERRQ(ierr);
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sfB);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfB,0,0,NULL,PETSC_USE_POINTER,NULL,PETSC_USE_POINTER);CHKERRQ(ierr);
  ierr = PetscSFCompose(sfA,sfB,&sfBA);CHKERRQ(ierr);
  ierr = CheckGraphEmpty(sfBA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfBA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfB);CHKERRQ(ierr);

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

TEST*/
