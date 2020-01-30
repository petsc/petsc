
#include <petscsys.h>        /*I  "petscsys.h"   I*/

PetscStack *petscstack = NULL;

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>

static PetscBool amsmemstack = PETSC_FALSE;

/*@C
   PetscStackSAWsGrantAccess - Grants access of the PETSc stack frames to the SAWs publisher

   Collective on PETSC_COMM_WORLD?

   Level: developer

   Developers Note: Cannot use PetscFunctionBegin/Return() or PetscStackCallSAWs() since it may be used within those routines

.seealso: PetscObjectSetName(), PetscObjectSAWsViewOff(), PetscObjectSAWsTakeAccess()

@*/
void  PetscStackSAWsGrantAccess(void)
{
  if (amsmemstack) {
    /* ignore any errors from SAWs */
    SAWs_Unlock();
  }
}

/*@C
   PetscStackSAWsTakeAccess - Takes access of the PETSc stack frames to the SAWs publisher

   Collective on PETSC_COMM_WORLD?

   Level: developer

   Developers Note: Cannot use PetscFunctionBegin/Return() or PetscStackCallSAWs() since it may be used within those routines

.seealso: PetscObjectSetName(), PetscObjectSAWsViewOff(), PetscObjectSAWsTakeAccess()

@*/
void  PetscStackSAWsTakeAccess(void)
{
  if (amsmemstack) {
    /* ignore any errors from SAWs */
    SAWs_Lock();
  }
}

PetscErrorCode PetscStackViewSAWs(void)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (rank) return 0;
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Stack/functions",petscstack->function,20,SAWs_READ,SAWs_STRING));
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Stack/__current_size",&petscstack->currentsize,1,SAWs_READ,SAWs_INT));
  amsmemstack = PETSC_TRUE;
  return 0;
}

PetscErrorCode PetscStackSAWsViewOff(void)
{
  PetscFunctionBegin;
  if (!amsmemstack) PetscFunctionReturn(0);
  PetscStackCallSAWs(SAWs_Delete,("/PETSc/Stack"));
  amsmemstack = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#  endif


PetscErrorCode PetscStackCreate(void)
{
  PetscStack *petscstack_in;
  PetscInt   i;

  if (PetscStackActive()) return 0;

  petscstack_in              = (PetscStack*)malloc(sizeof(PetscStack));
  petscstack_in->currentsize = 0;
  petscstack_in->hotdepth    = 0;
  for (i=0; i<PETSCSTACKSIZE; i++) {
    petscstack_in->function[i] = NULL;
    petscstack_in->file[i]     = NULL;
  }
  petscstack = petscstack_in;

#if defined(PETSC_HAVE_SAWS)
  {
  PetscBool flg = PETSC_FALSE;
  PetscOptionsHasName(NULL,NULL,"-stack_view",&flg);
  if (flg) PetscStackViewSAWs();
  }
#endif
  return 0;
}


PetscErrorCode  PetscStackView(FILE *file)
{
  int        i;

  if (!file) file = PETSC_STDOUT;

  if (file == PETSC_STDOUT) {
    (*PetscErrorPrintf)("Note: The EXACT line numbers in the stack are not available,\n");
    (*PetscErrorPrintf)("      INSTEAD the line number of the start of the function\n");
    (*PetscErrorPrintf)("      is given.\n");
    for (i=petscstack->currentsize-1; i>=0; i--) (*PetscErrorPrintf)("[%d] %s line %d %s\n",PetscGlobalRank,petscstack->function[i],petscstack->line[i],petscstack->file[i]);
  } else {
    fprintf(file,"Note: The EXACT line numbers in the stack are not available,\n");
    fprintf(file,"      INSTEAD the line number of the start of the function\n");
    fprintf(file,"      is given.\n");
    for (i=petscstack->currentsize-1; i>=0; i--) fprintf(file,"[%d] %s line %d %s\n",PetscGlobalRank,petscstack->function[i],petscstack->line[i],petscstack->file[i]);
  }
  return 0;
}

PetscErrorCode PetscStackDestroy(void)
{
  if (PetscStackActive()) {
    free(petscstack);
    petscstack = NULL;
  }
  return 0;
}

/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackCopy(PetscStack *sint,PetscStack *sout)
{
  int i;

  if (!sint) sout->currentsize = 0;
  else {
    for (i=0; i<sint->currentsize; i++) {
      sout->function[i]     = sint->function[i];
      sout->file[i]         = sint->file[i];
      sout->line[i]         = sint->line[i];
      sout->petscroutine[i] = sint->petscroutine[i];
    }
    sout->currentsize = sint->currentsize;
  }
  return 0;
}

/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackPrint(PetscStack *sint,FILE *fp)
{
  int i;

  if (!sint) return(0);
  for (i=sint->currentsize-2; i>=0; i--) fprintf(fp,"      [%d]  %s() line %d in %s\n",PetscGlobalRank,sint->function[i],sint->line[i],sint->file[i]);
  return 0;
}

