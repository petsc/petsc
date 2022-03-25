
#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/

#if PetscDefined(USE_DEBUG)
PetscStack petscstack;
#endif

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

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank) return 0;
#if PetscDefined(USE_DEBUG)
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Stack/functions",petscstack.function,20,SAWs_READ,SAWs_STRING));
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Stack/__current_size",&petscstack.currentsize,1,SAWs_READ,SAWs_INT));
#endif
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
#endif /* PETSC_HAVE_SAWS */

#if PetscDefined(USE_DEBUG)
PetscErrorCode PetscStackSetCheck(PetscBool check)
{
  petscstack.check = check;
  return 0;
}

PetscErrorCode PetscStackReset(void)
{
  memset(&petscstack,0,sizeof(petscstack));
  return 0;
}

PetscErrorCode  PetscStackView(FILE *file)
{
  if (!file) file = PETSC_STDOUT;
  if (petscstack.currentsize < 0) {
    /* < 0 is absolutely a corrupted stack, but this function is usually called in an error
     * handler, which are not capable of recovering from errors so best we can do is print
     * this warning */
    fprintf(file,"PetscStack is definitely corrupted with stack size %d\n",petscstack.currentsize);
  } else if (petscstack.currentsize == 0) {
    if (file == PETSC_STDOUT) {
      (*PetscErrorPrintf)("No error traceback is available, the problem could be in the main program. \n");
    } else {
      fprintf(file,"No error traceback is available, the problem could be in the main program. \n");
    }
  } else {
    if (file == PETSC_STDOUT) {
      (*PetscErrorPrintf)("The EXACT line numbers in the error traceback are not available.\n");
      (*PetscErrorPrintf)("instead the line number of the start of the function is given.\n");
      for (int i = petscstack.currentsize-1, j = 1; i >= 0; --i, ++j) {
        (*PetscErrorPrintf)("#%d %s() at %s:%d\n",j,petscstack.function[i],petscstack.file[i],petscstack.line[i]);
      }
    } else {
      fprintf(file,"The EXACT line numbers in the error traceback are not available.\n");
      fprintf(file,"Instead the line number of the start of the function is given.\n");
      for (int i = petscstack.currentsize-1, j = 1; i >= 0; --i, ++j) {
        fprintf(file,"[%d] #%d %s() at %s:%d\n",PetscGlobalRank,j,petscstack.function[i],petscstack.file[i],petscstack.line[i]);
      }
    }
  }
  return 0;
}

/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackCopy(PetscStack *sint,PetscStack *sout)
{
  if (sint) {
    for (int i = 0; i < sint->currentsize; ++i) {
      sout->function[i]     = sint->function[i];
      sout->file[i]         = sint->file[i];
      sout->line[i]         = sint->line[i];
      sout->petscroutine[i] = sint->petscroutine[i];
    }
    sout->currentsize = sint->currentsize;
  } else {
    sout->currentsize = 0;
  }
  return 0;
}

/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackPrint(PetscStack *sint,FILE *fp)
{
  if (sint) {
    for (int i = sint->currentsize-2; i >= 0; --i) {
      fprintf(fp,"      [%d]  %s() at %s:%d\n",PetscGlobalRank,sint->function[i],sint->file[i],sint->line[i]);
    }
  }
  return 0;
}
#endif /* PetscDefined(USE_DEBUG) */
