
#include <petscsys.h>        /*I  "petscsys.h"   I*/

#if defined(PETSC_USE_DEBUG)

#if defined(PETSC_HAVE_PTHREADCLASSES)
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscStack *petscstack = 0;
#endif
#else
PetscStack *petscstack = 0;
#endif


#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>

static SAWs_Directory amsmemstack = NULL;

#undef __FUNCT__
#define __FUNCT__ "PetscStackSAWsGrantAccess"
/*@C
   PetscStackSAWsGrantAccess - Grants access of the PETSc stack frames to the SAWs publisher

   Collective on PETSC_COMM_WORLD?

   Level: developer

   Concepts: publishing object

   Developers Note: Cannot use PetscFunctionBegin/Return() or PetscStackCallSAWs() since it may be used within those routines

.seealso: PetscObjectSetName(), PetscObjectSAWsViewOff(), PetscObjectSAWsTakeAccess()

@*/
void  PetscStackSAWsGrantAccess(void)
{
  if (amsmemstack) {
    SAWs_Unlock_Directory(amsmemstack);
  }
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackSAWsTakeAccess"
/*@C
   PetscStackSAWsTakeAccess - Takes access of the PETSc stack frames to the SAWs publisher

   Collective on PETSC_COMM_WORLD?

   Level: developer

   Concepts: publishing object

   Developers Note: Cannot use PetscFunctionBegin/Return() or PetscStackCallSAWs() since it may be used within those routines

.seealso: PetscObjectSetName(), PetscObjectSAWsViewOff(), PetscObjectSAWsTakeAccess()

@*/
void  PetscStackSAWsTakeAccess(void)
{
  if (amsmemstack) {
    SAWs_Lock_Directory(amsmemstack);
  }
}

PetscErrorCode PetscStackViewSAWs(void)
{
  SAWs_Directory mem;
  PetscStack*    petscstackp;

  petscstackp = (PetscStack*)PetscThreadLocalGetValue(petscstack);
  PetscStackCallSAWs(SAWs_Add_Directory,(SAWs_ROOT_DIRECTORY,"Stack",&mem));
  PetscStackCallSAWs(SAWs_Add_Variable,(mem,"functions",petscstackp->function,10,SAWs_READ,SAWs_STRING));
  PetscStackCallSAWs(SAWs_Add_Variable,(mem,"current size",&petscstackp->currentsize,1,SAWs_READ,SAWs_INT));
  amsmemstack = mem;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackSAWsViewOff"
PetscErrorCode PetscStackSAWsViewOff(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!amsmemstack) PetscFunctionReturn(0);
  ierr        = SAWs_Destroy_Directory(&amsmemstack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

PetscErrorCode PetscStackCreate(void)
{
  PetscStack *petscstack_in;
  if (PetscStackActive()) return 0;

  petscstack_in              = (PetscStack*)malloc(sizeof(PetscStack));
  petscstack_in->currentsize = 0;
  PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,petscstack_in);

#if defined(PETSC_HAVE_SAWS)
  {
  PetscBool flg = PETSC_FALSE;
  PetscOptionsHasName(NULL,"-stack_view",&flg);
  if (flg) PetscStackViewSAWs();
  }
#endif
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "PetscStackView"
PetscErrorCode  PetscStackView(FILE *file)
{
  int        i;
  PetscStack *petscstackp;

  petscstackp = (PetscStack*)PetscThreadLocalGetValue(petscstack);
  if (!file) file = PETSC_STDOUT;

  if (file == PETSC_STDOUT) {
    (*PetscErrorPrintf)("Note: The EXACT line numbers in the stack are not available,\n");
    (*PetscErrorPrintf)("      INSTEAD the line number of the start of the function\n");
    (*PetscErrorPrintf)("      is given.\n");
    for (i=petscstackp->currentsize-1; i>=0; i--) (*PetscErrorPrintf)("[%d] %s line %d %s%s\n",PetscGlobalRank,petscstackp->function[i],petscstackp->line[i],petscstackp->directory[i],petscstackp->file[i]);
  } else {
    fprintf(file,"Note: The EXACT line numbers in the stack are not available,\n");
    fprintf(file,"      INSTEAD the line number of the start of the function\n");
    fprintf(file,"      is given.\n");
    for (i=petscstackp->currentsize-1; i>=0; i--) fprintf(file,"[%d] %s line %d %s%s\n",PetscGlobalRank,petscstackp->function[i],petscstackp->line[i],petscstackp->directory[i],petscstackp->file[i]);
  }
  return 0;
}

PetscErrorCode PetscStackDestroy(void)
{
  if (PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)PetscThreadLocalGetValue(petscstack);
    free(petscstack_in);
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,(PetscStack*)0);
    PetscThreadLocalDestroy(petscstack); /* Deletes pthread_key if it was used */
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackCopy"
/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackCopy(PetscStack *sint,PetscStack *sout)
{
  int i;

  if (!sint) sout->currentsize = 0;
  else {
    for (i=0; i<sint->currentsize; i++) {
      sout->function[i]     = sint->function[i];
      sout->file[i]         = sint->file[i];
      sout->directory[i]    = sint->directory[i];
      sout->line[i]         = sint->line[i];
      sout->petscroutine[i] = sint->petscroutine[i];
    }
    sout->currentsize = sint->currentsize;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackPrint"
/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackPrint(PetscStack *sint,FILE *fp)
{
  int i;

  if (!sint) return(0);
  for (i=sint->currentsize-2; i>=0; i--) fprintf(fp,"      [%d]  %s() line %d in %s%s\n",PetscGlobalRank,sint->function[i],sint->line[i],sint->directory[i],sint->file[i]);
  return 0;
}

#else

#undef __FUNCT__
#define __FUNCT__ "PetscStackCreate"
PetscErrorCode  PetscStackCreate(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PetscStackView"
PetscErrorCode  PetscStackView(FILE *file)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PetscStackDestroy"
PetscErrorCode  PetscStackDestroy(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)     /* SAWs stack functions do nothing in optimized mode */
void PetscStackSAWsGrantAccess(void) {}
void PetscStackSAWsTakeAccess(void) {}

PetscErrorCode PetscStackViewSAWs(void)
{
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackSAWsViewOff"
PetscErrorCode  PetscStackSAWsViewOff(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#endif

