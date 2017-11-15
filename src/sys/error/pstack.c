
#include <petscsys.h>        /*I  "petscsys.h"   I*/

#if defined(PETSC_USE_DEBUG)

#if defined(PETSC_HAVE_PTHREADCLASSES)
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscStack *petscstack = 0;
#else
PetscThreadKey petscstack;
#endif
#else
PetscStack *petscstack = 0;
#endif


#if defined(PETSC_HAVE_AMS)
#include <petscviewerams.h>

static AMS_Memory amsmemstack = -1;

#undef __FUNCT__
#define __FUNCT__ "PetscStackAMSGrantAccess"
/*@C
   PetscStackAMSGrantAccess - Grants access of the PETSc stack frames to the AMS publisher

   Collective on PETSC_COMM_WORLD?

   Level: developer

   Concepts: publishing object

   Developers Note: Cannot use PetscFunctionBegin/Return() or PetscStackCallAMS() since it may be used within those routines

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff(), PetscObjectAMSTakeAccess()

@*/
void  PetscStackAMSGrantAccess(void)
{
  if (amsmemstack != -1) {
    AMS_Memory_grant_access(amsmemstack);
  }
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackAMSTakeAccess"
/*@C
   PetscStackAMSTakeAccess - Takes access of the PETSc stack frames to the AMS publisher

   Collective on PETSC_COMM_WORLD?

   Level: developer

   Concepts: publishing object

   Developers Note: Cannot use PetscFunctionBegin/Return() or PetscStackCallAMS() since it may be used within those routines

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff(), PetscObjectAMSTakeAccess()

@*/
void  PetscStackAMSTakeAccess(void)
{
  if (amsmemstack != -1) {
    AMS_Memory_take_access(amsmemstack);
  }
}

PetscErrorCode PetscStackViewAMS(void)
{
  AMS_Comm       acomm;
  PetscErrorCode ierr;
  AMS_Memory     mem;
  PetscStack*    petscstackp;

  petscstackp = (PetscStack*)PetscThreadLocalGetValue(petscstack);
  ierr = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_WORLD,&acomm);CHKERRQ(ierr);
  PetscStackCallAMS(AMS_Memory_create,(acomm,"Stack",&mem));
  PetscStackCallAMS(AMS_Memory_take_access,(mem));
  PetscStackCallAMS(AMS_Memory_add_field,(mem,"functions",petscstackp->function,10,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF));
  PetscStackCallAMS(AMS_Memory_add_field,(mem,"current size",&petscstackp->currentsize,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF));
  PetscStackCallAMS(AMS_Memory_publish,(mem));
  PetscStackCallAMS(AMS_Memory_grant_access,(mem));
  amsmemstack = mem;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackAMSViewOff"
PetscErrorCode PetscStackAMSViewOff(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (amsmemstack == -1) PetscFunctionReturn(0);
  ierr        = AMS_Memory_destroy(amsmemstack);CHKERRQ(ierr);
  amsmemstack = -1;
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

#if defined(PETSC_HAVE_AMS)
  {
  PetscBool flg = PETSC_FALSE;
  PetscOptionsHasName(NULL,"-stack_view",&flg);
  if (flg) PetscStackViewAMS();
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
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,NULL);
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

#if defined(PETSC_HAVE_PTHREADCLASSES)
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL void *petscstack = 0;
#else
PetscThreadKey petscstack;
#endif
#else
void *petscstack = 0;
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscStackCreate"
PetscErrorCode  PetscStackCreate(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PetscStackView"
PetscErrorCode  PetscStackView(PETSC_UNUSED FILE *file)
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
#undef __FUNCT__
#define __FUNCT__ "PetscStackCopy"
PetscErrorCode  PetscStackCopy(PETSC_UNUSED void *sint,PETSC_UNUSED void *sout)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PetscStackPrint"
PetscErrorCode  PetscStackPrint(PETSC_UNUSED void *sint,PETSC_UNUSED FILE *fp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)     /* AMS stack functions do nothing in optimized mode */
void PetscStackAMSGrantAccess(void) {}
void PetscStackAMSTakeAccess(void) {}

PetscErrorCode PetscStackViewAMS(void)
{
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscStackAMSViewOff"
PetscErrorCode  PetscStackAMSViewOff(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#endif

