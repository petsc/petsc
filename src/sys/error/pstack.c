
#include <petscsys.h>        /*I  "petscsys.h"   I*/

#if defined(PETSC_USE_DEBUG)

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscStack  *petscstack = 0;
#else
PetscStack *petscstack = 0;
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscStackPublish"
PetscErrorCode  PetscStackPublish(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStackDepublish"
PetscErrorCode  PetscStackDepublish(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "PetscStackCreate"
PetscErrorCode  PetscStackCreate(void)
{
  PetscErrorCode ierr;

  PetscStack *petscstack_in;
  if (petscstack) return 0;
  
  ierr = PetscNew(PetscStack,&petscstack_in);CHKERRQ(ierr);
  petscstack_in->currentsize = 0;
  petscstack = petscstack_in;
  PetscThreadLocalSetValue(petscstack,petscstack); /* Sets the value for the pthread_key_t if it is used */
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStackView"
PetscErrorCode  PetscStackView(PetscViewer viewer)
{
  PetscErrorCode ierr;
  int  i;
  FILE *file;

  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  ierr = PetscViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);

  if (file == PETSC_STDOUT) {
    (*PetscErrorPrintf)("Note: The EXACT line numbers in the stack are not available,\n");
    (*PetscErrorPrintf)("      INSTEAD the line number of the start of the function\n");
    (*PetscErrorPrintf)("      is given.\n");
    for (i=petscstack->currentsize-1; i>=0; i--) {
      (*PetscErrorPrintf)("[%d] %s line %d %s%s\n",PetscGlobalRank,
                                                   petscstack->function[i],
                                                   petscstack->line[i],
                                                   petscstack->directory[i],
                                                   petscstack->file[i]);
    }
  } else {
    fprintf(file,"Note: The EXACT line numbers in the stack are not available,\n");
    fprintf(file,"      INSTEAD the line number of the start of the function\n");
    fprintf(file,"      is given.\n");
    for (i=petscstack->currentsize-1; i>=0; i--) {
      fprintf(file,"[%d] %s line %d %s%s\n",PetscGlobalRank,
                                            petscstack->function[i],
                                            petscstack->line[i],
                                            petscstack->directory[i],
                                            petscstack->file[i]);
    }
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStackDestroy"
/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackDestroy(void)
{
  PetscErrorCode ierr;
  if (petscstack){
    PetscStack *petscstack_in = petscstack;
    petscstack = 0;
    ierr = PetscFree(petscstack_in);CHKERRQ(ierr);
    PetscThreadLocalDestroy(petscstack); /* Deletes pthread_key if it was used */
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStackCopy"
/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackCopy(PetscStack* sint,PetscStack* sout)
{
  int i;

  if (!sint) {
    sout->currentsize = 0;
  } else {
    for (i=0; i<sint->currentsize; i++) {
      sout->function[i]  = sint->function[i];
      sout->file[i]      = sint->file[i];
      sout->directory[i] = sint->directory[i];
      sout->line[i]      = sint->line[i];
    }
    sout->currentsize = sint->currentsize;
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscStackPrint"
/*  PetscFunctionBegin;  so that make rule checkbadPetscFunctionBegin works */
PetscErrorCode  PetscStackPrint(PetscStack* sint,FILE *fp)
{
  int i;

  if (!sint) return(0);
  for (i=sint->currentsize-3; i>=0; i--) {
    fprintf(fp,"      [%d]  %s() line %d in %s%s\n",PetscGlobalRank,sint->function[i],sint->line[i],sint->directory[i],sint->file[i]);
  }
  return 0;
}

#else
#undef __FUNCT__  
#define __FUNCT__ "PetscStackPublish"
PetscErrorCode  PetscStackPublish(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PetscStackDepublish"
PetscErrorCode  PetscStackDepublish(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PetscStackCreate"
PetscErrorCode  PetscStackCreate(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PetscStackView"
PetscErrorCode  PetscStackView(PetscViewer viewer)
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

#endif

