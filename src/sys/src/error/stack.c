#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stack.c,v 1.21 1999/06/30 23:49:24 balay Exp bsmith $";
#endif

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "sys.h"

#if defined(PETSC_USE_STACK)

PetscStack *petscstack = 0;

#if defined(PETSC_HAVE_AMS)
/* AMS Variables */
AMS_Memory stack_mem      = -1;
AMS_Comm   Petsc_AMS_Comm = -1;
int        stack_err;
char       *msg;
#endif

#undef __FUNC__  
#define __FUNC__ "PetscStackPublish"
int PetscStackPublish(void)
{
#if defined(PETSC_HAVE_AMS)
  /*
        Publishes the stack to AMS
  */
  int      ierr;
  AMS_Comm acomm;

  PetscFunctionBegin;
  if (!petscstack) SETERRQ(1,1,"Stack not available to publish");
  ierr = ViewerAMSGetAMSComm(VIEWER_AMS_WORLD,&acomm);CHKERRQ(ierr);
  ierr = AMS_Memory_create(acomm, "stack_memory", &stack_mem);CHKERRQ(ierr);
         
  /* Add a field to the memory */
  ierr = AMS_Memory_add_field(stack_mem, "stack",petscstack->function,petscstack->currentsize,
                              AMS_STRING, AMS_READ, AMS_COMMON, AMS_REDUCT_UNDEF);CHKERRQ(ierr);
                
  /* Publish the memory */
  ierr = AMS_Memory_publish(stack_mem);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscStackDepublish"
int PetscStackDepublish(void)
{
#if defined(PETSC_HAVE_AMS)
  int ierr;

  PetscFunctionBegin;
  if (stack_mem >= 0) {
    ierr      = AMS_Memory_destroy(stack_mem);CHKERRQ(ierr);
    stack_mem = -1;
  }
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ "PetscStackCreate"
int PetscStackCreate(void)
{
  int ierr;

  PetscStack *petscstack_in;
  if (petscstack) return 0;
  
  petscstack_in              = (PetscStack *) PetscMalloc(sizeof(PetscStack));CHKPTRQ(petscstack_in);
  ierr = PetscMemzero(petscstack_in,sizeof(PetscStack));CHKERRQ(ierr);
  petscstack_in->currentsize = 0;
  petscstack = petscstack_in;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStackView"
int PetscStackView(Viewer viewer)
{
  int  i,ierr;
  FILE *file;

  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  ierr = ViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);

  if (file == stderr) {
    (*PetscErrorPrintf)("Note: The EXACT line numbers in the stack are not available,\n");
    (*PetscErrorPrintf)("      INSTEAD the line number of the start of the function\n");
    (*PetscErrorPrintf)("      is given.\n");
    for ( i=petscstack->currentsize-1; i>=0; i-- ) {
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
    for ( i=petscstack->currentsize-1; i>=0; i-- ) {
      fprintf(file,"[%d] %s line %d %s%s\n",PetscGlobalRank,
                                            petscstack->function[i],
                                            petscstack->line[i],
                                            petscstack->directory[i],
                                            petscstack->file[i]);
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStackDestroy"
int PetscStackDestroy(void) 
{
  int ierr;
#if defined(PETSC_HAVE_AMS)
  ierr = PetscStackDepublish();CHKERRQ(ierr);
#endif
  if (petscstack){
    PetscStack *petscstack_in = petscstack;
    petscstack = 0;
    ierr = PetscFree(petscstack_in);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscStackCopy"
int PetscStackCopy(PetscStack* sint,PetscStack* sout)
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


#undef __FUNC__  
#define __FUNC__ "PetscStackPrint"
int PetscStackPrint(PetscStack* sint,FILE *fp)
{
  int i;

  if (!sint) return(0);
  for (i=sint->currentsize-3; i>=0; i--) {
    fprintf(fp,"      [%d]  %s() line %d in %s%s\n",PetscGlobalRank,sint->function[i],sint->line[i],
            sint->directory[i],sint->file[i]);
  }
  return 0;
}

#else
int PetscStackPublish(void)
{
  return 0;
}
int PetscStackDepublish(void)
{
  return 0;
}
int PetscStackCreate(void)
{
  return 0;
}
int PetscStackView(Viewer viewer)
{
  return 0;
}
int PetscStackDestroy(void) 
{
  return 0;
}

#endif
