#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stack.c,v 1.17 1999/01/04 03:24:56 bsmith Exp balay $";
#endif

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "sys.h"

#if defined(USE_PETSC_STACK)

int         petscstacksize = 0;   /* current size of stack */
int         petscstacksize_max;   /* maximum size we've allocated for */
PetscStack *petscstack = 0;

#if defined(HAVE_AMS)
/* AMS Variables */
AMS_Memory stack_mem = -1;
AMS_Comm   Petsc_AMS_Comm = -1;
int        stack_err;
char       *msg;
#endif

int PetscStackPublish(void)
{
#if defined(HAVE_AMS)
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
  ierr = AMS_Memory_add_field(stack_mem, "stack",petscstack->function ,
	                      petscstacksize,AMS_STRING, AMS_READ, AMS_COMMON, AMS_REDUCT_UNDEF);CHKERRQ(ierr);
                
  /* Publish the memory */
  ierr = AMS_Memory_publish(stack_mem);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

int PetscStackDepublish(void)
{
#if defined(HAVE_AMS)
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
  
int PetscStackCreate(int stacksize)
{
  int ierr;

  PetscStack *petscstack_in;
  if (stacksize <=0 ) return 0;
  if (petscstack) return 0;
  
  petscstack_in      = (PetscStack *) PetscMalloc(sizeof(PetscStack));CHKPTRQ(petscstack_in);
  petscstacksize     = 0;
  petscstacksize_max = stacksize;

  petscstack_in->function =(char **) PetscMalloc(stacksize*sizeof(char*));CHKPTRQ(petscstack_in->function);
  petscstack_in->line     =(int *) PetscMalloc(stacksize*sizeof(int));CHKPTRQ(petscstack_in->line);
  petscstack_in->directory=(char **) PetscMalloc(stacksize*sizeof(char*));CHKPTRQ(petscstack_in->directory);
  petscstack_in->file     =(char **) PetscMalloc(stacksize*sizeof(char*));CHKPTRQ(petscstack_in->file);

  ierr = PetscMemzero(petscstack_in->function,stacksize*sizeof(char*));CHKERRQ(ierr);
  ierr = PetscMemzero(petscstack_in->line,stacksize*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(petscstack_in->function,stacksize*sizeof(char*));CHKERRQ(ierr);
  ierr = PetscMemzero(petscstack_in->function,stacksize*sizeof(char*));CHKERRQ(ierr);

  petscstack = petscstack_in;


  return 0;
}

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
    for ( i=petscstacksize-1; i>=0; i-- ) {
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
    for ( i=petscstacksize-1; i>=0; i-- ) {
      fprintf(file,"[%d] %s line %d %s%s\n",PetscGlobalRank,
                                            petscstack->function[i],
                                            petscstack->line[i],
                                            petscstack->directory[i],
                                            petscstack->file[i]);
    }
  }
  return 0;
}

int PetscStackDestroy(void) 
{
#if defined(HAVE_AMS)
  int ierr;
  ierr = PetscStackDepublish();CHKERRQ(ierr);
#endif
  if (petscstack){
    PetscStack *petscstack_in = petscstack;
    petscstack = 0;
    PetscFree(petscstack_in->line);
    PetscFree(petscstack_in->function);
    PetscFree(petscstack_in->file);
    PetscFree(petscstack_in->directory);
    PetscFree(petscstack_in);
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

int PetscStackCreate(int stacksize)
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
