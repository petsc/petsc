
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stack.c,v 1.7 1998/04/09 04:10:48 bsmith Exp bsmith $";
#endif
/*

*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "sys.h"

#if defined(USE_PETSC_STACK)

int         petscstacksize = 0;   /* current size of stack */
int         petscstacksize_max;   /* maximum size we've allocated for */
PetscStack *petscstack = 0;

int PetscStackCreate(int stacksize)
{
  PetscStack *petscstack_in;
  if (stacksize <=0 ) return 0;
  if (petscstack) return 0;
  
  petscstack_in      = (PetscStack *) PetscMalloc(sizeof(PetscStack));CHKPTRQ(petscstack_in);
  petscstacksize     = 0;
  petscstacksize_max = stacksize;

  petscstack_in->function  = (char **) PetscMalloc(stacksize*sizeof(char*));CHKPTRQ(petscstack_in->function);
  petscstack_in->line      = (int *) PetscMalloc(stacksize*sizeof(int));CHKPTRQ(petscstack_in->line);
  petscstack_in->directory = (char **) PetscMalloc(stacksize*sizeof(char*));CHKPTRQ(petscstack_in->directory);
  petscstack_in->file      = (char **) PetscMalloc(stacksize*sizeof(char*));CHKPTRQ(petscstack_in->file);

  PetscMemzero(petscstack_in->function,stacksize*sizeof(char*));
  PetscMemzero(petscstack_in->line,stacksize*sizeof(int));
  PetscMemzero(petscstack_in->function,stacksize*sizeof(char*));
  PetscMemzero(petscstack_in->function,stacksize*sizeof(char*));

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
    for ( i=petscstacksize-1; i>=0; i-- ) {
      (*PetscErrorPrintf)("[%d] %s line %d %s%s\n",PetscGlobalRank,
                                                petscstack->function[i],
                                                petscstack->line[i],
                                                petscstack->directory[i],
                                                petscstack->file[i]);
    }
  } else {
    for ( i=petscstacksize-1; i>=0; i-- ) {
      fprintf(stdout,"[%d] %s line %d %s%s\n",PetscGlobalRank,
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
