
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stack.c,v 1.4 1997/12/12 19:37:06 bsmith Exp bsmith $";
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
  if (stacksize <=0 ) return 0;
  if (petscstack) return 0;
  
  petscstack = (PetscStack *) PetscMalloc(stacksize*sizeof(PetscStack));CHKPTRQ(petscstack);
  petscstacksize = 0;
  petscstacksize_max = stacksize;

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
                                                petscstack[i].function,
                                                petscstack[i].line,
                                                petscstack[i].directory,
                                                petscstack[i].file);
    }
  } else {
    for ( i=petscstacksize-1; i>=0; i-- ) {
      fprintf(stdout,"[%d] %s line %d %s%s\n",PetscGlobalRank,
                                              petscstack[i].function,
                                              petscstack[i].line,
                                              petscstack[i].directory,
                                              petscstack[i].file);
    }
  }
  return 0;
}

int PetscStackDestroy(void) 
{
  if (petscstack) PetscFree(petscstack);
  petscstack = 0;
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

int PetscStackDestroy() 
{
  return 0;
}

#endif
