#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mal.c,v 1.34 1998/04/20 19:28:40 bsmith Exp curfman $";
#endif
/*
    Code that allows a user to dictate what malloc() PETSc uses.
*/
#include "petsc.h"             /*I   "petsc.h"   I*/
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/petscfix.h"


/*
        We want to make sure that all mallocs of double or complex numbers are complex aligned.
    1) on systems with memalign() we call that routine to get an aligned memory location
    2) on systems without memalign() we 
       - allocate one sizeof(Scalar) extra space
       - we shift the pointer up slightly if needed to get Scalar aligned
       - if shifted we store at ptr[-1] the amount of shift (plus a cookie)
*/
#define SHIFT_COOKIE 456123

void *PetscMallocAlign(int mem)
{
  /* if size is not a multiple of align then don't bother aligning */
  if ((mem % sizeof(Scalar))) {
    return malloc(mem);
  }
#if defined(HAVE_DOUBLE_ALIGN_MALLOC) && !defined(USE_PETSC_COMPLEX)
  return malloc(mem);
#elif defined(HAVE_MEMALIGN)
  return memalign(sizeof(Scalar),mem);
#else
  {
    int *ptr,shift;
    /*
      malloc space for one extra Scalar and shift ptr enough to get it Scalar aligned
    */
    ptr = (int *) malloc(mem + sizeof(Scalar));
    if (!ptr) return 0;
    shift = ((unsigned long) ptr) % sizeof(Scalar);
    shift = (sizeof(Scalar) - shift)/sizeof(int);
    if (shift) {
      ptr     += shift;
      ptr[-1]  = shift + SHIFT_COOKIE ;
    }
    return (void *) ptr;
  }
#endif
}

int PetscFreeAlign(void *ptr)
{
#if defined(HAVE_DOUBLE_ALIGN_MALLOC) && !defined(USE_PETSC_COMPLEX)
  free(ptr);
  return 0;
#elif defined(HAVE_MEMALIGN)
  free(ptr);
  return 0;
#else
  int shift;
  /*
       Previous int tells us how many ints the pointer has been shifted from
    the original address provided by the system malloc().
  */
  shift = ((int *)ptr)[-1] - SHIFT_COOKIE;   
  if (shift == 1 || shift == 2 || shift == 3) {
    ptr   = (void *) (((int *) ptr) - shift);
  }
  free(ptr);
  return 0;
#endif
}

/*
    Set the default malloc and free to be the usual system versions unless using complex
*/
#if defined(USE_PETSC_COMPLEX)
void *(*PetscTrMalloc)(unsigned int,int,char*,char*,char*) = 
     (void*(*)(unsigned int,int,char*,char*,char*)) PetscMallocAlign;
int  (*PetscTrFree)(void *,int,char*,char *,char*)         = 
     (int (*)(void*,int,char*,char*,char*)) PetscFreeAlign;
#else
void *(*PetscTrMalloc)(unsigned int,int,char*,char*,char*) = 
     (void*(*)(unsigned int,int,char*,char*,char*))malloc;
int  (*PetscTrFree)(void *,int,char*,char *,char*)         = 
     (int (*)(void*,int,char*,char*,char*))free;
#endif


static int petscsetmallocvisited = 0;

#undef __FUNC__  
#define __FUNC__ "PetscSetMalloc"
/*@C
   PetscSetMalloc - Sets the routines used to do mallocs and frees.
   This routine MUST be called before PetscInitialize() and may be
   called only once.

   Not Collective

   Input Parameters:
+  malloc - the malloc routine
-  free - the free routine

.keywords: Petsc, set, malloc, free, memory allocation
@*/
int PetscSetMalloc(void *(*imalloc)(unsigned int,int,char*,char*,char*),
                   int (*ifree)(void*,int,char*,char*,char*))
{
  PetscFunctionBegin;
  if (petscsetmallocvisited) SETERRQ(PETSC_ERR_SUP,0,"cannot call multiple times");
  PetscTrMalloc               = imalloc;
  PetscTrFree                 = ifree;
  petscsetmallocvisited       = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscClearMalloc"
/*@C
   PetscClearMalloc - Clears the routines used to do mallocs and frees.

   Not Collective

.keywords: Petsc, set, malloc, free, memory allocation
@*/
int PetscClearMalloc(void)
{
  PetscFunctionBegin;
#if defined(HAVE_MEMALIGN) && defined(USE_PETSC_COMPLEX)
  PetscTrMalloc               = (void*(*)(unsigned int,int,char*,char*,char*))PetscMallocAlign;
  PetscTrFree                 = (int (*)(void*,int,char*,char*,char*))PetscFreeAlign;
#else
  PetscTrMalloc               = (void*(*)(unsigned int,int,char*,char*,char*))malloc;
  PetscTrFree                 = (int (*)(void*,int,char*,char*,char*))free;
#endif
  petscsetmallocvisited       = 0;
  PetscFunctionReturn(0);
}
