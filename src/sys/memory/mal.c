
/*
    Code that allows a user to dictate what malloc() PETSc uses.
*/
#include <petscsys.h>             /*I   "petscsys.h"   I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

/*
        We want to make sure that all mallocs of double or complex numbers are complex aligned.
    1) on systems with memalign() we call that routine to get an aligned memory location
    2) on systems without memalign() we 
       - allocate one sizeof(PetscScalar) extra space
       - we shift the pointer up slightly if needed to get PetscScalar aligned
       - if shifted we store at ptr[-1] the amount of shift (plus a classid)
*/
#define SHIFT_CLASSID 456123

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocAlign"
PetscErrorCode  PetscMallocAlign(size_t mem,int line,const char func[],const char file[],const char dir[],void** result)
{
#if defined(PETSC_HAVE_DOUBLE_ALIGN_MALLOC) && (PETSC_MEMALIGN == 8)
  *result = malloc(mem);
#elif defined(PETSC_HAVE_MEMALIGN)
  *result = memalign(PETSC_MEMALIGN,mem);
#else
  {
    int *ptr,shift;
    /*
      malloc space for two extra chunks and shift ptr 1 + enough to get it PetscScalar aligned
    */
    ptr = (int*)malloc(mem + 2*PETSC_MEMALIGN);
    if (ptr) {
      shift        = (int)(((PETSC_UINTPTR_T) ptr) % PETSC_MEMALIGN);
      shift        = (2*PETSC_MEMALIGN - shift)/sizeof(int);
      ptr[shift-1] = shift + SHIFT_CLASSID ;
      ptr         += shift;
      *result      = (void*)ptr;
    } else {
      *result      = 0;
    }
  }
#endif
  if (!*result)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEM,"Memory requested %.0f",(PetscLogDouble)mem); 
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscFreeAlign"
PetscErrorCode  PetscFreeAlign(void *ptr,int line,const char func[],const char file[],const char dir[])
{
#if (!(defined(PETSC_HAVE_DOUBLE_ALIGN_MALLOC) && (PETSC_MEMALIGN == 8)) && !defined(PETSC_HAVE_MEMALIGN))
  int shift;
  /*
       Previous int tells us how many ints the pointer has been shifted from
    the original address provided by the system malloc().
  */
  shift = *(((int*)ptr)-1) - SHIFT_CLASSID;   
  if (shift > PETSC_MEMALIGN-1) return PetscError(PETSC_COMM_SELF,line,func,file,dir,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap");
  if (shift < 0) return PetscError(PETSC_COMM_SELF,line,func,file,dir,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap");
  ptr   = (void*)(((int*)ptr) - shift);
#endif

#if defined(PETSC_HAVE_FREE_RETURN_INT)
  int err = free(ptr); 
  if (err) {
    return PetscError(PETSC_COMM_SELF,line,func,file,dir,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"System free returned error %d\n",err);
  }
#else 
  free(ptr);
#endif
  return 0;
}

/*
        We never use the system free directly because on many machines it 
    does not return an error code.
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscFreeDefault"
PetscErrorCode  PetscFreeDefault(void *ptr,int line,char *func,char *file,char *dir)
{
#if defined(PETSC_HAVE_FREE_RETURN_INT)
  int err = free(ptr); 
  if (err) {
    return PetscError(PETSC_COMM_SELF,line,func,file,dir,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"System free returned error %d\n",err);
  }
#else 
  free(ptr);
#endif
  return 0;
}

PetscErrorCode   (*PetscTrMalloc)(size_t,int,const char[],const char[],const char[],void**) = PetscMallocAlign;
PetscErrorCode   (*PetscTrFree)(void*,int,const char[],const char[],const char[])          = PetscFreeAlign;

PetscBool  petscsetmallocvisited = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocSet"
/*@C
   PetscMallocSet - Sets the routines used to do mallocs and frees.
   This routine MUST be called before PetscInitialize() and may be
   called only once.

   Not Collective

   Input Parameters:
+  malloc - the malloc routine
-  free - the free routine

   Level: developer

   Concepts: malloc
   Concepts: memory^allocation 

@*/
PetscErrorCode  PetscMallocSet(PetscErrorCode (*imalloc)(size_t,int,const char[],const char[],const char[],void**),
                                              PetscErrorCode (*ifree)(void*,int,const char[],const char[],const char[]))
{
  PetscFunctionBegin;
  if (petscsetmallocvisited && (imalloc != PetscTrMalloc || ifree != PetscTrFree)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cannot call multiple times");
  PetscTrMalloc               = imalloc;
  PetscTrFree                 = ifree;
  petscsetmallocvisited       = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocClear"
/*@C
   PetscMallocClear - Resets the routines used to do mallocs and frees to the 
        defaults.

   Not Collective

   Level: developer

   Notes:
    In general one should never run a PETSc program with different malloc() and 
    free() settings for different parts; this is because one NEVER wants to 
    free() an address that was malloced by a different memory management system

@*/
PetscErrorCode  PetscMallocClear(void)
{
  PetscFunctionBegin;
  PetscTrMalloc         = PetscMallocAlign;
  PetscTrFree           = PetscFreeAlign;
  petscsetmallocvisited = PETSC_FALSE;
  PetscFunctionReturn(0);
}
