/*
    Code that allows a user to dictate what malloc() PETSc uses.
*/
#include <petscsys.h>             /*I   "petscsys.h"   I*/
#include <stdarg.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_MEMKIND)
#include <errno.h>
#include <memkind.h>
typedef enum {PETSC_MK_DEFAULT=0,PETSC_MK_HBW_PREFERRED=1} PetscMemkindType;
PetscMemkindType currentmktype = PETSC_MK_HBW_PREFERRED;
PetscMemkindType previousmktype = PETSC_MK_HBW_PREFERRED;
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

PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t mem,PetscBool clear,int line,const char func[],const char file[],void **result)
{
  if (!mem) {*result = NULL; return 0;}
#if PetscDefined(HAVE_MEMKIND)
  {
    int err;

    err = memkind_posix_memalign(currentmktype ? MEMKIND_HBW_PREFERRED : MEMKIND_DEFAULT,result,PETSC_MEMALIGN,mem);
    PetscCheck(err != EINVAL,PETSC_COMM_SELF,PETSC_ERR_MEM,"Memkind: invalid 3rd or 4th argument of memkind_posix_memalign()");
    if (err == ENOMEM) PetscInfo(NULL,"Memkind: fail to request HBW memory %.0f, falling back to normal memory\n",(PetscLogDouble)mem);
    PetscCheck(*result,PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
    if (clear) PetscCall(PetscMemzero(*result,mem));
  }
#else /* PetscDefined(HAVE_MEMKIND) */
#  if PetscDefined(HAVE_DOUBLE_ALIGN_MALLOC) && (PETSC_MEMALIGN == 8)
  if (clear) *result = calloc(1+mem/sizeof(int),sizeof(int));
  else       *result = malloc(mem);

  PetscCheck(*result,PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
  if (PetscLogMemory) PetscCall(PetscMemzero(*result,mem));

#  elif PetscDefined(HAVE_MEMALIGN)
  *result = memalign(PETSC_MEMALIGN,mem);
  PetscCheck(*result,PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
  if (clear || PetscLogMemory) PetscCall(PetscMemzero(*result,mem));
#  else /* PetscDefined(HAVE_DOUBLE_ALIGN_MALLOC) || PetscDefined(HAVE_MEMALIGN) */
  {
    int *ptr,shift;
    /*
      malloc space for two extra chunks and shift ptr 1 + enough to get it PetscScalar aligned
    */
    if (clear) {
      ptr = (int*)calloc(1+(mem + 2*PETSC_MEMALIGN)/sizeof(int),sizeof(int));
    } else {
      ptr = (int*)malloc(mem + 2*PETSC_MEMALIGN);
    }
    PetscCheck(ptr,PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
    shift        = (int)(((PETSC_UINTPTR_T) ptr) % PETSC_MEMALIGN);
    shift        = (2*PETSC_MEMALIGN - shift)/sizeof(int);
    ptr[shift-1] = shift + SHIFT_CLASSID;
    ptr         += shift;
    *result      = (void*)ptr;
    if (PetscLogMemory) PetscCall(PetscMemzero(*result,mem));
  }
#  endif /* PetscDefined(HAVE_DOUBLE_ALIGN_MALLOC) || PetscDefined(HAVE_MEMALIGN) */
#endif /* PetscDefined(HAVE_MEMKIND) */
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscFreeAlign(void *ptr,int line,const char func[],const char file[])
{
  if (!ptr) return 0;
#if PetscDefined(HAVE_MEMKIND)
  memkind_free(0,ptr); /* specify the kind to 0 so that memkind will look up for the right type */
#else /* PetscDefined(HAVE_MEMKIND) */
#  if (!(PetscDefined(HAVE_DOUBLE_ALIGN_MALLOC) && (PETSC_MEMALIGN == 8)) && !PetscDefined(HAVE_MEMALIGN))
  {
    /*
      Previous int tells us how many ints the pointer has been shifted from
      the original address provided by the system malloc().
    */
    const int shift = *(((int*)ptr)-1) - SHIFT_CLASSID;

    PetscCheck(shift <= PETSC_MEMALIGN-1,PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap");
    PetscCheck(shift >= 0,PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap");
    ptr = (void*)(((int*)ptr) - shift);
  }
#  endif

#  if PetscDefined(HAVE_FREE_RETURN_INT)
  int err = free(ptr);
  PetscCheck(!err,PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"System free returned error %d\n",err);
#  else
  free(ptr);
#  endif
#endif
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscReallocAlign(size_t mem, int line, const char func[], const char file[], void **result)
{
  if (!mem) {
    PetscCall(PetscFreeAlign(*result, line, func, file));
    *result = NULL;
    return 0;
  }
#if PetscDefined(HAVE_MEMKIND)
  *result = memkind_realloc(currentmktype ? MEMKIND_HBW_PREFERRED : MEMKIND_DEFAULT,*result,mem);
#else
#  if (!(PetscDefined(HAVE_DOUBLE_ALIGN_MALLOC) && (PETSC_MEMALIGN == 8)) && !PetscDefined(HAVE_MEMALIGN))
  {
    /*
      Previous int tells us how many ints the pointer has been shifted from
      the original address provided by the system malloc().
    */
    int shift = *(((int*)*result)-1) - SHIFT_CLASSID;
    PetscCheck(shift <= PETSC_MEMALIGN-1,PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap");
    PetscCheck(shift >= 0,PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap");
    *result = (void*)(((int*)*result) - shift);
  }
#  endif

#  if (PetscDefined(HAVE_DOUBLE_ALIGN_MALLOC) && (PETSC_MEMALIGN == 8)) || PetscDefined(HAVE_MEMALIGN)
  *result = realloc(*result, mem);
#  else
  {
    /*
      malloc space for two extra chunks and shift ptr 1 + enough to get it PetscScalar aligned
    */
    int *ptr = (int*)realloc(*result,mem + 2*PETSC_MEMALIGN);
    if (ptr) {
      int shift    = (int)(((PETSC_UINTPTR_T) ptr) % PETSC_MEMALIGN);
      shift        = (2*PETSC_MEMALIGN - shift)/sizeof(int);
      ptr[shift-1] = shift + SHIFT_CLASSID;
      ptr         += shift;
      *result      = (void*)ptr;
    } else {
      *result      = NULL;
    }
  }
#  endif
#endif
  PetscCheck(*result,PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
#if PetscDefined(HAVE_MEMALIGN)
  /* There are no standard guarantees that realloc() maintains the alignment of memalign(), so I think we have to
   * realloc and, if the alignment is wrong, malloc/copy/free. */
  if (((size_t) (*result)) % PETSC_MEMALIGN) {
    void *newResult;
#  if PetscDefined(HAVE_MEMKIND)
    {
      int err;
      err = memkind_posix_memalign(currentmktype ? MEMKIND_HBW_PREFERRED : MEMKIND_DEFAULT,&newResult,PETSC_MEMALIGN,mem);
      PetscCheck(err != EINVAL,PETSC_COMM_SELF,PETSC_ERR_MEM,"Memkind: invalid 3rd or 4th argument of memkind_posix_memalign()");
      if (err == ENOMEM) PetscInfo(NULL,"Memkind: fail to request HBW memory %.0f, falling back to normal memory\n",(PetscLogDouble)mem);
    }
#  else
    newResult = memalign(PETSC_MEMALIGN,mem);
#  endif
    PetscCheck(newResult,PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
    PetscCall(PetscMemcpy(newResult,*result,mem));
#  if PetscDefined(HAVE_FREE_RETURN_INT)
    {
      int err = free(*result);
      PetscCheck(!err,PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"System free returned error %d\n",err);
    }
#  else
#    if defined(PETSC_HAVE_MEMKIND)
    memkind_free(0,*result);
#    else
    free(*result);
#    endif
#  endif
    *result = newResult;
  }
#endif
  return 0;
}

PetscErrorCode (*PetscTrMalloc)(size_t,PetscBool,int,const char[],const char[],void**) = PetscMallocAlign;
PetscErrorCode (*PetscTrFree)(void*,int,const char[],const char[])                     = PetscFreeAlign;
PetscErrorCode (*PetscTrRealloc)(size_t,int,const char[],const char[],void**)          = PetscReallocAlign;

PETSC_INTERN PetscBool petscsetmallocvisited;
PetscBool petscsetmallocvisited = PETSC_FALSE;

/*@C
   PetscMallocSet - Sets the routines used to do mallocs and frees.
   This routine MUST be called before PetscInitialize() and may be
   called only once.

   Not Collective

   Input Parameters:
+ imalloc - the routine that provides the malloc (also provides calloc(), which is used depends on the second argument)
. ifree - the routine that provides the free
- iralloc - the routine that provides the realloc

   Level: developer

@*/
PetscErrorCode PetscMallocSet(PetscErrorCode (*imalloc)(size_t,PetscBool,int,const char[],const char[],void**),
                              PetscErrorCode (*ifree)(void*,int,const char[],const char[]),
                              PetscErrorCode (*iralloc)(size_t, int, const char[], const char[], void **))
{
  PetscFunctionBegin;
  PetscCheckFalse(petscsetmallocvisited && (imalloc != PetscTrMalloc || ifree != PetscTrFree),PETSC_COMM_SELF,PETSC_ERR_SUP,"cannot call multiple times");
  PetscTrMalloc         = imalloc;
  PetscTrFree           = ifree;
  PetscTrRealloc        = iralloc;
  petscsetmallocvisited = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocClear - Resets the routines used to do mallocs and frees to the defaults.

   Not Collective

   Level: developer

   Notes:
    In general one should never run a PETSc program with different malloc() and
    free() settings for different parts; this is because one NEVER wants to
    free() an address that was malloced by a different memory management system

    Called in PetscFinalize() so that if PetscInitialize() is called again it starts with a fresh slate of allocation information

@*/
PetscErrorCode PetscMallocClear(void)
{
  PetscFunctionBegin;
  PetscTrMalloc         = PetscMallocAlign;
  PetscTrFree           = PetscFreeAlign;
  PetscTrRealloc        = PetscReallocAlign;
  petscsetmallocvisited = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMemoryTrace(const char label[])
{
  PetscLogDouble        mem,mal;
  static PetscLogDouble oldmem = 0,oldmal = 0;

  PetscFunctionBegin;
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  PetscCall(PetscMallocGetCurrentUsage(&mal));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s High water  %8.3f MB increase %8.3f MB Current %8.3f MB increase %8.3f MB\n",label,mem*1e-6,(mem - oldmem)*1e-6,mal*1e-6,(mal - oldmal)*1e-6));
  oldmem = mem;
  oldmal = mal;
  PetscFunctionReturn(0);
}

static PetscErrorCode (*PetscTrMallocOld)(size_t,PetscBool,int,const char[],const char[],void**) = PetscMallocAlign;
static PetscErrorCode (*PetscTrReallocOld)(size_t,int,const char[],const char[],void**)          = PetscReallocAlign;
static PetscErrorCode (*PetscTrFreeOld)(void*,int,const char[],const char[])                     = PetscFreeAlign;

/*@C
   PetscMallocSetDRAM - Set PetscMalloc to use DRAM.
     If memkind is available, change the memkind type. Otherwise, switch the
     current malloc and free routines to the PetscMallocAlign and
     PetscFreeAlign (PETSc default).

   Not Collective

   Level: developer

   Notes:
     This provides a way to do the allocation on DRAM temporarily. One
     can switch back to the previous choice by calling PetscMallocReset().

.seealso: PetscMallocReset()
@*/
PetscErrorCode PetscMallocSetDRAM(void)
{
  PetscFunctionBegin;
  if (PetscTrMalloc == PetscMallocAlign) {
#if defined(PETSC_HAVE_MEMKIND)
    previousmktype = currentmktype;
    currentmktype  = PETSC_MK_DEFAULT;
#endif
  } else {
    /* Save the previous choice */
    PetscTrMallocOld  = PetscTrMalloc;
    PetscTrReallocOld = PetscTrRealloc;
    PetscTrFreeOld    = PetscTrFree;
    PetscTrMalloc     = PetscMallocAlign;
    PetscTrFree       = PetscFreeAlign;
    PetscTrRealloc    = PetscReallocAlign;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocResetDRAM - Reset the changes made by PetscMallocSetDRAM

   Not Collective

   Level: developer

.seealso: PetscMallocSetDRAM()
@*/
PetscErrorCode PetscMallocResetDRAM(void)
{
  PetscFunctionBegin;
  if (PetscTrMalloc == PetscMallocAlign) {
#if defined(PETSC_HAVE_MEMKIND)
    currentmktype = previousmktype;
#endif
  } else {
    /* Reset to the previous choice */
    PetscTrMalloc  = PetscTrMallocOld;
    PetscTrRealloc = PetscTrReallocOld;
    PetscTrFree    = PetscTrFreeOld;
  }
  PetscFunctionReturn(0);
}

static PetscBool petscmalloccoalesce =
#if defined(PETSC_USE_MALLOC_COALESCED)
  PETSC_TRUE;
#else
  PETSC_FALSE;
#endif

/*@C
   PetscMallocSetCoalesce - Use coalesced malloc when allocating groups of objects

   Not Collective

   Input Parameters:
.  coalesce - PETSC_TRUE to use coalesced malloc for multi-object allocation.

   Options Database Keys:
.  -malloc_coalesce - turn coalesced malloc on or off

   Note:
   PETSc uses coalesced malloc by default for optimized builds and not for debugging builds.  This default can be changed via the command-line option -malloc_coalesce or by calling this function.
   This function can only be called immediately after PetscInitialize()

   Level: developer

.seealso: PetscMallocA()
@*/
PetscErrorCode PetscMallocSetCoalesce(PetscBool coalesce)
{
  PetscFunctionBegin;
  petscmalloccoalesce = coalesce;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocA - Allocate and optionally clear one or more objects, possibly using coalesced malloc

   Not Collective

   Input Parameters:
+  n - number of objects to allocate (at least 1)
.  clear - use calloc() to allocate space initialized to zero
.  lineno - line number to attribute allocation (typically __LINE__)
.  function - function to attribute allocation (typically PETSC_FUNCTION_NAME)
.  filename - file name to attribute allocation (typically __FILE__)
-  bytes0 - first of n object sizes

   Output Parameters:
.  ptr0 - first of n pointers to allocate

   Notes:
   This function is not normally called directly by users, but rather via the macros PetscMalloc1(), PetscMalloc2(), or PetscCalloc1(), etc.

   Level: developer

.seealso: PetscMallocAlign(), PetscMallocSet(), PetscMalloc1(), PetscMalloc2(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5(), PetscMalloc6(), PetscMalloc7(), PetscCalloc1(), PetscCalloc2(), PetscCalloc3(), PetscCalloc4(), PetscCalloc5(), PetscCalloc6(), PetscCalloc7(), PetscFreeA()
@*/
PetscErrorCode PetscMallocA(int n,PetscBool clear,int lineno,const char *function,const char *filename,size_t bytes0,void *ptr0,...)
{
  va_list        Argp;
  size_t         bytes[8],sumbytes;
  void           **ptr[8];
  int            i;

  PetscFunctionBegin;
  PetscCheck(n <= 8,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Attempt to allocate %d objects but only 8 supported",n);
  bytes[0] = bytes0;
  ptr[0] = (void**)ptr0;
  sumbytes = (bytes0 + PETSC_MEMALIGN-1) & ~(PETSC_MEMALIGN-1);
  va_start(Argp,ptr0);
  for (i=1; i<n; i++) {
    bytes[i] = va_arg(Argp,size_t);
    ptr[i] = va_arg(Argp,void**);
    sumbytes += (bytes[i] + PETSC_MEMALIGN-1) & ~(PETSC_MEMALIGN-1);
  }
  va_end(Argp);
  if (petscmalloccoalesce) {
    char *p;
    PetscCall((*PetscTrMalloc)(sumbytes,clear,lineno,function,filename,(void**)&p));
    for (i=0; i<n; i++) {
      *ptr[i] = bytes[i] ? p : NULL;
      p = (char*)PetscAddrAlign(p + bytes[i]);
    }
  } else {
    for (i=0; i<n; i++) {
      PetscCall((*PetscTrMalloc)(bytes[i],clear,lineno,function,filename,(void**)ptr[i]));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscFreeA - Free one or more objects, possibly allocated using coalesced malloc

   Not Collective

   Input Parameters:
+  n - number of objects to free (at least 1)
.  lineno - line number to attribute deallocation (typically __LINE__)
.  function - function to attribute deallocation (typically PETSC_FUNCTION_NAME)
.  filename - file name to attribute deallocation (typically __FILE__)
-  ptr0 ... - first of n pointers to free

   Note:
   This function is not normally called directly by users, but rather via the macros PetscFree(), PetscFree2(), etc.

   The pointers are zeroed to prevent users from accidentally reusing space that has been freed.

   Level: developer

.seealso: PetscMallocAlign(), PetscMallocSet(), PetscMallocA(), PetscFree1(), PetscFree2(), PetscFree3(), PetscFree4(), PetscFree5(), PetscFree6(), PetscFree7()
@*/
PetscErrorCode PetscFreeA(int n,int lineno,const char *function,const char *filename,void *ptr0,...)
{
  va_list   Argp;
  void    **ptr[8];
  int       i;

  PetscFunctionBegin;
  PetscCheck(n <= 8,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Attempt to allocate %d objects but only up to 8 supported",n);
  ptr[0] = (void**)ptr0;
  va_start(Argp,ptr0);
  for (i=1; i<n; i++) {
    ptr[i] = va_arg(Argp,void**);
  }
  va_end(Argp);
  if (petscmalloccoalesce) {
    for (i=0; i<n; i++) {       /* Find first nonempty allocation */
      if (*ptr[i]) break;
    }
    while (--n > i) {
      *ptr[n] = NULL;
    }
    PetscCall((*PetscTrFree)(*ptr[n],lineno,function,filename));
    *ptr[n] = NULL;
  } else {
    while (--n >= 0) {
      PetscCall((*PetscTrFree)(*ptr[n],lineno,function,filename));
      *ptr[n] = NULL;
    }
  }
  PetscFunctionReturn(0);
}
