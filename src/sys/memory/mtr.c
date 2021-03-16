
/*
     Interface to malloc() and free(). This code allows for logging of memory usage and some error checking
*/
#include <petscsys.h>           /*I "petscsys.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

/*
     These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t,PetscBool,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode PetscFreeAlign(void*,int,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscReallocAlign(size_t,int,const char[],const char[],void**);

#define CLASSID_VALUE  ((PetscClassId) 0xf0e0d0c9)
#define ALREADY_FREED  ((PetscClassId) 0x0f0e0d9c)

/*  this is the header put at the beginning of each malloc() using for tracking allocated space and checking of allocated space heap */
typedef struct _trSPACE {
  size_t          size, rsize; /* Aligned size and requested size */
  int             id;
  int             lineno;
  const char      *filename;
  const char      *functionname;
  PetscClassId    classid;
#if defined(PETSC_USE_DEBUG)
  PetscStack      stack;
#endif
  struct _trSPACE *next,*prev;
} TRSPACE;

/* HEADER_BYTES is the number of bytes in a PetscMalloc() header.
   It is sizeof(trSPACE) padded to be a multiple of PETSC_MEMALIGN.
*/
#define HEADER_BYTES  ((sizeof(TRSPACE)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

/* This union is used to insure that the block passed to the user retains
   a minimum alignment of PETSC_MEMALIGN.
*/
typedef union {
  TRSPACE sp;
  char    v[HEADER_BYTES];
} TrSPACE;

#define MAXTRMAXMEMS 50
static size_t    TRallocated          = 0;
static int       TRfrags              = 0;
static TRSPACE   *TRhead              = NULL;
static int       TRid                 = 0;
static PetscBool TRdebugLevel         = PETSC_FALSE;
static PetscBool TRdebugIinitializenan= PETSC_FALSE;
static PetscBool TRrequestedSize      = PETSC_FALSE;
static size_t    TRMaxMem             = 0;
static int       NumTRMaxMems         = 0;
static size_t    TRMaxMems[MAXTRMAXMEMS];
static int       TRMaxMemsEvents[MAXTRMAXMEMS];
/*
      Arrays to log information on mallocs for PetscMallocView()
*/
static int        PetscLogMallocMax       = 10000;
static int        PetscLogMalloc          = -1;
static size_t     PetscLogMallocThreshold = 0;
static size_t     *PetscLogMallocLength;
static const char **PetscLogMallocFile,**PetscLogMallocFunction;
static int        PetscLogMallocTrace          = -1;
static size_t     PetscLogMallocTraceThreshold = 0;
static PetscViewer PetscLogMallocTraceViewer   = NULL;

/*@C
   PetscMallocValidate - Test the memory for corruption.  This can be called at any time between PetscInitialize() and PetscFinalize()

   Input Parameters:
+  line - line number where call originated.
.  function - name of function calling
-  file - file where function is

   Return value:
   The number of errors detected.

   Options Database:.
+  -malloc_test - turns this feature on when PETSc was not configured with --with-debugging=0
-  -malloc_debug - turns this feature on anytime

   Output Effect:
   Error messages are written to stdout.

   Level: advanced

   Notes:
    This is only run if PetscMallocSetDebug() has been called which is set by -malloc_test (if debugging is turned on) or -malloc_debug (any time)

    You should generally use CHKMEMQ as a short cut for calling this  routine.

    The Fortran calling sequence is simply PetscMallocValidate(ierr)

   No output is generated if there are no problems detected.

   Developers Note:
     Uses the flg TRdebugLevel (set as the first argument to PetscMallocSetDebug()) to determine if it should run

.seealso: CHKMEMQ

@*/
PetscErrorCode  PetscMallocValidate(int line,const char function[],const char file[])
{
  TRSPACE      *head,*lasthead;
  char         *a;
  PetscClassId *nend;

  if (!TRdebugLevel) return 0;
  head = TRhead; lasthead = NULL;
  if (head && head->prev) {
    (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s\n",function,line,file);
    (*PetscErrorPrintf)("Root memory header %p has invalid back pointer %p\n",head,head->prev);
    return PETSC_ERR_MEMC;
  }
  while (head) {
    if (head->classid != CLASSID_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at  %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write before beginning of or past end of array\n");
      if (lasthead){
        a    = (char*)(((TrSPACE*)head) + 1);
        (*PetscErrorPrintf)("Last intact block [id=%d(%.0f)] at address %p allocated in %s() line %d in %s\n",lasthead->id,(PetscLogDouble)lasthead->size,a,lasthead->functionname,lasthead->lineno,lasthead->filename);
      }
      abort();
      return PETSC_ERR_MEMC;
    }
    a    = (char*)(((TrSPACE*)head) + 1);
    nend = (PetscClassId*)(a + head->size);
    if (*nend != CLASSID_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s\n",function,line,file);
      if (*nend == ALREADY_FREED) {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p already freed\n",head->id,(PetscLogDouble)head->size,a);
        return PETSC_ERR_MEMC;
      } else {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
        (*PetscErrorPrintf)("Memory originally allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
        return PETSC_ERR_MEMC;
      }
    }
    if (head->prev && head->prev != lasthead) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Backpointer %p is invalid, should be %p\n",head->prev,lasthead);
      (*PetscErrorPrintf)("Previous memory originally allocated in %s() line %d in %s\n",lasthead->functionname,lasthead->lineno,lasthead->filename);
      (*PetscErrorPrintf)("Memory originally allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      return PETSC_ERR_MEMC;
    }
    lasthead = head;
    head     = head->next;
  }
  return 0;
}

/*
    PetscTrMallocDefault - Malloc with tracing.

    Input Parameters:
+   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
-   filename  - file name where used.  Use __FILE__ for this

    Returns:
    double aligned pointer to requested storage, or null if not  available.
 */
PetscErrorCode  PetscTrMallocDefault(size_t a,PetscBool clear,int lineno,const char function[],const char filename[],void **result)
{
  TRSPACE        *head;
  char           *inew;
  size_t         nsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Do not try to handle empty blocks */
  if (!a) { *result = NULL; PetscFunctionReturn(0); }

  ierr = PetscMallocValidate(lineno,function,filename); if (ierr) PetscFunctionReturn(ierr);

  nsize = (a + (PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1);
  ierr  = PetscMallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscClassId),clear,lineno,function,filename,(void**)&inew);CHKERRQ(ierr);

  head  = (TRSPACE*)inew;
  inew += sizeof(TrSPACE);

  if (TRhead) TRhead->prev = head;
  head->next   = TRhead;
  TRhead       = head;
  head->prev   = NULL;
  head->size   = nsize;
  head->rsize  = a;
  head->id     = TRid++;
  head->lineno = lineno;

  head->filename                 = filename;
  head->functionname             = function;
  head->classid                  = CLASSID_VALUE;
  *(PetscClassId*)(inew + nsize) = CLASSID_VALUE;

  TRallocated += TRrequestedSize ? head->rsize : head->size;
  if (TRallocated > TRMaxMem) TRMaxMem = TRallocated;
  if (PetscLogMemory) {
    PetscInt i;
    for (i=0; i<NumTRMaxMems; i++) {
      if (TRallocated > TRMaxMems[i]) TRMaxMems[i] = TRallocated;
    }
  }
  TRfrags++;

#if defined(PETSC_USE_DEBUG)
  if (PetscStackActive()) {
    ierr = PetscStackCopy(petscstack,&head->stack);CHKERRQ(ierr);
    /* fix the line number to where the malloc() was called, not the PetscFunctionBegin; */
    head->stack.line[head->stack.currentsize-2] = lineno;
  } else {
    head->stack.currentsize = 0;
  }
#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL_DOUBLE)
  if (!clear && TRdebugIinitializenan) {
    size_t     i, n = a/sizeof(PetscReal);
    PetscReal *s = (PetscReal*) inew;
    /* from https://www.doc.ic.ac.uk/~eedwards/compsys/float/nan.html */
#if defined(PETSC_USE_REAL_SINGLE)
    int        nas = 0x7F800002;
#else
    PetscInt64 nas = 0x7FF0000000000002;
#endif
    for (i=0; i<n; i++) {
      memcpy(s+i,&nas,sizeof(PetscReal));
    }
  }
#endif
#endif

  /*
         Allow logging of all mallocs made.
         TODO: Currently this memory is never freed, it should be freed during PetscFinalize()
  */
  if (PetscLogMalloc > -1 && PetscLogMalloc < PetscLogMallocMax && a >= PetscLogMallocThreshold) {
    if (!PetscLogMalloc) {
      PetscLogMallocLength = (size_t*)malloc(PetscLogMallocMax*sizeof(size_t));
      if (!PetscLogMallocLength) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");

      PetscLogMallocFile = (const char**)malloc(PetscLogMallocMax*sizeof(char*));
      if (!PetscLogMallocFile) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");

      PetscLogMallocFunction = (const char**)malloc(PetscLogMallocMax*sizeof(char*));
      if (!PetscLogMallocFunction) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");
    }
    PetscLogMallocLength[PetscLogMalloc]     = nsize;
    PetscLogMallocFile[PetscLogMalloc]       = filename;
    PetscLogMallocFunction[PetscLogMalloc++] = function;
  }
  if (PetscLogMallocTrace > -1 && a >= PetscLogMallocTraceThreshold) {
    ierr = PetscViewerASCIIPrintf(PetscLogMallocTraceViewer,"Alloc %zu %s:%d (%s)\n", a, filename ? filename : "null", lineno, function ? function : "null");CHKERRQ(ierr);
  }
  *result = (void*)inew;
  PetscFunctionReturn(0);
}

/*
   PetscTrFreeDefault - Free with tracing.

   Input Parameters:
.   a    - pointer to a block allocated with PetscTrMalloc
.   lineno - line number where used.  Use __LINE__ for this
.   filename  - file name where used.  Use __FILE__ for this
 */
PetscErrorCode  PetscTrFreeDefault(void *aa,int lineno,const char function[],const char filename[])
{
  char           *a = (char*)aa;
  TRSPACE        *head;
  char           *ahead;
  size_t         asize;
  PetscErrorCode ierr;
  PetscClassId   *nend;

  PetscFunctionBegin;
  /* Do not try to handle empty blocks */
  if (!a) PetscFunctionReturn(0);

  ierr = PetscMallocValidate(lineno,function,filename);CHKERRQ(ierr);

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE*)a;

  if (head->classid != CLASSID_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,lineno,filename);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\nmay be block not allocated with PetscMalloc()\n",a);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (PetscClassId*)(ahead + head->size);
  if (*nend != CLASSID_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,lineno,filename);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p was already freed\n",head->id,(PetscLogDouble)head->size,a + sizeof(TrSPACE));
      if (head->lineno > 0 && head->lineno < 50000 /* sanity check */) {
        (*PetscErrorPrintf)("Block freed in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,-head->lineno,head->filename);
      }
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Memory already freed");
    } else {
      /* Damaged tail */
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,lineno,filename);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Corrupted memory");
    }
  }
  if (PetscLogMallocTrace > -1 && head->rsize >= PetscLogMallocTraceThreshold) {
    ierr = PetscViewerASCIIPrintf(PetscLogMallocTraceViewer, "Free  %zu %s:%d (%s)\n", head->rsize, filename ? filename : "null", lineno, function ? function : "null");CHKERRQ(ierr);
  }
  /* Mark the location freed */
  *nend = ALREADY_FREED;
  /* Save location where freed.  If we suspect the line number, mark as  allocated location */
  if (lineno > 0 && lineno < 50000) {
    head->lineno       = lineno;
    head->filename     = filename;
    head->functionname = function;
  } else {
    head->lineno = -head->lineno;
  }
  asize = TRrequestedSize ? head->rsize : head->size;
  if (TRallocated < asize) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"TRallocate is smaller than memory just freed");
  TRallocated -= asize;
  TRfrags--;
  if (head->prev) head->prev->next = head->next;
  else TRhead = head->next;

  if (head->next) head->next->prev = head->prev;
  ierr = PetscFreeAlign(a,lineno,function,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  PetscTrReallocDefault - Realloc with tracing.

  Input Parameters:
+ len      - number of bytes to allocate
. lineno   - line number where used.  Use __LINE__ for this
. filename - file name where used.  Use __FILE__ for this
- result - original memory

  Output Parameter:
. result - double aligned pointer to requested storage, or null if not available.

  Level: developer

.seealso: PetscTrMallocDefault(), PetscTrFreeDefault()
*/
PetscErrorCode PetscTrReallocDefault(size_t len, int lineno, const char function[], const char filename[], void **result)
{
  char           *a = (char *) *result;
  TRSPACE        *head;
  char           *ahead, *inew;
  PetscClassId   *nend;
  size_t         nsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Realloc requests zero space so just free the current space */
  if (!len) {
    ierr = PetscTrFreeDefault(*result,lineno,function,filename);CHKERRQ(ierr);
    *result = NULL;
    PetscFunctionReturn(0);
  }
  /* If the orginal space was NULL just use the regular malloc() */
  if (!*result) {
    ierr = PetscTrMallocDefault(len,PETSC_FALSE,lineno,function,filename,result);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscMallocValidate(lineno,function,filename); if (ierr) PetscFunctionReturn(ierr);

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *) a;
  inew  = a;

  if (head->classid != CLASSID_VALUE) {
    (*PetscErrorPrintf)("PetscTrReallocDefault() called from %s() line %d in %s\n",function,lineno,filename);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\nmay be block not allocated with PetscMalloc()\n",a);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (PetscClassId *)(ahead + head->size);
  if (*nend != CLASSID_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrReallocDefault() called from %s() line %d in %s\n",function,lineno,filename);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p was already freed\n",head->id,(PetscLogDouble)head->size,a + sizeof(TrSPACE));
      if (head->lineno > 0 && head->lineno < 50000 /* sanity check */) {
        (*PetscErrorPrintf)("Block freed in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,-head->lineno,head->filename);
      }
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Memory already freed");
    } else {
      /* Damaged tail */
      (*PetscErrorPrintf)("PetscTrReallocDefault() called from %s() line %d in %s\n",function,lineno,filename);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Corrupted memory");
    }
  }

  /* remove original reference to the memory allocated from the PETSc debugging heap */
  TRallocated -= TRrequestedSize ? head->rsize : head->size;
  TRfrags--;
  if (head->prev) head->prev->next = head->next;
  else TRhead = head->next;
  if (head->next) head->next->prev = head->prev;

  nsize = (len + (PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1);
  ierr  = PetscReallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscClassId),lineno,function,filename,(void**)&inew);CHKERRQ(ierr);

  head  = (TRSPACE*)inew;
  inew += sizeof(TrSPACE);

  if (TRhead) TRhead->prev = head;
  head->next   = TRhead;
  TRhead       = head;
  head->prev   = NULL;
  head->size   = nsize;
  head->rsize  = len;
  head->id     = TRid++;
  head->lineno = lineno;

  head->filename                 = filename;
  head->functionname             = function;
  head->classid                  = CLASSID_VALUE;
  *(PetscClassId*)(inew + nsize) = CLASSID_VALUE;

  TRallocated += TRrequestedSize ? head->rsize : head->size;
  if (TRallocated > TRMaxMem) TRMaxMem = TRallocated;
  if (PetscLogMemory) {
    PetscInt i;
    for (i=0; i<NumTRMaxMems; i++) {
      if (TRallocated > TRMaxMems[i]) TRMaxMems[i] = TRallocated;
    }
  }
  TRfrags++;

#if defined(PETSC_USE_DEBUG)
  if (PetscStackActive()) {
    ierr = PetscStackCopy(petscstack,&head->stack);CHKERRQ(ierr);
    /* fix the line number to where the malloc() was called, not the PetscFunctionBegin; */
    head->stack.line[head->stack.currentsize-2] = lineno;
  } else {
    head->stack.currentsize = 0;
  }
#endif

  /*
         Allow logging of all mallocs made. This adds a new entry to the list of allocated memory
         and does not remove the previous entry to the list hence this memory is "double counted" in PetscMallocView()
  */
  if (PetscLogMalloc > -1 && PetscLogMalloc < PetscLogMallocMax && len >= PetscLogMallocThreshold) {
    if (!PetscLogMalloc) {
      PetscLogMallocLength = (size_t*)malloc(PetscLogMallocMax*sizeof(size_t));
      if (!PetscLogMallocLength) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");

      PetscLogMallocFile = (const char**)malloc(PetscLogMallocMax*sizeof(char*));
      if (!PetscLogMallocFile) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");

      PetscLogMallocFunction = (const char**)malloc(PetscLogMallocMax*sizeof(char*));
      if (!PetscLogMallocFunction) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM," ");
    }
    PetscLogMallocLength[PetscLogMalloc]     = nsize;
    PetscLogMallocFile[PetscLogMalloc]       = filename;
    PetscLogMallocFunction[PetscLogMalloc++] = function;
  }
  *result = (void*)inew;
  PetscFunctionReturn(0);
}

/*@C
    PetscMemoryView - Shows the amount of memory currently being used in a communicator.

    Collective on PetscViewer

    Input Parameter:
+    viewer - the viewer that defines the communicator
-    message - string printed before values

    Options Database:
+    -malloc_debug - have PETSc track how much memory it has allocated
-    -memory_view - during PetscFinalize() have this routine called

    Level: intermediate

.seealso: PetscMallocDump(), PetscMemoryGetCurrentUsage(), PetscMemorySetGetMaximumUsage(), PetscMallocView()
 @*/
PetscErrorCode  PetscMemoryView(PetscViewer viewer,const char message[])
{
  PetscLogDouble allocated,allocatedmax,resident,residentmax,gallocated,gallocatedmax,gresident,gresidentmax,maxgallocated,maxgallocatedmax,maxgresident,maxgresidentmax;
  PetscLogDouble mingallocated,mingallocatedmax,mingresident,mingresidentmax;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscMallocGetCurrentUsage(&allocated);CHKERRQ(ierr);
  ierr = PetscMallocGetMaximumUsage(&allocatedmax);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&resident);CHKERRQ(ierr);
  ierr = PetscMemoryGetMaximumUsage(&residentmax);CHKERRQ(ierr);
  if (residentmax > 0) residentmax = PetscMax(resident,residentmax);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,message);CHKERRQ(ierr);
  if (resident && residentmax && allocated) {
    ierr = MPI_Reduce(&residentmax,&gresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&residentmax,&maxgresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&residentmax,&mingresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Maximum (over computational time) process memory:        total %5.4e max %5.4e min %5.4e\n",gresidentmax,maxgresidentmax,mingresidentmax);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&gresident,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&resident,&maxgresident,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&resident,&mingresident,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current process memory:                                  total %5.4e max %5.4e min %5.4e\n",gresident,maxgresident,mingresident);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocatedmax,&gallocatedmax,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocatedmax,&maxgallocatedmax,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocatedmax,&mingallocatedmax,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Maximum (over computational time) space PetscMalloc()ed: total %5.4e max %5.4e min %5.4e\n",gallocatedmax,maxgallocatedmax,mingallocatedmax);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&gallocated,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocated,&maxgallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocated,&mingallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current space PetscMalloc()ed:                           total %5.4e max %5.4e min %5.4e\n",gallocated,maxgallocated,mingallocated);CHKERRQ(ierr);
  } else if (resident && residentmax) {
    ierr = MPI_Reduce(&residentmax,&gresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&residentmax,&maxgresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&residentmax,&mingresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Maximum (over computational time) process memory:        total %5.4e max %5.4e min %5.4e\n",gresidentmax,maxgresidentmax,mingresidentmax);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&gresident,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&resident,&maxgresident,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&resident,&mingresident,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current process memory:                                  total %5.4e max %5.4e min %5.4e\n",gresident,maxgresident,mingresident);CHKERRQ(ierr);
  } else if (resident && allocated) {
    ierr = MPI_Reduce(&resident,&gresident,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&resident,&maxgresident,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&resident,&mingresident,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current process memory:                                  total %5.4e max %5.4e min %5.4e\n",gresident,maxgresident,mingresident);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&gallocated,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocated,&maxgallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocated,&mingallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current space PetscMalloc()ed:                           total %5.4e max %5.4e min %5.4e\n",gallocated,maxgallocated,mingallocated);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -memory_view to get maximum memory usage\n");CHKERRQ(ierr);
  } else if (allocated) {
    ierr = MPI_Reduce(&allocated,&gallocated,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocated,&maxgallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Reduce(&allocated,&mingallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current space PetscMalloc()ed:                           total %5.4e max %5.4e min %5.4e\n",gallocated,maxgallocated,mingallocated);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -memory_view to get maximum memory usage\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"OS cannot compute process memory\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -malloc_debug to get statistics on PetscMalloc() calls\nOS cannot compute process memory\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscMallocGetCurrentUsage - gets the current amount of memory used that was PetscMalloc()ed

    Not Collective

    Output Parameters:
.   space - number of bytes currently allocated

    Level: intermediate

.seealso: PetscMallocDump(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMemoryGetMaximumUsage()
 @*/
PetscErrorCode  PetscMallocGetCurrentUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) TRallocated;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocGetMaximumUsage - gets the maximum amount of memory used that was PetscMalloc()ed at any time
        during this run.

    Not Collective

    Output Parameters:
.   space - maximum number of bytes ever allocated at one time

    Level: intermediate

.seealso: PetscMallocDump(), PetscMallocView(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMallocPushMaximumUsage()
 @*/
PetscErrorCode  PetscMallocGetMaximumUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) TRMaxMem;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocPushMaximumUsage - Adds another event to collect the maximum memory usage over an event

    Not Collective

    Input Parameter:
.   event - an event id; this is just for error checking

    Level: developer

.seealso: PetscMallocDump(), PetscMallocView(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMallocPopMaximumUsage()
 @*/
PetscErrorCode  PetscMallocPushMaximumUsage(int event)
{
  PetscFunctionBegin;
  if (++NumTRMaxMems > MAXTRMAXMEMS) PetscFunctionReturn(0);
  TRMaxMems[NumTRMaxMems-1]       = TRallocated;
  TRMaxMemsEvents[NumTRMaxMems-1] = event;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocPopMaximumUsage - collect the maximum memory usage over an event

    Not Collective

    Input Parameter:
.   event - an event id; this is just for error checking

    Output Parameter:
.   mu - maximum amount of memory malloced during this event; high water mark relative to the beginning of the event

    Level: developer

.seealso: PetscMallocDump(), PetscMallocView(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMallocPushMaximumUsage()
 @*/
PetscErrorCode  PetscMallocPopMaximumUsage(int event,PetscLogDouble *mu)
{
  PetscFunctionBegin;
  *mu = 0;
  if (NumTRMaxMems-- > MAXTRMAXMEMS) PetscFunctionReturn(0);
  if (TRMaxMemsEvents[NumTRMaxMems] != event) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"PetscMallocPush/PopMaximumUsage() are not nested");
  *mu = TRMaxMems[NumTRMaxMems];
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
/*@C
   PetscMallocGetStack - returns a pointer to the stack for the location in the program a call to PetscMalloc() was used to obtain that memory

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.    ptr - the memory location

   Output Parameter:
.    stack - the stack indicating where the program allocated this memory

   Level: intermediate

.seealso:  PetscMallocGetCurrentUsage(), PetscMallocView()
@*/
PetscErrorCode  PetscMallocGetStack(void *ptr,PetscStack **stack)
{
  TRSPACE *head;

  PetscFunctionBegin;
  head   = (TRSPACE*) (((char*)ptr) - HEADER_BYTES);
  *stack = &head->stack;
  PetscFunctionReturn(0);
}
#else
PetscErrorCode  PetscMallocGetStack(void *ptr,void **stack)
{
  PetscFunctionBegin;
  *stack = NULL;
  PetscFunctionReturn(0);
}
#endif

/*@C
   PetscMallocDump - Dumps the currently allocated memory blocks to a file. The information
   printed is: size of space (in bytes), address of space, id of space,
   file in which space was allocated, and line number at which it was
   allocated.

   Not Collective

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stdout is assumed.

   Options Database Key:
.  -malloc_dump <optional filename> - Dumps unfreed memory during call to PetscFinalize()

   Level: intermediate

   Fortran Note:
   The calling sequence in Fortran is PetscMallocDump(integer ierr)
   The fp defaults to stdout.

   Notes:
     Uses MPI_COMM_WORLD to display rank, because this may be called in PetscFinalize() after PETSC_COMM_WORLD has been freed.

     When called in PetscFinalize() dumps only the allocations that have not been properly freed

     PetscMallocView() prints a list of all memory ever allocated

.seealso:  PetscMallocGetCurrentUsage(), PetscMallocView(), PetscMallocViewSet(), PetscMallocValidate()
@*/
PetscErrorCode  PetscMallocDump(FILE *fp)
{
  TRSPACE        *head;
  size_t         libAlloc = 0;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (!fp) fp = PETSC_STDOUT;
  head = TRhead;
  while (head) {
    libAlloc += TRrequestedSize ? head->rsize : head->size;
    head = head->next;
  }
  if (TRallocated - libAlloc > 0) fprintf(fp,"[%d]Total space allocated %.0f bytes\n",rank,(PetscLogDouble)TRallocated);
  head = TRhead;
  while (head) {
    PetscBool isLib;

    ierr = PetscStrcmp(head->functionname, "PetscDLLibraryOpen", &isLib);CHKERRQ(ierr);
    if (!isLib) {
      fprintf(fp,"[%2d]%.0f bytes %s() line %d in %s\n",rank,(PetscLogDouble) (TRrequestedSize ? head->rsize : head->size),head->functionname,head->lineno,head->filename);
#if defined(PETSC_USE_DEBUG)
      ierr = PetscStackPrint(&head->stack,fp);CHKERRQ(ierr);
#endif
    }
    head = head->next;
  }
  PetscFunctionReturn(0);
}

/*@
    PetscMallocViewSet - Activates logging of all calls to PetscMalloc() with a minimum size to view

    Not Collective

    Input Arguments:
.   logmin - minimum allocation size to log, or PETSC_DEFAULT

    Options Database Key:
+  -malloc_view <optional filename> - Activates PetscMallocView() in PetscFinalize()
.  -malloc_view_threshold <min> - Sets a minimum size if -malloc_view is used
-  -log_view_memory - view the memory usage also with the -log_view option

    Level: advanced

    Notes: Must be called after PetscMallocSetDebug()

    Uses MPI_COMM_WORLD to determine rank because PETSc communicators may not be available

.seealso: PetscMallocDump(), PetscMallocView(), PetscMallocViewSet(), PetscMallocTraceSet(), PetscMallocValidate()
@*/
PetscErrorCode PetscMallocViewSet(PetscLogDouble logmin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogMalloc = 0;
  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  if (logmin < 0) logmin = 0.0; /* PETSC_DEFAULT or PETSC_DECIDE */
  PetscLogMallocThreshold = (size_t)logmin;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocViewGet - Determine whether all calls to PetscMalloc() are being logged

    Not Collective

    Output Arguments
.   logging - PETSC_TRUE if logging is active

    Options Database Key:
.  -malloc_view <optional filename> - Activates PetscMallocView()

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocView(), PetscMallocTraceGet()
@*/
PetscErrorCode PetscMallocViewGet(PetscBool *logging)
{

  PetscFunctionBegin;
  *logging = (PetscBool)(PetscLogMalloc >= 0);
  PetscFunctionReturn(0);
}

/*@
  PetscMallocTraceSet - Trace all calls to PetscMalloc()

  Not Collective

  Input Arguments:
+ viewer - The viewer to use for tracing, or NULL to use stdout
. active - Flag to activate or deactivate tracing
- logmin - The smallest memory size that will be logged

  Note:
  The viewer should not be collective.

  Level: advanced

.seealso: PetscMallocTraceGet(), PetscMallocViewGet(), PetscMallocDump(), PetscMallocView()
@*/
PetscErrorCode PetscMallocTraceSet(PetscViewer viewer, PetscBool active, PetscLogDouble logmin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!active) {PetscLogMallocTrace = -1; PetscFunctionReturn(0);}
  PetscLogMallocTraceViewer = !viewer ? PETSC_VIEWER_STDOUT_SELF : viewer;
  PetscLogMallocTrace = 0;
  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  if (logmin < 0) logmin = 0.0; /* PETSC_DEFAULT or PETSC_DECIDE */
  PetscLogMallocTraceThreshold = (size_t) logmin;
  PetscFunctionReturn(0);
}

/*@
  PetscMallocTraceGet - Determine whether all calls to PetscMalloc() are being traced

  Not Collective

  Output Argument:
. logging - PETSC_TRUE if logging is active

  Options Database Key:
. -malloc_view <optional filename> - Activates PetscMallocView()

  Level: advanced

.seealso: PetscMallocTraceSet(), PetscMallocViewGet(), PetscMallocDump(), PetscMallocView()
@*/
PetscErrorCode PetscMallocTraceGet(PetscBool *logging)
{

  PetscFunctionBegin;
  *logging = (PetscBool) (PetscLogMallocTrace >= 0);
  PetscFunctionReturn(0);
}

/*@C
    PetscMallocView - Saves the log of all calls to PetscMalloc(); also calls
       PetscMemoryGetMaximumUsage()

    Not Collective

    Input Parameter:
.   fp - file pointer; or NULL

    Options Database Key:
.  -malloc_view <optional filename> - Activates PetscMallocView() in PetscFinalize()

    Level: advanced

   Fortran Note:
   The calling sequence in Fortran is PetscMallocView(integer ierr)
   The fp defaults to stdout.

   Notes:
     PetscMallocDump() dumps only the currently unfreed memory, this dumps all memory ever allocated

     PetscMemoryView() gives a brief summary of current memory usage

.seealso: PetscMallocGetCurrentUsage(), PetscMallocDump(), PetscMallocViewSet(), PetscMemoryView()
@*/
PetscErrorCode  PetscMallocView(FILE *fp)
{
  PetscInt       i,j,n,*perm;
  size_t         *shortlength;
  int            *shortcount,err;
  PetscMPIInt    rank;
  PetscBool      match;
  const char     **shortfunction;
  PetscLogDouble rss;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRMPI(ierr);
  err = fflush(fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");

  if (PetscLogMalloc < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscMallocView() called without call to PetscMallocViewSet() this is often due to\n                      setting the option -malloc_view AFTER PetscInitialize() with PetscOptionsInsert() or PetscOptionsInsertFile()");

  if (!fp) fp = PETSC_STDOUT;
  ierr = PetscMemoryGetMaximumUsage(&rss);CHKERRQ(ierr);
  if (rss) {
    (void) fprintf(fp,"[%d] Maximum memory PetscMalloc()ed %.0f maximum size of entire process %.0f\n",rank,(PetscLogDouble)TRMaxMem,rss);
  } else {
    (void) fprintf(fp,"[%d] Maximum memory PetscMalloc()ed %.0f OS cannot compute size of entire process\n",rank,(PetscLogDouble)TRMaxMem);
  }
  shortcount    = (int*)malloc(PetscLogMalloc*sizeof(int));if (!shortcount) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  shortlength   = (size_t*)malloc(PetscLogMalloc*sizeof(size_t));if (!shortlength) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  shortfunction = (const char**)malloc(PetscLogMalloc*sizeof(char*));if (!shortfunction) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  for (i=0,n=0; i<PetscLogMalloc; i++) {
    for (j=0; j<n; j++) {
      ierr = PetscStrcmp(shortfunction[j],PetscLogMallocFunction[i],&match);CHKERRQ(ierr);
      if (match) {
        shortlength[j] += PetscLogMallocLength[i];
        shortcount[j]++;
        goto foundit;
      }
    }
    shortfunction[n] = PetscLogMallocFunction[i];
    shortlength[n]   = PetscLogMallocLength[i];
    shortcount[n]    = 1;
    n++;
foundit:;
  }

  perm = (PetscInt*)malloc(n*sizeof(PetscInt));if (!perm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Out of memory");
  for (i=0; i<n; i++) perm[i] = i;
  ierr = PetscSortStrWithPermutation(n,(const char**)shortfunction,perm);CHKERRQ(ierr);

  (void) fprintf(fp,"[%d] Memory usage sorted by function\n",rank);
  for (i=0; i<n; i++) {
    (void) fprintf(fp,"[%d] %d %.0f %s()\n",rank,shortcount[perm[i]],(PetscLogDouble)shortlength[perm[i]],shortfunction[perm[i]]);
  }
  free(perm);
  free(shortlength);
  free(shortcount);
  free((char**)shortfunction);
  err = fflush(fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

/*@
    PetscMallocSetDebug - Set's PETSc memory debugging

    Not Collective

    Input Parameter:
+   eachcall - checks the entire heap of allocated memory for issues on each call to PetscMalloc() and PetscFree()
-   initializenan - initializes all memory with NaN to catch use of uninitialized floating point arrays

    Options Database:
+   -malloc_debug <true or false> - turns on or off debugging
.   -malloc_test - turns on all debugging if PETSc was configured with debugging including -malloc_dump, otherwise ignored
.   -malloc_view_threshold t - log only allocations larger than t
.   -malloc_dump <filename> - print a list of all memory that has not been freed
.   -malloc no - (deprecated) same as -malloc_debug no
-   -malloc_log - (deprecated) same as -malloc_view

   Level: developer

    Notes: This is called in PetscInitialize() and should not be called elsewhere

.seealso: CHKMEMQ(), PetscMallocValidate(), PetscMallocGetDebug()
@*/
PetscErrorCode PetscMallocSetDebug(PetscBool eachcall, PetscBool initializenan)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscTrMalloc == PetscTrMallocDefault) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot call this routine more than once, it can only be called in PetscInitialize()");
  ierr = PetscMallocSet(PetscTrMallocDefault,PetscTrFreeDefault,PetscTrReallocDefault);CHKERRQ(ierr);

  TRallocated         = 0;
  TRfrags             = 0;
  TRhead              = NULL;
  TRid                = 0;
  TRdebugLevel        = eachcall;
  TRMaxMem            = 0;
  PetscLogMallocMax   = 10000;
  PetscLogMalloc      = -1;
  TRdebugIinitializenan = initializenan;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocGetDebug - Indicates what PETSc memory debugging it is doing.

    Not Collective

    Output Parameters:
+    basic - doing basic debugging
.    eachcall - checks the entire memory heap at each PetscMalloc()/PetscFree()
-    initializenan - initializes memory with NaN

   Level: intermediate

   Notes:
     By default, the debug version always does some debugging unless you run with -malloc_debug no

.seealso: CHKMEMQ(), PetscMallocValidate(), PetscMallocSetDebug()
@*/
PetscErrorCode PetscMallocGetDebug(PetscBool *basic, PetscBool *eachcall, PetscBool *initializenan)
{
  PetscFunctionBegin;
  if (basic) *basic = (PetscTrMalloc == PetscTrMallocDefault) ? PETSC_TRUE : PETSC_FALSE;
  if (eachcall) *eachcall           = TRdebugLevel;
  if (initializenan) *initializenan = TRdebugIinitializenan;
  PetscFunctionReturn(0);
}

/*@
  PetscMallocLogRequestedSizeSet - Whether to log the requested or aligned memory size

  Not Collective

  Input Parameter:
. flg - PETSC_TRUE to log the requested memory size

  Options Database:
. -malloc_requested_size <bool> - Sets this flag

  Level: developer

.seealso: PetscMallocLogRequestedSizeGet(), PetscMallocViewSet()
@*/
PetscErrorCode PetscMallocLogRequestedSizeSet(PetscBool flg)
{
  PetscFunctionBegin;
  TRrequestedSize = flg;
  PetscFunctionReturn(0);
}

/*@
  PetscMallocLogRequestedSizeGet - Whether to log the requested or aligned memory size

  Not Collective

  Output Parameter:
. flg - PETSC_TRUE if we log the requested memory size

  Level: developer

.seealso: PetscMallocLogRequestedSizeSetinalSizeSet(), PetscMallocViewSet()
@*/
PetscErrorCode PetscMallocLogRequestedSizeGet(PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = TRrequestedSize;
  PetscFunctionReturn(0);
}
