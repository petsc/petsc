
/*
     Interface to malloc() and free(). This code allows for
  logging of memory usage and some error checking
*/
#include <petscsys.h>           /*I "petscsys.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif


/*
     These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
extern PetscErrorCode  PetscMallocAlign(size_t,int,const char[],const char[],void**);
extern PetscErrorCode  PetscFreeAlign(void*,int,const char[],const char[]);
extern PetscErrorCode  PetscReallocAlign(size_t,int,const char[],const char[],void**);
extern PetscErrorCode  PetscTrMallocDefault(size_t,int,const char[],const char[],void**);
extern PetscErrorCode  PetscTrFreeDefault(void*,int,const char[],const char[]);
extern PetscErrorCode  PetscTrReallocDefault(size_t,int,const char[],const char[],void**);


#define CLASSID_VALUE  ((PetscClassId) 0xf0e0d0c9)
#define ALREADY_FREED  ((PetscClassId) 0x0f0e0d9c)

typedef struct _trSPACE {
  size_t       size;
  int          id;
  int          lineno;
  const char   *filename;
  const char   *functionname;
  PetscClassId classid;
#if defined(PETSC_USE_DEBUG)
  PetscStack   stack;
#endif
  struct _trSPACE *next,*prev;
} TRSPACE;

/* HEADER_BYTES is the number of bytes in a PetscMalloc() header.
   It is sizeof(TRSPACE) padded to be a multiple of PETSC_MEMALIGN.
*/

#define HEADER_BYTES  ((sizeof(TRSPACE)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))


/* This union is used to insure that the block passed to the user retains
   a minimum alignment of PETSC_MEMALIGN.
*/
typedef union {
  TRSPACE sp;
  char    v[HEADER_BYTES];
} TrSPACE;


static size_t    TRallocated  = 0;
static int       TRfrags      = 0;
static TRSPACE   *TRhead      = NULL;
static int       TRid         = 0;
static PetscBool TRdebugLevel = PETSC_FALSE;
static size_t    TRMaxMem     = 0;
/*
      Arrays to log information on all Mallocs
*/
static int        PetscLogMallocMax       = 10000;
static int        PetscLogMalloc          = -1;
static size_t     PetscLogMallocThreshold = 0;
static size_t     *PetscLogMallocLength;
static const char **PetscLogMallocFile,**PetscLogMallocFunction;

PetscErrorCode PetscSetUseTrMalloc_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMallocSet(PetscTrMallocDefault,PetscTrFreeDefault);CHKERRQ(ierr);
  PetscTrRealloc = PetscTrReallocDefault;

  TRallocated       = 0;
  TRfrags           = 0;
  TRhead            = NULL;
  TRid              = 0;
  TRdebugLevel      = PETSC_FALSE;
  TRMaxMem          = 0;
  PetscLogMallocMax = 10000;
  PetscLogMalloc    = -1;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocValidate - Test the memory for corruption.  This can be used to
   check for memory overwrites.

   Input Parameter:
+  line - line number where call originated.
.  function - name of function calling
-  file - file where function is

   Return value:
   The number of errors detected.

   Output Effect:
   Error messages are written to stdout.

   Level: advanced

   Notes:
    You should generally use CHKMEMQ as a short cut for calling this
    routine.

    The line, function, file are given by the C preprocessor as

    The Fortran calling sequence is simply PetscMallocValidate(ierr)

   No output is generated if there are no problems detected.

.seealso: CHKMEMQ

@*/
PetscErrorCode  PetscMallocValidate(int line,const char function[],const char file[])
{
  TRSPACE      *head,*lasthead;
  char         *a;
  PetscClassId *nend;

  PetscFunctionBegin;
  head = TRhead; lasthead = NULL;
  while (head) {
    if (head->classid != CLASSID_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at  %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      if (lasthead) (*PetscErrorPrintf)("Last intact block allocated in %s() line %d in %s\n",lasthead->functionname,lasthead->lineno,lasthead->filename);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC," ");
    }
    a    = (char*)(((TrSPACE*)head) + 1);
    nend = (PetscClassId*)(a + head->size);
    if (*nend != CLASSID_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s\n",function,line,file);
      if (*nend == ALREADY_FREED) {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p already freed\n",head->id,(PetscLogDouble)head->size,a);
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC," ");
      } else {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
        (*PetscErrorPrintf)("Memory originally allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC," ");
      }
    }
    lasthead = head;
    head     = head->next;
  }
  PetscFunctionReturn(0);
}

/*
    PetscTrMallocDefault - Malloc with tracing.

    Input Parameters:
+   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
-   filename  - file name where used.  Use __FILE__ for this

    Returns:
    double aligned pointer to requested storage, or null if not
    available.
 */
PetscErrorCode  PetscTrMallocDefault(size_t a,int lineno,const char function[],const char filename[],void **result)
{
  TRSPACE        *head;
  char           *inew;
  size_t         nsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Do not try to handle empty blocks */
  if (!a) { *result = NULL; PetscFunctionReturn(0); }

  if (TRdebugLevel) {
    ierr = PetscMallocValidate(lineno,function,filename); if (ierr) PetscFunctionReturn(ierr);
  }

  nsize = (a + (PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1);
  ierr  = PetscMallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscClassId),lineno,function,filename,(void**)&inew);CHKERRQ(ierr);

  head  = (TRSPACE*)inew;
  inew += sizeof(TrSPACE);

  if (TRhead) TRhead->prev = head;
  head->next   = TRhead;
  TRhead       = head;
  head->prev   = NULL;
  head->size   = nsize;
  head->id     = TRid;
  head->lineno = lineno;

  head->filename                 = filename;
  head->functionname             = function;
  head->classid                  = CLASSID_VALUE;
  *(PetscClassId*)(inew + nsize) = CLASSID_VALUE;

  TRallocated += nsize;
  if (TRallocated > TRMaxMem) TRMaxMem = TRallocated;
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
         Allow logging of all mallocs made
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
  *result = (void*)inew;
  PetscFunctionReturn(0);
}


/*
   PetscTrFreeDefault - Free with tracing.

   Input Parameters:
.   a    - pointer to a block allocated with PetscTrMalloc
.   lineno - line number where used.  Use __LINE__ for this
.   file  - file name where used.  Use __FILE__ for this
 */
PetscErrorCode  PetscTrFreeDefault(void *aa,int line,const char function[],const char file[])
{
  char           *a = (char*)aa;
  TRSPACE        *head;
  char           *ahead;
  PetscErrorCode ierr;
  PetscClassId   *nend;

  PetscFunctionBegin;
  /* Do not try to handle empty blocks */
  if (!a) PetscFunctionReturn(0);

  if (TRdebugLevel) {
    ierr = PetscMallocValidate(line,function,file);CHKERRQ(ierr);
  }

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE*)a;

  if (head->classid != CLASSID_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,line,file);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\nmay be block not allocated with PetscMalloc()\n",a);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (PetscClassId*)(ahead + head->size);
  if (*nend != CLASSID_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p was already freed\n",head->id,(PetscLogDouble)head->size,a + sizeof(TrSPACE));
      if (head->lineno > 0 && head->lineno < 50000 /* sanity check */) {
        (*PetscErrorPrintf)("Block freed in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,-head->lineno,head->filename);
      }
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Memory already freed");
    } else {
      /* Damaged tail */
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s\n",function,line,file);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s\n",head->functionname,head->lineno,head->filename);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEMC,"Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend = ALREADY_FREED;
  /* Save location where freed.  If we suspect the line number, mark as  allocated location */
  if (line > 0 && line < 50000) {
    head->lineno       = line;
    head->filename     = file;
    head->functionname = function;
  } else {
    head->lineno = -head->lineno;
  }
  /* zero out memory - helps to find some reuse of already freed memory */
  ierr = PetscMemzero(aa,head->size);CHKERRQ(ierr);

  TRallocated -= head->size;
  TRfrags--;
  if (head->prev) head->prev->next = head->next;
  else TRhead = head->next;

  if (head->next) head->next->prev = head->prev;
  ierr = PetscFreeAlign(a,line,function,file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*
  PetscTrReallocDefault - Realloc with tracing.

  Input Parameters:
+ len      - number of bytes to allocate
. lineno   - line number where used.  Use __LINE__ for this
. filename - file name where used.  Use __FILE__ for this
- result   - double aligned pointer to initial storage.

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
  /* Realloc to zero = free */
  if (!len) {
    ierr = PetscTrFreeDefault(*result,lineno,function,filename);CHKERRQ(ierr);
    *result = NULL;
    PetscFunctionReturn(0);
  }

  if (TRdebugLevel) {ierr = PetscMallocValidate(lineno,function,filename); if (ierr) PetscFunctionReturn(ierr);}

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

  TRallocated -= head->size;
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
  head->id     = TRid;
  head->lineno = lineno;

  head->filename                 = filename;
  head->functionname             = function;
  head->classid                  = CLASSID_VALUE;
  *(PetscClassId*)(inew + nsize) = CLASSID_VALUE;

  TRallocated += nsize;
  if (TRallocated > TRMaxMem) TRMaxMem = TRallocated;
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
         Allow logging of all mallocs made
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
    PetscMemoryView - Shows the amount of memory currently being used
        in a communicator.

    Collective on PetscViewer

    Input Parameter:
+    viewer - the viewer that defines the communicator
-    message - string printed before values

    Options Database:
+    -malloc - have PETSc track how much memory it has allocated
-    -memory_view - during PetscFinalize() have this routine called

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMemoryGetCurrentUsage(), PetscMemorySetGetMaximumUsage()
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
    ierr = MPI_Reduce(&residentmax,&gresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&residentmax,&maxgresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&residentmax,&mingresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Maximum (over computational time) process memory:        total %5.4e max %5.4e min %5.4e\n",gresidentmax,maxgresidentmax,mingresidentmax);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&gresident,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&maxgresident,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&mingresident,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current process memory:                                  total %5.4e max %5.4e min %5.4e\n",gresident,maxgresident,mingresident);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocatedmax,&gallocatedmax,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocatedmax,&maxgallocatedmax,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocatedmax,&mingallocatedmax,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Maximum (over computational time) space PetscMalloc()ed: total %5.4e max %5.4e min %5.4e\n",gallocatedmax,maxgallocatedmax,mingallocatedmax);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&gallocated,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&maxgallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&mingallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current space PetscMalloc()ed:                           total %5.4e max %5.4e min %5.4e\n",gallocated,maxgallocated,mingallocated);CHKERRQ(ierr);
  } else if (resident && residentmax) {
    ierr = MPI_Reduce(&residentmax,&gresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&residentmax,&maxgresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&residentmax,&mingresidentmax,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Maximum (over computational time) process memory:        total %5.4e max %5.4e min %5.4e\n",gresidentmax,maxgresidentmax,mingresidentmax);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&gresident,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&maxgresident,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&mingresident,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current process memory:                                  total %5.4e max %5.4e min %5.4e\n",gresident,maxgresident,mingresident);CHKERRQ(ierr);
  } else if (resident && allocated) {
    ierr = MPI_Reduce(&resident,&gresident,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&maxgresident,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&resident,&mingresident,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current process memory:                                  total %5.4e max %5.4e min %5.4e\n",gresident,maxgresident,mingresident);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&gallocated,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&maxgallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&mingallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current space PetscMalloc()ed:                           total %5.4e max %5.4e min %5.4e\n",gallocated,maxgallocated,mingallocated);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -memory_view to get maximum memory usage\n");CHKERRQ(ierr);
  } else if (allocated) {
    ierr = MPI_Reduce(&allocated,&gallocated,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&maxgallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&allocated,&mingallocated,1,MPIU_PETSCLOGDOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Current space PetscMalloc()ed:                           total %5.4e max %5.4e min %5.4e\n",gallocated,maxgallocated,mingallocated);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -memory_view to get maximum memory usage\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"OS cannot compute process memory\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -malloc to get statistics on PetscMalloc() calls\nOS cannot compute process memory\n");CHKERRQ(ierr);
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

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
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

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMemoryGetCurrentUsage()
 @*/
PetscErrorCode  PetscMallocGetMaximumUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) TRMaxMem;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
/*@C
   PetscMallocGetStack - returns a pointer to the stack for the location in the program a call to PetscMalloc() was used to obtain that memory

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.    ptr - the memory location

   Output Paramter:
.    stack - the stack indicating where the program allocated this memory

   Level: intermediate

.seealso:  PetscMallocGetCurrentUsage(), PetscMallocDumpLog()
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
   PetscMallocDump - Dumps the allocated memory blocks to a file. The information
   printed is: size of space (in bytes), address of space, id of space,
   file in which space was allocated, and line number at which it was
   allocated.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stdout is assumed.

   Options Database Key:
.  -malloc_dump - Dumps unfreed memory during call to PetscFinalize()

   Level: intermediate

   Fortran Note:
   The calling sequence in Fortran is PetscMallocDump(integer ierr)
   The fp defaults to stdout.

   Notes: uses MPI_COMM_WORLD, because this may be called in PetscFinalize() after PETSC_COMM_WORLD
          has been freed.

   Concepts: memory usage
   Concepts: memory bleeding
   Concepts: bleeding memory

.seealso:  PetscMallocGetCurrentUsage(), PetscMallocDumpLog()
@*/
PetscErrorCode  PetscMallocDump(FILE *fp)
{
  TRSPACE        *head;
  PetscInt       libAlloc = 0;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fp) fp = PETSC_STDOUT;
  head = TRhead;
  while (head) {
    PetscBool isLib;

    ierr = PetscStrcmp(head->functionname, "PetscDLLibraryOpen", &isLib);CHKERRQ(ierr);
    libAlloc += head->size;
    head = head->next;
  }
  if (TRallocated - libAlloc > 0) fprintf(fp,"[%d]Total space allocated %.0f bytes\n",rank,(PetscLogDouble)TRallocated);
  head = TRhead;
  while (head) {
    PetscBool isLib;

    ierr = PetscStrcmp(head->functionname, "PetscDLLibraryOpen", &isLib);CHKERRQ(ierr);
    if (!isLib) {
      fprintf(fp,"[%2d]%.0f bytes %s() line %d in %s\n",rank,(PetscLogDouble)head->size,head->functionname,head->lineno,head->filename);
#if defined(PETSC_USE_DEBUG)
      ierr = PetscStackPrint(&head->stack,fp);CHKERRQ(ierr);
#endif
    }
    head = head->next;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

/*@
    PetscMallocSetDumpLog - Activates logging of all calls to PetscMalloc().

    Not Collective

    Options Database Key:
+  -malloc_log <filename> - Activates PetscMallocDumpLog()
-  -malloc_log_threshold <min> - Activates logging and sets a minimum size

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocSetDumpLogThreshold()
@*/
PetscErrorCode PetscMallocSetDumpLog(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogMalloc = 0;

  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscMallocSetDumpLogThreshold - Activates logging of all calls to PetscMalloc().

    Not Collective

    Input Arguments:
.   logmin - minimum allocation size to log, or PETSC_DEFAULT

    Options Database Key:
+  -malloc_log <filename> - Activates PetscMallocDumpLog()
-  -malloc_log_threshold <min> - Activates logging and sets a minimum size

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocSetDumpLog()
@*/
PetscErrorCode PetscMallocSetDumpLogThreshold(PetscLogDouble logmin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMallocSetDumpLog();CHKERRQ(ierr);
  if (logmin < 0) logmin = 0.0; /* PETSC_DEFAULT or PETSC_DECIDE */
  PetscLogMallocThreshold = (size_t)logmin;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocGetDumpLog - Determine whether all calls to PetscMalloc() are being logged

    Not Collective

    Output Arguments
.   logging - PETSC_TRUE if logging is active

    Options Database Key:
.  -malloc_log - Activates PetscMallocDumpLog()

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog()
@*/
PetscErrorCode PetscMallocGetDumpLog(PetscBool *logging)
{

  PetscFunctionBegin;
  *logging = (PetscBool)(PetscLogMalloc >= 0);
  PetscFunctionReturn(0);
}

/*@C
    PetscMallocDumpLog - Dumps the log of all calls to PetscMalloc(); also calls
       PetscMemoryGetMaximumUsage()

    Collective on PETSC_COMM_WORLD

    Input Parameter:
.   fp - file pointer; or NULL

    Options Database Key:
.  -malloc_log - Activates PetscMallocDumpLog()

    Level: advanced

   Fortran Note:
   The calling sequence in Fortran is PetscMallocDumpLog(integer ierr)
   The fp defaults to stdout.

.seealso: PetscMallocGetCurrentUsage(), PetscMallocDump(), PetscMallocSetDumpLog()
@*/
PetscErrorCode  PetscMallocDumpLog(FILE *fp)
{
  PetscInt       i,j,n,dummy,*perm;
  size_t         *shortlength;
  int            *shortcount,err;
  PetscMPIInt    rank,size,tag = 1212 /* very bad programming */;
  PetscBool      match;
  const char     **shortfunction;
  PetscLogDouble rss;
  MPI_Status     status;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);
  /*
       Try to get the data printed in order by processor. This will only sometimes work
  */
  err = fflush(fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");

  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(&dummy,1,MPIU_INT,rank-1,tag,MPI_COMM_WORLD,&status);CHKERRQ(ierr);
  }

  if (PetscLogMalloc < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscMallocDumpLog() called without call to PetscMallocSetDumpLog() this is often due to\n                      setting the option -malloc_log AFTER PetscInitialize() with PetscOptionsInsert() or PetscOptionsInsertFile()");

  if (!fp) fp = PETSC_STDOUT;
  ierr = PetscMemoryGetMaximumUsage(&rss);CHKERRQ(ierr);
  if (rss) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %.0f maximum size of entire process %.0f\n",rank,(PetscLogDouble)TRMaxMem,rss);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %.0f OS cannot compute size of entire process\n",rank,(PetscLogDouble)TRMaxMem);CHKERRQ(ierr);
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

  ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Memory usage sorted by function\n",rank);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] %d %.0f %s()\n",rank,shortcount[perm[i]],(PetscLogDouble)shortlength[perm[i]],shortfunction[perm[i]]);CHKERRQ(ierr);
  }
  free(perm);
  free(shortlength);
  free(shortcount);
  free((char**)shortfunction);
  err = fflush(fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  if (rank != size-1) {
    ierr = MPI_Send(&dummy,1,MPIU_INT,rank+1,tag,MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

/*@
    PetscMallocDebug - Turns on/off debugging for the memory management routines.

    Not Collective

    Input Parameter:
.   level - PETSC_TRUE or PETSC_FALSE

   Level: intermediate

.seealso: CHKMEMQ(), PetscMallocValidate()
@*/
PetscErrorCode  PetscMallocDebug(PetscBool level)
{
  PetscFunctionBegin;
  TRdebugLevel = level;
  PetscFunctionReturn(0);
}

/*@
    PetscMallocGetDebug - Indicates if any PETSc is doing ANY memory debugging.

    Not Collective

    Output Parameter:
.    flg - PETSC_TRUE if any debugger

   Level: intermediate

    Note that by default, the debug version always does some debugging unless you run with -malloc no


.seealso: CHKMEMQ(), PetscMallocValidate()
@*/
PetscErrorCode  PetscMallocGetDebug(PetscBool *flg)
{
  PetscFunctionBegin;
  if (PetscTrMalloc == PetscTrMallocDefault) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}
