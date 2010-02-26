#define PETSC_DLL
/*
     Interface to malloc() and free(). This code allows for 
  logging of memory usage and some error checking 
*/
#include "petscsys.h"           /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif


/*
     These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMallocAlign(size_t,int,const char[],const char[],const char[],void**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFreeAlign(void*,int,const char[],const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTrMallocDefault(size_t,int,const char[],const char[],const char[],void**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTrFreeDefault(void*,int,const char[],const char[],const char[]);


#define COOKIE_VALUE   ((PetscCookie) 0xf0e0d0c9)
#define ALREADY_FREED  ((PetscCookie) 0x0f0e0d9c)

typedef struct _trSPACE {
    size_t          size;
    int             id;
    int             lineno;
    const char      *filename;
    const char      *functionname;
    const char      *dirname;
    PetscCookie     cookie;        
#if defined(PETSC_USE_DEBUG)
    PetscStack      stack;
#endif
    struct _trSPACE *next,*prev;
} TRSPACE;

/* HEADER_BYTES is the number of bytes in a PetscMalloc() header.
   It is sizeof(TRSPACE) padded to be a multiple of PETSC_MEMALIGN.
*/

#define HEADER_BYTES      (sizeof(TRSPACE)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1)


/* This union is used to insure that the block passed to the user retains
   a minimum alignment of PETSC_MEMALIGN.
*/
typedef union {
    TRSPACE sp;
    char    v[HEADER_BYTES];
} TrSPACE;


static size_t     TRallocated  = 0;
static int        TRfrags      = 0;
static TRSPACE    *TRhead      = 0;
static int        TRid         = 0;
static PetscTruth TRdebugLevel = PETSC_FALSE;
static size_t     TRMaxMem     = 0;
/*
      Arrays to log information on all Mallocs
*/
static int        PetscLogMallocMax = 10000,PetscLogMalloc = -1;
static size_t     *PetscLogMallocLength;
static const char **PetscLogMallocDirectory,**PetscLogMallocFile,**PetscLogMallocFunction;

#undef __FUNCT__  
#define __FUNCT__ "PetscSetUseTrMalloc_Private"
PetscErrorCode PetscSetUseTrMalloc_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr              = PetscMallocSet(PetscTrMallocDefault,PetscTrFreeDefault);CHKERRQ(ierr);
  TRallocated       = 0;
  TRfrags           = 0;
  TRhead            = 0;
  TRid              = 0;
  TRdebugLevel      = PETSC_FALSE;
  TRMaxMem          = 0;
  PetscLogMallocMax = 10000;
  PetscLogMalloc    = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocValidate"
/*@C
   PetscMallocValidate - Test the memory for corruption.  This can be used to
   check for memory overwrites.

   Input Parameter:
+  line - line number where call originated.
.  function - name of function calling
.  file - file where function is
-  dir - directory where function is

   Return value:
   The number of errors detected.
   
   Output Effect:
   Error messages are written to stdout.  

   Level: advanced

   Notes:
    You should generally use CHKMEMQ as a short cut for calling this 
    routine.

    The line, function, file and dir are given by the C preprocessor as 
    __LINE__, __FUNCT__, __FILE__, and __DIR__

    The Fortran calling sequence is simply PetscMallocValidate(ierr)

   No output is generated if there are no problems detected.

.seealso: CHKMEMQ

@*/
PetscErrorCode PETSC_DLLEXPORT PetscMallocValidate(int line,const char function[],const char file[],const char dir[])
{
  TRSPACE     *head,*lasthead;
  char        *a;
  PetscCookie *nend;

  PetscFunctionBegin;
  head = TRhead; lasthead = NULL;
  while (head) {
    if (head->cookie != COOKIE_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at  %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      if (lasthead)
	(*PetscErrorPrintf)("Last intact block allocated in %s() line %d in %s%s\n",lasthead->functionname,lasthead->lineno,lasthead->dirname,lasthead->filename);
      SETERRQ(PETSC_ERR_MEMC," ");
    }
    a    = (char *)(((TrSPACE*)head) + 1);
    nend = (PetscCookie *)(a + head->size);
    if (*nend != COOKIE_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s%s\n",function,line,dir,file);
      if (*nend == ALREADY_FREED) { 
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p already freed\n",head->id,(PetscLogDouble)head->size,a);
        SETERRQ(PETSC_ERR_MEMC," ");
      } else {
        (*PetscErrorPrintf)("Memory [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
        (*PetscErrorPrintf)("Memory originally allocated in %s() line %d in %s%s\n",head->functionname,head->lineno,head->dirname,head->filename);
        SETERRQ(PETSC_ERR_MEMC," ");
      }
    }
    lasthead = head;
    head = head->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTrMallocDefault"
/*
    PetscTrMallocDefault - Malloc with tracing.

    Input Parameters:
+   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNCT__ for this
.   filename  - file name where used.  Use __FILE__ for this
-   dir - directory where file is. Use __SDIR__ for this

    Returns:
    double aligned pointer to requested storage, or null if not
    available.
 */
PetscErrorCode PETSC_DLLEXPORT PetscTrMallocDefault(size_t a,int lineno,const char function[],const char filename[],const char dir[],void**result)
{
  TRSPACE        *head;
  char           *inew;
  size_t         nsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Trying to malloc zero size array");
  }

  if (TRdebugLevel) {
    ierr = PetscMallocValidate(lineno,function,filename,dir); if (ierr) PetscFunctionReturn(ierr);
  }

  nsize = (a + (PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1);
  ierr = PetscMallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscCookie),lineno,function,filename,dir,(void**)&inew);CHKERRQ(ierr);

  head   = (TRSPACE *)inew;
  inew  += sizeof(TrSPACE);

  if (TRhead) TRhead->prev = head;
  head->next     = TRhead;
  TRhead         = head;
  head->prev     = 0;
  head->size     = nsize;
  head->id       = TRid;
  head->lineno   = lineno;

  head->filename     = filename;
  head->functionname = function;
  head->dirname      = dir;
  head->cookie       = COOKIE_VALUE;
  *(PetscCookie *)(inew + nsize) = COOKIE_VALUE;

  TRallocated += nsize;
  if (TRallocated > TRMaxMem) {
    TRMaxMem   = TRallocated;
  }
  TRfrags++;

#if defined(PETSC_USE_DEBUG)
  ierr = PetscStackCopy(petscstack,&head->stack);CHKERRQ(ierr);
#endif

  /*
         Allow logging of all mallocs made
  */
  if (PetscLogMalloc > -1 && PetscLogMalloc < PetscLogMallocMax) {
    if (!PetscLogMalloc) {
      PetscLogMallocLength    = (size_t*)malloc(PetscLogMallocMax*sizeof(size_t));
      if (!PetscLogMallocLength) SETERRQ(PETSC_ERR_MEM," ");
      PetscLogMallocDirectory = (const char**)malloc(PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocDirectory) SETERRQ(PETSC_ERR_MEM," ");
      PetscLogMallocFile      = (const char**)malloc(PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFile) SETERRQ(PETSC_ERR_MEM," ");
      PetscLogMallocFunction  = (const char**)malloc(PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFunction) SETERRQ(PETSC_ERR_MEM," "); 
    }
    PetscLogMallocLength[PetscLogMalloc]      = nsize;
    PetscLogMallocDirectory[PetscLogMalloc]   = dir;
    PetscLogMallocFile[PetscLogMalloc]        = filename;
    PetscLogMallocFunction[PetscLogMalloc++]  = function; 
  }
  *result = (void*)inew;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscTrFreeDefault"
/*
   PetscTrFreeDefault - Free with tracing.

   Input Parameters:
.   a    - pointer to a block allocated with PetscTrMalloc
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNCT__ for this
.   file  - file name where used.  Use __FILE__ for this
.   dir - directory where file is. Use __SDIR__ for this
 */
PetscErrorCode PETSC_DLLEXPORT PetscTrFreeDefault(void *aa,int line,const char function[],const char file[],const char dir[])
{
  char           *a = (char*)aa;
  TRSPACE        *head;
  char           *ahead;
  PetscErrorCode ierr;
  PetscCookie    *nend;
  
  PetscFunctionBegin; 
  /* Do not try to handle empty blocks */
  if (!a) {
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    SETERRQ4(PETSC_ERR_ARG_OUTOFRANGE,"Trying to free null block: Free called from %s() line %d in %s%s\n",function,line,dir,file);
  }
  
  if (TRdebugLevel) {
    ierr = PetscMallocValidate(line,function,file,dir);CHKERRQ(ierr);
  }
  
  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *)a;
  
  if (head->cookie != COOKIE_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s%s\n",function,line,dir,file);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\nmay be block not allocated with PetscMalloc()\n",a);
    SETERRQ(PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (PetscCookie *)(ahead + head->size);
  if (*nend != COOKIE_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p was already freed\n",head->id,(PetscLogDouble)head->size,a + sizeof(TrSPACE));
      if (head->lineno > 0 && head->lineno < 50000 /* sanity check */) {
	(*PetscErrorPrintf)("Block freed in %s() line %d in %s%s\n",head->functionname,head->lineno,head->dirname,head->filename);	
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n",head->functionname,-head->lineno,head->dirname,head->filename);	
      }
      SETERRQ(PETSC_ERR_ARG_WRONG,"Memory already freed");
    } else {
      /* Damaged tail */ 
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Block [id=%d(%.0f)] at address %p is corrupted (probably write past end of array)\n",head->id,(PetscLogDouble)head->size,a);
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n",head->functionname,head->lineno,head->dirname,head->filename);
      SETERRQ(PETSC_ERR_MEMC,"Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend        = ALREADY_FREED; 
  /* Save location where freed.  If we suspect the line number, mark as  allocated location */
  if (line > 0 && line < 50000) {
    head->lineno       = line;
    head->filename     = file;
    head->functionname = function;
    head->dirname      = dir;
  } else {
    head->lineno = - head->lineno;
  }
  /* zero out memory - helps to find some reuse of already freed memory */
  ierr = PetscMemzero(aa,head->size);CHKERRQ(ierr);
  
  TRallocated -= head->size;
  TRfrags     --;
  if (head->prev) head->prev->next = head->next;
  else TRhead = head->next;
  
  if (head->next) head->next->prev = head->prev;
  ierr = PetscFreeAlign(a,line,function,file,dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscMemoryShowUsage"
/*@C
    PetscMemoryShowUsage - Shows the amount of memory currently being used 
        in a communicator.
   
    Collective on PetscViewer

    Input Parameter:
+    viewer - the viewer that defines the communicator
-    message - string printed before values

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMemoryGetCurrentUsage()
 @*/
PetscErrorCode PETSC_DLLEXPORT PetscMemoryShowUsage(PetscViewer viewer,const char message[])
{
  PetscLogDouble allocated,maximum,resident,residentmax;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscMallocGetCurrentUsage(&allocated);CHKERRQ(ierr);
  ierr = PetscMallocGetMaximumUsage(&maximum);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&resident);CHKERRQ(ierr);
  ierr = PetscMemoryGetMaximumUsage(&residentmax);CHKERRQ(ierr);
  if (residentmax > 0) residentmax = PetscMax(resident,residentmax);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,message);CHKERRQ(ierr);
  if (resident && residentmax && allocated) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Current space PetscMalloc()ed %g, max space PetscMalloced() %g\n[%d]Current process memory %g max process memory %g\n",rank,allocated,maximum,rank,resident,residentmax);CHKERRQ(ierr);
  } else if (resident && residentmax) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Run with -malloc to get statistics on PetscMalloc() calls\n[%d]Current process memory %g max process memory %g\n",rank,rank,resident,residentmax);CHKERRQ(ierr);
  } else if (resident && allocated) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Current space PetscMalloc()ed %g, max space PetscMalloced() %g\n[%d]Current process memory %g, run with -memory_info to get max memory usage\n",rank,allocated,maximum,rank,resident);CHKERRQ(ierr);
  } else if (allocated) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Current space PetscMalloc()ed %g, max space PetscMalloced() %g\n[%d]OS cannot compute process memory\n",rank,allocated,maximum,rank);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"Run with -malloc to get statistics on PetscMalloc() calls\nOS cannot compute process memory\n");CHKERRQ(ierr);    
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocGetCurrentUsage"
/*@C
    PetscMallocGetCurrentUsage - gets the current amount of memory used that was PetscMalloc()ed
   
    Not Collective

    Output Parameters:
.   space - number of bytes currently allocated

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMallocDump(), PetscMallocDumpLog(), PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(),
          PetscMemoryGetMaximumUsage()
 @*/
PetscErrorCode PETSC_DLLEXPORT PetscMallocGetCurrentUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) TRallocated;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocGetMaximumUsage"
/*@C
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
PetscErrorCode PETSC_DLLEXPORT PetscMallocGetMaximumUsage(PetscLogDouble *space)
{
  PetscFunctionBegin;
  *space = (PetscLogDouble) TRMaxMem;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocDump"
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
PetscErrorCode PETSC_DLLEXPORT PetscMallocDump(FILE *fp)
{
  TRSPACE        *head;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fp) fp = PETSC_STDOUT;
  if (TRallocated > 0) {
    fprintf(fp,"[%d]Total space allocated %.0f bytes\n",rank,(PetscLogDouble)TRallocated);
  }
  head = TRhead;
  while (head) {
    fprintf(fp,"[%2d]%.0f bytes %s() line %d in %s%s\n",rank,(PetscLogDouble)head->size,head->functionname,head->lineno,head->dirname,head->filename);
#if defined(PETSC_USE_DEBUG)
    ierr = PetscStackPrint(&head->stack,fp);CHKERRQ(ierr);
#endif
    head = head->next;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocSetDumpLog"
/*@C
    PetscMallocSetDumpLog - Activates logging of all calls to PetscMalloc().

    Not Collective

    Options Database Key:
.  -malloc_log - Activates PetscMallocDumpLog()

    Level: advanced

.seealso: PetscMallocDump(), PetscMallocDumpLog()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMallocSetDumpLog(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogMalloc = 0;
  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocDumpLog"
/*@C
    PetscMallocDumpLog - Dumps the log of all calls to PetscMalloc(); also calls
       PetscMemoryGetMaximumUsage()

    Collective on PETSC_COMM_WORLD

    Input Parameter:
.   fp - file pointer; or PETSC_NULL

    Options Database Key:
.  -malloc_log - Activates PetscMallocDumpLog()

    Level: advanced

   Fortran Note:
   The calling sequence in Fortran is PetscMallocDumpLog(integer ierr)
   The fp defaults to stdout.

.seealso: PetscMallocGetCurrentUsage(), PetscMallocDump(), PetscMallocSetDumpLog()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMallocDumpLog(FILE *fp)
{
  PetscInt       i,j,n,dummy,*perm;
  size_t         *shortlength;
  int            *shortcount,err;
  PetscMPIInt    rank,size,tag = 1212 /* very bad programming */;
  PetscTruth     match;
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
  if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on file");    

  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(&dummy,1,MPIU_INT,rank-1,tag,MPI_COMM_WORLD,&status);CHKERRQ(ierr);
  }

  if (!fp) fp = PETSC_STDOUT;
  ierr = PetscMemoryGetMaximumUsage(&rss);CHKERRQ(ierr);
  if (rss) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %.0f maximum size of entire process %.0f\n",rank,(PetscLogDouble)TRMaxMem,rss);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %.0f OS cannot compute size of entire process\n",rank,(PetscLogDouble)TRMaxMem);CHKERRQ(ierr);
  }
  if (PetscLogMalloc < 0) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"PetscMallocDumpLog() called without call to PetscMallocSetDumpLog() this is often due to\n                      setting the option -malloc_log AFTER PetscInitialize() with PetscOptionsInsert() or PetscOptionsInsertFile()");
  shortcount       = (int*)malloc(PetscLogMalloc*sizeof(int));if (!shortcount) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  shortlength      = (size_t*)malloc(PetscLogMalloc*sizeof(size_t));if (!shortlength) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  shortfunction    = (const char**)malloc(PetscLogMalloc*sizeof(char *));if (!shortfunction) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  shortfunction[0] = PetscLogMallocFunction[0];
  shortlength[0]   = PetscLogMallocLength[0]; 
  shortcount[0]    = 0;
  n = 1;
  for (i=1; i<PetscLogMalloc; i++) {
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

  perm = (PetscInt*)malloc(n*sizeof(PetscInt));if (!perm) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  for (i=0; i<n; i++) perm[i] = i;
  ierr = PetscSortStrWithPermutation(n,(const char **)shortfunction,perm);CHKERRQ(ierr);

  ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Memory usage sorted by function\n",rank);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] %d %.0f %s()\n",rank,shortcount[perm[i]],(PetscLogDouble)shortlength[perm[i]],shortfunction[perm[i]]);CHKERRQ(ierr);
  }
  free(perm);
  free(shortlength);
  free(shortcount);
  free((char **)shortfunction);
  err = fflush(fp);
  if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on file");    
  if (rank != size-1) {
    ierr = MPI_Send(&dummy,1,MPIU_INT,rank+1,tag,MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

/* ---------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocDebug"
/*@C
    PetscMallocDebug - Turns on/off debugging for the memory management routines.

    Not Collective

    Input Parameter:
.   level - PETSC_TRUE or PETSC_FALSE

   Level: intermediate

.seealso: CHKMEMQ(), PetscMallocValidate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMallocDebug(PetscTruth level)
{
  PetscFunctionBegin;
  TRdebugLevel = level;
  PetscFunctionReturn(0);
}
