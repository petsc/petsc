#define PETSC_DLL
/*
     Interface to malloc() and free(). This code allows for 
  logging of memory usage and some error checking 
*/
#include "petsc.h"           /*I "petsc.h" I*/
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#include "petscfix.h"


/*
     These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMallocAlign(size_t,int,const char[],const char[],const char[],void**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFreeAlign(void*,int,const char[],const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTrMallocDefault(size_t,int,const char[],const char[],const char[],void**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTrFreeDefault(void*,int,const char[],const char[],const char[]);

#undef __FUNCT__  
#define __FUNCT__ "PetscSetUseTrMalloc_Private"
PetscErrorCode PetscSetUseTrMalloc_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr         = PetscSetMalloc(PetscTrMallocDefault,PetscTrFreeDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if (PETSC_SIZEOF_VOID_P == 8)
#define TR_ALIGN_BYTES      8
#define TR_ALIGN_MASK       0x7
#else
#define TR_ALIGN_BYTES      4
#define TR_ALIGN_MASK       0x3
#endif

#define COOKIE_VALUE   0xf0e0d0c9
#define ALREADY_FREED  0x0f0e0d9c
#define MAX_TR_STACK   20
#define TR_MALLOC      0x1
#define TR_FREE        0x2

typedef struct _trSPACE {
    unsigned long   size;
    int             id;
    int             lineno;
    const char      *filename;
    const char      *functionname;
    const char      *dirname;
    unsigned long   cookie;        
#if defined(PETSC_USE_DEBUG)
    PetscStack      stack;
#endif
    struct _trSPACE *next,*prev;
} TRSPACE;

/* HEADER_DOUBLES is the number of doubles in a PetscMalloc() header */
/* We have to be careful about alignment rules here */

#define HEADER_DOUBLES      sizeof(TRSPACE)/sizeof(double)+1


/* This union is used to insure that the block passed to the user is
   aligned on a double boundary */
typedef union {
    TRSPACE sp;
    double  v[HEADER_DOUBLES];
} TrSPACE;

static long       TRallocated    = 0,TRfrags = 0;
static TRSPACE    *TRhead      = 0;
static int        TRid         = 0;
static PetscTruth TRdebugLevel = PETSC_FALSE;
static long       TRMaxMem     = 0;
/*
      Arrays to log information on all Mallocs
*/
static int  PetscLogMallocMax = 10000,PetscLogMalloc = -1,*PetscLogMallocLength;
static const char **PetscLogMallocDirectory,**PetscLogMallocFile,**PetscLogMallocFunction;

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
  TRSPACE       *head;
  char          *a;
  unsigned long *nend;

  PetscFunctionBegin;
  head = TRhead;
  while (head) {
    if (head->cookie != COOKIE_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at  %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      SETERRQ(PETSC_ERR_MEMC," ");
    }
    if (head->size <=0) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at  %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      SETERRQ(PETSC_ERR_MEMC," ");
    }
    a    = (char *)(((TrSPACE*)head) + 1);
    nend = (unsigned long *)(a + head->size);
    if (nend[0] != COOKIE_VALUE) {
      (*PetscErrorPrintf)("PetscMallocValidate: error detected at %s() line %d in %s%s\n",function,line,dir,file);
      if (nend[0] == ALREADY_FREED) {
        (*PetscErrorPrintf)("Memory [id=%d(%lx)] at address %p already freed\n",head->id,head->size,a);
        SETERRQ(PETSC_ERR_MEMC," ");
      } else {
        (*PetscErrorPrintf)("Memory [id=%d(%lx)] at address %p is corrupted (probably write past end)\n",
	        head->id,head->size,a);
        (*PetscErrorPrintf)("Memory originally allocated in %s() line %d in %s%s\n",head->functionname,
                head->lineno,head->dirname,head->filename);
        SETERRQ(PETSC_ERR_MEMC," ");
      }
    }
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
  TRSPACE          *head;
  char             *inew;
  unsigned long    *nend;
  size_t           nsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TRdebugLevel) {
    ierr = PetscMallocValidate(lineno,function,filename,dir); if (ierr) PetscFunctionReturn(ierr);
  }
  if (!a) SETERRQ(PETSC_ERR_MEM_MALLOC_0,"Cannot malloc size zero");

  nsize = a;
  if (nsize & TR_ALIGN_MASK) nsize += (TR_ALIGN_BYTES - (nsize & TR_ALIGN_MASK));
  ierr = PetscMallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscScalar),lineno,function,filename,dir,(void**)&inew);CHKERRQ(ierr);

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
  nend               = (unsigned long *)(inew + nsize);
  nend[0]            = COOKIE_VALUE;

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
      PetscLogMallocLength    = (int*)malloc(PetscLogMallocMax*sizeof(int));
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
  unsigned long  *nend;
  
  PetscFunctionBegin; 
  /* Do not try to handle empty blocks */
  if (!a) {
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Trying to free null block");
  }
  
  if (TRdebugLevel) {
    ierr = PetscMallocValidate(line,function,file,dir);CHKERRQ(ierr);
  }
  
  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *)a;
  
  if (head->cookie != COOKIE_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s%s\n",function,line,dir,file);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\n\
may be block not allocated with PetscTrMalloc or PetscMalloc\n",a);
    SETERRQ(PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (unsigned long *)(ahead + head->size);
  if (*nend != COOKIE_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Block [id=%d(%lx)] at address %p was already freed\n",
                          head->id,head->size,a + sizeof(TrSPACE));
      if (head->lineno > 0 && head->lineno < 5000 /* sanity check */) {
	(*PetscErrorPrintf)("Block freed in %s() line %d in %s%s\n",head->functionname,
                            head->lineno,head->dirname,head->filename);	
      } else {
        (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n",head->functionname,
                            -head->lineno,head->dirname,head->filename);	
      }
      SETERRQ(PETSC_ERR_ARG_WRONG,"Memory already freed");
    } else {
      /* Damaged tail */ 
      (*PetscErrorPrintf)("PetscTrFreeDefault() called from %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n",
                          head->id,head->size,a);
      (*PetscErrorPrintf)("Block allocated in %s() line %d in %s%s\n",head->functionname,
                          head->lineno,head->dirname,head->filename);
      SETERRQ(PETSC_ERR_MEMC,"Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend        = ALREADY_FREED; 
  /* Save location where freed.  If we suspect the line number, mark as 
     allocated location */
  if (line > 0 && line < 50000) {
    head->lineno       = line;
    head->filename     = file;
    head->functionname = function;
    head->dirname      = dir;
  } else {
    head->lineno = - head->lineno;
  }
  /* zero out memory - helps to find some reuse of already freed memory */
  ierr = PetscMemzero(aa,(int)(head->size));CHKERRQ(ierr);
  
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
/*@
    PetscMemoryShowUsage - Shows the amount of memory currently being used 
        in a communicator.
   
    Collective on PetscViewer

    Input Parameter:
+    viewer - the viewer that defines the communicator
-    message - string printed before values

    Level: intermediate

    Concepts: memory usage

.seealso: PetscMemoryDump(), PetscMemoryGetCurrentUsage()
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

.seealso:  PetscMallocGetCurrentSize(), PetscMallocDumpLog() 
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMallocDump(FILE *fp)
{
  TRSPACE        *head;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fp) fp = stdout;
  if (TRallocated > 0) {
    fprintf(fp,"[%d]Total space allocated %d bytes\n",rank,(int)TRallocated);
  }
  head = TRhead;
  while (head) {
    fprintf(fp,"[%2d]%d bytes %s() line %d in %s%s\n",rank,(int)head->size,
            head->functionname,head->lineno,head->dirname,head->filename);
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
  PetscFunctionBegin;
  PetscLogMalloc = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMallocDumpLog"
/*@C
    PetscMallocDumpLog - Dumps the log of all calls to PetscMalloc(); also calls 
       PetscMemoryGetCurrentUsage() and PetscMemoryGetMaximumUsage()

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
  PetscInt       i,j,n,*shortlength,dummy,*perm;
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
  fflush(fp);
  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(&dummy,1,MPIU_INT,rank-1,tag,MPI_COMM_WORLD,&status);CHKERRQ(ierr);
  }

  if (!fp) fp = stdout;
  ierr = PetscMemoryGetCurrentUsage(&rss);CHKERRQ(ierr);
  if (rss) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %D maximum size of entire process %D\n",rank,(PetscInt)TRMaxMem,(PetscInt)rss);CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory PetscMalloc()ed %D OS cannot compute size of entire process\n",rank,(PetscInt)TRMaxMem);CHKERRQ(ierr);
  }
  shortlength      = (PetscInt*)malloc(PetscLogMalloc*sizeof(PetscInt));if (!shortlength) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  shortfunction    = (const char**)malloc(PetscLogMalloc*sizeof(char *));if (!shortfunction) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  shortfunction[0] = PetscLogMallocFunction[0];
  shortlength[0]   = PetscLogMallocLength[0]; 
  n = 1;
  for (i=1; i<PetscLogMalloc; i++) {
    for (j=0; j<n; j++) {
      ierr = PetscStrcmp(shortfunction[j],PetscLogMallocFunction[i],&match);CHKERRQ(ierr);
      if (match) {
        shortlength[j] += PetscLogMallocLength[i];
        goto foundit;
      }
    }
    shortfunction[n] = PetscLogMallocFunction[i];
    shortlength[n]   = PetscLogMallocLength[i]; 
    n++;
    foundit:;
  }

  perm = (PetscInt*)malloc(n*sizeof(PetscInt));if (!perm) SETERRQ(PETSC_ERR_MEM,"Out of memory");
  for (i=0; i<n; i++) perm[i] = i;
  ierr = PetscSortStrWithPermutation(n,(const char **)shortfunction,perm);CHKERRQ(ierr);

  ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Memory usage sorted by function\n",rank);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] % 10d %s()\n",rank,shortlength[perm[i]],shortfunction[perm[i]]);CHKERRQ(ierr);
  }
  free(perm);
  free(shortlength);
  free((char **)shortfunction);
  fflush(fp);
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
