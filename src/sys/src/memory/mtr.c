/*$Id: mtr.c,v 1.157 2001/08/07 03:02:00 balay Exp $*/
/*
     Interface to malloc() and free(). This code allows for 
  logging of memory usage and some error checking 
*/
#include "petsc.h"           /*I "petsc.h" I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "petscfix.h"


/*
     These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
EXTERN int   PetscMallocAlign(size_t,int,char*,char*,char*,void**);
EXTERN int   PetscFreeAlign(void*,int,char*,char*,char*);
EXTERN int   PetscTrMallocDefault(size_t,int,char*,char*,char*,void**);
EXTERN int   PetscTrFreeDefault(void*,int,char*,char*,char*);

/*
  Code for checking if a pointer is out of the range 
  of malloced memory. This will only work on flat memory models and 
  even then is suspicious.
*/
#if (PETSC_SIZEOF_VOID_P == 8)
void *PetscLow = (void*)0x0 ,*PetscHigh = (void*)0xEEEEEEEEEEEEEEEE;
#else
void *PetscLow  = (void*)0x0,*PetscHigh = (void*)0xEEEEEEEE;  
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscSetUseTrMalloc_Private"
int PetscSetUseTrMalloc_Private(void)
{
  int ierr;

  PetscFunctionBegin;
#if (PETSC_SIZEOF_VOID_P == 8)
  PetscLow     = (void*)0xEEEEEEEEEEEEEEEE;
#else
  PetscLow     = (void*)0xEEEEEEEE; 
#endif
  PetscHigh    = (void*)0x0; 
  ierr         = PetscSetMalloc(PetscTrMallocDefault,PetscTrFreeDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    PetscTrSpace - Routines for tracing space usage.

    Description:
    PetscTrMalloc replaces malloc and PetscTrFree replaces free.  These routines
    have the same syntax and semantics as the routines that they replace,
    In addition, there are routines to report statistics on the memory
    usage, and to report the currently allocated space.  These routines
    are built on top of malloc and free, and can be used together with
    them as long as any space allocated with PetscTrMalloc is only freed with
    PetscTrFree.
 */


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
    char            *filename;
    char            *functionname;
    char            *dirname;
    unsigned long   cookie;        
#if defined(PETSC_USE_STACK)
    PetscStack      stack;
#endif
    struct _trSPACE *next,*prev;
} TRSPACE;

/* HEADER_DOUBLES is the number of doubles in a PetscTrSpace header */
/* We have to be careful about alignment rules here */

#define HEADER_DOUBLES      sizeof(TRSPACE)/sizeof(double)+1


/* This union is used to insure that the block passed to the user is
   aligned on a double boundary */
typedef union {
    TRSPACE sp;
    double  v[HEADER_DOUBLES];
} TrSPACE;

static long    TRallocated    = 0,TRfrags = 0;
static TRSPACE *TRhead      = 0;
static int     TRid         = 0;
static int     TRdebugLevel = 0;
static long    TRMaxMem     = 0;
/*
      Arrays to log information on all Mallocs
*/
static int  PetscLogMallocMax = 10000,PetscLogMalloc = -1,*PetscLogMallocLength;
static char **PetscLogMallocDirectory,**PetscLogMallocFile,**PetscLogMallocFunction;

#undef __FUNCT__  
#define __FUNCT__ "PetscTrValid"
/*@C
   PetscTrValid - Test the memory for corruption.  This can be used to
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
    You should generally use CHKMEMQ or CHKMEMA as a short cut for calling this 
    routine.

    The line, function, file and dir are given by the C preprocessor as 
    __LINE__, __FUNCT__, __FILE__, and __DIR__

    The Fortran calling sequence is simply PetscTrValid(ierr)

   No output is generated if there are no problems detected.

.seealso: CHKMEMQ, CHKMEMA

@*/
int PetscTrValid(int line,const char function[],const char file[],const char dir[])
{
  TRSPACE  *head;
  char     *a;
  unsigned long *nend;

  PetscFunctionBegin;
  head = TRhead;
  while (head) {
    if (head->cookie != COOKIE_VALUE) {
      (*PetscErrorPrintf)("error detected at  %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      SETERRQ(PETSC_ERR_MEMC," ");
    }
    if (head->size <=0) {
      (*PetscErrorPrintf)("error detected at  %s() line %d in %s%s\n",function,line,dir,file);
      (*PetscErrorPrintf)("Memory at address %p is corrupted\n",head);
      (*PetscErrorPrintf)("Probably write past beginning or end of array\n");
      SETERRQ(PETSC_ERR_MEMC," ");
    }
    a    = (char *)(((TrSPACE*)head) + 1);
    nend = (unsigned long *)(a + head->size);
    if (nend[0] != COOKIE_VALUE) {
      (*PetscErrorPrintf)("error detected at %s() line %d in %s%s\n",function,line,dir,file);
      if (nend[0] == ALREADY_FREED) {
        (*PetscErrorPrintf)("Memory [id=%d(%lx)] at address %p already freed\n",head->id,head->size,a);
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
int PetscTrMallocDefault(size_t a,int lineno,char *function,char *filename,char *dir,void**result)
{
  TRSPACE          *head;
  char             *inew;
  unsigned long    *nend;
  size_t           nsize;
  int              ierr;

  PetscFunctionBegin;
  if (TRdebugLevel > 0) {
    ierr = PetscTrValid(lineno,function,filename,dir); if (ierr) PetscFunctionReturn(ierr);
  }

  if (!a) {
    (*PetscErrorPrintf)("PETSC ERROR: PetscTrMalloc: malloc zero length, this is illegal!\n");
    PetscFunctionReturn(1);
  }
  if (a < 0) {
    (*PetscErrorPrintf)("PETSC ERROR: PetscTrMalloc: malloc negative length, this is illegal!\n");
    PetscFunctionReturn(1);
  }
  nsize = a;
  if (nsize & TR_ALIGN_MASK) nsize += (TR_ALIGN_BYTES - (nsize & TR_ALIGN_MASK));
  ierr = PetscMallocAlign(nsize+sizeof(TrSPACE)+sizeof(PetscScalar),lineno,function,filename,dir,(void**)&inew);CHKERRQ(ierr);


  /*
   Keep track of range of memory locations we have malloced in 
  */
  if (PetscLow > (void*)inew) PetscLow = (void*)inew;
  if (PetscHigh < (void*)(inew+nsize+sizeof(TrSPACE)+sizeof(unsigned long))) {
    PetscHigh = (void*)(inew+nsize+sizeof(TrSPACE)+sizeof(unsigned long)); 
  }

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

#if defined(PETSC_USE_STACK)
  ierr = PetscStackCopy(petscstack,&head->stack);CHKERRQ(ierr);
#endif

  /*
         Allow logging of all mallocs made
  */
  if (PetscLogMalloc > -1 && PetscLogMalloc < PetscLogMallocMax) {
    if (PetscLogMalloc == 0) {
      PetscLogMallocLength    = (int*)malloc(PetscLogMallocMax*sizeof(int));
      if (!PetscLogMallocLength) SETERRQ(PETSC_ERR_MEM," ");
      PetscLogMallocDirectory = (char**)malloc(PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocDirectory) SETERRQ(PETSC_ERR_MEM," ");
      PetscLogMallocFile      = (char**)malloc(PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFile) SETERRQ(PETSC_ERR_MEM," ");
      PetscLogMallocFunction  = (char**)malloc(PetscLogMallocMax*sizeof(char**));
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
int PetscTrFreeDefault(void *aa,int line,char *function,char *file,char *dir)
{
  char     *a = (char*)aa;
  TRSPACE  *head;
  char     *ahead;
  int      ierr;
  unsigned long *nend;
  
  PetscFunctionBegin; 
  /* Do not try to handle empty blocks */
  if (!a) {
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Trying to free null block");
  }
  
  if (TRdebugLevel > 0) {
    ierr = PetscTrValid(line,function,file,dir);CHKERRQ(ierr);
  }
  
  if (PetscLow > aa || PetscHigh < aa){
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"PetscTrFreeDefault called with address not allocated by PetscTrMallocDefault");
  }
  
  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *)a;
  
  if (head->cookie != COOKIE_VALUE) {
    (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
    (*PetscErrorPrintf)("Block at address %p is corrupted; cannot free;\n\
may be block not allocated with PetscTrMalloc or PetscMalloc\n",a);
    SETERRQ(PETSC_ERR_MEMC,"Bad location or corrupted memory");
  }
  nend = (unsigned long *)(ahead + head->size);
  if (*nend != COOKIE_VALUE) {
    if (*nend == ALREADY_FREED) {
      (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
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
      (*PetscErrorPrintf)("PetscTrFreeDefault called from %s() line %d in %s%s\n",function,line,dir,file);
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
#define __FUNCT__ "PetscShowMemoryUsage"
/*@
    PetscShowMemoryUsage - Shows the amount of memory currently being used 
        in a communicator.
   
    Collective on PetscViewer

    Input Parameter:
+    viewer - the viewer that defines the communicator
-    message - string printed before values

    Level: intermediate

    Concepts: memory usage

.seealso: PetscTrDump(),PetscTrSpace(), PetscGetResidentSetSize()
 @*/
int PetscShowMemoryUsage(PetscViewer viewer,char *message)
{
  PetscLogDouble allocated,maximum,resident;
  int            ierr,rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscTrSpace(&allocated,PETSC_NULL,&maximum);CHKERRQ(ierr);
  ierr = PetscGetResidentSetSize(&resident);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,message);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Space allocated %g, max space allocated %g, process memory %g\n",rank,allocated,maximum,resident);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTrSpace"
/*@C
    PetscTrSpace - Returns space statistics.
   
    Not Collective

    Output Parameters:
+   space - number of bytes currently allocated
.   frags - number of blocks currently allocated
-   maxs - maximum number of bytes ever allocated

    Level: intermediate

    Concepts: memory usage

.seealso: PetscTrDump()
 @*/
int PetscTrSpace(PetscLogDouble *space,PetscLogDouble *fr,PetscLogDouble *maxs)
{
  PetscFunctionBegin;

  if (space) *space = (PetscLogDouble) TRallocated;
  if (fr)    *fr    = (PetscLogDouble) TRfrags;
  if (maxs)  *maxs  = (PetscLogDouble) TRMaxMem;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTrDump"
/*@C
   PetscTrDump - Dumps the allocated memory blocks to a file. The information 
   printed is: size of space (in bytes), address of space, id of space, 
   file in which space was allocated, and line number at which it was 
   allocated.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stdout is assumed.

   Options Database Key:
.  -trdump - Dumps unfreed memory during call to PetscFinalize()

   Level: intermediate

   Fortran Note:
   The calling sequence in Fortran is PetscTrDump(integer ierr)
   The fp defaults to stdout.

   Notes: uses MPI_COMM_WORLD, because this may be called in PetscFinalize() after PETSC_COMM_WORLD
          has been freed.

   Concepts: memory usage
   Concepts: memory bleeding
   Concepts: bleeding memory

.seealso:  PetscTrSpace(), PetscTrLogDump() 
@*/
int PetscTrDump(FILE *fp)
{
  TRSPACE *head;
  int     rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fp) fp = stdout;
  if (TRallocated > 0) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d]Total space allocated %d bytes\n",rank,(int)TRallocated);CHKERRQ(ierr);
  }
  head = TRhead;
  while (head) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%2d]%8d bytes %s() line %d in %s%s\n",rank,(int)head->size,
            head->functionname,head->lineno,head->dirname,head->filename);CHKERRQ(ierr);
#if defined(PETSC_USE_STACK)
    ierr = PetscStackPrint(&head->stack,fp);CHKERRQ(ierr);
#endif
    head = head->next;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PetscTrLog"
/*@C
    PetscTrLog - Activates logging of all calls to malloc.

    Not Collective

    Options Database Key:
.  -trmalloc_log - Activates PetscTrLog() and PetscTrLogDump()

    Level: advanced

.seealso: PetscTrLogDump()
@*/
int PetscTrLog(void)
{
  PetscFunctionBegin;

  PetscLogMalloc = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTrLogDump"
/*@C
    PetscTrLogDump - Dumps the log of all calls to malloc; also calls 
    PetscGetResidentSetSize().

    Collective on PETSC_COMM_WORLD

    Input Parameter:
.   fp - file pointer; or PETSC_NULL

    Options Database Key:
.  -trmalloc_log - Activates PetscTrLog() and PetscTrLogDump()

    Level: advanced

   Fortran Note:
   The calling sequence in Fortran is PetscTrLogDump(integer ierr)
   The fp defaults to stdout.

.seealso: PetscTrLog(), PetscTrDump()
@*/
int PetscTrLogDump(FILE *fp)
{
  int            i,rank,j,n,*shortlength,ierr,dummy,size,tag = 1212 /* very bad programming */;
  PetscTruth     match;
  char           **shortfunction;
  PetscLogDouble rss;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);
  /*
       Try to get the data printed in order by processor. This will only sometimes work 
  */  
  fflush(fp);
  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(&dummy,1,MPI_INT,rank-1,tag,MPI_COMM_WORLD,&status);CHKERRQ(ierr);
  }

  if (!fp) fp = stdout;
  ierr = PetscGetResidentSetSize(&rss);CHKERRQ(ierr);
  ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Maximum memory used %d Size of entire process %d\n",rank,(int)TRMaxMem,(int)rss);CHKERRQ(ierr);

  shortlength      = (int*)malloc(PetscLogMalloc*sizeof(int));
  shortfunction    = (char**)malloc(PetscLogMalloc*sizeof(char *));
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

  ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] Memory usage sorted by function\n",rank);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscFPrintf(MPI_COMM_WORLD,fp,"[%d] % 9d %s()\n",rank,shortlength[i],shortfunction[i]);CHKERRQ(ierr);
  }
  free(shortlength);
  free(shortfunction);
  fflush(fp);
  if (rank != size-1) {
    ierr = MPI_Send(&dummy,1,MPI_INT,rank+1,tag,MPI_COMM_WORLD);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
} 

/* ---------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PetscTrDebugLevel"
/*
    PetscTrDebugLevel - Set the level of debugging for the space management 
                   routines.

    Input Parameter:
.   level - level of debugging.  Currently, either 0 (no checking) or 1
    (use PetscTrValid at each PetscTrMalloc or PetscTrFree).
*/
int  PetscTrDebugLevel(int level)
{
  PetscFunctionBegin;

  TRdebugLevel = level;
  PetscFunctionReturn(0);
}




