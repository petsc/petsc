

#ifndef lint
static char vcid[] = "$Id: mtr.c,v 1.75 1997/02/22 02:23:29 bsmith Exp bsmith $";
#endif
/*
     PETSc's interface to malloc() and free(). This code allows for 
  logging of memory usage and some error checking 
*/
#include <stdio.h>
#include "petsc.h"           /*I "petsc.h" I*/
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/petscfix.h"

void *PetscTrMallocDefault(unsigned int, int, char *,char *,char *);
int  PetscTrFreeDefault( void *, int, char *,char *,char *);

/*
  Code for checking if a pointer is out of the range 
  of malloced memory. This will only work on flat memory models and 
  even then is suspicious.
*/
void *PetscLow = (void *) 0x0  , *PetscHigh = (void *) 0xEEEEEEEE;
static int TrMallocUsed = 0;
static int TrUseNan;   /* unitialize Scalar arrays with Nans */

#undef __FUNC__  
#define __FUNC__ "PetscSetUseTrMalloc_Private" /* ADIC Ignore */
int PetscSetUseTrMalloc_Private(int usenan)
{
  int ierr;
#if !defined(PETSC_INSIGHT)
  PetscLow     = (void *) 0xEEEEEEEE;
  PetscHigh    = (void *) 0x0;
#endif
  ierr         = PetscSetMalloc(PetscTrMallocDefault,PetscTrFreeDefault); CHKERRQ(ierr);
  TrMallocUsed = 1;
  TrUseNan     = usenan;
  return 0;
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

/* HEADER_DOUBLES is the number of doubles in a PetscTrSpace header */
/* We have to be careful about alignment rules here */

#define TR_FILENAME_LEN     16
#define TR_FUNCTIONNAME_LEN 32
#define TR_DIRNAME_LEN      224
#define HEADER_DOUBLES      38

#if defined(HAVE_64BITS)
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
    char            filename[TR_FILENAME_LEN];
    char            functionname[TR_FUNCTIONNAME_LEN];
    char            dirname[TR_DIRNAME_LEN];
    unsigned long   cookie;        
    struct _trSPACE *next, *prev;
} TRSPACE;
/* This union is used to insure that the block passed to the user is
   aligned on a double boundary */
typedef union {
    TRSPACE sp;
    double  v[HEADER_DOUBLES];
} TrSPACE;

static long    allocated = 0, frags = 0;
static TRSPACE *TRhead = 0;
static int     TRid = 0;
static int     TRdebugLevel = 0;
static long    TRMaxMem = 0;
static long    TRMaxMemId = 0;

#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
  extern int malloc_verify();
}
#elif defined(PARCH_sun4)
  extern int malloc_verify();
#endif

/*
    These are for MPICH version 1.0.13 only. This is for testing purposes
  only and should not be uncommented or used 

struct MPIR_OP {
  void              (*op)();
  unsigned long     cookie;
  int               commute;
  int               permanent;
};

int MPI_Corrupted()
{
  if ((MPI_SUM)->cookie != (unsigned long) 0xca01beaf) {
    SETERRQ(1,0,"MPI_SUM corrupted");
  }
  if ((MPI_MIN)->cookie != (unsigned long) 0xca01beaf) {
    SETERRQ(1,0,"MPI_MIN corrupted");
  }
  if ((MPI_MAX)->cookie != (unsigned long) 0xca01beaf) {
    SETERRQ(1,0,"MPI_MAX corrupted");
  }
  return 0;
}
*/

#undef __FUNC__  
#define __FUNC__ "PetscTrValid" /* ADIC Ignore */
/*
   PetscTrValid - Test the allocated blocks for validity.  This can be used to
   check for memory overwrites.

   Input Parameter:
.  line, file - line number and filename where call originated.

   Return value:
   The number of errors detected.
   
   Output Effect:
   Error messages are written to stdout.  

   No output is generated if there are no problems detected.
*/
int PetscTrValid(int line,char *function,char *file,char *dir )
{
  TRSPACE *head;
  char    *a;
  unsigned long *nend;

  head = TRhead;
  while (head) {
    if (head->cookie != COOKIE_VALUE) {
      fprintf( stderr, "called from %s() line %d in %s%s\n",function,line,dir,file );
      fprintf( stderr, "Block at address %p is corrupted\n", head );
      SETERRQ(1,0,"");
    }
    a    = (char *)(((TrSPACE*)head) + 1);
    nend = (unsigned long *)(a + head->size);
    if (nend[0] != COOKIE_VALUE) {
      fprintf( stderr, "called from %s() line %d in %s%s\n",function,line,dir,file );
      if (nend[0] == ALREADY_FREED) {
        fprintf(stderr,"Block [id=%d(%lx)] at address %p already freed\n", 
	        head->id, head->size, a );
        SETERRQ(1,0,"Freed block in memory list, corrupted memory");
      } else {
        fprintf( stderr, 
             "Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n", 
	     head->id, head->size, a );
        fprintf(stderr,"Block allocated in %s() line %d in %s%s\n",head->functionname,
                head->lineno,head->dirname,head->filename);
        SETERRQ(1,0,"Corrupted memory");
      }
    }
    head = head->next;
  }
#if defined(PARCH_sun4) && defined(PETSC_BOPT_g)
  malloc_verify();
#endif

  /*
  {
    int ierr;
    ierr = MPI_Corrupted(); CHKERRQ(ierr);
  }
  */

  return 0;
}

/*
      Arrays to log information on all Mallocs
*/
static int  PetscLogMallocMax = 10000, PetscLogMalloc = -1, *PetscLogMallocLength;
static char **PetscLogMallocDirectory, **PetscLogMallocFile,**PetscLogMallocFunction;


#undef __FUNC__  
#define __FUNC__ "PetscTrMallocDefault" /* ADIC Ignore */
/*
    PetscTrMallocDefault - Malloc with tracing.

    Input Parameters:
.   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNC__ for this
.   filename  - file name where used.  Use __FILE__ for this
.   dir - directory where file is. Use __SDIR__ for this

    Returns:
    double aligned pointer to requested storage, or null if not
    available.
 */
void *PetscTrMallocDefault(unsigned int a,int lineno,char *function,char *filename,char *dir)
{
  TRSPACE          *head;
  char             *inew;
  unsigned long    *nend;
  unsigned int     nsize;
  int              l,ierr;

  if (TRdebugLevel > 0) {
    ierr = PetscTrValid(lineno,function,filename,dir); if (ierr) return 0;
  }

  if (a == 0) {
    fprintf(stderr,"PETSC ERROR: PetscTrMalloc: malloc zero length, this is illegal!\n");
    return 0;
  }
  nsize = a;
  if (nsize & TR_ALIGN_MASK) nsize += (TR_ALIGN_BYTES - (nsize & TR_ALIGN_MASK));
  inew = (char *) malloc( (unsigned)(nsize+sizeof(TrSPACE)+sizeof(unsigned long)));
  if (!inew) return 0;

  
  /*
   Keep track of range of memory locations we have malloced in 
  */
#if !defined(PETSC_INSIGHT)
  if (PetscLow > (void *) inew) PetscLow = (void *) inew;
  if (PetscHigh < (void *) (inew+nsize+sizeof(TrSPACE)+sizeof(unsigned long)))
      PetscHigh = (void *) (inew+nsize+sizeof(TrSPACE)+sizeof(unsigned long));
#endif

  head = (TRSPACE *)inew;
  inew  += sizeof(TrSPACE);

  if (TRhead) TRhead->prev = head;
  head->next     = TRhead;
  TRhead         = head;
  head->prev     = 0;
  head->size     = nsize;
  head->id       = TRid;
  head->lineno   = lineno;

  if ((l = PetscStrlen(filename)) > TR_FILENAME_LEN-1) filename += (l - (TR_FILENAME_LEN-1));
  if (filename) PetscStrncpy( head->filename, filename, (TR_FILENAME_LEN-1) );
  head->filename[TR_FILENAME_LEN-1] = 0;

  if ((l = PetscStrlen(function)) > TR_FUNCTIONNAME_LEN-1) function += (l-(TR_FUNCTIONNAME_LEN-1));
  if (function) PetscStrncpy( head->functionname, function, (TR_FUNCTIONNAME_LEN-1) );
  head->functionname[TR_FUNCTIONNAME_LEN-1] = 0;

  if ((l = PetscStrlen(dir)) > TR_DIRNAME_LEN-1) dir += (l-(TR_DIRNAME_LEN-1));
  if (dir) PetscStrncpy( head->dirname, dir, (TR_DIRNAME_LEN-1) );
  head->dirname[TR_DIRNAME_LEN-1] = 0;

  head->cookie                = COOKIE_VALUE;
  nend                        = (unsigned long *)(inew + nsize);
  nend[0]                     = COOKIE_VALUE;

  allocated += nsize;
  if (allocated > TRMaxMem) {
    TRMaxMem   = allocated;
    TRMaxMemId = TRid;
  }
  frags     ++;

  if (TrUseNan && sizeof(Scalar)*(nsize/sizeof(Scalar)) == nsize) {
    ierr = PetscInitializeNans((Scalar*) inew,nsize/sizeof(Scalar)); 
    if (ierr) return 0;
  } else if (TrUseNan && sizeof(int)*(nsize/sizeof(int)) == nsize) {
    ierr = PetscInitializeLargeInts((int*) inew,nsize/sizeof(int)); 
    if (ierr) return 0;
  }

  /*
         Allow logging of all mallocs made
  */
  if (PetscLogMalloc > -1 && PetscLogMalloc < PetscLogMallocMax) {
    if (PetscLogMalloc == 0) {
      PetscLogMallocLength    = (int *) malloc( PetscLogMallocMax*sizeof(int));
      if (!PetscLogMallocLength) return 0;
      PetscLogMallocDirectory = (char **) malloc( PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocDirectory) return 0;
      PetscLogMallocFile = (char **) malloc( PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFile) return 0;
      PetscLogMallocFunction = (char **) malloc( PetscLogMallocMax*sizeof(char**));
      if (!PetscLogMallocFunction) return 0;
    }
    PetscLogMallocLength[PetscLogMalloc]      = nsize;
    PetscLogMallocDirectory[PetscLogMalloc]   = dir;
    PetscLogMallocFile[PetscLogMalloc]        = filename;
    PetscLogMallocFunction[PetscLogMalloc++]  = function; 
  }

  return (void *)inew;
}


#undef __FUNC__  
#define __FUNC__ "PetscTrFreeDefault" /* ADIC Ignore */
/*
   PetscTrFreeDefault - Free with tracing.

   Input Parameters:
.   a    - pointer to a block allocated with PetscTrMalloc
.   lineno - line number where used.  Use __LINE__ for this
.   function - function calling routine. Use __FUNC__ for this
.   file  - file name where used.  Use __FILE__ for this
.   dir - directory where file is. Use __SDIR__ for this
 */
int PetscTrFreeDefault( void *aa, int line, char *function, char *file, char *dir )
{
  char     *a = (char *) aa;
  TRSPACE  *head;
  char     *ahead;
  unsigned long *nend;
  int      ierr;

  /* Don't try to handle empty blocks */
  if (!a) {
    fprintf(stderr,"PetscTrFree called from %s() line %d in %s%s\n",function,line,file,dir);
    SETERRQ(1,0,"Trying to free null block");
  }

  if (TRdebugLevel > 0) {
    ierr = PetscTrValid(line,function,file,dir); CHKERRQ(ierr);
  }

#if !defined(PETSC_INSIGHT)
  if (PetscLow > aa || PetscHigh < aa){
    fprintf(stderr,"PetscTrFree called from %s() line %d in %s%s\n",function,line,file,dir);
    fprintf(stderr,"PetscTrFree called with address not allocated by PetscTrMalloc\n");
    SETERRQ(1,0,"Invalid Address");
  } 
#endif

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *)a;

  if (head->cookie != COOKIE_VALUE) {
    /* Damaged header */
    fprintf( stderr, "Block at address %p is corrupted; cannot free;\n\
may be block not allocated with PetscTrMalloc or PetscMalloc\n", a );
    SETERRQ(1,0,"Bad location or corrupted memory");
  }
  nend = (unsigned long *)(ahead + head->size);
  if (*nend != COOKIE_VALUE) {
    if (*nend == ALREADY_FREED) {
	fprintf(stderr,"Block [id=%d(%lx)] at address %p was already freed\n", 
		head->id, head->size, a + sizeof(TrSPACE) );
	if (head->lineno > 0) 
	  fprintf( stderr, "Block freed in %s() line %d in %s%s\n", head->functionname,
                 head->lineno,head->dirname,head->filename);	
	else
	fprintf( stderr, "Block allocated in %s() line %d in %s%s\n", head->functionname,
                 -head->lineno,head->dirname,head->filename);	
	SETERRQ(1,0,"Memory already freed");
    }
    else {
	/* Damaged tail */
	fprintf( stderr, 
  "Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n", 
		head->id, head->size, a );
	fprintf( stderr, "Block allocated in %s() line %d in %s%s\n", head->functionname,
                 head->lineno,head->dirname,head->filename);
	SETERRQ(1,0,"Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend        = ALREADY_FREED;
  /* Save location where freed.  If we suspect the line number, mark as 
     allocated location */
  if (line > 0 && line < 50000) {
    head->lineno = line;
    if (file) PetscStrncpy( head->filename, file, (TR_FILENAME_LEN-1) );
    head->filename[TR_FILENAME_LEN-1]= 0;  /* Just in case */
    if (function) PetscStrncpy( head->functionname, function, (TR_FUNCTIONNAME_LEN-1) );
    head->functionname[TR_FUNCTIONNAME_LEN-1]= 0;  /* Just in case */
    if (dir) PetscStrncpy( head->dirname, dir, (TR_DIRNAME_LEN-1) );
    head->dirname[TR_DIRNAME_LEN-1]= 0;  /* Just in case */
  }
  else {
    head->lineno = - head->lineno;
  }

  allocated -= head->size;
  frags     --;
  if (head->prev) head->prev->next = head->next;
  else TRhead = head->next;

  if (head->next) head->next->prev = head->prev;
  free( a );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscTrSpace" /* ADIC Ignore */
/*@
    PetscTrSpace - Returns space statistics.
   
    Output Parameters:
.   space - number of bytes currently allocated
.   frags - number of blocks currently allocated
.   maxs - maximum number of bytes ever allocated

.keywords: memory, allocation, tracing, space, statistics

.seealso: PetscTrDump()
 @*/
int PetscTrSpace( double *space, double *fr, double *maxs )
{
  if (space) *space = (double) allocated;
  if (fr)    *fr    = (double) frags;
  if (maxs)  *maxs  = (double) TRMaxMem;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscTrDump" /* ADIC Ignore */
/*@C
   PetscTrDump - Dumps the allocated memory blocks to a file. The information 
   printed is: size of space (in bytes), address of space, id of space, 
   file in which space was allocated, and line number at which it was 
   allocated.

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stderr is assumed.

   Options Database Key:
$  -trdump : dumps unfreed memory during call to PetscFinalize()

.keywords: memory, allocation, tracing, space, statistics

.seealso:  PetscTrSpace()
 @*/
int PetscTrDump( FILE *fp )
{
  TRSPACE *head;
  int     rank;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (fp == 0) fp = stderr;
  if (allocated > 0) {
    fprintf(fp,"[%d]Total space allocated %d bytes\n",rank,(int)allocated);
  }
  head = TRhead;
  while (head) {
    fprintf(fp,"[%d]%d bytes %s() line %d in %s%s\n",rank,(int) head->size,
            head->functionname,head->lineno,head->dirname,head->filename);
    head = head->next;
  }
  return 0;
}

/* ---------------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "PetscTrLog" /* ADIC Ignore */
/*@C
    PetscTrLog - Indicates that you wish all calls to malloc to be logged.

     Options Database:
.     -trmalloc_log

.seealso: PetscTrLogDump()
@*/
int PetscTrLog()
{
  PetscLogMalloc = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscTrLogDump" /* ADIC Ignore */
/*@C
    PetscTrLogDump - Dumps the log of all calls to malloc.

  Input Parameters:
.    fp - file pointer; or PETSC_NULL

     Options Database:
.     -trmalloc_log

.seealso: PetscTrLog()
@*/
int PetscTrLogDump(FILE *fp)
{
  int  i,rank,j,n,*shortlength;
  char **shortfunction;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (fp == 0) fp = stderr;
  fprintf(stderr,"Maximum memory used %d\n",(int) TRMaxMem);
  for ( i=0; i<PetscLogMalloc; i++ ) {
    fprintf(fp,"[%d] %d %s%s %s()\n",rank,PetscLogMallocLength[i],PetscLogMallocDirectory[i],
            PetscLogMallocFile[i],PetscLogMallocFunction[i]);
  }

  shortlength   = (int *) malloc(PetscLogMalloc*sizeof(int)); CHKPTRQ(shortlength);
  shortfunction = (char**) malloc(PetscLogMalloc*sizeof(char *));CHKPTRQ(shortfunction);
  shortfunction[0] = PetscLogMallocFunction[0];
  shortlength[0]   = PetscLogMallocLength[0]; 
  n = 1;
  for ( i=1; i<PetscLogMalloc; i++ ) {
    for ( j=0; j<n; j++ ) {
      if (!PetscStrcmp(shortfunction[j],PetscLogMallocFunction[i])) {
        shortlength[j] += PetscLogMallocLength[i];
        goto foundit;
      }
    }
    shortfunction[n] = PetscLogMallocFunction[i];
    shortlength[n]   = PetscLogMallocLength[i]; 
    n++;
    foundit:;
  }

  fprintf(fp,"Sorted by function\n");
  for ( i=0; i<n; i++ ) {
    fprintf(fp,"[%d] %d %s()\n",rank,shortlength[i],shortfunction[i]);
  }
  free(shortlength);
  free(shortfunction);
  return 0;
}

/* ---------------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "PetscTrDebugLevel" /* ADIC Ignore */
/*
    PetscTrDebugLevel - Set the level of debugging for the space management 
                   routines.

    Input Parameter:
.   level - level of debugging.  Currently, either 0 (no checking) or 1
    (use PetscTrValid at each PetscTrMalloc or PetscTrFree).
*/
int  PetscTrDebugLevel(int level )
{
  TRdebugLevel = level;
  return 0;
}

#if defined(PARCH_IRIX)
static long nanval[2] = {0x7fffffff,0xffffffff };  /* Signaling nan */
/* static long nanval[2] = {0x7ff7ffff,0xffffffff };  Quiet nan */
#elif defined(PARCH_sun4)
#elif defined(PARCH_rs6000)
struct sigcontext;
#include <fpxcp.h>
#else
static long nanval[2] = {-1,-1}; /* Probably a bad floating point value */
#endif

typedef union { long l[2]; double d; } NANDouble;

#include <math.h>
/*@
   PetscInitializeNans - Intialize certain memory locations with NANs.
   This routine is used to mark an array as unitialized so that
   if values are used for computation without first having been set,
   a floating point exception is generated.

   Input parameters:
.  p   - pointer to data
.  n   - length of data (in Scalars)

   Options Database Key:
$   -trmalloc_nan

   Notes:
   This routine is useful for tracking down the use of uninitialized
   array values.  If the code is run with the -fp_trap option, it will
   stop if one of the "unitialized" values is used in a computation.

.seealso: PetscInitializeLargeInts()
@*/
int PetscInitializeNans(Scalar *p,int n )
{
  double     *pp,nval;

#if defined(PARCH_sun4) 
  nval = signaling_nan();
#elif defined(PARCH_rs6000)
  nval = FP_INV_SNAN;
#else
  NANDouble  nd;
  nd.l[0] = nanval[0];
  nd.l[1] = nanval[1];
  nval = nd.d;
#endif
  pp = (double *) p;
#if defined(PETSC_COMPLEX)
  n *= 2;
#endif
  while (n--) *pp++   = nval;
  return 0;
}

/*@
   PetscInitializeLargeInts - Intializes an array of integers
   with very large values.

   Input parameters:
.  p   - pointer to data
.  n   - length of data (in ints)

   Options Database Key:
$   -trmalloc_nan

   Notes:
   This routine is useful for tracking down the use of uninitialized
   array values.  If an integer array value is absurdly large, then
   there's a good chance that it is being used before it was ever set.

.seealso: PetscInitializeNans()
@*/
int PetscInitializeLargeInts(int *p,int n )
{
  while (n--) *p++   = 1073741824;
  return 0;
}



