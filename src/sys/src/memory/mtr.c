
#ifndef lint
static char vcid[] = "$Id: mtr.c,v 1.44 1996/01/16 22:25:31 balay Exp bsmith $";
#endif
/*
     PETSc's interface to malloc() and free(). This code allows for 
  logging of memory usage and some error checking 
*/
#include <stdio.h>
#include "petsc.h"     /*I "petsc.h" I*/
/* rs6000 needs _XOPEN_SOURCE to use tsearch */
#if defined(PARCH_rs6000) && !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE
#endif
#if defined(PARCH_hpux) && !defined(_INCLUDE_XOPEN_SOURCE)
#define _INCLUDE_XOPEN_SOURCE
#endif
#if defined(HAVE_SEARCH_H)
#include <search.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/petscfix.h"

void *TrMalloc(unsigned int, int, char *);
int  TrFree( void *, int, char * );

/*
  Code for checking if a pointer is out of the range 
  of malloced memory. This will only work on flat memory models and 
  even then is suspicious.
*/
void *PetscLow = (void *) 0x0  , *PetscHigh = (void *) 0xEEEEEEEE;
int  TrMallocUsed = 0;

int PetscSetUseTrMalloc_Private()
{
  PetscLow  = (void *) 0xEEEEEEEE;
  PetscHigh = (void *) 0x0;
  PetscSetMalloc(TrMalloc,TrFree);
  TrMallocUsed = 1;
  return 0;
}

/*
    TrSpace - Routines for tracing space usage.

    Description:
    TrMalloc replaces malloc and TrFree replaces free.  These routines
    have the same syntax and semantics as the routines that they replace,
    In addition, there are routines to report statistics on the memory
    usage, and to report the currently allocated space.  These routines
    are built on top of malloc and free, and can be used together with
    them as long as any space allocated with TrMalloc is only freed with
    TrFree.
 */

/* HEADER_DOUBLES is the number of doubles in a trSPACE header */
/* We have to be careful about alignment rules here */
#if defined(HAVE_64BITS)
#define TR_ALIGN_BYTES 8
#define TR_ALIGN_MASK  0x7
#define TR_FNAME_LEN   16
#define HEADER_DOUBLES 8
#else
#define TR_ALIGN_BYTES 4
#define TR_ALIGN_MASK  0x3
#define TR_FNAME_LEN   12
#define HEADER_DOUBLES 5
#endif

#define COOKIE_VALUE   0xf0e0d0c9
#define ALREADY_FREED  0x0f0e0d9c
#define MAX_TR_STACK 20
#define TR_MALLOC 0x1
#define TR_FREE   0x2

typedef struct _trSPACE {
    unsigned long   size;
    int             id;
    int             lineno;
    char            fname[TR_FNAME_LEN];
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

/*
   TrValid - Test the allocated blocks for validity.  This can be used to
   check for memory overwrites.

   Input Parameter:
.  line, file - line number and filename where call originated.

   Return value:
   The number of errors detected.
   
   Output Effect:
   Error messages are written to stdout.  These have the form of either

$   Block [id=%d(%d)] at address %lx is corrupted (probably write past end)
$   Block allocated in <filename>[<linenumber>]

   if the sentinal at the end of the block has been corrupted, and

$   Block at address %lx is corrupted

   if the sentinal at the begining of the block has been corrupted.

   The address is the actual address of the block.  The id is the
   value of TRID.

   No output is generated if there are no problems detected.
*/
int TrValid(int line,char *file )
{
  TRSPACE *head;
  char    *a;
  unsigned long *nend;
  int      errs = 0;

  head = TRhead;
  while (head) {
    if (head->cookie != COOKIE_VALUE) {
      if (!errs) fprintf( stderr, "called from %s line %d \n",file,line );
      fprintf( stderr, "Block at address %p is corrupted\n", head );
      SETERRQ(1,"TrValid");
    }
    a    = (char *)(((TrSPACE*)head) + 1);
    nend = (unsigned long *)(a + head->size);
    if (nend[0] != COOKIE_VALUE) {
      if (!errs) fprintf( stderr, "called from %s line %d\n",file,line );
      errs++;
      head->fname[TR_FNAME_LEN-1]= 0;  /* Just in case */
      fprintf( stderr, 
             "Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n", 
	     head->id, head->size, a );
      fprintf( stderr, 
		"Block allocated in %s[%d]\n", head->fname, head->lineno );
      SETERRQ(1,"TrValid");
    }
    head = head->next;
  }
  return errs;
}

/*
    TrMalloc - Malloc with tracing.

    Input Parameters:
.   a   - number of bytes to allocate
.   lineno - line number where used.  Use __LINE__ for this
.   fname  - file name where used.  Use __FILE__ for this

    Returns:
    double aligned pointer to requested storage, or null if not
    available.
 */
void *TrMalloc(unsigned int a, int lineno, char *fname )
{
  TRSPACE          *head;
  char             *inew;
  unsigned long    *nend;
  unsigned int     nsize;
  int              l;

  if (TRdebugLevel > 0) {
    if (TrValid(lineno,fname )) return 0;
  }

  if (a == 0) {
    fprintf(stderr,"PETSC ERROR: TrMalloc: malloc zero length, this is illegal!\n");
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
  if ((l = PetscStrlen( fname )) > TR_FNAME_LEN-1 ) fname += (l - (TR_FNAME_LEN-1));
  PetscStrncpy( head->fname, fname, (TR_FNAME_LEN-1) );
  head->fname[TR_FNAME_LEN-1]= 0;
  head->cookie   = COOKIE_VALUE;
  nend           = (unsigned long *)(inew + nsize);
  nend[0]        = COOKIE_VALUE;

  allocated += nsize;
  if (allocated > TRMaxMem) {
    TRMaxMem   = allocated;
    TRMaxMemId = TRid;
  }
  frags     ++;
  return (void *)inew;
}


/*
   TrFree - Free with tracing.

   Input Parameters:
.  a    - pointer to a block allocated with TrMalloc
.  line - line in file where called
.  file - Name of file where called
 */
int TrFree( void *aa, int line, char *file )
{
  char     *a = (char *) aa;
  TRSPACE  *head;
  char     *ahead;
  unsigned long *nend;
  int      ierr;

  /* Don't try to handle empty blocks */
  if (!a) {
    fprintf(stderr,"TrFree called from line %d in %s\n",line,file);
    SETERRQ(1,"TrFree:Trying to free null block");
  }

  if (TRdebugLevel > 0) {
    if ((ierr = TrValid(line,file))) return ierr;
  }

  ahead = a;
  a     = a - sizeof(TrSPACE);
  head  = (TRSPACE *)a;
  if (head->cookie != COOKIE_VALUE) {
    /* Damaged header */
    fprintf( stderr, "Block at address %p is corrupted; cannot free;\n\
may be block not allocated with TrMalloc or MALLOC\n", a );
    SETERRQ(1,"TrFree:Bad location or corrupted memory");
  }
  nend = (unsigned long *)(ahead + head->size);
  if (*nend != COOKIE_VALUE) {
    if (*nend == ALREADY_FREED) {
	fprintf( stderr, 
  "Block [id=%d(%lx)] at address %p was already freed\n", 
		head->id, head->size, a + sizeof(TrSPACE) );
	head->fname[TR_FNAME_LEN-1]= 0;  /* Just in case */
	if (head->lineno > 0) 
	  fprintf( stderr, "Block freed in %s[%d]\n", head->fname, head->lineno );
	else
	  fprintf( stderr, "Block allocated at %s[%d]\n",head->fname,-head->lineno);
	  SETERRQ(1,"TrFree:Memory already freed");
    }
    else {
	/* Damaged tail */
	fprintf( stderr, 
  "Block [id=%d(%lx)] at address %p is corrupted (probably write past end)\n", 
		head->id, head->size, a );
	head->fname[TR_FNAME_LEN-1]= 0;  /* Just in case */
	fprintf( stderr, "Block allocated in %s[%d]\n", head->fname, head->lineno );
	SETERRQ(1,"TrFree:Corrupted memory");
    }
  }
  /* Mark the location freed */
  *nend        = ALREADY_FREED;
  /* Save location where freed.  If we suspect the line number, mark as 
     allocated location */
  if (line > 0 && line < 5000) {
    head->lineno = line;
    PetscStrncpy( head->fname, file, (TR_FNAME_LEN-1) );
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

/*@
    TrSpace - Returns space statistics.
   
    Output Parameters:
.   space - number of bytes currently allocated
.   frags - number of blocks currently allocated
.   maxs - maximum number of bytes ever allocated

.keywords: memory, allocation, tracing, space, statistics

.seealso: TrDump()
 @*/
int TrSpace( int *space, int *fr, int *maxs )
{
  if (space) *space = allocated;
  if (fr)    *fr    = frags;
  if (maxs)  *maxs  = (int)TRMaxMem;
  return 0;
}

/*@C
   TrDump - Dumps the allocated memory blocks to a file. The information 
   printed is: size of space (in bytes), address of space, id of space, 
   file in which space was allocated, and line number at which it was 
   allocated.

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stderr is assumed.

   Options Database Key:
$  -trdump : dumps unfreed memory during call to PetscFinalize()

.keywords: memory, allocation, tracing, space, statistics

.seealso:  TrSpace()
 @*/
int TrDump( FILE *fp )
{
  TRSPACE *head;
  int     id,rank;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (fp == 0) fp = stderr;
  head = TRhead;
  while (head) {
    fprintf( fp, "[%d]%d bytes at address [%p], id = ",rank, 
	     (int) head->size, head + sizeof(TrSPACE) );
    if (head->id >= 0) {
	head->fname[TR_FNAME_LEN-1] = 0;
	fprintf( fp, "%d, %s[%d]\n", head->id, head->fname, head->lineno );
    }
    else {
	/* Decode the package values */
	head->fname[TR_FNAME_LEN-1] = 0;
	id = head->id;
	fprintf( fp, "%d %s[%d]\n", id, head->fname, head->lineno );
    }
    head = head->next;
  }
  return 0;
}

#if defined(HAVE_SEARCH_H)

typedef struct { int id, size, lineno; char *fname; } TRINFO;
static FILE *TRFP;

static int IntCompare( TRINFO *a, TRINFO * b )
{
  return a->id - b->id;
}

/*ARGSUSED*/
static int  PrintSum(TRINFO ** a, VISIT order, int level )
{ 
  if (order == postorder || order == leaf) 
    fprintf( TRFP, "[%d]%s[%d] has %d\n",(*a)->id,(*a)->fname,(*a)->lineno,(*a)->size);
  return 0;
}

/*@C
  TrSummary - Summarize the allocate memory blocks by id.

  Input Parameter:
.  fp  - file pointer

  Note:
  This routine is the same as TrDump() on those systems that do not 
  include /usr/include/search.h .

.keywords: memory, allocation, tracing, space, statistics

.seealso: TrDump()
 @*/
int TrSummary( FILE *fp )
{
  TRSPACE *head;
  TRINFO  *root, *key, **fnd,nspace[1000];

  root = 0;
  head = TRhead;
  key  = nspace;
  while (head) {
    key->id     = head->id;
    key->size   = 0;
    key->lineno = head->lineno;
    key->fname  = head->fname;
#if defined(PARCH_solaris)
    fnd=(TRINFO **)tsearch((void *)key,(void **)&root, 
                                         (int (*)(const void*,const void*))IntCompare);
#elif !defined(PARCH_IRIX) && !defined(PARCH_hpux) && !defined(PARCH_rs6000)
    fnd=(TRINFO **)tsearch((char *)key,(char **)&root,(int (*)(void*,void*))IntCompare);
#else
    fnd=(TRINFO **)tsearch((void *)key,(void **)&root,(int (*)(void*,void*))IntCompare);
#endif
    if (*fnd == key) {
	key->size = 0;
	key++;
    }
    (*fnd)->size += head->size;
    head = head->next;
  }

  /* Print the data */
  TRFP = fp;
  twalk( (char *)root, (void (*)(void*,VISIT,int))PrintSum );
  fprintf(fp,"The maximum space allocated was %lx bytes [%lx]\n",TRMaxMem,TRMaxMemId);
  return 0;
}
#else
int TrSummary(FILE* fp )
{
  fprintf(fp,"The maximum space allocated was %ld bytes [%ld]\n",TRMaxMem,TRMaxMemId);
  return 0;
}	
#endif

/*@
   TrGetMaximumAllocated - If TrMalloc is used, returns the maximum amount 
   of space allocated.

   Output Parameter:
.  max - the maximum amount of space (in bytes).

.keywords: memory, maximum, allocation, tracing, space, statistics

.seealso: TrSummary(), TrDump()
@*/
int TrGetMaximumAllocated(double *max)
{
  if (TrMallocUsed)   *max = (double) TRMaxMem;
  else                *max = 0.0;
  return 0;
}

/*
    TrDebugLevel - Set the level of debugging for the space management 
                   routines.

    Input Parameter:
.   level - level of debugging.  Currently, either 0 (no checking) or 1
    (use TrValid at each TrMalloc or TrFree).
*/
int  TrDebugLevel(int level )
{
  TRdebugLevel = level;
  return 0;
}

#define TR_MAX_DUMP 100
/*
   The following routine attempts to give useful information about the
   memory usage when an "out-of-memory" error is encountered.  The rules are:
   If there are less than TR_MAX_DUMP blocks, output those.
   Otherwise, try to find multiple instances of the same routine/line #, and
   print a summary by number:
   file line number-of-blocks total-number-of-blocks

   We have to do a sort-in-place for this
 */

/*
  Sort by file/line number.  Do this without calling a system routine or
  allocating ANY space (space is being optimized here).

  We do this by first recursively sorting halves of the list and then
  merging them.  
 */

/* Merge two lists, returning the head of the merged list */
TRSPACE *TrImerge(TRSPACE * l1,TRSPACE * l2 )
{
  TRSPACE *head = 0, *tail = 0;
  int     sign;

  while (l1 && l2) {
    sign = PetscStrcmp(l1->fname, l2->fname);
    if (sign > 0 || (sign == 0 && l1->lineno >= l2->lineno)) {
      if (head) tail->next = l1; 
      else      head = tail = l1;
      tail = l1;
      l1   = l1->next;
    }
    else {
      if (head) tail->next = l2; 
      else      head = tail = l2;
      tail = l2;
      l2   = l2->next;
    }
  }
  /* Add the remaining elements to the end */
  if (l1) tail->next = l1;
  if (l2) tail->next = l2;
  return head;
}

/* Sort head with n elements, returning the head */
TRSPACE *TrIsort( TRSPACE * head,int n )
{
  TRSPACE *p, *l1, *l2;
  int     m, i;

  if (n <= 1) return head;

  /* This guarentees that m, n are both > 0 */
  m = n / 2;
  p = head;
  for (i=0; i<m-1; i++) p = p->next;
  /* p now points to the END of the first list */
  l2 = p->next;
  p->next = 0;
  l1 = TrIsort( head, m );
  l2 = TrIsort( l2,   n - m );
  return TrImerge( l1, l2 );
}

int TrSortBlocks()
{
  TRSPACE *head;
  int     cnt;

  head = TRhead;
  cnt  = 0;
  while (head) {
    cnt ++;
    head = head->next;
  }
  TRhead = TrIsort( TRhead, cnt );
  return 0;
}

/* Takes sorted input and dumps as an aggregate */
int TrDumpGrouped(FILE *fp )
{
  TRSPACE *head, *cur;
  int     nblocks, nbytes;

  if (fp == 0) fp = stderr;

  TrSortBlocks();
  head = TRhead;
  cur  = 0;
  while (head) {
    cur     = head->next;
    nblocks = 1;
    nbytes  = head->size;
    while (cur && !PetscStrcmp(cur->fname,head->fname) && cur->lineno == head->lineno){
	nblocks++;
	nbytes += cur->size;
	cur    = cur->next;
    }
    fprintf( fp, "File %13s line %5d: %d bytes in %d allocation%c\n", 
	     head->fname, head->lineno, nbytes, nblocks,(nblocks > 1) ? 's' : ' ');
    head = cur;
  }
  fflush( fp );
  return 0;
}

