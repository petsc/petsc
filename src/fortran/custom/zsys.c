#ifndef lint
static char vcid[] = "$Id: zsys.c,v 1.22 1996/09/14 03:34:18 curfman Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sys.h"
#include "vec.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define petscattachdebugger_  PETSCATTACHDEBUGGER
#define petscobjectsetname_   PETSCOBJECTSETNAME
#define petscobjectdestroy_   PETSCOBJECTDESTROY
#define petscobjectgetcomm_   PETSCOBJECTGETCOMM
#define petscobjectgetname_   PETSCOBJECTGETNAME
#define petscgettime_         PETSCGETTIME
#define petscgetflops_        PETSCGETFLOPS
#define petscerror_           PETSCERROR
#define petscrandomcreate_    PETSCRANDOMCREATE
#define petscrandomdestroy_   PETSCRANDOMDESTROY
#define petscrandomgetvalue_  PETSCRANDOMGETVALUE
#define vecsetrandom_         VECSETRANDOM
#define petsctrvalid_         PETSCTRVALID
#define petscdoubleview_      PETSCDOUBLEVIEW
#define petscintview_         PETSCINTVIEW
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscattachdebugger_  petscattachdebugger
#define petscobjectsetname_   petscobjectsetname
#define petscobjectdestroy_   petscobjectdestroy
#define petscobjectgetcomm_   petscobjectgetcomm
#define petscobjectgetname_   petscobjectgetname
#define petscgettime_         petscgettime  
#define petscgetflops_        petscgetflops 
#define petscerror_           petscerror
#define petscrandomcreate_    petscrandomcreate
#define petscrandomdestroy_   petscrandomdestroy
#define petscrandomgetvalue_  petscrandomgetvalue
#define vecsetrandom_         vecsetrandom
#define petsctrvalid_         petsctrvalid
#define petscdoubleview_      petscdoubleview
#define petscintview_         petscintview
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void petsctrvalid_(int *__ierr)
{
  *__ierr = PetscTrValid(0,"Unknown Fortran");
}

void petscrandomgetvalue_(PetscRandom r,Scalar *val, int *__ierr )
{
  *__ierr = PetscRandomGetValue((PetscRandom)PetscToPointer(*(int*)(r)),val);
}

void vecsetrandom_(PetscRandom r,Vec x, int *__ierr )
{
  *__ierr = VecSetRandom((PetscRandom)PetscToPointer(*(int*)(r)),
                         (Vec)PetscToPointer( *(int*)(x) ));
}

void petscobjectgetname(PetscObject obj, CHAR name, int *__ierr, int len)
{
  char *tmp;
  *__ierr = PetscObjectGetName((PetscObject)PetscToPointer(*(int*)(obj)),&tmp);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name);
  int  len1 = _fcdlen(name);
  PetscStrncpy(t,tmp,len1);
  }
#else
  PetscStrncpy(name,tmp,len);
#endif
}

void petscobjectdestroy_(PetscObject obj, int *__ierr )
{
  *__ierr = PetscObjectDestroy((PetscObject)PetscToPointer(*(int*)(obj)));
  PetscRmPointer(*(int*)(obj));
}

void petscobjectgetcomm_(PetscObject obj,int *comm, int *__ierr )
{
  MPI_Comm c;
  *__ierr = PetscObjectGetComm((PetscObject)PetscToPointer(*(int*)(obj)),&c);
  *(int*)comm = PetscFromPointerComm(c);
}

void petscattachdebugger_(int *__ierr)
{
  *__ierr = PetscAttachDebugger();
}

/*
      This bleeds memory, but no easy way to get around it
*/
void petscobjectsetname_(PetscObject obj,CHAR name,int *__ierr,int len)
{
  char *t1;

  FIXCHAR(name,len,t1);
  *__ierr = PetscObjectSetName((PetscObject)PetscToPointer(*(int*)(obj)),t1);
}

void petscerror_(int *number,CHAR message,int *__ierr,int len)
{
  char *t1;
  FIXCHAR(message,len,t1);
  *__ierr = PetscError(-1,0,"fortran_interface_unknown_file",*number,t1);
}

double petscgettime_()
{ 
  return PetscGetTime();
}

double  petscgetflops_()
{
#if defined(PETSC_LOG)
  return PetscGetFlops();
#else
  return 0.0;
#endif
}

void petscrandomcreate_(MPI_Comm comm,PetscRandomType *type,PetscRandom *r,int *__ierr )
{
  PetscRandom rr;
  *__ierr = PetscRandomCreate((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*type,&rr);
  *(int*)r = PetscFromPointer(rr);
}

void petscrandomdestroy_(PetscRandom r, int *__ierr )
{
  *__ierr = PetscRandomDestroy((PetscRandom )PetscToPointer( *(int*)(r) ));
   PetscRmPointer(*(int*)(r)); 
}

void petscdoubleview_(int *n,double *d,int *viwer,int *__ierr)
{
  *__ierr = PetscDoubleView(*n,d,0);
}

void petscintview_(int *n,int *d,int *viwer,int *__ierr)
{
  *__ierr = PetscIntView(*n,d,0);
}

#if defined(__cplusplus)
}
#endif

/*
    This is for when one is compiling with versions of MPI that use
   integers for the MPI objects, hence the PETSc objects require routines 
   provided here to do the conversion between C pointers and Fortran integers.
*/
#if defined(HAVE_64BITS)

/* 
   This file contains routines to convert to and from C pointers to Fortran integers
*/

typedef struct _PtrToIdx {
    int              idx;
    void             *ptr;
    struct _PtrToIdx *next;
} PtrToIdx;

#define MAX_PTRS 10000

static PtrToIdx PtrArray[MAX_PTRS];
static PtrToIdx *avail=0;
static int      DoInit = 1;

static void PetscInitPointer()
{
  int  i;

  for (i=0; i<MAX_PTRS-1; i++) {
    PtrArray[i].next = PtrArray + i + 1;
    PtrArray[i].idx  = i;
  }
  PtrArray[MAX_PTRS-1].next = 0;
  avail   = PtrArray + 1;
}

void *PetscToPointer(int idx )
{
  if (DoInit) {
    DoInit = 0;
    PetscInitPointer();
  }
  if (idx < 0 || idx >= MAX_PTRS) {
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    fprintf( stderr, "[%d]PETSC ERROR: Could not convert index %d into a pointer\n",rank, idx );
    fprintf( stderr, "[%d]PETSC ERROR: The index may be an incorrect argument.\n\
PETSC ERROR:Possible sources of this problem are a missing include file,\n\
PETSC ERROR:a misspelled PETSC object (e.g., VIEWER_STOUT_WORLD instead of VIEWER_STDOUT_WORLD)\n\
PETSC ERROR:or a misspelled user variable for an PETSc object (e.g., \n\
PETSC ERROR:com instead of comm).\n",rank );
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  if (idx == 0) return (void *)0;
  return PtrArray[idx].ptr;
}

int PetscFromPointer(void *ptr )
{
  int      idx,rank;
  PtrToIdx *newl;

  if (DoInit) {
    DoInit = 0;
    PetscInitPointer();
  }
  if (!ptr) return 0;
  if (avail) {
    newl	      = avail;
    avail	      = avail->next;
    newl->next	      = 0;
    idx		      = newl->idx;
    PtrArray[idx].ptr = ptr;
    return idx;
  }
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  /* This isn't the right thing to do, but it isn't too bad */
  fprintf( stderr, "[%d]PETSC ERROR:Pointer conversions exhausted\n",rank );
  fprintf(stderr, "[%d]PETSC ERROR:Too many PETSc objects may have been passed to/from Fortran\n\
  without being freed\n",rank );
  return MPI_Abort(PETSC_COMM_WORLD,1);
}

void PetscRmPointer(int idx )
{
  if (DoInit) {
    DoInit = 0;
    PetscInitPointer();
  }
  if (idx < 0 || idx >= MAX_PTRS) {
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    fprintf( stderr, "[%d]PETSC ERROR:Could not convert index %d into a pointer\n",rank, idx );
    fprintf( stderr, "[%d]PETSC ERROR:The index may be an incorrect argument.\n\
PETSC ERROR:Possible sources of this problem are a missing include file,\n\
PETSC ERROR:a misspelled PETSC object (e.g., VIEWER_STOUT_WORLD instead of VIEWER_STDOUT_WORLD)\n\
PETSC ERROR:or a misspelled user variable for an PETSc object (e.g., \n\
PETSC ERROR:com instead of comm).\n",rank );
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  if (idx == 0) return;
  if (PtrArray[idx].next) {
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    fprintf(stderr,"[%d] Error in recovering Fortran pointer; already freed\n",rank);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  PtrArray[idx].next = avail;
  PtrArray[idx].ptr  = 0;
  avail              = PtrArray + idx;
}
#endif




