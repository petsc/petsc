#ifndef lint
static char vcid[] = "$Id: zsys.c,v 1.15 1996/03/23 17:15:03 curfman Exp curfman $";
#endif

#include "zpetsc.h"
#include "sys.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define plogeventbegin_       PLOGEVENTBEGIN
#define plogeventend_         PLOGEVENTEND
#define plogflops_            PLOGFLOPS
#define petscattachdebugger_  PETSCATTACHDEBUGGER
#define plogallbegin_         PLOGALLBEGIN
#define plogdestroy_          PLOGDESTROY
#define plogbegin_            PLOGBEGIN
#define petscobjectsetname_   PETSCOBJECTSETNAME
#define petscobjectdestroy_   PETSCOBJECTDESTROY
#define petscobjectgetcomm_   PETSCOBJECTGETCOMM
#define petscobjectgetname_   PETSCOBJECTGETNAME
#define plogdump_             PLOGDUMP
#define plogeventregister_    PLOGEVENTREGISTER
#define plogstagepop_         PLOGSTAGEPOP
#define petscgettime_         PETSCGETTIME
#define petscgetflops_        PETSCGETFLOPS
#define petscerror_           PETSCERROR
#define petscrandomcreate_    PETSCRANDOMCREATE
#define petscrandomdestroy_   PETSCRANDOMDESTROY
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define plogeventbegin_       plogeventbegin
#define plogeventend_         plogeventend
#define plogflops_            plogflops
#define petscattachdebugger_  petscattachdebugger
#define plogallbegin_         plogallbegin
#define plogdestroy_          plogdestroy
#define plogbegin_            plogbegin
#define petscobjectsetname_   petscobjectsetname
#define petscobjectdestroy_   petscobjectdestroy
#define petscobjectgetcomm_   petscobjectgetcomm
#define petscobjectgetname_   petscobjectgetname
#define plogeventregister_    plogeventregister
#define plogdump_             plogdump
#define plogstagepop_         plogstagepop  
#define petscgettime_         petscgettime  
#define petscgetflops_        petscgetflops 
#define petscerror_           petscerror
#define petscrandomcreate_    petscrandomcreate
#define petscrandomdestroy_   petscrandomdestroy
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void plogdump_(CHAR name, int *__ierr,int len ){
#if defined(PETSC_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *__ierr = PLogDump(t1);
  FREECHAR(name,t1);
#endif
}
void plogeventregister_(int *e,CHAR string,CHAR color,int *__ierr,int len1,
                        int len2){
#if defined(PETSC_LOG)
  char *t1,*t2;
  FIXCHAR(string,len1,t1);
  FIXCHAR(color,len2,t2);
  *__ierr = PLogEventRegister(e,t1,t2);
#endif
}

void petscobjectgetname(PetscObject obj, CHAR name, int *__ierr, int len)
{
  char *tmp;
  *__ierr = PetscObjectGetName((PetscObject)MPIR_ToPointer(*(int*)(obj)),
                               &tmp);
#if defined(PARCH_t3d)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  PetscStrncpy(t,tmp,len1);
  }
#else
  PetscStrncpy(name,tmp,len);
#endif
}

void petscobjectdestroy_(PetscObject obj, int *__ierr ){
  *__ierr = PetscObjectDestroy((PetscObject)MPIR_ToPointer(*(int*)(obj)));
  MPIR_RmPointer(*(int*)(obj));
}

void petscobjectgetcomm_(PetscObject obj,MPI_Comm *comm, int *__ierr ){
  MPI_Comm c;
  *__ierr = PetscObjectGetComm((PetscObject)MPIR_ToPointer(*(int*)(obj)),&c);
  *(int*)comm = MPIR_FromPointer_Comm(c);
}

void petscattachdebugger_(int *__ierr){
  *__ierr = PetscAttachDebugger();
}

void plogallbegin_(int *__ierr){
#if defined(PETSC_LOG)
  *__ierr = PLogAllBegin();
#endif
}

void plogdestroy_(int *__ierr){
#if defined(PETSC_LOG)
  *__ierr = PLogDestroy();
#endif
}

void plogbegin_(int *__ierr){
#if defined(PETSC_LOG)
  *__ierr = PLogBegin();
#endif
}

void plogeventbegin_(int e,int o1,int o2,int o3,int o4){
#if defined(PETSC_LOG)
  PetscObject t1,t2,t3,t4;
  if (o1) t1 = (PetscObject) MPIR_ToPointer(*(int*)(o1)); else t1 = 0;
  if (o2) t2 = (PetscObject) MPIR_ToPointer(*(int*)(o2)); else t2 = 0;
  if (o3) t3 = (PetscObject) MPIR_ToPointer(*(int*)(o3)); else t3 = 0;
  if (o4) t4 = (PetscObject) MPIR_ToPointer(*(int*)(o4)); else t4 = 0;

  if (_PLB) (*_PLB)(e,1,t1,t2,t3,t4);
#if defined(HAVE_MPE)
  if (UseMPE && MPEFlags[e]) MPE_Log_event(MPEBEGIN+2*e,0,"");
#endif
#endif
}

void plogeventend_(int e,int o1,int o2,int o3,int o4){
#if defined(PETSC_LOG)
  PetscObject t1,t2,t3,t4;
  if (o1) t1 = (PetscObject) MPIR_ToPointer(*(int*)(o1)); else t1 = 0;
  if (o2) t2 = (PetscObject) MPIR_ToPointer(*(int*)(o2)); else t2 = 0;
  if (o3) t3 = (PetscObject) MPIR_ToPointer(*(int*)(o3)); else t3 = 0;
  if (o4) t4 = (PetscObject) MPIR_ToPointer(*(int*)(o4)); else t4 = 0;
  if (_PLE) (*_PLE)(e,1,t1,t2,t3,t4);
#if defined(HAVE_MPE)
  if (UseMPE && MPEFlags[e]) MPE_Log_event(MPEBEGIN+2*e+1,0,"");
#endif
#endif
}

void plogflops_(int f) {
  PLogFlops(f);
}

/*
      This bleeds memory, but no easy way to get around it
*/
void petscobjectsetname_(PetscObject obj,CHAR name,int *__ierr,int len)
{
  char *t1;

  FIXCHAR(name,len,t1);
  *__ierr = PetscObjectSetName((PetscObject)MPIR_ToPointer(*(int*)(obj)),t1);
}

void petscerror_(int *number,CHAR message,int *__ierr,int len)
{
  char *t1;
  FIXCHAR(message,len,t1);
  *__ierr = PetscError(-1,0,"fortran_interface_unknown_file",*number,t1);
}

void plogstagepop_(int *__ierr )
{
#if defined(PETSC_LOG)
  *__ierr = PLogStagePop();
#endif
}
double petscgettime_()
{ 
  return PetscGetTime();
}

double  petscgetflops_()
{
  return PetscGetFlops();
}

void petscrandomcreate_(MPI_Comm comm,PetscRandomType *type,PetscRandom *r,
                     int *__ierr )
{
  PetscRandom rr;
  *__ierr = PetscRandomCreate(
	(MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),*type,&rr);
  *(int*)r = MPIR_FromPointer(rr);
}

void petscrandomdestroy_(PetscRandom *r, int *__ierr ){
  *__ierr = PetscRandomDestroy((PetscRandom )MPIR_ToPointer( *(int*)(r) ));
   MPIR_RmPointer(*(int*)(r)); 
}

#if defined(__cplusplus)
}
#endif

/*
    This is for when one is compiling with the Edenburgh version of
   of MPI which uses integers for the MPI objects, hence the PETSc 
   objects require routines provided here to do the conversion between
   C pointers and Fortran integers.
*/
#if defined(_T3DMPI_RELEASE_ID)
/* ----------------------------------------------------------------*/
/*    This code was taken from the MPICH implementation of MPI.    */
/*
 *  $Id: zsys.c,v 1.15 1996/03/23 17:15:03 curfman Exp curfman $
 *
 *  (C) 1994 by Argonne National Laboratory and Mississipi State University.
 *      See COPYRIGHT in top-level directory.
 */

/* 
   This file contains routines to convert to and from pointers
*/

typedef struct _PtrToIdx {
    int idx;
    void *ptr;
    struct _PtrToIdx *next;
} PtrToIdx;
#define MAX_PTRS 10000

static PtrToIdx PtrArray[MAX_PTRS];
static PtrToIdx *avail=0;
static int      DoInit = 1;

static void MPIR_InitPointer()
{
  int  i;

  for (i=0; i<MAX_PTRS-1; i++) {
    PtrArray[i].next = PtrArray + i + 1;
    PtrArray[i].idx  = i;
  }
  PtrArray[MAX_PTRS-1].next = 0;
/* Don't start with the first one, whose index is 0. That could
   break some code. */
  avail   = PtrArray + 1;
}

void *MPIR_ToPointer(int idx )
{
  if (DoInit) {
    DoInit = 0;
    MPIR_InitPointer();
  }
  if (idx < 0 || idx >= MAX_PTRS) {
    fprintf( stderr, "Could not convert index %d into a pointer\n", idx );
    fprintf( stderr, "The index may be an incorrect argument.\n\
Possible sources of this problem are a missing \"include 'mpif.h'\",\n\
a misspelled MPI object (e.g., MPI_COM_WORLD instead of MPI_COMM_WORLD)\n\
or a misspelled user variable for an MPI object (e.g., \n\
com instead of comm).\n" );
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if (idx == 0) return (void *)0;
  return PtrArray[idx].ptr;
}

int MPIR_FromPointer(void *ptr )
{
  int      idx;
  PtrToIdx *new;
  if (DoInit) {
    DoInit = 0;
    MPIR_InitPointer();
  }
  if (!ptr) return 0;
  if (avail) {
    new		      = avail;
    avail	      = avail->next;
    new->next	      = 0;
    idx		      = new->idx;
    PtrArray[idx].ptr = ptr;
    return idx;
  }
  /* This isn't the right thing to do, but it isn't too bad */
  fprintf( stderr, "Pointer conversions exhausted\n" );
  fprintf(stderr, "Too many MPI objects may have been passed to/from Fortran\n\
  without being freed\n" );
  MPI_Abort(MPI_COMM_WORLD,1);
}

void MPIR_RmPointer(int idx )
{
  int myrank;
  if (DoInit) {
    DoInit = 0;
    MPIR_InitPointer();
  }
  if (idx < 0 || idx >= MAX_PTRS) {
    fprintf( stderr, "Could not convert index %d into a pointer\n", idx );
    fprintf( stderr, "The index may be an incorrect argument.\n\
Possible sources of this problem are a missing \"include 'mpif.h'\",\n\
a misspelled MPI object (e.g., MPI_COM_WORLD instead of MPI_COMM_WORLD)\n\
or a misspelled user variable for an MPI object (e.g., \n\
com instead of comm).\n" );
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if (idx == 0) return;
  if (PtrArray[idx].next) {
    /* In-use pointers NEVER have next set */
    MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    fprintf( stderr, 
	    "[%d] Error in recovering Fortran pointer; already freed\n", 
	    myrank );
    MPI_Abort(MPI_COMM_WORLD,1);
    return;
  }
  PtrArray[idx].next = avail;
  PtrArray[idx].ptr  = 0;
  avail              = PtrArray + idx;
}
#endif
