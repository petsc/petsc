/*$Id: zsys.c,v 1.71 1999/10/05 15:13:27 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "sys.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define chkmemfortran_             CHKMEMFORTRAN
#define petscattachdebugger_       PETSCATTACHDEBUGGER
#define petscobjectsetname_        PETSCOBJECTSETNAME
#define petscobjectdestroy_        PETSCOBJECTDESTROY
#define petscobjectgetcomm_        PETSCOBJECTGETCOMM
#define petscobjectgetname_        PETSCOBJECTGETNAME
#define petscgetflops_             PETSCGETFLOPS
#define petscerror_                PETSCERROR
#define petscrandomcreate_         PETSCRANDOMCREATE
#define petscrandomdestroy_        PETSCRANDOMDESTROY
#define petscrandomgetvalue_       PETSCRANDOMGETVALUE
#define petsctrvalid_              PETSCTRVALID
#define petscdoubleview_           PETSCDOUBLEVIEW
#define petscintview_              PETSCINTVIEW
#define petscsequentialphasebegin_ PETSCSEQUENTIALPHASEBEGIN
#define petscsequentialphaseend_   PETSCSEQUENTIALPHASEEND
#define petsctrlog_                PETSCTRLOG
#define petscmemcpy_               PETSCMEMCPY
#define petsctrdump_               PETSCTRDUMP
#define petsctrlogdump_            PETSCTRLOGDUMP
#define petscmemzero_              PETSCMEMZERO
#define petscbinaryopen_           PETSCBINARYOPEN
#define petscbinaryread_           PETSCBINARYREAD
#define petscbinarywrite_          PETSCBINARYWRITE
#define petscbinaryclose_          PETSCBINARYCLOSE
#define petscbinaryseek_           PETSCBINARYSEEK
#define petscfixfilename_          PETSCFIXFILENAME
#define petscreleasepointer_       PETSCRELEASEPOINTER
#define petscstrncpy_              PETSCSTRNCPY
#define petscbarrier_              PETSCBARRIER
#define petscsynchronizedflush_    PETSCSYNCHRONIZEDFLUSH
#define petscsplitownership_       PETSCSPLITOWNERSHIP
#define petscobjectgetnewtag_      PETSCOBJECTGETNEWTAG
#define petscobjectrestorenewtag_  PETSCOBJECTRESTORENEWTAG
#define petsccommgetnewtag_        PETSCCOMMGETNEWTAG
#define petsccommrestorenewtag_    PETSCCOMMRESTORENEWTAG
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define chkmemfortran_             chkmemfortran
#define petscobjectgetnewtag_      petscobjectgetnewtag
#define petscobjectrestorenewtag_  petscobjectrestorenewtag
#define petsccommgetnewtag_        petsccommgetnewtag
#define petsccommrestorenewtag_    petsccommrestorenewtag
#define petscsplitownership_       petscsplitownership
#define petscbarrier_              petscbarrier
#define petscstrncpy_              petscstrncpy
#define petscreleasepointer_       petscreleasepointer
#define petscfixfilename_          petscfixfilename
#define petsctrlog_                petsctrlog
#define petscattachdebugger_       petscattachdebugger
#define petscobjectsetname_        petscobjectsetname
#define petscobjectdestroy_        petscobjectdestroy
#define petscobjectgetcomm_        petscobjectgetcomm
#define petscobjectgetname_        petscobjectgetname
#define petscgetflops_             petscgetflops 
#define petscerror_                petscerror
#define petscrandomcreate_         petscrandomcreate
#define petscrandomdestroy_        petscrandomdestroy
#define petscrandomgetvalue_       petscrandomgetvalue
#define petsctrvalid_              petsctrvalid
#define petscdoubleview_           petscdoubleview
#define petscintview_              petscintview
#define petscsequentialphasebegin_ petscsequentialphasebegin
#define petscsequentialphaseend_   petscsequentialphaseend
#define petscmemcpy_               petscmemcpy
#define petsctrdump_               petsctrdump
#define petsctrlogdump_            petsctlogrdump
#define petscmemzero_              petscmemzero
#define petscbinaryopen_           petscbinaryopen
#define petscbinaryread_           petscbinaryread
#define petscbinarywrite_          petscbinarywrite
#define petscbinaryclose_          petscbinaryclose
#define petscbinaryseek_           petscbinaryseek
#define petscsynchronizedflush_    petscsynchronizedflush
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscobjectgetnewtag_(PetscObject *obj,int *tag, int *__ierr )
{
  *__ierr = PetscObjectGetNewTag(*obj,tag);
}

void PETSC_STDCALL petscobjectrestorenewtag_(PetscObject *obj,int *tag, int *__ierr )
{
  *__ierr = PetscObjectRestoreNewTag(*obj,tag);
}

void PETSC_STDCALL petsccommgetnewtag_(MPI_Comm *comm,int *tag, int *__ierr )
{
  *__ierr = PetscCommGetNewTag((MPI_Comm)PetscToPointerComm(*comm),tag);
}

void PETSC_STDCALL petsccommrestorenewtag_(MPI_Comm *comm,int *tag, int *__ierr )
{
  *__ierr = PetscCommRestoreNewTag((MPI_Comm)PetscToPointerComm(*comm),tag);
}

void PETSC_STDCALL petscsplitownership_(MPI_Comm *comm,int *n,int *N, int *__ierr )
{
  *__ierr = PetscSplitOwnership((MPI_Comm)PetscToPointerComm(*comm),n,N);
}

void PETSC_STDCALL petscbarrier_(PetscObject *obj, int *__ierr )
{
  *__ierr = PetscBarrier(*obj);
}

void PETSC_STDCALL petscstrncpy_(CHAR s1, CHAR s2, int *n,int *__ierr,int len1, int len2)
{
  char *t1,*t2;
  int  m;

#if defined(USES_CPTOFCD)
  t1 = _fcdtocp(s1); 
  t2 = _fcdtocp(s2); 
  m = *n; if (_fcdlen(s1) < m) m = _fcdlen(s1); if (_fcdlen(s2) < m) m = _fcdlen(s2);
#else
  t1 = s1;
  t2 = s2;
  m = *n; if (len1 < m) m = len1; if (len2 < m) m = len2;
#endif
  *__ierr = PetscStrncpy(t1,t2,m);
}

void PETSC_STDCALL petscfixfilename_(CHAR filein ,CHAR fileout,int *__ierr,int len1,int len2)
{
  int  i,n;
  char *in,*out;

#if defined(USES_CPTOFCD)
  in  = _fcdtocp(filein); 
  out = _fcdtocp(fileout); 
  n   = _fcdlen (filein); 
#else
  in  = filein;
  out = fileout;
  n   = len1;
#endif

  for (i=0; i<n; i++) {
#if defined(PARCH_win32)
    if (in[i] == '/') out[i] = '\\';
#else
    if (in[i] == '\\') out[i] = '/';
#endif
    else out[i] = in[i];
  }
  out[i] = 0;
}

void PETSC_STDCALL petscbinaryopen_(CHAR name,int *type,int *fd,int *__ierr,int len)
{
  int  ierr;
  char *c1;

  FIXCHAR(name,len,c1);
  ierr = PetscBinaryOpen(c1,*type,fd);
  FREECHAR(name,c1);
  *__ierr = ierr;
}

void PETSC_STDCALL petscbinarywrite_(int *fd,void *p,int *n,PetscDataType *type,int *istemp,int *__ierr)
{
  *__ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

void PETSC_STDCALL petscbinaryread_(int *fd,void *p,int *n,PetscDataType *type,int *__ierr)
{
  *__ierr = PetscBinaryRead(*fd,p,*n,*type);
}

void PETSC_STDCALL petscbinaryseek_(int *fd,int *size,PetscBinarySeekType *whence,int *__ierr)
{
  *__ierr = PetscBinarySeek(*fd,*size,*whence);
}

void PETSC_STDCALL petscbinaryclose_(int *fd,int *__ierr)
{
  *__ierr = PetscBinaryClose(*fd);
}

/* ---------------------------------------------------------------------------------*/
void PETSC_STDCALL petscmemzero_(void *a,int *n,int *__ierr) 
{
  *__ierr = PetscMemzero(a,*n);
}

void PETSC_STDCALL petsctrdump_(int *__ierr)
{
  *__ierr = PetscTrDump(stdout);
}
void PETSC_STDCALL petsctrlogdump_(int *__ierr)
{
  *__ierr = PetscTrLogDump(stdout);
}

void PETSC_STDCALL petscmemcpy_(int *out,int *in,int *length,int *__ierr)
{
  *__ierr = PetscMemcpy(out,in,*length);
}

void PETSC_STDCALL petsctrlog_(int *__ierr)
{
  *__ierr = PetscTrLog();
}

/*
        This version does not do a malloc 
*/
static char FIXCHARSTRING[1024];
#if defined(USES_CPTOFCD)
#include <fortran.h>

#define CHAR _fcd
#define FIXCHARNOMALLOC(a,n,b) \
{ \
  b = _fcdtocp(a); \
  n = _fcdlen (a); \
  if (b == PETSC_NULL_CHARACTER_Fortran) { \
      b = 0; \
  } else {  \
    while((n > 0) && (b[n-1] == ' ')) n--; \
    b = FIXCHARSTRING; \
    *__ierr = PetscStrncpy(b,_fcdtocp(a),n); \
    if (*__ierr) return; \
    b[n] = 0; \
  } \
}

#else

#define CHAR char*
#define FIXCHARNOMALLOC(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else { \
    while((n > 0) && (a[n-1] == ' ')) n--; \
    if (a[n] != 0) { \
      b = FIXCHARSTRING; \
      *__ierr = PetscStrncpy(b,a,n); \
      if (*__ierr) return; \
      b[n] = 0; \
    } else b = a;\
  } \
}

#endif

void PETSC_STDCALL chkmemfortran_(int *line, CHAR file,int *__ierr,int len)
{
  char *c1;

  FIXCHARNOMALLOC(file,len,c1);
  *__ierr = PetscTrValid(*line,"Userfunction",c1," ");
}

void PETSC_STDCALL petsctrvalid_(int *__ierr)
{
  *__ierr = PetscTrValid(0,"Unknown Fortran",0,0);
}

void PETSC_STDCALL petscrandomgetvalue_(PetscRandom *r,Scalar *val, int *__ierr )
{
  *__ierr = PetscRandomGetValue(*r,val);
}


void PETSC_STDCALL petscobjectgetname_(PetscObject *obj, CHAR name, int *__ierr, int len)
{
  char *tmp;
  *__ierr = PetscObjectGetName(*obj,&tmp);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name);
  int  len1 = _fcdlen(name);
  *__ierr = PetscStrncpy(t,tmp,len1);if (*__ierr) return;
  }
#else
  *__ierr = PetscStrncpy(name,tmp,len);if (*__ierr) return;
#endif
}

void PETSC_STDCALL petscobjectdestroy_(PetscObject *obj, int *__ierr )
{
  *__ierr = PetscObjectDestroy(*obj);
}

void PETSC_STDCALL petscobjectgetcomm_(PetscObject *obj,int *comm, int *__ierr )
{
  MPI_Comm c;
  *__ierr = PetscObjectGetComm(*obj,&c);
  *(int*)comm = PetscFromPointerComm(c);
}

void PETSC_STDCALL petscattachdebugger_(int *__ierr)
{
  *__ierr = PetscAttachDebugger();
}

void PETSC_STDCALL petscobjectsetname_(PetscObject *obj,CHAR name,int *__ierr,int len)
{
  char *t1;

  FIXCHAR(name,len,t1);
  *__ierr = PetscObjectSetName(*obj,t1);
  FREECHAR(name,t1);
}

void PETSC_STDCALL petscerror_(int *number,int *p,CHAR message,int *__ierr,int len)
{
  char *t1;
  FIXCHAR(message,len,t1);
  *__ierr = PetscError(-1,0,"fortran_interface_unknown_file",0,*number,*p,t1);
  FREECHAR(message,t1);
}

void PETSC_STDCALL petscgetflops_(PLogDouble *d,int *__ierr)
{
#if defined(PETSC_USE_LOG)
  *__ierr = PetscGetFlops(d);
#else
  __ierr = 0;
  *d     = 0.0;
#endif
}

void PETSC_STDCALL petscrandomcreate_(MPI_Comm *comm,PetscRandomType *type,PetscRandom *r,int *__ierr )
{
  *__ierr = PetscRandomCreate((MPI_Comm)PetscToPointerComm( *comm ),*type,r);
}

void PETSC_STDCALL petscrandomdestroy_(PetscRandom *r, int *__ierr )
{
  *__ierr = PetscRandomDestroy(*r);
}

void PETSC_STDCALL petscdoubleview_(int *n,double *d,int *viwer,int *__ierr)
{
  *__ierr = PetscDoubleView(*n,d,0);
}

void PETSC_STDCALL petscintview_(int *n,int *d,int *viwer,int *__ierr)
{
  *__ierr = PetscIntView(*n,d,0);
}

void PETSC_STDCALL petscsequentialphasebegin_(MPI_Comm *comm,int *ng, int *__ierr ){
*__ierr = PetscSequentialPhaseBegin(
	(MPI_Comm)PetscToPointerComm( *comm ),*ng);
}
void PETSC_STDCALL petscsequentialphaseend_(MPI_Comm *comm,int *ng, int *__ierr ){
*__ierr = PetscSequentialPhaseEnd(
	(MPI_Comm)PetscToPointerComm( *comm ),*ng);
}

void PETSC_STDCALL petscreleasepointer_(int *index,int *__ierr) 
{
   PetscRmPointer(index);
   *__ierr = 0;
}

void PETSC_STDCALL petscsynchronizedflush_(MPI_Comm *comm,int *__ierr)
{
  *__ierr = PetscSynchronizedFlush((MPI_Comm)PetscToPointerComm( *comm));
}

EXTERN_C_END


