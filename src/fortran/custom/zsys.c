#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zsys.c,v 1.64 1999/05/08 14:00:39 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sys.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
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

void petscobjectgetnewtag_(PetscObject *obj,int *tag, int *__ierr )
{
  *__ierr = PetscObjectGetNewTag(*obj,tag);
}

void petscobjectrestorenewtag_(PetscObject *obj,int *tag, int *__ierr )
{
  *__ierr = PetscObjectRestoreNewTag(*obj,tag);
}

void petsccommgetnewtag_(MPI_Comm *comm,int *tag, int *__ierr )
{
  *__ierr = PetscCommGetNewTag((MPI_Comm)PetscToPointerComm(*comm),tag);
}

void petsccommrestorenewtag_(MPI_Comm *comm,int *tag, int *__ierr )
{
  *__ierr = PetscCommRestoreNewTag((MPI_Comm)PetscToPointerComm(*comm),tag);
}

void petscsplitownership_(MPI_Comm *comm,int *n,int *N, int *__ierr )
{
  *__ierr = PetscSplitOwnership((MPI_Comm)PetscToPointerComm(*comm),n,N);
}

void petscbarrier_(PetscObject *A, int *__ierr )
{
  *__ierr = PetscBarrier(*A);
}

void petscstrncpy_(CHAR s1, CHAR s2, int *n,int *__ierr,int len1, int len2)
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

void petscfixfilename_(CHAR filein ,CHAR fileout,int *__ierr,int len1,int len2)
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

void petscbinaryopen_(CHAR name,int *type,int *fd,int *__ierr,int len)
{
  int  ierr;
  char *c1;

  FIXCHAR(name,len,c1);
  ierr = PetscBinaryOpen(c1,*type,fd);
  FREECHAR(name,c1);
  *__ierr = ierr;
}

void petscbinarywrite_(int *fd,void *p,int *n,PetscDataType *type,int *istemp,int *__ierr)
{
  *__ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

void petscbinaryread_(int *fd,void *p,int *n,PetscDataType *type,int *__ierr)
{
  *__ierr = PetscBinaryRead(*fd,p,*n,*type);
}

void petscbinaryseek_(int *fd,int *size,PetscBinarySeekType *whence,int *__ierr)
{
  *__ierr = PetscBinarySeek(*fd,*size,*whence);
}

void petscbinaryclose_(int *fd,int *__ierr)
{
  *__ierr = PetscBinaryClose(*fd);
}

/* ---------------------------------------------------------------------------------*/
void petscmemzero_(void *a,int *n,int *__ierr) 
{
  *__ierr = PetscMemzero(a,*n);
}

void petsctrdump_(int *__ierr)
{
  *__ierr = PetscTrDump(stdout);
}
void petsctrlogdump_(int *__ierr)
{
  *__ierr = PetscTrLogDump(stdout);
}

void petscmemcpy_(int *out,int *in,int *length,int *__ierr)
{
  *__ierr = PetscMemcpy(out,in,*length);
}

void petsctrlog_(int *__ierr)
{
  *__ierr = PetscTrLog();
}

void petsctrvalid_(int *__ierr)
{
  *__ierr = PetscTrValid(0,"Unknown Fortran",0,0);
}

void petscrandomgetvalue_(PetscRandom *r,Scalar *val, int *__ierr )
{
  *__ierr = PetscRandomGetValue(*r,val);
}


void petscobjectgetname(PetscObject *obj, CHAR name, int *__ierr, int len)
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

void petscobjectdestroy_(PetscObject *obj, int *__ierr )
{
  *__ierr = PetscObjectDestroy(*obj);
}

void petscobjectgetcomm_(PetscObject *obj,int *comm, int *__ierr )
{
  MPI_Comm c;
  *__ierr = PetscObjectGetComm(*obj,&c);
  *(int*)comm = PetscFromPointerComm(c);
}

void petscattachdebugger_(int *__ierr)
{
  *__ierr = PetscAttachDebugger();
}

void petscobjectsetname_(PetscObject *obj,CHAR name,int *__ierr,int len)
{
  char *t1;

  FIXCHAR(name,len,t1);
  *__ierr = PetscObjectSetName(*obj,t1);
  FREECHAR(name,t1);
}

void petscerror_(int *number,int *p,CHAR message,int *__ierr,int len)
{
  char *t1;
  FIXCHAR(message,len,t1);
  *__ierr = PetscError(-1,0,"fortran_interface_unknown_file",0,*number,*p,t1);
  FREECHAR(message,t1);
}

void petscgetflops_(PLogDouble *d,int *__ierr)
{
#if defined(PETSC_USE_LOG)
  *__ierr = PetscGetFlops(d);
#else
  __ierr = 0;
  *d     = 0.0;
#endif
}

void petscrandomcreate_(MPI_Comm *comm,PetscRandomType *type,PetscRandom *r,int *__ierr )
{
  *__ierr = PetscRandomCreate((MPI_Comm)PetscToPointerComm( *comm ),*type,r);
}

void petscrandomdestroy_(PetscRandom *r, int *__ierr )
{
  *__ierr = PetscRandomDestroy(*r);
}

void petscdoubleview_(int *n,double *d,int *viwer,int *__ierr)
{
  *__ierr = PetscDoubleView(*n,d,0);
}

void petscintview_(int *n,int *d,int *viwer,int *__ierr)
{
  *__ierr = PetscIntView(*n,d,0);
}

void petscsequentialphasebegin_(MPI_Comm *comm,int *ng, int *__ierr ){
*__ierr = PetscSequentialPhaseBegin(
	(MPI_Comm)PetscToPointerComm( *comm ),*ng);
}
void petscsequentialphaseend_(MPI_Comm *comm,int *ng, int *__ierr ){
*__ierr = PetscSequentialPhaseEnd(
	(MPI_Comm)PetscToPointerComm( *comm ),*ng);
}

void petscreleasepointer_(int *index,int *__ierr) 
{
   PetscRmPointer(index);
   *__ierr = 0;
}

void petscsynchronizedflush_(MPI_Comm *comm,int *__ierr){
*__ierr = PetscSynchronizedFlush(
	(MPI_Comm)PetscToPointerComm( *comm));
}

EXTERN_C_END


