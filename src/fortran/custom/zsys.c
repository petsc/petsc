
#include "src/fortran/custom/zpetsc.h"
#include "petscsys.h"
#include "petscmatlab.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscpopsignalhandler_     PETSCPOPSIGNALHANDLER
#define petscgetcputime_           PETSCGETCPUTIME
#define petscfopen_                PETSCFOPEN
#define petscfclose_               PETSCFCLOSE
#define petscfprintf_              PETSCFPRINTF
#define petscsynchronizedfprintf_  PETSCSYNCHRONIZEDFPRINTF
#define petscprintf_               PETSCPRINTF
#define petscsynchronizedprintf_   PETSCSYNCHRONIZEDPRINTF
#define petscsynchronizedflush_    PETSCSYNCHRONIZEDFLUSH
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
#define petscmallocvalidate_       PETSCMALLOCVALIDATE
#define petscrealview_             PETSCREALVIEW
#define petscintview_              PETSCINTVIEW
#define petscsequentialphasebegin_ PETSCSEQUENTIALPHASEBEGIN
#define petscsequentialphaseend_   PETSCSEQUENTIALPHASEEND
#define petsctrlog_                PETSCTRLOG
#define petscmemcpy_               PETSCMEMCPY
#define petscmallocdump_           PETSCMALLOCDUMP
#define petscmallocdumplog_        PETSCMALLOCDUMPLOG
#define petscmemzero_              PETSCMEMZERO
#define petscbinaryopen_           PETSCBINARYOPEN
#define petscbinaryread_           PETSCBINARYREAD
#define petscbinarywrite_          PETSCBINARYWRITE
#define petscbinaryclose_          PETSCBINARYCLOSE
#define petscbinaryseek_           PETSCBINARYSEEK
#define petscfixfilename_          PETSCFIXFILENAME
#define petscstrncpy_              PETSCSTRNCPY
#define petscbarrier_              PETSCBARRIER
#define petscsynchronizedflush_    PETSCSYNCHRONIZEDFLUSH
#define petscsplitownership_       PETSCSPLITOWNERSHIP
#define petscsplitownershipblock_  PETSCSPLITOWNERSHIPBLOCK
#define petscobjectgetnewtag_      PETSCOBJECTGETNEWTAG
#define petsccommgetnewtag_        PETSCCOMMGETNEWTAG
#define petscfptrap_               PETSCFPTRAP
#define petscoffsetfortran_        PETSCOFFSETFORTRAN
#define petscmatlabenginecreate_      PETSCMATLABENGINECREATE
#define petscmatlabenginedestroy_     PETSCMATLABENGINEDESTROY
#define petscmatlabengineevaluate_    PETSCMATLABENGINEEVALUATE
#define petscmatlabenginegetoutput_   PETSCMATLABENGINEGETOUTPUT
#define petscmatlabengineprintoutput_ PETSCMATLABENGINEPRINTOUTPUT
#define petscmatlabengineput_         PETSCMATLABENGINEPUT
#define petscmatlabengineget_         PETSCMATLABENGINEGET
#define petscmatlabengineputarray_    PETSCMATLABENGINEPUTARRAY
#define petscmatlabenginegetarray_    PETSCMATLABENGINEGETARRAY
#define petscgetmemoryusage    _      PETSCGETMEMORYUSAGE
#define petscviewerasciiprintf_       PETSCVIEWERASCIIPRINTF
#define petscviewerasciisynchronizedprintf_       PETSCVIEWERASCIISYNCHRONIZEDPRINTF
#define petscviewerasciisettab_       PETSCVIEWERASCIISETTAB
#define petscviewerasciipushtab_      PETSCVIEWERASCIIPUSHTAB
#define petscviewerasciipoptab_       PETSCVIEWERASCIIPOPTAB
#define petscviewerasciiusetabs_      PETSCVIEWERASCIIUSETABS
#define petscpusherrorhandler_        PETSCPUSHERRORHANDLER
#define petscpoperrorhandler_         PETSCPOPERRORHANDLER
#define petsctracebackerrorhandler_   PETSCTRACEBACKERRORHANDLER
#define petscaborterrorhandler_       PETSCABORTERRORHANDLER
#define petscignoreerrorhandler_      PETSCIGNOREERRORHANDLER
#define petscemacsclienterrorhandler_ PETSCEMACSCLIENTERRORHANDLER
#define petscattachdebuggererrorhandler_   PETSCATTACHDEBUGGERERRORHANDLER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscpopsignalhandler_        petscpopsignalhandler
#define petscfopen_                   petscfopen
#define petscfclose_                  petscfclose
#define petscfprintf_                 petscfprintf
#define petscsynchronizedfprintf_     petscsynchronizedfprintf
#define petscprintf_                  petscprintf
#define petscsynchronizedprintf_      petscsynchronizedprintf
#define petscsynchronizedflush_       petscsynchronizedflush
#define petscmatlabenginecreate_      petscmatlabenginecreate
#define petscmatlabenginedestroy_     petscmatlabenginedestroy
#define petscmatlabengineevaluate_    petscmatlabengineevaluate
#define petscmatlabenginegetoutput_   petscmatlabenginegetoutput
#define petscmatlabengineprintoutput_ petscmatlabengineprintoutput
#define petscmatlabengineput_         petscmatlabengineput
#define petscmatlabengineget_         petscmatlabengineget
#define petscmatlabengineputarray_    petscmatlabengineputarray
#define petscmatlabenginegetarray_    petscmatlabenginegetarray
#define petscoffsetfortran_        petscoffsetfortran     
#define chkmemfortran_             chkmemfortran
#define petscobjectgetnewtag_      petscobjectgetnewtag
#define petsccommgetnewtag_        petsccommgetnewtag
#define petscsplitownership_       petscsplitownership
#define petscsplitownershipblock_  petscsplitownershipblock
#define petscbarrier_              petscbarrier
#define petscstrncpy_              petscstrncpy
#define petscfixfilename_          petscfixfilename
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
#define petscmallocvalidate_       petscmallocvalidate
#define petscrealview_             petscrealview
#define petscintview_              petscintview
#define petscsequentialphasebegin_ petscsequentialphasebegin
#define petscsequentialphaseend_   petscsequentialphaseend
#define petscmemcpy_               petscmemcpy
#define petscmallocdump_           petscmallocdump
#define petscmallocdumplog_        petscmallocdumplog
#define petscmemzero_              petscmemzero
#define petscbinaryopen_           petscbinaryopen
#define petscbinaryread_           petscbinaryread
#define petscbinarywrite_          petscbinarywrite
#define petscbinaryclose_          petscbinaryclose
#define petscbinaryseek_           petscbinaryseek
#define petscsynchronizedflush_    petscsynchronizedflush
#define petscfptrap_               petscfptrap
#define petscgetcputime_           petscgetcputime
#define petscgetmemoryusage_       petscgetmemoryusage
#define petscviewerasciiprintf_    petscviewerasciiprintf
#define petscviewerasciisynchronizedprintf_    petscviewerasciisynchronizedprintf
#define petscviewerasciisettab_ petscviewerasciisettab
#define petscviewerasciipushtab_ petscviewerasciipushtab
#define petscviewerasciipoptab_ petscviewerasciipoptab
#define petscviewerasciiusetabs_ petscviewerasciiusetabs
#define petscpusherrorhandler_   petscpusherrorhandler
#define petscpoperrorhandler_    petscpoperrorhandler
#define petsctracebackerrorhandler_   petsctracebackerrorhandler
#define petscaborterrorhandler_       petscaborterrorhandler
#define petscignoreerrorhandler_      petscignoreerrorhandler
#define petscemacsclienterrorhandler_ petscemacsclienterrorhandler
#define petscattachdebuggererrorhandler_   petscattachdebuggererrorhandler
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f2)(int*,const CHAR PETSC_MIXED_LEN(len1),const CHAR PETSC_MIXED_LEN(len2),const CHAR PETSC_MIXED_LEN(len3),int*,int*,const CHAR PETSC_MIXED_LEN(len4),void*,PetscErrorCode* PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3) PETSC_END_LEN(len4));
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourerrorhandler(int line,const char *fun,const char *file,const char *dir,int n,int p,const char *mess,void *ctx)
{
  PetscErrorCode ierr = 0;
  size_t len1,len2,len3,len4;
  int l1,l2,l3,l4;

  PetscStrlen(fun,&len1); l1 = (int)len1;
  PetscStrlen(file,&len2);l2 = (int)len2;
  PetscStrlen(dir,&len3);l3 = (int)len3;
  PetscStrlen(mess,&len4);l4 = (int)len4;

#if defined(PETSC_USES_CPTOFCD)
 {
   CHAR fun_c,file_c,dir_c,mess_c;

   fun_c  = _cptofcd(fun,len1);
   file_c = _cptofcd(file,len2);
   dir_c  = _cptofcd(dir,len3);
   mess_c = _cptofcd(mess,len4);
   (*f2)(&line,fun_c,file_c,dir_c,&n,&p,mess_c,ctx,&ierr,len1,len2,len3,len4);

 }
#elif defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
  (*f2)(&line,fun,l1,file,l2,dir,l3,&n,&p,mess,l4,ctx,&ierr);
#else
  (*f2)(&line,fun,file,dir,&n,&p,mess,ctx,&ierr,l1,l2,l3,l4);
#endif
  return ierr;
}

EXTERN_C_BEGIN
/*
    integer i_x,i_y,shift
    Vec     x,y
    PetscScalar  v_x(1),v_y(1)

    call VecGetArray(x,v_x,i_x,ierr)
    if (x .eq. y) then
      call PetscOffsetFortran(y_v,x_v,shift,ierr)
      i_y = i_x + shift
    else 
      call VecGetArray(y,v_y,i_y,ierr)
    endif
*/

/*
        These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code
   
   functions, hence no STDCALL
*/
void petsctracebackerrorhandler_(int *line,const char *fun,const char *file,const char *dir,int *n,int *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscTraceBackErrorHandler(*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscaborterrorhandler_(int *line,const char *fun,const char *file,const char *dir,int *n,int *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscAbortErrorHandler(*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscattachdebuggererrorhandler_(int *line,const char *fun,const char *file,const char *dir,int *n,int *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscAttachDebuggerErrorHandler(*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscemacsclienterrorhandler_(int *line,const char *fun,const char *file,const char *dir,int *n,int *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscEmacsClientErrorHandler(*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscignoreerrorhandler_(int *line,const char *fun,const char *file,const char *dir,int *n,int *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscIgnoreErrorHandler(*line,fun,file,dir,*n,*p,mess,ctx);
}

void PETSC_STDCALL petscpusherrorhandler_(void (PETSC_STDCALL *handler)(int*,const CHAR PETSC_MIXED_LEN(len1),const CHAR PETSC_MIXED_LEN(len2),const CHAR PETSC_MIXED_LEN(len3),int*,int*,const CHAR PETSC_MIXED_LEN(len4),void*,PetscErrorCode* PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3) PETSC_END_LEN(len4)),void *ctx,PetscErrorCode *ierr)
{
  if ((void(*)(void))handler == (void(*)(void))petsctracebackerrorhandler_) {
    *ierr = PetscPushErrorHandler(PetscTraceBackErrorHandler,0);
  } else {
    f2    = handler;
    *ierr = PetscPushErrorHandler(ourerrorhandler,ctx);
  }
}

void PETSC_STDCALL petscpopsignalhandler_(PetscErrorCode *ierr)
{
  *ierr = PetscPopSignalHandler();
}

void PETSC_STDCALL petscpoperrorhandler_(PetscErrorCode *ierr)
{
  *ierr = PetscPopErrorHandler();
}

void PETSC_STDCALL petscviewerasciisettab_(PetscViewer *viewer,PetscInt *tabs,PetscErrorCode *ierr)
{
  *ierr = PetscViewerASCIISetTab(*viewer,*tabs);
}

void PETSC_STDCALL petscviewerasciipushtab_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  *ierr = PetscViewerASCIIPushTab(*viewer);
}

void PETSC_STDCALL petscviewerasciipoptab_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  *ierr = PetscViewerASCIIPopTab(*viewer);
}

void PETSC_STDCALL petscviewerasciiusetabs_(PetscViewer *viewer,PetscTruth *flg,PetscErrorCode *ierr)
{
  *ierr = PetscViewerASCIIUseTabs(*viewer,*flg);
}

void PETSC_STDCALL petscviewerasciiprintf_(PetscViewer *viewer,CHAR str PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(str,len1,c1);
  *ierr = PetscViewerASCIIPrintf(*viewer,c1);
  FREECHAR(str,c1);
}

void PETSC_STDCALL petscviewerasciisynchronizedprintf_(PetscViewer *viewer,CHAR str PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(str,len1,c1);
  *ierr = PetscViewerASCIISynchronizedPrintf(*viewer,c1);
  FREECHAR(str,c1);
}

void PETSC_STDCALL petscmemorygetcurrentusage_(PetscLogDouble *foo, PetscErrorCode *ierr)
{
  *ierr = PetscMemoryGetCurrentUsage(foo);
}

void PETSC_STDCALL petscmemorygetmaximumusage_(PetscLogDouble *foo, PetscErrorCode *ierr)
{
  *ierr = PetscMemoryGetMaximumUsage(foo);
}

void PETSC_STDCALL petscoffsetfortran_(PetscScalar *x,PetscScalar *y,PetscInt *shift,PetscErrorCode *ierr)
{
  *ierr = 0;
  *shift = y - x;
}

void PETSC_STDCALL petscgetcputime_(PetscLogDouble *t, PetscErrorCode *ierr)
{
  *ierr = PetscGetCPUTime(t);
}

void PETSC_STDCALL petscfopen_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),CHAR fmode PETSC_MIXED_LEN(len2),
                               FILE **file,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(fname,len1,c1);
  FIXCHAR(fmode,len2,c2);
  *ierr = PetscFOpen((MPI_Comm)PetscToPointerComm(*comm),c1,c2,file);
  FREECHAR(fname,c1);
  FREECHAR(fmode,c2);
}
  
void PETSC_STDCALL petscfclose_(MPI_Comm *comm,FILE **file,PetscErrorCode *ierr)
{
  *ierr = PetscFClose((MPI_Comm)PetscToPointerComm(*comm),*file);
}

void PETSC_STDCALL petscsynchronizedflush_(MPI_Comm *comm,PetscErrorCode *ierr)
{
  *ierr = PetscSynchronizedFlush((MPI_Comm)PetscToPointerComm(*comm));
}

void PETSC_STDCALL petscfprintf_(MPI_Comm *comm,FILE **file,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscFPrintf((MPI_Comm)PetscToPointerComm(*comm),*file,c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscprintf_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscPrintf((MPI_Comm)PetscToPointerComm(*comm),c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsynchronizedfprintf_(MPI_Comm *comm,FILE **file,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscSynchronizedFPrintf((MPI_Comm)PetscToPointerComm(*comm),*file,c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsynchronizedprintf_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscSynchronizedPrintf((MPI_Comm)PetscToPointerComm(*comm),c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsetfptrap_(PetscFPTrap *flag,PetscErrorCode *ierr)
{
  *ierr = PetscSetFPTrap(*flag);
}

void PETSC_STDCALL petscobjectgetnewtag_(PetscObject *obj,PetscMPIInt *tag,PetscErrorCode *ierr)
{
  *ierr = PetscObjectGetNewTag(*obj,tag);
}

void PETSC_STDCALL petsccommgetnewtag_(MPI_Comm *comm,PetscMPIInt *tag,PetscErrorCode *ierr)
{
  *ierr = PetscCommGetNewTag((MPI_Comm)PetscToPointerComm(*comm),tag);
}

void PETSC_STDCALL petscsplitownershipblock_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscErrorCode *ierr)
{
  *ierr = PetscSplitOwnershipBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,n,N);
}
void PETSC_STDCALL petscsplitownership_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscErrorCode *ierr)
{
  *ierr = PetscSplitOwnership((MPI_Comm)PetscToPointerComm(*comm),n,N);
}

void PETSC_STDCALL petscbarrier_(PetscObject *obj,PetscErrorCode *ierr)
{
  *ierr = PetscBarrier(*obj);
}

void PETSC_STDCALL petscstrncpy_(CHAR s1 PETSC_MIXED_LEN(len1),CHAR s2 PETSC_MIXED_LEN(len2),int *n,
                                 PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;
  int  m;

#if defined(PETSC_USES_CPTOFCD)
  t1 = _fcdtocp(s1); 
  t2 = _fcdtocp(s2); 
  m = *n; if (_fcdlen(s1) < m) m = _fcdlen(s1); if (_fcdlen(s2) < m) m = _fcdlen(s2);
#else
  t1 = s1;
  t2 = s2;
  m = *n; if (len1 < m) m = len1; if (len2 < m) m = len2;
#endif
  *ierr = PetscStrncpy(t1,t2,m);
}

void PETSC_STDCALL petscfixfilename_(CHAR filein PETSC_MIXED_LEN(len1),CHAR fileout PETSC_MIXED_LEN(len2),
                                     PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  PetscInt  i,n;
  char *in,*out;

#if defined(PETSC_USES_CPTOFCD)
  in  = _fcdtocp(filein); 
  out = _fcdtocp(fileout); 
  n   = _fcdlen (filein); 
#else
  in  = filein;
  out = fileout;
  n   = len1;
#endif

  for (i=0; i<n; i++) {
    if (in[i] == PETSC_REPLACE_DIR_SEPARATOR) out[i] = PETSC_DIR_SEPARATOR;
    else out[i] = in[i];
  }
  out[i] = 0;
}

void PETSC_STDCALL petscbinaryopen_(CHAR name PETSC_MIXED_LEN(len),PetscViewerFileType *type,int *fd,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscBinaryOpen(c1,*type,fd);
  FREECHAR(name,c1);
}

void PETSC_STDCALL petscbinarywrite_(int *fd,void *p,PetscInt *n,PetscDataType *type,PetscTruth *istemp,PetscErrorCode *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

void PETSC_STDCALL petscbinaryread_(int *fd,void *p,PetscInt *n,PetscDataType *type,PetscErrorCode *ierr)
{
  *ierr = PetscBinaryRead(*fd,p,*n,*type);
}

void PETSC_STDCALL petscbinaryseek_(int *fd,PetscInt *size,PetscBinarySeekType *whence,off_t *offset,PetscErrorCode *ierr)
{
  *ierr = PetscBinarySeek(*fd,*size,*whence,offset);
}

void PETSC_STDCALL petscbinaryclose_(int *fd,PetscErrorCode *ierr)
{
  *ierr = PetscBinaryClose(*fd);
}

/* ---------------------------------------------------------------------------------*/
void PETSC_STDCALL petscmemzero_(void *a,PetscInt *n,PetscErrorCode *ierr) 
{
  *ierr = PetscMemzero(a,*n);
}

void PETSC_STDCALL  petscmallocdump_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDump(stdout);
}
void PETSC_STDCALL petscmallocdumplog_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDumpLog(stdout);
}

void PETSC_STDCALL petscmemcpy_(int *out,int *in,int *length,PetscErrorCode *ierr)
{
  *ierr = PetscMemcpy(out,in,*length);
}

/*
        This version does not do a malloc 
*/
static char FIXCHARSTRING[1024];
#if defined(PETSC_USES_CPTOFCD)
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
    *ierr = PetscStrncpy(b,_fcdtocp(a),n); \
    if (*ierr) return; \
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
      *ierr = PetscStrncpy(b,a,n); \
      if (*ierr) return; \
      b[n] = 0; \
    } else b = a;\
  } \
}

#endif

void PETSC_STDCALL chkmemfortran_(int *line,CHAR file PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHARNOMALLOC(file,len,c1);
  *ierr = PetscMallocValidate(*line,"Userfunction",c1," ");
}

void PETSC_STDCALL petscmallocvalidate_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocValidate(0,"Unknown Fortran",0,0);
}

void PETSC_STDCALL petscrandomgetvalue_(PetscRandom *r,PetscScalar *val,PetscErrorCode *ierr)
{
  *ierr = PetscRandomGetValue(*r,val);
}


void PETSC_STDCALL petscobjectgetname_(PetscObject *obj,CHAR name PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tmp;
  *ierr = PetscObjectGetName(*obj,&tmp);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name);
  int  len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tmp,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tmp,len);if (*ierr) return;
#endif
}

void PETSC_STDCALL petscobjectdestroy_(PetscObject *obj,PetscErrorCode *ierr)
{
  *ierr = PetscObjectDestroy(*obj);
}

void PETSC_STDCALL petscobjectgetcomm_(PetscObject *obj,int *comm,PetscErrorCode *ierr)
{
  MPI_Comm c;
  *ierr = PetscObjectGetComm(*obj,&c);
  *(int*)comm = PetscFromPointerComm(c);
}

void PETSC_STDCALL petscattachdebugger_(PetscErrorCode *ierr)
{
  *ierr = PetscAttachDebugger();
}

void PETSC_STDCALL petscobjectsetname_(PetscObject *obj,CHAR name PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;

  FIXCHAR(name,len,t1);
  *ierr = PetscObjectSetName(*obj,t1);
  FREECHAR(name,t1);
}

void PETSC_STDCALL petscerror_(int *number,int *p,CHAR message PETSC_MIXED_LEN(len),
                               PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(message,len,t1);
  *ierr = PetscError(-1,0,0,0,*number,*p,t1);
  FREECHAR(message,t1);
}

void PETSC_STDCALL petscgetflops_(PetscLogDouble *d,PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscGetFlops(d);
#else
  ierr = 0;
  *d     = 0.0;
#endif
}

void PETSC_STDCALL petscrandomcreate_(MPI_Comm *comm,PetscRandomType *type,PetscRandom *r,PetscErrorCode *ierr)
{
  *ierr = PetscRandomCreate((MPI_Comm)PetscToPointerComm(*comm),*type,r);
}

void PETSC_STDCALL petscrandomdestroy_(PetscRandom *r,PetscErrorCode *ierr)
{
  *ierr = PetscRandomDestroy(*r);
}

void PETSC_STDCALL petscrealview_(PetscInt *n,PetscReal *d,int *viwer,PetscErrorCode *ierr)
{
  *ierr = PetscRealView(*n,d,0);
}

void PETSC_STDCALL petscintview_(PetscInt *n,PetscInt *d,int *viwer,PetscErrorCode *ierr)
{
  *ierr = PetscIntView(*n,d,0);
}

void PETSC_STDCALL petscsequentialphasebegin_(MPI_Comm *comm,PetscInt *ng,PetscErrorCode *ierr){
*ierr = PetscSequentialPhaseBegin(
	(MPI_Comm)PetscToPointerComm(*comm),*ng);
}
void PETSC_STDCALL petscsequentialphaseend_(MPI_Comm *comm,PetscInt *ng,PetscErrorCode *ierr){
*ierr = PetscSequentialPhaseEnd(
	(MPI_Comm)PetscToPointerComm(*comm),*ng);
}


#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_MAT_SINGLE)

void PETSC_STDCALL petscmatlabenginecreate_(MPI_Comm *comm,CHAR m PETSC_MIXED_LEN(len),PetscMatlabEngine *e,
                                            PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;

  FIXCHAR(m,len,ms);
  *ierr = PetscMatlabEngineCreate((MPI_Comm)PetscToPointerComm(*comm),ms,e);
  FREECHAR(m,ms);
}

void PETSC_STDCALL petscmatlabenginedestroy_(PetscMatlabEngine *e,PetscErrorCode *ierr)
{
  *ierr = PetscMatlabEngineDestroy(*e);
}

void PETSC_STDCALL petscmatlabengineevaluate_(PetscMatlabEngine *e,CHAR m PETSC_MIXED_LEN(len),
                                              PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;
  FIXCHAR(m,len,ms);
  *ierr = PetscMatlabEngineEvaluate(*e,ms);
  FREECHAR(m,ms);
}

void PETSC_STDCALL petscmatlabengineput_(PetscMatlabEngine *e,PetscObject *o,PetscErrorCode *ierr)
{
  *ierr = PetscMatlabEnginePut(*e,*o);
}

void PETSC_STDCALL petscmatlabengineget_(PetscMatlabEngine *e,PetscObject *o,PetscErrorCode *ierr)
{
  *ierr = PetscMatlabEngineGet(*e,*o);
}

void PETSC_STDCALL petscmatlabengineputarray_(PetscMatlabEngine *e,PetscInt *m,PetscInt *n,PetscScalar *a,
                                              CHAR s PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;
  FIXCHAR(s,len,ms);
  *ierr = PetscMatlabEnginePutArray(*e,*m,*n,a,ms);
  FREECHAR(s,ms);
}

void PETSC_STDCALL petscmatlabenginegetarray_(PetscMatlabEngine *e,PetscInt *m,PetscInt *n,PetscScalar *a,
                                              CHAR s PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;
  FIXCHAR(s,len,ms);
  *ierr = PetscMatlabEngineGetArray(*e,*m,*n,a,ms);
  FREECHAR(s,ms);
}

#endif
/*
EXTERN int PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
EXTERN int PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
*/

EXTERN_C_END


