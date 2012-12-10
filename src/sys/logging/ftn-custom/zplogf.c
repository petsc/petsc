
#include <petsc-private/fortranimpl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petsclogview_             PETSCLOGVIEW
#define petsclogprintDetailed_    PETSCLOGPRINTDETAILED
#define petsclogallbegin_         PETSCLOGALLBEGIN
#define petsclogdestroy_          PETSCLOGDESTROY
#define petsclogbegin_            PETSCLOGBEGIN
#define petsclogdump_             PETSCLOGDUMP
#define petsclogeventregister_    PETSCLOGEVENTREGISTER
#define petsclogstagepop_         PETSCLOGSTAGEPOP
#define petsclogstageregister_    PETSCLOGSTAGEREGISTER
#define petscclassidregister_     PETSCCLASSIDREGISTER
#define petsclogstagepush_        PETSCLOGSTAGEPUSH
#define petscgetflops_            PETSCGETFLOPS
#define petsclogstagegetid_       PETSCLOGSTAGEGETID
#define petsclogeventbegin_       PETSCLOGEVENTBEGIN
#define petsclogeventend_         PETSCLOGEVENTEND
#define petsclogflops_            PETSCLOGFLOPS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclogview_             petsclogview
#define petsclogprintDetailed_    petsclogprintDetailed
#define petsclogallbegin_         petsclogallbegin
#define petsclogdestroy_          petsclogdestroy
#define petsclogbegin_            petsclogbegin
#define petsclogeventregister_    petsclogeventregister
#define petsclogdump_             petsclogdump
#define petsclogstagepop_         petsclogstagepop
#define petsclogstageregister_    petsclogstageregister
#define petscclassidregister_     petscclassidregister
#define petsclogstagepush_        petsclogstagepush
#define petscgetflops_            petscgetflops
#define petsclogstagegetid_       petsclogstagegetid
#define petsclogeventbegin_       petsclogeventbegin
#define petsclogeventend_         petsclogeventend
#define petsclogflops_            petsclogflops
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petsclogeventbegin_(PetscLogEvent *e,PetscErrorCode *ierr){
  *ierr = PetscLogEventBegin(*e,0,0,0,0);
}

void PETSC_STDCALL petsclogeventend_(PetscLogEvent *e,PetscErrorCode *ierr){
  *ierr = PetscLogEventEnd(*e,0,0,0,0);
}

void PETSC_STDCALL petsclogflops_(PetscLogDouble *f,PetscErrorCode *ierr) {
  *ierr = PetscLogFlops(*f);
}

void PETSC_STDCALL petsclogview_(PetscViewer *viewer,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscLogView(v);
}

void PETSC_STDCALL petsclogprintDetailed_(MPI_Comm *comm,CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(filename,len,t);
  *ierr = PetscLogPrintDetailed(MPI_Comm_f2c(*(MPI_Fint *)&*comm),t);
  FREECHAR(filename,t);
#endif
}


void PETSC_STDCALL petsclogdump_(CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *ierr = PetscLogDump(t1);
  FREECHAR(name,t1);
#endif
}
void PETSC_STDCALL petsclogeventregister_(CHAR string PETSC_MIXED_LEN(len),PetscClassId *classid,PetscLogEvent *e,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);
  *ierr = PetscLogEventRegister(t1,*classid,e);
  FREECHAR(string,t1);
#endif
}
void PETSC_STDCALL petscclassidregister_(CHAR string PETSC_MIXED_LEN(len),PetscClassId *e,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);

  *ierr = PetscClassIdRegister(t1,e);
  FREECHAR(string,t1);
#endif
}

void PETSC_STDCALL petsclogallbegin_(PetscErrorCode *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogAllBegin();
#endif
}

void PETSC_STDCALL petsclogdestroy_(PetscErrorCode *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogDestroy();
#endif
}

void PETSC_STDCALL petsclogbegin_(PetscErrorCode *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogBegin();
#endif
}

void PETSC_STDCALL petsclogstagepop_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePop();
#endif
}

void PETSC_STDCALL petsclogstageregister_(CHAR sname PETSC_MIXED_LEN(len),PetscLogStage *stage,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageRegister(t,stage);
  FREECHAR(sname,t);
#endif
}

void PETSC_STDCALL petsclogstagepush_(PetscLogStage *stage,PetscErrorCode *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePush(*stage);
#endif
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

void PETSC_STDCALL   petsclogstagegetid_(CHAR sname PETSC_MIXED_LEN(len),PetscLogStage *stage, int *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageGetId(t,stage);
  FREECHAR(sname,t);
#endif
}

EXTERN_C_END
