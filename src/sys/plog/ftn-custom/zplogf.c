
#include "private/fortranimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petsclogprintsummary_     PETSCLOGPRINTSUMMARY
#define petsclogprintDetailed_    PETSCLOGPRINTDETAILED
#define petsclogallbegin_         PETSCLOGALLBEGIN
#define petsclogdestroy_          PETSCLOGDESTROY
#define petsclogbegin_            PETSCLOGBEGIN
#define petsclogdump_             PETSCLOGDUMP
#define petsclogeventregister_    PETSCLOGEVENTREGISTER
#define petsclogstagepop_         PETSCLOGSTAGEPOP
#define petsclogstageregister_    PETSCLOGSTAGEREGISTER
#define petsccookieregister_      PETSCCOOKIEREGISTER
#define petsclogstagepush_        PETSCLOGSTAGEPUSH
#define petscgetflops_            PETSCGETFLOPS
#define petsclogstagegetid_       PETSCLOGSTAGEGETID
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclogprintsummary_     petsclogprintsummary
#define petsclogprintDetailed_    petsclogprintDetailed
#define petsclogallbegin_         petsclogallbegin
#define petsclogdestroy_          petsclogdestroy
#define petsclogbegin_            petsclogbegin
#define petsclogeventregister_    petsclogeventregister
#define petsclogdump_             petsclogdump
#define petsclogstagepop_         petsclogstagepop  
#define petsclogstageregister_    petsclogstageregister
#define petsccookieregister_      petsccookieregister
#define petsclogstagepush_        petsclogstagepush
#define petscgetflops_            petscgetflops 
#define petsclogstagegetid_       petsclogstagegetid
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petsclogprintsummary_(MPI_Comm *comm,CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(filename,len,t);
  *ierr = PetscLogPrintSummary(MPI_Comm_f2c(*(MPI_Fint *)&*comm),t);
  FREECHAR(filename,t);
#endif
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
void PETSC_STDCALL petsclogeventregister_(CHAR string PETSC_MIXED_LEN(len),PetscCookie *cookie,PetscLogEvent *e,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);
  *ierr = PetscLogEventRegister(t1,*cookie,e);
  FREECHAR(string,t1);
#endif
}
void PETSC_STDCALL petsccookieregister_(CHAR string PETSC_MIXED_LEN(len),PetscCookie *e,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);

  *ierr = PetscCookieRegister(t1,e);
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
