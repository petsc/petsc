
#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclogview_             PETSCLOGVIEW
#define petsclogallbegin_         PETSCLOGALLBEGIN
#define petsclogdestroy_          PETSCLOGDESTROY
#define petsclogdefaultbegin_     PETSCLOGDEFAULTBEGIN
#define petsclognestedbegin_      PETSCLOGNESTEDBEGIN
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
#define petsclogallbegin_         petsclogallbegin
#define petsclogdestroy_          petsclogdestroy
#define petsclogdefaultbegin_     petsclogdefaultbegin
#define petsclognestedbegin_      petsclognestedbegin
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

PETSC_EXTERN void PETSC_STDCALL petsclogeventbegin_(PetscLogEvent *e,PetscErrorCode *ierr)
{
  *ierr = PetscLogEventBegin(*e,0,0,0,0);
}

PETSC_EXTERN void PETSC_STDCALL petsclogeventend_(PetscLogEvent *e,PetscErrorCode *ierr)
{
  *ierr = PetscLogEventEnd(*e,0,0,0,0);
}

PETSC_EXTERN void PETSC_STDCALL petsclogflops_(PetscLogDouble *f,PetscErrorCode *ierr)
{
  *ierr = PetscLogFlops(*f);
}

PETSC_EXTERN void PETSC_STDCALL petsclogview_(PetscViewer *viewer,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscLogView(v);
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogdump_(char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *ierr = PetscLogDump(t1);
  FREECHAR(name,t1);
#endif
}
PETSC_EXTERN void PETSC_STDCALL petsclogeventregister_(char* string PETSC_MIXED_LEN(len),PetscClassId *classid,PetscLogEvent *e,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);
  *ierr = PetscLogEventRegister(t1,*classid,e);
  FREECHAR(string,t1);
#endif
}
PETSC_EXTERN void PETSC_STDCALL petscclassidregister_(char* string PETSC_MIXED_LEN(len),PetscClassId *e,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);

  *ierr = PetscClassIdRegister(t1,e);
  FREECHAR(string,t1);
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogallbegin_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogAllBegin();
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogdestroy_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogDestroy();
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogdefaultbegin_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogDefaultBegin();
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclognestedbegin_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogNestedBegin();
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogstagepop_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePop();
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogstageregister_(char* sname PETSC_MIXED_LEN(len),PetscLogStage *stage,PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageRegister(t,stage);
  FREECHAR(sname,t);
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogstagepush_(PetscLogStage *stage,PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePush(*stage);
#endif
}

PETSC_EXTERN void PETSC_STDCALL petscgetflops_(PetscLogDouble *d,PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscGetFlops(d);
#else
  ierr = 0;
  *d   = 0.0;
#endif
}

PETSC_EXTERN void PETSC_STDCALL petsclogstagegetid_(char* sname PETSC_MIXED_LEN(len),PetscLogStage *stage, int *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageGetId(t,stage);
  FREECHAR(sname,t);
#endif
}
