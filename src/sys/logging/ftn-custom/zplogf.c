
#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclogview_             PETSCLOGVIEW
#define petsclogallbegin_         PETSCLOGALLBEGIN
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

PETSC_EXTERN void petsclogeventbegin_(PetscLogEvent *e,PetscErrorCode *ierr)
{
  *ierr = PetscLogEventBegin(*e,0,0,0,0);
}

PETSC_EXTERN void petsclogeventend_(PetscLogEvent *e,PetscErrorCode *ierr)
{
  *ierr = PetscLogEventEnd(*e,0,0,0,0);
}

PETSC_EXTERN void petsclogflops_(PetscLogDouble *f,PetscErrorCode *ierr)
{
  *ierr = PetscLogFlops(*f);
}

PETSC_EXTERN void petsclogview_(PetscViewer *viewer,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscLogView(v);
#endif
}

PETSC_EXTERN void petsclogdump_(char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *ierr = PetscLogDump(t1);if (*ierr) return;
  FREECHAR(name,t1);
#endif
}
PETSC_EXTERN void petsclogeventregister_(char* string,PetscClassId *classid,PetscLogEvent *e,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);
  *ierr = PetscLogEventRegister(t1,*classid,e);if (*ierr) return;
  FREECHAR(string,t1);
#endif
}
PETSC_EXTERN void petscclassidregister_(char* string,PetscClassId *e,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(string,len,t1);

  *ierr = PetscClassIdRegister(t1,e);if (*ierr) return;
  FREECHAR(string,t1);
#endif
}

PETSC_EXTERN void petsclogallbegin_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogAllBegin();
#endif
}

PETSC_EXTERN void petsclogdefaultbegin_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogDefaultBegin();
#endif
}

PETSC_EXTERN void petsclognestedbegin_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogNestedBegin();
#endif
}

PETSC_EXTERN void petsclogstagepop_(PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePop();
#endif
}

PETSC_EXTERN void petsclogstageregister_(char* sname,PetscLogStage *stage,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageRegister(t,stage);if (*ierr) return;
  FREECHAR(sname,t);
#endif
}

PETSC_EXTERN void petsclogstagepush_(PetscLogStage *stage,PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePush(*stage);
#endif
}

PETSC_EXTERN void petscgetflops_(PetscLogDouble *d,PetscErrorCode *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscGetFlops(d);
#else
  ierr = 0;
  *d   = 0.0;
#endif
}

PETSC_EXTERN void petsclogstagegetid_(char* sname,PetscLogStage *stage, int *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageGetId(t,stage);if (*ierr) return;
  FREECHAR(sname,t);
#endif
}
