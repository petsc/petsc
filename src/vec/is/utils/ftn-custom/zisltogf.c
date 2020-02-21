#include <petsc/private/fortranimpl.h>
#include <petscis.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define islocaltoglobalmappingview_       ISLOCALTOGLOBALMAPPINGVIEW
#define islocaltoglobalmpnggetinfosize_   ISLOCALTOGLOBALMPNGGETINFOSIZE
#define islocaltoglobalmappinggetinfo_    ISLOCALTOGLOBALMAPPINGGETINFO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define islocaltoglobalmappingview_       islocaltoglobalmappingview
#define islocaltoglobalmpnggetinfosize_   islocaltoglobalmpnggetinfosize
#define islocaltoglobalmappinggetinfo_    islocaltoglobalmappinggetinfo
#endif

PETSC_EXTERN void islocaltoglobalmappingview_(ISLocalToGlobalMapping *mapping,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISLocalToGlobalMappingView(*mapping,v);
}

static PetscInt  *sprocs, *snumprocs, **sindices;
static PetscBool called;
PETSC_EXTERN void islocaltoglobalmpnggetinfosize_(ISLocalToGlobalMapping *mapping,PetscInt *size,PetscInt *maxnumprocs,PetscErrorCode *ierr)
{
  PetscInt i;
  if (called) {*ierr = PETSC_ERR_ARG_WRONGSTATE; return;}
  *ierr        = ISLocalToGlobalMappingGetInfo(*mapping,size,&sprocs,&snumprocs,&sindices); if (*ierr) return;
  *maxnumprocs = 0;
  for (i=0; i<*size; i++) *maxnumprocs = PetscMax(*maxnumprocs,snumprocs[i]);
  called = PETSC_TRUE;
}

PETSC_EXTERN void islocaltoglobalmappinggetinfo_(ISLocalToGlobalMapping *mapping,PetscInt *size,PetscInt *procs,PetscInt *numprocs,
                                                  PetscInt *indices,PetscErrorCode *ierr)
{
  PetscInt i,j;
  if (!called) {*ierr = PETSC_ERR_ARG_WRONGSTATE; return;}
  *ierr = PetscArraycpy(procs,sprocs,*size); if (*ierr) return;
  *ierr = PetscArraycpy(numprocs,snumprocs,*size); if (*ierr) return;
  for (i=0; i<*size; i++) {
    for (j=0; j<numprocs[i]; j++) indices[i + (*size)*j] = sindices[i][j];
  }
  *ierr  = ISLocalToGlobalMappingRestoreInfo(*mapping,size,&sprocs,&snumprocs,&sindices); if (*ierr) return;
  called = PETSC_FALSE;
}

PETSC_EXTERN void islocaltoglobalmappingviewfromoptions_(ISLocalToGlobalMapping *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = ISLocalToGlobalMappingViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
