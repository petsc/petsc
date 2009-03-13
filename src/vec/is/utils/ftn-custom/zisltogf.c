#include "private/fortranimpl.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define islocaltoglobalmappingview_       ISLOCALTOGLOBALMAPPINGVIEW
#define islocaltoglobalmpnggetinfosize_   ISLOCALTOGLOBALMPNGGETINFOSIZE
#define islocaltoglobalmappinggetinfo_    ISLOCALTOGLOBALMAPPINGGETINFO
#define iscompressindicesgeneral_         ISCOMPRESSINDICESGENERAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define islocaltoglobalmappingview_       islocaltoglobalmappingview
#define islocaltoglobalmpnggetinfosize_   islocaltoglobalmpnggetinfosize
#define islocaltoglobalmappinggetinfo_    islocaltoglobalmappinggetinfo
#define iscompressindicesgeneral_         iscompressindicesgeneral
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscompressindicesgeneral_(PetscInt *n,PetscInt *bs,PetscInt *imax,IS *is_in,IS *is_out,PetscErrorCode *ierr)
{
  *ierr = ISCompressIndicesGeneral(*n,*bs,*imax,is_in,is_out);
}

void PETSC_STDCALL islocaltoglobalmappingview_(ISLocalToGlobalMapping *mapping,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISLocalToGlobalMappingView(*mapping,v);
}

static PetscInt   *sprocs, *snumprocs, **sindices;
static PetscTruth called;
void PETSC_STDCALL islocaltoglobalmpnggetinfosize_(ISLocalToGlobalMapping *mapping,PetscInt *nprocs,PetscInt *maxnumprocs,PetscErrorCode *ierr)
{
  PetscInt i;
  if (called) {*ierr = PETSC_ERR_ARG_WRONGSTATE; return;}
  *ierr        = ISLocalToGlobalMappingGetInfo(*mapping,nprocs,&sprocs,&snumprocs,&sindices); if (*ierr) return;
  *maxnumprocs = 0;
  for (i=0; i<*nprocs; i++) {
    *maxnumprocs = PetscMax(*maxnumprocs,snumprocs[i]);
  }
  called = PETSC_TRUE;
}

void PETSC_STDCALL islocaltoglobalmappinggetinfo_(ISLocalToGlobalMapping *mapping,PetscInt *nprocs,PetscInt *procs,PetscInt *numprocs,
                                                  PetscInt *indices,PetscErrorCode *ierr)
{
  PetscInt i,j;
  if (!called) {*ierr = PETSC_ERR_ARG_WRONGSTATE; return;}
  *ierr = PetscMemcpy(procs,sprocs,*nprocs*sizeof(PetscInt)); if (*ierr) return;
  *ierr = PetscMemcpy(numprocs,snumprocs,*nprocs*sizeof(PetscInt)); if (*ierr) return;
  for (i=0; i<*nprocs; i++) {
    for (j=0; j<numprocs[i]; j++) {
      indices[i + (*nprocs)*j] = sindices[i][j];
    }
  }
  *ierr = ISLocalToGlobalMappingRestoreInfo(*mapping,nprocs,&sprocs,&snumprocs,&sindices); if (*ierr) return;
  called = PETSC_FALSE;
}

EXTERN_C_END
