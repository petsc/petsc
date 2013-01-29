#include <petsc-private/fortranimpl.h>
#include <petscdmshell.h>       /*I    "petscdmshell.h"  I*/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmshellsetcreatematrix_                DMSHELLSETCREATEMATRIX
#define dmshellsetcreateglobalvector_          DMSHELLSETCREATEGLOBALVECTOR_
#define dmshellsetcreatelocalvector_           DMSHELLSETCREATELOCALVECTOR_
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmshellsetcreatematrix_                dmshellsetcreatematrix
#define dmshellsetcreateglobalvector_          dmshellsetcreateglobalvector
#define dmshellsetcreatelocalvector_           dmshellsetcreatelocalvector
#endif

/*
 * C routines are required for matrix and global vector creation.  We define C routines here that call the corresponding
 * Fortran routine (indexed by _cb) that was set by the user.
 */

static struct {
  PetscFortranCallbackId creatematrix;
  PetscFortranCallbackId createglobalvector;
  PetscFortranCallbackId createlocalvector;
} _cb;

static PetscErrorCode ourcreatematrix(DM dm,MatType type,Mat *A)
{
  int len;
  char *ftype = (char*)type;
  if (type) {
    size_t slen;
    PetscStrlen(type,&slen);
    len = (int)slen;
  } else {
    type = PETSC_NULL_CHARACTER_Fortran;
    len = 0;
  }
  PetscObjectUseFortranCallback(dm,_cb.creatematrix,(DM*,CHAR PETSC_MIXED_LEN(),Mat*,PetscErrorCode* PETSC_END_LEN()),
                                (&dm,ftype PETSC_MIXED_LEN_CALL(len),A,&ierr PETSC_END_LEN_CALL(len)));
  return 0;
}

static PetscErrorCode ourcreateglobalvector(DM dm,Vec *v)
{
  PetscObjectUseFortranCallback(dm,_cb.createglobalvector,(DM*,Vec*,PetscErrorCode*),(&dm,v,&ierr));
  return 0;
}

static PetscErrorCode ourcreatelocalvector(DM dm,Vec *v)
{
  PetscObjectUseFortranCallback(dm,_cb.createlocalvector,(DM*,Vec*,PetscErrorCode*),(&dm,v,&ierr));
  return 0;
}

PETSC_EXTERN_C void PETSC_STDCALL dmshellsetcreatematrix_(DM *dm,void (PETSC_STDCALL *func)(DM*,CHAR type PETSC_MIXED_LEN(len),Mat*,PetscErrorCode* PETSC_END_LEN(len)),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.creatematrix,(PetscVoidFunction)func,PETSC_NULL);
  if (*ierr) return;
  *ierr = DMShellSetCreateMatrix(*dm,ourcreatematrix);
}

PETSC_EXTERN_C void PETSC_STDCALL dmshellsetcreateglobalvector_(DM *dm,void (PETSC_STDCALL *func)(DM*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.createglobalvector,(PetscVoidFunction)func,PETSC_NULL);
  if (*ierr) return;
  *ierr = DMShellSetCreateGlobalVector(*dm,ourcreateglobalvector);
}

PETSC_EXTERN_C void PETSC_STDCALL dmshellsetcreatelocalvector_(DM *dm,void (PETSC_STDCALL *func)(DM*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.createlocalvector,(PetscVoidFunction)func,PETSC_NULL);
  if (*ierr) return;
  *ierr = DMShellSetCreateLocalVector(*dm,ourcreatelocalvector);
}
