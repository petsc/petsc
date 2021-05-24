#include <petsc/private/fortranimpl.h>
#include <petscdmshell.h>       /*I    "petscdmshell.h"  I*/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmshellsetcreatematrix_                DMSHELLSETCREATEMATRIX
#define dmshellsetcreateglobalvector_          DMSHELLSETCREATEGLOBALVECTOR_
#define dmshellsetcreatelocalvector_           DMSHELLSETCREATELOCALVECTOR_
#define dmshellsetglobaltolocal_               DMSHELLSETGLOBALTOLOCAL_
#define dmshellsetlocaltoglobal_               DMSHELLSETLOCALTOGLOBAL_
#define dmshellsetlocaltolocal_                DMSHELLSETLOCALTOLOCAL_
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmshellsetcreatematrix_                dmshellsetcreatematrix
#define dmshellsetcreateglobalvector_          dmshellsetcreateglobalvector
#define dmshellsetcreatelocalvector_           dmshellsetcreatelocalvector
#define dmshellsetglobaltolocal_               dmshellsetglobaltolocal
#define dmshellsetlocaltoglobal_               dmshellsetlocaltoglobal
#define dmshellsetlocaltolocal_                dmshellsetlocaltolocal_
#endif

/*
 * C routines are required for matrix and global vector creation.  We define C routines here that call the corresponding
 * Fortran routine (indexed by _cb) that was set by the user.
 */

static struct {
  PetscFortranCallbackId creatematrix;
  PetscFortranCallbackId createglobalvector;
  PetscFortranCallbackId createlocalvector;
  PetscFortranCallbackId globaltolocalbegin;
  PetscFortranCallbackId globaltolocalend;
  PetscFortranCallbackId localtoglobalbegin;
  PetscFortranCallbackId localtoglobalend;
  PetscFortranCallbackId localtolocalbegin;
  PetscFortranCallbackId localtolocalend;
} _cb;

static PetscErrorCode ourcreatematrix(DM dm,Mat *A)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.creatematrix,(DM*,Mat*,PetscErrorCode*),(&dm,A,&ierr));
}

static PetscErrorCode ourcreateglobalvector(DM dm,Vec *v)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.createglobalvector,(DM*,Vec*,PetscErrorCode*),(&dm,v,&ierr));
}

static PetscErrorCode ourcreatelocalvector(DM dm,Vec *v)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.createlocalvector,(DM*,Vec*,PetscErrorCode*),(&dm,v,&ierr));
}

static PetscErrorCode ourglobaltolocalbegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.globaltolocalbegin,(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),(&dm,&g,&mode,&l,&ierr));
}

static PetscErrorCode ourglobaltolocalend(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.globaltolocalend,(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),(&dm,&g,&mode,&l,&ierr));
}

static PetscErrorCode ourlocaltoglobalbegin(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.localtoglobalbegin,(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),(&dm,&l,&mode,&g,&ierr));
}

static PetscErrorCode ourlocaltoglobalend(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.localtoglobalend,(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),(&dm,&l,&mode,&g,&ierr));
}

static PetscErrorCode ourlocaltolocalbegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.localtolocalbegin,(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),(&dm,&g,&mode,&l,&ierr));
}

static PetscErrorCode ourlocaltolocalend(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscObjectUseFortranCallbackSubType(dm,_cb.localtolocalend,(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),(&dm,&g,&mode,&l,&ierr));
}

PETSC_EXTERN void dmshellsetcreatematrix_(DM *dm,void (*func)(DM*,Mat*,PetscErrorCode*,PETSC_FORTRAN_CHARLEN_T len),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.creatematrix,(PetscVoidFunction)func,NULL);if (*ierr) return;
  *ierr = DMShellSetCreateMatrix(*dm,ourcreatematrix);
}

PETSC_EXTERN void dmshellsetcreateglobalvector_(DM *dm,void (*func)(DM*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.createglobalvector,(PetscVoidFunction)func,NULL);if (*ierr) return;
  *ierr = DMShellSetCreateGlobalVector(*dm,ourcreateglobalvector);
}

PETSC_EXTERN void dmshellsetcreatelocalvector_(DM *dm,void (*func)(DM*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.createlocalvector,(PetscVoidFunction)func,NULL);if (*ierr) return;
  *ierr = DMShellSetCreateLocalVector(*dm,ourcreatelocalvector);
}

PETSC_EXTERN void dmshellsetglobaltolocal_(DM *dm,void (*begin)(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),void (*end)(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.globaltolocalbegin,(PetscVoidFunction)begin,NULL);if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.globaltolocalend,(PetscVoidFunction)end,NULL);if (*ierr) return;
  *ierr = DMShellSetGlobalToLocal(*dm,ourglobaltolocalbegin,ourglobaltolocalend);
}

PETSC_EXTERN void dmshellsetlocaltoglobal_(DM *dm,void (*begin)(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),void (*end)(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.localtoglobalbegin,(PetscVoidFunction)begin,NULL);if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.localtoglobalend,(PetscVoidFunction)end,NULL);if (*ierr) return;
  *ierr = DMShellSetLocalToGlobal(*dm,ourlocaltoglobalbegin,ourlocaltoglobalend);
}

PETSC_EXTERN void dmshellsetlocaltolocal_(DM *dm,void (*begin)(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),void (*end)(DM*,Vec*,InsertMode*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.localtolocalbegin,(PetscVoidFunction)begin,NULL);if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm,PETSC_FORTRAN_CALLBACK_SUBTYPE,&_cb.localtolocalend,(PetscVoidFunction)end,NULL);if (*ierr) return;
  *ierr = DMShellSetLocalToLocal(*dm,ourlocaltolocalbegin,ourlocaltolocalend);
}
