#include <petsc/private/fortranimpl.h>
#include <petsc/private/vecimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecsetvalue_              VECSETVALUE
#define vecsetvaluelocal_         VECSETVALUELOCAL
#define vecload_                  VECLOAD
#define vecview_                  VECVIEW
#define vecgetarray_              VECGETARRAY
#define vecgetarrayread_          VECGETARRAYREAD
#define vecgetarrayaligned_       VECGETARRAYALIGNED
#define vecrestorearray_          VECRESTOREARRAY
#define vecrestorearrayread_      VECRESTOREARRAYREAD
#define vecduplicatevecs_         VECDUPLICATEVECS
#define vecdestroyvecs_           VECDESTROYVECS
#define vecmax_                   VECMAX
#define vecgetownershiprange_     VECGETOWNERSHIPRANGE
#define vecgetownershipranges_    VECGETOWNERSHIPRANGES
#define vecsetoptionsprefix_      VECSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecgetarrayaligned_       vecgetarrayaligned
#define vecsetvalue_              vecsetvalue
#define vecsetvaluelocal_         vecsetvaluelocal
#define vecload_                  vecload
#define vecview_                  vecview
#define vecgetarray_              vecgetarray
#define vecrestorearray_          vecrestorearray
#define vecgetarrayaligned_       vecgetarrayaligned
#define vecgetarrayread_          vecgetarrayread
#define vecrestorearrayread_      vecrestorearrayread
#define vecduplicatevecs_         vecduplicatevecs
#define vecdestroyvecs_           vecdestroyvecs
#define vecmax_                   vecmax
#define vecgetownershiprange_     vecgetownershiprange
#define vecgetownershipranges_    vecgetownershipranges
#define vecsetoptionsprefix_      vecsetoptionsprefix
#endif

PETSC_EXTERN void PETSC_STDCALL vecsetvalue_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  *ierr = VecSetValues(*v,1,i,va,*mode);
}
PETSC_EXTERN void PETSC_STDCALL vecsetvaluelocal_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  *ierr = VecSetValuesLocal(*v,1,i,va,*mode);
}

PETSC_EXTERN void PETSC_STDCALL vecload_(Vec *vec, PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoad(*vec,v);
}

PETSC_EXTERN void PETSC_STDCALL vecview_(Vec *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = VecView(*x,v);
}

/*MC
         VecGetArrayAligned - FORTRAN only. Forces alignment of vector
      arrays so that arrays of derived types may be used.

   Synopsis:
   VecGetArrayAligned(PetscErrorCode ierr)

     Not Collective

     Notes: Allows code such as

$     type  :: Field
$        PetscScalar :: p1
$        PetscScalar :: p2
$      end type Field
$
$      type(Field)       :: lx_v(0:1)
$
$      call VecGetArray(localX, lx_v, lx_i, ierr)
$      call InitialGuessLocal(lx_v(lx_i/2),ierr)
$
$      subroutine InitialGuessLocal(a,ierr)
$      type(Field)     :: a(*)

     If you have not called VecGetArrayAligned() the code may generate incorrect data
     or crash.

     lx_i needs to be divided by the number of entries in Field (in this case 2)

     You do NOT need VecGetArrayAligned() if lx_v and a are arrays of PetscScalar

.seealso: VecGetArray(), VecGetArrayF90()
M*/
static PetscBool VecGetArrayAligned = PETSC_FALSE;
PETSC_EXTERN void PETSC_STDCALL vecgetarrayaligned_(PetscErrorCode *ierr)
{
  VecGetArrayAligned = PETSC_TRUE;
}

PETSC_EXTERN void PETSC_STDCALL vecgetarray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt    m,bs;

  *ierr = VecGetArray(*x,&lx); if (*ierr) return;
  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  bs    = 1;
  if (VecGetArrayAligned) {
    *ierr = VecGetBlockSize(*x,&bs);if (*ierr) return;
  }
  *ierr = PetscScalarAddressToFortran((PetscObject)*x,bs,fa,lx,m,ia);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
PETSC_EXTERN void PETSC_STDCALL vecrestorearray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt    m;
  PetscScalar *lx;

  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,&lx);if (*ierr) return;
  *ierr = VecRestoreArray(*x,&lx);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL vecgetarrayread_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscScalar *lx;
  PetscInt          m,bs;

  *ierr = VecGetArrayRead(*x,&lx); if (*ierr) return;
  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  bs    = 1;
  if (VecGetArrayAligned) {
    *ierr = VecGetBlockSize(*x,&bs);if (*ierr) return;
  }
  *ierr = PetscScalarAddressToFortran((PetscObject)*x,bs,fa,(PetscScalar*)lx,m,ia);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
PETSC_EXTERN void PETSC_STDCALL vecrestorearrayread_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt          m;
  const PetscScalar *lx;

  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,(PetscScalar**)&lx);if (*ierr) return;
  *ierr = VecRestoreArrayRead(*x,&lx);if (*ierr) return;
}

/*
      vecduplicatevecs() and vecdestroyvecs() are slightly different from C since the
    Fortran provides the array to hold the vector objects,while in C that
    array is allocated by the VecDuplicateVecs()
*/
PETSC_EXTERN void PETSC_STDCALL vecduplicatevecs_(Vec *v,PetscInt *m,Vec *newv,PetscErrorCode *ierr)
{
  Vec      *lV;
  PetscInt i;
  *ierr = VecDuplicateVecs(*v,*m,&lV); if (*ierr) return;
  for (i=0; i<*m; i++) newv[i] = lV[i];
  *ierr = PetscFree(lV);
}

PETSC_EXTERN void PETSC_STDCALL vecdestroyvecs_(PetscInt *m,Vec *vecs,PetscErrorCode *ierr)
{
  PetscInt i;
  for (i=0; i<*m; i++) {
    *ierr = VecDestroy(&vecs[i]);if (*ierr) return;
  }
}

PETSC_EXTERN void PETSC_STDCALL vecmax_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x,p,val);
}

PETSC_EXTERN void PETSC_STDCALL vecgetownershiprange_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}

PETSC_EXTERN void PETSC_STDCALL vecgetownershipranges_(Vec *x,PetscInt *range,PetscErrorCode *ierr)
{
  PetscMPIInt    size;
  const PetscInt *r;

  *ierr = MPI_Comm_size((*x)->map->comm,&size);if (*ierr) return;
  *ierr = VecGetOwnershipRanges(*x,&r);if (*ierr) return;
  *ierr = PetscMemcpy(range,r,(size+1)*sizeof(PetscInt));
}

PETSC_EXTERN void PETSC_STDCALL vecsetoptionsprefix_(Vec *v,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = VecSetOptionsPrefix(*v,t);
  FREECHAR(prefix,t);
}
