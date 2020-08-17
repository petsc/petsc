#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecsetrandom_             VECSETRANDOM
#define vecsetvalueslocal0_       VECSETVALUESLOCAL0
#define vecsetvalueslocal11_      VECSETVALUESLOCAL11
#define vecsetvalueslocal1_       VECSETVALUESLOCAL1
#define vecgetvalues_             VECGETVALUES
#define vecgetvalues0_            VECGETVALUES0
#define vecgetvalues1_            VECGETVALUES1
#define vecgetvalues11_           VECGETVALUES11
#define vecsetvalues_             VECSETVALUES
#define vecsetvalues0_            VECSETVALUES0
#define vecsetvalues1_            VECSETVALUES1
#define vecsetvalues11_           VECSETVALUES11
#define vecsetvaluesblocked       VECSETVALUESBLOCKED
#define vecsetvaluesblocked0_     VECSETVALUESBLOCKED0
#define vecsetvaluesblocked1_     VECSETVALUESBLOCKED1
#define vecsetvaluesblocked11_    VECSETVALUESBLOCKED11
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
#define vecmin1_                  VECMIN1
#define vecmin2_                  VECMIN2
#define vecmax1_                  VECMAX1
#define vecmax2_                  VECMAX2
#define vecgetownershiprange1_    VECGETOWNERSHIPRANGE1
#define vecgetownershiprange2_    VECGETOWNERSHIPRANGE2
#define vecgetownershiprange3_    VECGETOWNERSHIPRANGE3
#define vecgetownershipranges_    VECGETOWNERSHIPRANGES
#define vecsetoptionsprefix_      VECSETOPTIONSPREFIX
#define vecviewfromoptions_       VECVIEWFROMOPTIONS
#define vecstashviewfromoptions_  VECSTASHVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecsetrandom_             vecsetrandom
#define vecsetvalueslocal0_       vecsetvalueslocal0
#define vecsetvalueslocal1_       vecsetvalueslocal1
#define vecsetvalueslocal11_      vecsetvalueslocal11
#define vecgetvalues_             vecgetvalues
#define vecgetvalues0_            vecgetvalues0
#define vecgetvalues1_            vecgetvalues1
#define vecgetvalues11_           vecgetvalues11
#define vecsetvalues_             vecsetvalues
#define vecsetvalues0_            vecsetvalues0
#define vecsetvalues1_            vecsetvalues1
#define vecsetvalues11_           vecsetvalues11
#define vecsetvaluesblocked_      vecsetvaluesblocked
#define vecsetvaluesblocked0_     vecsetvaluesblocked0
#define vecsetvaluesblocked1_     vecsetvaluesblocked1
#define vecsetvaluesblocked11_    vecsetvaluesblocked11
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
#define vecmin1_                  vecmin1
#define vecmin2_                  vecmin2
#define vecmax1_                  vecmax1
#define vecmax2_                  vecmax2
#define vecgetownershiprange1_    vecgetownershiprange1
#define vecgetownershiprange2_    vecgetownershiprange2
#define vecgetownershiprange3_    vecgetownershiprange3
#define vecgetownershipranges_    vecgetownershipranges
#define vecsetoptionsprefix_      vecsetoptionsprefix
#define vecviewfromoptions_       vecviewfromoptions
#define vecstashviewfromoptions_  vecstashviewfromoptions
#endif

PETSC_EXTERN void vecsetvalueslocal_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  *ierr = VecSetValuesLocal(*x,*ni,ix,y,*iora);
}

PETSC_EXTERN void vecsetvalueslocal0_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvalueslocal_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void vecsetvalueslocal1_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvalueslocal_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void vecsetvalueslocal11_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvalueslocal_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void vecgetvalues_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],int *ierr)
{
  *ierr = VecGetValues(*x,*ni,ix,y);
}

PETSC_EXTERN void vecgetvalues0_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],int *ierr)
{
  vecgetvalues_(x,ni,ix,y,ierr);
}

PETSC_EXTERN void vecgetvalues1_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],int *ierr)
{
  vecgetvalues_(x,ni,ix,y,ierr);
}

PETSC_EXTERN void vecgetvalues11_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],int *ierr)
{
  vecgetvalues_(x,ni,ix,y,ierr);
}

PETSC_EXTERN void  vecsetvalues_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  *ierr = VecSetValues(*x,*ni,ix,y,*iora);
}

PETSC_EXTERN void  vecsetvalues0_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvalues_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void  vecsetvalues1_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvalues_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void  vecsetvalues11_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvalues_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void  vecsetvaluesblocked_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  *ierr = VecSetValuesBlocked(*x,*ni,ix,y,*iora);
}

PETSC_EXTERN void  vecsetvaluesblocked0_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvaluesblocked_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void  vecsetvaluesblocked1_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvaluesblocked_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void  vecsetvaluesblocked11_(Vec *x,PetscInt *ni, PetscInt ix[], PetscScalar y[],InsertMode *iora, int *ierr)
{
  vecsetvaluesblocked_(x,ni,ix,y,iora,ierr);
}

PETSC_EXTERN void vecsetvalue_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  *ierr = VecSetValues(*v,1,i,va,*mode);
}
PETSC_EXTERN void vecsetvaluelocal_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  *ierr = VecSetValuesLocal(*v,1,i,va,*mode);
}

PETSC_EXTERN void vecload_(Vec *vec, PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoad(*vec,v);
}

PETSC_EXTERN void vecview_(Vec *x,PetscViewer *vin,PetscErrorCode *ierr)
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

     Notes:
    Allows code such as

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
PETSC_EXTERN void vecgetarrayaligned_(PetscErrorCode *ierr)
{
  VecGetArrayAligned = PETSC_TRUE;
}

PETSC_EXTERN void vecgetarray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
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
PETSC_EXTERN void vecrestorearray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt    m;
  PetscScalar *lx;

  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,&lx);if (*ierr) return;
  *ierr = VecRestoreArray(*x,&lx);if (*ierr) return;
}

PETSC_EXTERN void vecgetarrayread_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
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
PETSC_EXTERN void vecrestorearrayread_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
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
PETSC_EXTERN void vecduplicatevecs_(Vec *v,PetscInt *m,Vec *newv,PetscErrorCode *ierr)
{
  Vec      *lV;
  PetscInt i;
  *ierr = VecDuplicateVecs(*v,*m,&lV); if (*ierr) return;
  for (i=0; i<*m; i++) newv[i] = lV[i];
  *ierr = PetscFree(lV);
}

PETSC_EXTERN void vecdestroyvecs_(PetscInt *m,Vec *vecs,PetscErrorCode *ierr)
{
  PetscInt i;
  for (i=0; i<*m; i++) {
    *ierr = VecDestroy(&vecs[i]);if (*ierr) return;
  }
}

PETSC_EXTERN void vecmin1_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMin(*x,p,val);
}

PETSC_EXTERN void vecmin2_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMin(*x,p,val);
}

PETSC_EXTERN void vecmax1_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x,p,val);
}

PETSC_EXTERN void vecmax2_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x,p,val);
}

PETSC_EXTERN void vecgetownershiprange1_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}

PETSC_EXTERN void vecgetownershiprange2_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}

PETSC_EXTERN void vecgetownershiprange3_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}

PETSC_EXTERN void vecgetownershipranges_(Vec *x,PetscInt *range,PetscErrorCode *ierr)
{
  PetscMPIInt    size;
  const PetscInt *r;

  *ierr = MPI_Comm_size(PetscObjectComm((PetscObject)*x),&size);if (*ierr) return;
  *ierr = VecGetOwnershipRanges(*x,&r);if (*ierr) return;
  *ierr = PetscArraycpy(range,r,size+1);
}

PETSC_EXTERN void vecsetoptionsprefix_(Vec *v,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = VecSetOptionsPrefix(*v,t);if (*ierr) return;
  FREECHAR(prefix,t);
}
PETSC_EXTERN void vecviewfromoptions_(Vec *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = VecViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
PETSC_EXTERN void vecstashviewfromoptions_(Vec *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = VecStashViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
