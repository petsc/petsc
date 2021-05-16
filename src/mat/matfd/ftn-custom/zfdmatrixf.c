#include <petsc/private/f90impl.h>
#include <petsc/private/matimpl.h>

/* Declare these pointer types instead of void* for clarity, but do not include petscts.h so that this code does have an actual reverse dependency. */
typedef struct _p_TS *TS;
typedef struct _p_SNES *SNES;

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matfdcoloringsetfunctionts_              MATFDCOLORINGSETFUNCTIONTS
#define matfdcoloringsetfunction_                MATFDCOLORINGSETFUNCTION
#define matfdcoloringview_                       MATFDCOLORINGVIEW
#define matfdcoloingsettype_                     MATFDCOLORINGSETTYPE
#define matfdcoloringgetperturbedcolumnsf90_     MATFDCOLORINGGETPERTURBEDCOLUMNSF90
#define matfdcoloringrestoreperturbedcolumnsf90_ MATFDCOLORINGRESTOREPERTURBEDCOLUMNSF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matfdcoloringsetfunctionts_              matfdcoloringsetfunctionts
#define matfdcoloringsetfunction_                matfdcoloringsetfunction
#define matfdcoloringview_                       matfdcoloringview
#define matfdcoloingsettype_                     matfdcoloringsettype
#define matfdcoloringgetperturbedcolumnsf90_     matfdcoloringgetperturbedcolumnsf90
#define matfdcoloringrestoreperturbedcolumnsf90_ matfdcoloringrestoreperturbedcolumnsf90
#endif

PETSC_EXTERN void matfdcoloringgetperturbedcolumnsf90_(MatFDColoring *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt       len;

  *__ierr = MatFDColoringGetPerturbedColumns(*x,&len,&fa);   if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*)fa,MPIU_INT,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matfdcoloringrestoreperturbedcolumnsf90_(MatFDColoring *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr,MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmatfdcoloringfunctionts(TS ts,PetscReal t,Vec x,Vec y,MatFDColoring fd)
{
  PetscErrorCode ierr = 0;
  (*(void (*)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*))(fd->ftn_func_pointer)) (&ts,&t,&x,&y,fd->ftn_func_cntx,&ierr);
  return ierr;
}

static PetscErrorCode ourmatfdcoloringfunctionsnes(SNES snes,Vec x,Vec y,MatFDColoring fd)
{
  PetscErrorCode ierr = 0;
  (*(void (*)(SNES*,Vec*,Vec*,void*,PetscErrorCode*))(fd->ftn_func_pointer)) (&snes,&x,&y,fd->ftn_func_cntx,&ierr);
  return ierr;
}

/*
        MatFDColoringSetFunction sticks the Fortran function and its context into the MatFDColoring structure and passes the MatFDColoring object
    in as the function context. ourmafdcoloringfunctionsnes() and ourmatfdcoloringfunctionts()  then access the function and its context from the
    MatFDColoring that is passed in. This is the same way that fortran_func_pointers is used in PETSc objects.

   NOTE: FORTRAN USER CANNOT PUT IN A NEW J OR B currently.
*/

PETSC_EXTERN void matfdcoloringsetfunctionts_(MatFDColoring *fd,void (*f)(TS*,double*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  (*fd)->ftn_func_pointer =  (void (*)(void)) f;
  (*fd)->ftn_func_cntx    = ctx;

  *ierr = MatFDColoringSetFunction(*fd,(PetscErrorCodeFunction)ourmatfdcoloringfunctionts,*fd);
}

PETSC_EXTERN void matfdcoloringsetfunction_(MatFDColoring *fd,void (*f)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  (*fd)->ftn_func_pointer = (void (*)(void)) f;
  (*fd)->ftn_func_cntx    = ctx;

  *ierr = MatFDColoringSetFunction(*fd,(PetscErrorCodeFunction)ourmatfdcoloringfunctionsnes,*fd);
}

PETSC_EXTERN void matfdcoloringview_(MatFDColoring *c,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatFDColoringView(*c,v);
}

PETSC_EXTERN void matfdcoloringsettype_(MatFDColoring *matfdcoloring,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MatFDColoringSetType(*matfdcoloring,t);if (*ierr) return;
  FREECHAR(type,t);
}
