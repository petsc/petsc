#include "private/fortranimpl.h"
#include "private/matimpl.h"
#include "petscts.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matfdcoloringsetfunctionts_      MATFDCOLORINGSETFUNCTIONTS
#define matfdcoloringsetfunction_        MATFDCOLORINGSETFUNCTION
#define matfdcoloringview_               MATFDCOLORINGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matfdcoloringsetfunctionts_      matfdcoloringsetfunctionts
#define matfdcoloringsetfunction_        matfdcoloringsetfunction
#define matfdcoloringview_               matfdcoloringview
#endif


/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmatfdcoloringfunctionts(TS ts,PetscReal t,Vec x,Vec y,MatFDColoring fd)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*))(fd->ftn_func_pointer)) (&ts,&t,&x,&y,fd->ftn_func_cntx,&ierr);
  return ierr;
}

static PetscErrorCode ourmatfdcoloringfunctionsnes(SNES snes,Vec x,Vec y,MatFDColoring fd)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(SNES*,Vec*,Vec*,void*,PetscErrorCode*))(fd->ftn_func_pointer)) (&snes,&x,&y,fd->ftn_func_cntx,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

/*
        MatFDColoringSetFunction sticks the Fortran function and its context into the MatFDColoring structure and passes the MatFDColoring object
    in as the function context. ourmafdcoloringfunctionsnes() and ourmatfdcoloringfunctionts()  then access the function and its context from the
    MatFDColoring that is passed in. This is the same way that fortran_func_pointers is used in PETSc objects.

   NOTE: FORTRAN USER CANNOT PUT IN A NEW J OR B currently.
*/


void PETSC_STDCALL matfdcoloringsetfunctionts_(MatFDColoring *fd,void (PETSC_STDCALL *f)(TS*,double*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  (*fd)->ftn_func_pointer = (void*) f;
  (*fd)->ftn_func_cntx = ctx;
  *ierr = MatFDColoringSetFunction(*fd,(PetscErrorCodeFunction)ourmatfdcoloringfunctionts,*fd);
}

void PETSC_STDCALL matfdcoloringsetfunction_(MatFDColoring *fd,void (PETSC_STDCALL *f)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  (*fd)->ftn_func_pointer = (void*) f;
  (*fd)->ftn_func_cntx = ctx;
  *ierr = MatFDColoringSetFunction(*fd,(PetscErrorCodeFunction)ourmatfdcoloringfunctionsnes,*fd);
}

void PETSC_STDCALL matfdcoloringview_(MatFDColoring *c,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatFDColoringView(*c,v);
}


EXTERN_C_END
