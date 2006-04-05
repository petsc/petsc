#include "zpetsc.h"
#include "petscmat.h"
#include "petscts.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matfdcoloringsetfunctionts_      MATFDCOLORINGSETFUNCTIONTS
#define matfdcoloringsetfunctionsnes_    MATFDCOLORINGSETFUNCTIONSNES
#define matfdcoloringview_               MATFDCOLORINGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matfdcoloringsetfunctionts_      matfdcoloringsetfunctionts
#define matfdcoloringsetfunctionsnes_    matfdcoloringsetfunctionsnes
#define matfdcoloringview_               matfdcoloringview
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f7)(TS*,double*,Vec*,Vec*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f8)(SNES*,Vec*,Vec*,void*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmatfdcoloringfunctionts(TS ts,double t,Vec x,Vec y,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*f7)(&ts,&t,&x,&y,ctx,&ierr);
  return ierr;
}

static PetscErrorCode ourmatfdcoloringfunctionsnes(SNES ts,Vec x,Vec y,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*f8)(&ts,&x,&y,ctx,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

/*
        MatFDColoringSetFunction sticks the Fortran function into the fortran_func_pointers
    this function is then accessed by ourmatfdcoloringfunction()

   NOTE: FORTRAN USER CANNOT PUT IN A NEW J OR B currently.

   USER CAN HAVE ONLY ONE MatFDColoring in code Because there is no place to hang f7!
*/


void PETSC_STDCALL matfdcoloringsetfunctionts_(MatFDColoring *fd,void (PETSC_STDCALL *f)(TS*,double*,Vec*,Vec*,void*,PetscErrorCode*),
                                 void *ctx,PetscErrorCode *ierr)
{
  f7 = f;
  *ierr = MatFDColoringSetFunction(*fd,(PetscErrorCodeFunction)ourmatfdcoloringfunctionts,ctx);
}

void PETSC_STDCALL matfdcoloringsetfunctionsnes_(MatFDColoring *fd,void (PETSC_STDCALL *f)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),
                                 void *ctx,PetscErrorCode *ierr)
{
  f8 = f;
  *ierr = MatFDColoringSetFunction(*fd,(PetscErrorCodeFunction)ourmatfdcoloringfunctionsnes,ctx);
}

void PETSC_STDCALL matfdcoloringview_(MatFDColoring *c,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatFDColoringView(*c,v);
}


EXTERN_C_END
