/*$Id: zao.c,v 1.15 2000/04/30 22:25:45 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "ao.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define aocreatebasic_   AOCREATEBASIC
#define aocreatebasicis_ AOCREATEBASICIS
#define aoview           AOVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aocreatebasic_   aocreatebasic
#define aocreatebasicis_ aocreatebasicis
#define aoview           aoview_
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aoview_(AO *ao,Viewer *viewer, int *ierr)
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = AOView(*ao,v);
}

void PETSC_STDCALL aocreatebasic_(MPI_Comm *comm,int *napp,int *myapp,int *mypetsc,AO *aoout,int *ierr)
{
  *ierr = AOCreateBasic((MPI_Comm)PetscToPointerComm(*comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatebasicis_(IS *isapp,IS *ispetsc,AO *aoout,int *ierr)
{
  *ierr = AOCreateBasicIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
