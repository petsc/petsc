#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zda.c,v 1.22 1998/08/05 17:28:00 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "mat.h"
#include "da.h"

#ifdef HAVE_FORTRAN_CAPS
#define dacreate1d_             DACREATE1D
#define dacreate3d_             DACREATE3D
#define dacreate2d_             DACREATE2D
#define dadestroy_              DADESTROY
#define dacreateglobalvector_   DACREATEGLOBALVECTOR
#define dacreatelocalvector_    DACREATELOCALVECTOR
#define dagetscatter_           DAGETSCATTER
#define dagetglobalindices_     DAGETGLOBALINDICES
#define daview_                 DAVIEW
#define dagetinfo_              DAGETINFO
#define dagetcoloring_          DAGETCOLORING
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define dacreateglobalvector_   dacreateglobalvector
#define dacreatelocalvector_    dacreatelocalvector
#define daview_                 daview
#define dacreate1d_             dacreate1d
#define dacreate3d_             dacreate3d
#define dacreate2d_             dacreate2d
#define dadestroy_              dadestroy
#define dagetscatter_           dagetscatter
#define dagetglobalindices_     dagetglobalindices
#define dagetinfo_              dagetinfo
#define dagetcoloring_          dagetcoloring
#endif

#if defined(__cplusplus)
extern "C" {
#endif
void dagetcoloring_(DA *da, ISColoring *coloring, Mat *J,int *__ierr)
{
  *__ierr = DAGetColoring(*da,coloring,J);
}

void daview_(DA *da,Viewer v, int *__ierr )
{
  PetscPatchDefaultViewers_Fortran(v);
  *__ierr = DAView(*da,v);
}

void dagetglobalindices_(DA *da,int *n, int *indices, long *ia,int *__ierr )
{
  int *idx;
  *__ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia     = PetscIntAddressToFortran(indices,idx);
}

void dacreateglobalvector_(DA *da,Vec* g, int *__ierr )
{
  *__ierr = DACreateGlobalVector(*da,g);
}

void dacreatelocalvector_(DA *da,Vec* l, int *__ierr )
{
  *__ierr = DACreateLocalVector(*da,l);
}

void dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,
                   int *__ierr )
{
  VecScatter l,g,ll;
  *__ierr = DAGetScatter(*da,&l,&g,&ll);
  if (!FORTRANNULLINTEGER(ltog)) *(PetscFortranAddr*) ltog = PetscFromPointer(l);
  if (!FORTRANNULLINTEGER(gtol)) *(PetscFortranAddr*) gtol = PetscFromPointer(g);
  if (!FORTRANNULLINTEGER(ltol)) *(PetscFortranAddr*) ltol = PetscFromPointer(ll);
}

void dadestroy_(DA *da, int *__ierr )
{
  *__ierr = DADestroy(*da);
}

void dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  *__ierr = DACreate2d((MPI_Comm)PetscToPointerComm( *comm ),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lc)) lc = PETSC_NULL;
  *__ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  if (FORTRANNULLINTEGER(lz)) lz = PETSC_NULL;
  *__ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

void dagetinfo_(DA *da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap, int *__ierr )
{
  *__ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap);
}

#if defined(__cplusplus)
}
#endif

