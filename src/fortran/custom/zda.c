#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zda.c,v 1.19 1998/03/20 22:42:42 bsmith Exp balay $";
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
void dagetcoloring_(DA da, ISColoring *coloring, Mat *J,int *__ierr)
{
  ISColoring ocoloring;
  Mat        oJ;
  *__ierr = DAGetColoring((DA)PetscToPointer(da),&ocoloring,&oJ);
  *(PetscFortranAddr*) coloring = PetscFromPointer(ocoloring);
  *(PetscFortranAddr*) J = PetscFromPointer(oJ);
}

void daview_(DA da,Viewer v, int *__ierr )
{
  PetscPatchDefaultViewers_Fortran(v);
  *__ierr = DAView((DA)PetscToPointer(da),v);
}

void dagetglobalindices_(DA da,int *n, int *indices, long *ia,int *__ierr )
{
  int *idx;
  *__ierr = DAGetGlobalIndices((DA)PetscToPointer(da),n,&idx);
  *ia     = PetscIntAddressToFortran(indices,idx);
}

void dacreateglobalvector_(DA da,Vec* g, int *__ierr )
{
  Vec v;
  *__ierr = DACreateGlobalVector((DA)PetscToPointer(da),&v);
  *(PetscFortranAddr*) g = PetscFromPointer(v);
}

void dacreatelocalvector_(DA da,Vec* l, int *__ierr )
{
  Vec v;
  *__ierr = DACreateLocalVector((DA)PetscToPointer(da),&v);
  *(PetscFortranAddr*) l = PetscFromPointer(v);
}

void dagetscatter_(DA da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,
                   int *__ierr )
{
  VecScatter l,g,ll;
  *__ierr = DAGetScatter((DA)PetscToPointer(da),&l,&g,&ll);
  if (!FORTRANNULL(ltog)) *(PetscFortranAddr*) ltog = PetscFromPointer(l);
  if (!FORTRANNULL(gtol)) *(PetscFortranAddr*) gtol = PetscFromPointer(g);
  if (!FORTRANNULL(ltol)) *(PetscFortranAddr*) ltol = PetscFromPointer(ll);
}

void dadestroy_(DA da, int *__ierr )
{
  *__ierr = DADestroy((DA)PetscToPointer(da));
  PetscRmPointer(da);
}

void dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,DA *inra, int *__ierr )
{
  DA da;
  if (FORTRANNULL(lx)) lx = PETSC_NULL;
  if (FORTRANNULL(ly)) ly = PETSC_NULL;
  *__ierr = DACreate2d((MPI_Comm)PetscToPointerComm( *comm ),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,&da);
  *(PetscFortranAddr*) inra = PetscFromPointer(da);
}

void dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra, int *__ierr )
{
  DA da;
  if (FORTRANNULL(lc)) lc = PETSC_NULL;
  *__ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,&da);
  *(PetscFortranAddr*) inra = PetscFromPointer(da);
}

void dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra, int *__ierr )
{
  DA da;
  if (FORTRANNULL(lx)) lx = PETSC_NULL;
  if (FORTRANNULL(ly)) ly = PETSC_NULL;
  if (FORTRANNULL(lz)) lz = PETSC_NULL;
  *__ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,&da);
  *(PetscFortranAddr*) inra = PetscFromPointer(da);
}

void dagetinfo_(DA da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap, int *__ierr )
{
  *__ierr = DAGetInfo((DA)PetscToPointer(da),dim,M,N,P,m,n,p,w,s,wrap);
}

#if defined(__cplusplus)
}
#endif
