#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zda.c,v 1.15 1997/12/03 14:06:20 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "mat.h"
#include "da.h"

#ifdef HAVE_FORTRAN_CAPS
#define dacreate1d_             DACREATE1D
#define dacreate3d_             DACREATE3D
#define dacreate2d_             DACREATE2D
#define dadestroy_              DADESTROY
#define dagetdistributedvector_ DAGETDISTRIBUTEDVECTOR
#define dagetlocalvector_       DAGETLOCALVECTOR
#define dagetscatter_           DAGETSCATTER
#define dagetglobalindices_     DAGETGLOBALINDICES
#define daview_                 DAVIEW
#define dagetinfo_              DAGETINFO
#define dagetcoloring_          DAGETCOLORING
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define daview_                 daview
#define dacreate1d_             dacreate1d
#define dacreate3d_             dacreate3d
#define dacreate2d_             dacreate2d
#define dadestroy_              dadestroy
#define dagetdistributedvector_ dagetdistributedvector
#define dagetlocalvector_       dagetlocalvector
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
  *__ierr = DAGetColoring((DA)PetscToPointer(*(int*)(da)),&ocoloring,&oJ);
  *(int*) coloring = PetscFromPointer(ocoloring);
  *(int*) J = PetscFromPointer(oJ);
}

void daview_(DA da,Viewer v, int *__ierr )
{
  PetscPatchDefaultViewers_Fortran(v);
  *__ierr = DAView((DA)PetscToPointer( *(int*)(da) ),v);
}

void dagetglobalindices_(DA da,int *n, int *indices, int *ia,int *__ierr )
{
#if defined(PARCH_IRIX64)
  (*PetscErrorPrintf)("PETSC ERROR: Cannot use DAGetGlobalIndices() from Fortran under IRIX\n");
  (*PetscErrorPrintf)("PETSC ERROR: Refer to troubleshooting.html for more details\n");
  MPI_Abort(PETSC_COMM_WORLD,1);
#else
  int *idx;
  *__ierr = DAGetGlobalIndices((DA)PetscToPointer(*(int*)(da)),n,&idx);
  *ia     = PetscIntAddressToFortran(indices,idx);
#endif
}

void dagetdistributedvector_(DA da,Vec* g, int *__ierr )
{
  Vec v;
  *__ierr = DAGetDistributedVector((DA)PetscToPointer(*(int*)(da)),&v);
  *(int*) g = PetscFromPointer(v);
}
void dagetlocalvector_(DA da,Vec* l, int *__ierr )
{
  Vec v;
  *__ierr = DAGetLocalVector((DA)PetscToPointer(*(int*)(da)),&v);
  *(int*) l = PetscFromPointer(v);
}
void dagetscatter_(DA da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,
                   int *__ierr )
{
  VecScatter l,g,ll;
  *__ierr = DAGetScatter((DA)PetscToPointer(*(int*)(da)),&l,&g,&ll);
  if (!FORTRANNULL(ltog)) *(int*) ltog = PetscFromPointer(l);
  if (!FORTRANNULL(gtol)) *(int*) gtol = PetscFromPointer(g);
  if (!FORTRANNULL(ltol)) *(int*) ltol = PetscFromPointer(ll);
}

void dadestroy_(DA da, int *__ierr )
{
  *__ierr = DADestroy((DA)PetscToPointer( *(int*)(da) ));
  PetscRmPointer(*(int*)(da));
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
  *(int*) inra = PetscFromPointer(da);
}

void dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra, int *__ierr )
{
  DA da;
  if (FORTRANNULL(lc)) lc = PETSC_NULL;
  *__ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,&da);
  *(int*) inra = PetscFromPointer(da);
}

void dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra, int *__ierr )
{
  DA da;
  *__ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,&da);
  *(int*) inra = PetscFromPointer(da);
}

void dagetinfo_(DA da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap, int *__ierr )
{
  *__ierr = DAGetInfo((DA)PetscToPointer( *(int*)(da) ),dim,M,N,P,m,n,p,w,s,wrap);
}

#if defined(__cplusplus)
}
#endif
