/*$Id: zda.c,v 1.39 2001/01/15 21:49:49 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscmat.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dagetinterpolation_          DAGETINTERPOLATION
#define dacreate1d_                  DACREATE1D
#define dacreate3d_                  DACREATE3D
#define dacreate2d_                  DACREATE2D
#define dadestroy_                   DADESTROY
#define dacreateglobalvector_        DACREATEGLOBALVECTOR
#define dacreatelocalvector_         DACREATELOCALVECTOR
#define dagetscatter_                DAGETSCATTER
#define dagetglobalindices_          DAGETGLOBALINDICES
#define daview_                      DAVIEW
#define dagetinfo_                   DAGETINFO
#define dagetcoloring_               DAGETCOLORING
#define dagetislocaltoglobalmapping_ DAGETISLOCALTOGLOBALMAPPING
#define dagetislocaltoglobalmappingblck_ DAGETISLOCALTOGLOBALMAPPINGBLCK
#define daload_                      DALOAD
#define dasetfieldname_              DASETFIELDNAME
#define dagetfieldname_              DAGETFIELDNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetinterpolation_          dagetinterpolation
#define daload_                      daload
#define dacreateglobalvector_        dacreateglobalvector
#define dacreatelocalvector_         dacreatelocalvector
#define daview_                      daview
#define dacreate1d_                  dacreate1d
#define dacreate3d_                  dacreate3d
#define dacreate2d_                  dacreate2d
#define dadestroy_                   dadestroy
#define dagetscatter_                dagetscatter
#define dagetglobalindices_          dagetglobalindices
#define dagetinfo_                   dagetinfo
#define dagetcoloring_               dagetcoloring
#define dagetislocaltoglobalmapping_ dagetislocaltoglobalmapping
#define dagetislocaltoglobalmappingblck_ dagetislocaltoglobalmappingblck
#define dasetfieldname_              dasetfieldname
#define dagetfieldname_              dagetfieldname
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dagetinterpolation_(DA *dac,DA *daf,Mat *A,Vec *scale,int *ierr)
{
  if (FORTRANNULLOBJECT(scale)) scale = PETSC_NULL;
  *ierr = DAGetInterpolation(*dac,*daf,A,scale);
}

void PETSC_STDCALL dasetfieldname_(DA *da,int *nf,CHAR name PETSC_MIXED_LEN(len),
                                   int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = DASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}
void PETSC_STDCALL dagetfieldname(DA *da,int *nf,CHAR name PETSC_MIXED_LEN(len),
                                  int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = DAGetFieldName(*da,*nf,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
}

void PETSC_STDCALL daload_(PetscViewer *viewer,int *M,int *N,int *P,DA *da,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DALoad(v,*M,*N,*P,da);
}

void PETSC_STDCALL dagetislocaltoglobalmapping_(DA *da,ISLocalToGlobalMapping *map,int *ierr)
{
  *ierr = DAGetISLocalToGlobalMapping(*da,map);
}

void PETSC_STDCALL dagetislocaltoglobalmappingblck_(DA *da,ISLocalToGlobalMapping *map,int *ierr)
{
  *ierr = DAGetISLocalToGlobalMappingBlck(*da,map);
}

void PETSC_STDCALL dagetcoloring_(DA *da,ISColoring *coloring,Mat *J,int *ierr)
{
  if (FORTRANNULLOBJECT(coloring)) coloring = PETSC_NULL;
  if (FORTRANNULLOBJECT(J))        J        = PETSC_NULL;
  *ierr = DAGetColoring(*da,coloring,J);
}

void PETSC_STDCALL daview_(DA *da,PetscViewer *vin,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DAView(*da,v);
}

void PETSC_STDCALL dagetglobalindices_(DA *da,int *n,int *indices,long *ia,int *ierr)
{
  int *idx;
  *ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia   = PetscIntAddressToFortran(indices,idx);
}

void PETSC_STDCALL dacreateglobalvector_(DA *da,Vec* g,int *ierr)
{
  *ierr = DACreateGlobalVector(*da,g);
}

void PETSC_STDCALL dacreatelocalvector_(DA *da,Vec* l,int *ierr)
{
  *ierr = DACreateLocalVector(*da,l);
}

void PETSC_STDCALL dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,int *ierr)
{
  if (!FORTRANNULLINTEGER(ltog)) ltog = PETSC_NULL;
  if (!FORTRANNULLINTEGER(gtol)) gtol = PETSC_NULL;
  if (!FORTRANNULLINTEGER(ltol)) ltol = PETSC_NULL;
  *ierr = DAGetScatter(*da,ltog,gtol,ltol);
}

void PETSC_STDCALL dadestroy_(DA *da,int *ierr)
{
  *ierr = DADestroy(*da);
}

void PETSC_STDCALL dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,DA *inra,int *ierr)
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  *ierr = DACreate2d((MPI_Comm)PetscToPointerComm(*comm),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra,int *ierr)
{
  if (FORTRANNULLINTEGER(lc)) lc = PETSC_NULL;
  *ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void PETSC_STDCALL dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra,int *ierr)
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  if (FORTRANNULLINTEGER(lz)) lz = PETSC_NULL;
  *ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

void PETSC_STDCALL dagetinfo_(DA *da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap,DAStencilType *st,int *ierr)
{
  if (FORTRANNULLINTEGER(dim)) dim  = PETSC_NULL;
  if (FORTRANNULLINTEGER(M))   M    = PETSC_NULL;
  if (FORTRANNULLINTEGER(N))   N    = PETSC_NULL;
  if (FORTRANNULLINTEGER(P))   P    = PETSC_NULL;
  if (FORTRANNULLINTEGER(m))   m    = PETSC_NULL;
  if (FORTRANNULLINTEGER(n))   n    = PETSC_NULL;
  if (FORTRANNULLINTEGER(p))   p    = PETSC_NULL;
  if (FORTRANNULLINTEGER(w))   w    = PETSC_NULL;
  if (FORTRANNULLINTEGER(s))   s    = PETSC_NULL;
  if (FORTRANNULLINTEGER(wrap))wrap = PETSC_NULL;
  if (FORTRANNULLINTEGER(st))  st   = PETSC_NULL;
  *ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}

EXTERN_C_END


