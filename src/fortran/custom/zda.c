
#include "src/dm/da/daimpl.h"
#include "src/fortran/custom/zpetsc.h"
#include "petscmat.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dasetblockfills_             DASETBLOCKFILLS
#define dasetlocalfunction_          DASETLOCALFUNCTION
#define dasetLocaladiforfunction_    DASETLOCALADIFORFUNCTION
#define dasetlocaladiformffunction_  DASETLOCALADIFORMFFUNCTION
#define dasetlocaljacobian_          DASETLOCALJACOBIAN
#define dagetlocalinfo_              DAGETLOCALINFO
#define dagetinterpolation_          DAGETINTERPOLATION
#define dacreate1d_                  DACREATE1D
#define dacreate3d_                  DACREATE3D
#define dacreate2d_                  DACREATE2D
#define dadestroy_                   DADESTROY
#define dacreateglobalvector_        DACREATEGLOBALVECTOR
#define dacreatenaturalvector_       DACREATENATURALVECTOR
#define dacreatelocalvector_         DACREATELOCALVECTOR
#define dagetlocalvector_            DAGETLOCALVECTOR
#define dagetglobalvector_           DAGETGLOBALVECTOR
#define darestorelocalvector_        DARESTORELOCALVECTOR
#define darestoreglobalvector_       DARESTOREGLOBALVECTOR
#define dagetscatter_                DAGETSCATTER
#define dagetglobalindices_          DAGETGLOBALINDICES
#define daview_                      DAVIEW
#define dagetinfo_                   DAGETINFO
#define dagetcoloring_               DAGETCOLORING
#define dagetmatrix_                 DAGETMATRIX
#define dagetislocaltoglobalmapping_ DAGETISLOCALTOGLOBALMAPPING
#define dagetislocaltoglobalmappingblck_ DAGETISLOCALTOGLOBALMAPPINGBLCK
#define daload_                      DALOAD
#define dasetfieldname_              DASETFIELDNAME
#define dagetfieldname_              DAGETFIELDNAME
#define darefine_                    DAREFINE
#define dagetao_                     DAGETAO
#define darestoreglobalvector_       DARESTOREGLOBALVECTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define darestoreglobalvector_       darestoreglobalvector
#define dasetblockfills_             dasetblockfills
#define dagetlocalinfo_              dagetlocalinfo
#define dagetlocalvector_            dagetlocalvector
#define dagetglobalvector_           dagetglobalvector
#define darestorelocalvector_        darestorelocalvector
#define darestoreglobalvector_       darestoreglobalvector
#define dagetinterpolation_          dagetinterpolation
#define daload_                      daload
#define dacreateglobalvector_        dacreateglobalvector
#define dacreatenaturalvector_       dacreatenaturalvector
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
#define dagetmatrix_                 dagetmatrix
#define dagetislocaltoglobalmapping_ dagetislocaltoglobalmapping
#define dagetislocaltoglobalmappingblck_ dagetislocaltoglobalmappingblck
#define dasetfieldname_              dasetfieldname
#define dagetfieldname_              dagetfieldname
#define darefine_                    darefine
#define dagetao_                     dagetao
#define dasetlocalfunction_          dasetlocalfunction
#define dasetlocaladiforfunction_       dasetlocaladiforfunction
#define dasetlocaladiformffunction_       dasetlocaladiformffunction
#define dasetlocaljacobian_          dasetlocaljacobian
#endif



EXTERN_C_BEGIN

void PETSC_STDCALL dasetblockfills_(DA *da,PetscInt *dfill,PetscInt *ofill,PetscErrorCode *ierr)
{
  *ierr = DASetBlockFills(*da,dfill,ofill);
}

static void (PETSC_STDCALL *j1d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlj1d(DALocalInfo *info,PetscScalar *in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*j1d)(info,&in[info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *j2d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlj2d(DALocalInfo *info,PetscScalar **in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*j2d)(info,&in[info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *j3d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlj3d(DALocalInfo *info,PetscScalar ***in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*j3d)(info,&in[info->gzs][info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f1d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlf1d(DALocalInfo *info,PetscScalar *in,PetscScalar *out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*f1d)(info,&in[info->dof*info->gxs],&out[info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f2d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlf2d(DALocalInfo *info,PetscScalar **in,PetscScalar **out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*f2d)(info,&in[info->gys][info->dof*info->gxs],&out[info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f3d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlf3d(DALocalInfo *info,PetscScalar ***in,PetscScalar ***out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*f3d)(info,&in[info->gzs][info->gys][info->dof*info->gxs],&out[info->zs][info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dasetlocalfunction_(DA *da,void (PETSC_STDCALL *func)(DALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
     f2d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf2d);
  } else if (dim == 3) {
     f3d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf3d);
  } else if (dim == 1) {
     f1d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf1d);
  } else *ierr = 1;
}


void PETSC_STDCALL dasetlocaladiforfunction_(DA *da,
void (PETSC_STDCALL *jfunc)(PetscInt*,DALocalInfo*,void*,void*,PetscInt*,void*,void*,PetscInt*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  (*da)->adifor_lf = (DALocalFunction1)jfunc;
}

void PETSC_STDCALL dasetlocaladiformffunction_(DA *da,
void (PETSC_STDCALL *jfunc)(DALocalInfo*,void*,void*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  (*da)->adiformf_lf = (DALocalFunction1)jfunc;
}

void PETSC_STDCALL dasetlocaljacobian_(DA *da,void (PETSC_STDCALL *jac)(DALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
     j2d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))jac; 
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj2d);
  } else if (dim == 3) {
     j3d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))jac;
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj3d);
  } else if (dim == 1) {
     j1d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))jac; 
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj1d);
  } else *ierr = 1;
}

void PETSC_STDCALL dagetlocalinfo_(DA *da,DALocalInfo *ao,PetscErrorCode *ierr)
{
  *ierr = DAGetLocalInfo(*da,ao);
}

void PETSC_STDCALL dagetao_(DA *da,AO *ao,PetscErrorCode *ierr)
{
  *ierr = DAGetAO(*da,ao);
}

void PETSC_STDCALL darefine_(DA *da,MPI_Comm *comm,DA *daref, PetscErrorCode *ierr)
{
  *ierr = DARefine(*da,(MPI_Comm)PetscToPointerComm(*comm),daref);
}

void PETSC_STDCALL dagetinterpolation_(DA *dac,DA *daf,Mat *A,Vec *scale,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(scale);
  *ierr = DAGetInterpolation(*dac,*daf,A,scale);
}

void PETSC_STDCALL dasetfieldname_(DA *da,PetscInt *nf,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = DASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}
void PETSC_STDCALL dagetfieldname(DA *da,PetscInt *nf,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
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

void PETSC_STDCALL daload_(PetscViewer *viewer,PetscInt *M,PetscInt *N,PetscInt *P,DA *da,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DALoad(v,*M,*N,*P,da);
}

void PETSC_STDCALL dagetislocaltoglobalmapping_(DA *da,ISLocalToGlobalMapping *map,PetscErrorCode *ierr)
{
  *ierr = DAGetISLocalToGlobalMapping(*da,map);
}

void PETSC_STDCALL dagetislocaltoglobalmappingblck_(DA *da,ISLocalToGlobalMapping *map,PetscErrorCode *ierr)
{
  *ierr = DAGetISLocalToGlobalMappingBlck(*da,map);
}

void PETSC_STDCALL dagetcoloring_(DA *da,ISColoringType *ctype,ISColoring *coloring,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  *ierr = DAGetColoring(*da,*ctype,coloring);
}

void PETSC_STDCALL dagetmatrix_(DA *da,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DAGetMatrix(*da,t,J);
  FREECHAR(mat_type,t);
}

void PETSC_STDCALL daview_(DA *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DAView(*da,v);
}

void PETSC_STDCALL dagetglobalindices_(DA *da,PetscInt *n,PetscInt *indices,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt *idx;
  *ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia   = PetscIntAddressToFortran(indices,idx);
}

void PETSC_STDCALL dacreateglobalvector_(DA *da,Vec* g,PetscErrorCode *ierr)
{
  *ierr = DACreateGlobalVector(*da,g);
}

void PETSC_STDCALL dacreatenaturalvector_(DA *da,Vec* g,PetscErrorCode *ierr)
{
  *ierr = DACreateNaturalVector(*da,g);
}

void PETSC_STDCALL dacreatelocalvector_(DA *da,Vec* l,PetscErrorCode *ierr)
{
  *ierr = DACreateLocalVector(*da,l);
}

void PETSC_STDCALL dagetlocalvector_(DA *da,Vec* l,PetscErrorCode *ierr)
{
  *ierr = DAGetLocalVector(*da,l);
}

void PETSC_STDCALL dagetglobalvector_(DA *da,Vec* l,PetscErrorCode *ierr)
{
  *ierr = DAGetGlobalVector(*da,l);
}

void PETSC_STDCALL darestoreglobalvector_(DA *da,Vec* l,PetscErrorCode *ierr)
{
  *ierr = DARestoreGlobalVector(*da,l);
}

void PETSC_STDCALL darestorelocalvector_(DA *da,Vec* l,PetscErrorCode *ierr)
{
  *ierr = DARestoreLocalVector(*da,l);
}

void PETSC_STDCALL darestoreglobalvector_(DA *da,Vec* g,PetscErrorCode *ierr)
{
  *ierr = DARestoreGlobalVector(*da,g);
}

void PETSC_STDCALL dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ltog);
  CHKFORTRANNULLOBJECT(gtol);
  CHKFORTRANNULLOBJECT(ltol);
  *ierr = DAGetScatter(*da,ltog,gtol,ltol);
}

void PETSC_STDCALL dadestroy_(DA *da,PetscErrorCode *ierr)
{
  *ierr = DADestroy(*da);
}

void PETSC_STDCALL dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,PetscInt *M,PetscInt *N,PetscInt *m,PetscInt *n,PetscInt *w,
                  PetscInt *s,PetscInt *lx,PetscInt *ly,DA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = DACreate2d((MPI_Comm)PetscToPointerComm(*comm),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,PetscInt *M,PetscInt *w,PetscInt *s,
                 PetscInt *lc,DA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lc);
  *ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void PETSC_STDCALL dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,
                 PetscInt *w,PetscInt *s,PetscInt *lx,PetscInt *ly,PetscInt *lz,DA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

void PETSC_STDCALL dagetinfo_(DA *da,PetscInt *dim,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,PetscInt *w,PetscInt *s,
                DAPeriodicType *wrap,DAStencilType *st,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(dim);
  CHKFORTRANNULLINTEGER(M);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLINTEGER(P);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  CHKFORTRANNULLINTEGER(w);
  CHKFORTRANNULLINTEGER(s);
  CHKFORTRANNULLINTEGER(wrap);
  CHKFORTRANNULLINTEGER(st);
  *ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}

EXTERN_C_END


