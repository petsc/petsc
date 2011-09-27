#ifndef _COMPAT_PETSC_FWK_H
#define _COMPAT_PETSC_FWK_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))

static PetscClassId PETSC_FWK_COOKIE = 0;

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkInitializePackage"
static PetscErrorCode PetscFwkInitializePackage(const char path[])
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode ierr;
  if (initialized) return 0;
  initialized = PETSC_TRUE;
  PetscFunctionBegin;
  ierr = PetscCookieRegister("PetscFwk",&PETSC_FWK_COOKIE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct _p_PetscFwk;
typedef struct _p_PetscFwk *PetscFwk;

#define PetscFwk_ERR_SUP                                                    \
  PetscFunctionBegin;                                                       \
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version"); \
  PetscFunctionReturn(PETSC_ERR_SUP);

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCreate"
static PetscErrorCode PetscFwkCreate(MPI_Comm comm,PetscFwk *fwk){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkDestroy"
static PetscErrorCode PetscFwkDestroy(PetscFwk fwk){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetURL"
static PetscErrorCode PetscFwkGetURL(PetscFwk fwk, const char**_url) {PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkSetURL"
static PetscErrorCode PetscFwkSetURL(PetscFwk fwk, const char*_url) {PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCall"
static PetscErrorCode PetscFwkCall(PetscFwk fwk, const char message[]){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkView"
static PetscErrorCode PetscFwkView(PetscFwk fwk,PetscViewer viewer){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponent"
static PetscErrorCode PetscFwkRegisterComponent(PetscFwk fwk,const char key[]){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponentURL"
static PetscErrorCode PetscFwkRegisterComponentURL(PetscFwk fwk,const char key[],const char url[]){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterDependence"
static PetscErrorCode PetscFwkRegisterDependence(PetscFwk fwk,const char client_key[],const char server_key[]){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetComponent"
static PetscErrorCode PetscFwkGetComponent(PetscFwk fwk,const char key[],PetscFwk *component,PetscTruth *found){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetParent"
static PetscErrorCode PetscFwkGetParent(PetscFwk fwk, PetscFwk *parent){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkVisit"
static PetscErrorCode PetscFwkVisit(PetscFwk fwk, const char *message){PetscFwk_ERR_SUP}

static PetscFwk PETSC_FWK_DEFAULT_(MPI_Comm comm) {return PETSC_NULL;}

#endif

#endif /* _COMPAT_PETSC_FWK_H */
