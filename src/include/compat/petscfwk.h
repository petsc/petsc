#ifndef _COMPAT_PETSC_FWK_H
#define _COMPAT_PETSC_FWK_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))

static PetscClassId PETSC_FWK_COOKIE = 0;

struct _p_PetscFwk;
typedef struct _p_PetscFwk *PetscFwk;

#define PetscFwk_ERR_SUP                                                    \
  PetscFunctionBegin;                                                       \
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version"); \
  PetscFunctionReturn(0);

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCreate"
PetscErrorCode PetscFwkCreate(MPI_Comm comm,PetscFwk *fwk){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkDestroy"
static PetscErrorCode PetscFwkDestroy(PetscFwk fwk){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkViewConfigurationOrder"
static PetscErrorCode PetscFwkViewConfigurationOrder(PetscFwk fwk,PetscViewer viewer){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponent"
static PetscErrorCode PetscFwkRegisterComponent(PetscFwk fwk,const char url[]){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterDependence"
static PetscErrorCode PetscFwkRegisterDependence(PetscFwk fwk,const char client_url[],const char server_url[]){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetComponent"
static PetscErrorCode PetscFwkGetComponent(PetscFwk fwk,const char url[],PetscObject *component,PetscTruth *found){PetscFwk_ERR_SUP}

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkConfigure"
static PetscErrorCode PetscFwkConfigure(PetscFwk fwk,PetscInt state){PetscFwk_ERR_SUP}

#endif

#endif /* _COMPAT_PETSC_FWK_H */
