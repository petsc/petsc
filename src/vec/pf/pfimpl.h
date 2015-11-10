
#if !defined(_PFIMPL)
#define _PFIMPL

#include <petscpf.h>
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_EXTERN PetscBool      PFRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PFRegisterAll(void);

typedef struct _PFOps *PFOps;
struct _PFOps {
  PetscErrorCode (*apply)(void*,PetscInt,const PetscScalar*,PetscScalar*);
  PetscErrorCode (*applyvec)(void*,Vec,Vec);
  PetscErrorCode (*destroy)(void*);
  PetscErrorCode (*view)(void*,PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PF);
};

struct _p_PF {
  PETSCHEADER(struct _PFOps);
  PetscInt dimin,dimout;             /* dimension of input and output spaces */
  void     *data;
};

#endif
