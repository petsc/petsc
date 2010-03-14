#ifndef __PETSCFWK_H
#define __PETSCFWK_H

#include "petsc.h"

struct _p_PetscFwk;
typedef struct _p_PetscFwk *PetscFwk;

typedef PetscErrorCode (*PetscFwkComponentConfigure)(PetscFwk fwk, PetscInt state, PetscObject *component);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkInitializePackage(const char path[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkFinalizePackage(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkCreate(MPI_Comm comm, PetscFwk *fwk);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkDestroy(PetscFwk fwk);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponent(PetscFwk fwk, const char componenturl[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponentWithID(PetscFwk fwk, const char componenturl[], PetscInt *id);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterDependence(PetscFwk fwk, const char clienturl[], const char serverurl[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponent(PetscFwk fwk, const char url[], PetscObject *component);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponentByID(PetscFwk fwk, PetscInt id, PetscObject *component);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkConfigure(PetscFwk fwk, PetscInt state);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkViewConfigurationOrder(PetscFwk fwk, PetscViewer viewerASCII);

/* 
   1) 'Create' a PetscFwk fwk, which is created with fwk.state=0
   2) 'Require' some dependencies by listing the dependent components' URLs. For each newly encountered URL, 
      which has the form [<path>/<lib>]:<name>, the following is done:
      a) <lib> is located along the <path>, and is loaded  
      b) the configuration subroutine Configure of type PetscFwkComponentConfigure, with the symbol '<name>Configure',
         is loaded from the library or from the current object, if <path>/<lib> is missing.  
      c) Configure is then run with 'fwk'=fwk, 'state'=fwk.state (which is zero in this case),
         'component'=component (return parameter), to initialize the component.  
         component is expected to use fwk's comm for its own creation/initialization.
      d) More dependency requirements may be posted during each Configure 
   3) Run PetscFwkConfigure on fwk with 'state' equal to the number of cycles to be executed: 
      fwk is configured to be in state='state' by going through that many cycles. 
      a) Components are sorted topologically according to the dependency graph and the state of the fwk is set to 0.
      b) During each cycle the state of the framework is incremented, components are traversed in the topological order
         and the corresponding Configure routine is run with 'fwk'=fwk, 'state'=fwk.state, 'component'=component.
*/
#endif
