#ifndef __PETSCFWK_H
#define __PETSCFWK_H

#include "petscsys.h"

extern PETSC_DLLEXPORT PetscClassId PETSC_FWK_CLASSID;

/* 
   There is only one type implementing PetscFwk class, 
   so all the code is in the interface and PETSCFWK (below) 
   is the implementing type name, rather than something like PETSCFWK_BASIC.
*/
#define PETSCFWK "petscfwk" 

struct _p_PetscFwk;
typedef struct _p_PetscFwk *PetscFwk;

EXTERN PetscFwk PETSC_DLLEXPORT PETSC_FWK_DEFAULT_(MPI_Comm);
#define PETSC_FWK_DEFAULT_SELF  PETSC_FWK_DEFAULT_(PETSC_COMM_SELF)
#define PETSC_FWK_DEFAULT_WORLD PETSC_FWK_DEFAULT_(PETSC_COMM_WORLD)


typedef PetscErrorCode (*PetscFwkConfigureComponentFunction)(PetscFwk fwk, const char *configuration, PetscObject *component);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkInitializePackage(const char path[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkFinalizePackage(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkCreate(MPI_Comm comm, PetscFwk *fwk);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkDestroy(PetscFwk fwk);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponent(PetscFwk fwk, const char key[], const char url[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterDependence(PetscFwk fwk, const char client_key[], const char server_key[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponent(PetscFwk fwk, const char key[], PetscObject *component, PetscTruth *found);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkGetURL(PetscFwk fwk, const char *key, const char **url, PetscTruth *found);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkConfigure(PetscFwk fwk, const char *configuration);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFwkView(PetscFwk fwk, PetscViewer viewerASCII);

/* 
   1) 'Create' a PetscFwk fwk.
   2) 'Register' some components by associating keys with urls.  
      URLs associated with a key can be overwritten with subsequent 'register' calls.
      Empty urls can be used in 'register' calls, so that dependencies on this key can be declared before its URL is known.
   3) 'Require' some dependencies by listing the dependent components' keys. 
   4) Run PetscFwkConfigure on fwk with the 'configuration' string encoding the state each component must be put in.
   
   During registration, for each newly encountered URL, which has the form [<path>/<lib>]:<name>, the following is done:
      a) <lib> is located along the <path>, and is loaded  
      b) the configuration subroutine Configure of type PetscFwkComponentConfigure, with the symbol '<name>Configure',
         is loaded from the library or from the current object, if <path>/<lib> is missing.  
      c) Configure is then run with 'fwk'=fwk, 'configuration'=PETSC_NULL, 'component'=component (return parameter), 
         to initialize the component.  component is expected to use fwk's comm for its own creation/initialization.
      d) More dependency requirements may be posted during the initial Configure call.

   During configuration Components are traversed in the topological order and the corresponding Configure routine is run 
      with 'fwk'=fwk, 'configuration'=configuration, 'component'=component.

*/
#endif
