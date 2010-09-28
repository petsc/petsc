#ifndef __PETSCFWK_H
#define __PETSCFWK_H

#include "petscsys.h"

extern PETSCSYS_DLLEXPORT PetscClassId PETSC_FWK_CLASSID;

/* 
   There is only one type implementing PetscFwk, 
   so all the code is in the interface and implements only one class PETSCFWK (below) 
   rather than using something like PETSCFWK_BASIC, etc.
*/
#define PETSCFWK "petscfwk" 


struct _p_PetscFwk;
typedef struct _p_PetscFwk *PetscFwk;

EXTERN PetscFwk PETSCSYS_DLLEXPORT PETSC_FWK_DEFAULT_(MPI_Comm);
#define PETSC_FWK_DEFAULT_SELF  PETSC_FWK_DEFAULT_(PETSC_COMM_SELF)
#define PETSC_FWK_DEFAULT_WORLD PETSC_FWK_DEFAULT_(PETSC_COMM_WORLD)



EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkInitializePackage(const char path[]);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkFinalizePackage(void);

/**/
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkCall(PetscFwk component,       const char *message);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkGetURL(PetscFwk component,     const char **outurl);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkSetURL(PetscFwk component,     const char *inurl);
/**/
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkCreate(MPI_Comm comm, PetscFwk *fwk);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkView(PetscFwk fwk, PetscViewer viewerASCII);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkRegisterComponent(PetscFwk fwk, const char key[]);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkRegisterDependence(PetscFwk fwk, const char server_key[], const char client_key[]);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkRegisterComponentURL(PetscFwk fwk, const char key[], const char url[]);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkGetComponent(PetscFwk fwk, const char key[], PetscFwk *component, PetscBool  *found);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkGetParent(PetscFwk fwk, PetscFwk *parent);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkVisit(PetscFwk fwk, const char *message);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscFwkDestroy(PetscFwk fwk);



/* 
   This library deals with components and frameworks.

PetscFwk can (i) act as a  "framework" (more or less as before), or
(ii) as a "component" (more about it below).

(i) As a "framework", PetscFwk is a bag of "components" (other
PetscFwk objects).
One can register components (PetscFwkRegisterComponent) and
dependencies between them (PetscRegisterDependence) through keys.
For each new key the framework creates a new PetscFwk component with
its PetscObject name equal to the key, and inserts a vertex into the
dependency graph.
For each dependency between two keys, it inserts a corresponding edge
into the dependency graph.
A framework can "visit" its components in the topological sort order
of the dependency graph, and "call" each component
with a string "<message>": PetscFwkVisit

(ii) As a "component", PetscFwk supports essentially one interface:
"call" with two arguments,
the component itself and a string "<message>".  This call is forwarded
to an implementing function in two different ways:
 (a) the component's vtable (to be defined below) is searched for a
message handler for <message>
     if the subroutine is found, its called with the component as the
sole argument.
 (b) if (a) fails to locate an appropriate message handler, and only
then,  the component's vtable is searched for a
      message handler for message "call"; if the subroutine is found,
it is called with the component and "<message>" as the
     two arguments;
 Otherwise an error occurs, since the component is unable to handle
the message.
The message handler acts on the component state, which is accessible
through the public PETSc API.
In particular, the message handler can extract PetscObjects attached
to the component via PetscObjectCompose.
This is a slow and somewhat cumbersome way of passing arguments to a
message handler, but it's one that can span
language boundaries (e.g., from C to Python).

vtable:
A component can implement message handling routines two different ways:
 (1) Message handlers can be composed with the component object via
PetscObjectComposeFunction
      A component with PetscObject name <name> handles message
<message> using function with the name
      "<name><Message>" (i.e., appending <message> to <name> and
capitalizing the first letter of the <message> string).
 (2) Message handlers can be found using a URL associated with a
component.  The URL is of the one of the two forms:
    (2.1) "[<path>/<lib>.a:]<name>" or  "[<path>/<lib>.so:]<name>",
in which case all message handler searches are done not
            among the composed functions, but among the dynamic
symbols in the lib.
            If the lib is absent, then the symbols are searched for
in the main executable, and have to be exported as dynamic,
            in order to be found.
    (2.2) "<path>/<module>.py:<name>", in which case message handlers
are supposed to be static methods in a Python class <name>,
            located in <module>, found at <path>. In this case
message handlers must have names matching <message>,
            if a <message>-specific handler is absent.  These
handlers are passed a petsc4py.Fwk object wrapping the component,
            and a Python string, encapsulating <message>, if necessary.

A URL can be associated with an PetscFwk using PetscFwkSetURL.
A URL can be reset using repeated calls to PetscFwkSetURL.
The URL can be retrieved using PetscFwkGetURL.

A component attached to a framework using a key can be extracted with
PetscFwkGetComponent,
and then its vtable can be manipulated either by composing functions
with it using PetscObjectComposeFunction,
or by setting the component's URL.
There is a shorthand way of associating a URL to a component being
attached to a framework: PetscFwkRegisterComponentURL.
       
*/
#endif
