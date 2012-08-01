#ifndef __PETSCSHELL_H
#define __PETSCSHELL_H

#include <petscsys.h>

PETSC_EXTERN PetscClassId PETSC_SHELL_CLASSID;

/* 
   There is only one type implementing PetscShell, 
   so all the code is in the interface and implements only one class PETSCSHELL (below) 
   rather than using something like PETSCSHELL_BASIC, etc.
*/
#define PETSCSHELL "petscshell" 


/*S
     PetscShell - a simple interpreter of string messages.
                  Responds to PetscShellCall(shell,message) by calling 
                  nameMessage(shell) or nameCall(shell,message),
                  where name is the shell's object name and functions nameMessage and nameCall
                  are attached to shell via PetscObjectComposeFunction() or found in a dynamic
                  library specified by the shell's url (see PetscShellSetURL()).

   Level: intermediate

   Note: PetscShellSetURL() allows for Python backends.  In this case name.message() or name()
         are called, where name is a Python class.

.seealso:  PetscShellSetURL(), PetscShellCall(), PetscShellRegisterComponent()
S*/
typedef struct _p_PetscShell *PetscShell;

PETSC_EXTERN PetscShell  PETSC_SHELL_DEFAULT_(MPI_Comm);
#define PETSC_SHELL_DEFAULT_SELF  PETSC_SHELL_DEFAULT_(PETSC_COMM_SELF)
#define PETSC_SHELL_DEFAULT_WORLD PETSC_SHELL_DEFAULT_(PETSC_COMM_WORLD)



PETSC_EXTERN PetscErrorCode PetscShellInitializePackage(const char[]);
PETSC_EXTERN PetscErrorCode PetscShellFinalizePackage(void);

/**/
PETSC_EXTERN PetscErrorCode PetscShellCreate(MPI_Comm, PetscShell*);
PETSC_EXTERN PetscErrorCode PetscShellSetURL(PetscShell,     const char*);
PETSC_EXTERN PetscErrorCode PetscShellCall(PetscShell,       const char *);
PETSC_EXTERN PetscErrorCode PetscShellGetURL(PetscShell,     const char **);
PETSC_EXTERN PetscErrorCode PetscShellView(PetscShell, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscShellDestroy(PetscShell*);
/**/
PETSC_EXTERN PetscErrorCode PetscShellRegisterComponentShell(PetscShell, const char[], PetscShell);
PETSC_EXTERN PetscErrorCode PetscShellRegisterComponentURL(PetscShell, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscShellRegisterDependence(PetscShell, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscShellGetComponent(PetscShell, const char[], PetscShell *, PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscShellVisit(PetscShell, const char *);
PETSC_EXTERN PetscErrorCode PetscShellGetVisitor(PetscShell, PetscShell *);




/* 
   This library deals with components and frameworks.

PetscShell can (i) act as a  "framework" (more or less as before), or
(ii) as a "component" (more about it below).

(i) As a "framework", PetscShell is a bag of "components" (other
PetscShell objects).
One can register components (PetscShellRegisterComponent) and
dependencies between them (PetscRegisterDependence) through keys.
For each new key the framework creates a new PetscShell component with
its PetscObject name equal to the key, and inserts a vertex into the
dependency graph.
For each dependency between two keys, it inserts a corresponding edge
into the dependency graph.
A framework can "visit" its components in the topological sort order
of the dependency graph, and "call" each component
with a string "<message>": PetscShellVisit

(ii) As a "component", PetscShell supports essentially one interface:
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
handlers are passed a petsc4py.Shell object wrapping the component,
            and a Python string, encapsulating <message>, if necessary.

A URL can be associated with an PetscShell using PetscShellSetURL.
A URL can be reset using repeated calls to PetscShellSetURL.
The URL can be retrieved using PetscShellGetURL.

A component attached to a framework using a key can be extracted with
PetscShellGetComponent,
and then its vtable can be manipulated either by composing functions
with it using PetscObjectComposeFunction,
or by setting the component's URL.
There is a shorthand way of associating a URL to a component being
attached to a framework: PetscShellyRegisterComponentURL.
       
*/
#endif
