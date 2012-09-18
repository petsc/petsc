/*
   Routines to determine options set in the options database.
*/
#if !defined(__PETSCOPTIONS_H)
#define __PETSCOPTIONS_H
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscOptionsHasName(const char[],const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetInt(const char[],const char [],PetscInt *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetBool(const char[],const char [],PetscBool  *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetReal(const char[],const char[],PetscReal *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetScalar(const char[],const char[],PetscScalar *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetIntArray(const char[],const char[],PetscInt[],PetscInt *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetRealArray(const char[],const char[],PetscReal[],PetscInt *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetBoolArray(const char[],const char[],PetscBool [],PetscInt *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetString(const char[],const char[],char[],size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetStringArray(const char[],const char[],char*[],PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetEList(const char[],const char[],const char*const*,PetscInt,PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetEnum(const char[],const char[],const char*const*,PetscEnum*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsValidKey(const char[],PetscBool *);

PETSC_EXTERN PetscErrorCode PetscOptionsSetAlias(const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsSetValue(const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsClearValue(const char[]);

PETSC_EXTERN PetscErrorCode PetscOptionsAllUsed(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscOptionsUsed(const char *,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscOptionsLeft(void);
PETSC_EXTERN PetscErrorCode PetscOptionsView(PetscViewer);

PETSC_EXTERN PetscErrorCode PetscOptionsCreate(void);
PETSC_EXTERN PetscErrorCode PetscOptionsInsert(int*,char ***,const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsInsertFile(MPI_Comm,const char[],PetscBool );
#if defined(PETSC_HAVE_YAML)
PETSC_EXTERN PetscErrorCode PetscOptionsInsertFile_YAML(MPI_Comm,const char[],PetscBool);
#endif
PETSC_EXTERN PetscErrorCode PetscOptionsInsertString(const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsDestroy(void);
PETSC_EXTERN PetscErrorCode PetscOptionsClear(void);
PETSC_EXTERN PetscErrorCode PetscOptionsPrefixPush(const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsPrefixPop(void);

PETSC_EXTERN PetscErrorCode PetscOptionsReject(const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsGetAll(char*[]);

PETSC_EXTERN PetscErrorCode PetscOptionsGetenv(MPI_Comm,const char[],char[],size_t,PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToInt(const char[],PetscInt*);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToReal(const char[],PetscReal*);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToBool(const char[],PetscBool*);

PETSC_EXTERN PetscErrorCode PetscOptionsMonitorSet(PetscErrorCode (*)(const char[], const char[], void*), void *, PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode PetscOptionsMonitorCancel(void);
PETSC_EXTERN PetscErrorCode PetscOptionsMonitorDefault(const char[], const char[], void *);

PETSC_EXTERN PetscBool PetscOptionsPublish;
PETSC_EXTERN PetscInt PetscOptionsPublishCount;

/*MC
    PetscOptionsBegin - Begins a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

   Synopsis: PetscErrorCode PetscOptionsBegin(MPI_Comm comm,const char prefix[],const char title[],const char mansec[])

    Collective on MPI_Comm

  Input Parameters:
+   comm - communicator that shares GUI
.   prefix - options prefix for all options displayed on window
.   title - short descriptive text, for example "Krylov Solver Options"
-   mansec - section of manual pages for options, for example KSP

  Level: intermediate

  Notes: Needs to be ended by a call the PetscOptionsEnd()
         Can add subheadings with PetscOptionsHead()

  Developer notes: PetscOptionsPublish is set in PetscOptionsCheckInitial_Private() with -options_gui. When PetscOptionsPublish is set the
$             loop between PetscOptionsBegin() and PetscOptionsEnd() is run THREE times with PetscOptionsPublishCount of values -1,0,1 otherwise
$             the loop is run ONCE with a PetscOptionsPublishCount of 1.
$             = -1 : The PetscOptionsInt() etc just call the PetscOptionsGetInt() etc
$             = 0  : The GUI objects are created in PetscOptionsInt() etc and displayed in PetscOptionsEnd() and the options
$                    database updated updated with user changes; PetscOptionsGetInt() etc are also called
$             = 1 : The PetscOptionsInt() etc again call the PetscOptionsGetInt() etc (possibly getting new values), in addition the help message and
$                   default values are printed if -help was given.
$           When PetscOptionsObject.changedmethod is set this causes PetscOptionsPublishCount to be reset to -2 (so in the next loop iteration it is -1)
$           and the whole process is repeated. This is to handle when, for example, the KSPType is changed thus changing the list of
$           options available so they need to be redisplayed so the user can change the. Chaning PetscOptionsObjects.changedmethod is never
$           currently set.


.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscObjectOptionsBegin()

M*/
#define    PetscOptionsBegin(comm,prefix,mess,sec) 0; do {\
             for (PetscOptionsPublishCount=(PetscOptionsPublish?-1:1); PetscOptionsPublishCount<2; PetscOptionsPublishCount++) {\
             PetscErrorCode _5_ierr = PetscOptionsBegin_Private(comm,prefix,mess,sec);CHKERRQ(_5_ierr);

/*MC
    PetscObjectOptionsBegin - Begins a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

   Synopsis: PetscErrorCode PetscObjectOptionsBegin(PetscObject obj)

    Collective on PetscObject

  Input Parameters:
.   obj - object to set options for

  Level: intermediate

  Notes: Needs to be ended by a call the PetscOptionsEnd()
         Can add subheadings with PetscOptionsHead()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

M*/
#define PetscObjectOptionsBegin(obj) 0; do {                            \
  for (PetscOptionsPublishCount=(PetscOptionsPublish?-1:1); PetscOptionsPublishCount<2; PetscOptionsPublishCount++) { \
  PetscErrorCode _5_ierr = PetscObjectOptionsBegin_Private(obj);CHKERRQ(_5_ierr);

/*MC
    PetscOptionsEnd - Ends a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

    Collective on the MPI_Comm used in PetscOptionsBegin()

   Synopsis: PetscErrorCode PetscOptionsEnd(void)

  Level: intermediate

  Notes: Needs to be preceded by a call to PetscOptionsBegin()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscObjectOptionsBegin()

M*/
#define    PetscOptionsEnd() _5_ierr = PetscOptionsEnd_Private();CHKERRQ(_5_ierr);}} while (0)

PETSC_EXTERN PetscErrorCode PetscOptionsBegin_Private(MPI_Comm,const char[],const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectOptionsBegin_Private(PetscObject);
PETSC_EXTERN PetscErrorCode PetscOptionsEnd_Private(void);
PETSC_EXTERN PetscErrorCode PetscOptionsHead(const char[]);

/*MC
     PetscOptionsTail - Ends a section of options begun with PetscOptionsHead()
            See, for example, KSPSetFromOptions_GMRES().

   Collective on the communicator passed in PetscOptionsBegin()

   Synopsis: PetscErrorCode PetscOptionsTail(void)

  Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          Must be preceded by a call to PetscOptionsHead() in the same function.

          This needs to be used only if the code below PetscOptionsTail() can be run ONLY once.
      See, for example, PCSetFromOptions_Composite(). This is a return(0) in it for early exit
      from the function.

          This is only for use with the PETSc options GUI; which does not currently exist.

   Concepts: options database^subheading

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsEnum()
M*/
#define    PetscOptionsTail() 0; {if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);}

PETSC_EXTERN PetscErrorCode PetscOptionsEnum(const char[],const char[],const char[],const char *const*,PetscEnum,PetscEnum*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsInt(const char[],const char[],const char[],PetscInt,PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsReal(const char[],const char[],const char[],PetscReal,PetscReal*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsScalar(const char[],const char[],const char[],PetscScalar,PetscScalar*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsName(const char[],const char[],const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsString(const char[],const char[],const char[],const char[],char*,size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBool(const char[],const char[],const char[],PetscBool ,PetscBool *,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroupBegin(const char[],const char[],const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroup(const char[],const char[],const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroupEnd(const char[],const char[],const char[],PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsList(const char[],const char[],const char[],PetscFList,const char[],char[],size_t,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsEList(const char[],const char[],const char[],const char*const*,PetscInt,const char[],PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsRealArray(const char[],const char[],const char[],PetscReal[],PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsIntArray(const char[],const char[],const char[],PetscInt[],PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringArray(const char[],const char[],const char[],char*[],PetscInt*,PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolArray(const char[],const char[],const char[],PetscBool [],PetscInt*,PetscBool *);

PETSC_EXTERN PetscErrorCode PetscOptionsSetFromOptions(void);
PETSC_EXTERN PetscErrorCode PetscOptionsAMSDestroy(void);

/*
    See manual page for PetscOptionsBegin()
*/
typedef enum {OPTION_INT,OPTION_LOGICAL,OPTION_REAL,OPTION_LIST,OPTION_STRING,OPTION_REAL_ARRAY,OPTION_HEAD,OPTION_INT_ARRAY,OPTION_ELIST,OPTION_LOGICAL_ARRAY,OPTION_STRING_ARRAY} PetscOptionType;
typedef struct _n_PetscOptions* PetscOptions;
struct _n_PetscOptions {
  char              *option;
  char              *text;
  void              *data;         /* used to hold the default value and then any value it is changed to by GUI */
  PetscFList        flist;         /* used for available values for PetscOptionsList() */
  const char *const *list;        /* used for available values for PetscOptionsEList() */
  char              nlist;         /* number of entries in list */
  char              *man;
  size_t            arraylength;   /* number of entries in data in the case that it is an array (of PetscInt etc) */
  PetscBool         set;           /* the user has changed this value in the GUI */
  PetscOptionType   type;
  PetscOptions      next;
  char              *pman;
  void              *edata;
};

typedef struct {
  PetscOptions     next;
  char             *prefix,*pprefix;
  char             *title;
  MPI_Comm         comm;
  PetscBool        printhelp,changedmethod,alreadyprinted;
  PetscObject      object;
} PetscOptionsObjectType;
#endif
