/*
   Routines to determine options set in the options database.
*/
#if !defined(__PETSCOPTIONS_H)
#define __PETSCOPTIONS_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsHasName(const char[],const char[],PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsHasName,(const char b[],PetscTruth *f),(PETSC_NULL,b,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetInt(const char[],const char [],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetInt,(const char b[],PetscInt *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetTruth(const char[],const char [],PetscTruth *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetTruth,(const char b[],PetscTruth *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetReal(const char[],const char[],PetscReal *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetReal,(const char b[],PetscReal *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetScalar(const char[],const char[],PetscScalar *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetScalar,(const char b[],PetscScalar i[],PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetIntArray(const char[],const char[],PetscInt[],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetIntArray,(const char b[],PetscInt i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetRealArray(const char[],const char[],PetscReal[],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetRealArray,(const char b[],PetscReal i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetTruthArray(const char[],const char[],PetscTruth[],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetTruthArray,(const char b[],PetscTruth i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetString(const char[],const char[],char[],size_t,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetString,(const char b[],char i[],size_t s,PetscTruth *f),(PETSC_NULL,b,i,s,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetStringArray(const char[],const char[],char*[],PetscInt*,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetStringArray,(const char b[],char *i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetEList(const char[],const char[],const char*const*,PetscInt,PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetEnum(const char[],const char[],const char*const*,PetscEnum*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsValidKey(const char[],PetscTruth*);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsSetAlias(const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsSetValue(const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsClearValue(const char[]);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsAllUsed(int*);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsLeft(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsPrint(FILE *);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsCreate(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsInsert(int*,char ***,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsInsertFile(MPI_Comm,const char[],PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsInsertString(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsDestroy(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsClear(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsReject(const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetAll(char*[]);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetenv(MPI_Comm,const char[],char[],size_t,PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsAtoi(const char[],PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsAtod(const char[],PetscReal*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsMonitorSet(PetscErrorCode (*)(const char[], const char[], void*), void *, PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsMonitorCancel(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsMonitorDefault(const char[], const char[], void *);

extern PETSC_DLLEXPORT PetscTruth PetscOptionsPublish;
extern PETSC_DLLEXPORT PetscInt   PetscOptionsPublishCount;

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

M*/
#define    PetscOptionsBegin(comm,prefix,mess,sec) 0; {\
             for (PetscOptionsPublishCount=(PetscOptionsPublish?-1:1); PetscOptionsPublishCount<2; PetscOptionsPublishCount++) {\
             PetscErrorCode _5_ierr = PetscOptionsBegin_Private(comm,prefix,mess,sec);CHKERRQ(_5_ierr);

/*MC
    PetscOptionsEnd - Ends a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

    Collective on the MPI_Comm used in PetscOptionsBegin()

   Synopsis: PetscErrorCode PetscOptionsEnd(void)

  Level: intermediate

  Notes: Needs to be preceded by a call to PetscOptionsBegin()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

M*/
#define    PetscOptionsEnd() _5_ierr = PetscOptionsEnd_Private();CHKERRQ(_5_ierr);}}

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsBegin_Private(MPI_Comm,const char[],const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsEnd_Private(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsHead(const char[]);

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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsEnum()
M*/
#define    PetscOptionsTail() 0; {if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);}

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsEnum(const char[],const char[],const char[],const char *const*,PetscEnum,PetscEnum*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsInt(const char[],const char[],const char[],PetscInt,PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsReal(const char[],const char[],const char[],PetscReal,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsScalar(const char[],const char[],const char[],PetscScalar,PetscScalar*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsName(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsString(const char[],const char[],const char[],const char[],char*,size_t,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruth(const char[],const char[],const char[],PetscTruth,PetscTruth*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthGroupBegin(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthGroup(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthGroupEnd(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsList(const char[],const char[],const char[],PetscFList,const char[],char[],size_t,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsEList(const char[],const char[],const char[],const char*const*,PetscInt,const char[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsRealArray(const char[],const char[],const char[],PetscReal[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsIntArray(const char[],const char[],const char[],PetscInt[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsStringArray(const char[],const char[],const char[],char*[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthArray(const char[],const char[],const char[],PetscTruth[],PetscInt*,PetscTruth*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsSetFromOptions(void);
PETSC_EXTERN_CXX_END

/* 
    See manual page for PetscOptionsBegin() 
*/
typedef enum {OPTION_INT,OPTION_LOGICAL,OPTION_REAL,OPTION_LIST,OPTION_STRING,OPTION_REAL_ARRAY,OPTION_HEAD,OPTION_INT_ARRAY} PetscOptionType;
typedef struct _p_PetscOptions* PetscOptions;
struct _p_PetscOptions {
  char            *option;
  char            *text;
  void            *data;         /* used to hold the default value and then any value it is changed to by GUI */
  PetscFList      flist;         /* used for available values for PetscOptionsList() */
  char            *man;
  size_t          arraylength;   /* number of entries in data in the case that it is an array (of PetscInt etc) */
  PetscTruth      set;           /* the user has changed this value in the GUI */
  PetscOptionType type;
  PetscOptions    next;
};

typedef struct {
  PetscOptions     next;
  char             *prefix,*mprefix;  
  char             *title;
  MPI_Comm         comm;
  PetscTruth       printhelp,changedmethod,alreadyprinted;
} PetscOptionsObjectType;
#endif
