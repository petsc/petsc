/*
   Routines to determine options set in the options database.
*/
#if !defined(__PETSCOPTIONS_H)
#define __PETSCOPTIONS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsHasName(const char[],const char[],PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsHasName,(const char b[],PetscTruth *f),(PETSC_NULL,b,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetInt(const char[],const char [],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetInt,(const char b[],PetscInt *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetLogical(const char[],const char [],PetscTruth *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetLogical,(const char b[],PetscTruth *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetReal(const char[],const char[],PetscReal *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetReal,(const char b[],PetscReal *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetScalar(const char[],const char[],PetscScalar *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetScalar,(const char b[],PetscScalar i[],PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetIntArray(const char[],const char[],PetscInt[],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetIntArray,(const char b[],PetscInt i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetRealArray(const char[],const char[],PetscReal[],PetscInt *,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetRealArray,(const char b[],PetscReal i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetString(const char[],const char[],char[],size_t,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetString,(const char b[],char i[],size_t s,PetscTruth *f),(PETSC_NULL,b,i,s,f))
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetStringArray(const char[],const char[],char*[],PetscInt*,PetscTruth*);
PetscPolymorphicSubroutine(PetscOptionsGetStringArray,(const char b[],char *i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsSetAlias(const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsSetValue(const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsClearValue(const char[]);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsAllUsed(int*);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsLeft(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsPrint(FILE *);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsCreate(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsInsert(int*,char ***,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsInsertFile(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsInsertString(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsDestroy(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsReject(const char[],const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetAll(char*[]);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsGetenv(MPI_Comm,const char[],char[],size_t,PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsAtoi(const char[],PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscOptionsAtod(const char[],PetscReal*);

extern PETSC_DLLEXPORT PetscTruth PetscOptionsPublish;
extern PETSC_DLLEXPORT int        PetscOptionsPublishCount;

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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

M*/
#define    PetscOptionsBegin(comm,prefix,mess,sec) 0; {\
             for (PetscOptionsPublishCount=(PetscOptionsPublish?-1:1); PetscOptionsPublishCount<2; PetscOptionsPublishCount++) {\
             int _5_ierr = PetscOptionsBegin_Private(comm,prefix,mess,sec);CHKERRQ(_5_ierr);

/*MC
    PetscOptionsEnd - Ends a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

    Collective on the MPI_Comm used in PetscOptionsBegin()

   Synopsis: PetscErrorCode PetscOptionsEnd(void)

  Level: intermediate

  Notes: Needs to be preceded by a call to PetscOptionsBegin()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
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

   Concepts: options database^subheading

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
M*/
#define    PetscOptionsTail() 0; {if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);}

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsInt(const char[],const char[],const char[],PetscInt,PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsReal(const char[],const char[],const char[],PetscReal,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsScalar(const char[],const char[],const char[],PetscScalar,PetscScalar*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsName(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsString(const char[],const char[],const char[],const char[],char*,size_t,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogical(const char[],const char[],const char[],PetscTruth,PetscTruth*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogicalGroupBegin(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogicalGroup(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogicalGroupEnd(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsList(const char[],const char[],const char[],PetscFList,const char[],char[],PetscInt,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsEList(const char[],const char[],const char[],const char*[],PetscInt,const char[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsRealArray(const char[],const char[],const char[],PetscReal[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsIntArray(const char[],const char[],const char[],PetscInt[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscOptionsStringArray(const char[],const char[],const char[],char*[],PetscInt*,PetscTruth*);

PETSC_EXTERN_CXX_END
#endif
