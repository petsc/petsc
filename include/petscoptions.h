/*
   Routines to determine options set in the options database.
*/
#if !defined(__PETSCOPTIONS_H)
#define __PETSCOPTIONS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode  PetscOptionsHasName(const char[],const char[],PetscTruth*);
PetscPolymorphicFunction(PetscOptionsHasName,(const char b[],PetscTruth *f),(PETSC_NULL,b,f))
EXTERN PetscErrorCode  PetscOptionsGetInt(const char[],const char [],PetscInt *,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetInt,(const char b[],PetscInt *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode  PetscOptionsGetLogical(const char[],const char [],PetscTruth *,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetLogical,(const char b[],PetscTruth *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode  PetscOptionsGetReal(const char[],const char[],PetscReal *,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetReal,(const char b[],PetscReal *i,PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode  PetscOptionsGetScalar(const char[],const char[],PetscScalar *,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetScalar,(const char b[],PetscScalar i[],PetscTruth *f),(PETSC_NULL,b,i,f))
EXTERN PetscErrorCode  PetscOptionsGetIntArray(const char[],const char[],PetscInt[],PetscInt *,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetIntArray,(const char b[],PetscInt i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode  PetscOptionsGetRealArray(const char[],const char[],PetscReal[],PetscInt *,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetRealArray,(const char b[],PetscReal i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))
EXTERN PetscErrorCode  PetscOptionsGetString(const char[],const char[],char[],size_t,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetString,(const char b[],char i[],size_t s,PetscTruth *f),(PETSC_NULL,b,i,s,f))
EXTERN PetscErrorCode  PetscOptionsGetStringArray(const char[],const char[],char*[],PetscInt*,PetscTruth*);
PetscPolymorphicFunction(PetscOptionsGetStringArray,(const char b[],char *i[],PetscInt *ii,PetscTruth *f),(PETSC_NULL,b,i,ii,f))

EXTERN PetscErrorCode  PetscOptionsSetAlias(const char[],const char[]);
EXTERN PetscErrorCode  PetscOptionsSetValue(const char[],const char[]);
EXTERN PetscErrorCode  PetscOptionsClearValue(const char[]);

EXTERN PetscErrorCode  PetscOptionsAllUsed(int*);
EXTERN PetscErrorCode  PetscOptionsLeft(void);
EXTERN PetscErrorCode  PetscOptionsPrint(FILE *);

EXTERN PetscErrorCode  PetscOptionsCreate(void);
EXTERN PetscErrorCode  PetscOptionsInsert(int*,char ***,const char[]);
EXTERN PetscErrorCode  PetscOptionsInsertFile(const char[]);
EXTERN PetscErrorCode  PetscOptionsInsertString(const char[]);
EXTERN PetscErrorCode  PetscOptionsDestroy(void);

EXTERN PetscErrorCode  PetscOptionsReject(const char[],const char[]);
EXTERN PetscErrorCode  PetscOptionsGetAll(char*[]);

EXTERN PetscErrorCode  PetscOptionsGetenv(MPI_Comm,const char[],char[],size_t,PetscTruth *);
EXTERN PetscErrorCode  PetscOptionsAtoi(const char[],PetscInt*);
EXTERN PetscErrorCode  PetscOptionsAtod(const char[],PetscReal*);

extern PetscTruth PetscOptionsPublish;
extern int        PetscOptionsPublishCount;

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

EXTERN PetscErrorCode PetscOptionsBegin_Private(MPI_Comm,const char[],const char[],const char[]);
EXTERN PetscErrorCode PetscOptionsEnd_Private(void);
EXTERN PetscErrorCode PetscOptionsHead(const char[]);

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

EXTERN PetscErrorCode PetscOptionsInt(const char[],const char[],const char[],PetscInt,PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsReal(const char[],const char[],const char[],PetscReal,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsScalar(const char[],const char[],const char[],PetscScalar,PetscScalar*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsName(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PetscOptionsString(const char[],const char[],const char[],const char[],char*,size_t,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsLogical(const char[],const char[],const char[],PetscTruth,PetscTruth*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsLogicalGroupBegin(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PetscOptionsLogicalGroup(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PetscOptionsLogicalGroupEnd(const char[],const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PetscOptionsList(const char[],const char[],const char[],PetscFList,const char[],char[],PetscInt,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsEList(const char[],const char[],const char[],const char*[],PetscInt,const char[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsRealArray(const char[],const char[],const char[],PetscReal[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsIntArray(const char[],const char[],const char[],PetscInt[],PetscInt*,PetscTruth*);
EXTERN PetscErrorCode PetscOptionsStringArray(const char[],const char[],const char[],char*[],PetscInt*,PetscTruth*);

PETSC_EXTERN_CXX_END
#endif
