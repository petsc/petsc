#if !defined(PETSCSECTION_H)
#define PETSCSECTION_H
#include <petscsys.h>
#include <petscis.h>
#include <petscsectiontypes.h>

PETSC_EXTERN PetscClassId PETSC_SECTION_CLASSID;

PETSC_EXTERN PetscErrorCode PetscSectionCreate(MPI_Comm,PetscSection*);
PETSC_EXTERN PetscErrorCode PetscSectionClone(PetscSection, PetscSection*);
PETSC_EXTERN PetscErrorCode PetscSectionSetFromOptions(PetscSection);
PETSC_EXTERN PetscErrorCode PetscSectionCopy(PetscSection, PetscSection);
PETSC_EXTERN PetscErrorCode PetscSectionCompare(PetscSection, PetscSection, PetscBool*);
PETSC_EXTERN PetscErrorCode PetscSectionGetNumFields(PetscSection, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSectionSetNumFields(PetscSection, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldName(PetscSection, PetscInt, const char *[]);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldName(PetscSection, PetscInt, const char []);
PETSC_EXTERN PetscErrorCode PetscSectionGetComponentName(PetscSection, PetscInt, PetscInt, const char *[]);
PETSC_EXTERN PetscErrorCode PetscSectionSetComponentName(PetscSection, PetscInt, PetscInt, const char []);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldComponents(PetscSection, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldComponents(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetChart(PetscSection, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSectionSetChart(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetPermutation(PetscSection, IS *);
PETSC_EXTERN PetscErrorCode PetscSectionSetPermutation(PetscSection, IS);
PETSC_EXTERN PetscErrorCode PetscSectionGetPointMajor(PetscSection, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSectionSetPointMajor(PetscSection, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSectionGetIncludesConstraints(PetscSection, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSectionSetIncludesConstraints(PetscSection, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSectionGetDof(PetscSection, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetDof(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionAddDof(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldDof(PetscSection, PetscInt, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldDof(PetscSection, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionAddFieldDof(PetscSection, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionHasConstraints(PetscSection, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSectionGetConstraintDof(PetscSection, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetConstraintDof(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionAddConstraintDof(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldConstraintDof(PetscSection, PetscInt, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldConstraintDof(PetscSection, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionAddFieldConstraintDof(PetscSection, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetConstraintIndices(PetscSection, PetscInt, const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSectionSetConstraintIndices(PetscSection, PetscInt, const PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldConstraintIndices(PetscSection, PetscInt, PetscInt, const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldConstraintIndices(PetscSection, PetscInt, PetscInt, const PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetUpBC(PetscSection);
PETSC_EXTERN PetscErrorCode PetscSectionSetUp(PetscSection);
PETSC_EXTERN PetscErrorCode PetscSectionGetMaxDof(PetscSection, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionGetStorageSize(PetscSection, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionGetConstrainedStorageSize(PetscSection, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionGetOffset(PetscSection, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetOffset(PetscSection, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldOffset(PetscSection, PetscInt, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldOffset(PetscSection, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldPointOffset(PetscSection, PetscInt, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSectionGetOffsetRange(PetscSection, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSectionView(PetscSection, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSectionViewFromOptions(PetscSection,PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscSectionReset(PetscSection);
PETSC_EXTERN PetscErrorCode PetscSectionDestroy(PetscSection*);
PETSC_EXTERN PetscErrorCode PetscSectionCreateGlobalSection(PetscSection, PetscSF, PetscBool, PetscBool, PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionCreateGlobalSectionCensored(PetscSection, PetscSF, PetscBool, PetscInt, const PetscInt [], PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionCreateSubsection(PetscSection, PetscInt, const PetscInt [], PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionCreateSupersection(PetscSection[], PetscInt, PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionCreateSubmeshSection(PetscSection, IS, PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionGetPointLayout(MPI_Comm, PetscSection, PetscLayout *);
PETSC_EXTERN PetscErrorCode PetscSectionGetValueLayout(MPI_Comm, PetscSection, PetscLayout *);
PETSC_EXTERN PetscErrorCode PetscSectionPermute(PetscSection, IS, PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionGetField(PetscSection, PetscInt, PetscSection *);
PETSC_EXTERN PetscErrorCode PetscSectionSetUseFieldOffsets(PetscSection, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSectionGetUseFieldOffsets(PetscSection, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSectionExtractDofsFromArray(PetscSection, MPI_Datatype, const void *, IS, PetscSection *, void *[]);

PETSC_EXTERN PetscErrorCode PetscSectionSetClosureIndex(PetscSection, PetscObject, PetscSection, IS);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosureIndex(PetscSection, PetscObject, PetscSection *, IS *);
PETSC_EXTERN PetscErrorCode PetscSectionSetClosurePermutation(PetscSection, PetscObject, PetscInt, IS);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosurePermutation(PetscSection, PetscObject, PetscInt, PetscInt, IS *);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosureInversePermutation(PetscSection, PetscObject, PetscInt, PetscInt, IS *);

PETSC_EXTERN PetscClassId PETSC_SECTION_SYM_CLASSID;

PETSC_EXTERN PetscFunctionList PetscSectionSymList;
PETSC_EXTERN PetscErrorCode PetscSectionSymSetType(PetscSectionSym, PetscSectionSymType);
PETSC_EXTERN PetscErrorCode PetscSectionSymGetType(PetscSectionSym, PetscSectionSymType*);
PETSC_EXTERN PetscErrorCode PetscSectionSymRegister(const char[],PetscErrorCode (*)(PetscSectionSym));

PETSC_EXTERN PetscErrorCode PetscSectionSymCreate(MPI_Comm, PetscSectionSym*);
PETSC_EXTERN PetscErrorCode PetscSectionSymDestroy(PetscSectionSym*);
PETSC_EXTERN PetscErrorCode PetscSectionSymView(PetscSectionSym,PetscViewer);

PETSC_EXTERN PetscErrorCode PetscSectionSetSym(PetscSection, PetscSectionSym);
PETSC_EXTERN PetscErrorCode PetscSectionGetSym(PetscSection, PetscSectionSym*);
PETSC_EXTERN PetscErrorCode PetscSectionSetFieldSym(PetscSection, PetscInt, PetscSectionSym);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldSym(PetscSection, PetscInt, PetscSectionSym*);

PETSC_EXTERN PetscErrorCode PetscSectionGetPointSyms(PetscSection, PetscInt, const PetscInt *, const PetscInt ***, const PetscScalar ***);
PETSC_EXTERN PetscErrorCode PetscSectionRestorePointSyms(PetscSection, PetscInt, const PetscInt *, const PetscInt ***, const PetscScalar ***);
PETSC_EXTERN PetscErrorCode PetscSectionGetFieldPointSyms(PetscSection, PetscInt, PetscInt, const PetscInt *, const PetscInt ***, const PetscScalar ***);
PETSC_EXTERN PetscErrorCode PetscSectionRestoreFieldPointSyms(PetscSection, PetscInt, PetscInt, const PetscInt *, const PetscInt ***, const PetscScalar ***);

#endif
