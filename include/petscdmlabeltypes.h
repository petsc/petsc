#pragma once

/* SUBMANSEC = DM */

typedef const char *DMLabelType;
#define DMLABELCONCRETE  "concrete"
#define DMLABELEPHEMERAL "ephemeral"

/*S
  DMLabel - Object which encapsulates a subset of the mesh from a `DM`

  Level: developer

  Note:
  A label consists of a set of points on a `DM`

.seealso: [](ch_dmbase), `DM`, `DMPlexCreate()`, `DMLabelCreate()`, `DMLabelView()`, `DMLabelDestroy()`, `DMPlexCreateLabelField()`,
          `DMLabelGetDefaultValue()`, `DMLabelSetDefaultValue()`, `DMLabelDuplicate()`, `DMLabelGetValue()`, `DMLabelSetValue()`,
          `DMLabelAddStratum()`, `DMLabelAddStrata()`, `DMLabelInsertIS()`, `DMLabelGetNumValues()`, `DMLabelGetValueIS()`,
          `DMLabelGetStratumSize()`, `DMLabelComputeIndex()`, `DMLabelDestroyIndex()`, `DMLabelDistribute()`, `DMLabelConvertToSection()`
S*/
typedef struct _p_DMLabel *DMLabel;
