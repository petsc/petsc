#pragma once

/* SUBMANSEC = DM */

/*S
  DMAdaptor - An object that constructs a `DMLabel` or metric `Vec` that can be used to modify a `DM` based on error estimators or other criteria

  Level: developer

.seealso: [](ch_dmbase), `DM`, `DMAdaptorCreate()`, `DMAdaptorSetSolver()`, `DMAdaptorGetSolver()`, `DMAdaptorSetSequenceLength()`, `DMAdaptorGetSequenceLength()`, `DMAdaptorSetFromOptions()`,
          `DMAdaptorSetUp()`, `DMAdaptorAdapt()`, `DMAdaptorDestroy()`, `DMAdaptorGetTransferFunction()`, `PetscConvEstCreate()`, `PetscConvEstDestroy()`
S*/
typedef struct _p_DMAdaptor *DMAdaptor;
