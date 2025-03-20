#pragma once

#include <petscbm.h>

struct _PetscBenchOps {
  PetscErrorCode (*setfromoptions)(PetscBench, PetscOptionItems);
  PetscErrorCode (*setup)(PetscBench);
  PetscErrorCode (*run)(PetscBench);
  PetscErrorCode (*view)(PetscBench, PetscViewer);
  PetscErrorCode (*reset)(PetscBench);
  PetscErrorCode (*destroy)(PetscBench);
};

struct _p_PetscBench {
  PETSCHEADER(struct _PetscBenchOps);
  PetscBool       setupcalled;
  PetscInt        size;
  PetscLogHandler lhdlr;
  void           *data;
};
