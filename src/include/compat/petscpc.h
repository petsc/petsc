#ifndef _COMPAT_PETSC_PC_H
#define _COMPAT_PETSC_PC_H

#include "private/pcimpl.h"

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PCLSC          "lsc"
#define PCPFMG         "pfmg"
#define PCSYSPFMG      "syspfmg"
#define PCREDISTRIBUTE "redistribute"
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PCEXOTIC       "exotic"
#define PCSUPPORTGRAPH "supportgraph"
#define PCASA	       "asa"
#define PCCP	       "cp"
#define PCBFBT         "bfbt"
#endif

#if (PETSC_VERSION_(2,3,2))
#define PCOPENMP       "openmp"
#endif

#endif /* _COMPAT_PETSC_PC_H */
