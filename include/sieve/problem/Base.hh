#ifndef included_ALE_Problem_Base_hh
#define included_ALE_Problem_Base_hh

#include <DMBuilder.hh>

#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

namespace ALE {
  namespace Problem {
    typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
    typedef enum {NEUMANN, DIRICHLET} BCType;
    typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
    typedef union {SectionReal section; Vec vec;} ExactSolType;
  }
}

#endif
