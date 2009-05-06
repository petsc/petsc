

#include "petscda.h"   /*I "petscda.h" I*/
#include "HYPRE_struct_mv.h"
#include "HYPRE_struct_ls.h"

typedef struct {
  MPI_Comm            hcomm;
  DA                  da;
  HYPRE_StructGrid    hgrid;
  HYPRE_StructStencil hstencil;
  HYPRE_StructMatrix  hmat;
  HYPRE_StructVector  hb,hx;

  PetscTruth          needsinitialization;
} Mat_HYPREStruct;
