

#include "petscda.h"   /*I "petscda.h" I*/
#include "HYPRE_struct_mv.h"
#include "HYPRE_struct_ls.h"
#include "_hypre_struct_mv.h"

typedef struct {
  MPI_Comm            hcomm;
  DA                  da;
  HYPRE_StructGrid    hgrid;
  HYPRE_StructStencil hstencil;
  HYPRE_StructMatrix  hmat;
  HYPRE_StructVector  hb,hx;
  hypre_Box           hbox;

  PetscTruth          needsinitialization;

  /* variables that are stored here so they need not be reloaded for each MatSetValuesLocal() or MatZeroRowsLocal() call */
  PetscInt            *gindices,rstart,gnx,gnxgny,xs,ys,zs,nx,ny,nxny;
} Mat_HYPREStruct;
