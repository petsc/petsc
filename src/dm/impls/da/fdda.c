
#include <petsc/private/dmdaimpl.h> /*I      "petscdmda.h"     I*/
#include <petscmat.h>

extern PetscErrorCode DMCreateColoring_DA_1d_MPIAIJ(DM,ISColoringType,ISColoring*);
extern PetscErrorCode DMCreateColoring_DA_2d_MPIAIJ(DM,ISColoringType,ISColoring*);
extern PetscErrorCode DMCreateColoring_DA_2d_5pt_MPIAIJ(DM,ISColoringType,ISColoring*);
extern PetscErrorCode DMCreateColoring_DA_3d_MPIAIJ(DM,ISColoringType,ISColoring*);

/*
   For ghost i that may be negative or greater than the upper bound this
  maps it into the 0:m-1 range using periodicity
*/
#define SetInRange(i,m) ((i < 0) ? m+i : ((i >= m) ? i-m : i))

static PetscErrorCode DMDASetBlockFills_Private(const PetscInt *dfill,PetscInt w,PetscInt **rfill)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nz,*fill;

  PetscFunctionBegin;
  if (!dfill) PetscFunctionReturn(0);

  /* count number nonzeros */
  nz = 0;
  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      if (dfill[w*i+j]) nz++;
    }
  }
  ierr = PetscMalloc1(nz + w + 1,&fill);CHKERRQ(ierr);
  /* construct modified CSR storage of nonzero structure */
  /*  fill[0 -- w] marks starts of each row of column indices (and end of last row)
   so fill[1] - fill[0] gives number of nonzeros in first row etc */
  nz = w + 1;
  for (i=0; i<w; i++) {
    fill[i] = nz;
    for (j=0; j<w; j++) {
      if (dfill[w*i+j]) {
        fill[nz] = j;
        nz++;
      }
    }
  }
  fill[w] = nz;

  *rfill = fill;
  PetscFunctionReturn(0);
}


static PetscErrorCode DMDASetBlockFillsSparse_Private(const PetscInt *dfillsparse,PetscInt w,PetscInt **rfill)
{
  PetscErrorCode ierr;
  PetscInt       nz;

  PetscFunctionBegin;
  if (!dfillsparse) PetscFunctionReturn(0);

  /* Determine number of non-zeros */
  nz = (dfillsparse[w] - w - 1);

  /* Allocate space for our copy of the given sparse matrix representation. */
  ierr = PetscMalloc1(nz + w + 1,rfill);CHKERRQ(ierr);
  ierr = PetscArraycpy(*rfill,dfillsparse,nz+w+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode DMDASetBlockFills_Private2(DM_DA *dd)
{
  PetscErrorCode ierr;
  PetscInt       i,k,cnt = 1;

  PetscFunctionBegin;

  /* ofillcount tracks the columns of ofill that have any nonzero in thems; the value in each location is the number of
   columns to the left with any nonzeros in them plus 1 */
  ierr = PetscCalloc1(dd->w,&dd->ofillcols);CHKERRQ(ierr);
  for (i=0; i<dd->w; i++) {
    for (k=dd->ofill[i]; k<dd->ofill[i+1]; k++) dd->ofillcols[dd->ofill[k]] = 1;
  }
  for (i=0; i<dd->w; i++) {
    if (dd->ofillcols[i]) {
      dd->ofillcols[i] = cnt++;
    }
  }
  PetscFunctionReturn(0);
}



/*@
    DMDASetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DMCreateMatrix().

    Logically Collective on da

    Input Parameter:
+   da - the distributed array
.   dfill - the fill pattern in the diagonal block (may be NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks


    Level: developer

    Notes:
    This only makes sense when you are doing multicomponent problems but using the
       MPIAIJ matrix format

           The format for dfill and ofill is a 2 dimensional dof by dof matrix with 1 entries
       representing coupling and 0 entries for missing coupling. For example
$             dfill[9] = {1, 0, 0,
$                         1, 1, 0,
$                         0, 1, 1}
       means that row 0 is coupled with only itself in the diagonal block, row 1 is coupled with
       itself and row 0 (in the diagonal block) and row 2 is coupled with itself and row 1 (in the
       diagonal block).

     DMDASetGetMatrix() allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the dfill, ofill format

   Contributed by Glenn Hammond

.seealso DMCreateMatrix(), DMDASetGetMatrix(), DMSetMatrixPreallocateOnly()

@*/
PetscErrorCode  DMDASetBlockFills(DM da,const PetscInt *dfill,const PetscInt *ofill)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* save the given dfill and ofill information */
  ierr = DMDASetBlockFills_Private(dfill,dd->w,&dd->dfill);CHKERRQ(ierr);
  ierr = DMDASetBlockFills_Private(ofill,dd->w,&dd->ofill);CHKERRQ(ierr);

  /* count nonzeros in ofill columns */
  ierr = DMDASetBlockFills_Private2(dd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*@
    DMDASetBlockFillsSparse - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DMCreateMatrix(), using sparse representations
    of fill patterns.

    Logically Collective on da

    Input Parameter:
+   da - the distributed array
.   dfill - the sparse fill pattern in the diagonal block (may be NULL, means use dense block)
-   ofill - the sparse fill pattern in the off-diagonal blocks


    Level: developer

    Notes: This only makes sense when you are doing multicomponent problems but using the
       MPIAIJ matrix format

           The format for dfill and ofill is a sparse representation of a
           dof-by-dof matrix with 1 entries representing coupling and 0 entries
           for missing coupling.  The sparse representation is a 1 dimensional
           array of length nz + dof + 1, where nz is the number of non-zeros in
           the matrix.  The first dof entries in the array give the
           starting array indices of each row's items in the rest of the array,
           the dof+1st item contains the value nz + dof + 1 (i.e. the entire length of the array)
           and the remaining nz items give the column indices of each of
           the 1s within the logical 2D matrix.  Each row's items within
           the array are the column indices of the 1s within that row
           of the 2D matrix.  PETSc developers may recognize that this is the
           same format as that computed by the DMDASetBlockFills_Private()
           function from a dense 2D matrix representation.

     DMDASetGetMatrix() allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the dfill, ofill format

   Contributed by Philip C. Roth

.seealso DMDASetBlockFills(), DMCreateMatrix(), DMDASetGetMatrix(), DMSetMatrixPreallocateOnly()

@*/
PetscErrorCode  DMDASetBlockFillsSparse(DM da,const PetscInt *dfillsparse,const PetscInt *ofillsparse)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* save the given dfill and ofill information */
  ierr = DMDASetBlockFillsSparse_Private(dfillsparse,dd->w,&dd->dfill);CHKERRQ(ierr);
  ierr = DMDASetBlockFillsSparse_Private(ofillsparse,dd->w,&dd->ofill);CHKERRQ(ierr);

  /* count nonzeros in ofill columns */
  ierr = DMDASetBlockFills_Private2(dd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode  DMCreateColoring_DA(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode   ierr;
  PetscInt         dim,m,n,p,nc;
  DMBoundaryType   bx,by,bz;
  MPI_Comm         comm;
  PetscMPIInt      size;
  PetscBool        isBAIJ;
  DM_DA            *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           yn  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     xn                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */

  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,NULL,NULL,NULL,&m,&n,&p,&nc,NULL,&bx,&by,&bz,NULL);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (ctype == IS_COLORING_LOCAL) {
    if (size == 1) {
      ctype = IS_COLORING_GLOBAL;
    } else if (dim > 1) {
      if ((m==1 && bx == DM_BOUNDARY_PERIODIC) || (n==1 && by == DM_BOUNDARY_PERIODIC) || (p==1 && bz == DM_BOUNDARY_PERIODIC)) {
        SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"IS_COLORING_LOCAL cannot be used for periodic boundary condition having both ends of the domain  on the same process");
      }
    }
  }

  /* Tell the DMDA it has 1 degree of freedom per grid point so that the coloring for BAIJ
     matrices is for the blocks, not the individual matrix elements  */
  ierr = PetscStrbeginswith(da->mattype,MATBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (!isBAIJ) {ierr = PetscStrbeginswith(da->mattype,MATMPIBAIJ,&isBAIJ);CHKERRQ(ierr);}
  if (!isBAIJ) {ierr = PetscStrbeginswith(da->mattype,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);}
  if (isBAIJ) {
    dd->w  = 1;
    dd->xs = dd->xs/nc;
    dd->xe = dd->xe/nc;
    dd->Xs = dd->Xs/nc;
    dd->Xe = dd->Xe/nc;
  }

  /*
     We do not provide a getcoloring function in the DMDA operations because
   the basic DMDA does not know about matrices. We think of DMDA as being
   more low-level then matrices.
  */
  if (dim == 1) {
    ierr = DMCreateColoring_DA_1d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMCreateColoring_DA_2d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMCreateColoring_DA_3d_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Not done for %D dimension, send us mail petsc-maint@mcs.anl.gov for code",dim);
  if (isBAIJ) {
    dd->w  = nc;
    dd->xs = dd->xs*nc;
    dd->xe = dd->xe*nc;
    dd->Xs = dd->Xs*nc;
    dd->Xe = dd->Xe*nc;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateColoring_DA_2d_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,ys,nx,ny,i,j,ii,gxs,gys,gnx,gny,m,n,M,N,dim,s,k,nc,col;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DMBoundaryType  bx,by;
  DMDAStencilType st;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,NULL,&M,&N,NULL,&nc,&s,&bx,&by,NULL,&st);CHKERRQ(ierr);
  col  = 2*s + 1;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* special case as taught to us by Paul Hovland */
  if (st == DMDA_STENCIL_STAR && s == 1) {
    ierr = DMCreateColoring_DA_2d_5pt_MPIAIJ(da,ctype,coloring);CHKERRQ(ierr);
  } else {

    if (bx == DM_BOUNDARY_PERIODIC && (m % col)) SETERRQ2(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X (%d) is divisible\n\
                                                            by 2*stencil_width + 1 (%d)\n", m, col);
    if (by == DM_BOUNDARY_PERIODIC && (n % col)) SETERRQ2(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y (%d) is divisible\n\
                                                            by 2*stencil_width + 1 (%d)\n", n, col);
    if (ctype == IS_COLORING_GLOBAL) {
      if (!dd->localcoloring) {
        ierr = PetscMalloc1(nc*nx*ny,&colors);CHKERRQ(ierr);
        ii   = 0;
        for (j=ys; j<ys+ny; j++) {
          for (i=xs; i<xs+nx; i++) {
            for (k=0; k<nc; k++) {
              colors[ii++] = k + nc*((i % col) + col*(j % col));
            }
          }
        }
        ncolors = nc + nc*(col-1 + col*(col-1));
        ierr    = ISColoringCreate(comm,ncolors,nc*nx*ny,colors,PETSC_OWN_POINTER,&dd->localcoloring);CHKERRQ(ierr);
      }
      *coloring = dd->localcoloring;
    } else if (ctype == IS_COLORING_LOCAL) {
      if (!dd->ghostedcoloring) {
        ierr = PetscMalloc1(nc*gnx*gny,&colors);CHKERRQ(ierr);
        ii   = 0;
        for (j=gys; j<gys+gny; j++) {
          for (i=gxs; i<gxs+gnx; i++) {
            for (k=0; k<nc; k++) {
              /* the complicated stuff is to handle periodic boundaries */
              colors[ii++] = k + nc*((SetInRange(i,m) % col) + col*(SetInRange(j,n) % col));
            }
          }
        }
        ncolors = nc + nc*(col - 1 + col*(col-1));
        ierr    = ISColoringCreate(comm,ncolors,nc*gnx*gny,colors,PETSC_OWN_POINTER,&dd->ghostedcoloring);CHKERRQ(ierr);
        /* PetscIntView(ncolors,(PetscInt*)colors,0); */

        ierr = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_LOCAL);CHKERRQ(ierr);
      }
      *coloring = dd->ghostedcoloring;
    } else SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  }
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateColoring_DA_3d_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,ys,nx,ny,i,j,gxs,gys,gnx,gny,m,n,p,dim,s,k,nc,col,zs,gzs,ii,l,nz,gnz,M,N,P;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DMBoundaryType  bx,by,bz;
  DMDAStencilType st;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col  = 2*s + 1;
  if (bx == DM_BOUNDARY_PERIODIC && (m % col)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                                                         by 2*stencil_width + 1\n");
  if (by == DM_BOUNDARY_PERIODIC && (n % col)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                                                         by 2*stencil_width + 1\n");
  if (bz == DM_BOUNDARY_PERIODIC && (p % col)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Z is divisible\n\
                                                         by 2*stencil_width + 1\n");

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      ierr = PetscMalloc1(nc*nx*ny*nz,&colors);CHKERRQ(ierr);
      ii   = 0;
      for (k=zs; k<zs+nz; k++) {
        for (j=ys; j<ys+ny; j++) {
          for (i=xs; i<xs+nx; i++) {
            for (l=0; l<nc; l++) {
              colors[ii++] = l + nc*((i % col) + col*(j % col) + col*col*(k % col));
            }
          }
        }
      }
      ncolors = nc + nc*(col-1 + col*(col-1)+ col*col*(col-1));
      ierr    = ISColoringCreate(comm,ncolors,nc*nx*ny*nz,colors,PETSC_OWN_POINTER,&dd->localcoloring);CHKERRQ(ierr);
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!dd->ghostedcoloring) {
      ierr = PetscMalloc1(nc*gnx*gny*gnz,&colors);CHKERRQ(ierr);
      ii   = 0;
      for (k=gzs; k<gzs+gnz; k++) {
        for (j=gys; j<gys+gny; j++) {
          for (i=gxs; i<gxs+gnx; i++) {
            for (l=0; l<nc; l++) {
              /* the complicated stuff is to handle periodic boundaries */
              colors[ii++] = l + nc*((SetInRange(i,m) % col) + col*(SetInRange(j,n) % col) + col*col*(SetInRange(k,p) % col));
            }
          }
        }
      }
      ncolors = nc + nc*(col-1 + col*(col-1)+ col*col*(col-1));
      ierr    = ISColoringCreate(comm,ncolors,nc*gnx*gny*gnz,colors,PETSC_OWN_POINTER,&dd->ghostedcoloring);CHKERRQ(ierr);
      ierr    = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_LOCAL);CHKERRQ(ierr);
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateColoring_DA_1d_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,nx,i,i1,gxs,gnx,l,m,M,dim,s,nc,col;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DMBoundaryType  bx;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,NULL,NULL,&M,NULL,NULL,&nc,&s,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  col  = 2*s + 1;

  if (bx == DM_BOUNDARY_PERIODIC && (m % col)) SETERRQ2(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points %d is divisible\n\
                                                          by 2*stencil_width + 1 %d\n",(int)m,(int)col);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&nx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,NULL,NULL,&gnx,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      ierr = PetscMalloc1(nc*nx,&colors);CHKERRQ(ierr);
      if (dd->ofillcols) {
        PetscInt tc = 0;
        for (i=0; i<nc; i++) tc += (PetscInt) (dd->ofillcols[i] > 0);
        i1 = 0;
        for (i=xs; i<xs+nx; i++) {
          for (l=0; l<nc; l++) {
            if (dd->ofillcols[l] && (i % col)) {
              colors[i1++] =  nc - 1 + tc*((i % col) - 1) + dd->ofillcols[l];
            } else {
              colors[i1++] = l;
            }
          }
        }
        ncolors = nc + 2*s*tc;
      } else {
        i1 = 0;
        for (i=xs; i<xs+nx; i++) {
          for (l=0; l<nc; l++) {
            colors[i1++] = l + nc*(i % col);
          }
        }
        ncolors = nc + nc*(col-1);
      }
      ierr = ISColoringCreate(comm,ncolors,nc*nx,colors,PETSC_OWN_POINTER,&dd->localcoloring);CHKERRQ(ierr);
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!dd->ghostedcoloring) {
      ierr = PetscMalloc1(nc*gnx,&colors);CHKERRQ(ierr);
      i1   = 0;
      for (i=gxs; i<gxs+gnx; i++) {
        for (l=0; l<nc; l++) {
          /* the complicated stuff is to handle periodic boundaries */
          colors[i1++] = l + nc*(SetInRange(i,m) % col);
        }
      }
      ncolors = nc + nc*(col-1);
      ierr    = ISColoringCreate(comm,ncolors,nc*gnx,colors,PETSC_OWN_POINTER,&dd->ghostedcoloring);CHKERRQ(ierr);
      ierr    = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_LOCAL);CHKERRQ(ierr);
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  ierr = ISColoringReference(*coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateColoring_DA_2d_5pt_MPIAIJ(DM da,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode  ierr;
  PetscInt        xs,ys,nx,ny,i,j,ii,gxs,gys,gnx,gny,m,n,dim,s,k,nc;
  PetscInt        ncolors;
  MPI_Comm        comm;
  DMBoundaryType  bx,by;
  ISColoringValue *colors;
  DM_DA           *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,NULL,NULL,NULL,NULL,&nc,&s,&bx,&by,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  if (bx == DM_BOUNDARY_PERIODIC && (m % 5)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible by 5\n");
  if (by == DM_BOUNDARY_PERIODIC && (n % 5)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible by 5\n");

  /* create the coloring */
  if (ctype == IS_COLORING_GLOBAL) {
    if (!dd->localcoloring) {
      ierr = PetscMalloc1(nc*nx*ny,&colors);CHKERRQ(ierr);
      ii   = 0;
      for (j=ys; j<ys+ny; j++) {
        for (i=xs; i<xs+nx; i++) {
          for (k=0; k<nc; k++) {
            colors[ii++] = k + nc*((3*j+i) % 5);
          }
        }
      }
      ncolors = 5*nc;
      ierr    = ISColoringCreate(comm,ncolors,nc*nx*ny,colors,PETSC_OWN_POINTER,&dd->localcoloring);CHKERRQ(ierr);
    }
    *coloring = dd->localcoloring;
  } else if (ctype == IS_COLORING_LOCAL) {
    if (!dd->ghostedcoloring) {
      ierr = PetscMalloc1(nc*gnx*gny,&colors);CHKERRQ(ierr);
      ii = 0;
      for (j=gys; j<gys+gny; j++) {
        for (i=gxs; i<gxs+gnx; i++) {
          for (k=0; k<nc; k++) {
            colors[ii++] = k + nc*((3*SetInRange(j,n) + SetInRange(i,m)) % 5);
          }
        }
      }
      ncolors = 5*nc;
      ierr    = ISColoringCreate(comm,ncolors,nc*gnx*gny,colors,PETSC_OWN_POINTER,&dd->ghostedcoloring);CHKERRQ(ierr);
      ierr    = ISColoringSetType(dd->ghostedcoloring,IS_COLORING_LOCAL);CHKERRQ(ierr);
    }
    *coloring = dd->ghostedcoloring;
  } else SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Unknown ISColoringType %d",(int)ctype);
  PetscFunctionReturn(0);
}

/* =========================================================================== */
extern PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ(DM,Mat,PetscBool);
extern PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ_Fill(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ(DM,Mat,PetscBool);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ_Fill(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ(DM,Mat,PetscBool);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ_Fill(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPIBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPIBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPISBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPISBAIJ(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_2d_MPISELL(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_3d_MPISELL(DM,Mat);
extern PetscErrorCode DMCreateMatrix_DA_IS(DM,Mat);

/*@C
   MatSetupDM - Sets the DMDA that is to be used by the HYPRE_StructMatrix PETSc matrix

   Logically Collective on mat

   Input Parameters:
+  mat - the matrix
-  da - the da

   Level: intermediate

@*/
PetscErrorCode MatSetupDM(Mat mat,DM da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  ierr = PetscTryMethod(mat,"MatSetupDM_C",(Mat,DM),(mat,da));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatView_MPI_DA(Mat A,PetscViewer viewer)
{
  DM                da;
  PetscErrorCode    ierr;
  const char        *prefix;
  Mat               Anatural;
  AO                ao;
  PetscInt          rstart,rend,*petsc,i;
  IS                is;
  MPI_Comm          comm;
  PetscViewerFormat format;

  PetscFunctionBegin;
  /* Check whether we are just printing info, in which case MatView() already viewed everything we wanted to view */
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetDM(A, &da);CHKERRQ(ierr);
  if (!da) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix not generated from a DMDA");

  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc1(rend-rstart,&petsc);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) petsc[i-rstart] = i;
  ierr = AOApplicationToPetsc(ao,rend-rstart,petsc);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,rend-rstart,petsc,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);

  /* call viewer on natural ordering */
  ierr = MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&Anatural);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)A,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)Anatural,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Anatural,((PetscObject)A)->name);CHKERRQ(ierr);
  ((PetscObject)Anatural)->donotPetscObjectPrintClassNamePrefixType = PETSC_TRUE;
  ierr = MatView(Anatural,viewer);CHKERRQ(ierr);
  ((PetscObject)Anatural)->donotPetscObjectPrintClassNamePrefixType = PETSC_FALSE;
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatLoad_MPI_DA(Mat A,PetscViewer viewer)
{
  DM             da;
  PetscErrorCode ierr;
  Mat            Anatural,Aapp;
  AO             ao;
  PetscInt       rstart,rend,*app,i,m,n,M,N;
  IS             is;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetDM(A, &da);CHKERRQ(ierr);
  if (!da) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix not generated from a DMDA");

  /* Load the matrix in natural ordering */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&Anatural);CHKERRQ(ierr);
  ierr = MatSetType(Anatural,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatSetSizes(Anatural,m,n,M,N);CHKERRQ(ierr);
  ierr = MatLoad(Anatural,viewer);CHKERRQ(ierr);

  /* Map natural ordering to application ordering and create IS */
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Anatural,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc1(rend-rstart,&app);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) app[i-rstart] = i;
  ierr = AOPetscToApplication(ao,rend-rstart,app);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,rend-rstart,app,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);

  /* Do permutation and replace header */
  ierr = MatCreateSubMatrix(Anatural,is,is,MAT_INITIAL_MATRIX,&Aapp);CHKERRQ(ierr);
  ierr = MatHeaderReplace(A,&Aapp);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA(DM da, Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       dim,dof,nx,ny,nz,dims[3],starts[3],M,N,P;
  Mat            A;
  MPI_Comm       comm;
  MatType        Atype;
  void           (*aij)(void)=NULL,(*baij)(void)=NULL,(*sbaij)(void)=NULL,(*sell)(void)=NULL,(*is)(void)=NULL;
  MatType        mtype;
  PetscMPIInt    size;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  mtype = da->mattype;

  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           ny  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     nx                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */

  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  M   = dd->M;
  N   = dd->N;
  P   = dd->P;
  dim = da->dim;
  dof = dd->w;
  /* ierr = DMDAGetInfo(da,&dim,&M,&N,&P,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr); */
  ierr = DMDAGetCorners(da,NULL,NULL,NULL,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,dof*nx*ny*nz,dof*nx*ny*nz,dof*M*N*P,dof*M*N*P);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetDM(A,da);CHKERRQ(ierr);
  if (da->structure_only) {
    ierr = MatSetOption(A,MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatGetType(A,&Atype);CHKERRQ(ierr);
  /*
     We do not provide a getmatrix function in the DMDA operations because
   the basic DMDA does not know about matrices. We think of DMDA as being more
   more low-level than matrices. This is kind of cheating but, cause sometimes
   we think of DMDA has higher level than matrices.

     We could switch based on Atype (or mtype), but we do not since the
   specialized setting routines depend only on the particular preallocation
   details of the matrix, not the type itself.
  */
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  if (!aij) {
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  }
  if (!aij) {
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&baij);CHKERRQ(ierr);
    if (!baij) {
      ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&baij);CHKERRQ(ierr);
    }
    if (!baij) {
      ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&sbaij);CHKERRQ(ierr);
      if (!sbaij) {
        ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&sbaij);CHKERRQ(ierr);
      }
      if (!sbaij) {
        ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISELLSetPreallocation_C",&sell);CHKERRQ(ierr);
        if (!sell) {
          ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSELLSetPreallocation_C",&sell);CHKERRQ(ierr);
        }
      }
      if (!sell) {
        ierr = PetscObjectQueryFunction((PetscObject)A,"MatISSetPreallocation_C",&is);CHKERRQ(ierr);
      }
    }
  }
  if (aij) {
    if (dim == 1) {
      if (dd->ofill) {
        ierr = DMCreateMatrix_DA_1d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DMCreateMatrix_DA_1d_MPIAIJ(da,A,PETSC_FALSE);CHKERRQ(ierr);
      }
    } else if (dim == 2) {
      if (dd->ofill) {
        ierr = DMCreateMatrix_DA_2d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DMCreateMatrix_DA_2d_MPIAIJ(da,A,PETSC_FALSE);CHKERRQ(ierr);
      }
    } else if (dim == 3) {
      if (dd->ofill) {
        ierr = DMCreateMatrix_DA_3d_MPIAIJ_Fill(da,A);CHKERRQ(ierr);
      } else {
        ierr = DMCreateMatrix_DA_3d_MPIAIJ(da,A,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  } else if (baij) {
    if (dim == 2) {
      ierr = DMCreateMatrix_DA_2d_MPIBAIJ(da,A);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DMCreateMatrix_DA_3d_MPIBAIJ(da,A);CHKERRQ(ierr);
    } else SETERRQ3(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension! Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
  } else if (sbaij) {
    if (dim == 2) {
      ierr = DMCreateMatrix_DA_2d_MPISBAIJ(da,A);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DMCreateMatrix_DA_3d_MPISBAIJ(da,A);CHKERRQ(ierr);
    } else SETERRQ3(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension! Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
  } else if (sell) {
     if (dim == 2) {
       ierr = DMCreateMatrix_DA_2d_MPISELL(da,A);CHKERRQ(ierr);
     } else if (dim == 3) {
       ierr = DMCreateMatrix_DA_3d_MPISELL(da,A);CHKERRQ(ierr);
     } else SETERRQ3(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Not implemented for %D dimension and Matrix Type: %s in %D dimension! Send mail to petsc-maint@mcs.anl.gov for code",dim,Atype,dim);
  } else if (is) {
    ierr = DMCreateMatrix_DA_IS(da,A);CHKERRQ(ierr);
  } else {
    ISLocalToGlobalMapping ltog;

    ierr = MatSetBlockSize(A,dof);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(A,ltog,ltog);CHKERRQ(ierr);
  }
  ierr = DMDAGetGhostCorners(da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);CHKERRQ(ierr);
  ierr = MatSetStencil(A,dim,dims,starts,dof);CHKERRQ(ierr);
  ierr = MatSetDM(A,da);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    /* change viewer to display matrix in natural ordering */
    ierr = MatSetOperation(A, MATOP_VIEW, (void (*)(void))MatView_MPI_DA);CHKERRQ(ierr);
    ierr = MatSetOperation(A, MATOP_LOAD, (void (*)(void))MatLoad_MPI_DA);CHKERRQ(ierr);
  }
  *J = A;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
PETSC_EXTERN PetscErrorCode MatISSetPreallocation_IS(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);

PetscErrorCode DMCreateMatrix_DA_IS(DM dm,Mat J)
{
  DM_DA                  *da = (DM_DA*)dm->data;
  Mat                    lJ;
  ISLocalToGlobalMapping ltog;
  IS                     is_loc_filt, is_glob;
  const PetscInt         *e_loc,*idx;
  PetscInt               nel,nen,nv,dof,dim,*gidx,nb;
  PetscBool              flg;
  PetscErrorCode         ierr;

  /* The l2g map of DMDA has all ghosted nodes, and e_loc is a subset of all the local nodes (including the ghosted)
     We need to filter the local indices that are represented through the DMDAGetElements decomposition
     This is because the size of the local matrices in MATIS is the local size of the l2g map */
  PetscFunctionBegin;
  dof  = da->w;
  dim  = dm->dim;

  ierr = MatSetBlockSize(J,dof);CHKERRQ(ierr);

  /* get local elements indices in local DMDA numbering */
  ierr = DMDAGetElements(dm,&nel,&nen,&e_loc);CHKERRQ(ierr); /* this will throw an error if the stencil type is not DMDA_STENCIL_BOX */
  ierr = ISCreateBlock(PetscObjectComm((PetscObject)dm),dof,nel*nen,e_loc,PETSC_COPY_VALUES,&is_loc_filt);CHKERRQ(ierr);
  ierr = DMDARestoreElements(dm,&nel,&nen,&e_loc);CHKERRQ(ierr);

  /* obtain a consistent local ordering for MATIS */
  ierr = ISSortRemoveDups(is_loc_filt);CHKERRQ(ierr);
  ierr = ISBlockGetLocalSize(is_loc_filt,&nb);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(ltog,&nv);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(nb,nv/dof),&gidx);CHKERRQ(ierr);
  ierr = ISBlockGetIndices(is_loc_filt,&idx);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(ltog,nb,idx,gidx);CHKERRQ(ierr);
  ierr = ISBlockRestoreIndices(is_loc_filt,&idx);CHKERRQ(ierr);
  ierr = ISCreateBlock(PetscObjectComm((PetscObject)dm),dof,nb,gidx,PETSC_USE_POINTER,&is_glob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is_glob,&ltog);CHKERRQ(ierr);
  ierr = ISDestroy(&is_glob);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);

  /* We also attach a l2g map to the local matrices to have MatSetValueLocal to work */
  ierr = MatISGetLocalMat(J,&lJ);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is_loc_filt,&ltog);CHKERRQ(ierr);
  ierr = ISDestroy(&is_loc_filt);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)lJ),nv/dof,0,1,&is_glob);CHKERRQ(ierr);
  ierr = ISGetIndices(is_glob,&idx);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(ltog,IS_GTOLM_MASK,nv/dof,idx,&nb,gidx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_glob,&idx);CHKERRQ(ierr);
  ierr = ISDestroy(&is_glob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  ierr = ISCreateBlock(PETSC_COMM_SELF,dof,nb,gidx,PETSC_USE_POINTER,&is_loc_filt);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is_loc_filt,&ltog);CHKERRQ(ierr);
  ierr = ISDestroy(&is_loc_filt);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(lJ,ltog,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  ierr = PetscFree(gidx);CHKERRQ(ierr);

  /* Preallocation (not exact): we reuse the preallocation routines of the assembled version  */
  flg = dm->prealloc_only;
  dm->prealloc_only = PETSC_TRUE;
  switch (dim) {
  case 1:
    ierr = PetscObjectComposeFunction((PetscObject)J,"MatMPIAIJSetPreallocation_C",MatISSetPreallocation_IS);CHKERRQ(ierr);
    ierr = DMCreateMatrix_DA_1d_MPIAIJ(dm,J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)J,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
    break;
  case 2:
    ierr = PetscObjectComposeFunction((PetscObject)J,"MatMPIAIJSetPreallocation_C",MatISSetPreallocation_IS);CHKERRQ(ierr);
    ierr = DMCreateMatrix_DA_2d_MPIAIJ(dm,J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)J,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
    break;
  case 3:
    ierr = PetscObjectComposeFunction((PetscObject)J,"MatMPIAIJSetPreallocation_C",MatISSetPreallocation_IS);CHKERRQ(ierr);
    ierr = DMCreateMatrix_DA_3d_MPIAIJ(dm,J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)J,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unhandled dimension %d",dim);
    break;
  }
  dm->prealloc_only = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPISELL(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny,m,n,dim,s,*cols = NULL,k,nc,*rows = NULL,col,cnt,l,p;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,NULL,NULL,NULL,NULL,&nc,&s,&bx,&by,NULL,&st);CHKERRQ(ierr);
  col  = 2*s + 1;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,&rows,col*col*nc*nc,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    pend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      lend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

      cnt = 0;
      for (k=0; k<nc; k++) {
        for (l=lstart; l<lend+1; l++) {
          for (p=pstart; p<pend+1; p++) {
            if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
              cols[cnt++] = k + nc*(slot + gnx*l + p);
            }
          }
        }
        rows[k] = k + nc*(slot);
      }
      ierr = MatPreallocateSetLocal(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqSELLSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPISELLSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(col*col*nc*nc,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {

      pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      pend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

      for (j=ys; j<ys+ny; j++) {
        slot = i - gxs + gnx*(j - gys);

        lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        lend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

        cnt = 0;
        for (k=0; k<nc; k++) {
          for (l=lstart; l<lend+1; l++) {
            for (p=pstart; p<pend+1; p++) {
              if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
                cols[cnt++] = k + nc*(slot + gnx*l + p);
              }
            }
          }
          rows[k] = k + nc*(slot);
        }
        ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPISELL(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols = NULL,k,nc,*rows = NULL,col,cnt,l,p,*dnz = NULL,*onz = NULL;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk,M,N,P;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by,bz;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col  = 2*s + 1;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,&rows,col*col*col*nc*nc,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

        cnt = 0;
        for (l=0; l<nc; l++) {
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                  cols[cnt++] = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                }
              }
            }
          }
          rows[l] = l + nc*(slot);
        }
        ierr = MatPreallocateSetLocal(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqSELLSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPISELLSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(col*col*col*nc*nc*nc,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        for (k=zs; k<zs+nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

          slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

          cnt = 0;
          for (l=0; l<nc; l++) {
            for (ii=istart; ii<iend+1; ii++) {
              for (jj=jstart; jj<jend+1; jj++) {
                for (kk=kstart; kk<kend+1; kk++) {
                  if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                    cols[cnt++] = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                  }
                }
              }
            }
            rows[l] = l + nc*(slot);
          }
          ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ(DM da,Mat J,PetscBool isIS)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny,m,n,dim,s,*cols = NULL,k,nc,*rows = NULL,col,cnt,l,p,M,N;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  MPI_Comm               comm;
  DMBoundaryType         bx,by;
  ISLocalToGlobalMapping ltog,mltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&M,&N,NULL,NULL,&nc,&s,&bx,&by,NULL,&st);CHKERRQ(ierr);
  if (!isIS && bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE) {
    ierr = MatSetOption(J,MAT_SORTED_FULL,PETSC_TRUE);CHKERRQ(ierr);
  }
  col  = 2*s + 1;
  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2*s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2*s >= n) removedups = PETSC_TRUE;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,&rows,col*col*nc*nc,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    pend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      lend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

      cnt = 0;
      for (k=0; k<nc; k++) {
        for (l=lstart; l<lend+1; l++) {
          for (p=pstart; p<pend+1; p++) {
            if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
              cols[cnt++] = k + nc*(slot + gnx*l + p);
            }
          }
        }
        rows[k] = k + nc*(slot);
      }
      if (removedups) {
        ierr = MatPreallocateSetLocalRemoveDups(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
      } else {
        ierr = MatPreallocateSetLocal(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(J,&mltog,NULL);CHKERRQ(ierr);
  if (!mltog) {
    ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  }

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    for (i=xs; i<xs+nx; i++) {

      pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      pend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

      for (j=ys; j<ys+ny; j++) {
        slot = i - gxs + gnx*(j - gys);

        lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        lend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

        cnt = 0;
        for (l=lstart; l<lend+1; l++) {
          for (p=pstart; p<pend+1; p++) {
            if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star have either l = 0 or p = 0 */
              cols[cnt++] = nc*(slot + gnx*l + p);
              for (k=1; k<nc; k++) {
                cols[cnt] = 1 + cols[cnt-1];cnt++;
              }
            }
          }
        }
        for (k=0; k<nc; k++) rows[k] = k + nc*(slot);
        ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    if (bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE) {
      ierr = MatSetOption(J,MAT_SORTED_FULL,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPIAIJ_Fill(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols,k,nc,row,col,cnt,maxcnt = 0,l,p,M,N;
  PetscInt               lstart,lend,pstart,pend,*dnz,*onz;
  DM_DA                  *dd = (DM_DA*)da->data;
  PetscInt               ifill_col,*ofill = dd->ofill, *dfill = dd->dfill;
  MPI_Comm               comm;
  DMBoundaryType         bx,by;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&M,&N,NULL,NULL,&nc,&s,&bx,&by,NULL,&st);CHKERRQ(ierr);
  col  = 2*s + 1;
  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2*s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2*s >= n) removedups = PETSC_TRUE;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc1(col*col*nc,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    pend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      lend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

      for (k=0; k<nc; k++) {
        cnt = 0;
        for (l=lstart; l<lend+1; l++) {
          for (p=pstart; p<pend+1; p++) {
            if (l || p) {
              if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star */
                for (ifill_col=ofill[k]; ifill_col<ofill[k+1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc*(slot + gnx*l + p);
              }
            } else {
              if (dfill) {
                for (ifill_col=dfill[k]; ifill_col<dfill[k+1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc*(slot + gnx*l + p);
              } else {
                for (ifill_col=0; ifill_col<nc; ifill_col++) cols[cnt++] = ifill_col + nc*(slot + gnx*l + p);
              }
            }
          }
        }
        row    = k + nc*(slot);
        maxcnt = PetscMax(maxcnt,cnt);
        if (removedups) {
          ierr   = MatPreallocateSetLocalRemoveDups(ltog,1,&row,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
        } else {
          ierr   = MatPreallocateSetLocal(ltog,1,&row,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    for (i=xs; i<xs+nx; i++) {

      pstart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      pend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));

      for (j=ys; j<ys+ny; j++) {
        slot = i - gxs + gnx*(j - gys);

        lstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        lend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));

        for (k=0; k<nc; k++) {
          cnt = 0;
          for (l=lstart; l<lend+1; l++) {
            for (p=pstart; p<pend+1; p++) {
              if (l || p) {
                if ((st == DMDA_STENCIL_BOX) || (!l || !p)) {  /* entries on star */
                  for (ifill_col=ofill[k]; ifill_col<ofill[k+1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc*(slot + gnx*l + p);
                }
              } else {
                if (dfill) {
                  for (ifill_col=dfill[k]; ifill_col<dfill[k+1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc*(slot + gnx*l + p);
                } else {
                  for (ifill_col=0; ifill_col<nc; ifill_col++) cols[cnt++] = ifill_col + nc*(slot + gnx*l + p);
                }
              }
            }
          }
          row  = k + nc*(slot);
          ierr = MatSetValuesLocal(J,1,&row,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ(DM da,Mat J,PetscBool isIS)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols = NULL,k,nc,*rows = NULL,col,cnt,l,p,*dnz = NULL,*onz = NULL;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk,M,N,P;
  MPI_Comm               comm;
  DMBoundaryType         bx,by,bz;
  ISLocalToGlobalMapping ltog,mltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  if (!isIS && bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE && bz == DM_BOUNDARY_NONE) {
    ierr = MatSetOption(J,MAT_SORTED_FULL,PETSC_TRUE);CHKERRQ(ierr);
  }
  col  = 2*s + 1;

  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2*s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2*s >= n) removedups = PETSC_TRUE;
  if (P == 1 && 2*s >= p) removedups = PETSC_TRUE;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc2(nc,&rows,col*col*col*nc*nc,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

        cnt = 0;
        for (l=0; l<nc; l++) {
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                  cols[cnt++] = l + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                }
              }
            }
          }
          rows[l] = l + nc*(slot);
        }
        if (removedups) {
          ierr = MatPreallocateSetLocalRemoveDups(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
        } else {
          ierr = MatPreallocateSetLocal(ltog,nc,rows,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(J,&mltog,NULL);CHKERRQ(ierr);
  if (!mltog) {
    ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  }

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        for (k=zs; k<zs+nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

          slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

          cnt = 0;
          for (kk=kstart; kk<kend+1; kk++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (ii=istart; ii<iend+1; ii++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                  cols[cnt++] = nc*(slot + ii + gnx*jj + gnx*gny*kk);
                    for (l=1; l<nc; l++) {
                      cols[cnt] = 1 + cols[cnt-1];cnt++;
                  }
                }
              }
            }
          }
          rows[0] = nc*(slot); for (l=1; l<nc; l++) rows[l] = 1 + rows[l-1];
          ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (!isIS && bx == DM_BOUNDARY_NONE && by == DM_BOUNDARY_NONE && bz == DM_BOUNDARY_NONE) {
      ierr = MatSetOption(J,MAT_SORTED_FULL,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ_Fill(DM da,Mat J)
{
  PetscErrorCode         ierr;
  DM_DA                  *dd = (DM_DA*)da->data;
  PetscInt               xs,nx,i,j,gxs,gnx,row,k,l;
  PetscInt               m,dim,s,*cols = NULL,nc,cnt,maxcnt = 0,*ocols;
  PetscInt               *ofill = dd->ofill,*dfill = dd->dfill;
  DMBoundaryType         bx;
  ISLocalToGlobalMapping ltog;
  PetscMPIInt            rank,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)da),&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)da),&size);CHKERRQ(ierr);

  /*
         nc - number of components per grid point

  */
  ierr = DMDAGetInfo(da,&dim,&m,NULL,NULL,NULL,NULL,NULL,&nc,&s,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  if (s > 1) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Matrix creation for 1d not implemented correctly for stencil width larger than 1");
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&nx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,NULL,NULL,&gnx,NULL,NULL);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = PetscCalloc2(nx*nc,&cols,nx*nc,&ocols);CHKERRQ(ierr);

  /*
        note should be smaller for first and last process with no periodic
        does not handle dfill
  */
  cnt = 0;
  /* coupling with process to the left */
  for (i=0; i<s; i++) {
    for (j=0; j<nc; j++) {
      ocols[cnt] = ((!rank) ? 0 : (s - i)*(ofill[j+1] - ofill[j]));
      cols[cnt]  = dfill[j+1] - dfill[j] + (s + i)*(ofill[j+1] - ofill[j]);
      if (!rank && (dd->bx == DM_BOUNDARY_PERIODIC)) {
        if (size > 1) ocols[cnt] += (s - i)*(ofill[j+1] - ofill[j]);
        else cols[cnt] += (s - i)*(ofill[j+1] - ofill[j]);
      }
      maxcnt = PetscMax(maxcnt,ocols[cnt]+cols[cnt]);
      cnt++;
    }
  }
  for (i=s; i<nx-s; i++) {
    for (j=0; j<nc; j++) {
      cols[cnt] = dfill[j+1] - dfill[j] + 2*s*(ofill[j+1] - ofill[j]);
      maxcnt = PetscMax(maxcnt,ocols[cnt]+cols[cnt]);
      cnt++;
    }
  }
  /* coupling with process to the right */
  for (i=nx-s; i<nx; i++) {
    for (j=0; j<nc; j++) {
      ocols[cnt] = ((rank == (size-1)) ? 0 : (i - nx + s + 1)*(ofill[j+1] - ofill[j]));
      cols[cnt]  = dfill[j+1] - dfill[j] + (s + nx - i - 1)*(ofill[j+1] - ofill[j]);
      if ((rank == size-1) && (dd->bx == DM_BOUNDARY_PERIODIC)) {
        if (size > 1) ocols[cnt] += (i - nx + s + 1)*(ofill[j+1] - ofill[j]);
        else cols[cnt] += (i - nx + s + 1)*(ofill[j+1] - ofill[j]);
      }
      maxcnt = PetscMax(maxcnt,ocols[cnt]+cols[cnt]);
      cnt++;
    }
  }

  ierr = MatSeqAIJSetPreallocation(J,0,cols);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,cols,0,ocols);CHKERRQ(ierr);
  ierr = PetscFree2(cols,ocols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc1(maxcnt,&cols);CHKERRQ(ierr);
    row = xs*nc;
    /* coupling with process to the left */
    for (i=xs; i<xs+s; i++) {
      for (j=0; j<nc; j++) {
        cnt = 0;
        if (rank) {
          for (l=0; l<s; l++) {
            for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i - s + l)*nc + ofill[k];
          }
        }
        if (!rank && (dd->bx == DM_BOUNDARY_PERIODIC)) {
          for (l=0; l<s; l++) {
            for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (m + i - s - l)*nc + ofill[k];
          }
        }
        if (dfill) {
          for (k=dfill[j]; k<dfill[j+1]; k++) {
            cols[cnt++] = i*nc + dfill[k];
          }
        } else {
          for (k=0; k<nc; k++) {
            cols[cnt++] = i*nc + k;
          }
        }
        for (l=0; l<s; l++) {
          for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i + s - l)*nc + ofill[k];
        }
        ierr = MatSetValues(J,1,&row,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
        row++;
      }
    }
    for (i=xs+s; i<xs+nx-s; i++) {
      for (j=0; j<nc; j++) {
        cnt = 0;
        for (l=0; l<s; l++) {
          for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i - s + l)*nc + ofill[k];
        }
        if (dfill) {
          for (k=dfill[j]; k<dfill[j+1]; k++) {
            cols[cnt++] = i*nc + dfill[k];
          }
        } else {
          for (k=0; k<nc; k++) {
            cols[cnt++] = i*nc + k;
          }
        }
        for (l=0; l<s; l++) {
          for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i + s - l)*nc + ofill[k];
        }
        ierr = MatSetValues(J,1,&row,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
        row++;
      }
    }
    /* coupling with process to the right */
    for (i=xs+nx-s; i<xs+nx; i++) {
      for (j=0; j<nc; j++) {
        cnt = 0;
        for (l=0; l<s; l++) {
          for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i - s + l)*nc + ofill[k];
        }
        if (dfill) {
          for (k=dfill[j]; k<dfill[j+1]; k++) {
            cols[cnt++] = i*nc + dfill[k];
          }
        } else {
          for (k=0; k<nc; k++) {
            cols[cnt++] = i*nc + k;
          }
        }
        if (rank < size-1) {
          for (l=0; l<s; l++) {
            for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i + s - l)*nc + ofill[k];
          }
        }
        if ((rank == size-1) && (dd->bx == DM_BOUNDARY_PERIODIC)) {
          for (l=0; l<s; l++) {
            for (k=ofill[j]; k<ofill[j+1]; k++) cols[cnt++] = (i - s - l - m + 2)*nc + ofill[k];
          }
        }
        ierr = MatSetValues(J,1,&row,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
        row++;
      }
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateMatrix_DA_1d_MPIAIJ(DM da,Mat J,PetscBool isIS)
{
  PetscErrorCode         ierr;
  PetscInt               xs,nx,i,i1,slot,gxs,gnx;
  PetscInt               m,dim,s,*cols = NULL,nc,*rows = NULL,col,cnt,l;
  PetscInt               istart,iend;
  DMBoundaryType         bx;
  ISLocalToGlobalMapping ltog,mltog;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,NULL,NULL,NULL,NULL,NULL,&nc,&s,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!isIS && bx == DM_BOUNDARY_NONE) {
    ierr = MatSetOption(J,MAT_SORTED_FULL,PETSC_TRUE);CHKERRQ(ierr);
  }
  col  = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&nx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,NULL,NULL,&gnx,NULL,NULL);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,col*nc,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,col*nc,NULL,col*nc,NULL);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(J,&mltog,NULL);CHKERRQ(ierr);
  if (!mltog) {
    ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);
  }

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscMalloc2(nc,&rows,col*nc*nc,&cols);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = PetscMax(-s,gxs - i);
      iend   = PetscMin(s,gxs + gnx - i - 1);
      slot   = i - gxs;

      cnt = 0;
      for (i1=istart; i1<iend+1; i1++) {
        cols[cnt++] = nc*(slot + i1);
        for (l=1; l<nc; l++) {
          cols[cnt] = 1 + cols[cnt-1];cnt++;
        }
      }
      rows[0] = nc*(slot); for (l=1; l<nc; l++) rows[l] = 1 + rows[l-1];
      ierr = MatSetValuesLocal(J,nc,rows,cnt,cols,NULL,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (!isIS && bx == DM_BOUNDARY_NONE) {
      ierr = MatSetOption(J,MAT_SORTED_FULL,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPIBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols,nc,col,cnt,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,ii,jj;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
     nc - number of components per grid point
     col - number of colors needed in one direction for single component problem
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,NULL,NULL,NULL,NULL,&nc,&s,&bx,&by,NULL,&st);CHKERRQ(ierr);
  col  = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc1(col*col*nc*nc,&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny,nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      slot   = i - gxs + gnx*(j - gys);

      /* Find block columns in block row */
      cnt = 0;
      for (ii=istart; ii<iend+1; ii++) {
        for (jj=jstart; jj<jend+1; jj++) {
          if (st == DMDA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
            cols[cnt++] = slot + ii + gnx*jj;
          }
        }
      }
      ierr = MatPreallocateSetLocalBlock(ltog,1,&slot,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatSeqBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(col*col*nc*nc,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        slot = i - gxs + gnx*(j - gys);
        cnt  = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            if (st == DMDA_STENCIL_BOX || !ii || !jj) { /* BOX or on the STAR */
              cols[cnt++] = slot + ii + gnx*jj;
            }
          }
        }
        ierr = MatSetValuesBlockedLocal(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPIBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by,bz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,NULL,NULL,NULL,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col  = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc1(col*col*col,&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny*nz,nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

        /* Find block columns in block row */
        cnt = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            for (kk=kstart; kk<kend+1; kk++) {
              if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                cols[cnt++] = slot + ii + gnx*jj + gnx*gny*kk;
              }
            }
          }
        }
        ierr = MatPreallocateSetLocalBlock(ltog,1,&slot,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(col*col*col*nc*nc,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        for (k=zs; k<zs+nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

          slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

          cnt = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                  cols[cnt++] = slot + ii + gnx*jj + gnx*gny*kk;
                }
              }
            }
          }
          ierr = MatSetValuesBlockedLocal(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This helper is for of SBAIJ preallocation, to discard the lower-triangular values which are difficult to
  identify in the local ordering with periodic domain.
*/
static PetscErrorCode L2GFilterUpperTriangular(ISLocalToGlobalMapping ltog,PetscInt *row,PetscInt *cnt,PetscInt col[])
{
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingApplyBlock(ltog,1,row,row);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(ltog,*cnt,col,col);CHKERRQ(ierr);
  for (i=0,n=0; i<*cnt; i++) {
    if (col[i] >= *row) col[n++] = col[i];
  }
  *cnt = n;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_2d_MPISBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols,nc,col,cnt,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,ii,jj;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
     nc - number of components per grid point
     col - number of colors needed in one direction for single component problem
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,NULL,NULL,NULL,NULL,&nc,&s,&bx,&by,NULL,&st);CHKERRQ(ierr);
  col  = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gnx,&gny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc1(col*col*nc*nc,&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny,nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      slot   = i - gxs + gnx*(j - gys);

      /* Find block columns in block row */
      cnt = 0;
      for (ii=istart; ii<iend+1; ii++) {
        for (jj=jstart; jj<jend+1; jj++) {
          if (st == DMDA_STENCIL_BOX || !ii || !jj) {
            cols[cnt++] = slot + ii + gnx*jj;
          }
        }
      }
      ierr = L2GFilterUpperTriangular(ltog,&slot,&cnt,cols);CHKERRQ(ierr);
      ierr = MatPreallocateSymmetricSetBlock(slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatSeqSBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(col*col*nc*nc,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        slot   = i - gxs + gnx*(j - gys);

        /* Find block columns in block row */
        cnt = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            if (st == DMDA_STENCIL_BOX || !ii || !jj) {
              cols[cnt++] = slot + ii + gnx*jj;
            }
          }
        }
        ierr = L2GFilterUpperTriangular(ltog,&slot,&cnt,cols);CHKERRQ(ierr);
        ierr = MatSetValuesBlocked(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_DA_3d_MPISBAIJ(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols,k,nc,col,cnt,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by,bz;
  DMDAStencilType        st;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  /*
     nc - number of components per grid point
     col - number of colors needed in one direction for single component problem
  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,NULL,NULL,NULL,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col  = 2*s + 1;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* create the matrix */
  ierr = PetscMalloc1(col*col*col,&cols);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nx*ny*nz,nx*ny*nz,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

        /* Find block columns in block row */
        cnt = 0;
        for (ii=istart; ii<iend+1; ii++) {
          for (jj=jstart; jj<jend+1; jj++) {
            for (kk=kstart; kk<kend+1; kk++) {
              if ((st == DMDA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) {
                cols[cnt++] = slot + ii + gnx*jj + gnx*gny*kk;
              }
            }
          }
        }
        ierr = L2GFilterUpperTriangular(ltog,&slot,&cnt,cols);CHKERRQ(ierr);
        ierr = MatPreallocateSymmetricSetBlock(slot,cnt,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatSeqSBAIJSetPreallocation(J,nc,0,dnz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(J,nc,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(col*col*col*nc*nc,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        for (k=zs; k<zs+nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

          slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

          cnt = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if ((st == DMDA_STENCIL_BOX) || (!ii && !jj) || (!jj && !kk) || (!ii && !kk)) {
                  cols[cnt++] = slot + ii + gnx*jj + gnx*gny*kk;
                }
              }
            }
          }
          ierr = L2GFilterUpperTriangular(ltog,&slot,&cnt,cols);CHKERRQ(ierr);
          ierr = MatSetValuesBlocked(J,1,&slot,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

PetscErrorCode DMCreateMatrix_DA_3d_MPIAIJ_Fill(DM da,Mat J)
{
  PetscErrorCode         ierr;
  PetscInt               xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;
  PetscInt               m,n,dim,s,*cols,k,nc,row,col,cnt, maxcnt = 0,l,p,*dnz,*onz;
  PetscInt               istart,iend,jstart,jend,kstart,kend,zs,nz,gzs,gnz,ii,jj,kk,M,N,P;
  DM_DA                  *dd = (DM_DA*)da->data;
  PetscInt               ifill_col,*dfill = dd->dfill,*ofill = dd->ofill;
  MPI_Comm               comm;
  PetscScalar            *values;
  DMBoundaryType         bx,by,bz;
  ISLocalToGlobalMapping ltog;
  DMDAStencilType        st;
  PetscBool              removedups = PETSC_FALSE;

  PetscFunctionBegin;
  /*
         nc - number of components per grid point
         col - number of colors needed in one direction for single component problem

  */
  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&nc,&s,&bx,&by,&bz,&st);CHKERRQ(ierr);
  col  = 2*s + 1;
  if (bx == DM_BOUNDARY_PERIODIC && (m % col)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in X is divisible\n\
                 by 2*stencil_width + 1\n");
  if (by == DM_BOUNDARY_PERIODIC && (n % col)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Y is divisible\n\
                 by 2*stencil_width + 1\n");
  if (bz == DM_BOUNDARY_PERIODIC && (p % col)) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"For coloring efficiency ensure number of grid points in Z is divisible\n\
                 by 2*stencil_width + 1\n");

  /*
       With one processor in periodic domains in a skinny dimension the code will label nonzero columns multiple times
       because of "wrapping" around the end of the domain hitting an entry already counted in the other direction.
  */
  if (M == 1 && 2*s >= m) removedups = PETSC_TRUE;
  if (N == 1 && 2*s >= n) removedups = PETSC_TRUE;
  if (P == 1 && 2*s >= p) removedups = PETSC_TRUE;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gnx,&gny,&gnz);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = PetscMalloc1(col*col*col*nc,&cols);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);

  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny*nz,nc*nx*ny*nz,dnz,onz);CHKERRQ(ierr);

  ierr = MatSetBlockSize(J,nc);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {
    istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
    iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
    for (j=ys; j<ys+ny; j++) {
      jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
      jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
      for (k=zs; k<zs+nz; k++) {
        kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
        kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

        slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

        for (l=0; l<nc; l++) {
          cnt = 0;
          for (ii=istart; ii<iend+1; ii++) {
            for (jj=jstart; jj<jend+1; jj++) {
              for (kk=kstart; kk<kend+1; kk++) {
                if (ii || jj || kk) {
                  if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                    for (ifill_col=ofill[l]; ifill_col<ofill[l+1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                  }
                } else {
                  if (dfill) {
                    for (ifill_col=dfill[l]; ifill_col<dfill[l+1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                  } else {
                    for (ifill_col=0; ifill_col<nc; ifill_col++) cols[cnt++] = ifill_col + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                  }
                }
              }
            }
          }
          row  = l + nc*(slot);
          maxcnt = PetscMax(maxcnt,cnt);
          if (removedups) {
            ierr = MatPreallocateSetLocalRemoveDups(ltog,1,&row,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
          } else {
            ierr = MatPreallocateSetLocal(ltog,1,&row,ltog,cnt,cols,dnz,onz);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(J,ltog,ltog);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  if (!da->prealloc_only) {
    ierr = PetscCalloc1(maxcnt,&values);CHKERRQ(ierr);
    for (i=xs; i<xs+nx; i++) {
      istart = (bx == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-i));
      iend   = (bx == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,m-i-1));
      for (j=ys; j<ys+ny; j++) {
        jstart = (by == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-j));
        jend   = (by == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,n-j-1));
        for (k=zs; k<zs+nz; k++) {
          kstart = (bz == DM_BOUNDARY_PERIODIC) ? -s : (PetscMax(-s,-k));
          kend   = (bz == DM_BOUNDARY_PERIODIC) ?  s : (PetscMin(s,p-k-1));

          slot = i - gxs + gnx*(j - gys) + gnx*gny*(k - gzs);

          for (l=0; l<nc; l++) {
            cnt = 0;
            for (ii=istart; ii<iend+1; ii++) {
              for (jj=jstart; jj<jend+1; jj++) {
                for (kk=kstart; kk<kend+1; kk++) {
                  if (ii || jj || kk) {
                    if ((st == DMDA_STENCIL_BOX) || ((!ii && !jj) || (!jj && !kk) || (!ii && !kk))) {/* entries on star*/
                      for (ifill_col=ofill[l]; ifill_col<ofill[l+1]; ifill_col++) cols[cnt++] = ofill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                    }
                  } else {
                    if (dfill) {
                      for (ifill_col=dfill[l]; ifill_col<dfill[l+1]; ifill_col++) cols[cnt++] = dfill[ifill_col] + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                    } else {
                      for (ifill_col=0; ifill_col<nc; ifill_col++) cols[cnt++] = ifill_col + nc*(slot + ii + gnx*jj + gnx*gny*kk);
                    }
                  }
                }
              }
            }
            row  = l + nc*(slot);
            ierr = MatSetValuesLocal(J,1,&row,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
    /* do not copy values to GPU since they are all zero and not yet needed there */
    ierr = MatBindToCPU(J,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatBindToCPU(J,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
