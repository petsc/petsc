
#include "appctx.h"

int AppPartitionGetOwnedSize(AppPartition *part, int *m);

#undef  __FUNCT__
#define __FUNCT__ "AppPartitionSetUp"
int AppPartitionSetUp(AppPartition *part, MPI_Comm comm, int Nelx, int Nely, int nsdx, int nsdy,
                      double xmin, double xmax, double ymin, double ymax)
{
  int ierr;

  PetscFunctionBegin;

  part->comm = comm;
  ierr = MPI_Comm_rank(comm,&part->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&part->size);CHKERRQ(ierr);

  if ((Nelx<=0) || (Nely<=0) || (nsdx<=0) || (nsdy<=0)) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"The number of subdomains and the number of elements must be strictly positive!");
  }
  if ((Nelx % nsdx) || (Nely % nsdy)) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"The number of subdomains must divide the number of elements in each direction!");
  }
  if (part->size != (nsdx*nsdy)) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"The number of processors doesn't match the number of subdomains!");
  }

  part->nsdx = nsdx;
  part->nsdy = nsdy;
  part->nelx = Nelx/nsdx;
  part->nely = Nely/nsdy;
  part->xmin = xmin;
  part->ymin = ymin;
  part->delx = (xmax-xmin)/Nelx;
  part->dely = (ymax-ymin)/Nely;

  ierr = AppPartitionGetOwnedSize(part,&(part->m));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
   AppPartitionGetNodes -

   Input argument
.  part - AppPartition context
.  el - element number (in local numbering)

   Output argument
.  nodes - points to the indices (in local numbering) of the vertices of el

*/
#undef  __FUNCT__
#define __FUNCT__ "AppPartitionGetNodes"
int AppPartitionGetNodes(AppPartition *part, int el, int **nodes)
{
  int i, j;

  PetscFunctionBegin;

  i = el / part->nely;
  j = el % part->nely;

  part->local_nodes[0] = (i)*(part->nely+1)+(j);
  part->local_nodes[1] = (i)*(part->nely+1)+(j+1);
  part->local_nodes[2] = (i+1)*(part->nely+1)+(j+1);
  part->local_nodes[3] = (i+1)*(part->nely+1)+(j);

  *nodes = part->local_nodes;

  PetscFunctionReturn(0);
}

/*
   AppPartitionGetCoords -

   Input argument
.  part - AppPartition context
.  el - element number (in local numbering)

   Output argument
.  nodes - points to the coordinates of the vertices of el

*/
#undef  __FUNCT__
#define __FUNCT__ "AppPartitionGetCoords"
int AppPartitionGetCoords(AppPartition *part, int el, double **coords)
{
  int i, j;
  double offsetx, offsety;

  PetscFunctionBegin;

  offsetx = part->xmin + (part->rank / part->nsdy) * part->nelx * part->delx;
  offsety = part->ymin + (part->rank % part->nsdy) * part->nely * part->dely;

  i = el / part->nely;
  j = el % part->nely;

  part->coords[0] = (i)*(part->delx) + offsetx; part->coords[1] = (j)*(part->dely) + offsety;
  part->coords[2] = (i)*(part->delx) + offsetx; part->coords[3] = (j+1)*(part->dely) + offsety;
  part->coords[4] = (i+1)*(part->delx) + offsetx; part->coords[5] = (j+1)*(part->dely) + offsety;
  part->coords[6] = (i+1)*(part->delx) + offsetx; part->coords[7] = (j)*(part->dely) + offsety;

  *coords = part->coords;

  PetscFunctionReturn(0);
}

/*
   AppPartitionGetOwnedSize -

   Input argument
.  part - AppPartition context

   Output argument
.  m - number of nodes owned by the processor

*/
#undef  __FUNCT__
#define __FUNCT__ "AppPartitionGetOwnedSize"
int AppPartitionGetOwnedSize(AppPartition *part, int *m)
{
  int I, J;

  PetscFunctionBegin;

  I = (part->rank / part->nsdy);
  J = (part->rank % part->nsdy);

  if (I != (part->nsdx-1)) {
    if (J != (part->nsdy-1)) {
      *m = (part->nelx) * (part->nely);
    } else {
      *m = (part->nelx) * (part->nely+1);
    }
  } else {
    if (J != (part->nsdy-1)) {
      *m = (part->nelx+1) * (part->nely);
    } else {
      *m = (part->nelx+1) * (part->nely+1);
    }
  }

  PetscFunctionReturn(0);
}

/*
   AppPartitionCreateLocalToGlobalMapping -

   Input argument
.  part - AppPartition context

   Output argument
.  mapping - mapping from local to global numbering

*/
#undef  __FUNCT__
#define __FUNCT__ "AppPartitionCreateLocalToGlobalMapping"
#define offset(I,J) ((I)*typ_col+(J)*typ_sd+(((I)==(part->nsdx-1))?((J)*(part->nely)):0))
int AppPartitionCreateLocalToGlobalMapping(AppPartition *part, ISLocalToGlobalMapping *mapping)
{
  int i, j, I, J, ierr, *indices, *p, typ_sd, typ_col;

  PetscFunctionBegin;

  ierr = PetscMalloc((part->nelx+1)*(part->nely+1)*sizeof(int),&indices);CHKERRQ(ierr);
  p = indices;

  I = (part->rank / part->nsdy);
  J = (part->rank % part->nsdy);

  typ_sd  = part->nelx * part->nely;
  typ_col = part->nsdy * typ_sd + part->nelx;

  if (I != (part->nsdx-1)) {
    if (J != (part->nsdy-1)) {
      for (i=0; i<part->nelx; i++) {
        for (j=0; j<part->nely; j++) {
          *(p++) = offset(I,J) + i*(part->nely) + j;
        }
        if (J != (part->nsdy-2)) {
          *(p++) = offset(I,J+1) + i*(part->nely);
        } else {
          *(p++) = offset(I,J+1) + i*(part->nely+1);
	}
      }
      {
        for (j=0; j<part->nely; j++) {
          *(p++) = offset(I+1,J) + j;
        }
        {
          *(p++) = offset(I+1,J+1);
	}
      }
    } else { /* (I != (part->nsdx-1)) && (J == (part->nsdy-1)) */
      for (i=0; i<part->nelx; i++) {
        for (j=0; j<=part->nely; j++) {
          *(p++) = offset(I,J) + i*(part->nely+1) + j;
	}
      }
      {
        for (j=0; j<=part->nely; j++) {
          *(p++) = offset(I+1,J) + j;
	}
      }      
    }
  } else { /* (I == (part->nsdx-1)) && (J != (part->nsdy-1)) */
    if (J != (part->nsdy-1)) {
      for (i=0; i<=part->nelx; i++) {
        for (j=0; j<part->nely; j++) {
          *(p++) = offset(I,J) + i*(part->nely) + j;   /* FIX HERE!!! */
        }
        if (J != (part->nsdy-2)) {
          *(p++) = offset(I,J+1) + i*(part->nely);
	} else {
          *(p++) = offset(I,J+1) + i*(part->nely+1);
	}
      }
    } else { /* (I == (part->nsdx-1)) && (J == (part->nsdy-1)) */
      for (i=0; i<=part->nelx; i++) {
        for (j=0; j<=part->nely; j++) {
          *(p++) = offset(I,J) + i*(part->nely+1) + j;
        }
      }
    }
  }

  ierr = ISLocalToGlobalMappingCreate(part->comm,(part->nelx+1)*(part->nely+1),indices,mapping);CHKERRQ(ierr);

  ierr = PetscFree(indices);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef offset

/*
   AppPartitionGetBoundaryNodesAndCoords

   Input argument
.  part - AppPartition context

   Output argument
.  boundary -
.  coords -

*/
#undef  __FUNCT__
#define __FUNCT__ "AppPartitionGetBoundaryNodesAndCoords"
int AppPartitionGetBoundaryNodesAndCoords(AppPartition *part, int *n, int **boundary, double **coords)
{
  int i, j, I, J, ierr, *pb;
  double *pc, offsetx, offsety;
  PetscTruth flg;

  ierr = PetscOptionsHasName(PETSC_NULL,"-dirichlet_on_left",&flg);CHKERRQ(ierr);

  PetscFunctionBegin;

  I = (part->rank / part->nsdy);
  J = (part->rank % part->nsdy);

  offsetx = part->xmin + I * part->nelx * part->delx;
  offsety = part->ymin + J * part->nely * part->dely;

  {
    int counter = 0;
    if (I==0) { counter += part->nely+1; }
    if (!flg) {
      if (I==(part->nsdx-1)) { counter += part->nely+1; }
      if (J==0) {
	counter += part->nelx+1;
	if (I==0) { counter--; }
	if (I==(part->nsdx-1)) { counter--; }
      }
      if (J==(part->nsdy-1)) {
	counter += part->nelx+1;
	if (I==0) { counter--; }
	if (I==(part->nsdx-1)) { counter--; }
      }
    }

    *n = counter;

    counter++; /* This is to avoid zero-length allocations. */
    ierr = PetscMalloc(counter*sizeof(int),boundary);CHKERRQ(ierr);
    pb   = *boundary;
    ierr = PetscMalloc(2*counter*sizeof(double),coords);CHKERRQ(ierr);
    pc   = *coords;
  }

  if (I==0) {
    for (j=0; j<=part->nely; j++) {
      *(pb++) = j;
      *(pc++) = offsetx;
      *(pc++) = offsety + j*(part->dely);
    }
  }
  if (!flg) {
    if (J==(part->nsdy-1)) {
      if (I != 0) {
        *(pb++) = part->nely;
        *(pc++) = offsetx;
        *(pc++) = offsety + (part->nely)*(part->dely);
      }
      for(i=1; i<part->nelx; i++) {
        *(pb++) = i*(part->nely+1) + part->nely;
        *(pc++) = offsetx + i*(part->delx);
        *(pc++) = offsety + (part->nely)*(part->dely);
      }
      if (I != (part->nsdx-1)) {
        *(pb++) = (part->nelx)*(part->nely+1) + part->nely;
        *(pc++) = offsetx + (part->nelx)*(part->delx);
        *(pc++) = offsety + (part->nely)*(part->dely);
      }
    }
    if (I==(part->nsdx-1)) {
      for(j=part->nely; j>=0; j--) {
        *(pb++) = part->nelx*(part->nely+1)+j;
        *(pc++) = offsetx + (part->nelx)*(part->delx);
        *(pc++) = offsety + j*(part->dely);
      }
    }
    if (J==0) {
      if (I != (part->nsdx-1)) {
        *(pb++) = (part->nelx)*(part->nely+1);
        *(pc++) = offsetx + (part->nelx)*(part->delx);
        *(pc++) = offsety;
      }
      for (i=part->nelx-1; i>=1; i--) {
        *(pb++) = i*(part->nely+1);
        *(pc++) = offsetx + i*(part->delx);
        *(pc++) = offsety;
      }
      if (I != 0) {
        *(pb++) = 0;
        *(pc++) = offsetx;
        *(pc++) = offsety;
      }
    }
  }

  PetscFunctionReturn(0);
}





