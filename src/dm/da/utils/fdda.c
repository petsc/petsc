#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdda.c,v 1.9 1997/07/09 21:01:12 balay Exp bsmith $";
#endif
 
#include "da.h"     /*I      "da.h"     I*/
#include "mat.h"    /*I      "mat.h"    I*/

#undef __FUNC__  
#define __FUNC__ "DAGetColoring2dBox" 
/*@C
      DAGetColoring2dBox - Gets the coloring required for computing the Jacobian via
          finite differences on a function defined using the nine point stencil
          on a two dimensional grid. 

          This is a utility routine that will change over time, not part of the 
          core PETSc package.

  Input Parameter:
.    da - the distributed array

  Output Parameters:
.    coloring - matrix coloring for compute Jacobians
.    J  - matrix with the correct nonzero structured 
            (obviously without the correct Jacobian values)

.seealso: DAGetColoring2dBoxMC()

@*/
int DAGetColoring2dBox(DA da,ISColoring *coloring,Mat *J)
{
  int      ierr, xs,ys,nx,ny,*colors,i,j,ii,slot,gxs,gys,ys1,ny1,nx1,gnx,gny;           
  int      m,n,dim,w,s,N,indices[9],*gindices,k,xs1;
  MPI_Comm comm;
  Scalar   values[9];
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
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s,0); CHKERRQ(ierr);
  if (dim != 2) SETERRQ(1,0,"2d only");
  if (w != 1)   SETERRQ(1,0,"Scalar problems only");
  if (s != 1)   SETERRQ(1,0,"Stencil width 1 only");
  /* also no support for periodic boundary conditions */
  N = m*n;
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm); CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(da,PETSC_NULL,&gindices);

  /* create the coloring */
  colors = (int *) PetscMalloc( nx*ny*sizeof(int) ); CHKPTRQ(colors);
  ii = 0;
  for ( j=ys; j<ys+ny; j++ ) {
    for ( i=xs; i<xs+nx; i++ ) {
      colors[ii++] = (i % 3) + 3*(j % 3);
    }
  }
  ierr = ISColoringCreate(comm,nx*ny,colors,coloring); CHKERRQ(ierr);
  PetscFree(colors);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nx*ny,nx*ny,PETSC_DECIDE,PETSC_DECIDE,9,0,0,0,J);CHKERRQ(ierr);  

  values[0] = 0; values[1] = 0; values[2] = 0; values[3] = 0; values[4] = 0; 
  values[5] = 0; values[6] = 0; values[7] = 0; values[8] = 0; 


  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then map those indices to the global PETSc ordering
    before inserting in the matrix
  */
  
  /* fill up Jacobian for left edge */
  if (xs == 0) {
    ys1 = ys;
    ny1 = ny;
    /* lower left corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs - gxs + gnx*(ys - gys);
      indices[0] = slot;         indices[1] = slot + 1; 
      indices[2] = slot + gnx;   indices[3] = slot + gnx + 1;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    /* upper left corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs - gxs + gnx*(ys + (ny - 1) - gys);
      indices[0] = slot - gnx;   indices[1] = slot - gnx + 1;
      indices[2] = slot;         indices[3] = slot + 1;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs - gxs + gnx*(j - gys);
      indices[0] = slot - gnx;   indices[1] = slot - gnx + 1;
      indices[2] = slot;         indices[3] = slot + 1;
      indices[4] = slot + gnx;   indices[5] = slot + gnx + 1;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);  
    }
  }      

  /* fill up Jacobian for right edge */
  if (xs + nx == m) {
    ys1 = ys;
    ny1 = ny;
    /* lower right corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys - gys);
      indices[0] = slot - 1;         indices[1] = slot; 
      indices[2] = slot + gnx - 1;   indices[3] = slot + gnx;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    /* upper right corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys + (ny - 1) - gys);
      indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;
      indices[2] = slot-1;           indices[3] = slot;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs + (nx - 1) - gxs + gnx*(j - gys);
      indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;
      indices[2] = slot - 1;         indices[3] = slot;
      indices[4] = slot + gnx - 1;   indices[5] = slot + gnx;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);   
    }
  }   

  /* fill up Jacobian for bottom */
  if (ys == 0) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys - gys);
      indices[0] = slot - 1;       indices[1] = slot;        indices[2] = slot + 1;
      indices[3] = slot + gnx - 1; indices[4] = slot + gnx;  indices[5] = slot + gnx + 1;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for top */
  if (ys + ny == n) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys + (ny - 1) - gys);
      indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;  indices[2] = slot - gnx + 1;
      indices[3] = slot - 1;         indices[4] = slot;        indices[5] = slot  + 1;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for interior of grid */
  nx1 = nx;
  ny1 = ny;
  if (xs == 0)    {xs++; nx1--;}
  if (ys == 0)    {ys++; ny1--;}
  if (xs+nx1 == m) {nx1--;}
  if (ys+ny1 == n) {ny1--;}
  for ( i=xs; i<xs+nx1; i++ ) {
    for ( j=ys; j<ys+ny1; j++ ) {
      slot = i - gxs + gnx*(j - gys);
      indices[0] = slot - gnx - 1;indices[1] = slot - gnx; indices[2] = slot - gnx + 1;
      indices[3] = slot - 1  ;    indices[4] = slot;       indices[5] = slot + 1;
      indices[6] = slot + gnx - 1;indices[7] = slot + gnx; indices[8] = slot + gnx + 1;
      for ( k=0; k<9; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,9,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "DAGetColoring2dBoxMC" 
/*@C
      DAGetColoring2dBoxMC - Gets the coloring required for computing the Jacobian via
          finite differences on a function defined using the nine point stencil
          on a two dimensional grid. 

          This is a utility routine that will change over time, not part of the 
          core PETSc package.

  Input Parameter:
.    da - the distributed array

  Output Parameters:
.    coloring - matrix coloring for compute Jacobians
.    J  - matrix with the correct nonzero structured 
            (obviously without the correct Jacobian values)

.seealso: DAGetColoring2dBox()

@*/
int DAGetColoring2dBoxMC(DA da,ISColoring *coloring,Mat *J)
{
  int      ierr, xs,ys,nx,ny,*colors,i,j,ii,slot,gxs,gys,ys1,ny1,nx1,gnx,gny;           
  int      m,n,dim,w,s,N,*indices,*gindices,k,xs1,nc,ng,*cols;
  MPI_Comm comm;
  Scalar   *values;
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
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s,0); CHKERRQ(ierr);
  if (dim != 2) SETERRQ(1,0,"2d only");
  if (s != 1)   SETERRQ(1,0,"Stencil width 1 only");
  /* also no support for periodic boundary conditions */
  nc     = w;
  values = (Scalar *) PetscMalloc( 9*nc*nc*sizeof(Scalar) ); CHKPTRQ(values);
  PetscMemzero(values,9*nc*nc*sizeof(Scalar));
  cols    = (int *) PetscMalloc( nc*sizeof(int) ); CHKPTRQ(cols);
  indices = (int *) PetscMalloc( 9*nc*nc*sizeof(int) ); CHKPTRQ(cols);
  N       = m*n;

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm); CHKERRQ(ierr);


  /* create the coloring */
  colors = (int *) PetscMalloc( nc*nx*ny*sizeof(int) ); CHKPTRQ(colors);
  ii = 0;
  for ( j=ys; j<ys+ny; j++ ) {
    for ( i=xs; i<xs+nx; i++ ) {
      for ( k=0; k<nc; k++ ) {
        colors[ii++] = k + nc*((i % 3) + 3*(j % 3));
      }
    }
  }
  ierr = ISColoringCreate(comm,nc*nx*ny,colors,coloring); CHKERRQ(ierr);
  PetscFree(colors);

  /* create empty Jacobian matrix */
  ierr = MatCreateMPIAIJ(comm,nc*nx*ny,nc*nx*ny,PETSC_DECIDE,PETSC_DECIDE,9*nc,0,0,0,J);CHKERRQ(ierr);  

  ierr = DAGetGlobalIndices(da,&ng,&gindices);
  ierr = MatSetLocalToGlobalMapping(*J,ng,gindices); CHKERRQ(ierr);

  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global PETSc ordering
  */
  
  /* fill up Jacobian for left edge */
  if (xs == 0) {
    ys1 = ys;
    ny1 = ny;
    /* lower left corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs - gxs + gnx*(ys - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0+k]    = slot;         indices[1*nc+k] = slot + 1; 
        indices[2*nc+k] = slot + gnx;   indices[3*nc+k] = slot + gnx + 1;
        cols[k]         = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    /* upper left corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs - gxs + gnx*(ys + (ny - 1) - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0] = slot - gnx;   indices[1] = slot - gnx + 1;
        indices[2] = slot;         indices[3] = slot + 1;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs - gxs + gnx*(j - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0] = slot - gnx;   indices[1] = slot - gnx + 1;
        indices[2] = slot;         indices[3] = slot + 1;
        indices[4] = slot + gnx;   indices[5] = slot + gnx + 1;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);  
    }
  }      

  /* fill up Jacobian for right edge */
  if (xs + nx == m) {
    ys1 = ys;
    ny1 = ny;
    /* lower right corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0] = slot - 1;         indices[1] = slot; 
        indices[2] = slot + gnx - 1;   indices[3] = slot + gnx;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    /* upper right corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys + (ny - 1) - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;
        indices[2] = slot-1;           indices[3] = slot;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,4*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs + (nx - 1) - gxs + gnx*(j - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;
        indices[2] = slot - 1;         indices[3] = slot;
        indices[4] = slot + gnx - 1;   indices[5] = slot + gnx;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);   
    }
  }   

  /* fill up Jacobian for bottom */
  if (ys == 0) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys - gys);
      for ( k=0; k<nc; k++ ) {
        indices[0] = slot - 1;       indices[1] = slot;        indices[2] = slot + 1;
        indices[3] = slot + gnx - 1; indices[4] = slot + gnx;  indices[5] = slot + gnx + 1;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for top */
  if (ys + ny == n) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys + (ny - 1) - gys);
      for ( k=0; k<nc; k++ ) {    
        indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;  indices[2] = slot - gnx + 1;
        indices[3] = slot - 1;         indices[4] = slot;        indices[5] = slot  + 1;
        cols[0]    = slot;
      }
      ierr = MatSetValuesLocal(*J,nc,cols,6*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for interior of grid */
  nx1 = nx;
  ny1 = ny;
  if (xs == 0)    {xs++; nx1--;}
  if (ys == 0)    {ys++; ny1--;}
  if (xs+nx1 == m) {nx1--;}
  if (ys+ny1 == n) {ny1--;}
  for ( i=xs; i<xs+nx1; i++ ) {
    for ( j=ys; j<ys+ny1; j++ ) {
      slot = i - gxs + gnx*(j - gys);
      for ( k=0; k<nc; k++ ) {    
        indices[0] = slot - gnx - 1;indices[1] = slot - gnx; indices[2] = slot - gnx + 1;
        indices[3] = slot - 1  ;    indices[4] = slot;       indices[5] = slot + 1;
        indices[6] = slot + gnx - 1;indices[7] = slot + gnx; indices[8] = slot + gnx + 1;
        cols[0]    = slot;
    }
      ierr = MatSetValuesLocal(*J,nc,cols,9*nc,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  PetscFree(values); 
  PetscFree(cols);
  PetscFree(indices);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  return 0;
}


