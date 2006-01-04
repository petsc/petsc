#define PETSCDM_DLL

/*
  Code for interpolating between grids represented by DAs
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/
#include "petscmg.h"

#undef __FUNCT__  
#define __FUNCT__ "DMGetInterpolationScale"
PetscErrorCode PETSCDM_DLLEXPORT DMGetInterpolationScale(DM dac,DM daf,Mat mat,Vec *scale)
{
  PetscErrorCode ierr;
  Vec            fine;
  PetscScalar    one = 1.0;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(daf,&fine);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac,scale);CHKERRQ(ierr);
  ierr = VecSet(fine,one);CHKERRQ(ierr);
  ierr = MatRestrict(mat,fine,*scale);CHKERRQ(ierr);
  ierr = VecDestroy(fine);CHKERRQ(ierr);
  ierr = VecReciprocal(*scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_1D_Q1"
PetscErrorCode DAGetInterpolation_1D_Q1(DA dac,DA daf,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       i,i_start,m_f,Mx,*idx_f;
  PetscInt       m_ghost,*idx_c,m_ghost_c;
  PetscInt       row,col,i_start_ghost,mx,m_c,nc,ratio;
  PetscInt       i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar    v[2],x,*coors = 0,*ccoors;
  Mat            mat;
  DAPeriodicType pt;
  Vec            vcoors,cvcoors;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (pt == DA_XPERIODIC) {
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }

  ierr = DAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreate(dac->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f,m_c,mx,Mx);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,2,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  if (!DAXPeriodic(pt)){ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  ierr = DAGetCoordinates(daf,&vcoors);CHKERRQ(ierr);
  if (vcoors) {
    ierr = DAGetGhostedCoordinates(dac,&cvcoors);CHKERRQ(ierr);
    ierr = DAVecGetArray(daf->da_coordinates,vcoors,&coors);CHKERRQ(ierr);
    ierr = DAVecGetArray(dac->da_coordinates,cvcoors,&ccoors);CHKERRQ(ierr);
  }
  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and then to PETSc global numbering */
    row    = idx_f[dof*(i-i_start_ghost)]/dof;

    i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
    if (i_c < i_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

    /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x direction; since they have no right neighbor
    */
    if (coors) {
      x = (coors[i] - ccoors[i_c]); 
      /* only access the next coors point if we know there is one */
      /* note this is dangerous because x may not exactly equal ZERO */
      if (PetscAbsScalar(x) != 0.0) x = x/(ccoors[i_c+1] - ccoors[i_c]);
    } else {
      x  = ((double)(i - i_c*ratio))/((double)ratio);
    }
    nc = 0;
      /* one left and below; or we are right on it */
    col      = dof*(i_c-i_start_ghost_c);
    cols[nc] = idx_c[col]/dof; 
    v[nc++]  = - x + 1.0;
    /* one right? */
    if (i_c*ratio != i) { 
      cols[nc] = idx_c[col+dof]/dof;
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
  }
  if (vcoors) {
    ierr = DAVecRestoreArray(daf->da_coordinates,vcoors,&coors);CHKERRQ(ierr);
    ierr = DAVecRestoreArray(dac->da_coordinates,cvcoors,&ccoors);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(5*m_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_1D_Q0"
PetscErrorCode DAGetInterpolation_1D_Q0(DA dac,DA daf,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       i,i_start,m_f,Mx,*idx_f;
  PetscInt       m_ghost,*idx_c,m_ghost_c;
  PetscInt       row,col,i_start_ghost,mx,m_c,nc,ratio;
  PetscInt       i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar    v[2],x;
  Mat            mat;
  DAPeriodicType pt;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (pt == DA_XPERIODIC) {
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }

  ierr = DAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreate(dac->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f,m_c,mx,Mx);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,2,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  if (!DAXPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and then to PETSc global numbering */
    row    = idx_f[dof*(i-i_start_ghost)]/dof;

    i_c = (i/ratio);    /* coarse grid node to left of fine grid node */

    /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x direction; since they have no right neighbor
    */
    x  = ((double)(i - i_c*ratio))/((double)ratio);
    nc = 0;
      /* one left and below; or we are right on it */
    col      = dof*(i_c-i_start_ghost_c);
    cols[nc] = idx_c[col]/dof; 
    v[nc++]  = - x + 1.0;
    /* one right? */
    if (i_c*ratio != i) { 
      cols[nc] = idx_c[col+dof]/dof;
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(5*m_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_2D_Q1"
PetscErrorCode DAGetInterpolation_2D_Q1(DA dac,DA daf,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  PetscInt       m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,*dnz,*onz;
  PetscInt       row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
  PetscInt       i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col_shift,col_scale;
  PetscMPIInt    size_c,size_f,rank_f;
  PetscScalar    v[4],x,y;
  Mat            mat;
  DAPeriodicType pt;
  DACoor2d       **coors = 0,**ccoors;
  Vec            vcoors,cvcoors;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (DAXPeriodic(pt)){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (DAYPeriodic(pt)){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }


  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /*
     Used for handling a coarse DA that lives on 1/4 the processors of the fine DA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the 
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr = MPI_Comm_size(dac->comm,&size_c);CHKERRQ(ierr);
  ierr = MPI_Comm_size(daf->comm,&size_f);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(daf->comm,&rank_f);CHKERRQ(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);

  ierr = MatPreallocateInitialize(daf->comm,m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col]/dof; 
      /* one right and below */
      if (i_c*ratioi != i) { 
        cols[nc++] = col_shift + idx_c[col+dof]/dof;
      }
      /* one left and above */
      if (j_c*ratioj != j) { 
        cols[nc++] = col_shift + idx_c[col+m_ghost_c*dof]/dof;
      }
      /* one right and above */
      if (j_c*ratioi != j && i_c*ratioj != i) { 
        cols[nc++] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof;
      }
      ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreate(daf->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (!DAXPeriodic(pt) && !DAYPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  ierr = DAGetCoordinates(daf,&vcoors);CHKERRQ(ierr);
  if (vcoors) {
    ierr = DAGetGhostedCoordinates(dac,&cvcoors);CHKERRQ(ierr);
    ierr = DAVecGetArray(daf->da_coordinates,vcoors,&coors);CHKERRQ(ierr);
    ierr = DAVecGetArray(dac->da_coordinates,cvcoors,&ccoors);CHKERRQ(ierr);
  }

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      if (coors) {
        /* only access the next coors point if we know there is one */
        /* note this is dangerous because x may not exactly equal ZERO */
        x = (coors[j][i].x - ccoors[j_c][i_c].x);
        if (PetscAbsScalar(x) != 0.0) x = x/(ccoors[j_c][i_c+1].x - ccoors[j_c][i_c].x);
        y = (coors[j][i].y - ccoors[j_c][i_c].y);
        if (PetscAbsScalar(y) != 0.0) y = y/(ccoors[j_c+1][i_c].y - ccoors[j_c][i_c].y);
      } else {
        x  = ((double)(i - i_c*ratioi))/((double)ratioi);
        y  = ((double)(j - j_c*ratioj))/((double)ratioj);
      }
      nc = 0;
      /* one left and below; or we are right on it */
      col      = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc] = col_shift + idx_c[col]/dof; 
      v[nc++]  = x*y - x - y + 1.0;
      /* one right and below */
      if (i_c*ratioi != i) { 
        cols[nc] = col_shift + idx_c[col+dof]/dof;
        v[nc++]  = -x*y + x;
      }
      /* one left and above */
      if (j_c*ratioj != j) { 
        cols[nc] = col_shift + idx_c[col+m_ghost_c*dof]/dof;
        v[nc++]  = -x*y + y;
      }
      /* one right and above */
      if (j_c*ratioj != j && i_c*ratioi != i) { 
        cols[nc] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof;
        v[nc++]  = x*y;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  if (vcoors) {
    ierr = DAVecRestoreArray(daf->da_coordinates,vcoors,&coors);CHKERRQ(ierr);
    ierr = DAVecRestoreArray(dac->da_coordinates,cvcoors,&ccoors);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13*m_f*n_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
       Contributed by Andrei Draganescu <aidraga@sandia.gov>
*/
#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_2D_Q0"
PetscErrorCode DAGetInterpolation_2D_Q0(DA dac,DA daf,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  PetscInt       m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,*dnz,*onz;
  PetscInt       row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
  PetscInt       i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col_shift,col_scale;
  PetscMPIInt    size_c,size_f,rank_f;
  PetscScalar    v[4];
  Mat            mat;
  DAPeriodicType pt;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (DAXPeriodic(pt)) SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in x");
  if (DAYPeriodic(pt)) SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in y");
  ratioi = mx/Mx;
  ratioj = my/My;
  if (ratioi*Mx != mx) SETERRQ(PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in x");
  if (ratioj*My != my) SETERRQ(PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in y");
  if (ratioi != 2) SETERRQ(PETSC_ERR_ARG_WRONG,"Coarsening factor in x must be 2");
  if (ratioj != 2) SETERRQ(PETSC_ERR_ARG_WRONG,"Coarsening factor in y must be 2");

  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /*
     Used for handling a coarse DA that lives on 1/4 the processors of the fine DA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the 
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr = MPI_Comm_size(dac->comm,&size_c);CHKERRQ(ierr);
  ierr = MPI_Comm_size(daf->comm,&size_f);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(daf->comm,&rank_f);CHKERRQ(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);

  ierr = MatPreallocateInitialize(daf->comm,m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col]/dof; 
      ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreate(daf->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (!DAXPeriodic(pt) && !DAYPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */
      nc = 0;
      /* one left and below; or we are right on it */
      col      = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc] = col_shift + idx_c[col]/dof; 
      v[nc++]  = 1.0;
     
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13*m_f*n_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_3D_Q1"
PetscErrorCode DAGetInterpolation_3D_Q1(DA dac,DA daf,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof,l;
  PetscInt       m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,Mz,mz;
  PetscInt       row,col,i_start_ghost,j_start_ghost,cols[8],mx,m_c,my,nc,ratioi,ratioj,ratiok;
  PetscInt       i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  PetscInt       l_start,p_f,l_start_ghost,p_ghost,l_start_c,p_c;
  PetscInt       l_start_ghost_c,p_ghost_c,l_c,*dnz,*onz;
  PetscScalar    v[8],x,y,z;
  Mat            mat;
  DAPeriodicType pt;
  DACoor3d       ***coors = 0,***ccoors;
  Vec            vcoors,cvcoors;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,&Mz,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,&mz,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (mx == Mx) {
    ratioi = 1;
  } else if (DAXPeriodic(pt)){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (my == My) {
    ratioj = 1;
  } else if (DAYPeriodic(pt)){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }
  if (mz == Mz) {
    ratiok = 1;
  } else if (DAZPeriodic(pt)){
    ratiok = mz/Mz;
    if (ratiok*Mz != mz) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: mz/Mz  must be integer: mz %D Mz %D",mz,Mz);
  } else {
    ratiok = (mz-1)/(Mz-1);
    if (ratiok*(Mz-1) != mz-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mz - 1)/(Mz - 1) must be integer: mz %D Mz %D",mz,Mz);
  }

  ierr = DAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix, determining exact preallocation */
  ierr = MatPreallocateInitialize(dac->comm,m_f*n_f*p_f,m_c*n_c*p_c,dnz,onz);CHKERRQ(ierr);
  /* loop over local fine grid nodes counting interpolating points */
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
        i_c = (i/ratioi);
        j_c = (j/ratioj);
        l_c = (l/ratiok);
        if (l_c < l_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
          l_start %D l_c %D l_start_ghost_c %D",l_start,l_c,l_start_ghost_c);
        if (j_c < j_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
          j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
        if (i_c < i_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

        /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
        */
        nc       = 0;
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc++] = idx_c[col]/dof; 
        if (i_c*ratioi != i) { 
          cols[nc++] = idx_c[col+dof]/dof;
        }
        if (j_c*ratioj != j) { 
          cols[nc++] = idx_c[col+m_ghost_c*dof]/dof;
        }
        if (l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof;
        }
        if (j_c*ratioj != j && i_c*ratioi != i) { 
          cols[nc++] = idx_c[col+(m_ghost_c+1)*dof]/dof;
        }
        if (j_c*ratioj != j && l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;
        }
        if (i_c*ratioi != i && l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof;
        }
        if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof;
        }
        ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatCreate(dac->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f*p_f,m_c*n_c*p_c,mx*my*mz,Mx*My*Mz);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (!DAXPeriodic(pt) && !DAYPeriodic(pt) && !DAZPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  ierr = DAGetCoordinates(daf,&vcoors);CHKERRQ(ierr);
  if (vcoors) {
    ierr = DAGetGhostedCoordinates(dac,&cvcoors);CHKERRQ(ierr);
    ierr = DAVecGetArray(daf->da_coordinates,vcoors,&coors);CHKERRQ(ierr);
    ierr = DAVecGetArray(dac->da_coordinates,cvcoors,&ccoors);CHKERRQ(ierr);
  }

  /* loop over local fine grid nodes setting interpolation for those*/
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

        i_c = (i/ratioi);
        j_c = (j/ratioj);
        l_c = (l/ratiok);

        /* 
           Only include those interpolation points that are truly 
           nonzero. Note this is very important for final grid lines
           in x and y directions; since they have no right/top neighbors
        */
	if (coors) {
	  /* only access the next coors point if we know there is one */
	  /* note this is dangerous because x may not exactly equal ZERO */
	  x = (coors[l][j][i].x - ccoors[l_c][j_c][i_c].x);
	  if (PetscAbsScalar(x) != 0.0) x = x/(ccoors[l_c][j_c][i_c+1].x - ccoors[l_c][j_c][i_c].x);
	  y = (coors[l][j][i].y - ccoors[l_c][j_c][i_c].y);
	  if (PetscAbsScalar(y) != 0.0) y = y/(ccoors[l_c][j_c+1][i_c].y - ccoors[l_c][j_c][i_c].y);
	  z = (coors[l][j][i].z - ccoors[l_c][j_c][i_c].z);
	  if (PetscAbsScalar(z) != 0.0) z = z/(ccoors[l_c+1][j_c][i_c].z - ccoors[l_c][j_c][i_c].z);
	} else {
	  x  = ((double)(i - i_c*ratioi))/((double)ratioi);
	  y  = ((double)(j - j_c*ratioj))/((double)ratioj);
	  z  = ((double)(l - l_c*ratiok))/((double)ratiok);
        }
        nc = 0;
        /* one left and below; or we are right on it */
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));

        cols[nc] = idx_c[col]/dof; 
        v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));

        if (i_c*ratioi != i) { 
          cols[nc] = idx_c[col+dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (j_c*ratioj != j) { 
          cols[nc] = idx_c[col+m_ghost_c*dof]/dof;
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (l_c*ratiok != l) { 
          cols[nc] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof;
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (j_c*ratioj != j && i_c*ratioi != i) { 
          cols[nc] = idx_c[col+(m_ghost_c+1)*dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (j_c*ratioj != j && l_c*ratiok != l) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (i_c*ratioi != i && l_c*ratiok != l) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
        }
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
  }
  if (vcoors) {
    ierr = DAVecRestoreArray(daf->da_coordinates,vcoors,&coors);CHKERRQ(ierr);
    ierr = DAVecRestoreArray(dac->da_coordinates,cvcoors,&ccoors);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13*m_f*n_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation"
/*@C
   DAGetInterpolation - Gets an interpolation matrix that maps between 
   grids associated with two DAs.

   Collective on DA

   Input Parameters:
+  dac - the coarse grid DA
-  daf - the fine grid DA

   Output Parameters:
+  A - the interpolation matrix
-  scale - a scaling vector used to scale the coarse grid restricted vector before applying the 
           grid function or grid Jacobian to it.

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: DARefine(), DAGetInjection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetInterpolation(DA dac,DA daf,Mat *A,Vec *scale)
{
  PetscErrorCode ierr;
  PetscInt       dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DAPeriodicType wrapc,wrapf;
  DAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DA_COOKIE,1);
  PetscValidHeaderSpecific(daf,DA_COOKIE,2);
  PetscValidPointer(A,3);
  if (scale) PetscValidPointer(scale,4);

  ierr = DAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&wrapc,&stc);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&wrapf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Dimensions of DA do not match %D %D",dimc,dimf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(PETSC_ERR_ARG_INCOMP,"DOF of DA do not match %D %D",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Stencil width of DA do not match %D %D",sc,sf);CHKERRQ(ierr);
  if (wrapc != wrapf) SETERRQ(PETSC_ERR_ARG_INCOMP,"Periodic type different in two DAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(PETSC_ERR_ARG_INCOMP,"Stencil type different in two DAs");CHKERRQ(ierr);
  if (Mc < 2 && Mf > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2 && Nf > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2 && Pf > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in z direction");

  if (dac->interptype == DA_Q1){
    if (dimc == 1){
      ierr = DAGetInterpolation_1D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 2){
      ierr = DAGetInterpolation_2D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 3){
      ierr = DAGetInterpolation_3D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else {
      SETERRQ2(PETSC_ERR_SUP,"No support for this DA dimension %D for interpolation type %d",dimc,(int)dac->interptype);
    }
  } else if (dac->interptype == DA_Q0){
    if (dimc == 1){
      ierr = DAGetInterpolation_1D_Q0(dac,daf,A);CHKERRQ(ierr);
    } if (dimc == 2){
       ierr = DAGetInterpolation_2D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else {
      SETERRQ2(PETSC_ERR_SUP,"No support for this DA dimension %D for interpolation type %d",dimc,(int)dac->interptype);
    }
  }
  if (scale) {
    ierr = DMGetInterpolationScale((DM)dac,(DM)daf,*A,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "DAGetInjection_2D"
PetscErrorCode DAGetInjection_2D(DA dac,DA daf,VecScatter *inject)
{
  PetscErrorCode ierr;
  PetscInt       i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  PetscInt       m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c;
  PetscInt       row,i_start_ghost,j_start_ghost,mx,m_c,my,nc,ratioi,ratioj;
  PetscInt       i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  PetscInt       *cols;
  DAPeriodicType pt;
  Vec            vecf,vecc;
  IS             isf;

  PetscFunctionBegin;

  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (DAXPeriodic(pt)){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (DAYPeriodic(pt)){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }


  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);


  /* loop over local fine grid nodes setting interpolation for those*/
  nc = 0;
  ierr = PetscMalloc(n_f*m_f*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_ERR_ARG_INCOMP,"Processor's coarse DA must lie over fine DA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      if (i_c*ratioi == i && j_c*ratioj == j) { 
	/* convert to local "natural" numbering and then to PETSc global numbering */
	row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];
        cols[nc++] = row; 
      }
    }
  }

  ierr = ISCreateBlock(daf->comm,dof,nc,cols,&isf);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,PETSC_NULL,inject);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(isf);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInjection"
/*@C
   DAGetInjection - Gets an injection matrix that maps between 
   grids associated with two DAs.

   Collective on DA

   Input Parameters:
+  dac - the coarse grid DA
-  daf - the fine grid DA

   Output Parameters:
.  inject - the injection scatter

   Level: intermediate

.keywords: interpolation, restriction, multigrid, injection 

.seealso: DARefine(), DAGetInterpolation()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetInjection(DA dac,DA daf,VecScatter *inject)
{
  PetscErrorCode ierr;
  PetscInt       dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DAPeriodicType wrapc,wrapf;
  DAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DA_COOKIE,1);
  PetscValidHeaderSpecific(daf,DA_COOKIE,2);
  PetscValidPointer(inject,3);

  ierr = DAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&wrapc,&stc);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&wrapf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Dimensions of DA do not match %D %D",dimc,dimf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(PETSC_ERR_ARG_INCOMP,"DOF of DA do not match %D %D",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Stencil width of DA do not match %D %D",sc,sf);CHKERRQ(ierr);
  if (wrapc != wrapf) SETERRQ(PETSC_ERR_ARG_INCOMP,"Periodic type different in two DAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(PETSC_ERR_ARG_INCOMP,"Stencil type different in two DAs");CHKERRQ(ierr);
  if (Mc < 2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in z direction");

  if (dimc == 2){
    ierr = DAGetInjection_2D(dac,daf,inject);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"No support for this DA dimension %D",dimc);
  }
  PetscFunctionReturn(0);
} 

