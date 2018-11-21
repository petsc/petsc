#include <petscdm.h>
#include <petscdmda.h>


/*
  Test to verify that the zero order scalar evaluation of forward mode AD yields the same
  result as evaluating the PDE residual.

  Input parameters:
  da   - distributed array upon which variables are defined
  f    - values of residual evaluation
  u    - independent variable values upon which to evaluate
  tag  - identifier for active session
  view - sanity check: toggle whether to print comparisons of nonzero values (to check
         there are some)

  TODO: Move this test to Jacobian driver, to avoid unravelling into 1-array
*/
PetscErrorCode TestZOS2d(DM da,PetscScalar **f,PetscScalar **u,PetscBool view)
{
  PetscErrorCode ierr;
  PetscInt       m,n,gxs,gys,gxm,gym,i,j,k = 0;
  PetscScalar    diff = 0,norm = 0,*u_vec,*fz;
  MPI_Comm       comm = MPI_COMM_WORLD;

  PetscFunctionBegin;

  /* Get extent of region owned by processor */
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  m = gxm*gym;
  n = m;

  /* Convert to a 1-array */
  ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_vec[k++] = u[j][i];
  }
  k = 0;

  /* Zero order scalar evaluation vs. calling RHS function */
  ierr = PetscMalloc1(m,&fz);CHKERRQ(ierr);
  zos_forward(1,m,n,0,u_vec,fz);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      if ((view) && ((fabs(f[j][i]) > 1.e-16) || (fabs(fz[k]) > 1.e-16)))
        PetscPrintf(comm,"(%2d,%2d): F_rhs = %+.4e, F_zos = %+.4e\n",j,i,f[j][i],fz[k]);
      diff += (f[j][i]-fz[k])*(f[j][i]-fz[k]);k++;
      norm += f[j][i]*f[j][i];
    }
  }
  diff = sqrt(diff);
  norm = diff/sqrt(norm);
  PetscPrintf(comm,"    ----- Testing Zero Order evaluation -----\n");
  PetscPrintf(comm,"    ||Fzos - Frhs||_2/||Frhs||_2 = %.4e, ||Fzos - Frhs||_2 = %.4e\n",norm,diff);
  ierr = PetscFree(fz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

