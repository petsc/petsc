static char help[] = "Solves -Laplacian u - exp(u) = 0,  0 < x < 1 using GPU\n\n";
/*
   Same as ex47.c except it also uses the GPU to evaluate the function
*/

#include <petscdmda.h>
#include <petscsnes.h>
#include <petsccusp.h>

extern PetscErrorCode ComputeFunction(SNES,Vec,Vec,void*), ComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscBool  useCUSP = PETSC_FALSE;

int main(int argc,char **argv) 
{
  SNES           snes; 
  Vec            x,f;  
  Mat            J;
  DM             da;
  PetscErrorCode ierr;
  char           *tmp,typeName[256];
  PetscBool      flg;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-dm_vec_type",typeName,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrstr(typeName,"cusp",&tmp);CHKERRQ(ierr);
    if (tmp) useCUSP = PETSC_TRUE;
  }

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-8,1,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x); VecDuplicate(x,&f);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATAIJ,&J);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,f,ComputeFunction,da);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,ComputeJacobian,da);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}

struct ApplyStencil
{
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		/* f = (2*x_i - x_(i+1) - x_(i-1))/h - h*exp(x_i) */
	     thrust::get<0>(t) = 1;
		if ((thrust::get<4>(t) > 0) && (thrust::get<4>(t) < thrust::get<5>(t)-1)) {
		  thrust::get<0>(t) = (2.0*thrust::get<1>(t) - thrust::get<2>(t) - thrust::get<3>(t)) / (thrust::get<6>(t)) - (thrust::get<6>(t))*exp(thrust::get<1>(t));
		} else if (thrust::get<4>(t) == 0) {
		  thrust::get<0>(t) = thrust::get<1>(t) / (thrust::get<6>(t));
		} else if (thrust::get<4>(t) == thrust::get<5>(t)-1) {
		  thrust::get<0>(t) = thrust::get<1>(t) / (thrust::get<6>(t));
		} 
		
	}
};

PetscErrorCode ComputeFunction(SNES snes,Vec x,Vec f,void *ctx) 
{
  PetscInt       i,Mx,xs,xm,xstartshift,xendshift,fstart;
  PetscScalar    *xx,*ff,hx;
  DM             da = (DM) ctx; 
  Vec            xlocal;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  MPI_Comm       comm;
  CUSPARRAY      *xarray,*farray;

  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx     = 1.0/(PetscReal)(Mx-1);
  ierr = DMGetLocalVector(da,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);

  if (useCUSP) {
    ierr = VecCUSPGetArrayRead(xlocal,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(f,&farray);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (rank) xstartshift = 1; else xstartshift = 0;
    if (rank != size-1) xendshift = 1; else xendshift = 0;
    ierr = VecGetOwnershipRange(f,&fstart,PETSC_NULL);CHKERRQ(ierr);
    try {
      thrust::for_each(
		       thrust::make_zip_iterator(
						 thrust::make_tuple(
								    farray->begin(),
								    xarray->begin()+xstartshift,
								    xarray->begin()+xstartshift + 1,
								    xarray->begin()+xstartshift - 1,
								    thrust::counting_iterator<int>(fstart),
								    thrust::constant_iterator<int>(Mx),
								    thrust::constant_iterator<PetscScalar>(hx))),
		       thrust::make_zip_iterator(
						 thrust::make_tuple(
								    farray->end(),
								    xarray->end()-xendshift,
								    xarray->end()-xendshift + 1,
								    xarray->end()-xendshift - 1,
								    thrust::counting_iterator<int>(fstart) + x->map->n,
								    thrust::constant_iterator<int>(Mx),
								    thrust::constant_iterator<PetscScalar>(hx))),
		       ApplyStencil());
    }
    catch(char* all){
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Thrust is not working\n");CHKERRQ(ierr);
    }
    ierr = VecCUSPRestoreArrayRead(xlocal,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(f,&farray);CHKERRQ(ierr);
  } else {
    ierr = DMDAVecGetArray(da,xlocal,&xx);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,f,&ff);CHKERRQ(ierr);
    ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || i == Mx-1) ff[i] = xx[i]/hx; 
      else  ff[i] =  (2.0*xx[i] - xx[i-1] - xx[i+1])/hx - hx*PetscExpScalar(xx[i]); 
    }
    ierr = DMDAVecRestoreArray(da,xlocal,&xx);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,f,&ff);CHKERRQ(ierr);
  }
    ierr = DMRestoreLocalVector(da,&xlocal);CHKERRQ(ierr);
  //  VecView(x,0);printf("f\n");
  //  VecView(f,0);
  return 0;

}
PetscErrorCode ComputeJacobian(SNES snes,Vec x,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  DM             da = (DM) ctx; 
  PetscInt       i,Mx,xm,xs; 
  PetscScalar    hx,*xx; 
  Vec            xlocal;
  PetscErrorCode ierr;

  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(Mx-1);
  ierr = DMGetLocalVector(da,&xlocal);DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    if (i == 0 || i == Mx-1) { 
      ierr = MatSetValue(*J,i,i,1.0/hx,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValue(*J,i,i-1,-1.0/hx,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(*J,i,i,2.0/hx - hx*PetscExpScalar(xx[i]),INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(*J,i,i+1,-1.0/hx,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  ierr = DMDAVecRestoreArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xlocal);CHKERRQ(ierr);
  return 0;}

