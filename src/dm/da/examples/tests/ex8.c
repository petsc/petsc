      
static char help[] = "Demonstrates generating a slice from a DA Vector.\n\n";

#include "petscda.h"
#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "GenerateSliceScatter"
/*
    Given a DA generates a VecScatter context that will deliver a slice
  of the global vector to each processor. In this example, each processor
  receives the values i=*, j=*, k=rank, i.e. one z plane.

  Note: This code is written assuming only one degree of freedom per node.
  For multiple degrees of freedom per node use ISCreateBlock()
  instead of ISCreateGeneral().
*/
PetscErrorCode GenerateSliceScatter(DA da,VecScatter *scatter,Vec *vslice)
{
  AO             ao;
  PetscInt       M,N,P,nslice,*sliceindices,count,i,j;
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Vec            vglobal;
  IS             isfrom,isto;

  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = DAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0);CHKERRQ(ierr);

  /* 
     nslice is number of degrees of freedom in this processors slice
   if there are more processors then z plans the extra processors get 0
   elements in their slice.
  */
  if (rank < P) {nslice = M*N;} else nslice = 0;

  /* 
     Generate the local vector to hold this processors slice
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,nslice,vslice);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&vglobal);CHKERRQ(ierr);

  /*
       Generate the indices for the slice in the "natural" global ordering
     Note: this is just an example, one could select any subset of nodes 
    on each processor. Just list them in the global natural ordering.

  */
  ierr = PetscMalloc((nslice+1)*sizeof(PetscInt),&sliceindices);CHKERRQ(ierr);
  count = 0;
  if (rank < P) {
    for (j=0; j<N; j++) {
      for (i=0; i<M; i++) {
         sliceindices[count++] = rank*M*N + j*M + i;
      }
    }
  }
  /*
      Convert the indices to the "PETSc" global ordering
  */
  ierr = AOApplicationToPetsc(ao,nslice,sliceindices);CHKERRQ(ierr); 
  
  /* Create the "from" and "to" index set */
  /* This is to scatter from the global vector */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nslice,sliceindices,&isfrom);CHKERRQ(ierr);
  /* This is to gather into the local vector */
  ierr = ISCreateStride(PETSC_COMM_SELF,nslice,0,1,&isto);CHKERRQ(ierr);
  ierr = VecScatterCreate(vglobal,isfrom,*vslice,isto,scatter);CHKERRQ(ierr);
  ierr = ISDestroy(isfrom);CHKERRQ(ierr); 
  ierr = ISDestroy(isto);CHKERRQ(ierr);
  ierr = PetscFree(sliceindices);CHKERRQ(ierr);
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,M = 3,N = 5,P=3,s=1;
  PetscInt       *lx = PETSC_NULL,*ly = PETSC_NULL,*lz = PETSC_NULL;
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;
  DA             da;
  Vec            local,global,vslice;
  PetscScalar    value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  VecScatter     scatter;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Read options */  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,1,s,
                    lx,ly,lz,&da);CHKERRQ(ierr);
  ierr = DAView(da,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);

  ierr = GenerateSliceScatter(da,&scatter,&vslice);CHKERRQ(ierr);

  /* Put the value rank+1 into all locations of vslice and transfer back to global vector */
  value = 1.0 + rank;
  ierr = VecSet(vslice,value);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,vslice,global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,vslice,global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecView(global,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
