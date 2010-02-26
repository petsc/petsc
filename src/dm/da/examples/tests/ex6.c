      
static char help[] = "Tests various 3-dimensional DA routines.\n\n";

#include "petscda.h"
#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       M = 3,N = 5,P=3,s=1,w=2,nloc,l,i,j,k,kk,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode ierr;
  PetscInt       Xs,Xm,Ys,Ym,Zs,Zm,iloc,*ltog,*iglobal;
  PetscInt       *lx = PETSC_NULL,*ly = PETSC_NULL,*lz = PETSC_NULL;
  PetscTruth     test_order = PETSC_FALSE;
  DA             da;
  PetscViewer    viewer;
  Vec            local,global;
  PetscScalar    value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  AO             ao;
  PetscTruth     flg = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,300,&viewer);CHKERRQ(ierr);

  /* Read options */  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-test_order",&test_order,PETSC_NULL);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-distribute",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRQ(1,"Must set -m option with -distribute option");
    ierr = PetscMalloc(m*sizeof(PetscInt),&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRQ(1,"Must set -n option with -distribute option");
    ierr = PetscMalloc(n*sizeof(PetscInt),&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
    if (p == PETSC_DECIDE) SETERRQ(1,"Must set -p option with -distribute option");
    ierr = PetscMalloc(p*sizeof(PetscInt),&lz);CHKERRQ(ierr);
    for (i=0; i<p-1; i++) { lz[i] = 2;}
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,w,s,
                    lx,ly,lz,&da);CHKERRQ(ierr);
  if (lx) {
    ierr = PetscFree(lx);CHKERRQ(ierr);
    ierr = PetscFree(ly);CHKERRQ(ierr);
    ierr = PetscFree(lz);CHKERRQ(ierr);
  }
  ierr = DAView(da,viewer);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(global,value);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  value = rank;
  ierr = VecScale(local,value);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  if (!test_order) { /* turn off printing when testing ordering mappings */
    if (M*N*P<40) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vector:\n");CHKERRQ(ierr);
      ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    }
  }

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-local_print",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer sviewer;
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
    ierr = VecView(local,sviewer);CHKERRQ(ierr); 
    ierr = PetscViewerRestoreSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (test_order) {
    ierr = DAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
    ierr = DAGetGlobalIndices(da,&nloc,&ltog);CHKERRQ(ierr);
    ierr = DAGetAO(da,&ao);CHKERRQ(ierr);
    /* ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = PetscMalloc(nloc*sizeof(PetscInt),&iglobal);CHKERRQ(ierr);

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DA ordering of grid points */
    kk = 0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs); 
          for (l=0; l<w; l++) {
            iglobal[kk++] = ltog[iloc+l];
          }
        }
      }
    } 

    /* Map this to the application ordering (which for DAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y, then z) */
    ierr = AOPetscToApplication(ao,nloc,iglobal);CHKERRQ(ierr); 

    /* Then map the application ordering back to the PETSc DA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal);CHKERRQ(ierr); 

    /* Verify the mappings */
    kk=0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs); 
          for (l=0; l<w; l++) {
            if (iglobal[kk] != ltog[iloc+l]) {
              ierr = PetscPrintf(MPI_COMM_WORLD,"[%D] Problem with mapping: z=%D, j=%D, i=%D, l=%D, petsc1=%D, petsc2=%D\n",
                      rank,k,j,i,l,ltog[iloc+l],iglobal[kk]);
            }
            kk++;
          }
        }
      }
    }
    ierr = PetscFree(iglobal);CHKERRQ(ierr);
  } 

  /* Free memory */
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
  




















