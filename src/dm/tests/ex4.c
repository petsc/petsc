
static char help[] = "Tests various 2-dimensional DMDA routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt      rank;
  PetscInt         M = 10,N = 8,m = PETSC_DECIDE;
  PetscInt         s =2,w=2,n = PETSC_DECIDE,nloc,l,i,j,kk;
  PetscInt         Xs,Xm,Ys,Ym,iloc,*iglobal;
  const PetscInt   *ltog;
  PetscInt         *lx       = NULL,*ly = NULL;
  PetscBool        testorder = PETSC_FALSE,flg;
  DMBoundaryType   bx        = DM_BOUNDARY_NONE,by= DM_BOUNDARY_NONE;
  DM               da;
  PetscViewer      viewer;
  Vec              local,global;
  PetscScalar      value;
  DMDAStencilType  st = DMDA_STENCIL_BOX;
  AO               ao;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,400,&viewer));

  /* Readoptions */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-NX",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-NY",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-xperiodic",&flg,NULL)); if (flg) bx = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-yperiodic",&flg,NULL)); if (flg) by = DM_BOUNDARY_PERIODIC;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-xghosted",&flg,NULL)); if (flg) bx = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-yghosted",&flg,NULL)); if (flg) by = DM_BOUNDARY_GHOSTED;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL)); if (flg) st = DMDA_STENCIL_STAR;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-box",&flg,NULL)); if (flg) st = DMDA_STENCIL_BOX;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testorder",&testorder,NULL));
  /*
      Test putting two nodes in x and y on each processor, exact last processor
      in x and y gets the rest.
  */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-distribute",&flg,NULL));
  if (flg) {
    PetscCheck(m != PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -m option with -distribute option");
    PetscCall(PetscMalloc1(m,&lx));
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    PetscCheck(n != PETSC_DECIDE,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must set -n option with -distribute option");
    PetscCall(PetscMalloc1(n,&ly));
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
  }

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,st,M,N,m,n,w,s,lx,ly,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(PetscFree(lx));
  PetscCall(PetscFree(ly));

  PetscCall(DMView(da,viewer));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  PetscCall(VecSet(global,value));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  /* Scale local vectors according to processor rank; pass to global vector */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank;
  PetscCall(VecScale(local,value));
  PetscCall(DMLocalToGlobalBegin(da,local,INSERT_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(da,local,INSERT_VALUES,global));

  if (!testorder) { /* turn off printing when testing ordering mappings */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vectors:\n"));
    PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n\n"));
  }

  /* Send ghost points to local vectors */
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-local_print",&flg,NULL));
  if (flg) {
    PetscViewer sviewer;

    PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    PetscCall(VecView(local,sviewer));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Tests mappings between application/PETSc orderings */
  if (testorder) {
    ISLocalToGlobalMapping ltogm;

    PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
    PetscCall(ISLocalToGlobalMappingGetSize(ltogm,&nloc));
    PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&ltog));
    PetscCall(DMDAGetGhostCorners(da,&Xs,&Ys,NULL,&Xm,&Ym,NULL));
    PetscCall(DMDAGetAO(da,&ao));
    PetscCall(PetscMalloc1(nloc,&iglobal));

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DMDA ordering of grid points */
    kk = 0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs);
        for (l=0; l<w; l++) {
          iglobal[kk++] = ltog[iloc+l];
        }
      }
    }

    /* Map this to the application ordering (which for DMDAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y) */
    PetscCall(AOPetscToApplication(ao,nloc,iglobal));

    /* Then map the application ordering back to the PETSc DMDA ordering */
    PetscCall(AOApplicationToPetsc(ao,nloc,iglobal));

    /* Verify the mappings */
    kk=0;
    for (j=Ys; j<Ys+Ym; j++) {
      for (i=Xs; i<Xs+Xm; i++) {
        iloc = w*((j-Ys)*Xm + i-Xs);
        for (l=0; l<w; l++) {
          if (iglobal[kk] != ltog[iloc+l]) {
            PetscCall(PetscFPrintf(PETSC_COMM_SELF,stdout,"[%d] Problem with mapping: j=%" PetscInt_FMT ", i=%" PetscInt_FMT ", l=%" PetscInt_FMT ", petsc1=%" PetscInt_FMT ", petsc2=%" PetscInt_FMT "\n",rank,j,i,l,ltog[iloc+l],iglobal[kk]));
          }
          kk++;
        }
      }
    }
    PetscCall(PetscFree(iglobal));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog));
  }

  /* Free memory */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      args: -nox
      filter: grep -v -i Object
      requires: x

   test:
      suffix: 2
      args: -testorder -nox
      requires: x

TEST*/
