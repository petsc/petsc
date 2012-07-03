
static char help[] = "Test MatGetMultiProcBlock() \n\
Reads a PETSc matrix and vector from a file and solves a linear system.\n\n";
/*
  Example:
  mpiexec -n 4 ./ex37 -f <input_file> -nsubcomm 2 -subcomm_view -subcomm_type <1 or 2>
*/

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  KSP            subksp;             
  Mat            A,subA;        
  Vec            x,b,u,subb,subx,subu;           
  PetscViewer    fd;            
  char           file[PETSC_MAX_PATH_LEN];     
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscInt       i,m,n,its;
  PetscReal      norm;
  PetscMPIInt    rank,size;
  MPI_Comm       comm,subcomm;
  PetscSubcomm   psubcomm;
  PetscInt       nsubcomm=1,id;
  PetscScalar    *barray,*xarray,*uarray,*array,one=1.0;
  PetscInt       type=1;

  PetscInitialize(&argc,&args,(char *)0,help); 
  /* Load the matrix */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    
  /* Load the matrix; then destroy the viewer.*/
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr); 

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    
  /* Create rhs vector b */
  ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecSet(b,one);CHKERRQ(ierr);

  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /* Test MatGetMultiProcBlock() */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nsubcomm",&nsubcomm,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-subcomm_type",&type,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscSubcommCreate(comm,&psubcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(psubcomm,nsubcomm);CHKERRQ(ierr);
  if (type == PETSC_SUBCOMM_GENERAL){/* user provides color, subrank and duprank */
    PetscMPIInt color,subrank,duprank,subsize;
    duprank = size-1 - rank;
    subsize = size/nsubcomm;
    if (subsize*nsubcomm != size) SETERRQ2(comm,PETSC_ERR_SUP,"This example requires nsubcomm %D divides nproc %D",nsubcomm,size);
    color   = duprank/subsize;
    subrank = duprank - color*subsize;
    ierr = PetscSubcommSetTypeGeneral(psubcomm,color,subrank,duprank);CHKERRQ(ierr);
  } else if (type == PETSC_SUBCOMM_CONTIGUOUS){
    ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
  } else if(type == PETSC_SUBCOMM_INTERLACED){
    ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
  } else {
    SETERRQ1(psubcomm->parent,PETSC_ERR_SUP,"PetscSubcommType %D is not supported yet",type);
  }
  subcomm = psubcomm->comm;

  ierr = PetscOptionsHasName(PETSC_NULL, "-subcomm_view", &flg);CHKERRQ(ierr);
  if (flg){
    PetscMPIInt subsize,subrank,duprank;
    ierr = MPI_Comm_size((MPI_Comm)subcomm,&subsize);CHKERRQ(ierr);
    ierr = MPI_Comm_rank((MPI_Comm)subcomm,&subrank);CHKERRQ(ierr);
    ierr = MPI_Comm_rank((MPI_Comm)psubcomm->dupparent,&duprank);CHKERRQ(ierr);

    ierr = PetscSynchronizedPrintf(comm,"[%D], color %D, sub-size %D, sub-rank %D, duprank %D\n",rank,psubcomm->color,subsize,subrank,duprank);
    ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  }

  /* Create subA */
  ierr = MatGetMultiProcBlock(A,subcomm,MAT_INITIAL_MATRIX,&subA);CHKERRQ(ierr);

  /* Create sub vectors without arrays. Place b's and x's local arrays into subb and subx */
  ierr = MatGetLocalSize(subA,&m,&n);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,PETSC_NULL,&subb);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,PETSC_NULL,&subx);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,PETSC_NULL,&subu);CHKERRQ(ierr);

  ierr = VecGetArray(b,&barray);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(u,&uarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(subb,barray);CHKERRQ(ierr);
  ierr = VecPlaceArray(subx,xarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(subu,uarray);CHKERRQ(ierr);

  /* Create linear solvers associated with subA */
  ierr = KSPCreate(subcomm,&subksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(subksp,subA,subA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(subksp);CHKERRQ(ierr);

  /* Solve sub systems */
  ierr = KSPSolve(subksp,subb,subx);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(subksp,&its);CHKERRQ(ierr);

  /* check residual */
  ierr = MatMult(subA,subx,subu);CHKERRQ(ierr);
  ierr = VecAXPY(subu,-1.0,subb);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > 1.e-4 && !rank){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%D]  Number of iterations = %3D\n",rank,its);CHKERRQ(ierr);
    printf("Error: Residual norm of each block |subb - subA*subx |= %G\n",norm);
  }
  ierr = VecResetArray(subb);CHKERRQ(ierr);
  ierr = VecResetArray(subx);CHKERRQ(ierr);
  ierr = VecResetArray(subu);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-subvec_view",&id,&flg);CHKERRQ(ierr);
  if (flg && rank == id){
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%D] subb:\n", rank);
    ierr = VecGetArray(subb,&array);CHKERRQ(ierr);
    for (i=0; i<m; i++) printf("%G\n",PetscRealPart(array[i]));
    ierr = VecRestoreArray(subb,&array);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%D] subx:\n", rank);
    ierr = VecGetArray(subx,&array);CHKERRQ(ierr);
    for (i=0; i<m; i++) printf("%G\n",PetscRealPart(array[i]));
    ierr = VecRestoreArray(subx,&array);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&barray);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&uarray);CHKERRQ(ierr);
  ierr = MatDestroy(&subA);CHKERRQ(ierr);
  ierr = VecDestroy(&subb);CHKERRQ(ierr);
  ierr = VecDestroy(&subx);CHKERRQ(ierr);
  ierr = VecDestroy(&subu);CHKERRQ(ierr);
  ierr = KSPDestroy(&subksp);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&psubcomm);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&x);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}

