#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.3 1998/04/20 19:30:04 bsmith Exp bsmith $";
#endif

static char help[] = 
"Reads a a simple unstructured grid from a file, partitions it\n\
 and distributes the grid data accordingly\n";

/*T
   Concepts: Mat^Partitioning a matrix;
   Processors: n
T*/

/*
       This is a very basic, even crude example of managing an unstructured
    grid in parallel.

    This is for a Galerkin style finite element method. 

    After the calls below each processor will have 
      1) a list of elements it "owns"; for each element it will have the global 
         (and local) numbering of the three vertices it has
      2) a list of vertices it "owns" plus any additional ghost vertices that 
         are contained by a local element but not owned locally. For each owned 
         (and ghost vertex) it will have the x and y coordinates

    It will not have 
      1) list of ghost elements (since they are not needed for traditional 
         Galerkin style finite element methods). For various finite volume methods
         you may need the ghost element lists, these may be generated using the 
         element neighbor information given in the file database.

*/

/* 
  Include "mat.h" so that we can use matrices.
  automatically includes:
     petsc.h  - base PETSc routines   vec.h    - vectors
     sys.h    - system routines       mat.h    - matrices
     is.h     - index sets            viewer.h - viewers               
*/
#include "mat.h"

typedef struct {
  int n_vert,n_ele;
  int mlocal_vert,mlocal_ele;
  int *ia,*ja;
} GridData;

/*

  Variables on all processors:
     n_vert - total number of vertices
     mlocal_vert - number of vertices on this processor
     vert - x,y coordinates of local vertices

     n_ele - total number of elements
     mlocal_ele - number of vertices on this processor
     ele  - vertices of elements on this processor

     ia,ja - adjacency graph of elements
    
  Variables on processor 0 during data reading from file:
     mmlocal_vert[i] - number of vertices on each processor
     tmpvert - x,y coordinates of vertices on any processor (as read in)

     mmlocal_ele[i] - number of elements on each processor

     tmpia, tmpja - adjacency graph of elements for other processors

*/

extern int ReadData(GridData *);
extern int PartitionData(GridData *);
extern int MoveData(GridData *);

int main(int argc,char **args)
{
  int          ierr;
  GridData     gdata;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = ReadData(&gdata); CHKERRA(ierr);
  ierr = PartitionData(&gdata); CHKERRA(ierr);
  ierr = MoveData(&gdata); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}


/*
     Reads in the grid data from a file; each processor is naively 
  assigned a continuous chunk of vertex and element data.
*/
int ReadData(GridData *gdata)
{
  int          rank,size,n_vert,*mmlocal_vert,mlocal_vert,i,*ia,*ja,cnt,j;
  int          mlocal_ele,*mmlocal_ele,*ele,*tmpele,n_ele,net,a1,a2,a3;
  int          *iatmp,*jatmp;
  char         msg[128];
  double       *vert,*tmpvert;
  MPI_Status   status;

  /*
     Processor 0 opens the file, reads in data and send a portion off 
   each other processor.

     Note: For a truely scalable IO portion of the code, one would store
   the grid data in a binary file and use MPI-IO commands to have each 
   processor read in the parts that it needs.
  */
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  if (!rank) {
    FILE *fd;
    fd = fopen("usgdata","r"); if (!fd) SETERRA(1,1,"Cannot open grid file");

    /* read in number of vertices */
    fgets(msg,128,fd);
    printf("File msg:%s",msg);
    fscanf(fd,"Number Vertices = %d\n",&n_vert);
    printf("Number of grid vertices %d\n",n_vert);

    /* broadcast number of vertices to all processors */
    MPI_Bcast(&n_vert,1,MPI_INT,0,PETSC_COMM_WORLD);
    mlocal_vert  = n_vert/size + ((n_vert % size) > 0);

    /* 
      allocate enough room for the first processor to keep track of how many 
      vertices are assigned to each processor.
    */ 
    mmlocal_vert = (int *) PetscMalloc(size*sizeof(int));CHKPTRA(mmlocal_vert);
    for ( i=0; i<size; i++ ) {
      mmlocal_vert[i] = n_vert/size + ((n_vert % size) > i);
      printf("Processor %d assigned %d vertices\n",i,mmlocal_vert[i]);
    }
 
    vert = (double *) PetscMalloc(2*mmlocal_vert[0]*sizeof(double));CHKPTRA(vert);   
    printf("Vertices assigned to processor 0\n");
    for ( i=0; i<mlocal_vert; i++ ) {
      fscanf(fd,"%d %lf %lf\n",&cnt,vert+2*i,vert+2*i+1);
      printf("%d %g %g\n",cnt,vert[2*i],vert[2*i+1]);
    }

    /* read in vertices for all the other processors */
    tmpvert = (double *) PetscMalloc(2*mmlocal_vert[0]*sizeof(double));CHKPTRA(tmpvert);
    for ( j=1; j<size; j++ ) {
      printf("Vertices assigned to processor %d\n",j);
      for ( i=0; i<mmlocal_vert[j]; i++ ) {
        fscanf(fd,"%d %lf %lf\n",&cnt,tmpvert+2*i,tmpvert+2*i+1);
        printf("%d %g %g\n",cnt,tmpvert[2*i],tmpvert[2*i+1]);
      }
      MPI_Send(tmpvert,2*mmlocal_vert[j],MPI_DOUBLE,j,0,PETSC_COMM_WORLD);
    }
    PetscFree(tmpvert);
    PetscFree(mmlocal_vert);

    fscanf(fd,"Number Elements = %d\n",&n_ele);
    printf("Number of grid elements %d\n",n_ele);

    /* broadcast number of elements to all processors */
    MPI_Bcast(&n_ele,1,MPI_INT,0,PETSC_COMM_WORLD);
    mlocal_ele  = n_ele/size + ((n_ele % size) > 0);

    /* 
      allocate enough room for the first processor to keep track of how many 
      elements are assigned to each processor.
    */ 
    mmlocal_ele = (int *) PetscMalloc(size*sizeof(int));CHKPTRA(mmlocal_ele);
    for ( i=0; i<size; i++ ) {
      mmlocal_ele[i] = n_ele/size + ((n_ele % size) > i);
      printf("Processor %d assigned %d elements\n",i,mmlocal_ele[i]);
    }
 
    ele = (int *) PetscMalloc(3*mmlocal_ele[0]*sizeof(int));CHKPTRA(ele);   
    printf("Elements assigned to processor 0\n");
    for ( i=0; i<mlocal_ele; i++ ) {
      fscanf(fd,"%d %d %d %d\n",&cnt,ele+3*i,ele+3*i+1,ele+3*i+2);
      printf("%d %d %d %d\n",cnt,ele[3*i],ele[3*i+1],ele[3*i+2]);
    }

    /* read in elements for all the other processors */
    tmpele = (int *) PetscMalloc(3*mmlocal_ele[0]*sizeof(int));CHKPTRA(tmpele);
    for ( j=1; j<size; j++ ) {
      printf("Elements assigned to processor %d\n",j);
      for ( i=0; i<mmlocal_ele[j]; i++ ) {
        fscanf(fd,"%d %d %d %d\n",&cnt,tmpele+3*i,tmpele+3*i+1,tmpele+3*i+2);
        printf("%d %d %d %d\n",cnt,tmpele[3*i],tmpele[3*i+1],tmpele[3*i+2]);
      }
      MPI_Send(tmpele,3*mmlocal_ele[j],MPI_INT,j,0,PETSC_COMM_WORLD);
    }
    PetscFree(tmpele);

    /* 
         read in element neighbors for processor 0 
         We don't know how many spaces in ja[] to allocate so we allocate 
       3*the number of local elements, this is the maximum it could be
    */
    ia    = (int *) PetscMalloc((mlocal_ele+1)*sizeof(int));CHKPTRA(ia);
    ja    = (int *) PetscMalloc((3*mlocal_ele+1)*sizeof(int));CHKPTRA(ja);
    net   = 0;
    ia[0] = 0;
    printf("Element neighbors on processor 0\n");
    fgets(msg,128,fd);
    for ( i=0; i<mlocal_ele; i++ ) {
      fscanf(fd,"%d %d %d %d\n",&cnt,&a1,&a2,&a3);
      printf("%d %d %d %d\n",cnt,a1,a2,a3);
      if (a1 >= 0) {ja[net++] = a1;}
      if (a2 >= 0) {ja[net++] = a2;}
      if (a3 >= 0) {ja[net++] = a3;}
      ia[i+1] = net;
    }

    printf("ia values for processor 0\n");
    for ( i=0; i<mlocal_ele+1; i++ ) {
      printf("%d ",ia[i]);
    }
    printf("\n");
    printf("ja values for processor 0\n");
    for ( i=0; i<ia[mlocal_ele]; i++ ) {
      printf("%d ",ja[i]);
    }
    printf("\n");

    iatmp    = (int *) PetscMalloc((mlocal_ele+1)*sizeof(int));CHKPTRA(iatmp);
    jatmp    = (int *) PetscMalloc((3*mlocal_ele+1)*sizeof(int));CHKPTRA(jatmp);
    for ( j=1; j<size; j++ ) {
      net   = 0;
      iatmp[0] = 0;
      printf("Element neighbors on processor %d\n",j);
      for ( i=0; i<mmlocal_ele[j]; i++ ) {
        fscanf(fd,"%d %d %d %d\n",&cnt,&a1,&a2,&a3);
        printf("%d %d %d %d\n",cnt,a1,a2,a3);
        if (a1 >= 0) {jatmp[net++] = a1;}
        if (a2 >= 0) {jatmp[net++] = a2;}
        if (a3 >= 0) {jatmp[net++] = a3;}
        iatmp[i+1] = net;
      }

      printf("ia values for processor %d\n",j);
      for ( i=0; i<mmlocal_ele[j]+1; i++ ) {
        printf("%d ",iatmp[i]);
      }
      printf("\n");
      printf("ja values for processor %d\n",j);
      for ( i=0; i<iatmp[mmlocal_ele[j]]; i++ ) {
        printf("%d ",jatmp[i]);
      }
      printf("\n");

      /* send graph off to appropriate processor */
      MPI_Send(iatmp,mmlocal_ele[j]+1,MPI_INT,j,0,PETSC_COMM_WORLD);
      MPI_Send(jatmp,iatmp[mmlocal_ele[j]],MPI_INT,j,0,PETSC_COMM_WORLD);
    }
    PetscFree(iatmp);
    PetscFree(jatmp);
    PetscFree(mmlocal_ele);

    fclose(fd);
  } else {
    /* receive total number of vertices */
    MPI_Bcast(&n_vert,1,MPI_INT,0,PETSC_COMM_WORLD);
    mlocal_vert = n_vert/size + ((n_vert % size) > rank);

    /* receive vertices */
    vert = (double *) PetscMalloc(2*(mlocal_vert+1)*sizeof(double));CHKPTRQ(vert);
    MPI_Recv(vert,2*mlocal_vert,MPI_DOUBLE,0,0,PETSC_COMM_WORLD,&status);

    /* receive total number of elements */
    MPI_Bcast(&n_ele,1,MPI_INT,0,PETSC_COMM_WORLD);
    mlocal_ele = n_ele/size + ((n_ele % size) > rank);

    /* receive elements */
    ele = (int *) PetscMalloc(3*(mlocal_ele+1)*sizeof(int));CHKPTRQ(ele);
    MPI_Recv(ele,3*mlocal_ele,MPI_INT,0,0,PETSC_COMM_WORLD,&status);

    /* receive element adjacency graph */
    ia    = (int *) PetscMalloc((mlocal_ele+1)*sizeof(int));CHKPTRA(ia);
    MPI_Recv(ia,mlocal_ele+1,MPI_INT,0,0,PETSC_COMM_WORLD,&status);

    ja    = (int *) PetscMalloc((ia[mlocal_ele]+1)*sizeof(int));CHKPTRA(ja);
    MPI_Recv(ja,ia[mlocal_ele],MPI_INT,0,0,PETSC_COMM_WORLD,&status);
  }

  gdata->n_vert      = n_vert;
  gdata->n_ele       = n_ele;
  gdata->mlocal_vert = mlocal_vert;
  gdata->mlocal_ele  = mlocal_ele;

  gdata->ia          = ia;
  gdata->ja          = ja;

  return 0;
}


/*
         Given the grid data spread across the processors, determines a
   new partitioning of the cells to reduce the number of cut edges between
   cells.
*/
int PartitionData(GridData *gdata)
{
  Mat          Adj;                /* adjacency matrix */
  int          *ia,*ja;
  int          mlocal_ele,n_ele,ierr;
  Partitioning part;
  IS           isnewproc; 

  n_ele  = gdata->n_ele;
  mlocal_ele  = gdata->mlocal_ele;

  ia          = gdata->ia;
  ja          = gdata->ja;

  /*
      Create the adjacency graph
  */
  ierr = MatCreateMPIAdj(PETSC_COMM_WORLD,mlocal_ele,n_ele,ia,ja,&Adj);CHKERRQ(ierr);

  /*
      Create the partioning object
  */
  ierr = PartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
  ierr = PartitioningSetAdjacency(part,Adj); CHKERRQ(ierr);
  ierr = PartitioningSetFromOptions(part);CHKERRQ(ierr);
  ierr = PartitioningApply(part,&isnewproc);CHKERRQ(ierr);
  ierr = PartitioningDestroy(part); CHKERRQ(ierr);

  /*
       isnewproc - indicates for each local element the new processor it is assigned to
  */
  PetscPrintf(PETSC_COMM_WORLD,"New processor assignment for each element");
  ierr = ISView(isnewproc,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
      Free the adjacency graph data structures
  */
  ierr = MatDestroy(Adj); CHKERRQ(ierr);


  return 0;
}

/*
      Moves the grid data to be on the correct processor for the new
   partitioning.
*/
int MoveData(GridData *gdata)
{

  /* 
      We first must move the element vertex information to the processor
    that needs it.
  */

  return 0;
}



