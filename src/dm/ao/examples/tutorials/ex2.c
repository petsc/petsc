#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.1 1998/04/15 20:22:50 bsmith Exp bsmith $";
#endif

static char help[] = 
"Reads a a simple unstructured grid from a file, partitions it\n\
 and distributes the grid data accordingly\n";

/*T
   Concepts: Mat^Partitioning a matrix;
   Processors: n
T*/

/* 
  Include "mat.h" so that we can use matrices.
  automatically includes:
     petsc.h  - base PETSc routines   vec.h    - vectors
     sys.h    - system routines       mat.h    - matrices
     is.h     - index sets            viewer.h - viewers               
*/
#include "mat.h"

int main(int argc,char **args)
{
  Mat    Adj;                /* adjacency matrix */
  int    rank,size,n,*mlocal,i,*ia,*ja;
  char   msg[128];
  double *vertices,*tmpvertices;

  PetscInitialize(&argc,&args,(char *)0,help);

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
    fscanf(fd,"Number Vertices = %d\n",&n);
    printf("Number of grid vertices %d\n",n);

    /* 
      allocate enough room for the first processor to keep track of how many 
      vertices are assigned to each processor.
    */ 
    mlocal = (int *) PetscMalloc(size*sizeof(int));CHKPTRA(mlocal);
    for ( i=0; i<size; i++ ) {
      mlocal[i] = n/size + ((n % size) > i);
      printf("Processor %d assigned %d vertices\n",i,mlocal[i]);
    }
 
    
    
    /* 
       Read in vertices for this processor (processor 0)
    */


    fclose(fd);
  }

  PetscFinalize();
  return 0;
}

