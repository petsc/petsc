#define PETSCMAT_DLL

#include "../src/mat/impls/adj/mpi/mpiadj.h"       /*I "petscmat.h" I*/

#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef PETSC_HAVE_STDLIB_H
#include <stdlib.h>
#endif

/* 
   Currently using Scotch-3.4
*/
EXTERN_C_BEGIN
#include "scotch.h"
EXTERN_C_END

/*************************************
 * 				     *
 * Note:			     *
 * 				     *
 * To make scotch compile I 	     *
 * modified all old mat->m/M into    *
 * mat->rmap->n/N		     *
 *				     *
 * Hope I was right		     *
 *				     *
 *************************************/
typedef struct {
    char arch[PETSC_MAX_PATH_LEN];
    int multilevel;
    char strategy[30];
    int global_method;          /* global method */
    int local_method;           /* local method */
    int nbvtxcoarsed;           /* number of vertices for the coarse graph */
    int map;                    /* to know if we map on archptr or just partionate the graph */
    char *mesg_log;
    char host_list[PETSC_MAX_PATH_LEN];
} MatPartitioning_Scotch;

#define SIZE_LOG 10000          /* size of buffer for msg_log */

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_Scotch"
static PetscErrorCode MatPartitioningApply_Scotch(MatPartitioning part, IS * partitioning)
{
    PetscErrorCode ierr;
    int  *parttab, *locals = PETSC_NULL, rank, i, size;
    size_t                 j;
    Mat                    mat = part->adj, matMPI, matSeq;
    int                    nb_locals = mat->rmap->n;
    Mat_MPIAdj             *adj = (Mat_MPIAdj *) mat->data;
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;
    PetscTruth             flg;
#ifdef PETSC_HAVE_UNISTD_H
    int                    fd_stdout, fd_pipe[2], count,err;
#endif

    PetscFunctionBegin;

    /* check if the matrix is sequential, use MatGetSubMatrices if necessary */
    ierr = MPI_Comm_size(((PetscObject)mat)->comm, &size);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject) mat, MATMPIADJ, &flg);CHKERRQ(ierr);
    if (size > 1) {
        int M, N;
        IS isrow, iscol;
        Mat *A;

        if (flg) {
            SETERRQ(0, "Distributed matrix format MPIAdj is not supported for sequential partitioners");
        }
        PetscPrintf(((PetscObject)part)->comm, "Converting distributed matrix to sequential: this could be a performance loss\n");CHKERRQ(ierr);

        ierr = MatGetSize(mat, &M, &N);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF, M, 0, 1, &isrow);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &iscol);CHKERRQ(ierr);
        ierr = MatGetSubMatrices(mat, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &A);CHKERRQ(ierr);
        matSeq = *A; 
        ierr = PetscFree(A);CHKERRQ(ierr);
        ierr = ISDestroy(isrow);CHKERRQ(ierr);
        ierr = ISDestroy(iscol);CHKERRQ(ierr);
    } else
        matSeq = mat;

    /* convert the the matrix to MPIADJ type if necessary */
    if (!flg) {
        ierr = MatConvert(matSeq, MATMPIADJ, MAT_INITIAL_MATRIX, &matMPI);CHKERRQ(ierr);
    } else {
        matMPI = matSeq;
    }

    adj = (Mat_MPIAdj *) matMPI->data;  /* finaly adj contains adjacency graph */

    ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);

    {
        /* definition of Scotch library arguments */
        SCOTCH_Strat stratptr;      /* scotch strategy */
        SCOTCH_Graph grafptr;       /* scotch graph */
#if defined(DOES_NOT_COMPILE_DUE_TO_BROKEN_INTERFACE)
        int vertnbr = mat->rmap->N; /* number of vertices in full graph */
        int *verttab = adj->i;      /* start of edge list for each vertex */
        int *edgetab = adj->j;      /* edge list data */
        int edgenbr = adj->nz;      /* number of edges */
        int *velotab = NULL;        /* not used by petsc interface */
        int *vlbltab = NULL;    
        int *edlotab = NULL; 
        int flagval = 3;            /* (cf doc scotch no weight edge & vertices) */
#endif  
        int baseval = 0;            /* 0 for C array indexing */
        char strategy[256];

        ierr = PetscMalloc((mat->rmap->N) * sizeof(int), &parttab);CHKERRQ(ierr); 

        /* redirect output to buffer scotch -> mesg_log */
#ifdef PETSC_HAVE_UNISTD_H
        fd_stdout = dup(1);
        pipe(fd_pipe);
        close(1);
        dup2(fd_pipe[1], 1);
        ierr = PetscMalloc(SIZE_LOG * sizeof(char), &(scotch->mesg_log));CHKERRQ(ierr);
#endif

        /* library call */

        /* Construction of the scotch graph object */
        ierr = SCOTCH_graphInit(&grafptr);
#if defined(DOES_NOT_COMPILE_DUE_TO_BROKEN_INTERFACE)
        ierr = SCOTCH_graphBuild((SCOTCH_Graph *)   &grafptr, 
				 (const SCOTCH_Num)  vertnbr, 
				 (const SCOTCH_Num)  verttab, 
				 (const SCOTCH_Num *)velotab,
				 (const SCOTCH_Num *)vlbltab, 
				 (const SCOTCH_Num *)edgenbr, 
				 (const SCOTCH_Num *)edgetab, 
				 (const SCOTCH_Num)  edlotab, 
				 (const SCOTCH_Num *)baseval, 
				 (const SCOTCH_Num *)flagval);CHKERRQ(ierr);
#else
        SETERRQ(PETSC_ERR_SUP,"Scotch interface currently broken");
#endif
        ierr = SCOTCH_graphCheck(&grafptr);CHKERRQ(ierr);

        /* Construction of the strategy */
        if (scotch->strategy[0] != 0) {
            ierr = PetscStrcpy(strategy, scotch->strategy);CHKERRQ(ierr);
        } else {
            PetscStrcpy(strategy, "b{strat=");

            if (scotch->multilevel) {
                /* PetscStrcat(strategy,"m{vert=");
                   sprintf(strategy+strlen(strategy),"%d",scotch->nbvtxcoarsed);
                   PetscStrcat(strategy,",asc="); */
                sprintf(strategy, "b{strat=m{vert=%d,asc=",
                    scotch->nbvtxcoarsed);
            } else
                PetscStrcpy(strategy, "b{strat=");

            switch (scotch->global_method) {
            case MP_SCOTCH_GREEDY:
                PetscStrcat(strategy, "h");
                break;
            case MP_SCOTCH_GPS:
                PetscStrcat(strategy, "g");
                break;
            case MP_SCOTCH_GR_GPS:
                PetscStrcat(strategy, "g|h");
            }

            switch (scotch->local_method) {
            case MP_SCOTCH_KERNIGHAN_LIN:
                if (scotch->multilevel)
                    PetscStrcat(strategy, ",low=f}");
                else
                    PetscStrcat(strategy, " f");
                break;
            case MP_SCOTCH_NONE:
                if (scotch->multilevel)
                    PetscStrcat(strategy, ",asc=x}");
            default:
                break;
            }

            PetscStrcat(strategy, " x}");
        }

        PetscPrintf(((PetscObject)part)->comm, "strategy=[%s]\n", strategy);

        ierr = SCOTCH_stratInit(&stratptr);CHKERRQ(ierr);
	/*

	  TODO: Correct this part

	  Commented because this doesn't exists anymore 

	  
	  ierr = SCOTCH_stratMap(&stratptr, strategy);CHKERRQ(ierr);
	*/
        /* check for option mapping */
        if (!scotch->map) {
	  /* ********************************************
	   *						*
	   *        TODO: Correct this part		*
	   *						*
	   * Won't work with this tmp SCOTCH_Strat...	*
	   *						*
	   * I just modified it to make scotch compile, *
	   * to be able to use PaStiX...		*
	   *						*
	   **********************************************/
#if defined (DOES_NOT_COMPILE_DUE_TO_BROKEN_INTERFACE)
	  SCOTCH_Strat tmp;
	  ierr = SCOTCH_graphPart((const SCOTCH_Graph *)&grafptr, 
				  (const SCOTCH_Num)    &stratptr, 
				  (const SCOTCH_Strat *)&tmp,        /* The Argument changed from scotch 3.04 it was part->n, */ 
				  (SCOTCH_Num *)        parttab);CHKERRQ(ierr);
#else
        SETERRQ(PETSC_ERR_SUP,"Scotch interface currently broken");
#endif
            ierr = PetscPrintf(PETSC_COMM_SELF, "Partition simple without mapping\n");
        } else {
            SCOTCH_Graph grafarch;
            SCOTCH_Num *listtab;
            SCOTCH_Num listnbr = 0;
            SCOTCH_Arch archptr;        /* file in scotch architecture format */
            SCOTCH_Strat archstrat;
            int arch_total_size, *parttab_tmp,err;
            int cpt;
            char buf[256];
            FILE *file1, *file2;
            char host_buf[256];

            /* generate the graph that represents the arch */
            file1 = fopen(scotch->arch, "r");
            if (!file1) SETERRQ1(PETSC_ERR_FILE_OPEN, "Scotch: unable to open architecture file %s", scotch->arch);

            ierr = SCOTCH_graphInit(&grafarch);CHKERRQ(ierr);
            ierr = SCOTCH_graphLoad(&grafarch, file1, baseval, 3);CHKERRQ(ierr);

            ierr = SCOTCH_graphCheck(&grafarch);CHKERRQ(ierr);
            SCOTCH_graphSize(&grafarch, &arch_total_size, &cpt);

            err = fclose(file1);
            if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    

            printf("total size = %d\n", arch_total_size);

            /* generate the list of nodes currently working */
            ierr = PetscGetHostName(host_buf, 256);CHKERRQ(ierr);
            ierr = PetscStrlen(host_buf, &j);CHKERRQ(ierr);

            file2 = fopen(scotch->host_list, "r");
            if (!file2) SETERRQ1(PETSC_ERR_FILE_OPEN, "Scotch: unable to open host list file %s", scotch->host_list);

            i = -1;
            flg = PETSC_FALSE;
            while (!feof(file2) && !flg) {
                i++;
                fgets(buf, 256, file2);
                PetscStrncmp(buf, host_buf, j, &flg);
            }
            err = fclose(file2);
            if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
            if (!flg) SETERRQ1(PETSC_ERR_LIB, "Scotch: unable to find '%s' in host list file", host_buf);

            listnbr = size;
            ierr = PetscMalloc(sizeof(SCOTCH_Num) * listnbr, &listtab);CHKERRQ(ierr);

            ierr = MPI_Allgather(&i, 1, MPI_INT, listtab, 1, MPI_INT, ((PetscObject)part)->comm);CHKERRQ(ierr);

            printf("listnbr = %d, listtab = ", listnbr);
            for (i = 0; i < listnbr; i++)
                printf("%d ", listtab[i]);

            printf("\n");
            err = fflush(stdout);
            if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on file");    

            ierr = SCOTCH_stratInit(&archstrat);CHKERRQ(ierr);
	    /**************************************************************
	     *								  *
	     * TODO: Correct this part					  *
	     * 								  *
	     * Commented because this doesn't exists anymore 		  *
	     * 								  *
	     * ierr = SCOTCH_stratBipart(&archstrat, "fx");CHKERRQ(ierr); *
	     **************************************************************/
            ierr = SCOTCH_archInit(&archptr);CHKERRQ(ierr);
            ierr = SCOTCH_archBuild(&archptr, &grafarch, listnbr, listtab,
                &archstrat);CHKERRQ(ierr);

            ierr = PetscMalloc((mat->rmap->N) * sizeof(int), &parttab_tmp);CHKERRQ(ierr);
	    /************************************************************************************
	     *											*
	     * TODO: Correct this part								*
	     *											*
	     * Commented because this doesn't exists anymore 					*
	     *											*
	     * ierr = SCOTCH_mapInit(&mappptr, &grafptr, &archptr, parttab_tmp);CHKERRQ(ierr);	*
	     *											*
	     * ierr = SCOTCH_mapCompute(&mappptr, &stratptr);CHKERRQ(ierr);			*
	     * 											*
	     * ierr = SCOTCH_mapView(&mappptr, stdout);CHKERRQ(ierr);				*
	     ************************************************************************************/
            /* now we have to set in the real parttab at the good place */
            /* because the ranks order are different than position in */
            /* the arch graph */
            for (i = 0; i < mat->rmap->N; i++) {
                parttab[i] = parttab_tmp[i];
            }

            ierr = PetscFree(listtab);CHKERRQ(ierr);
            SCOTCH_archExit(&archptr);
	    /*************************************************
   	     * TODO: Correct this part			     *
	     * 						     *
	     * Commented because this doesn't exists anymore *
	     * SCOTCH_mapExit(&mappptr);		     *
	     *************************************************/
            SCOTCH_stratExit(&archstrat);
        }

        /* dump to mesg_log... */
#ifdef PETSC_HAVE_UNISTD_H
        err = fflush(stdout);
        if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on stdout");    

        count = read(fd_pipe[0], scotch->mesg_log, (SIZE_LOG - 1) * sizeof(char));
        if (count < 0)
            count = 0;
        scotch->mesg_log[count] = 0;
        close(1);
        dup2(fd_stdout, 1);
        close(fd_stdout);
        close(fd_pipe[0]);
        close(fd_pipe[1]);
#endif

        SCOTCH_graphExit(&grafptr);
        SCOTCH_stratExit(&stratptr);
    }

    if (ierr)
        SETERRQ(PETSC_ERR_LIB, scotch->mesg_log);

    /* Creation of the index set */

    ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)part)->comm, &size);CHKERRQ(ierr);
    nb_locals = mat->rmap->N / size;
    locals = parttab + rank * nb_locals;
    if (rank < mat->rmap->N % size) {
        nb_locals++;
        locals += rank;
    } else
        locals += mat->rmap->N % size;
    ierr = ISCreateGeneral(((PetscObject)part)->comm, nb_locals, locals, partitioning);CHKERRQ(ierr);

    /* destroying old objects */
    ierr = PetscFree(parttab);CHKERRQ(ierr);
    if (matSeq != mat) {
        ierr = MatDestroy(matSeq);CHKERRQ(ierr);
    }
    if (matMPI != mat) {
        ierr = MatDestroy(matMPI);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_Scotch"
PetscErrorCode MatPartitioningView_Scotch(MatPartitioning part, PetscViewer viewer)
{
  MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;
  PetscErrorCode         ierr;
  PetscMPIInt            rank;
  PetscTruth             iascii;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    if (!rank && scotch->mesg_log) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s\n", scotch->mesg_log);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Viewer type %s not supported for this Scotch partitioner",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetGlobal"
/*@
     MatPartitioningScotchSetGlobal - Set method for global partitioning.

  Input Parameter:
.  part - the partitioning context
.  method - MP_SCOTCH_GREED, MP_SCOTCH_GIBBS or MP_SCOTCH_GR_GI (the combination of two)
   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetGlobal(MatPartitioning part,
    MPScotchGlobalType global)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    switch (global) {
    case MP_SCOTCH_GREEDY:
    case MP_SCOTCH_GPS:
    case MP_SCOTCH_GR_GPS:
        scotch->global_method = global;
        break;
    default:
        SETERRQ(PETSC_ERR_SUP, "Scotch: Unknown or unsupported option");
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetCoarseLevel"
/*@
    MatPartitioningScotchSetCoarseLevel - Set the coarse level 
    
  Input Parameter:
.  part - the partitioning context
.  level - the coarse level in range [0.0,1.0]

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetCoarseLevel(MatPartitioning part, PetscReal level)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    if (level < 0 || level > 1.0) {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,
            "Scocth: level of coarsening out of range [0.0-1.0]");
    } else {
      /* ********************************************
       *					    *
       *        TODO: Correct this part		    *
       *					    *
       * Won't work with this nbvxtcoarsed          *
       *					    *
       * I just modified it to make scotch compile, *
       * to be able to use PaStiX...		    *
       *					    *
       **********************************************/
      scotch->nbvtxcoarsed = 0;
      /* with scotch 3.0.4 it was : scotch->nbvtxcoarsed = (int)(part->adj->N * level); */
    }
    if (scotch->nbvtxcoarsed < 20)
        scotch->nbvtxcoarsed = 20;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetStrategy"
/*@C
    MatPartitioningScotchSetStrategy - Set the strategy to be used by Scotch.
    This is an alternative way of specifying the global method, the local
    method, the coarse level and the multilevel option.
    
  Input Parameter:
.  part - the partitioning context
.  level - the strategy in Scotch format. Check Scotch documentation.

   Level: advanced

.seealso: MatPartitioningScotchSetGlobal(), MatPartitioningScotchSetLocal(), MatPartitioningScotchSetCoarseLevel(), MatPartitioningScotchSetMultilevel(), 
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetStrategy(MatPartitioning part, char *strat)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    PetscStrcpy(scotch->strategy, strat);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetLocal"
/*@
     MatPartitioningScotchSetLocal - Set method for local partitioning.

  Input Parameter:
.  part - the partitioning context
.  method - MP_SCOTCH_KERNIGHAN_LIN or MP_SCOTCH_NONE

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetLocal(MatPartitioning part, MPScotchLocalType local)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    switch (local) {
    case MP_SCOTCH_KERNIGHAN_LIN:
    case MP_SCOTCH_NONE:
        scotch->local_method = local;
        break;
    default:
        SETERRQ(PETSC_ERR_ARG_CORRUPT, "Scotch: Unknown or unsupported option");
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetArch"
/*@C
     MatPartitioningScotchSetArch - Specify the file that describes the
     architecture used for mapping. The format of this file is documented in
     the Scotch manual.

  Input Parameter:
.  part - the partitioning context
.  file - the name of file
   Level: advanced

  Note:
  If the name is not set, then the default "archgraph.src" is used.

.seealso: MatPartitioningScotchSetHostList(),MatPartitioningScotchSetMapping()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetArch(MatPartitioning part, const char *filename)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    PetscStrcpy(scotch->arch, filename);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetHostList"
/*@C
     MatPartitioningScotchSetHostList - Specify host list file for mapping.

  Input Parameter:
.  part - the partitioning context
.  file - the name of file

   Level: advanced

  Notes:
  The file must consist in a list of hostnames (one per line). These hosts
  are the ones referred to in the architecture file (see 
  MatPartitioningScotchSetArch()): the first host corresponds to index 0,
  the second one to index 1, and so on.
  
  If the name is not set, then the default "host_list" is used.
  
.seealso: MatPartitioningScotchSetArch(), MatPartitioningScotchSetMapping()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetHostList(MatPartitioning part, const char *filename)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    PetscStrcpy(scotch->host_list, filename);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetMultilevel"
/*@
     MatPartitioningScotchSetMultilevel - Activates multilevel partitioning.

  Input Parameter:
.  part - the partitioning context

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetMultilevel(MatPartitioning part)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    scotch->multilevel = 1;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningScotchSetMapping"
/*@
     MatPartitioningScotchSetMapping - Activates architecture mapping for the 
     partitioning algorithm. Architecture mapping tries to enhance the quality
     of partitioning by using network topology information. 

  Input Parameter:
.  part - the partitioning context

   Level: advanced

.seealso: MatPartitioningScotchSetArch(),MatPartitioningScotchSetHostList()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetMapping(MatPartitioning part)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;

    PetscFunctionBegin;

    scotch->map = 1;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_Scotch"
PetscErrorCode MatPartitioningSetFromOptions_Scotch(MatPartitioning part)
{
    PetscErrorCode ierr;
    PetscTruth flag;
    char name[PETSC_MAX_PATH_LEN];
    int i;
    PetscReal r;

    const char *global[] = { "greedy", "gps", "gr_gps" };
    const char *local[] = { "kernighan-lin", "none" };

    PetscFunctionBegin;
    ierr = PetscOptionsHead("Set Scotch partitioning options");CHKERRQ(ierr);

    ierr = PetscOptionsEList("-mat_partitioning_scotch_global",
        "Global method to use", "MatPartitioningScotchSetGlobal", global, 3,
        global[0], &i, &flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningScotchSetGlobal(part, (MPScotchGlobalType)i);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEList("-mat_partitioning_scotch_local",
        "Local method to use", "MatPartitioningScotchSetLocal", local, 2,
        local[0], &i, &flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningScotchSetLocal(part, (MPScotchLocalType)i);CHKERRQ(ierr);
    }

    flag = PETSC_FALSE;
    ierr = PetscOptionsTruth("-mat_partitioning_scotch_mapping", "Use mapping","MatPartitioningScotchSetMapping", flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningScotchSetMapping(part);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-mat_partitioning_scotch_arch",
        "architecture file in scotch format", "MatPartitioningScotchSetArch",
        "archgraph.src", name, PETSC_MAX_PATH_LEN, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningScotchSetArch(part, name);CHKERRQ(ierr);

    ierr = PetscOptionsString("-mat_partitioning_scotch_hosts",
        "host list filename", "MatPartitioningScotchSetHostList",
        "host_list", name, PETSC_MAX_PATH_LEN, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningScotchSetHostList(part, name);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-mat_partitioning_scotch_coarse_level",
        "coarse level", "MatPartitioningScotchSetCoarseLevel", 0, &r,
        &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningScotchSetCoarseLevel(part, r);CHKERRQ(ierr);

    flag = PETSC_FALSE;
    ierr = PetscOptionsTruth("-mat_partitioning_scotch_mul", "Use coarse level","MatPartitioningScotchSetMultilevel", flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningScotchSetMultilevel(part);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-mat_partitioning_scotch_strategy",
        "Scotch strategy string",
        "MatPartitioningScotchSetStrategy", "", name, PETSC_MAX_PATH_LEN,
        &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningScotchSetStrategy(part, name);CHKERRQ(ierr);

    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_Scotch"
PetscErrorCode MatPartitioningDestroy_Scotch(MatPartitioning part)
{
    MatPartitioning_Scotch *scotch = (MatPartitioning_Scotch *) part->data;
    PetscErrorCode         ierr;

    PetscFunctionBegin;
    ierr = PetscFree(scotch->mesg_log);CHKERRQ(ierr);
    ierr = PetscFree(scotch);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*MC
   MAT_PARTITIONING_SCOTCH - Creates a partitioning context via the external package SCOTCH.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+  -mat_partitioning_scotch_global <greedy> (one of) greedy gps gr_gps
.  -mat_partitioning_scotch_local <kernighan-lin> (one of) kernighan-lin none
.  -mat_partitioning_scotch_mapping: Use mapping (MatPartitioningScotchSetMapping)
.  -mat_partitioning_scotch_arch <archgraph.src>: architecture file in scotch format (MatPartitioningScotchSetArch)
.  -mat_partitioning_scotch_hosts <host_list>: host list filename (MatPartitioningScotchSetHostList)
.  -mat_partitioning_scotch_coarse_level <0>: coarse level (MatPartitioningScotchSetCoarseLevel)
.  -mat_partitioning_scotch_mul: Use coarse level (MatPartitioningScotchSetMultilevel)
-  -mat_partitioning_scotch_strategy <>: Scotch strategy string (MatPartitioningScotchSetStrategy)

   Level: beginner

   Notes: See http://www.labri.fr/Perso/~pelegrin/scotch/

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_Scotch"
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningCreate_Scotch(MatPartitioning part)
{
    PetscErrorCode ierr;
    MatPartitioning_Scotch *scotch;

    PetscFunctionBegin;
    SETERRQ(PETSC_ERR_SUP,"Sorry, the PETSc interface to scotch has not been updated to the latest Scotch version");
    ierr = PetscNewLog(part,MatPartitioning_Scotch, &scotch);CHKERRQ(ierr);
    part->data = (void*) scotch;

    scotch->map = 0;
    scotch->global_method = MP_SCOTCH_GR_GPS;
    scotch->local_method = MP_SCOTCH_KERNIGHAN_LIN;
    PetscStrcpy(scotch->arch, "archgraph.src");
    scotch->nbvtxcoarsed = 200;
    PetscStrcpy(scotch->strategy, "");
    scotch->multilevel = 0;
    scotch->mesg_log = NULL;

    PetscStrcpy(scotch->host_list, "host_list");

    part->ops->apply = MatPartitioningApply_Scotch;
    part->ops->view = MatPartitioningView_Scotch;
    part->ops->destroy = MatPartitioningDestroy_Scotch;
    part->ops->setfromoptions = MatPartitioningSetFromOptions_Scotch;

    PetscFunctionReturn(0);
}

EXTERN_C_END
