/*

      Contributed by Arvid Bessen, Columbia University, June 2007

       Extension of DA object to any number of dimensions.

*/
#include "../src/dm/adda/addaimpl.h"                          /*I "petscda.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "ADDACreate"
/*@C
  ADDACreate - Creates and ADDA object that translate between coordinates
  in a geometric grid of arbitrary dimension and data in a PETSc vector
  distributed on several processors.

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  dim - the dimension of the grid
.  nodes - array with d entries that give the number of nodes in each dimension
.  procs - array with d entries that give the number of processors in each dimension
          (or PETSC_NULL if to be determined automatically)
.  dof - number of degrees of freedom per node
-  periodic - array with d entries that, i-th entry is set to  true iff dimension i is periodic

   Output Parameters:
.  adda - pointer to ADDA data structure that is created

  Level: intermediate

@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDACreate(MPI_Comm comm, PetscInt dim, PetscInt *nodes,PetscInt *procs,
                                           PetscInt dof, PetscTruth *periodic,ADDA *adda_p)
{
  PetscErrorCode ierr;
  ADDA           adda;
  PetscInt       s=1; /* stencil width, fixed to 1 at the moment */
  PetscMPIInt    rank,size;
  PetscInt       i;
  PetscInt       nodes_total;
  PetscInt       nodesleft;
  PetscInt       procsleft;
  PetscInt       procsdimi;
  PetscInt       ranki;
  PetscInt       rpq;

  PetscFunctionBegin;
  PetscValidPointer(nodes,3);
  PetscValidPointer(adda_p,6);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(*adda_p,_p_ADDA,struct _ADDAOps,ADDA_COOKIE,0,"ADDA",comm,ADDADestroy,0);CHKERRQ(ierr);
  adda = *adda_p;
  adda->ops->view = ADDAView;
  adda->ops->createglobalvector = ADDACreateGlobalVector;
  adda->ops->getcoloring = ADDAGetColoring;
  adda->ops->getmatrix = ADDAGetMatrix;
  adda->ops->getinterpolation = ADDAGetInterpolation;
  adda->ops->refine = ADDARefine;
  adda->ops->coarsen = ADDACoarsen;
  adda->ops->getinjection = ADDAGetInjection;
  adda->ops->getaggregates = ADDAGetAggregates;
  
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 
  
  adda->dim = dim;
  adda->dof = dof;

  /* nodes */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->nodes));CHKERRQ(ierr);
  ierr = PetscMemcpy(adda->nodes, nodes, dim*sizeof(PetscInt));CHKERRQ(ierr);
  /* total number of nodes */
  nodes_total = 1;
  for(i=0; i<dim; i++) nodes_total *= nodes[i];

  /* procs */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->procs));CHKERRQ(ierr);
  /* create distribution of nodes to processors */
  if(procs == PETSC_NULL) {
    procs = adda->procs;
    nodesleft = nodes_total;
    procsleft = size;
    /* figure out a good way to split the array to several processors */
    for(i=0; i<dim; i++) {
      if(i==dim-1) {
	procs[i] = procsleft;
      } else {
	/* calculate best partition */
	procs[i] = (PetscInt)(((double) nodes[i])*pow(((double) procsleft)/((double) nodesleft),1./((double)(dim-i)))+0.5);
	if(procs[i]<1) procs[i]=1;
	while( procs[i] > 0 ) {
	  if( procsleft % procs[i] )
	    procs[i]--;
	  else
	    break;
	}
	nodesleft /= nodes[i];
	procsleft /= procs[i];
      }
    }
  } else {
    /* user provided the number of processors */
    ierr = PetscMemcpy(adda->procs, procs, dim*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* check for validity */
  procsleft = 1;
  for(i=0; i<dim; i++) {
    if (nodes[i] < procs[i]) {
      SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Partition in direction %d is too fine! %D nodes, %D processors", i, nodes[i], procs[i]);
    }
    procsleft *= procs[i];
  }
  if(procsleft != size) {
    SETERRQ(1, "Created or was provided with inconsistent distribution of processors");
  }

  /* periodicity */
  adda->periodic = periodic;
  
  /* find out local region */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->lcs));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->lce));CHKERRQ(ierr);
  procsdimi=size;
  ranki=rank;
  for(i=0; i<dim; i++) {
    /* What is the number of processor for dimensions i+1, ..., dim-1? */
    procsdimi /= procs[i];
    /* these are all nodes that come before our region */
    rpq = ranki / procsdimi;
    adda->lcs[i] = rpq * (nodes[i]/procs[i]);
    if( rpq + 1 < procs[i] ) {
      adda->lce[i] = (rpq + 1) * (nodes[i]/procs[i]);
    } else {
      /* last one gets all the rest */
      adda->lce[i] = nodes[i];
    }
    ranki = ranki - rpq*procsdimi;
  }
  
  /* compute local size */
  adda->lsize=1;
  for(i=0; i<dim; i++) {
    adda->lsize *= (adda->lce[i]-adda->lcs[i]);
  }
  adda->lsize *= dof;

  /* find out ghost points */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->lgs));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->lge));CHKERRQ(ierr);
  for(i=0; i<dim; i++) {
    if( periodic[i] ) {
      adda->lgs[i] = adda->lcs[i] - s;
      adda->lge[i] = adda->lce[i] + s;
    } else {
      adda->lgs[i] = PetscMax(adda->lcs[i] - s, 0);
      adda->lge[i] = PetscMin(adda->lce[i] + s, nodes[i]);
    }
  }
  
  /* compute local size with ghost points */
  adda->lgsize=1;
  for(i=0; i<dim; i++) {
    adda->lgsize *= (adda->lge[i]-adda->lgs[i]);
  }
  adda->lgsize *= dof;

  /* create global and local prototype vector */
  ierr = VecCreateMPIWithArray(comm,adda->lsize,PETSC_DECIDE,0,&(adda->global));CHKERRQ(ierr);
  ierr = VecSetBlockSize(adda->global,adda->dof);CHKERRQ(ierr);
#if ADDA_NEEDS_LOCAL_VECTOR
  /* local includes ghost points */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,adda->lgsize,0,&(adda->local));CHKERRQ(ierr);
  ierr = VecSetBlockSize(adda->local,dof);CHKERRQ(ierr);
#endif

  ierr = PetscMalloc(dim*sizeof(PetscInt), &(adda->refine));CHKERRQ(ierr);
  for(i=0; i<dim; i++) adda->refine[i] = 3;
  adda->dofrefine = 1;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDADestroy"
/*@
   ADDADestroy - Destroys a distributed array.

   Collective on ADDA

   Input Parameter:
.  adda - the distributed array to destroy 

   Level: beginner

.keywords: distributed array, destroy

.seealso: ADDACreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDADestroy(ADDA adda)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda,ADDA_COOKIE,1);

  /* check reference count */
  if(--((PetscObject)adda)->refct > 0) PetscFunctionReturn(0);

  /* destroy the allocated data */
  ierr = PetscFree(adda->nodes);CHKERRQ(ierr);
  ierr = PetscFree(adda->procs);CHKERRQ(ierr);
  ierr = PetscFree(adda->lcs);CHKERRQ(ierr);
  ierr = PetscFree(adda->lce);CHKERRQ(ierr);
  ierr = PetscFree(adda->lgs);CHKERRQ(ierr);
  ierr = PetscFree(adda->lge);CHKERRQ(ierr);
  ierr = PetscFree(adda->refine);CHKERRQ(ierr);

  ierr = VecDestroy(adda->global);CHKERRQ(ierr);

  ierr = PetscHeaderDestroy(adda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAView"
/*@
   ADDAView - Views a distributed array.

   Collective on ADDA

    Input Parameter:
+   adda - the ADDA object to view
-   v - the viewer

    Level: developer

.keywords: distributed array, view

.seealso: DMView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAView(ADDA adda, PetscViewer v) {
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDACreateGlobalVector"
/*@
   ADDACreateGlobalVector - Creates global vector for distributed array.

   Collective on ADDA

   Input Parameter:
.  adda - the distributed array for which we create a global vector

   Output Parameter:
.  vec - the global vector

   Level: beginner

.keywords: distributed array, vector

.seealso: DMCreateGlobalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDACreateGlobalVector(ADDA adda, Vec *vec) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda,ADDA_COOKIE,1);
  PetscValidPointer(vec,2);
  ierr = VecDuplicate(adda->global, vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetColoring"
/*@
   ADDAGetColoring - Creates coloring for distributed array.

   Collective on ADDA

   Input Parameter:
+  adda - the distributed array for which we create a global vector
-  ctype - IS_COLORING_GHOSTED or IS_COLORING_LOCAL

   Output Parameter:
.  coloring - the coloring

   Level: developer

.keywords: distributed array, coloring

.seealso: DMGetColoring()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetColoring(ADDA adda, ISColoringType ctype,const MatType mtype,ISColoring *coloring) {
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetMatrix"
/*@
   ADDAGetMatrix - Creates matrix compatible with distributed array.

   Collective on ADDA

   Input Parameter:
.  adda - the distributed array for which we create the matrix
-  mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
           any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

   Output Parameter:
.  mat - the empty Jacobian 

   Level: beginner

.keywords: distributed array, matrix

.seealso: DMGetMatrix()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetMatrix(ADDA adda, const MatType mtype, Mat *mat) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda, ADDA_COOKIE, 1);
  ierr = MatCreate(((PetscObject)adda)->comm, mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat, adda->lsize, adda->lsize, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(*mat, mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetMatrixNS"
/*@
   ADDAGetMatrixNS - Creates matrix compatiable with two distributed arrays

   Collective on ADDA

   Input Parameter:
.  addar - the distributed array for which we create the matrix, which indexes the rows
.  addac - the distributed array for which we create the matrix, which indexes the columns
-  mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
           any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

   Output Parameter:
.  mat - the empty Jacobian 

   Level: beginner

.keywords: distributed array, matrix

.seealso: DMGetMatrix()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetMatrixNS(ADDA addar, ADDA addac, const MatType mtype, Mat *mat) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(addar, ADDA_COOKIE, 1);
  PetscValidHeaderSpecific(addac, ADDA_COOKIE, 2);
  PetscCheckSameComm(addar, 1, addac, 2);
  ierr = MatCreate(((PetscObject)addar)->comm, mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat, addar->lsize, addac->lsize, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(*mat, mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetInterpolation"
/*@
   ADDAGetInterpolation - Gets interpolation matrix between two ADDA objects

   Collective on ADDA

   Input Parameter:
+  adda1 - the fine ADDA object
-  adda2 - the second, coarser ADDA object

    Output Parameter:
+  mat - the interpolation matrix
-  vec - the scaling (optional)

   Level: developer

.keywords: distributed array, interpolation

.seealso: DMGetInterpolation()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetInterpolation(ADDA adda1,ADDA adda2,Mat *mat,Vec *vec) {
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDARefine"
/*@
   ADDARefine - Refines a distributed array.

   Collective on ADDA

   Input Parameter:
+  adda - the distributed array to refine
-  comm - the communicator to contain the new ADDA object (or PETSC_NULL)

   Output Parameter:
.  addaf - the refined ADDA

   Level: developer

.keywords: distributed array, refine

.seealso: DMRefine()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDARefine(ADDA adda, MPI_Comm comm, ADDA *addaf) {
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDACoarsen"
/*@
   ADDACoarsen - Coarsens a distributed array.

   Collective on ADDA

   Input Parameter:
+  adda - the distributed array to coarsen
-  comm - the communicator to contain the new ADDA object (or PETSC_NULL)

   Output Parameter:
.  addac - the coarsened ADDA

   Level: developer

.keywords: distributed array, coarsen

.seealso: DMCoarsen()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDACoarsen(ADDA adda, MPI_Comm comm,ADDA *addac) {
  PetscErrorCode ierr;
  PetscInt       *nodesc;
  PetscInt       dofc;
  PetscInt       i;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda, ADDA_COOKIE, 1);
  PetscValidPointer(addac, 3);
  ierr = PetscMalloc(adda->dim*sizeof(PetscInt), &nodesc);CHKERRQ(ierr);
  for(i=0; i<adda->dim; i++) {
    nodesc[i] = (adda->nodes[i] % adda->refine[i]) ? adda->nodes[i] / adda->refine[i] + 1 : adda->nodes[i] / adda->refine[i];
  }
  dofc = (adda->dof % adda->dofrefine) ? adda->dof / adda->dofrefine + 1 : adda->dof / adda->dofrefine;
  ierr = ADDACreate(((PetscObject)adda)->comm, adda->dim, nodesc, adda->procs, dofc, adda->periodic, addac);CHKERRQ(ierr);
  ierr = PetscFree(nodesc);CHKERRQ(ierr);
  /* copy refinement factors */
  ierr = ADDASetRefinement(*addac, adda->refine, adda->dofrefine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetInjection"
/*@
   ADDAGetInjection - Gets injection between distributed arrays.

   Collective on ADDA

   Input Parameter:
+  adda1 - the fine ADDA object
-  adda2 - the second, coarser ADDA object

    Output Parameter:
.  ctx - the injection

   Level: developer

.keywords: distributed array, injection

.seealso: DMGetInjection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetInjection(ADDA adda1, ADDA adda2, VecScatter *ctx) {
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

/*@C
  ADDAHCiterStartup - performs the first check for an iteration through a hypercube
  lc, uc, idx all have to be valid arrays of size dim
  This function sets idx to lc and then checks, whether the lower corner (lc) is less
  than thre upper corner (uc). If lc "<=" uc in all coordinates, it returns PETSC_TRUE,
  and PETSC_FALSE otherwise.
  
  Input Parameters:
+ dim - the number of dimension
. lc - the "lower" corner
- uc - the "upper" corner

  Output Parameters:
. idx - the index that this function increases

  Level: developer
@*/
PetscTruth ADDAHCiterStartup(const PetscInt dim, const PetscInt *const lc, const PetscInt *const uc, PetscInt *const idx) {
  PetscErrorCode ierr;
  PetscInt i;

  ierr = PetscMemcpy(idx, lc, sizeof(PetscInt)*dim);
  if(ierr) {
    PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,ierr,0," ");
    return PETSC_FALSE;
  }
  for(i=0; i<dim; i++) {
    if( lc[i] > uc[i] ) {
      return PETSC_FALSE;
    }
  }
  return PETSC_TRUE;
}

/*@C
  ADDAHCiter - iterates through a hypercube
  lc, uc, idx all have to be valid arrays of size dim
  This function return PETSC_FALSE, if idx exceeds uc, PETSC_TRUE otherwise.
  There are no guarantees on what happens if idx is not in the hypercube
  spanned by lc, uc, this should be checked with ADDAHCiterStartup.
  
  Use this code as follows:
  if( ADDAHCiterStartup(dim, lc, uc, idx) ) {
    do {
      ...
    } while( ADDAHCiter(dim, lc, uc, idx) );
  }
  
  Input Parameters:
+ dim - the number of dimension
. lc - the "lower" corner
- uc - the "upper" corner

  Output Parameters:
. idx - the index that this function increases

  Level: developer
@*/
PetscTruth ADDAHCiter(const PetscInt dim, const PetscInt *const lc, const PetscInt *const uc, PetscInt *const idx) {
  PetscInt i;
  for(i=dim-1; i>=0; i--) {
    idx[i] += 1;
    if( uc[i] > idx[i] ) {
      return PETSC_TRUE;
    } else {
      idx[i] -= uc[i] - lc[i];
    }
  }
  return PETSC_FALSE;
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetAggregates"
/*@C
   ADDAGetAggregates - Gets the aggregates that map between 
   grids associated with two ADDAs.

   Collective on ADDA

   Input Parameters:
+  addac - the coarse grid ADDA
-  addaf - the fine grid ADDA

   Output Parameters:
.  rest - the restriction matrix (transpose of the projection matrix)

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: ADDARefine(), ADDAGetInjection(), ADDAGetInterpolation()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetAggregates(ADDA addac,ADDA addaf,Mat *rest)
{
  PetscErrorCode ierr=0;
  PetscInt       i;
  PetscInt       dim;
  PetscInt       dofc, doff;
  PetscInt       *lcs_c, *lce_c;
  PetscInt       *lcs_f, *lce_f;
  PetscInt       *fgs, *fge;
  PetscInt       fgdofs, fgdofe;
  ADDAIdx        iter_c, iter_f;
  PetscInt       max_agg_size;
  PetscMPIInt    comm_size;
  ADDAIdx        *fine_nodes;
  PetscInt       fn_idx;
  PetscScalar    *one_vec;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(addac, ADDA_COOKIE, 1);
  PetscValidHeaderSpecific(addaf, ADDA_COOKIE, 2);
  PetscValidPointer(rest,3);
  if (addac->dim != addaf->dim) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Dimensions of ADDA do not match %D %D", addac->dim, addaf->dim);CHKERRQ(ierr);
/*   if (addac->dof != addaf->dof) SETERRQ2(PETSC_ERR_ARG_INCOMP,"DOF of ADDA do not match %D %D", addac->dof, addaf->dof);CHKERRQ(ierr); */
  dim = addac->dim;
  dofc = addac->dof;
  doff = addaf->dof;

  ierr = ADDAGetCorners(addac, &lcs_c, &lce_c);CHKERRQ(ierr);
  ierr = ADDAGetCorners(addaf, &lcs_f, &lce_f);CHKERRQ(ierr);
  
  /* compute maximum size of aggregate */
  max_agg_size = 1;
  for(i=0; i<dim; i++) {
    max_agg_size *= addaf->nodes[i] / addac->nodes[i] + 1;
  }
  max_agg_size *= doff / dofc + 1;

  /* create the matrix that will contain the restriction operator */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);CHKERRQ(ierr);

  /* construct matrix */
  if( comm_size == 1 ) {
    ierr = ADDAGetMatrixNS(addac, addaf, MATSEQAIJ, rest);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*rest, max_agg_size, PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = ADDAGetMatrixNS(addac, addaf, MATMPIAIJ, rest);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*rest, max_agg_size, PETSC_NULL, max_agg_size, PETSC_NULL);CHKERRQ(ierr);
  }
  /* store nodes in the fine grid here */
  ierr = PetscMalloc(sizeof(ADDAIdx)*max_agg_size, &fine_nodes);CHKERRQ(ierr);
  /* these are the values to set to, a collection of 1's */
  ierr = PetscMalloc(sizeof(PetscScalar)*max_agg_size, &one_vec);CHKERRQ(ierr);
  /* initialize */
  for(i=0; i<max_agg_size; i++) {
    ierr = PetscMalloc(sizeof(PetscInt)*dim, &(fine_nodes[i].x));CHKERRQ(ierr);
    one_vec[i] = 1.0;
  }

  /* get iterators */
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &(iter_c.x));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &(iter_f.x));CHKERRQ(ierr);

  /* the fine grid node corner for each coarse grid node */
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &fgs);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &fge);CHKERRQ(ierr);

  /* loop over all coarse nodes */
  ierr = PetscMemcpy(iter_c.x, lcs_c, sizeof(PetscInt)*dim);CHKERRQ(ierr);
  if( ADDAHCiterStartup(dim, lcs_c, lce_c, iter_c.x) ) {
    do {
      /* find corresponding fine grid nodes */
      for(i=0; i<dim; i++) {
	fgs[i] = iter_c.x[i]*addaf->nodes[i]/addac->nodes[i];
	fge[i] = PetscMin((iter_c.x[i]+1)*addaf->nodes[i]/addac->nodes[i], addaf->nodes[i]);
      }
      /* treat all dof of the coarse grid */
      for(iter_c.d=0; iter_c.d<dofc; iter_c.d++) {
	/* find corresponding fine grid dof's */
	fgdofs = iter_c.d*doff/dofc;
	fgdofe = PetscMin((iter_c.d+1)*doff/dofc, doff);
	/* we now know the "box" of all the fine grid nodes that are mapped to one coarse grid node */
	fn_idx = 0;
	/* loop over those corresponding fine grid nodes */
	if( ADDAHCiterStartup(dim, fgs, fge, iter_f.x) ) {
	  do {
	    /* loop over all corresponding fine grid dof */
	    for(iter_f.d=fgdofs; iter_f.d<fgdofe; iter_f.d++) {
	      ierr = PetscMemcpy(fine_nodes[fn_idx].x, iter_f.x, sizeof(PetscInt)*dim);CHKERRQ(ierr);
	      fine_nodes[fn_idx].d = iter_f.d;
	      fn_idx++;
	    }
	  } while( ADDAHCiter(dim, fgs, fge, iter_f.x) );
	}
	/* add all these points to one aggregate */
	ierr = ADDAMatSetValues(*rest, addac, 1, &iter_c, addaf, fn_idx, fine_nodes, one_vec, INSERT_VALUES);CHKERRQ(ierr);
      }
    } while( ADDAHCiter(dim, lcs_c, lce_c, iter_c.x) );
  }

  /* free memory */
  ierr = PetscFree(fgs);CHKERRQ(ierr);
  ierr = PetscFree(fge);CHKERRQ(ierr);
  ierr = PetscFree(iter_c.x);CHKERRQ(ierr);
  ierr = PetscFree(iter_f.x);CHKERRQ(ierr);
  ierr = PetscFree(lcs_c);CHKERRQ(ierr);
  ierr = PetscFree(lce_c);CHKERRQ(ierr);
  ierr = PetscFree(lcs_f);CHKERRQ(ierr);
  ierr = PetscFree(lce_f);CHKERRQ(ierr);
  ierr = PetscFree(one_vec);CHKERRQ(ierr);
  for(i=0; i<max_agg_size; i++) {
    ierr = PetscFree(fine_nodes[i].x);CHKERRQ(ierr);
  }
  ierr = PetscFree(fine_nodes);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDASetRefinement"
/*@
   ADDASetRefinement - Sets the refinement factors of the distributed arrays.

   Collective on ADDA

   Input Parameter:
+  adda - the ADDA object
.  refine - array of refinement factors
-  dofrefine - the refinement factor for the dof, usually just 1

   Level: developer

.keywords: distributed array, refinement
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDASetRefinement(ADDA adda, PetscInt *refine, PetscInt dofrefine) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda, ADDA_COOKIE, 1);
  PetscValidPointer(refine,3);
  ierr = PetscMemcpy(adda->refine, refine, adda->dim*sizeof(PetscInt));CHKERRQ(ierr);
  adda->dofrefine = dofrefine;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetCorners"
/*@
   ADDAGetCorners - Gets the corners of the local area

   Collective on ADDA

   Input Parameter:
.  adda - the ADDA object

   Output Parameter:
+  lcorner - the "lower" corner
-  ucorner - the "upper" corner

   Both lcorner and ucorner are allocated by this procedure and will point to an
   array of size adda->dim.

   Level: beginner

.keywords: distributed array, refinement
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetCorners(ADDA adda, PetscInt **lcorner, PetscInt **ucorner) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda, ADDA_COOKIE, 1);
  PetscValidPointer(lcorner,2);
  PetscValidPointer(ucorner,3);
  ierr = PetscMalloc(adda->dim*sizeof(PetscInt), lcorner);CHKERRQ(ierr);
  ierr = PetscMalloc(adda->dim*sizeof(PetscInt), ucorner);CHKERRQ(ierr);
  ierr = PetscMemcpy(*lcorner, adda->lcs, adda->dim*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(*ucorner, adda->lce, adda->dim*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetGhostCorners"
/*@
   ADDAGetGhostCorners - Gets the ghost corners of the local area

   Collective on ADDA

   Input Parameter:
.  adda - the ADDA object

   Output Parameter:
+  lcorner - the "lower" corner of the ghosted area
-  ucorner - the "upper" corner of the ghosted area

   Both lcorner and ucorner are allocated by this procedure and will point to an
   array of size adda->dim.

   Level: beginner

.keywords: distributed array, refinement
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetGhostCorners(ADDA adda, PetscInt **lcorner, PetscInt **ucorner) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adda, ADDA_COOKIE, 1);
  PetscValidPointer(lcorner,2);
  PetscValidPointer(ucorner,3);
  ierr = PetscMalloc(adda->dim*sizeof(PetscInt), lcorner);CHKERRQ(ierr);
  ierr = PetscMalloc(adda->dim*sizeof(PetscInt), ucorner);CHKERRQ(ierr);
  ierr = PetscMemcpy(*lcorner, adda->lgs, adda->dim*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(*ucorner, adda->lge, adda->dim*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "ADDAMatSetValues"
/*@C 
   ADDAMatSetValues - Inserts or adds a block of values into a matrix. The values
   are indexed geometrically with the help of the ADDA data structure.
   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to ADDAMatSetValues() have been completed.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  addam - the ADDA geometry information for the rows
.  m - the number of rows
.  idxm - the row indices, each of the a proper ADDAIdx
+  addan - the ADDA geometry information for the columns
.  n - the number of columns
.  idxn - the column indices, each of the a proper ADDAIdx
.  v - a logically two-dimensional array of values of size m*n
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented and unsorted.
   See MatSetOption() for other options.

   Calls to ADDAMatSetValues() (and MatSetValues()) with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   Efficiency Alert:
   The routine ADDAMatSetValuesBlocked() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), ADDAMatSetValuesBlocked(),
          InsertMode, INSERT_VALUES, ADD_VALUES
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAMatSetValues(Mat mat, ADDA addam, PetscInt m, const ADDAIdx idxm[],
						  ADDA addan, PetscInt n, const ADDAIdx idxn[],
						  const PetscScalar v[], InsertMode addv) {
  PetscErrorCode ierr;
  PetscInt       *nodemult;
  PetscInt       i, j;
  PetscInt       *matidxm, *matidxn;
  PetscInt       *x, d;
  PetscInt       idx;

  PetscFunctionBegin;
  /* find correct multiplying factors */
  ierr = PetscMalloc(addam->dim*sizeof(PetscInt), &nodemult);CHKERRQ(ierr);
  nodemult[addam->dim-1] = 1;
  for(j=addam->dim-2; j>=0; j--) {
    nodemult[j] = nodemult[j+1]*(addam->nodes[j+1]);
  }
  /* convert each coordinate in idxm to the matrix row index */
  ierr = PetscMalloc(m*sizeof(PetscInt), &matidxm);CHKERRQ(ierr);
  for(i=0; i<m; i++) {
    x = idxm[i].x; d = idxm[i].d;
    idx = 0;
    for(j=addam->dim-1; j>=0; j--) {
      if( x[j] < 0 ) { /* "left", "below", etc. of boundary */
	if( addam->periodic[j] ) { /* periodic wraps around */
	  x[j] += addam->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxm[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_m;
	}
      }
      if( x[j] >= addam->nodes[j] ) { /* "right", "above", etc. of boundary */
	if( addam->periodic[j] ) { /* periodic wraps around */
	  x[j] -= addam->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxm[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_m;
	}
      }
      idx += x[j]*nodemult[j];
    }
    matidxm[i] = idx*(addam->dof) + d;
  endofloop_m:
    ;
  }
  ierr = PetscFree(nodemult);CHKERRQ(ierr);

  /* find correct multiplying factors */
  ierr = PetscMalloc(addan->dim*sizeof(PetscInt), &nodemult);CHKERRQ(ierr);
  nodemult[addan->dim-1] = 1;
  for(j=addan->dim-2; j>=0; j--) {
    nodemult[j] = nodemult[j+1]*(addan->nodes[j+1]);
  }
  /* convert each coordinate in idxn to the matrix colum index */
  ierr = PetscMalloc(n*sizeof(PetscInt), &matidxn);CHKERRQ(ierr);
  for(i=0; i<n; i++) {
    x = idxn[i].x; d = idxn[i].d;
    idx = 0;
    for(j=addan->dim-1; j>=0; j--) {
      if( x[j] < 0 ) { /* "left", "below", etc. of boundary */
	if( addan->periodic[j] ) { /* periodic wraps around */
	  x[j] += addan->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxn[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_n;
	}
      }
      if( x[j] >= addan->nodes[j] ) { /* "right", "above", etc. of boundary */
	if( addan->periodic[j] ) { /* periodic wraps around */
	  x[j] -= addan->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxn[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_n;
	}
      }
      idx += x[j]*nodemult[j];
    }
    matidxn[i] = idx*(addan->dof) + d;
  endofloop_n:
    ;
  }
  /* call original MatSetValues() */
  ierr = MatSetValues(mat, m, matidxm, n, matidxn, v, addv);CHKERRQ(ierr);
  /* clean up */
  ierr = PetscFree(nodemult);CHKERRQ(ierr);
  ierr = PetscFree(matidxm);CHKERRQ(ierr);
  ierr = PetscFree(matidxn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

