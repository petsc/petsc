/*
     Loads the quadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#include "appctx.h"

#undef __FUNC__
#define __FUNC__ "AppCxtCreate"
int AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  int    ierr,flag;
  Viewer binary;
  char   filename[256];
  AppEquations *equations;
  AppView *view;
  AppElement *element;

  (*appctx) = (AppCtx *) PetscMalloc(sizeof(AppCtx));CHKPTRQ(*appctx);
  (*appctx)->comm = comm;
  equations =&(*appctx)->equations;
  view = &(*appctx)->view;
  element = &(*appctx)->element;

  /*-----------------------------------------------------------------------
     Load in the grid database
    ---------------------------------------------------------------------------*/
  ierr = OptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = ViewerFileOpenBinary((*appctx)->comm,filename,BINARY_RDONLY,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata); CHKERRQ(ierr);
  ierr = ViewerDestroy(binary); CHKERRQ(ierr);


 
  /*----------------------------------------------------
 setup viewing options 
--------------------------------------------------------*/

  ierr = OptionsHasName(PETSC_NULL,"-matlab_graphics",&view->matlabgraphics); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_matrix",&view->show_matrix); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_vector",&view->show_vector); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-monitor",&view->monitor); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_is",&view->show_is); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_ao",&view->show_ao); CHKERRQ(ierr);

  /*----------------------------------------------------
 setup the equations/ boundary conditions 
--------------------------------------------------------*/
  /* timestepping */
  equations->Nsteps = 8;
  equations->Nplot = 1;

  equations->initial_time = 0.0;
  equations->final_time = 1.0;
  equations->amp = 1;
  equations->offset = 0;
  equations->frequency = 1;
  ierr = OptionsGetInt(0,"-Nplot", &equations->Nplot, &flag);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-Nsteps", &equations->Nsteps, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-initial_time", &equations->initial_time, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-final_time", &equations->final_time, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-amp", &equations->amp, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-offset", &equations->offset, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-frequency", &equations->frequency, &flag);CHKERRQ(ierr);

  /* basic parameters */
  equations->eta = 1.0;
  ierr = OptionsGetDouble(0,"-viscosity", &equations->eta, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-tweak", &equations->tweak, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-penalty", &equations->penalty, &equations->penalty_flag);CHKERRQ(ierr);
  
  /* solution technique parameters */
  ierr = OptionsHasName(0,"-stokes",&equations->stokes_flag);CHKERRQ(ierr);
  ierr = OptionsHasName(0,"-convection",&equations->convection_flag);CHKERRQ(ierr);
  ierr = OptionsHasName(0,"-precon:conv",&equations->preconconv_flag);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-precon:conv",&equations->inner_steps, &equations->preconconv_flag);CHKERRQ(ierr);
  if (equations->preconconv_flag) equations->convection_flag = 1;
 /* manually set bc */
 ierr = OptionsHasName(0,"-vin",&equations->vin_flag);CHKERRQ(ierr);
 ierr = OptionsHasName(0,"-vout",&equations->vout_flag);CHKERRQ(ierr);
 ierr = OptionsHasName(0,"-wall",&equations->wall_flag);CHKERRQ(ierr);
 ierr = OptionsHasName(0,"-ywall",&equations->ywall_flag);CHKERRQ(ierr);
 ierr = OptionsHasName(0,"-pout",&equations->pout_flag);CHKERRQ(ierr);
 ierr = OptionsHasName(0,"-pin",&equations->pin_flag);CHKERRQ(ierr);
 ierr = OptionsHasName(0,"-dirichlet",&equations->dirichlet_flag);CHKERRQ(ierr);

  /* choose the problem and  set the appropriate bc */

 ierr = OptionsHasName(0,"-shear",&equations->shear_flag);CHKERRQ(ierr);
 if(equations->shear_flag){
   equations->vin_flag = 1; equations->vout_flag = 1;}

 ierr = OptionsHasName(0,"-parabolic",&equations->parabolic_flag);CHKERRQ(ierr);
 if(equations->parabolic_flag){
   equations->vin_flag = 1; equations->wall_flag = 1;}

 ierr = OptionsHasName(0,"-cylinder",&equations->cylinder_flag);CHKERRQ(ierr);
 if(equations->cylinder_flag){
   equations->vin_flag = 1; equations->ywall_flag = 1; equations->vout_flag = 1;}

 ierr = OptionsHasName(0,"-cavity",&equations->cavity_flag);CHKERRQ(ierr);
 if(equations->cavity_flag){
      equations->vin_flag = 1; equations->wall_flag = 1; equations->vout_flag = 1;}

 /* check for no bc set */
 if(equations->vin_flag == 0 && equations->vout_flag== 0){
   /* printf("No velocity bc set, setting vin\n");   equations->vin_flag = 1;*/
  printf("No velocity bc set,\n");   }

 if(equations->wall_flag == 0 && equations->ywall_flag== 0 && equations->dirichlet_flag == 0){
  /*  printf("No wall bc set, setting wall\n");   equations->wall_flag = 1; */  
printf("No wall bc set\n");  
  }

 /*-------------------------------------------------------------------------
 setup the types of elements and the quadrature to be used
---------------------------------------------------------------------------*/
element->vel_basis_count =9; 
element->vel_quad_count = 9;
element->p_basis_count = 4;
element->p_quad_count = 9;
element->dim = 2;
element->df_element_count = element->dim*element->vel_basis_count+element->p_basis_count; 

/* set the quadrature weights */
ierr = SetQuadrature(element);CHKERRQ(ierr);
/* assign the quadrature weights */
 element->vweights = element->BiquadWeights;


 /*-------------------------------------------------------------------------
 setup the debugging options 
---------------------------------------------------------------------------*/

 /*------------------------------------------------------------------------
      Setup the local data structures 
----------------------------------------------------------------------------*/

  /*     Generate the local numbering of cells and vertices  */
  ierr = AppCtxSetLocal(*appctx); CHKERRA(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetLocal"
/*  AppCtxSetLocal - Sets the local numbering data structures for the grid.*/

/* don't need cell_global and df_global IS after this fucntion, so could take them out of appctx.  */

/* Right now there is a lot of redundancy in the data we carry.  Later clean this up */
int AppCtxSetLocal(AppCtx *appctx)
{
  AOData  ao = appctx->aodata;
  AppGrid  *grid = &appctx->grid;
  AppEquations *equations = &appctx->equations;
  
  int   ierr, i;
  int *df_ptr;
  double *df_coords;
  ISLocalToGlobalMapping ltogcell;
  
  if(appctx->view.show_ao){printf("ao"); 
  ierr = AODataView(ao, VIEWER_STDOUT_SELF );CHKERRA(ierr); }
 
   /* get the local cells and vertices, and the local to global mapping for vertices */
  ierr = AODataPartitionAndSetupLocal(ao, "cell", "df",&grid->cell_global, &grid->df_global, &grid->dfltog); CHKERRQ(ierr);
 
  /* get the local cell_df  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","df",grid->cell_global,(void **)&grid->cell_df);CHKERRQ(ierr);

  /* Get the size of local objects   */
  ierr = ISGetSize(grid->cell_global,&grid->cell_n); CHKERRQ(ierr);
 
  /* this includes the ghosted dfs, since it comes from the cells.
   This is the number of dfs that this processor actually deals with. Size of local  vectors, etc. */
  ierr = ISGetSize(grid->df_global, &grid->df_count); CHKERRQ(ierr);
  /* get the number of local dfs - this comes from the actual partitioning.
   This is for constructing the global matrices and vectors, the local size is df_local_count */
 ierr = AODataKeyGetInfo(ao,"df",PETSC_NULL,&grid->df_local_count,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);


 /****** Don't really need 2 scatters for v1,v2 ,jsut do the local scatter once ************/

   /******* set up the velocity df array (for timestepping)*********/
  ierr = AODataKeyGetActiveLocalIS(ao, "df", "v", grid->df_global, 0, &grid->df_v); CHKERRQ(ierr);
  /******* set up the velocity v1 and v2df array (for statistics/data analysis)*********/
 ierr = AODataKeyGetActiveLocalIS(ao, "df", "v1", grid->df_global, 0, &grid->df_v1); CHKERRQ(ierr);
 ierr = AODataKeyGetActiveLocalIS(ao, "df", "v2", grid->df_global, 0, &grid->df_v2); CHKERRQ(ierr);

 /* view */

 if(appctx->view.show_is){  printf("grid cell_df\n");   PetscIntView(grid->cell_n*22, grid->cell_df, VIEWER_STDOUT_SELF);}
  if(appctx->view.show_is){ printf("the local to global mapping \n");  ierr = ISLocalToGlobalMappingView(grid->dfltog, VIEWER_STDOUT_SELF);}
  if(appctx->view.show_is){ printf("the local cells  \n");  ierr = ISView(grid->cell_global, VIEWER_STDOUT_SELF);}
  if(appctx->view.show_is){ printf("the local dfs  \n");  ierr = ISView(grid->df_global, VIEWER_STDOUT_SELF);}

 /*    Get the  coords corresponding to each cell */
 ierr = AODataSegmentGetIS(ao, "cell", "vcoords", grid->cell_global, (void **)&grid->cell_vcoords);CHKERRQ(ierr);
   if(0){  printf("cell_vcoords\n");   PetscDoubleView(grid->cell_n*18, grid->cell_vcoords, VIEWER_STDOUT_SELF);}

  /************************************************************/
  /*   Set up boundary related data structures                                    */
 /************************************************************/

 /* get the IS  for the different types of boundaries */ 
 /* get the sizes */
/* pre-allocate storage space for the boundary values to set */
 if(equations->vin_flag){ 
   ierr = AODataKeyGetActiveLocalIS(ao, "df", "vinlet", grid->df_global, 0, &grid->isinlet_vdf); CHKERRQ(ierr);
   ierr = ISGetSize(grid->isinlet_vdf, &grid->inlet_vcount); CHKERRQ(ierr); 
   grid->inlet_values = (double*)PetscMalloc((grid->inlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->inlet_values);
  grid->inlet_coords = (double*)PetscMalloc(2*(grid->inlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->inlet_coords);

 /* view */
 if(appctx->view.show_is){printf("isinlet_vdf\n"); ISView(grid->isinlet_vdf, VIEWER_STDOUT_SELF);}

 }

 if(equations->dirichlet_flag){
   ierr = AODataKeyGetActiveLocalIS(ao, "df", "vwall", grid->df_global, 0, &grid->iswall_vdf); CHKERRQ(ierr); 
   ierr = ISGetSize(grid->iswall_vdf, &grid->wall_vcount); CHKERRQ(ierr); 
   grid->wall_values =  (double*)PetscMalloc((grid->wall_vcount+1)*sizeof(double)); CHKPTRQ(grid->wall_values);
  grid->wall_coords = (double*)PetscMalloc(2*(grid->wall_vcount+1)*sizeof(double)); CHKPTRQ(grid->wall_coords);
 }

 if(equations->vout_flag){ 
   ierr = AODataKeyGetActiveLocalIS(ao, "df", "voutlet", grid->df_global, 0, &grid->isoutlet_vdf); CHKERRQ(ierr);
   ierr = ISGetSize(grid->isoutlet_vdf, &grid->outlet_vcount); CHKERRQ(ierr); 
  grid->outlet_values = (double*)PetscMalloc((grid->outlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->outlet_values);
  grid->outlet_coords = (double*)PetscMalloc(2*(grid->outlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->outlet_coords);

/* view */
 if(appctx->view.show_is){printf("isoutlet_vdf\n"); ISView(grid->isoutlet_vdf, VIEWER_STDOUT_SELF);}

}

 if(equations->pout_flag){ 
   ierr = AODataKeyGetActiveLocalIS(ao, "df", "poutlet", grid->df_global, 0, &grid->isoutlet_pdf); CHKERRQ(ierr); 
   ierr = ISGetSize(grid->isoutlet_pdf, &grid->outlet_pcount); CHKERRQ(ierr); 
}

 if(equations->pin_flag){ 
  ierr = AODataKeyGetActiveLocalIS(ao, "df", "pinlet", grid->df_global, 0, &grid->isinlet_pdf); CHKERRQ(ierr);
  ierr = ISGetSize(grid->isinlet_pdf, &grid->inlet_pcount); CHKERRQ(ierr); 
  grid->inlet_pvalues = (double*)PetscMalloc((grid->inlet_pcount+1)*sizeof(double)); CHKPTRQ(grid->inlet_pvalues);
}

 if( equations->wall_flag ) {
  ierr = AODataKeyGetActiveLocalIS(ao, "df", "vwall", grid->df_global, 0, &grid->iswall_vdf); CHKERRQ(ierr); 
   ierr = ISGetSize(grid->iswall_vdf, &grid->wall_vcount); CHKERRQ(ierr); 
}




  if( equations->ywall_flag ) {
  ierr = AODataKeyGetActiveLocalIS(ao, "df", "ywall", grid->df_global, 0, &grid->isywall_vdf); CHKERRQ(ierr); 
   ierr = ISGetSize(grid->isywall_vdf, &grid->ywall_vcount); CHKERRQ(ierr); 
}

 
 df_coords = (double*)PetscMalloc(2*(grid->df_count+1)*sizeof(double)); CHKPTRQ(df_coords);  
/* Now extract the actual df coords */
  ierr = AODataSegmentGetIS(ao,"df","coords",grid->df_global,(void **)&df_coords);CHKERRQ(ierr);
  /* now extract the needed ones */
  if(equations->vin_flag){
    ierr = ISGetIndices(grid->isinlet_vdf, &df_ptr); CHKERRQ(ierr);
    for( i = 0;  i < grid->inlet_vcount; i++ ){
      grid->inlet_coords[2*i] = df_coords[2*df_ptr[i]];
      grid->inlet_coords[2*i+1]  = df_coords[2*df_ptr[i]+1];
    }
  }
  if(equations->vout_flag){
    ierr = ISGetIndices(grid->isoutlet_vdf, &df_ptr); CHKERRQ(ierr);
    for( i = 0;  i < grid->outlet_vcount; i++ ){
      grid->outlet_coords[2*i] = df_coords[2*df_ptr[i]];
      grid->outlet_coords[2*i+1]  = df_coords[2*df_ptr[i]+1];
    }
  }
  if(equations->dirichlet_flag){
    ierr = ISGetIndices(grid->iswall_vdf, &df_ptr);CHKERRQ(ierr);
    for(i=0;i<grid->wall_vcount;i++){
      grid->wall_coords[2*i] = df_coords[2*df_ptr[i]];
      grid->wall_coords[2*i+1]  = df_coords[2*df_ptr[i]+1];
    }
  }


/* free up unneeded df_coords */
  ierr = PetscFree(df_coords);CHKERRQ(ierr);


 PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtDestroy"
int AppCtxDestroy(AppCtx *appctx)
{
  int        ierr;
  AOData     ao = appctx->aodata;
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;

  /* fix this */

/*   ierr = AODataDestroy(ao);CHKERRQ(ierr); */

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.dfgtol);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.dfltog);CHKERRQ(ierr);

  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AODataPartitionAndSetupLocal"
/*     AppDataSetupLocal - Sets the local numbering data structures for the grid.
Input:
        AOData ao,  - the AO
        string cell	- the name of the key
        string segment    -the name of the segment 
        IS baseIS          - the local indices of the key

Output:
       IS   issegment     - the local indices in global numbering of the segment
        ISLocalToGlobalMapping ltogsegment - the local to global mapping for the segment

do I need to do more derefencing???
possible side effect: attaching ltogcell to the database
*/
int AODataPartitionAndSetupLocal(AOData ao, char *keyname,  char *segmentname, IS *iscell, IS *issegment, ISLocalToGlobalMapping *ltog){
  ISLocalToGlobalMapping ltogcell;
  int ierr,rstart,rend,rank;

  PetscFunctionBegin;  

  /*      Partition the grid cells   */
  ierr = AODataKeyPartition(ao,keyname); CHKERRA(ierr);  

  /*      Partition the vertices subservient to the cells  */ 
  ierr = AODataSegmentPartition(ao,keyname,segmentname); CHKERRA(ierr);  

 /*     Generate the list of on processor cells   */
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,iscell);CHKERRQ(ierr);

 /*       Get the list of vertices used by those cells   */
  ierr = AODataSegmentGetReducedIS(ao,keyname,segmentname,*iscell,issegment);CHKERRQ(ierr);
 /*     Make local to global mapping of cells  */
  ierr = ISLocalToGlobalMappingCreateIS(*iscell,&ltogcell);CHKERRQ(ierr);
  /*       Make local to global mapping of  vertices  */
  ierr = ISLocalToGlobalMappingCreateIS(*issegment,ltog);CHKERRQ(ierr);
  /*        Attach the local to global mapping to the database */
  ierr = AODataKeySetLocalToGlobalMapping(ao,keyname,ltogcell);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,segmentname,*ltog);CHKERRQ(ierr);
  /* Dereference the ltogcell */
  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);

 PetscFunctionReturn(0);
}

