 /* this stuff should be done in each routine..... it is a waste of memory to duplicate... */
 /* Now get the indices */
/*  ierr = ISGetIndices(grid->isinlet_vdf, &grid->inlet_vdf); CHKERRQ(ierr); */
/*  ierr = ISGetIndices(grid->isoutlet_vdf, &grid->outlet_vdf); CHKERRQ(ierr); */
/*  ierr = ISGetIndices(grid->iswall_vdf, &grid->wall_vdf); CHKERRQ(ierr); */
/*  ierr = ISGetIndices(grid->isoutlet_pdf, &grid->outlet_pdf); CHKERRQ(ierr); */
/*  ierr = ISGetIndices(grid->isinlet_pdf, &grid->inlet_pdf); CHKERRQ(ierr); */

 /* get the sizes */
 /*  ierr = ISGetSize(grid->isinlet_vdf, &grid->inlet_vcount); CHKERRQ(ierr); */
/*   ierr = ISGetSize(grid->isoutlet_vdf, &grid->outlet_vcount); CHKERRQ(ierr); */
/*   ierr = ISGetSize(grid->iswall_vdf, &grid->wall_vcount); CHKERRQ(ierr); */
/*   ierr = ISGetSize(grid->isoutlet_pdf, &grid->outlet_pcount); CHKERRQ(ierr); */
/*   ierr = ISGetSize(grid->isinlet_pdf, &grid->inlet_pcount); CHKERRQ(ierr); */


  /* pre-allocate storage space for the boundary values to set */
 grid->inlet_values = (double*)PetscMalloc((grid->inlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->inlet_values);
 grid->outlet_values = (double*)PetscMalloc((grid->outlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->outlet_values);
 grid->inlet_pvalues = (double*)PetscMalloc((grid->inlet_pcount+1)*sizeof(double)); CHKPTRQ(grid->inlet_pvalues);
 /* space for inlet_coords */
  grid->inlet_coords = (double*)PetscMalloc((grid->inlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->inlet_coords);
  grid->outlet_coords = (double*)PetscMalloc((grid->outlet_vcount+1)*sizeof(double)); CHKPTRQ(grid->outlet_coords);


/* Nice stiffness test */

if(1){
    /* TEST - the stiffness matrix should have zero row sums */
    for(i=0;i<16;i++){
      test[i] = 0;
      for(j=0;j<16;j++){
	test[i] += phi->vstiff[i][j];
      }
    }
    printf("STIFFNESS test  should be all zeros \n");
    PetscDoubleView(16, test, VIEWER_STDOUT_SELF );

    /* TEST - incompressibility matrix should be zero on u =x, v= -y  */
      for(i=0;i<8;i++){ /* set u =x, v= -y  */
	test[2*i] = coords_ptr[2*i];
	test[2*i+1] = -coords_ptr[2*i+1];
      }
     for(i=0;i<4;i++){
       comp[i] = 0;
	for(j=0;j<16;j++){
	  comp[i] =+ pvalues[16*i+j]*test[j];
	}
     }
printf("INCOMPRESSIBILITY test, in the following, the sums should all be zero (integrate against constant), r2+r3 -r1 -r4 = 0 (against x), r3+r4-r1-r2 (against y), (r1+r3-r2-r4 -against xy):\n, so should be all zeros");
  printf("INCOMPRESSIBILITY test, on u =x, v= -y:\n");
    PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );

    /* TEST - incompressibility matrix should be zero  */
      for(i=0;i<8;i++){ /* set u =3x+7, v= -3y+10x  */
	test[2*i] = 3*coords_ptr[2*i]+7;
	test[2*i+1] = -3*coords_ptr[2*i+1]+ 10*coords_ptr[2*i];
      }
     for(i=0;i<4;i++){
       comp[i] = 0;
	for(j=0;j<16;j++){
	  comp[i] =+ pvalues[16*i+j]*test[j];
	}
     }
  printf("INCOMPRESSIBILITY test, on  u =3x+7, v= -3y+10x  \n");
  PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );
  

   /* TEST - incompressibility matrix should be zero  */
      for(i=0;i<8;i++){ /* set u =1, v= 2  */
	test[2*i] = 1;
	test[2*i+1] = 2;
      }
     for(i=0;i<4;i++){
       comp[i] = 0;
	for(j=0;j<16;j++){
	  comp[i] =+ pvalues[16*i+j]*test[j];
	}
     }
  printf("INCOMPRESSIBILITY test, on  u =1, v=2  \n");
  PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );
  


 
      for(i=0;i<8;i++){ /* set u = x*x, v = -2x*y */
	test[2*i] = coords_ptr[2*i]*coords_ptr[2*i];
	test[2*i+1] = -2*coords_ptr[2*i]*coords_ptr[2*i+1];
      }
     for(i=0;i<4;i++){
       comp[i] = 0;
	for(j=0;j<16;j++){
	  comp[i] =+ pvalues[16*i+j]*test[j];
	}
     }
  printf("INCOMPRESSIBILITY test, should be all zeros on set u = x*x, v = -2x*y  \n");
    PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );
    }
 

if(1){
   /* test for dx and dy, since sum of the elements is 1, sum of the dx, dy should be zero at each point*/
 /* TEST - the stiffness matrix should have zero row sums */
    for(ii=0;ii<9;ii++){
      test[ii] = 0;
      for(jj=0;jj<8;jj++){
	test[ii] += phi->dx[jj][ii];
      }
    }
    printf(" sum of dx  should be all zeros \n");
    PetscDoubleView(8, test, VIEWER_STDOUT_SELF );

   for(ii=0;ii<9;ii++){
      test[ii] = 0;
      for(jj=0;jj<8;jj++){
	test[ii] += phi->dy[jj][ii];
      }
    }
    printf(" sum of dy  should be all zeros \n");
    PetscDoubleView(8, test, VIEWER_STDOUT_SELF );

    /* TEST - the stiffness matrix should have zero row sums */
    for(ii=0;ii<16;ii++){
      test[ii] = 0;
      for(jj=0;jj<16;jj++){
	test[ii] += phi->vstiff[ii][jj];
      }
    }
    printf("STIFFNESS test  should be all zeros \n");
    PetscDoubleView(16, test, VIEWER_STDOUT_SELF );

/* TEST - incompressibility matrix should be zero on u =x, v= -y  */
      for(ii=0;ii<8;ii++){ /* set u =x, v= -y  */
	test[2*ii] = coords_ptr[2*ii];
	test[2*ii+1] = -coords_ptr[2*ii+1];
      }
      for(ii=0;ii<4;ii++){comp[ii] = 0;  for(jj=0;jj<16;jj++){  comp[ii] =+ pvalues[16*ii+jj]*test[jj];}}
      printf("INCOMPRESSIBILITY test, in the following, the sums should all be zero (integrate against constant), r2+r3 -r1 -r4 = 0 (against x), r3+r4-r1-r2 (against y), (r1+r3-r2-r4 -against xy):\n, so should be all zeros");
  printf("INCOMPRESSIBILITY test, on u =x, v= -y:\n");
    PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );


    /* TEST - incompressibility matrix should be zero  */
      for(ii=0;ii<8;ii++){ 
	test[2*ii] = 1;
	test[2*ii+1] = 1;
      }
     for(ii=0;ii<4;ii++){comp[ii] = 0;  for(jj=0;jj<16;jj++){  comp[ii] =+ phi->presult[ii][jj]*test[jj];}}
  printf("INCOMPRESSIBILITY test, on  u =1, v= 1  \n");
  PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );



   /* TEST - incompressibility matrix should be zero  */
      for(ii=0;ii<8;ii++){ /* set u =1, v= 2  */
	test[2*ii] = 1;
	test[2*ii+1] = 2;
      }
      for(ii=0;ii<4;ii++){comp[ii] = 0;  for(jj=0;jj<16;jj++){  comp[ii] =+ phi->presult[ii][jj]*test[jj];}}
  printf("INCOMPRESSIBILITY test, on  u =1, v=2  \n");
  PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );
   /* TEST - incompressibility matrix should be zero  */
      for(ii=0;ii<8;ii++){ /* set u =2, v= 2  */
	test[2*ii] = 2;
	test[2*ii+1] = 2;
      }
      for(ii=0;ii<4;ii++){comp[ii] = 0;  for(jj=0;jj<16;jj++){  comp[ii] =+ phi->presult[ii][jj]*test[jj];}}
  printf("INCOMPRESSIBILITY test, on  u =2, v=2  \n");
  PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );
   /* TEST - incompressibility matrix should be zero  */
      for(ii=0;ii<8;ii++){ /* set u =20, v= 2  */
	test[2*ii] = 20;
	test[2*ii+1] = 2;
      }
      for(ii=0;ii<4;ii++){comp[ii] = 0;  for(jj=0;jj<16;jj++){  comp[ii] =+ phi->presult[ii][jj]*test[jj];}}
  printf("INCOMPRESSIBILITY test, on  u =20, v=2  \n");
  PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );




      for(ii=0;ii<8;ii++){ /* set u = x*x, v = -2x*y */
	test[2*ii] = coords_ptr[2*ii]*coords_ptr[2*ii];
	test[2*ii+1] = -2*coords_ptr[2*ii]*coords_ptr[2*ii+1];
      }
      for(ii=0;ii<4;ii++){comp[ii] = 0;  for(jj=0;jj<16;jj++){  comp[ii] =+phi->presult[ii][jj]*test[jj];}}
      printf("INCOMPRESSIBILITY test, should be all zeros on set u = x*x, v = -2x*y  \n");
      PetscDoubleView(4, comp, VIEWER_STDOUT_SELF );

   
}
