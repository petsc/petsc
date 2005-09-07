
/* Prototypes missing from PLAPACK. Contributed by Opale Hullen (opale.hullen@utoronto.ca) */
int PLA_Comm_1D_to_2D_ratio (MPI_Comm comm1d, double ratio, MPI_Comm *comm2d);
int PLA_Comm_1D_to_2D(MPI_Comm comm1d, int nprows, int npcols, MPI_Comm *comm2d);
int PLA_Init(MPI_Comm comm);
int PLA_Initialized(int*);
int PLA_Finalize();
int PLA_Temp_create (int nb, int zero_or_one, PLA_Template *templ_ret);
int PLA_Temp_free(PLA_Template *);
int PLA_Temp_comm_all_info(PLA_Template templ, MPI_Comm *comm, int *me, int *nprocs);
int PLA_Temp_comm_row_info(PLA_Template temp1, MPI_Comm *comm, int *me, int *nprocs);
int PLA_Temp_comm_col_info(PLA_Template temp1, MPI_Comm *comm, int *me, int *nprocs);
int PLA_Temp_comm_row_rank(PLA_Template templ, int *);
int PLA_Temp_comm_row_size(PLA_Template templ, int *);
int PLA_Temp_comm_col_rank(PLA_Template templ, int *);
int PLA_Temp_comm_col_size(PLA_Template templ, int *);
int PLA_Temp_zero_or_one(PLA_Template templ, int *);
int PLA_Temp_nb(PLA_Template templ, int *nb);
int PLA_Environ_nb_alg(int operation, PLA_Template templ, int *sug_nb);
int pla_Environ_set_nb_alg(int, int nb_alg); 
int PLA_Obj_free(PLA_Obj *);
int PLA_Matrix_create( MPI_Datatype datatype, int, int, PLA_Template, int, int, PLA_Obj * );
int PLA_Set_error_checking( int, int, int, int );
int PLA_Obj_global_info( PLA_Obj, int *, int *, int *, int *, int *,  int *, int * );
int PLA_Obj_local_info( PLA_Obj, int *, int *, void **, int *, int * );
int PLA_Obj_local_length( PLA_Obj, int * );
int PLA_Obj_local_width( PLA_Obj, int * );
int PLA_Obj_local_ldim( PLA_Obj, int * );

int PLA_Obj_global_length(PLA_Obj, int *);
int PLA_Obj_global_width(PLA_Obj, int *);
int PLA_API_begin();
int PLA_API_end();
int PLA_Obj_API_open(PLA_Obj);
int PLA_Obj_API_close(PLA_Obj);
int PLA_API_axpy_matrix_to_global(int, int, void *, void *, int, PLA_Obj, int, int);
/* int PLA_API_axpy_vector_to_global(int, void*, void*, int, PLA_Obj_view_struct*, int); */
int PLA_API_axpy_vector_to_global(int, void*, void*, int, PLA_Obj, int);
int PLA_Mscalar_create  ( MPI_Datatype, int, int, int, int, PLA_Template, PLA_Obj * );
int PLA_Obj_set_to_zero(PLA_Obj);
int PLA_General_solve(PLA_Obj    A, PLA_Obj   ipiv,    PLA_Obj B );
int PLA_Mvector_create( MPI_Datatype, int, int, PLA_Template, int, PLA_Obj * );
int PLA_Pos_def_solve ( PLA_Obj A, PLA_Obj B);
int PLA_LU(PLA_Obj A, PLA_Obj pivots);
int PLA_LU_enter(PLA_Obj A, PLA_Obj pivots);
int PLA_LU_right( int nb_alg, PLA_Obj A, PLA_Obj pivots );
int PLA_Obj_view_all( PLA_Obj, PLA_Obj * );

int PLA_Apply_pivots_to_rows(PLA_Obj, PLA_Obj);
int PLA_Gemm(int, int, PLA_Obj, PLA_Obj, PLA_Obj, PLA_Obj, PLA_Obj);
int PLA_Trsm(int, int, int, int, PLA_Obj, PLA_Obj, PLA_Obj);
int PLA_Trsv(int, int, int, PLA_Obj, PLA_Obj);
int PLA_Chol(int,  PLA_Obj);
int PLA_Create_constants_conf_to( PLA_Obj , PLA_Obj *, PLA_Obj *, PLA_Obj *);
int PLA_Obj_API_sync(PLA_Obj);
