#define DESTROY(PetscType, CLASSID)                     \
PETSC_STATIC_INLINE                                     \
PetscErrorCode PetscType##Destroy_new(PetscType *obj)   \
{                                                       \
  PetscType      tmp = 0;                               \
  PetscErrorCode ierr;                                  \
  PetscFunctionBegin;                                   \
  PetscValidPointer((obj),1);                           \
  if (!(*(obj))) PetscFunctionReturn(0);                \
  tmp = *(obj); *(obj) = 0;                             \
  if (CLASSID == -1)                                    \
    {PetscValidPointer(tmp,1); }                        \
  else if (CLASSID == PETSC_OBJECT_CLASSID)             \
    {PetscValidHeader(tmp,1);}                          \
  else                                                  \
    {PetscValidHeaderSpecific(tmp,CLASSID,1);}          \
  ierr = PetscType##Destroy(tmp);CHKERRQ(ierr);         \
  PetscFunctionReturn(0);                               \
}                                                       \
/**/
#undef  __FUNCT__
#define __FUNCT__ "User provided function\0:Destroy"
DESTROY(PetscObject            , PETSC_OBJECT_CLASSID   )
DESTROY(PetscFwk               , PETSC_FWK_CLASSID      )
DESTROY(PetscViewer            , PETSC_VIEWER_CLASSID   )
DESTROY(PetscRandom            , PETSC_RANDOM_CLASSID   )
DESTROY(IS                     , IS_CLASSID             )
DESTROY(ISColoring             , -1                     )
DESTROY(ISLocalToGlobalMapping , IS_LTOGM_CLASSID       )
DESTROY(Vec                    , VEC_CLASSID            )
DESTROY(VecScatter             , VEC_SCATTER_CLASSID    )
DESTROY(Mat                    , MAT_CLASSID            )
DESTROY(MatFDColoring          , MAT_FDCOLORING_CLASSID )
DESTROY(PC                     , PC_CLASSID             )
DESTROY(KSP                    , KSP_CLASSID            )
DESTROY(SNES                   , SNES_CLASSID           )
DESTROY(TS                     , TS_CLASSID             )
DESTROY(AO                     , AO_CLASSID             )
DESTROY(DM                     , DM_CLASSID             )

#undef PetscObjetDestroy
#undef PetscFwkDestroy
#undef PetscViewerDestroy
#undef PetscRandomDestroy
#undef ISDestroy
#undef ISColoringDestroy
#undef ISLocalToGlobalMappingDestroy
#undef VecDestroy
#undef VecScatterDestroy
#undef MatDestroy
#undef MatFDColoringDestroy
#undef PCDestroy
#undef KSPDestroy
#undef SNESDestroy
#undef TSDestroy
#undef AODestroy
#undef DMDestroy

#define PetscObjectDestroy            PetscObjectDestroy_new
#define PetscFwkDestroy               PetscFwkDestroy_new
#define PetscViewerDestroy            PetscViewerDestroy_new
#define PetscRandomDestroy            PetscRandomDestroy_new
#define ISDestroy                     ISDestroy_new
#define ISColoringDestroy             ISColoringDestroy_new
#define ISLocalToGlobalMappingDestroy ISLocalToGlobalMappingDestroy_new
#define VecDestroy                    VecDestroy_new
#define VecScatterDestroy             VecScatterDestroy_new
#define MatDestroy                    MatDestroy_new
#define MatFDColoringDestroy          MatFDColoringDestroy_new
#define PCDestroy                     PCDestroy_new
#define KSPDestroy                    KSPDestroy_new
#define SNESDestroy                   SNESDestroy_new
#define TSDestroy                     TSDestroy_new
#define AODestroy                     AODestroy_new
#define DMDestroy                     DMDestroy_new
