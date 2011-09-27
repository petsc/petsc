#ifndef _COMPAT_PETSC_DESTROY_H
#define _COMPAT_PETSC_DESTROY_H
#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))

#define DESTROY(PetscType, COOKIE)                      \
PETSC_STATIC_INLINE                                     \
PetscErrorCode PetscType##Destroy_new(PetscType *obj)   \
{                                                       \
  PetscType      tmp = 0;                               \
  PetscErrorCode ierr;                                  \
  PetscFunctionBegin;                                   \
  PetscValidPointer((obj),1);                           \
  if (!(*(obj))) PetscFunctionReturn(0);                \
  tmp = *(obj); *(obj) = 0;                             \
  if (COOKIE == -1)                                     \
    {PetscValidPointer(tmp,1); }                        \
  else if (COOKIE == PETSC_OBJECT_COOKIE)               \
    {PetscValidHeader(tmp,1);}                          \
  else                                                  \
    {PetscValidHeaderSpecific(tmp,COOKIE,1);}           \
  ierr = PetscType##Destroy(tmp);CHKERRQ(ierr);         \
  PetscFunctionReturn(0);                               \
}                                                       \
/**/
#undef  __FUNCT__
#define __FUNCT__ "User provided function\0:Destroy"
DESTROY(PetscObject            , PETSC_OBJECT_COOKIE   )
DESTROY(PetscFwk               , PETSC_FWK_COOKIE      )
DESTROY(PetscViewer            , PETSC_VIEWER_COOKIE   )
DESTROY(PetscRandom            , PETSC_RANDOM_COOKIE   )
DESTROY(IS                     , IS_COOKIE             )
DESTROY(ISColoring             , -1                    )
DESTROY(ISLocalToGlobalMapping , IS_LTOGM_COOKIE       )
DESTROY(Vec                    , VEC_COOKIE            )
DESTROY(VecScatter             , VEC_SCATTER_COOKIE    )
DESTROY(Mat                    , MAT_COOKIE            )
DESTROY(MatFDColoring          , MAT_FDCOLORING_COOKIE )
DESTROY(MatNullSpace           , MAT_NULLSPACE_COOKIE  )
DESTROY(PC                     , PC_COOKIE             )
DESTROY(KSP                    , KSP_COOKIE            )
DESTROY(SNES                   , SNES_COOKIE           )
DESTROY(TS                     , TS_COOKIE             )
DESTROY(AO                     , AO_COOKIE             )
DESTROY(DM                     , DM_COOKIE             )

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
#undef MatNullSpaceDestroy
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
#define MatNullSpaceDestroy           MatNullSpaceDestroy_new
#define PCDestroy                     PCDestroy_new
#define KSPDestroy                    KSPDestroy_new
#define SNESDestroy                   SNESDestroy_new
#define TSDestroy                     TSDestroy_new
#define AODestroy                     AODestroy_new
#define DMDestroy                     DMDestroy_new

#endif
#endif
