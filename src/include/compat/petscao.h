#ifndef _COMPAT_PETSC_AO_H
#define _COMPAT_PETSC_AO_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define AOType char*
#define AOBASIC               "basic"
#define AOADVANCED            "advanced"
#define AOMAPPING             "mapping"
#define AOMEMORYSCALABLE      "memoryscalable"
#undef __FUNCT__
#define __FUNCT__ "AOGetType"
static PetscErrorCode
AOGetType(AO ao, const AOType *aotype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  PetscValidPointer(aotype,3);
  switch (((PetscObject)ao)->type) {
  case AO_BASIC:    *aotype = AOBASIC;    break;
  case AO_ADVANCED: *aotype = AOADVANCED; break;
  case AO_MAPPING:  *aotype = AOMAPPING;  break;
  default:          *aotype = 0;          break;
  }
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define AOInitializePackage(p) (0)
#endif

#endif /* _COMPAT_PETSC_AO_H */
