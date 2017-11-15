#include <petsc.h>
#include <lua.h>

int lua_VecCreate(lua_State *L)
{
  PetscErrorCode ierr;
  Vec            vec;

  ierr = VecCreate(PETSC_COMM_SELF,&vec);
  lua_pushlightuserdata (L,vec);
  return 1;
}

int lua_VecSetSize(lua_State *L)
{
  PetscErrorCode ierr;
  Vec            vec;
  PetscInt       n;
  int            isnum;

  vec = (Vec) lua_touserdata(L,1);
  n   = (PetscInt) lua_tointegerx(L,2,&isnum);
  ierr = VecSetSizes(vec,n,n);
  return 0;
}

int luaopen_libpetsc(lua_State *L)
{
  PetscInitializeNoArguments();
  lua_register(L,"VecCreate",(lua_CFunction)lua_VecCreate);
  lua_register(L,"VecSetSize",(lua_CFunction)lua_VecSetSize);
  return(0);
}
