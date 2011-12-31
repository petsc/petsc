function err = PetscObjectsView(viewer)
%
if (nargin == 0)
  err = calllib('libpetsc', 'PetscObjectsView',0);PetscCHKERRQ(err);
else
  err = calllib('libpetsc', 'PetscObjectsView',viewer.pobj);PetscCHKERRQ(err);
end


