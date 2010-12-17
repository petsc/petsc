function err = PetscObjectsView(viewer)
%
if (nargin == 0)
  err = calllib('libpetsc', 'PetscOptionsView',0);PetscCHKERRQ(err);
else
  err = calllib('libpetsc', 'PetscOptionsView',viewer.pobj);PetscCHKERRQ(err);
end


