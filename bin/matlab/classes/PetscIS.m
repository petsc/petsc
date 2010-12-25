classdef PetscIS < PetscObject
  methods
    function obj = PetscIS()
      comm = PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libpetsc', 'ISCreate',comm ,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'ISSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = GeneralSetIndices(obj,indices)
      indices = indices - 1;  
      err = calllib('libpetsc', 'ISGeneralSetIndices', obj.pobj,length(indices),indices);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'ISView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'ISView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'ISDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 

