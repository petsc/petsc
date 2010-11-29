classdef PetscIS < PetscObject
  methods
    function obj = PetscIS()
      [err,obj.pobj] = calllib('libpetsc', 'ISCreate', 0,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'ISSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = GeneralSetIndices(obj,indices)
      indices = indices - 1;  
      err = calllib('libpetsc', 'ISGeneralSetIndices', obj.pobj,length(indices),indices);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'ISView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'ISDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 

