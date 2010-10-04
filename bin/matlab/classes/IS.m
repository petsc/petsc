classdef IS < PetscObject
  methods
    function obj = IS()
      [err,obj.pobj] = calllib('libpetsc', 'ISCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'ISSetType', obj.pobj,name);
    end
    function err = GeneralSetIndices(obj,indices)
      err = calllib('libpetsc', 'ISGeneralSetIndices', obj.pobj,length(indices),indices);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'ISView', obj.pobj,viewer.pobj);
    end
  end
end

 

