function[xl,xu] = MSA_Plate(xl,xu,user)

%% This file only implements a rectangular plate.
xl.Set(-10000000);
xu.Set(100000000);
xl_2d = user.dm.VecGetArray(xl);
bmx = user.bmx; bmy = user.bmy;
mx = user.mx; my = user.my;
bheight = user.bheight;
for (j = 1:my)
  for (i = 1:mx)
    if ((i >(mx-bmx)/2) && (i <= mx-(mx-bmx)/2) && (j > (my-bmy)/2) && (j <= my-(my-bmy)/2))
      xl_2d(i,j) = bheight;
    end
  end
end
