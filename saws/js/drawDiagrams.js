//this recursive function should always be called on the root solver (endtag = "0").
//target_endtag is the endtag of the solver option that is currently being asked. the reason we have this as a parameter is because there is simply not enough space to display a diagram of the ENTIRE solver. so, we only display a path to the current solver option
//x,y are the x and y coordinates of the upper lefthand corner of the svg (should initially be called with 0,0)
//this function returns a string that needs to be put into an svg in the html page
function drawDiagrams(endtag,target_endtag,x,y) {

    alert("called on:"+endtag+"     and:"+target_endtag);

    var numChildren = getSawsNumChildren(endtag);

    if(numChildren == 0) //base case. no children.
        return "";
    if(target_endtag.indexOf(endtag) != 0) //base case. endtag is not on the path to target_endtag.
        return "";

    var index = getSawsIndex(endtag);
    var ret   = "";

    if(sawsInfo[index].pc == "fieldsplit") { //draw fieldsplit diagram
        var colors = ["green","blue","red"];
        var layer = 0;
        ret += "<polygon points=\""+x+","+y+" "+(x+400)+","+y+" "+(x+400)+","+(y+400)+" "+x+","+(y+400)+"\" style=\"fill:khaki;stroke:black;stroke-width:1\"></polygon>";
        layer = 1;

        ret += drawFieldsplit(endtag,0,target_endtag,x,y,400);

        function drawFieldsplit(endtag,level,target_endtag,x,y,size) {//input is the id of the fieldsplit. for example "0". (x,y) is the upper lefthand corner. size is the size of one side of the parent square (in pixels)
            //work = draw the children of the fieldsplit then call draw on each child
            if(target_endtag.indexOf(endtag) != 0)
                return ""; //endtag is not on the path to the target_endtag

            var ret = "";

            var numChildren = getSawsNumChildren(endtag);
            if(numChildren == 0)
                return;
            var colorNum = level;//the depth of the fieldsplit within other fieldsplits

            for(var i=0; i<numChildren; i++) {
                var side   = size/(numChildren+1);//leave one extra block of space
                var curr_x = x + i*side;
                var curr_y = y + i*side;

                ret += "<polygon points=\""+curr_x+","+curr_y+" "+(curr_x+side)+","+curr_y+" "+(curr_x+side)+","+(curr_y+side)+" "+curr_x+","+(curr_y+side)+"\" style=\"fill:"+colors[colorNum]+";stroke:black;stroke-width:1\"></polygon>";

                var childID = endtag + "_" + i;
                //only draw if child is indeed a fieldsplit
                var child_index = getSawsIndex(childID);
                if(child_index != -1 && sawsInfo[child_index].pc == "fieldsplit")
                    ret += drawFieldsplit(childID, level+1, target_endtag, curr_x, curr_y, size/numChildren);

                //if child is mg, then it is time to switch drawing methods
                else if(child_index != -1 && sawsInfo[child_index].pc == "mg") {
                    var possible = drawDiagrams(childID,target_endtag,x+size+20+146,y+i*side+side/2);
                    if(possible != "") {//don't draw the arrow if there is no diagram following
                        ret += "<image x=\""+(x+size+20)+"\" y=\""+(y+i*side+side/2)+"\" width=\"146\" height=\"26\" xlink:href=\"images/arrow.png\"></image>";
                        ret += possible;
                    }
                }
            }
            var side = size/(numChildren+1);//side of the blank square
            var blank_x = x + numChildren*side;
            var blank_y = y + numChildren*side;

            var inc = side/4;//the increment
            for(var i=1; i<4; i++) {
                var x_coord = blank_x + i*inc;
                var y_coord = blank_y + i*inc;
                ret += "<circle cx=\""+x_coord+"\" cy=\"" + y_coord + "\" r=\"2\" stroke=\"black\" stroke-width=\"2\" fill=\"black\"></circle>";
            }

            return ret;
        }
    }

    else if(sawsInfo[index].pc == "mg") { //draw multigrid diagram. multigrid diagram doesn't use an inner recursive function because it's not that complex to draw.

        var selectedChild = "";

        //generate a parallelogram for each layer
        for(var i=0; i<numChildren; i++) { //i represents the multigrid level (i=0 would be coarse)
            var dim = 3+2*i;//dimxdim grid
            //ret += "<polygon points=\"0,141 141,0 465,0 324,141\" style=\"fill:khaki;stroke:black;stroke-width:1\"> </polygon>";
            var global_downshift = 141*i + 68*i;

            ret += "<polygon points=\""+x+","+(y+141+global_downshift)+" "+ (x+141)+","+(y+global_downshift)+" "+(x+465)+","+(y+global_downshift)+" "+(x+324)+","+(y+141+global_downshift)+"\" style=\"fill:khaki;stroke:black;stroke-width:1\"> </polygon>";

            for(var j=1; j<dim; j++) {//draw 'vertical' lines
                var inc = 324/dim;//parallogram is 324 wide and 200 on the slant side (1.6x)
                var shift = j*inc;
                var top_shift = shift + 141;
                ret += "<line x1=\""+(x+shift)+"\" y1=\""+(y+141+global_downshift)+"\" x2=\""+(x+top_shift)+"\" y2=\""+(y+global_downshift)+"\" style='stroke:black;stroke-width:1'></line>";
            }
            for(var j=1; j<dim; j++) {//draw horizonal lines
                var inc = 141/dim;//parallelogram is 141 tall
                var horiz_shift = (141/dim) * j;
                var horiz_shift_end = horiz_shift + 324;

                var shift = 141 - inc * j;
                ret += "<line x1=\""+(x+horiz_shift)+"\" y1=\""+(y+shift+global_downshift) +"\" x2=\""+(x+horiz_shift_end)+"\" y2=\""+(y+shift+global_downshift)+"\" style='stroke:black;stroke-width:1'></line>";
            }

            if(i != numChildren-1)//add transition arrows image if there are more grids left
                ret += "<image x=\""+x+"\" y=\""+(y+141+global_downshift)+"\" width=\"349\" height=\"68\" xlink:href=\"images/transition.bmp\"></image>"; //images in svg are handled differently. can't simply use <img>

            //if the current child is the one that is on the path to the target, then record it
            var child_endtag = endtag + "_" + i;

            if(target_endtag.indexOf(child_endtag) == 0) { //this can only happen with 1 child (the one that is on the path to the target)
                selectedChild = i;
            }
        }

        var new_x = x + 465;
        var new_y = y + (selectedChild) * (141+68);//manipulate based on selectedChild

        //recursively draw the rest of the path to target_endtag

        var possible  = drawDiagrams(endtag+"_"+selectedChild,target_endtag,new_x+146,new_y);
        if(possible != "") {//only add the arrow if something was actually drawn
            ret += "<image x=\""+(new_x-45)+"\" y=\""+(new_y+70)+"\" width=\"146\" height=\"26\" xlink:href=\"images/arrow.png\"></image>";
            ret += possible
        }

    }

    return ret;

}