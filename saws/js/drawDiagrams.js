//this recursive function should always be called on the root solver (endtag = "0")
function drawDiagrams(endtag) {

    var numChildren = getSawsNumChildren(endtag);

    if(numChildren == 0) //base case
        return;

    var index = getSawsIndex(endtag);
    var ret   = "";

    if(sawsInfo[index].pc == "fieldsplit") { //draw fieldsplit diagram
        var colors = ["green","blue"];
        var layer = 0;
        $("#fieldsplitDiagram").html("<svg id=\"svgFieldsplit\" width='400' height='400'> <polygon points=\"0,0 0,400 400,400 400,0\" style=\"fill:khaki;stroke:black;stroke-width:1\"> </svg>");
        layer = 1;

        drawFieldsplit("0",0,0,400);

        function drawFieldsplit(fieldsplit,x,y,size) {//input is the id of the fieldsplit. for example "0". (x,y) is the upper lefthand corner. size is the size of one side of the parent square (in pixels)
            //work = draw the children of the fieldsplit then call draw on each child
            var numChildren = getSawsNumChildren(fieldsplit);
            if(numChildren == 0)
                return;
            var colorNum = fieldsplit.length - 1;

            for(var i=0; i<numChildren; i++) {
                var side   = size/(numChildren+1);//leave one extra block of space
                var curr_x = x + i*side;
                var curr_y = y + i*side;

                var string = "<polygon points=\""+curr_x+","+curr_y+" "+(curr_x+side)+","+curr_y+" "+(curr_x+side)+","+(curr_y+side)+" "+curr_x+","+(curr_y+side)+"\" style=\"fill:"+colors[colorNum]+";stroke:black;stroke-width:1\"> </svg>";

                $("#svgFieldsplit").append(string);
                var childID = fieldsplit + i;
                drawFieldsplit(childID, curr_x, curr_y, size/numChildren);
            }
            var side = size/(numChildren+1);//side of the blank square
            var blank_x = x + numChildren*side;
            var blank_y = y + numChildren*side;

            var inc = side/4;//the increment
            for(var i=1; i<4; i++) {
                var x_coord = blank_x + i*inc;
                var y_coord = blank_y + i*inc;
                $("#svgFieldsplit").append("<circle cx=\""+x_coord+"\" cy=\"" + y_coord + "\" r=\"1\" stroke=\"black\" stroke-width=\"2\" fill=\"black\">");
            }
        }
    }

    else if(sawsInfo[index].pc == "mg") { //draw multigrid diagram

        //generate a parallelogram for each layer
            for(var i=0; i<=_level; i++) { //i represents the multigrid level (i=0 would be coarse)
                var dim = 3+2*i;//dimxdim grid
                $("#multigridDiagram").append("<svg id=\"svg"+i+"\" width='465' height='142'> <polygon points=\"0,141 141,0 465,0 324,141\" style=\"fill:khaki;stroke:black;stroke-width:1\"> </svg>");//the sides of the parallogram follow the golden ratio so that the original figure was a golden rectangle. the diagram is slanted at a 45 degree angle.

                for(var j=1; j<dim; j++) {//draw 'vertical' lines
                    var inc = 324/dim;//parallogram is 324 wide and 200 on the slant side (1.6x)
                    var shift = j*inc;
                    var top_shift = shift + 141;
                    $("#svg"+i).append("<line x1=\""+shift+"\" y1='141' x2=\""+top_shift+"\" y2='0' style='stroke:black;stroke-width:1'></line>");
                }
                for(var j=1; j<dim; j++) {//draw horizonal lines
                    var inc = 141/dim;//parallelogram is 141 tall
                    var horiz_shift = (141/dim) * j;
                    var horiz_shift_end = horiz_shift + 324;

                    var shift = 141 - inc * j;
                    $("#svg"+i).append("<line x1=\""+horiz_shift+"\" y1=\""+shift +"\" x2=\""+horiz_shift_end+"\" y2=\""+shift+"\" style='stroke:black;stroke-width:1'></line>");
                }
                //put text here
                if(i!=0)
                    $("#multigridDiagram").append("<span>Level "+i+"</span><br>");
                else
                    $("#multigridDiagram").append("<span>Coarse Grid (Level 0)</span><br>");

                if(i != _level)//add transition arrows image if there are more grids left
                    $("#multigridDiagram").append("<img src='images/transition.bmp' alt='Error Loading Multigrid Transition Arrows'><br>");
            }

    }

}