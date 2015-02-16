/*
 * box-algorithm that I wrote for drawing trees with different node sizes and splitting in different directions
 *
 * The following aesthetic criteria are satisfied:
 *
 * 1) No overlapping text
 * 2) When switching directions, the entire subtree in the new direction should be seen as one of the nodes in the original direction
 * 3) Sister nodes are shown on the same line
 * 4) Parent is centered at the middle of its children
 * 5) Parent is above/to the left of all children
 *
 */

var node_radius = 4; //adjust this global variable to change the size of the drawn node

//generates svg code for the given input parameters
//the x, y coordinates are the upper left hand coordinate of the drawing. should start with (0,0)
function getBoxTree(data, endtag, x, y) {

    if(data[endtag] == undefined)
        return;

    var ret         = ""; //the svg code to return
    var numChildren = getNumChildren(data,endtag);
    var pc_type     = data[endtag].pc_type;

    var total_size = data[endtag].total_size;
    var text_size  = getTextSize(data,endtag);

    //draw the node itself (centering it properly)
    var visualLoc  = data[endtag].visual_loc;

    var description = getSimpleDescription(data,endtag);
    var numLines    = countNumOccurances("<br>",description);

    //recursively draw all the children (if any)
    var elapsedDist = 0;

    for(var i = 0; i<numChildren; i++) {
        var childEndtag    = endtag + "_" + i;

        if(data[childEndtag] == undefined)
            return;

        var childTotalSize = data[childEndtag].total_size;

        if(pc_type == "mg" || pc_type == "gamg") {
            //draw the appropriate line from the parent to the child
            ret += getCurve(x + visualLoc.x, y + visualLoc.y, x+text_size.width+data[endtag].indentations[i]+data[childEndtag].visual_loc.x, y+elapsedDist+data[childEndtag].visual_loc.y,"east");

            //draw the child
            ret += getBoxTree(data, childEndtag, x+text_size.width+data[endtag].indentations[i], y+elapsedDist); //remember to indent !!

            elapsedDist += childTotalSize.height;
        }
        else {
            //draw the appropriate line from the parent to the child
            ret += getCurve(x + visualLoc.x, y + visualLoc.y, x+elapsedDist+data[childEndtag].visual_loc.x, y+text_size.height+data[endtag].indentations[i]+data[childEndtag].visual_loc.y ,"south");

            //draw the child
            ret += getBoxTree(data, childEndtag, x+elapsedDist, y+text_size.height+data[endtag].indentations[i]); //remember to indent !!

            elapsedDist += childTotalSize.width;
        }
    }

    //useful for debugging purposes (don't delete)
    //ret += "<rect x=\"" + x + "\" y=\"" + y + "\" width=\"" + total_size.width + "\" height=\"" + total_size.height + "\" style=\"fill:black;fill-opacity:.1\" />";

    //draw the node itself last so that the text is on top of everything
    var color = colors[getNumUnderscores(endtag) % colors.length];
    ret += "<circle id=\"" + "node" + endtag + "\" cx=\"" + (x + visualLoc.x) + "\" cy=\"" + (y + visualLoc.y) + "\" r=\"" + node_radius + "\" stroke=\"black\" stroke-width=\"1\" fill=\"" + color + "\" />";

    for(var i = 0; i<numLines; i++) {
        var indx  = description.indexOf("<br>");
        var chunk = description.substring(0,indx);
        description = description.substring(indx+4,description.length);
        ret += "<text x=\"" + (x + visualLoc.x + 1.2*node_radius) + "\" y=\"" + (y + visualLoc.y + 2*node_radius + 12*i) + "\" fill=\"black\" font-size=\"12px\">" + chunk + "</text>";
    }

    return ret;
}

//this function should be pretty straightforward
function getTextSize(data, endtag) {

    var pc_type = data[endtag].pc_type;
    var ret     = new Object();
    ret.width   = 100;//70 is enough for chrome, but svg font in safari/firefox shows up bigger so we need more space (although the font-size is always 12px)

    var description = getSimpleDescription(data,endtag);
    var height = 2*node_radius + 12 * countNumOccurances("<br>",description); //make each line 15 pixels tall
    ret.height = height;

    return ret;
}

/*
 * This method recursively calculates each node's total-size (the total size the its subtree takes up)
 * Children of mg and gamg are put to the east of the parent node and children of anything else are put to the south
 * If the node has children, this method also calculates the data on how the child nodes should be indented (so that all sister nodes line up)
 * Also calculates and records the location (the center coordinates) of the visual node
 *
 */

function calculateSizes(data, endtag) {

    if(data[endtag] == undefined)
        return;
    var text_size   = getTextSize(data,endtag); //return an object containing 'width' and 'height'
    var numChildren = getNumChildren(data,endtag);
    var pc_type     = data[endtag].pc_type;

    if(numChildren == 0) {
	data[endtag].total_size = text_size; //simply set total_size to text_size
        data[endtag].visual_loc = {
            x: node_radius,
            y: node_radius
        }; //the location of the visual node
	return;
    }

    //otherwise, first recursively calculate the properties of the child nodes
    for(var i = 0; i<numChildren; i++) {
	var childEndtag = endtag + "_" + i;
	calculateSizes(data,childEndtag); //recursively calculate the data of all the children !!
    }

    if(pc_type == "mg" || pc_type == "gamg") { //put children to the east

	var totalHeight = 0; //get the total heights of all the children. and the most extreme visual node location
        var mostShifted = 0; //of the child nodes, find the most x_shifted visual node

	for(var i=0; i<numChildren; i++) { //iterate thru the children to get the total height and most extreme visual node location
	    var childEndtag  = endtag + "_" + i;

            if(data[childEndtag] == undefined)
                return;

            var childSize    = data[childEndtag].total_size;
            var visualLoc    = data[childEndtag].visual_loc;

	    totalHeight += childSize.height;
            if(visualLoc.x > mostShifted)
                mostShifted = visualLoc.x;
	}

        var indentations  = new Array();
        var rightFrontier = 0;

        for(var i=0; i<numChildren; i++) { //iterate through the children again and indent each child such that their visual nodes line up
            var childEndtag  = endtag + "_" + i;
            var childSize    = data[childEndtag].total_size;
            var visualLoc    = data[childEndtag].visual_loc;

            indentations[i] = 0;

            if(visualLoc.x < mostShifted) {
                indentations[i] = mostShifted - visualLoc.x; //record to let the drawing algorithm know to indent these children
            }
            if(indentations[i] + childSize.width > rightFrontier) //at the same time, calculate how wide the total_size must be
                rightFrontier = indentations[i] + childSize.width;
        }

        //find where the parent node must be (if there is an odd number of children, simply align it with the center child. for even, take average of the middle 2 children)
        var visualLoc = new Object();
        visualLoc.x = node_radius;

        if(numChildren % 2 == 0) { //even number of children (take avg of middle two visual nodes)
            var elapsedDist = 0;
            for(var i = 0; i<numChildren/2 - 1; i++) {
                var childEndtag  = endtag + "_" + i;
                elapsedDist += data[childEndtag].total_size.height;
            }
            var child1 = numChildren/2 - 1;
            var child2 = numChildren/2;
            var child1_endtag = endtag + "_" + child1;
            var child2_endtag = endtag + "_" + child2;

            var child1_pos    = elapsedDist + data[child1_endtag].visual_loc.y;
            var child2_pos    = elapsedDist + data[child1_endtag].total_size.height + data[child2_endtag].visual_loc.y;

            var mid_y = (child1_pos + child2_pos)/2;
            visualLoc.y = mid_y;
        }
        else { //odd number of children (simply take the visual y-coord of the middle child)
            var elapsedDist = 0;
            for(var i = 0; i<Math.floor(numChildren/2); i++) {
                var childEndtag  = endtag + "_" + i;
                elapsedDist += data[childEndtag].total_size.height;
            }
            var child = Math.floor(numChildren/2);
            var child_endtag = endtag + "_" + child;

            var mid_y = elapsedDist + data[child_endtag].visual_loc.y;
            visualLoc.y = mid_y;
        }

	var total_size = new Object();

        var southFrontier = visualLoc.y - node_radius + text_size.height;
	if(southFrontier > totalHeight) //should be rare, but certainly possible. (this is when the parent node is absurdly long)
	    total_size.height = southFrontier;
        else
            total_size.height = totalHeight;

	total_size.width = text_size.width + rightFrontier; //total width depends on how far the right frontier got pushed

	data[endtag].total_size   = total_size;
        data[endtag].indentations = indentations;
        data[endtag].visual_loc   = visualLoc;
    }
    else { //put children to the south

	var totalWidth = 0; //get the total widths of all the children. and the most extreme visual node location
        var mostShifted = 0; //of the child nodes, find the most y_shifted visual node

	for(var i=0; i<numChildren; i++) { //iterate thru the children to get the total width and most extreme visual node location
	    var childEndtag = endtag + "_" + i;
            if(data[childEndtag] == undefined)
                return;

	    var childSize   = data[childEndtag].total_size;
            var visualLoc   = data[childEndtag].visual_loc;

	    totalWidth += childSize.width;
	    if(visualLoc.y > mostShifted)
		mostShifted = visualLoc.y;
	}

        var indentations  = new Array();
        var southFrontier = 0;

        for(var i=0; i<numChildren; i++) { //iterate through the children again and indent each child such that their visual nodes line up
            var childEndtag  = endtag + "_" + i;
            var childSize    = data[childEndtag].total_size;
            var visualLoc    = data[childEndtag].visual_loc;

            indentations[i] = 0;

            if(visualLoc.y < mostShifted) {
                indentations[i] = mostShifted - visualLoc.y; //record to let the drawing algorithm know to indent these children
            }
            if(indentations[i] + childSize.height > southFrontier) //at the same time, calculate how long the total_size must be
                southFrontier = indentations[i] + childSize.height;
        }

        //find where the parent node must be (if there is an odd number of children, simply align it with the center child. for even, take average of the middle 2 children)
        var visualLoc = new Object();
        visualLoc.y   = node_radius;

        if(numChildren % 2 == 0) { //even number of children (take avg of middle two visual nodes)
            var elapsedDist = 0;
            for(var i = 0; i<numChildren/2 - 1; i++) {
                var childEndtag  = endtag + "_" + i;
                elapsedDist += data[childEndtag].total_size.width;
            }
            var child1 = numChildren/2 - 1;
            var child2 = numChildren/2;
            var child1_endtag = endtag + "_" + child1;
            var child2_endtag = endtag + "_" + child2;

            var child1_pos    = elapsedDist + data[child1_endtag].visual_loc.x;
            var child2_pos    = elapsedDist + data[child1_endtag].total_size.width + data[child2_endtag].visual_loc.x;

            var mid_x = (child1_pos + child2_pos)/2;
            visualLoc.x = mid_x;
        }
        else { //odd number of children (simply take the visual y-coord of the middle child)
            var elapsedDist = 0;
            for(var i = 0; i<Math.floor(numChildren/2); i++) {
                var childEndtag  = endtag + "_" + i;
                elapsedDist += data[childEndtag].total_size.width;
            }
            var child = Math.floor(numChildren/2);
            var child_endtag = endtag + "_" + child;

            var mid_x = elapsedDist + data[child_endtag].visual_loc.x;
            visualLoc.x = mid_x;
        }

	var total_size = new Object();

        var rightFrontier = visualLoc.x - node_radius + text_size.width;
	if(rightFrontier > totalWidth) //should be rare, but certainly possible. (this is when the parent node is absurdly wide)
	    total_size.width = rightFrontier;
        else
            total_size.width = totalWidth;

	total_size.height = text_size.height + southFrontier; //total height depends on how far the south frontier got pushed

	data[endtag].total_size   = total_size;
        data[endtag].indentations = indentations;
        data[endtag].visual_loc   = visualLoc;
    }
    return;
}


//use svg to generate a smooth Bézier curve from one point to another
//this simple algorithm to find the 2 control points for the bezier curve to generate a logistic-looking curve is taken from the d3 graphics library
function getCurve(x1,y1,x2,y2,direction) {

    var ret = "";

    if(direction == "east") {
        var mid_x = (x1+x2)/2.0;

        var control1 = new Object();
        control1.x = mid_x;
        control1.y = y1;

        var control2 = new Object();
        control2.x = mid_x;
        control2.y = y2;

        ret = "<path d=\"M " + x1 + "," + y1 + " " + "C" + control1.x + "," + control1.y + " " + control2.x + "," + control2.y + " " + x2 + "," + y2 + "\" stroke =\"blue\" stroke-width=\"2\" stroke-opacity=\".5\" fill=\"none\" />";
    }

    else if(direction == "south") {
        var mid_y = (y1+y2)/2.0;

        var control1 = new Object();
        control1.x = x1;
        control1.y = mid_y;

        var control2 = new Object();
        control2.x = x2;
        control2.y = mid_y;

        ret = "<path d=\"M " + x1 + "," + y1 + " " + "C" + control1.x + "," + control1.y + " " + control2.x + "," + control2.y + " " + x2 + "," + y2 + "\" stroke =\"blue\" stroke-width=\"2\" stroke-opacity=\".5\" fill=\"none\" />";
    }

    return ret;
}
