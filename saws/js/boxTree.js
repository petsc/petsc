/*
 * box-algorithm that I wrote for drawing trees with different node sizes and splitting in different directions
 *
 * The following aesthetic criteria are satisfied:
 *
 * 1) No overlapping text
 * 2) When switching directions, the entire subtree in the new direction should be seen as one of the nodes in the original direction
 * 3) Sister nodes are shown on the same line (not yet implemented)
 * 4) No overlapping subtrees
 * 5) Parent is centered at the middle of its children (not yet implemented)
 *
 */

//generates svg code for the given input parameters
//the x, y coordinates are the upper left hand coordinate of the drawing. should start with (0,0)
function getBoxTree(data, endtag, x, y) {

    var ret         = ""; //the svg code to return
    var index       = getIndex(data,endtag);
    var numChildren = getNumChildren(data,endtag);
    var pc_type     = data[index].pc_type;

    var total_size = data[index].total_size;
    var text_size  = getTextSize(data,endtag);

    //draw the node itself (centering it properly)
    //this centering algorithm can be easily changed
    var centered_x = x;
    var centered_y = y;
    if(pc_type == "mg") {
        centered_y = y + .5*total_size.height - .5*text_size.height;
    }
    else {
        centered_x = x + .5*total_size.width - .5*text_size.width;
    }

    var node_radius = 5;
    var description = getSimpleDescription(endtag);
    var numLines    = countNumOccurances("<br>",description);

    //ret += "<rect x=\"" + centered_x + "\" y=\"" + centered_y + "\" width=\"" + text_size.width + "\" height=\"" + text_size.height + "\" style=\"fill:rgb(0,0,255);stroke-width:2;stroke:rgb(0,0,0)\" />"; //don't delete this code. this is very useful for debugging purposes
    ret += "<circle cx=\"" + (centered_x + node_radius) + "\" cy=\"" + (centered_y + node_radius) + "\" r=\"" + node_radius + "\" stroke=\"black\" stroke-width=\"1\" fill=\"blue\" />";

    for(var i = 0; i<numLines; i++) {
        var index = description.indexOf("<br>");
        var chunk = description.substring(0,index);
        description = description.substring(index+4,description.length);
        ret += "<text x=\"" + (centered_x+6) + "\" y=\"" + (centered_y+19+12*i) + "\" fill=\"black\" font-size=\"12px\">" + chunk + "</text>"; //for debugging purposes, I'm putting the endtag here. this will eventually be replaced by the proper solver description
    }



    var elapsedDist = 0;

    //recursively draw all the children (if any)
    for(var i = 0; i<numChildren; i++) {
        var childEndtag    = endtag + "_" + i;
        var childIndex     = getIndex(data,childEndtag);
        var childTotalSize = data[childIndex].total_size;

        if(pc_type == "mg") {
            ret += getBoxTree(data, childEndtag, x+text_size.width, y+elapsedDist);
            //calculate where child is located and draw the appropriate line
            var child_pc_type  = data[childIndex].pc_type;
            var childTextSize  = getTextSize(data,childEndtag);

            var child_centered_x = x+text_size.width;
            var child_centered_y = y+elapsedDist;

            if(child_pc_type == "mg")
                child_centered_y = (y+elapsedDist) + .5*childTotalSize.height - .5*childTextSize.height;
            else
                child_centered_x = (x+text_size.width) + .5*childTotalSize.width - .5*childTextSize.width;

            ret += getCurve(centered_x + node_radius, centered_y + node_radius, child_centered_x + node_radius, child_centered_y + node_radius,"east");


            elapsedDist += childTotalSize.height;
        }
        else {
            ret += getBoxTree(data, childEndtag, x+elapsedDist, y+text_size.height);
            //calculate where child is located and draw the appropriate line
            var child_pc_type  = data[childIndex].pc_type;
            var childTextSize  = getTextSize(data,childEndtag);

            var child_centered_x = x+elapsedDist;
            var child_centered_y = y+text_size.height;

            if(child_pc_type == "mg")
                child_centered_y = (y+text_size.height) + .5*childTotalSize.height - .5*childTextSize.height;
            else
                child_centered_x = (x+elapsedDist) + .5*childTotalSize.width - .5*childTextSize.width;

            ret += getCurve(centered_x + node_radius, centered_y + node_radius, child_centered_x + node_radius, child_centered_y + node_radius,"south");


            elapsedDist += childTotalSize.width;
        }
    }

    return ret;
}

//this function should be pretty straightforward
function getTextSize(data, endtag) {

    var index   = getIndex(data,endtag);
    var pc_type = data[index].pc_type;
    var ret     = new Object();
    ret.width   = 150;

    var description = getSimpleDescription(endtag);
    var height = 20 + 15 * countNumOccurances("<br>",description); //make each line 15 pixels tall
    ret.height = height;

    return ret;
}

/*
 * This method recursively calculates each node's total-size (the total size the its subtree takes up)
 * This method puts children of fieldsplit to the south of the parent node and the children of mg to the east of the parent node
 *
 */

function calculateSizes(data, endtag) {

    var index       = getIndex(data,endtag);
    var text_size   = getTextSize(data,endtag); //return an object containing 'width' and 'height'
    var numChildren = getNumChildren(data,endtag);
    var pc_type     = data[index].pc_type;

    if(numChildren == 0) {
	data[index].total_size = text_size; //simply set total_size to text_size
	return;
    }

    //otherwise, first recursively calculate the properties of the child nodes
    for(var i = 0; i<numChildren; i++) {
	var childEndtag = endtag + "_" + i;
	calculateSizes(data,childEndtag); //recursively calculate the sizes of all the children !!
    }

    if(pc_type == "mg") { //put children to the east

	var totalHeight = 0; //get the total heights of all the children. and the most extreme width
	var maxWidth    = 0;

	for(var i=0; i<numChildren; i++) {
	    var childEndtag = endtag + "_" + i;
	    var childIndex  = getIndex(data,childEndtag);
	    var childSize   = data[childIndex].total_size;

	    totalHeight += childSize.height;
	    if(childSize.width > maxWidth)
		maxWidth = childSize.width;
	}

	var total_size = new Object();

	if(text_size.height > totalHeight) //should be rare, but certainly possible.
	    total_size.height = text_size.height;
        else
            total_size.height = totalHeight;
	total_size.width = text_size.width + maxWidth;

	data[index].total_size = total_size;
    }
    else { //put children to the south

	var totalWidth = 0; //get the total widths of all the children. and the most extreme height
	var maxHeight  = 0;

	for(var i=0; i<numChildren; i++) {
	    var childEndtag = endtag + "_" + i;
	    var childIndex  = getIndex(data,childEndtag);
	    var childSize   = data[childIndex].total_size;

	    totalWidth += childSize.width;
	    if(childSize.height > maxHeight)
		maxHeight = childSize.height;
	}

	var total_size = new Object();

	if(text_size.width > totalWidth) //should be rare, but certainly possible.
	    total_size.width = text_size.width;
        else
            total_size.width = totalWidth;
	total_size.height = text_size.height + maxHeight;

	data[index].total_size = total_size;
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

        ret = "<path d=\"M " + x1 + "," + y1 + " " + "C" + control1.x + "," + control1.y + " " + control2.x + "," + control2.y + " " + x2 + "," + y2 + "\" stroke =\"blue\" stroke-width=\"2\" fill=\"none\" />";
    }

    else if(direction == "south") {
        var mid_y = (y1+y2)/2.0;

        var control1 = new Object();
        control1.x = x1;
        control1.y = mid_y;

        var control2 = new Object();
        control2.x = x2;
        control2.y = mid_y;

        ret = "<path d=\"M " + x1 + "," + y1 + " " + "C" + control1.x + "," + control1.y + " " + control2.x + "," + control2.y + " " + x2 + "," + y2 + "\" stroke =\"blue\" stroke-width=\"2\" fill=\"none\" />";
    }

    /*$("#tree").html("<svg id=\"demo\" width=\"" + matInfo[0].total_size.width + "\" height=\"" + matInfo[0].total_size.height + "\" viewBox=\"0 0 " + matInfo[0].total_size.width + " " + matInfo[0].total_size.height + "\">" + ret + "</svg>");*/ //this was for debugging purposes

    return ret;
}