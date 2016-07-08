/*
 * World.h
 *
 *  Created on: Mar 28, 2016
 *      Author: andy
 */

#ifndef World_H_
#define World_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace cv;
using namespace std;

class World {
public:
	World(int gSpace, float obsThresh, float comThresh);
	virtual ~World();

	// build a walls object, so nodes that are built that is constructed as detectected, for inference
	void saveWorldToYML(); // save a world map to YML
	void pullWorldFromYML(string fName); // pull world from yml
	void getObsGraph(); // find nodes that can see eachother
	void getCommGraph(); // find nodes that can communicate with eachother
	void getDistGraph(); // calc distance between each node
	void initializeMaps(); // initialize cost map, point map

	Mat createMiniMapImg(); // for making perfect miniMapImg

	void plotPath(vector<vector<int> >, int[3], int);
	void plotMap();
	void clearPlot();
	void plotTravelGraph();
	void plotExploreGraph();
	void plotFrontierGraph(); //
	void addCommLine(vector<int> b,vector<int> c); //
	void plotCommLines(); // plot inter agent commo
	vector<vector<int> > commLine;

	Mat createExplImage();

	vector<vector<int> > costMap; // what the agent uses to navigate
	vector<vector<Point> > pointMap; // image locations of all points
	vector<vector<vector<vector<int> > > > obsGraph; // [xLoc][yLoc][listIndex][0=their xLoc, 1 = their yLoc] list contains all that are observable
	vector<vector<float> > distGraph; // [dx][dy] = distance
	float getEuclidDist(int x0, int y0, int x1, int y1);


	int nCols;
	int nRows;
	int gSpace; // spacing, in pixels, between nodes
	Mat image;
	Mat imgGray;
	Mat imgFrnt;
	String fileName; // filename image is taken from

	float obsThresh; // how far can I see? LOS
	float commThresh; // how far can I communicate, LOS


};

#endif /* World_H_ */
