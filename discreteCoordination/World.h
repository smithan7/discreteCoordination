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

#include "Costmap.h"

using namespace cv;
using namespace std;

class World {
public:
	World(string fName, int gSpace, float obsThresh, float comThresh);
	virtual ~World();

	// initialize world
	void saveWorldToYML(string fName); // save a world map to YML
	void pullWorldFromYML(string fName); // pull world from yml
	void getObsGraph(); // find nodes that can see eachother
	void getCommGraph(); // find nodes that can communicate with eachother
	void initializeMaps(); // initialize cost map, point map

	Mat createMiniMapImg(); // for making perfect miniMapImg

	// for working in the world
	Costmap costmap; // what the agent uses to navigate
	vector<vector<vector<vector<int> > > > obsGraph; // [xLoc][yLoc][listIndex][0=their xLoc, 1 = their yLoc] list contains all that are observable
	void observe(vector<int> cLoc, Costmap &costmap);

	int gSpace; // spacing, in pixels, between nodes

	float obsThresh; // how far can I see? LOS
	float commThresh; // how far can I communicate, LOS


};

#endif /* World_H_ */
