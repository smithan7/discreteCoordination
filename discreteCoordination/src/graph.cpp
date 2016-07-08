/*
 * graph.cpp
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */


#include "graph.h"

Point findIntersection(Vec4i w1, Vec4i w2);
float distToLine(Vec4i w, Point a);
Point extendLine(Point a, Point m);

using namespace cv;
using namespace std;

graph::graph(){
	// open descriptor names yml file and store to vector
	FileStorage fsN("/home/andy/Dropbox/workspace/fabmap2Test/masterList.yml", FileStorage::READ);
	fsN["names"] >> this->masterNameList;
	fsN["histogramList"] >> this->masterHistogramList;
	fsN["sequenceList"] >> this->masterSequenceList;
	fsN["pointList"] >> this->masterPointList;
	fsN["centerList"] >> this->masterCenterList;
	fsN["meanLength"] >> this->masterMeanLengthList;
	fsN.release();
}

void graph::createGraph(world &gMap, float obsThresh, float comThresh, int gSpace){
	this->gSpace = gSpace;
	this->nRows = gMap.nRows;
	this->nCols = gMap.nCols;
	this->initializeCostMap();

	this->frntsExist = true;
	this->image = gMap.image;
	this->tempImage = gMap.image;
	this->obsThresh = obsThresh;
	this->comThresh = comThresh;
}

float graph::dStarDist(frontier &frnt, agent bot, world &gMap){

	cerr << "  fLoc: " << frnt.centroid[0] << ", " << frnt.centroid[1] << endl;
	cerr << "  cLoc: " << bot.cLoc[0] << ", " << bot.cLoc[1] << endl;

	// is bot in the cSet?
	if(frnt.cSet[bot.cLoc[0]][bot.cLoc[1]] == 1){
		cerr << "   in cSet" << endl;
		return frnt.gScore[bot.cLoc[0]][bot.cLoc[1]];
	}
	// do I need to seed the openSet?
	if(frnt.oSet[frnt.centroid[0]][frnt.centroid[1]] == 0){
		cerr << "   seeding oSet" << endl;
		frnt.oSet[frnt.centroid[0]][frnt.centroid[1]] = 1;
		frnt.gScore[frnt.centroid[0]][frnt.centroid[1]] = 0;
		frnt.fScore[frnt.centroid[0]][frnt.centroid[1]] = frnt.gScore[frnt.centroid[0]][frnt.centroid[1]] + gMap.getEuclidDist(frnt.centroid[0],frnt.centroid[1],bot.cLoc[0],bot.cLoc[1]);
	}
	int foo = 1;
	int finishFlag = 0;
	cerr << "going into while loop" << endl;
	while(finishFlag == 0 ){

		// am I at the goal?
		if(frnt.cSet[bot.cLoc[0]][bot.cLoc[1]] == 1){
			cerr << "   cLoc in cSet, finished" << endl;
			return frnt.gScore[bot.cLoc[0]][bot.cLoc[1]];
		} // end at the goal

		/////////////////// this finds oSet node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> tLoc;
		tLoc.push_back(-1);
		tLoc.push_back(-1);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(frnt.oSet[i][j] > 0 && frnt.fScore[i][j] < min){
					min = frnt.fScore[i][j];
					tLoc[0] = i;
					tLoc[1] = j;
				}
			}
		}

		if(tLoc[0] == tLoc[1] && tLoc[0] == -1){
			return(INFINITY);
		}
		/////////////////////// end finding current node

		frnt.oSet[tLoc[0]][tLoc[1]] = 0; // take out of openset
		frnt.cSet[tLoc[0]][tLoc[1]] = 1; // add to closed set

		for(int nbrRow=tLoc[0]-1;nbrRow<tLoc[0]+2;nbrRow++){
			if(nbrRow > 0 && nbrRow < gMap.nRows){ // make sure it's on the map

				for(int nbrCol=tLoc[1]-1; nbrCol<tLoc[1]+2; nbrCol++){
					if(nbrCol > 0 && nbrCol < gMap.nCols){ // make sure its on the map

						if(this->costMap[nbrRow][nbrCol] < 51){ // make sure its observed / or frontier before adding to oSet

							if(frnt.cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
								continue;
							}
							float tGScore;
							tGScore = frnt.gScore[tLoc[0]][tLoc[1]] + gMap.getEuclidDist(tLoc[0],tLoc[1],nbrRow,nbrCol); // calc temporary gscore
							if(frnt.oSet[nbrRow][nbrCol] == 0){
								frnt.oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
							}
							else if(tGScore >= frnt.gScore[nbrRow][nbrCol]){ // is temp gscore worse than stored g score of nbr? don't update
								continue;
							}
							frnt.cameFrom[nbrRow][nbrCol] = tLoc;
							frnt.gScore[nbrRow][nbrCol] = tGScore;
							frnt.fScore[nbrRow][nbrCol] = tGScore + gMap.getEuclidDist(bot.cLoc[0],bot.cLoc[1],nbrRow,nbrCol) + this->costMap[nbrRow][nbrCol];
						}
					}
				}
			}
		}
	}
	return INFINITY;
}

vector<vector<int> > graph::dStarPath(frontier &frnt, agent &bot, world &gMap){
	// is bot in the cSet?
	if(frnt.cSet[bot.cLoc[0]][bot.cLoc[1]] == 100){
		cerr << "in cSet" << endl;
		vector<vector<int> > totalPath;
		totalPath.push_back(bot.cLoc);
		vector<int> temp = bot.cLoc;
		cerr << "a" << endl;
		cerr << "frnt.Centroid: " << frnt.centroid[0] << " , " << frnt.centroid[1] << endl;
		cerr << "cLoc: " << bot.cLoc[0] << " , " << bot.cLoc[1] << endl;
		bool finish = false;
		while(!finish){ // work backwards to start
			if(temp == frnt.centroid){
				finish =true;
				totalPath.push_back(temp); // append path
			}
			cerr << "   " << temp[0] << " , " << temp[1] << endl;
			temp = frnt.cameFrom[temp[0]][temp[1]]; // work backwards
			totalPath.push_back(temp); // append path
		}
		cerr << " returning totalPath" << endl;
		return totalPath;
	}

	// do I need to seed the openSet?
	if(frnt.oSet[frnt.centroid[0]][frnt.centroid[1]] == 0){
		frnt.oSet[frnt.centroid[0]][frnt.centroid[1]] = 1;
		frnt.gScore[frnt.centroid[0]][frnt.centroid[1]] = 0;
		frnt.fScore[frnt.centroid[0]][frnt.centroid[1]] = frnt.gScore[frnt.centroid[0]][frnt.centroid[1]] + gMap.getEuclidDist(frnt.centroid[0],frnt.centroid[1],bot.cLoc[0],bot.cLoc[1]);
	}

	int foo = 1;
	int finishFlag = 0;

	while(foo>0 && finishFlag == 0){

		// am I at the goal?
		if(frnt.cSet[bot.cLoc[0]][bot.cLoc[1]] == 1){
			vector<vector<int> > totalPath;
			totalPath.push_back(bot.cLoc);
			vector<int> temp = bot.cLoc;
			cerr << "frnt.Centroid: " << frnt.centroid[0] << " , " << frnt.centroid[1] << endl;
			cerr << "cLoc: " << bot.cLoc[0] << " , " << bot.cLoc[1] << endl;
			bool finish = false;
			while(temp[0] >= 0){ // work backwards to start
				if(temp == frnt.centroid){
					finish =true;
					totalPath.push_back(temp); // append path
				}
				cerr << "   " << temp[0] << " , " << temp[1] << endl;
				temp = frnt.cameFrom[temp[0]][temp[1]]; // work backwards
				totalPath.push_back(temp); // append path
			}
			cerr << " returning totalPath" << endl;
			return totalPath;
		} // end am I at the goal


		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> tLoc;
		tLoc.push_back(0);
		tLoc.push_back(0);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(frnt.oSet[i][j] > 0 && frnt.fScore[i][j] < min){
					min = frnt.fScore[i][j];
					tLoc[0] = i;
					tLoc[1] = j;
				}
			}
		}
		//cerr << "tLoc: " << tLoc[0] << "<" << tLoc[1] << endl;
		/////////////////////// end finding current node

		frnt.oSet[tLoc[0]][tLoc[1]] = 0; // take out of openset
		frnt.cSet[tLoc[0]][tLoc[1]] = 1; // add to closed set

		for(int nbrRow=tLoc[0]-1;nbrRow<tLoc[0]+2;nbrRow++){
			if(nbrRow > 0 && nbrRow < gMap.nRows){ // make sure it's on the map

				for(int nbrCol=tLoc[1]-1; nbrCol<tLoc[1]+2; nbrCol++){
					if(nbrCol > 0 && nbrCol < gMap.nCols){ // make sure its on the map

						if(this->costMap[nbrRow][nbrCol] < 51){ // make sure its observed / or frontier before adding to oSet

							if(frnt.cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
								continue;
							}
							float tGScore;
							tGScore = frnt.gScore[tLoc[0]][tLoc[1]] + gMap.getEuclidDist(tLoc[0],tLoc[1],nbrRow,nbrCol); // calc temporary gscore
							if(frnt.oSet[nbrRow][nbrCol] == 0){
								frnt.oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
							}
							else if(tGScore >= frnt.gScore[nbrRow][nbrCol]){ // is temp gscore worse than stored g score of nbr? don't update
								continue;
							}
							frnt.cameFrom[nbrRow][nbrCol] = tLoc;
							frnt.gScore[nbrRow][nbrCol] = tGScore;
							frnt.fScore[nbrRow][nbrCol] = tGScore + gMap.getEuclidDist(bot.cLoc[0],bot.cLoc[1],nbrRow,nbrCol) + this->costMap[nbrRow][nbrCol];
						}
					}
				}
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
	}
	vector<vector<int> > totalPath;
	for(int i=0; i<4; i++){
		vector<int> t = bot.cLoc;
		totalPath.push_back(t);
	}
	return totalPath;
}

void graph::centralMarket(world &gMap, vector<frontier> &frnt, vector<agent> &bot){
	int nBot = bot.size();
	int nFrnt = frnt.size();

	for(int i=0; i<nBot; i++){
		bot[i].fCost.erase(bot[i].fCost.begin(),bot[i].fCost.end());
		for(int j=0; j<nFrnt; j++){
			cerr << "   going into D* dist" << endl;
			bot[i].fCost.push_back(this->dStarDist(frnt[j], bot[i], gMap));
			cerr << "   out of D* dist" << endl;
		}
	}

	cerr << "frontier costs: " << endl;
	for(int i=0; i<nFrnt; i++){
		for(int j=0; j<nBot; j++){
			cerr << "   cost: " << bot[j].fCost[i] << endl;;
			cerr << "   reward: " << frnt[i].reward << endl;
		}
	}

	if(nBot <= nFrnt){ // more frontiers than agents
		cerr << "more frontiers than agents" << endl;
		vector<vector<float> > fValueList; // list of frontier values for all agents, [agent][frontier]
		for(int i=0; i<nBot; i++){
			vector<float> cVal;
			for(int j=0; j<nFrnt; j++){
				cVal.push_back( frnt[i].reward - bot[i].fCost[j] );
			}
			fValueList.push_back( cVal );
		}

		cerr << "fValueList: " << endl;
		for(int i=0; i<nBot; i++){
			cerr << "   ";
			for(int j=0; j<nFrnt; j++){
				cerr << fValueList[i][j] << " , ";
			}
			cerr << endl;
		}

		cerr << "a" << endl;


		bool fin = false;
		vector<int> maxDex;
		vector<float> maxVal;

		while(!fin){
			maxDex.erase(maxDex.begin(), maxDex.end());
			maxVal.erase(maxVal.begin(), maxVal.end());
			fin = true;

			for(int i=0; i<nBot; i++){ // get each agents best frontier
				maxDex.push_back( -1 );
				maxVal.push_back( -INFINITY );

				for(int j=0; j<(int)fValueList[i].size(); j++){
					if(fValueList[i][j] > maxVal[i]){
						maxDex[i] = j; // agent's max value is frontier j
						maxVal[i] = fValueList[i][j];
					}
				}
			}

			cerr << "b" << endl;


			// make sure no one shares the same frontier
			for(int i=0; i<nBot; i++){
				for(int j=i+1; j<nBot; j++){
					if(i!=j && maxDex[i] == maxDex[j]){ // not me and has the same goal;
						fin = false;
						if(maxVal[i] >= maxVal[j]){
							fValueList[j][maxDex[j]] = -INFINITY;
						}
						else{
							fValueList[i][maxDex[i]] = -INFINITY;
						}
					}
				}
			}
		}
		for(int i=0; i<nBot; i++){
			bot[i].gLoc = frnt[maxDex[i]].centroid;
			bot[i].gIndex = maxDex[i];
		}
	}
	else{ // more agents than frontiers
		cerr << "more agents than frontiers" << endl;
		for(int i=0; i<nFrnt; i++){ // go through all frontiers and find the best agent
			float mV = INFINITY;
			int mI;
			for(int j=0; j<nBot; j++){
				if(bot[j].fCost[i] < mV){
					mV = bot[j].fCost[i];
					mI = j;
				}
			}
			bot[mI].gLoc = frnt[i].centroid;
			bot[mI].gIndex = i;
			for(int j = 0; j<(int)frnt.size(); j++){ // erase all of the value for the worst agent
				bot[mI].fCost[j] = INFINITY;
			}
		}
	}
}

void graph::initializeCostMap(){
	for(int i=0; i<this->nRows; i++){
		vector<float> tC;
		for(int j=0; j<this->nCols; j++){
			tC.push_back(100);
		}
		this->costMap.push_back(tC);
	}
}

Mat graph::createMiniMapImg(){
	Mat temp = Mat::zeros(this->nRows, this->nCols,CV_8UC1);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] > 100){
				temp.at<uchar>(i,j,0) = 101;
			}
			else{
				temp.at<uchar>(i,j,0) = this->costMap[i][j];
			}
		}
	}
	return temp;
}

Mat graph::createMiniMapInferImg(){
	Mat temp = Mat::zeros(this->nRows, this->nCols,CV_8UC1);
	for(size_t i=0; i<this->hullPts.size(); i++){
		temp.at<uchar>(this->hullPts[i][1]/this->gSpace,this->hullPts[i][0]/this->gSpace,0) = 255;
	}

	Scalar color;
	color[0] = 255;

    for(size_t i=0; i<this->hullPts.size()-1; i++){
    	Point pa, pb;
    	pa.x = this->hullPts[i][0]/this->gSpace;
    	pa.y = this->hullPts[i][1]/this->gSpace;
    	pb.x = this->hullPts[i+1][0]/this->gSpace;
    	pb.y = this->hullPts[i+1][1]/this->gSpace;
    	line(temp, pa, pb, color, 2, 8);
    }

    for(int i=0; i<1; i++){
    	Point pa, pb;
    	pa.x = this->hullPts[0][0]/this->gSpace;
    	pa.y = this->hullPts[0][1]/this->gSpace;

    	pb.x = this->hullPts[this->hullPts.size()-1][0]/this->gSpace;
    	pb.y = this->hullPts[this->hullPts.size()-1][1]/this->gSpace;
    	line(temp, pa, pb, color, 2, 8);
    }

	return temp;
}

void graph::makeInference(vector<frontier> &frnt){
	Mat inferCalc = Mat::zeros(this->gSpace*(this->nRows+1), this->gSpace*(this->nCols+1), CV_8UC1);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			Point pa, pb;
			pa.x = this->gSpace*j + this->gSpace/2;
			pa.y = this->gSpace*i + this->gSpace/2;
			pb.x = this->gSpace*j - this->gSpace/2;
			pb.y = this->gSpace*i - this->gSpace/2;
			if(this->costMap[i][j] < 51){ // free space and frontiers
				rectangle(inferCalc,pa,pb,Scalar(255),-1);
			}
			else if(this->costMap[i][j] == INFINITY){ // obstacle
				rectangle(inferCalc,pa,pb,Scalar(255),-1);
			}
			/*
			else if(this->costMap[i][j] == 100){ // unknown space
				rectangle(inferCalc,pa,pb,Scalar(255),-1);
			}
			*/
		}
	}

	// get points that make up the obstacle
	vector<Point> obsPoints;
    obsPoints = this->getImagePoints(inferCalc);

    vector<Point> hull(obsPoints.size() );
   	convexHull( Mat(obsPoints), hull, true, CV_CHAIN_APPROX_NONE);

    /// Draw contours + hull results
    RNG rng(12345);
    Mat hullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
    Scalar red = Scalar(255);

    for(int i=0; i<(int)hull.size()-1; i++){
    	line(hullDrawing, hull[i], hull[i+1], red, 1,8);
    }
    line(hullDrawing, hull[0], hull[hull.size()-1], red, 1,8);


    Mat plot = Mat::zeros(inferCalc.size(), CV_8UC1);
    /*
    resize(hullDrawing, plot, Size(), 1,1);
    imshow( "HullDrawing", plot );
    cerr << "out of draw contours" << endl;
	waitKey(0);
	*/
	// btwise and between hull and main image to get outer boundaries, extend to find intersection points
	Mat hullWalls = Mat::zeros( hullDrawing.size(), CV_8UC3 );
	bitwise_and(hullDrawing, inferCalc, hullWalls);
	/*
	resize(hullWalls, plot, Size(), 1,1);
    imshow( "Hull walls", plot );
	waitKey(0);
	*/
	Mat wallLines = Mat::zeros(inferCalc.size(), CV_8UC1);
	vector<Vec4i> lines;
	HoughLinesP(hullWalls, lines, 1, CV_PI/180, 1  ,3,2);

    for( size_t i = 0; i < lines.size(); i++){
        line( wallLines, Point(lines[i][0], lines[i][1]),Point(lines[i][2], lines[i][3]), Scalar(255), 3, 8 );
    }
    /*
    resize(wallLines, plot, Size(), 1,1);
    imshow( "wall lines", plot );
	waitKey(0);
	*/

	// find intersection points of hull walls
	vector<Point> intPts;
	for( size_t i = 0; i < lines.size(); i++){
		for(size_t j=i; j<lines.size(); j++){
			Point t = findIntersection(lines[i],lines[j]);
			if(t.x > 0 && t.x < wallLines.rows && t.y > 0 && t.y < wallLines.cols){
				intPts.push_back(t);
			}
		}
    }

	Mat intImage = Mat::zeros(inferCalc.size(), CV_8UC1);
	for(int i=0; i<(int)intPts.size(); i++){
		circle(intImage, intPts[i], 1, Scalar(255), -1);
	}
	// create new hull with intersection points
    bitwise_or(intImage,hullDrawing, intImage);
    /*
    resize(intImage, plot, Size(), 1,1);
    imshow( "Intersection Points and hull", plot);
	waitKey(0);
	*/

	// add intersection points
	vector<Point> hullPoints;
    hullPoints = this->getImagePoints(intImage);

    // get new convex hull with projected boundaries
    vector<Point> outerHull(hullPoints.size() );
   	convexHull( Mat(hullPoints), outerHull, true, CV_CHAIN_APPROX_NONE);

    /// Draw contours + hull results
    Mat outerHullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
    for(int i=0; i<(int)outerHull.size()-1; i++){
    	line(outerHullDrawing, outerHull[i], outerHull[i+1], red, 1,8);
    }
    line(outerHullDrawing, outerHull[0], outerHull[outerHull.size()-1], red, 3,8);
    /*
    resize(outerHullDrawing, plot, Size(), 1,1);
    imshow( "outerHullDrawing", plot );
	waitKey(0);
	*/
	// project frontiers
	this->getFrontierOrientation(frnt);
	for(int i=0; i<(int)frnt.size(); i++){
		frnt[i].getFrontierProjection();
    }
	// find frontiers likely to exit the outer hull
	vector<int> frntExits;
	cerr << frnt.size() << endl;
	for(int i=0; i<(int)frnt.size(); i++){
		double t = pointPolygonTest(outerHull,Point(frnt[i].centroid[1]*this->gSpace + this->gSpace/2,frnt[i].centroid[0]*this->gSpace + this->gSpace/2), false); // +1 means inside
		if(t <= 0){
			frntExits.push_back(i);
		}
	}

	// find unexplored areas inside the outer hull
	Mat frontierSpace = Mat::zeros(inferCalc.size(), CV_8UC1);
	bitwise_or(inferCalc, outerHullDrawing, frontierSpace);

	/*
	imshow("frontierSpace", frontierSpace);
	waitKey(0);
	*/
	vector<vector<Point> > goalAreaContour;
	vector<Vec4i> hierarchyF;
	findContours(frontierSpace,goalAreaContour, hierarchyF, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Mat cont = Mat::zeros( inferCalc.size(), CV_8UC3);
	for(int i=0; i<(int)goalAreaContour.size(); i++){
		drawContours(cont, goalAreaContour, i, Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),3,8);
	}
	/*
	imshow("cont", cont);
	waitKey(0);
	*/
    // identify contours with frontiers in them
    vector<float> goalAreaArea;
    vector<int> goalAreaCount;
    vector<vector<int> > goalAreaMembers;
    for(int i=0; i<(int)goalAreaContour.size(); i++){
    	//Mat cont2 = Mat::zeros( inferCalc.size(), CV_8UC3);
    	int count = 0;
    	vector<int> m;
    	for(int j=0; j<(int)frnt.size(); j++){
    		//circle(cont2,Point(frntProjection[j][1]*this->gSpace + this->gSpace/2*this->nRows,frntProjection[j][0]*this->gSpace + this->gSpace/2*this->nCols),10,Scalar(0,0,255),-1);
    		double t0 = pointPolygonTest(goalAreaContour[i],Point(frnt[j].centroid[1]*this->gSpace + this->gSpace/2,frnt[j].centroid[0]*this->gSpace + this->gSpace/2), false); // +1 means inside
    		if(t0 >= 0){
    			m.push_back(j);
    			count++;
    		}
    	}
		goalAreaMembers.push_back(m);
		goalAreaCount.push_back(count);
		goalAreaArea.push_back(contourArea(goalAreaContour[i]));

		/*
    	cerr << " = " << count << endl;
		drawContours(cont, goalAreaContour, i, Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),3,8);
		imshow("cont", cont);
		waitKey(0);
	*/
    }
    // determine if children of contour has frontiers in it
    for(size_t i=0; i<goalAreaMembers.size(); i++){
    	int child = hierarchyF[i][2];
		while(child >= 0){ // check every contour in child level
			// compare child list against mine, erase from my list frontiers that my child has
			for(size_t p=0; p<goalAreaMembers[i].size(); p++){
				for(size_t c=0; c<goalAreaMembers[child].size(); c++){
					//for(int j=0; j<goalAreaMembers[child].size(); j++){
					//	cout << goalAreaMembers[child][j] << ", ";
					//}
					//cout << endl;


					if(goalAreaMembers[i][p] == goalAreaMembers[child][c]){ // matching pair?
						goalAreaMembers[i].erase(goalAreaMembers[i].begin()+p,goalAreaMembers[i].begin()+p+1);
					}

				}

			}
			child = hierarchyF[child][0];
		}
    }


    /*
    for(int i=0; i<goalAreaContour.size(); i++){
		Mat cont2 = Mat::zeros( inferCalc.size(), CV_8UC3);
		for(int j=0; j<frntProjection.size(); j++){
			circle(cont2,Point(frntProjection[j][1]*this->gSpace + this->gSpace/2*this->nRows,frntProjection[j][0]*this->gSpace + this->gSpace/2*this->nCols),this->gSpace,Scalar(0,0,255),-1);
		}
		drawContours(cont2, goalAreaContour, i, Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),3,8);
		imshow("goalAreaContours", cont2);
		waitKey(0);

    }
    cout << "Goal Area Members" << endl;
    for(int i=0; i<goalAreaMembers.size(); i++){
    	cout << "   ";
    	for(int j=0; j<goalAreaMembers[i].size(); j++){
    		cout << goalAreaMembers[i][j] << ", ";
    	}
    	cout << endl;
    }
	*/

	// assign values (area) to frontiers

    // set value for all frontiers in each internal contour
    vector<float> goalAreaValue;

    float maxValue = 0;
    for(int i=0; i<(int)goalAreaArea.size(); i++){
    	//cerr << "goalAreaArea[" << i << "]: " << goalAreaArea[i] << endl;
    	//cerr << "goalAreaCount[" << i << "]: " << goalAreaCount[i] << endl;
    	goalAreaValue.push_back(goalAreaArea[i] / goalAreaCount[i]);
    	//cout << "gAV: " << goalAreaArea[i] / goalAreaCount[i] << endl;
    	if(goalAreaArea[i] / goalAreaCount[i] > maxValue && goalAreaCount[i] > 0){
    		maxValue = goalAreaArea[i] / goalAreaCount[i];
    	}
    	for(int j=0; j<(int)goalAreaMembers[i].size(); j++){
    		frnt[goalAreaMembers[i][j]].reward = goalAreaValue[i];
    	}
    }
    /*
    cout << "FnrtValue: " << endl;
    for(int i=0; i<(int)this->frntCentroid.size(); i++){
    	cout << "   " << this->frntValue[i] << endl;
    }
	*/
    // set value for all external frontiers
    for(int i=0;i<(int)frntExits.size(); i++){
    	frnt[frntExits[i]].reward = 100*maxValue;
    }
    /*
    // review value of all frontiers
    cout << "FnrtValue: " << endl;
    for(int i=0; i<(int)this->frntCentroid.size(); i++){
    	cout << "   " << this->frntValue[i] << endl;
    }
	*/
	// find frontiers likely to intersect based upon orientation and placement

	////////////////////////////////////////////////////////////////////////////////// Display inference Map ////////////////////////////////////////////////////////////////////////////////////

	Mat frontiers = this->getFrontiersImage();
	resize(frontiers, plot, Size(), 1,1);
	for(int i=0; i<(int)frnt.size(); i++){
		circle(frontiers, Point(frnt[i].projection[1], frnt[i].projection[0]), 1, Scalar(100), -1);
		circle(frontiers, Point(frnt[i].centroid[1], frnt[i].centroid[0]),    1, Scalar(200), -1);
		line(frontiers, Point(frnt[i].projection[1], frnt[i].projection[0]), Point(frnt[i].centroid[1], frnt[i].centroid[0]), Scalar(100), 1, 8);
	}

	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy;
	findContours(outerHullDrawing, contours2, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	drawContours(frontiers, contours2, 0, Scalar(127), 1, 8);

	Mat inferDisplay = Mat::zeros(inferCalc.size(), CV_8UC3);
	Mat obstaclesAndHull = Mat::zeros(inferCalc.size(), CV_8UC1);
	this->inferredMiniMap = Mat::zeros(this->nCols,this->nRows, CV_8UC1);

	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			Point pa, pb;
			pa.x = this->gSpace*j + this->gSpace/2;
			pa.y = this->gSpace*i + this->gSpace/2;
			pb.x = this->gSpace*j - this->gSpace/2;
			pb.y = this->gSpace*i - this->gSpace/2;
			if(this->costMap[i][j] < 50){ // free space with inflation
				Scalar color;
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
				rectangle(inferDisplay,pa,pb,color,-1);
				this->inferredMiniMap.at<uchar>(i,j,0) = 0;
			}
			else if(this->costMap[i][j] == INFINITY){ // obstacle
				Scalar color;
				color[0] = 127;
				color[1] = 127;
				color[2] = 127;
				rectangle(inferDisplay,pa,pb,color,-1);
				rectangle(obstaclesAndHull,pa,pb,Scalar(255),-1);
				this->inferredMiniMap.at<uchar>(i,j,0) = 255;
			}
			else if(this->costMap[i][j] == 100){ // unknown space
				Scalar color;
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
				rectangle(inferDisplay,pa,pb,color,-1);
				this->inferredMiniMap.at<uchar>(i,j,0) = 100;
			}
			else if(this->costMap[i][j] == 50){ // unknown space
				Scalar color;
				color[0] = 0;
				color[1] = 0;
				color[2] = 127;
				rectangle(inferDisplay,pa,pb,color,-1);
				this->inferredMiniMap.at<uchar>(i,j,0) = 100;
			}
		}
	}

	for(int i=0; i<(int)frnt.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 255;

		Point pa, pb;

		pa.x = this->gSpace*frnt[i].projection[1] + this->gSpace/2;
		pa.y = this->gSpace*frnt[i].projection[0] + this->gSpace/2;

		circle(inferDisplay,pa,5,color,-1);

		pb.x = this->gSpace*frnt[i].centroid[1] + this->gSpace/2;
		pb.y = this->gSpace*frnt[i].centroid[0] + this->gSpace/2;

		line(inferDisplay,pa,pb,color,5,8);
	}

	for(int i=0; i<(int)frntExits.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 255;
		color[2] = 0;

		Point pa;
		pa.x = this->gSpace*frnt[frntExits[i]].projection[1] + this->gSpace/2;
		pa.y = this->gSpace*frnt[frntExits[i]].projection[0] + this->gSpace/2;

		int radius = frnt[i].reward;

		circle(inferDisplay,pa,10,color,-1);
	}


    for(int i=0; i<(int)outerHull.size()-1; i++){
		Scalar color;
		color[0] = 0;
		color[1] = 255;
		color[2] = 0;

    	Point pa, pb;
    	pa.x = outerHull[i].x;
    	pa.y = outerHull[i].y;
    	pb.x = outerHull[i+1].x;
    	pb.y = outerHull[i+1].y;
    	line(inferDisplay, pa, pb, color, 2, 8);
    	line(obstaclesAndHull, pa, pb, Scalar(255), 2, 8);
    }

    for(int i=0; i<1; i++){
		Scalar color;
		color [0] = 0;
		color[1] = 255;
		color[2] = 0;

    	Point pa, pb;
    	pa.x = outerHull[0].x;
    	pa.y = outerHull[0].y;
    	pb.x = outerHull[outerHull.size()-1].x;
    	pb.y = outerHull[outerHull.size()-1].y;
    	line(inferDisplay, pa, pb,color, 2, 8);
    	line(obstaclesAndHull, pa, pb, Scalar(255), 2, 8);
    }

    this->hullPts.clear();
    for(size_t i=0; i<outerHull.size(); i++){
    	vector<int> t;
    	t.push_back(outerHull[i].x);
    	t.push_back(outerHull[i].y);
    	this->hullPts.push_back(t);
    }

    // for each frontier

    for(size_t i=0; i<frnt.size(); i++){
    	// find all obstacle points within distance r of each frontier member
    	for(size_t j=0; j<frnt[i].members.size(); j++){

    		for(int k = -2; k<3; k++){
    			for(int l=-2; l<3; l++){

    				if(this->costMap[frnt[i].members[j][0] + k][frnt[i].members[j][1] + l] > 255){
    					Scalar color;
						color[0] = 0;
						color[1] = 255;
						color[2] = 0;
    					vector<int> t;
    					t.push_back(frnt[i].members[j][0] + k);
    					t.push_back(frnt[i].members[j][1] + l);

    					Point pa;
    					pa.x = t[1]*this->gSpace;
    					pa.y = t[0]*this->gSpace;

    					frnt[i].obstacles.push_back(t);
    					circle(inferDisplay,pa,2,color,-1);
    				}
    			}
    		}
    	}
    }


    imshow( "infer miniMap", this->inferredMiniMap );
    waitKey(1);

    imshow( "infer display", inferDisplay );
    waitKey(1);


    Mat unobservedSpace;
    cvtColor(cont, unobservedSpace, CV_BGR2GRAY);
    threshold(unobservedSpace, unobservedSpace,1, 255, THRESH_BINARY);
    bitwise_not(unobservedSpace, unobservedSpace);

    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy2;

    /// Detect edges using canny
    Canny( unobservedSpace, canny_output, 100, 200, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    /// Get the moments
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

	for( size_t i = 0; i < contours.size(); i++ ){
		double myArea = contourArea(contours[i]);
		bool flag = true;

		if(myArea > 1000){ // not too small
			cerr << "Area" << contourArea(contours[i]) << endl;
			Mat mask = Mat::zeros(unobservedSpace.rows, unobservedSpace.cols, CV_8UC1);
			drawContours(mask, contours, i, Scalar(255), CV_FILLED);
			Scalar meanVal = mean(inferDisplay, mask);
			cerr << "meanVal: " << meanVal << endl;

			/// Draw contours
			if(meanVal[0] + meanVal[1] + meanVal[2] + meanVal[3] < 100){ // mostly unexplored space

				bool flag = true;
				for(size_t j=0; j<frnt.size(); j++){
					Point fP;
					fP.x = frnt[j].projection[1]*this->gSpace;
					fP.y = frnt[j].projection[0]*this->gSpace;
					float min = INFINITY;
					int mindex = -1;

					if(pointPolygonTest(contours[i], fP, true) >= 0){ // has at least 1 frontier

						Mat tempC = Mat::zeros(inferDisplay.size(), CV_8UC1);
						drawContours(tempC, contours, i, Scalar(255), -1);
						Size f1;
						double scale = 1/double(this->gSpace);
						resize( tempC, tempC, f1, scale, scale, INTER_AREA);

						vector<vector<Point> > tC;
					    vector<Vec4i> tH;

					    /// Detect edges using canny
					    Canny( tempC, tempC, 100, 200, 3 );
					    /// Find contours
					    findContours( tempC, tC, tH, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);


						drawContours( this->inferredMiniMap, tC, 0, Scalar(0), -1);
					    imshow( "i2", this->inferredMiniMap );
					    waitKey(1);


						Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
						drawContours( drawing, contours, i, color, 2, 8, hierarchy2, 0, Point() );

						/// Show in a window
						imshow( "cContours", drawing );
						waitKey(1);

						float xSum = 0;
						float ySum = 0;

						for(size_t j=0; j<contours[i].size(); j++){
							xSum += contours[i][j].x;
							ySum += contours[i][j].y;
						}

						Point center;
						center.x = xSum / contours[i].size();
						center.y = ySum / contours[i].size();

						vector<float> length;
						float meanLength = 0;
						for(size_t j=0; j<contours[i].size(); j++){
						  //theta.push_back( atan2( center.y - contours[i][j].y, center.x - contours[i][j].x));
							float l = sqrt( pow(center.y - contours[i][j].y, 2) +  pow(center.x - contours[i][j].x,2));
							length.push_back( l );
							meanLength += l;
						}
						meanLength /= length.size();

						vector<int> lengthHistogram;
						vector<float> lengthSequence;
						getLengthHistogram(length, meanLength, lengthHistogram, lengthSequence);

						for(size_t hi=0; hi<this->masterHistogramList.size(); hi++){
							float val = 0;
							for(int hl=0; hl< masterHistogramList[i].size(); hl++){
								val += abs(lengthHistogram[hl] - masterHistogramList[hi][hl]);
							}

							if(val < min && val > 0){
								min = val;
								mindex = i;
								break;
							}
						}

						Mat match = imread( "/home/andy/Dropbox/workspace/fabmap2Test/frontiersArea/training/" + masterNameList[mindex] + ".jpg", CV_LOAD_IMAGE_GRAYSCALE );
						float matchMeanLength = this->masterMeanLengthList[mindex];
						double fxy = meanLength / matchMeanLength;
						cv::Size f0;
						resize(match,match,f0,fxy,fxy, INTER_AREA); // scale to match the partial segment

						imshow("match", match);
						waitKey(1);

						Mat contourToInfer = Mat::zeros(unobservedSpace.rows, unobservedSpace.cols, CV_8UC1);
						drawContours(contourToInfer, contours, i, Scalar(255), CV_FILLED);
						fxy = 1.2;
						resize(contourToInfer, contourToInfer, f0, fxy, fxy, INTER_AREA);

						int rx = (contourToInfer.cols - obstaclesAndHull.cols)/2;
						int ry = (contourToInfer.rows - obstaclesAndHull.rows)/2;
						Rect roi(rx, ry, obstaclesAndHull.rows, obstaclesAndHull.cols);
						// Transform it into the C++ cv::Mat format
						contourToInfer = contourToInfer(roi);

						// generate a mask of the area being searched over that is 1.1 * the actual size to include exterior info
						imshow("contour to infer", contourToInfer);
						waitKey(1);
						// apply mask to obstacle + inferred obstacles

						Mat trial = Mat::zeros(unobservedSpace.rows, unobservedSpace.cols, CV_8UC1);

						bitwise_and(obstaclesAndHull, contourToInfer, trial);
						imshow("trial", trial);
						waitKey(0);

						// perform rotation based ransac over the images




						// get skeleton graph of found map

						// connect to existing skeleton graph in that area



						// geometric inference - make geometric image / costmap

						// 0 = free
						// 1 = inferred free
						// 255 = wall
						// 254 = inferred wall

					}
				}
			}
		}
	}


/*
    // clustering obstacles

    for(size_t i=0; i<frnt.size(); i++){
    	 // group obstacles into groups based on touching

    	vector<vector<int> > oL = frnt[i].obstacles;
		vector<vector<int> > oSet;
		vector<vector<int> > cSet;


		while(oL.size() > 0){

			oSet.push_back(oL[oL.size()-1]);
			oL.pop_back();
			vector<vector<int> > tCluster;

			while(oSet.size() > 0){
				cSet.push_back(oSet[oSet.size()-1]);
				tCluster.push_back(oSet[oSet.size()-1]);
				vector<int> pt = oSet[oSet.size()-1];
				oSet.pop_back();

				for(size_t j=0; j<frnt[i].obstacles.size(); j++){
					if(pow(frnt[i].obstacles[j][0]-pt[0],2) + pow(frnt[i].obstacles[j][1]-pt[1],2) < 2){
						bool flag = true;
						for(size_t k=0; k<cSet.size(); k++){
							if(frnt[i].obstacles[j] == cSet[k]){
								flag = false;
								break;
							}
						}
						if(flag){
							for(size_t k=0; k<oL.size(); k++){
								if(oL[k] == frnt[i].obstacles[j]){
									oL.erase(oL.begin()+k, oL.begin()+k+1);
									break;
								}
							}
							oSet.push_back(frnt[i].obstacles[j]);
						}
					}
				}
			}
			frnt[i].obsClusters.push_back(tCluster);
		}
    }

    // draw each obstacle cluster a different cluster
    for(size_t i=0; i<frnt.size(); i++){
    	for(size_t j=0; j< frnt[i].obsClusters.size(); j++){
    		Scalar color;
			color[0] = rand() % 255;
			color[1] = rand() % 255;
			color[2] = rand() % 255;
    		for(size_t k=0; k<frnt[i].obsClusters[j].size(); k++){
    			circle(inferDisplay, Point(frnt[i].obsClusters[j][k][1]*this->gSpace,frnt[i].obsClusters[j][k][0]*this->gSpace),2,color,-1,8);
    		}
    	}
    }

    // get orientation of obstacle clusters
    // extend obstacles into unobserved space, stop at obstacle or other observed space

     */

}

void graph::getLengthHistogram(vector<float> length, float meanLength, vector<int> &histogram, vector<float> &sequence){
	float binWidth = 0.03; // 100 bins over 3 unit
	for(float i = 0; i<3; i += binWidth){
		histogram.push_back(0);
	}

	float lIter = length.size() / 100;
	for(size_t i = 0; i<100; i++){
		int index = round(i*lIter);
		float normLength = length[index] / meanLength;

		sequence.push_back(normLength);

		if(normLength > 3){
			histogram[histogram.size()-1]++;
		}
		else if (normLength < 0.05){
			histogram[0]++;
		}
		else{
			float bin = round(normLength / binWidth); // normallize length and put in a bin;
			histogram[bin]++;
		}
	}
}


void graph::extractInferenceContour(){
	Mat temp;
	bitwise_or(this->obstacleMat, this->inferenceMat, temp);
	bitwise_or(temp, this->freeMat, temp);
	bitwise_not(temp,temp);

	imshow("q", temp);
}

void graph::getFrontierOrientation(vector<frontier> &frnt){
	for(int i=0; i<(int)frnt.size(); i++){
		vector<float> temp;
		temp.push_back(0);
		temp.push_back(0);
		frnt[i].orient = temp;
		float count = 0;
		// check each member of each cluster
		if(frnt[i].members.size() > 1){
			for(int j=0; j<(int)frnt[i].members.size(); j++){
				// check 4Nbr for being unobserved
				int xP = frnt[i].members[j][0];
				int yP = frnt[i].members[j][1];
				if(this->costMap[xP+1][yP] == 100){
					frnt[i].orient[0] += 1;
					count++;
				}
				if(this->costMap[xP-1][yP] == 100){
					frnt[i].orient[0] -= 1;
					count++;
				}
				if(this->costMap[xP][yP+1] == 100){
					frnt[i].orient[1] += 1;
					count++;
				}
				if(this->costMap[xP][yP-1] == 100){
					frnt[i].orient[1] -= 1;
					count++;
				}
			}
			if(count > 0){
				frnt[i].orient[0] /= count;
				frnt[i].orient[1] /= count;
			}
		}
	}
}


vector<Point> graph::getImagePoints(Mat &image){
	vector<Point> temp;
	for(int i=0; i<image.rows; i++){
		for(int j=0; j<image.cols; j++){
			Scalar intensity = image.at<uchar>(i,j);
			if(intensity[0] > 100){
				Point t;
				t.x = j;
				t.y = i;
				temp.push_back(t);
			}
		}
	}
	return temp;
}

Mat graph::getObstaclesImage(){
	Mat temp = Mat::zeros(this->nRows,this->nCols,CV_8UC1);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] > 100){ // free space with inflation
				temp.at<uchar>(i,j,0) = 255;
			}
			else{
				temp.at<uchar>(i,j,0) = 0;
			}
		}
	}
	//imshow("obstacles map", temp);
	return temp;
}

Mat graph::getFreeSpaceImage(){
	Mat temp = Mat::zeros(this->nRows, this->nCols,CV_8UC1);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] <50){ // free space with inflation
				temp.at<uchar>(i,j,0) = 255;
			}
			else{
				temp.at<uchar>(i,j,0) = 0;
			}
		}
	}
	//imshow("freespace map", temp);
	return temp;
}

Mat graph::getFrontiersImage(){
	Mat temp = Mat::zeros(this->nRows, this->nCols,CV_8UC1);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] == 50){ // free space with inflation
				temp.at<uchar>(i,j,0) = 255;
			}
			else{
				temp.at<uchar>(i,j,0) = 0;
			}
		}
	}
	//imshow("frontiers map", temp);
	return temp;
}

void graph::shareMap(vector<vector<float> >& in){
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if((this->costMap[i][j] == 100 || this->costMap[i][j] == 50) && in[i][j] > 100){ // I dont think its observed, they do
				this->costMap[i][j] = in[i][j];
			}
		}
	}
}

int graph::getMaxIndex(vector<float> value){
	int maxdex;
	float maxValue = -INFINITY;
	for(int i=0; i<(int)value.size(); i++){
		if(value[i] > maxValue){
			maxdex = i;
			maxValue  = value[i];
		}
	}
	return maxdex;
}

int graph::getMinIndex(vector<float> value){
	int mindex;
	float minValue = INFINITY;
	for(int i=0; i<(int)value.size(); i++){
		if(value[i] < minValue){
			mindex = i;
			minValue  = value[i];
		}
	}
	return mindex;
}

void graph::frontierCosts(world &gMap, vector<frontier> &frnt, vector<agent> &bot){
	for(int i=0; i<(int)bot.size(); i++){
		for(int j=0; j<(int)frnt.size(); j++){
			bot[i].fCost[j] = this->aStarDist(bot[i].cLoc, frnt[j].centroid, gMap);
		}
	}
}

void graph::findFrontiers(){
	this->frntList.erase(this->frntList.begin(),this->frntList.end());
	this->frntsExist = false;
	for(int i=1; i<this->nRows-1; i++){
		for(int j=1; j<this->nCols-1; j++){
			bool newFrnt = false;
			if(this->costMap[i][j] == 100 || this->costMap[i][j] == 50){ // i'm unobserved
				if(this->costMap[i+1][j] < 50){ //  but one of my nbrs is observed
					newFrnt = true;
				}
				else if(this->costMap[i-1][j] < 50){ //  but one of my nbrs is observed
					newFrnt = true;
				}
				else if(this->costMap[i][j+1] < 50){ //  but one of my nbrs is observed
					newFrnt = true;
				}
				else if(this->costMap[i][j-1] < 50){ //  but one of my nbrs is observed
					newFrnt = true;
				}
			}
			if(newFrnt){
				vector<int> fT;
				fT.push_back(i);
				fT.push_back(j);
				frntList.push_back(fT);
				this->frntsExist = true;
				this->costMap[i][j] = 50;
			}
		}
	}
}

void graph::observe(vector<int> cLoc, world &gMap){

	for(int i=0; i<gMap.obsGraph[cLoc[0]][cLoc[1]].size(); i++){
		this->costMap[gMap.obsGraph[cLoc[0]][cLoc[1]][i][0]][gMap.obsGraph[cLoc[0]][cLoc[1]][i][1]] = gMap.costMap[gMap.obsGraph[cLoc[0]][cLoc[1]][i][0]][gMap.obsGraph[cLoc[0]][cLoc[1]][i][1]];
	}

	// set obstacles in costmap
	int minObs[2], maxObs[2];
	if(cLoc[0] - this->obsThresh > 0){
		minObs[0] = cLoc[0]-this->obsThresh;
	}
	else{
		minObs[0] = 0;
	}

	if(cLoc[0] + this->obsThresh < this->nRows){
		maxObs[0] = cLoc[0]+this->obsThresh;
	}
	else{
		maxObs[0] = this->nRows;
	}

	if(cLoc[1] - this->obsThresh > 0){
		minObs[1] = cLoc[1]-this->obsThresh;
	}
	else{
		minObs[1] = 0;
	}

	if(cLoc[1] + this->obsThresh < this->nCols){
		maxObs[1] = cLoc[1]+this->obsThresh;
	}
	else{
		maxObs[1] = this->nCols;
	}

	for(int i=1+minObs[0]; i<maxObs[0]-1; i++){
		for(int j=1+minObs[1]; j<maxObs[1]-1; j++){
			if(gMap.costMap[i][j] != 0){ // am i an obstacle?
				for(int k=i-1; k<i+2; k++){
					for(int l=j-1; l<j+2; l++){
						if(this->costMap[k][l] < 50){ // are any of my nbrs visible?
							this->costMap[i][j] = INFINITY; // set my cost
						}

					}
				}
			}

		}
	}
}

void graph::showCostMapPlot(int index){
	char buffer[50];
	int n = sprintf(buffer, "CostMap %d",index);
	imshow(buffer, this->tempImage);
	waitKey(1);
}

void graph::addAgentToPlot(world &gMap, vector<int> cLoc, int myColor[3]){
	Scalar color;
	color[0] = myColor[0];
	color[1] = myColor[1];
	color[2] = myColor[2];

	circle(this->tempImage,gMap.pointMap[cLoc[0]][cLoc[1]],gMap.gSpace,color,-1);
}

void graph::buildCostMapPlot(world &gMap, vector<frontier> &frnt){
	this->tempImage = Mat::zeros(this->nRows*gMap.gSpace, this->nCols*gMap.gSpace,CV_8UC3);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			Point pa, pb;
			pa.x = gMap.pointMap[i][j].x + gMap.gSpace/2;
			pa.y = gMap.pointMap[i][j].y + gMap.gSpace/2;
			pb.x = gMap.pointMap[i][j].x - gMap.gSpace/2;
			pb.y = gMap.pointMap[i][j].y - gMap.gSpace/2;
			if(this->costMap[i][j] < 50){ // free space with inflation
				Scalar color;
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
				rectangle(this->tempImage,pa,pb,color,-1);
			}
			else if(this->costMap[i][j] == INFINITY){ // obstacle
				Scalar color;
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
				rectangle(this->tempImage,pa,pb,color,-1);
			}
			else if(this->costMap[i][j] == 100){ // unknown space
				Scalar color;
				color[0] = 127;
				color[1] = 127;
				color[2] = 127;
				rectangle(this->tempImage,pa,pb,color,-1);
			}
		}
	}
	for(int i=0; i<(int)this->frntList.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 127;

		Point pa, pb;
		pa.x = gMap.pointMap[this->frntList[i][0]][this->frntList[i][1]].x + gMap.gSpace/2;
		pa.y = gMap.pointMap[this->frntList[i][0]][this->frntList[i][1]].y + gMap.gSpace/2;
		pb.x = gMap.pointMap[this->frntList[i][0]][this->frntList[i][1]].x - gMap.gSpace/2;
		pb.y = gMap.pointMap[this->frntList[i][0]][this->frntList[i][1]].y - gMap.gSpace/2;

		rectangle(this->tempImage,pa,pb,color,-1);
	}

	for(int i=0; i<frnt.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 255;

		Point pa, pb;
		pa.x = gMap.pointMap[frnt[i].centroid[0]][frnt[i].centroid[1]].x + gMap.gSpace/2;
		pa.y = gMap.pointMap[frnt[i].centroid[0]][frnt[i].centroid[1]].y + gMap.gSpace/2;
		pb.x = gMap.pointMap[frnt[i].centroid[0]][frnt[i].centroid[1]].x - gMap.gSpace/2;
		pb.y = gMap.pointMap[frnt[i].centroid[0]][frnt[i].centroid[1]].y - gMap.gSpace/2;

		rectangle(this->tempImage,pa,pb,color,-1);
	}
}

vector<vector<int> > graph::aStarPath(vector<int> sLoc, vector<int> gLoc, world &gMap){
	if(sLoc == gLoc){
		vector<vector<int> > totalPath;
		for(int i=0; i<4; i++){
			vector<int> t = sLoc;
			totalPath.push_back(t);
		}
		return totalPath;
	}
	vector<vector<int> > cSet; // 1 means in closed set, 0 means not
	vector<vector<int> > oSet; // 1 means in open set, 0 means not
	vector<vector<float> > gScore; // known cost from initial node to n
	vector<vector<float> > fScore; // gScore + heuristic score (dist to goal + imposed cost)
	vector<vector<vector<int> > > cameFrom; // each square has a vector of the location it came from
	for(int i=0;i<this->nRows; i++){
		vector<int> tC;
		vector<int> tO;
		vector<vector<int> > tCF;
		vector<float> tG;
		vector<float> tF;
		for(int j=0; j<this->nCols; j++){
			tC.push_back(0);
			tO.push_back(0);
			vector<int> ttCF;
			ttCF.push_back(-1);
			ttCF.push_back(-1);

			tCF.push_back(ttCF);
			tG.push_back(INFINITY);
			tF.push_back(INFINITY);
		}
		cSet.push_back(tC);
		oSet.push_back(tC);
		cameFrom.push_back(tCF);
		gScore.push_back(tG); // init scores to inf
		fScore.push_back(tF); // init scores to inf
	}
	oSet[sLoc[0]][sLoc[1]] = 1; // starting node has score 0
	gScore[sLoc[0]][sLoc[1]] = 0; // starting node in open set
	fScore[sLoc[0]][sLoc[1]] = gScore[sLoc[0]][sLoc[1]] + gMap.getEuclidDist(sLoc[0],sLoc[1],gLoc[0],gLoc[1]) + this->costMap[sLoc[0]][sLoc[1]];
	int foo = 1;
	int finishFlag = 0;

	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> iMin;
		iMin.push_back(0);
		iMin.push_back(0);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(oSet[i][j] > 0 && fScore[i][j] < min){
					min = fScore[i][j];
					iMin[0] = i;
					iMin[1] = j;
				}
			}
		}
		vector<int> cLoc = iMin;
		/////////////////////// end finding current node
		if(cLoc == gLoc){ // if the current node equals goal, construct path
			finishFlag = 1;
			vector<vector<int> > totalPath;
			totalPath.push_back(gLoc);
			while(cLoc != sLoc){ // work backwards to start
				vector<int> temp;
				temp.push_back(cameFrom[cLoc[0]][cLoc[1]][0]); // work backwards
				temp.push_back(cameFrom[cLoc[0]][cLoc[1]][1]);
				cLoc = temp;
				totalPath.push_back(cLoc); // append path
			}
			reverse(totalPath.begin(),totalPath.end());
			return totalPath;
		} ///////////////////////////////// end construct path
		oSet[cLoc[0]][cLoc[1]] = 0;
		cSet[cLoc[0]][cLoc[1]] = 1;
		for(int nbrRow=cLoc[0]-1;nbrRow<cLoc[0]+2;nbrRow++){
			for(int nbrCol=cLoc[1]-1; nbrCol<cLoc[1]+2; nbrCol++){
				if(cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
					continue;
				}
				float tGScore;
				tGScore = gScore[cLoc[0]][cLoc[1]] + gMap.getEuclidDist(sLoc[0],sLoc[1],nbrRow,nbrCol); // calc temporary gscore
				if(oSet[nbrRow][nbrCol] == 0){
					oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
				}
				else if(tGScore >= gScore[nbrRow][nbrCol]){ // is temp gscore better than stored g score of nbr
					continue;
				}
				cameFrom[nbrRow][nbrCol][0] = cLoc[0];
				cameFrom[nbrRow][nbrCol][1] = cLoc[1];
				gScore[nbrRow][nbrCol] = tGScore;
				fScore[nbrRow][nbrCol] = gScore[nbrRow][nbrCol] + gMap.getEuclidDist(gLoc[0],gLoc[1],nbrRow,nbrCol) + this->costMap[nbrRow][nbrCol];
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
		foo = 0;
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				foo+= oSet[i][j];
			}
		}
	}
	vector<vector<int> > totalPath;
	for(int i=0; i<4; i++){
		vector<int> t = sLoc;
		totalPath.push_back(t);
	}
	return totalPath;
}

void graph::findNearestFrontier(agent &bot, vector<frontier> &frnt, world &gMap){
	vector<int> index;
	float minDist = INFINITY;
	for(int i=0; i<frnt.size(); i++){
		float t = this->aStarDist(bot.cLoc, frnt[i].centroid, gMap);
		if(minDist > t){
			minDist = t;
			index = frnt[i].centroid;
		}
	}
	bot.gLoc = index;
}

float graph::aStarDist(vector<int> sLoc, vector<int> gLoc, world &gMap){
	vector<vector<int> > cSet; // 1 means in closed set, 0 means not
	vector<vector<int> > oSet; // 1 means in open set, 0 means not
	vector<vector<float> > gScore; // known cost from initial node to n
	vector<vector<float> > fScore; // gScore + heuristic score (dist to goal + imposed cost)
	for(int i=0;i<this->nRows; i++){
		vector<int> tC;
		vector<int> tO;
		vector<float> tG;
		vector<float> tF;
		for(int j=0; j<this->nCols; j++){
			tC.push_back(0);
			tO.push_back(0);
			tG.push_back(INFINITY);
			tF.push_back(INFINITY);
		}
		cSet.push_back(tC);
		oSet.push_back(tC);
		gScore.push_back(tG); // init scores to inf
		fScore.push_back(tF); // init scores to inf
	}
	oSet[sLoc[0]][sLoc[1]] = 1; // starting node has score 0
	gScore[sLoc[0]][sLoc[1]] = 0; // starting node in open set
	fScore[sLoc[0]][sLoc[1]] = gScore[sLoc[0]][sLoc[1]] + gMap.getEuclidDist(sLoc[0],sLoc[1],gLoc[0],gLoc[1]) + this->costMap[sLoc[0]][sLoc[1]];
	int foo = 1;
	int finishFlag = 0;
	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> iMin;
		iMin.push_back(0);
		iMin.push_back(0);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(oSet[i][j] > 0 && fScore[i][j] < min){
					min = fScore[i][j];
					iMin[0] = i;
					iMin[1] = j;
				}
			}
		}
		vector<int> cLoc = iMin;
		/////////////////////// end finding current node
		if(cLoc == gLoc){ // if the current node equals goal, construct path
			finishFlag = 1;
			return fScore[gLoc[0]][gLoc[1]];
		} ///////////////////////////////// end construct path
		oSet[cLoc[0]][cLoc[1]] = 0;
		cSet[cLoc[0]][cLoc[1]] = 1;
		for(int nbrRow=cLoc[0]-1;nbrRow<cLoc[0]+2;nbrRow++){
			for(int nbrCol=cLoc[1]-1; nbrCol<cLoc[1]+2; nbrCol++){
				if(cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
					continue;
				}
				float tGScore;
				tGScore = gScore[cLoc[0]][cLoc[1]] + gMap.getEuclidDist(cLoc[0],cLoc[1],nbrRow,nbrCol); // calc temporary gscore
				if(oSet[nbrRow][nbrCol] == 0){
					oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
				}
				else if(tGScore >= gScore[nbrRow][nbrCol]){ // is temp gscore better than stored g score of nbr
					continue;
				}
				gScore[nbrRow][nbrCol] = tGScore;
				fScore[nbrRow][nbrCol] = gScore[nbrRow][nbrCol] + gMap.getEuclidDist(gLoc[0],gLoc[1],nbrRow,nbrCol) + this->costMap[nbrRow][nbrCol];
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
		foo = 0;
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				foo+= oSet[i][j];
			}
		}
	}
	return 0;
}

void graph::clusterFrontiers(world &gMap, vector<frontier> &frnt){

	cerr << "frnts in: " << frnt.size() << endl;
	for(int i=0; i<frnt.size(); i++){
		cerr << "   " << frnt[i].centroid[0] << " / " << frnt[i].centroid[1] << endl;
	}

	vector<vector<int> > qFrnts = this->frntList;

	// check to see if frnt.centroid is still a frontier cell, if so keep, else delete
	for(int i=0; i<(int)frnt.size(); i++){
		frnt[i].editFlag = true;
		bool flag = true;
		for(int j=0; j<(int)qFrnts.size(); j++){
			if(frnt[i].centroid == qFrnts[j]){
				flag = false;
				qFrnts.erase(qFrnts.begin()+j);
			}
		}
		if(flag){
			frnt.erase(frnt.begin()+i);
		}
		else{
			frnt[i].editFlag = false;
		}
	}

	cerr << "remaining frnts: " << frnt.size() << endl;
	for(int i=0; i<frnt.size(); i++){
		cerr << "   " << frnt[i].centroid[0] << " / " << frnt[i].centroid[1] << endl;
	}

	// breadth first search through known clusters
	for(int i=0; i<(int)frnt.size(); i++){ // keep checking for new frontier clusters while there are unclaimed frontiers
		vector<vector<int> > q; // current cluster
		vector<vector<int> > qP; // open set in cluster
		qP.push_back(frnt[i].centroid);

		while((int)qP.size() > 0){ // find all nbrs of those in q
			vector<int> seed = qP[0];
			q.push_back(qP[0]);
			qP.erase(qP.begin(),qP.begin()+1);
			for(int ni = seed[0]-2; ni<seed[0]+3; ni++){
				for(int nj = seed[1]-2; nj<seed[1]+3; nj++){
					for(int i=0; i<(int)qFrnts.size(); i++){
						if(qFrnts[i][0] == ni && qFrnts[i][1] == nj){
							qP.push_back(qFrnts[i]); // in range, add to open set
							qFrnts.erase(qFrnts.begin() + i);
						}
					}
				}
			}
		}
		frnt[i].members = q; // save to list of clusters
	}

	// breadth first search
	while((int)qFrnts.size() > 0){ // keep checking for new frontier clusters while there are unclaimed frontiers
		vector<vector<int> > q; // current cluster
		vector<vector<int> > qP; // open set in cluster
		qP.push_back(qFrnts[0]);
		qFrnts.erase(qFrnts.begin());

		while((int)qP.size() > 0){ // find all nbrs of those in q
			vector<int> seed = qP[0];
			q.push_back(qP[0]);
			qP.erase(qP.begin(),qP.begin()+1);
			for(int ni = seed[0]-1; ni<seed[0]+2; ni++){
				for(int nj = seed[1]-1; nj<seed[1]+2; nj++){
					for(int i=0; i<(int)qFrnts.size(); i++){
						if(qFrnts[i][0] == ni && qFrnts[i][1] == nj){
							qP.push_back(qFrnts[i]); // in range, add to open set
							qFrnts.erase(qFrnts.begin() + i, qFrnts.begin()+i+1);
						}
					}
				}
			}
		}
		frontier a(gMap.nRows, gMap.nCols);
		frnt.push_back(a);
		frnt[frnt.size()-1].members = q; // save to list of clusters
	}

	for(int i=0; i<(int)frnt.size(); i++){ // number of clusters
		if(frnt[i].editFlag){
			float minDist = INFINITY;
			int minDex;
			for(int j=0; j<(int)frnt[i].members.size(); j++){ // go through each cluster member
				float tempDist = 0;
				for(int k=0; k<(int)frnt[i].members.size(); k++){ // and get cumulative distance to all other members
					tempDist += gMap.getEuclidDist(frnt[i].members[j][0],frnt[i].members[j][1],frnt[i].members[k][0],frnt[i].members[k][1]);
				}
				if(tempDist < minDist){
					minDist = tempDist;
					minDex = j;
				}
			}
			frnt[i].centroid = frnt[i].members[minDex];
		}
	}
}

Point extendLine(Point a, Point m){
	float dx = m.x - a.x;
	float dy = m.y - a.y;
	Point p;
	p.x = m.x + dx;
	p.y = m.y + dy;
	return(p);
}

float distToLine(Vec4i w, Point a){
	float x1 = w[0];
	float y1 = w[1];
	float x2 = w[2];
	float y2 = w[3];

	float x0 = a.x;
	float y0 = a.y;

	float denom = sqrt(pow(x2-x1,2) + pow(y2-y1,2));
	if(denom != 0){
		float dist = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))/denom;
		return dist;
	}
	else{
		return(-1);
	}
}

Point findIntersection(Vec4i w1, Vec4i w2){
	float x1 = w1[0];
	float y1 = w1[1];
	float x2 = w1[2];
	float y2 = w1[3];

	float x3 = w2[0];
	float y3 = w2[1];
	float x4 = w2[2];
	float y4 = w2[3];

	float denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);

	if(denom != 0){
		Point p;
	    p.x = ((x1*y2-y1*x2)*(x3-x4) - (x3*y4-y3*x4)*(x1-x2))/denom;
	    p.y = ((x1*y2-y1*x2)*(y3-y4) - (x3*y4-y3*x4)*(y1-y2))/denom;
	    return(p);
	}
	else{
		Point p;
		p.x = -1;
		p.y = -1;
		return(p);
	}
}

/*
vector< vector<int> > graph::kMeansClusteringEuclid(vector<int> openFrnt, int numClusters){
	vector<int> tempFrnt = openFrnt;
	vector<int> cluster[numClusters]; // array of vectors
	float bestClusterDist = INFINITY;
	vector< vector<int> > bestClusterSet;
	for(int i=0; i<numClusters; i++){
		vector<int> temp;
		bestClusterSet.push_back(temp);
	}
	bool convergeFlag = false;
	bool initFlag = true;
	while(initFlag){
		for(int i=0; i<numClusters; i++){ // generate one random cluster per UAV
			if(tempFrnt.size() == 0){
				initFlag = false;
			}
			else{
				int temp = rand() % tempFrnt.size();
				cluster[i].push_back(tempFrnt[temp]); // assign random cluster centroid initially
				tempFrnt.erase(tempFrnt.begin()+temp); // don't allow center to be taken
			}
		}
	}

	tempFrnt = openFrnt;
	while(convergeFlag == false){ // until the solution converges
		float tempDist = 0;
		for(int i=0; i<numClusters; i++){ // compute centroid frontier of each cluster
			if(cluster[i].size() > 2){ // no point if a cluster has 1 or 2 members
				float minDist[2] = {INFINITY, -1};
				for(int j=0; j<int(cluster[i].size()); j++){ // find A* dist from each node in each cluster
					float distSum = 0;
					for(int k=0; k<int(cluster[i].size()); k++){ // to each other node in that cluster
						distSum += this->distGraph[ cluster[i][j] ][ cluster[i][k] ];
					} // end to each other node
					if(distSum < minDist[0]){ // is it the most central
						minDist[0] = distSum;
						minDist[1] = cluster[i][j]; // assign as new centroid
					} // end is it most central
				} // end find A* dist from each node in each cluster
				cluster[i].erase(cluster[i].begin(), cluster[i].end()); // erase cluster
				cluster[i].push_back(minDist[1]); // set the new cluster centroid
				tempDist += minDist[0];
			} // end if has more than 2 members
			else if(cluster[i].size() == 2){ // if it has two members calc distance between them
				cluster[i].erase(cluster[i].begin()+1); // erase cluster
				tempDist += this->distGraph[ cluster[i][0] ][ cluster[i][1] ];
			} // end if it has two members calc distance
			else{ // this cluster has one member
				tempDist += 0;
			} // end this cluster has one member
		} // end compute centroid frontier of each cluster

		for(int i=0; i<int(openFrnt.size()); i++){ // find A* dist from each frontier to each cluster and group frontier in closest cluster
			float minDist[2] = {INFINITY, -1}; // [dist, index];
			bool isCenter = false;
			for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
				if(openFrnt[i] == cluster[j][0]){
					isCenter = true;
				}
			}
			if(!isCenter){ // is it a centroid
				for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
					float tempDist = this->distGraph[ openFrnt[i]] [cluster[j][0]]; // calc dist between
					if(tempDist < minDist[0]){ // new centroid is closer, so switch
						minDist[0] = tempDist;
						minDist[1] = j; // closest centroid
					} // end new centroid is closer
				} // end find closest cluster center to current front
				if(minDist[1] >= 0){
					cluster[int(minDist[1])].push_back(openFrnt[i]); // add to appropriate cluster
				}
			} // end is it a centroid
		} // end check each cluster centroid
		if(tempDist < bestClusterDist){ // found a better solution, has not converged yet
			bestClusterDist = tempDist;
			for(int i=0; i<numClusters; i++){
				bestClusterSet[i].erase(bestClusterSet[i].begin(),bestClusterSet[i].end());
				for(int j=0; j<int(cluster[i].size()); j++){
					bestClusterSet[i].push_back(cluster[i][j]);
				}
			}
			convergeFlag = false;
		}
		else{ // it has converged
			convergeFlag = true;
		}
	} // end until solution converges
	return(bestClusterSet);
}

vector< vector<int> > graph::kMeansClusteringTravel(vector<int> openFrnt, int numClusters){
	vector<int> tempFrnt = openFrnt;
	vector<int> cluster[numClusters]; // array of vectors
	float bestClusterDist = INFINITY;
	vector< vector<int> > bestClusterSet;
	for(int i=0; i<numClusters; i++){
		vector<int> temp;
		bestClusterSet.push_back(temp);
	}
	bool convergeFlag = false;
	bool initFlag = true;
	while(initFlag){
		for(int i=0; i<numClusters; i++){ // generate one random cluster per UAV
			if(tempFrnt.size() == 0){
				initFlag = false;
			}
			else{
				int temp = rand() % tempFrnt.size();
				cluster[i].push_back(tempFrnt[temp]); // assign random cluster centroid initially
				tempFrnt.erase(tempFrnt.begin()+temp); // don't allow center to be taken
			}
		}
	}

	tempFrnt = openFrnt;
	while(convergeFlag == false){ // until the solution converges
		float tempDist = 0;
		for(int i=0; i<numClusters; i++){ // compute centroid frontier of each cluster
			if(cluster[i].size() > 2){ // no point if a cluster has 1 or 2 members
				float minDist[2] = {INFINITY, -1};
				for(int j=0; j<int(cluster[i].size()); j++){ // find A* dist from each node in each cluster
					float distSum = 0;
					for(int k=0; k<int(cluster[i].size()); k++){ // to each other node in that cluster
						distSum += this->aStarDist(cluster[i][j],cluster[i][k]);
					} // end to each other node
					if(distSum < minDist[0]){ // is it the most central
						minDist[0] = distSum;
						minDist[1] = cluster[i][j]; // assign as new centroid
					} // end is it most central
				} // end find A* dist from each node in each cluster
				cluster[i].erase(cluster[i].begin(), cluster[i].end()); // erase cluster
				cluster[i].push_back(minDist[1]); // set the new cluster centroid
				tempDist += minDist[0];
			} // end if has more than 2 members
			else if(cluster[i].size() == 2){ // if it has two members calc distance between them
				cluster[i].erase(cluster[i].begin()+1); // erase cluster
				tempDist += this->aStarDist(cluster[i][0],cluster[i][1]);
			} // end if it has two members calc distance
			else{ // this cluster has one member
				tempDist += 0;
			} // end this cluster has one member
		} // end compute centroid frontier of each cluster

		for(int i=0; i<int(openFrnt.size()); i++){ // find A* dist from each frontier to each cluster and group frontier in closest cluster
			float minDist[2] = {INFINITY, -1}; // [dist, index];
			bool isCenter = false;
			for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
				if(openFrnt[i] == cluster[j][0]){
					isCenter = true;
				}
			}
			if(!isCenter){ // is it a centroid
				for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
					float tempDist = this->aStarDist(openFrnt[i], cluster[j][0]); // calc dist between
					if(tempDist < minDist[0]){ // new centroid is closer, so switch
						minDist[0] = tempDist;
						minDist[1] = j; // closest centroid
					} // end new centroid is closer
				} // end find closest cluster center to current front
				if(minDist[1] >= 0){
					cluster[int(minDist[1])].push_back(openFrnt[i]); // add to appropriate cluster
				}
			} // end is it a centroid
		} // end check each cluster centroid
		if(tempDist < bestClusterDist){ // found a better solution, has not converged yet
			bestClusterDist = tempDist;
			for(int i=0; i<numClusters; i++){
				bestClusterSet[i].erase(bestClusterSet[i].begin(),bestClusterSet[i].end());
				for(int j=0; j<int(cluster[i].size()); j++){
					bestClusterSet[i].push_back(cluster[i][j]);
				}
			}
			convergeFlag = false;
		}
		else{ // it has converged
			convergeFlag = true;
		}
	} // end until solution converges
	return(bestClusterSet);
}
*/

graph::~graph() {
	// TODO Auto-generated destructor stub
}





