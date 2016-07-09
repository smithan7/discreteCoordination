/*
 * Graph.cpp
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */


#include "Graph.h"

Point findIntersection(Vec4i w1, Vec4i w2);
float distToLine(Vec4i w, Point a);
Point extendLine(Point a, Point m);
float distToLineSegment(Point p, Point v, Point w);


using namespace cv;
using namespace std;

Graph::Graph(){
	// open descriptor names yml file and store to vector
	FileStorage fsN("/home/andy/Dropbox/workspace/fabmap2Test/masterList.yml", FileStorage::READ);
	fsN["names"] >> this->masterNameList;
	fsN["histogramList"] >> this->masterHistogramList;
	fsN["sequenceList"] >> this->masterSequenceList;
	fsN["pointList"] >> this->masterPointList;
	fsN["centerList"] >> this->masterCenterList;
	fsN["meanLength"] >> this->masterMeanLengthList;
	fsN.release();

	this->minMatchStrength = 100;
	this->wallInflationDistance = 1;

}

void Graph::createGraph(World &gMap, float obsThresh, float comThresh, int gSpace){
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


void Graph::initializeCostMap(){
	for(int i=0; i<this->nRows; i++){
		vector<float> tC;
		for(int j=0; j<this->nCols; j++){
			tC.push_back(100);
		}
		this->costMap.push_back(tC);
	}
}

Mat Graph::createMiniMapImg(){
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




Mat Graph::makeNaiveMatForMiniMap(){
	Mat naiveCostMapOut = Mat::ones(this->nRows, this->nCols, CV_8UC1)*101; // set all cells as unknown

	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] < 51){
				naiveCostMapOut.at<uchar>(i,j,0) = 1;
			}
			else if(this->costMap[i][j] == INFINITY){
				naiveCostMapOut.at<uchar>(i,j,0) = 201;
			}
		}
	}

	for(size_t i=0; i<this->frontiersList.size(); i++){
		int x = this->frontiersList[i][0];
		int y = this->frontiersList[i][1];
		naiveCostMapOut.at<uchar>(x,y,0) = 2;
	}

	return naiveCostMapOut;
}


Mat Graph::makeGlobalInferenceMat(World &gMap){

	// convert gMap costmap to an image
	Mat globalFreeSpaceMat = gMap.createMiniMapImg();
	threshold(globalFreeSpaceMat,globalFreeSpaceMat, 10, 255, THRESH_BINARY_INV);
	// get graph.observedMat
	Mat graphObservedMat = this->getFreeSpaceImage();
	// get global unobserved mat
	Mat globalUnobservedMat;
	bitwise_xor(globalFreeSpaceMat, graphObservedMat, globalUnobservedMat);

	Mat globalInferenceCostMapOut = Mat::ones(this->nRows, this->nCols, CV_8UC1)*201;

	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] < 51){
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 1;
			}
			else if(this->costMap[i][j] == INFINITY){
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 101;
			}
			else if(globalFreeSpaceMat.at<uchar>(i,j,0) == 255){
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 2;
			}
		}
	}
	cerr << "returned global minimat" << endl;
	return globalInferenceCostMapOut;
}

Mat Graph::makeStructuralInferenceMatForMiniMap(){
	vector<Point> outerHull;
	Mat inferCalc = Mat::zeros(this->gSpace*(this->nRows+1), this->gSpace*(this->nCols+1), CV_8UC1);
	Mat outerHullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
	this->getOuterHull(inferCalc, outerHullDrawing, outerHull);

	// project Frontiers
	this->getFrontierOrientation();
	this->getFrontierProjections();

	// find Frontiers likely to exit the outer hull
	vector<int> frontierExits = getFrontierExits(outerHull);

	// find unexplored areas inside the outer hull;
	cerr << "3" << endl;
	bitwise_or(inferCalc, outerHullDrawing, inferCalc);
	vector<vector<Point> > inferenceContours = this->getMinimalInferenceContours(inferCalc);

	cerr << "4" << endl;
	Mat obstaclesAndHull = Mat::zeros(this->nRows, this->nCols, CV_8UC1);
	this->displayInferenceMat(outerHullDrawing, obstaclesAndHull, outerHull, frontierExits);

	cerr << "5" << endl;
	Mat obstaclesTemp= Mat::zeros(this->nRows, this->nCols, CV_8UC1);
	this->structuralBagOfWordsInference(inferenceContours,obstaclesAndHull);

	// use bitwise_and to match the inferred and structural contour
	// use bitwise_xor to subtract out the inferred contour
	// use bitwise_or to add in the the inferred contour, and draw a line from centroid to frontier to ensure they ling for thinning

	// add 1/2 area of total space sized rectangle to all external frontiers
	// redraw hull line to separate external rect and the inner hull
	// draw in frontier to connect the two for thinning

}


Mat Graph::makeVisualInferenceMatForMiniMap(){

}

Mat Graph::makeGeometricInferenceMatForMiniMap(){
	Mat geoInferenceCostMapOut = Mat::ones(this->gSpace*this->nRows, this->gSpace*this->nCols, CV_8UC1)*101; // set all cells as unknown

	vector<Point> outerHull;
	Mat inferCalc = Mat::zeros(this->gSpace*this->nRows, this->gSpace*this->nCols, CV_8UC1);
    Mat outerHullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
    this->getOuterHull(inferCalc, outerHullDrawing, outerHull);

	// find unexplored areas inside the outer hull;
	bitwise_or(inferCalc, outerHullDrawing, inferCalc);

	namedWindow("inferCalc", WINDOW_NORMAL);
	imshow("inferCalc", inferCalc);
	waitKey(1);

	vector<vector<Point> > inferenceContours = this->getMinimalInferenceContours(inferCalc);

	Mat tCont = Mat::zeros(inferCalc.size(), CV_8UC1);
	for(size_t i=0; i<inferenceContours.size(); i++){
		drawContours(tCont, inferenceContours, i, Scalar(255), -1);
	}

	namedWindow("inferCont", WINDOW_NORMAL);
	imshow("inferCont", tCont);
	waitKey(1);

	// find Frontiers likely to exit the outer hull
    // project Frontiers
    this->getFrontierOrientation();
	this->getFrontierProjections();
	vector<int> frontierExits = getFrontierExits(outerHull);
	cerr << "frontierExits: " << frontierExits.size() << endl;
	// add external contours for frontier exits
	vector<vector<Point> > externalContours;
	if(frontierExits.size() > 0){
		this->addExternalContours(outerHull, externalContours, frontierExits);
	}
	// draw everything on the costMap
	Mat inferredMatForMiniMap = Mat::zeros(this->nRows*this->gSpace, this->nCols*this->gSpace, CV_8UC1);
	for(size_t i=0; i<externalContours.size(); i++){
		drawContours(inferredMatForMiniMap, externalContours, i, Scalar(255), -1);
		drawContours(geoInferenceCostMapOut, externalContours, i, Scalar(2), -1); // add inferred free space
	}

	namedWindow("inferCont outer", WINDOW_NORMAL);
	imshow("inferCont outer", inferredMatForMiniMap);
	waitKey(1);

	for(size_t i=0; i<inferenceContours.size(); i++){
		drawContours(inferredMatForMiniMap, inferenceContours, i, Scalar(255), -1);
		drawContours(geoInferenceCostMapOut, inferenceContours, i, Scalar(2), -1); // add inferred free space
	}

	vector<vector<Point> > tHull;
	tHull.push_back(outerHull);
	drawContours(geoInferenceCostMapOut, tHull, 0, Scalar(202), 3); // add inferred walls

	Size f0;
	double val = 1/float(gSpace);
	resize(inferredMatForMiniMap, inferredMatForMiniMap, f0, val, val, INTER_AREA);
	resize(geoInferenceCostMapOut, geoInferenceCostMapOut, f0, val, val, INTER_AREA);

	// my observations are always right
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] < 100){
				geoInferenceCostMapOut.at<uchar>(i,j,0) = 1;
			}
			else if(this->costMap[i][j] == INFINITY){
				geoInferenceCostMapOut.at<uchar>(i,j,0) = 201;
			}
		}
	}

	// frontiers are invisible
	for(size_t i=0; i<this->frontiersList.size(); i++){
		Point f;
		f.x = this->frontiersList[i][0];
		f.y = this->frontiersList[i][1];
		//geoInferenceCostMapOut.at<uchar>(x,y,0) = 1;
		//circle(geoInferenceCostMapOut, f, 1, Scalar(1), 1);
	}

	//this->inflateWalls(geoInferenceCostMapOut);
	return geoInferenceCostMapOut;
}

void Graph::inflateWalls(Mat &costMat){
	for(int i=0; i<costMat.cols; i++){
		for(int j=0; j<costMat.rows; j++){
			if(costMat.at<uchar>(i,j,0) == 2){
				// check all cells within this->wallInflationDistance for either observed wall or inferred wall and make an inflated wall
				for(int k = -this->wallInflationDistance; k<this->wallInflationDistance + 1; k++){
					for(int l = -this->wallInflationDistance; l<this->wallInflationDistance + 1; l++){
						if(k + i < 0 || k + i >= costMat.cols){
							break;
						}
						if(l + j < 0 || l + j >= costMat.rows){
							break;
						}

						if(costMat.at<uchar>(i+k,j+l,0) == 201 || costMat.at<uchar>(i+k,j+l,0) == 202){ // unobserved and nbr is wall
							costMat.at<uchar>(i,j,0) = 203;
						}
					}
				}
			}
		}
	}
}

vector<float> Graph::getInferenceContourRewards(vector<int> frontierExits, vector<vector<Point> > contours){

	vector<float> contourAreas;

	for( size_t i = 0; i < contours.size(); i++ ){
		contourAreas.push_back(contourArea(contours[i]));
		cerr << "contourAreas: " << contourAreas[i] << endl;
	}

    // set value for all Frontiers in each internal contour
	vector<float> contourRewards;
    float maxReward = 0;
    for(size_t i=0; i<contourAreas.size(); i++){
    	Mat temp = Mat::zeros(this->nRows*this->gSpace, this->nCols*this->gSpace,CV_8UC3);
    	Scalar color;
    	color[0] = rand() % 255;
    	color[1] = rand() % 255;
    	color[2] = rand() % 255;
		drawContours(temp,contours,i,color);

    	float members = 0;
    	for(size_t j=0; j<this->frontiers.size(); j++){
    		Point fp;
    		fp.x = this->frontiers[j].projection[1]*this->gSpace;
    		fp.y = this->frontiers[j].projection[0]*this->gSpace;
    		Scalar color2;
    		color2[0] = rand() % 255;
    		color2[1] = rand() % 255;
    		color2[2] = rand() % 255;

    		circle(temp,fp,1,color2);

    		if(pointPolygonTest(contours[i],fp, false) >= 0){
    			members++;
    		}
    		imshow("inferred contours", temp);
    		waitKey(1);
    	}

    	contourRewards.push_back(contourAreas[i] / members);
    	if(contourRewards[i] > maxReward && members > 0){
    		maxReward = contourRewards[i];
    	}
    	cerr << "contourRewards: " << contourRewards[i] << endl;
    }
    waitKey(1);

    // set value for all external Frontiers
    for(size_t i=0;i<frontierExits.size(); i++){
    	contourRewards[frontierExits[i]] = 10*maxReward;
    }

    return contourRewards;
}

void Graph::setFrontierRewards(vector<float> contourRewards, vector<vector<Point> > inferenceContours){
	for(size_t i=0; i<inferenceContours.size(); i++){
		cerr << "reward to be set: " << contourRewards[i] << endl;
		for(size_t j=0; j<this->frontiers.size(); j++){
			Point pf;
			pf.x = this->frontiers[j].projection[0]*this->gSpace+this->gSpace/2;
			pf.y = this->frontiers[j].projection[1]*this->gSpace+this->gSpace/2;
			if(pointPolygonTest(inferenceContours[i],pf,false) >= 0){
				this->frontiers[j].reward = contourRewards[i];
				cerr << "   reward was set" << endl;
			}
		}
	}
}


void Graph::addExternalContours(vector<Point> outerHull, vector<vector<Point> > &externalContours, vector<int> frontierExits){
	// get original outerhull area
	float outerHullArea = contourArea(outerHull);
	for(size_t i=0; i<frontierExits.size(); i++){
		Point fC;
		fC.x = this->frontiers[frontierExits[i]].centroid[1] * this->gSpace;
		fC.y = this->frontiers[frontierExits[i]].centroid[0] * this->gSpace;
		Point fP;
		fC.x = this->frontiers[frontierExits[i]].projection[1] * this->gSpace;
		fC.y = this->frontiers[frontierExits[i]].projection[0] * this->gSpace;

		// find hull wall I intersect
		float minD = INFINITY;
		int index = -1;

		Mat outerHullCalc = Mat::zeros(400, 400, CV_8UC1);

		for(size_t j=1; j<outerHull.size(); j++){
			line(outerHullCalc, outerHull[j], outerHull[j-1], Scalar(127), 1, 8);
			float d = distToLineSegment(fC, outerHull[j], outerHull[j-1]);
			if(d < minD && d >= 0){
				minD = d;
				index = j;
			}
		}
		line(outerHullCalc, outerHull[outerHull.size()-1], outerHull[0], Scalar(127), 1, 8);
		float d = distToLineSegment(fC, outerHull[outerHull.size()-1], outerHull[0]);
		if(d < minD && d >= 0){
			minD = d;
			index = 0;
		}

		// get hull wall points
		Point w0, w1;
		w0.x = outerHull[index].x;
		w0.y = outerHull[index].y;
		if(index != 0){
			w1.x = outerHull[index-1].x;
			w1.y = outerHull[index-1].y;
		}
		else{
			w1.x = outerHull[outerHull.size() - 1].x;
			w1.y = outerHull[outerHull.size() - 1].y;
		}

		cout << "Hull Exits wall: " << w0.x << ", " << w0.y << " and " << w1.x << ", " << w1.y << endl;
		cout << "Frontier Exit Projection: " << fP.x << ", " << fP.y << endl;
		cout << "Frontier Exit Centroid: " << fC.x << ", " << fC.y << endl;

		circle(outerHullCalc, w0, 2, Scalar(127), -1, 8);
		circle(outerHullCalc, w1, 2, Scalar(127), -1, 8);
		circle(outerHullCalc, fC, 5, Scalar(255), -1, 8);
		circle(outerHullCalc, fP, 5, Scalar(255), 1, 8);

		namedWindow("outer hull calc", WINDOW_NORMAL);
		imshow("outer hull calc", outerHullCalc);
		waitKey(1);

		// get depth
		float fArea = outerHullArea;
		float depth = fArea / float(sqrt(pow(w0.x - w1.x,2) + pow(w0.y-w1.y,2)));

		cout << "fArea: " << fArea << endl;
		cout << "depth: " << depth << endl;

		// extend the direction of the projected frontier
		cout << "frontier.orient[0]: " << frontiers[frontierExits[i]].orient[0] << endl;
		cout << "frontier.orient[1]: " << frontiers[frontierExits[i]].orient[1] << endl;

		cout << "hull wall dy / dx: " << w0.y - w1.y << " / " << w0.x - w1.x << endl;

		float wl = sqrt(pow(w0.y-w1.y,2)+pow(w0.x-w1.x,2));
		cout << "hull wall length: " << wl << endl;
		float wly = (w0.x-w1.x)/wl;
		float wlx = (w0.y-w1.y)/wl;

		float fox = frontiers[frontierExits[i]].orient[0];
		float foy = frontiers[frontierExits[i]].orient[1];

		if(pow(wlx-fox,2)+pow(wly-foy,2) > pow(-wlx-fox,2)+pow(-wly-foy,2)){
			wlx *= -1;
			wly *= -1;
		}

		Point p0;
		p0.x = w0.x + depth * wlx;
		if(p0.x >= this->nCols*this->gSpace){ p0.x = (this->nCols-1)*this->gSpace; }
		else if(p0.x < 0){ p0.x = 0; }

		p0.y = w0.y + depth * wly;
		if(p0.y >= this->nRows*this->gSpace){ p0.y = (this->nRows-1)*this->gSpace; }
		else if(p0.y < 0){ p0.y = 0; }

		Point p1;
		p1.x = w1.x + depth * wlx;
		if(p1.x >= this->nCols*this->gSpace){ p1.x = (this->nCols-1)*this->gSpace; }
		else if(p1.x < 0){ p1.x = 0; }

		p1.y = w1.y + depth * wly;
		if(p1.y >= this->nRows*this->gSpace){ p1.y = (this->nRows-1)*this->gSpace; }
		else if(p1.y < 0){ p1.y = 0; }

		cout << "Point p0: " << p0.x << ", " << p0.y << endl;
		cout << "Point p1: " << p1.x << ", " << p1.y << endl;

		// make contour with projected point and hull line endpoints
		vector<Point> externalContour;
		externalContour.push_back(w0);
		externalContour.push_back(p0);
		externalContour.push_back(p1);
		externalContour.push_back(w1);

		externalContours.push_back(externalContour);
	}
}

vector<vector<Point> > Graph::getMinimalInferenceContours(Mat inferenceSpace){
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours(inferenceSpace,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	vector<vector<int> > contourFrontiers;
	// identify contours with Frontiers in them
	for(size_t i=0; i<contours.size(); i++){
		vector<int> m;
		for(size_t j=0; j<this->frontiers.size(); j++){
			Point pf;
			pf.x = this->frontiers[j].projection[1]*this->gSpace;
			pf.y = this->frontiers[j].projection[0]*this->gSpace;
			if(pointPolygonTest(contours[i],pf, false) >= 0){
				m.push_back(j);
			}
		}
		contourFrontiers.push_back(m);
	}

	// determine if children of contour has Frontiers in it
	vector<vector<Point> > contoursToKeep;
	for(size_t i=0; i<contours.size(); i++){
		if(contourFrontiers[i].size() > 0){ // has frontiers in it
			bool keepContour = true;
			int child = hierarchy[i][2];
			while(child >= 0){ // check every contour in child level
				// compare child list against mine, erase from my list Frontiers that my child has
				for(size_t p=0; p<contourFrontiers[i].size(); p++){
					for(size_t c=0; c<contourFrontiers[child].size(); c++){
						if(contourFrontiers[i][p] == contourFrontiers[child][c]){ // matching pair?
							keepContour = false; // dont keep contours without frontiers or that whose children have frontiers
						}
					}
				}
				child = hierarchy[child][0];
			}
			if(keepContour){
				contoursToKeep.push_back(contours[i]);
			}
		}
	}
	return contoursToKeep;
}

void Graph::getOuterHull(Mat &inferCalc, Mat &outerHullDrawing, vector<Point> &outerHull){
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			Point pa, pb;
			pa.x = this->gSpace*j + this->gSpace/2;
			pa.y = this->gSpace*i + this->gSpace/2;
			pb.x = this->gSpace*j - this->gSpace/2;
			pb.y = this->gSpace*i - this->gSpace/2;
			if(this->costMap[i][j] < 51){ // free space and Frontiers
				rectangle(inferCalc,pa,pb,Scalar(255),-1);
			}
			else if(this->costMap[i][j] == INFINITY){ // obstacle
				rectangle(inferCalc,pa,pb,Scalar(255),-1);
			}
		}
	}

	// get points that make up the obstacle
    Mat hullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
	vector<Point> obsPoints;
    obsPoints = this->getImagePoints(inferCalc);

    vector<Point> hull(obsPoints.size() );
   	convexHull( Mat(obsPoints), hull, true, CV_CHAIN_APPROX_NONE);

    /// Draw contours + hull results
    Scalar red = Scalar(255);

    for(int i=0; i<(int)hull.size()-1; i++){
    	line(hullDrawing, hull[i], hull[i+1], red, 1,8);
    }
    line(hullDrawing, hull[0], hull[hull.size()-1], red, 1,8);

    Mat plot = Mat::zeros(inferCalc.size(), CV_8UC1);

    // bitwise and between hull and main image to get outer boundaries, extend to find intersection points
	Mat hullWalls = Mat::zeros( hullDrawing.size(), CV_8UC3 );
	bitwise_and(hullDrawing, inferCalc, hullWalls);

	Mat wallLines = Mat::zeros(inferCalc.size(), CV_8UC1);
	vector<Vec4i> lines;
	HoughLinesP(hullWalls, lines, 1, CV_PI/180, 1  ,3,2);

    for( size_t i = 0; i < lines.size(); i++){
        line( wallLines, Point(lines[i][0], lines[i][1]),Point(lines[i][2], lines[i][3]), Scalar(255), 3, 8 );
    }

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

	// add intersection points
	vector<Point> hullPoints;
    hullPoints = this->getImagePoints(intImage);

    // get new convex hull with projected boundaries
   	convexHull( Mat(hullPoints), outerHull, true, CV_CHAIN_APPROX_NONE);

    /// Draw contours + hull results
    for(int i=0; i<(int)outerHull.size()-1; i++){
    	line(outerHullDrawing, outerHull[i], outerHull[i+1], red, 1,8);
    }
    line(outerHullDrawing, outerHull[0], outerHull[outerHull.size()-1], red, 3,8);
}

void Graph::visualBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull){

}


void Graph::clusteringObstacles(){
	/*
	    // clustering obstacles

	    for(size_t i=0; i<this->frontiers.size(); i++){
	    	 // group obstacles into groups based on touching

	    	vector<vector<int> > oL = this->frontiers[i].obstacles;
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

					for(size_t j=0; j<this->frontiers[i].obstacles.size(); j++){
						if(pow(this->frontiers[i].obstacles[j][0]-pt[0],2) + pow(this->frontiers[i].obstacles[j][1]-pt[1],2) < 2){
							bool flag = true;
							for(size_t k=0; k<cSet.size(); k++){
								if(this->frontiers[i].obstacles[j] == cSet[k]){
									flag = false;
									break;
								}
							}
							if(flag){
								for(size_t k=0; k<oL.size(); k++){
									if(oL[k] == this->frontiers[i].obstacles[j]){
										oL.erase(oL.begin()+k, oL.begin()+k+1);
										break;
									}
								}
								oSet.push_back(this->frontiers[i].obstacles[j]);
							}
						}
					}
				}
				this->frontiers[i].obsClusters.push_back(tCluster);
			}
	    }

	    // draw each obstacle cluster a different cluster
	    for(size_t i=0; i<this->frontiers.size(); i++){
	    	for(size_t j=0; j< this->frontiers[i].obsClusters.size(); j++){
	    		Scalar color;
				color[0] = rand() % 255;
				color[1] = rand() % 255;
				color[2] = rand() % 255;
	    		for(size_t k=0; k<this->frontiers[i].obsClusters[j].size(); k++){
	    			circle(inferDisplay, Point(frnt[i].obsClusters[j][k][1]*this->gSpace,frnt[i].obsClusters[j][k][0]*this->gSpace),2,color,-1,8);
	    		}
	    	}
	    }

	    // get orientation of obstacle clusters
	    // extend obstacles into unobserved space, stop at obstacle or other observed space

	     */
}

void Graph::structuralBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull){

	for( size_t i = 0; i < contours.size(); i++ ){
		double myArea = contourArea(contours[i]);
		bool flag = true;

		if(myArea > 1000){ // not too small
			cout << "Area: " << contourArea(contours[i]) << endl;

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

			float min = INFINITY;
			int mindex = -1;
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

			Mat matchMat = imread( "/home/andy/Dropbox/workspace/fabmap2Test/frontiersArea/training/" + masterNameList[mindex] + ".jpg", CV_LOAD_IMAGE_GRAYSCALE );

			float matchMeanLength = this->masterMeanLengthList[mindex];
			Point matchCenter = masterCenterList[mindex];
			double fxy = meanLength / matchMeanLength;
			Size f0;
			resize(matchMat,matchMat,f0,fxy,fxy, INTER_AREA); // scale to match the partial segment
			//Canny(matchMat, matchMat, 100, 200);

			imshow("match", matchMat);
			waitKey(1);

			Mat contourToInfer = Mat::zeros(this->nRows*this->gSpace, this->nCols*this->gSpace, CV_8UC1);
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

			Mat trial = Mat::zeros(this->nRows*this->gSpace, this->nCols*this->gSpace, CV_8UC1);

			bitwise_and(obstaclesAndHull, contourToInfer, trial);
			imshow("Obstacles near inference contour", trial);
			waitKey(1);

			// perform rotation based ransac over the images
			// in degrees, Mat getRotationMatrix2D(Point2f center, double angle, double scale);
			Mat rotMatrix = getRotationMatrix2D(matchCenter,5,1);
			//void warpAffine(matchMat, OutputArray dst, InputArray M);
			Mat rotatedMatchMat;
			warpAffine(matchMat, rotatedMatchMat, rotMatrix, matchMat.size());

			namedWindow("rotatedMatchMat", WINDOW_NORMAL);
			imshow("rotatedMatchMat", rotatedMatchMat);
			waitKey(1);

			float matchStrength = 0; // metric of how well the matched image aligns

			if(matchStrength >= this->minMatchStrength ){
				// get skeleton Graph of found map



				// connect to existing skeleton Graph in that area



			}
			else{
				// get skeleton Graph of contour



				// connect to existing skeleton Graph in that area



			}


		}
	}
}


void Graph::displayInferenceMat(Mat &outerHullDrawing, Mat &obstaclesAndHull, vector<Point> &outerHull, vector<int> frontierExits){
	Mat inferDisplay = Mat::zeros(this->nRows*this->gSpace, this->nCols*this->gSpace, CV_8UC3);
	Mat Frontiers = this->getFrontiersImage();
	for(size_t i=0; i<this->frontiers.size(); i++){
		circle(Frontiers, Point(this->frontiers[i].projection[1], this->frontiers[i].projection[0]), 1, Scalar(100), -1);
		circle(Frontiers, Point(this->frontiers[i].centroid[1], this->frontiers[i].centroid[0]),    1, Scalar(200), -1);
		line(Frontiers, Point(this->frontiers[i].projection[1], this->frontiers[i].projection[0]), Point(this->frontiers[i].centroid[1], this->frontiers[i].centroid[0]), Scalar(100), 1, 8);
	}

	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy;
	findContours(outerHullDrawing, contours2, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	drawContours(Frontiers, contours2, 0, Scalar(127), 1, 8);

	this->inferredMiniMap = Mat::zeros(this->nCols,this->nRows, CV_8UC1);

	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			Point pa, pb;
			pa.x = this->gSpace*j + this->gSpace/2;
			pa.y = this->gSpace*i + this->gSpace/2;
			pb.x = this->gSpace*j - this->gSpace/2;
			pb.y = this->gSpace*i - this->gSpace/2;
			if(this->costMap[i][j] < 10){ // free space with inflation
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

	for(size_t i=0; i<this->frontiers.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 200;

		Point pa, pb;

		pa.x = this->gSpace*this->frontiers[i].projection[1] + this->gSpace/2;
		pa.y = this->gSpace*this->frontiers[i].projection[0] + this->gSpace/2;

		circle(inferDisplay,pa,5,color,-1);

		pb.x = this->gSpace*this->frontiers[i].centroid[1] + this->gSpace/2;
		pb.y = this->gSpace*this->frontiers[i].centroid[0] + this->gSpace/2;


		int radius = round(sqrt(this->frontiers[i].reward) / 10);
		if(radius < 1){
			radius = 1;
		}
		circle(inferDisplay, pb, radius, color,-1,8);
		color[2] = 255;
		line(inferDisplay,pa,pb,color,5,8);
	}

	for(size_t i=0; i<frontierExits.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 255;
		color[2] = 0;

		Point pa;
		pa.x = this->gSpace*this->frontiers[frontierExits[i]].projection[1] + this->gSpace/2;
		pa.y = this->gSpace*this->frontiers[frontierExits[i]].projection[0] + this->gSpace/2;

		int radius = this->frontiers[i].reward;
		if(radius < 1){
			radius = 1;
		}
		circle(inferDisplay,pa,radius,color,-1);
	}


    for(size_t i=0; i<outerHull.size()-1; i++){
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

    // for each Frontier

    for(size_t i=0; i<this->frontiers.size(); i++){
    	// find all obstacle points within distance r of each Frontier member
    	for(size_t j=0; j<this->frontiers[i].members.size(); j++){

    		for(int k = -2; k<3; k++){
    			for(int l=-2; l<3; l++){

    				if(this->costMap[this->frontiers[i].members[j][0] + k][this->frontiers[i].members[j][1] + l] > 255){
    					Scalar color;
						color[0] = 0;
						color[1] = 255;
						color[2] = 0;
    					vector<int> t;
    					t.push_back(this->frontiers[i].members[j][0] + k);
    					t.push_back(this->frontiers[i].members[j][1] + l);

    					Point pa;
    					pa.x = t[1]*this->gSpace;
    					pa.y = t[0]*this->gSpace;

    					this->frontiers[i].obstacles.push_back(t);
    					circle(inferDisplay,pa,2,color,-1);
    				}
    			}
    		}
    	}
    }

    imshow( "infer miniMap", this->inferredMiniMap );
    imshow( "infer display", inferDisplay );
}

void Graph::getLengthHistogram(vector<float> length, float meanLength, vector<int> &histogram, vector<float> &sequence){
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


void Graph::extractInferenceContour(){
	Mat temp;
	bitwise_or(this->obstacleMat, this->inferenceMat, temp);
	bitwise_or(temp, this->freeMat, temp);
	bitwise_not(temp,temp);

	imshow("q", temp);
}

vector<int> Graph::getFrontierExits(vector<Point> &outerHull){
	vector<int> frontierExits;
	for(size_t i=0; i<this->frontiers.size(); i++){
		Point pf;
		pf.x = this->frontiers[i].projection[1]*this->gSpace + this->gSpace/2;
		pf.y = this->frontiers[i].projection[0]*this->gSpace + this->gSpace/2;
		double t = pointPolygonTest(outerHull,pf, false); // +1 means inside
		if(t <= 0){
			frontierExits.push_back(i);
		}
	}
	return frontierExits;
}

void Graph::getFrontierProjections(){
	for(size_t i=0; i<this->frontiers.size(); i++){
		this->frontiers[i].getFrontierProjection();
    }
}

void Graph::getFrontierOrientation(){
	for(int i=0; i<(int)this->frontiers.size(); i++){
		vector<float> temp;
		temp.push_back(0);
		temp.push_back(0);
		this->frontiers[i].orient = temp;
		float count = 0;
		// check each member of each cluster
		if(this->frontiers[i].members.size() > 1){
			for(int j=0; j<(int)this->frontiers[i].members.size(); j++){
				// check 4Nbr for being unobserved
				int xP = this->frontiers[i].members[j][0];
				int yP = this->frontiers[i].members[j][1];
				if(this->costMap[xP+1][yP] == 100){
					this->frontiers[i].orient[0] += 1;
					count++;
				}
				if(this->costMap[xP-1][yP] == 100){
					this->frontiers[i].orient[0] -= 1;
					count++;
				}
				if(this->costMap[xP][yP+1] == 100){
					this->frontiers[i].orient[1] += 1;
					count++;
				}
				if(this->costMap[xP][yP-1] == 100){
					this->frontiers[i].orient[1] -= 1;
					count++;
				}
			}
			if(count > 0){
				this->frontiers[i].orient[0] /= count;
				this->frontiers[i].orient[1] /= count;
			}
		}
	}
}


vector<Point> Graph::getImagePoints(Mat &image){
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

Mat Graph::getObstaclesImage(){
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

Mat Graph::getFreeSpaceImage(){
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

Mat Graph::getFrontiersImage(){
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
	//imshow("Frontiers map", temp);
	return temp;
}

void Graph::shareMap(vector<vector<float> >& in){
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if((this->costMap[i][j] == 100 || this->costMap[i][j] == 50) && in[i][j] > 100){ // I dont think its observed, they do
				this->costMap[i][j] = in[i][j];
			}
		}
	}
}

int Graph::getMaxIndex(vector<float> value){
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

int Graph::getMinIndex(vector<float> value){
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

void Graph::getFrontierCosts(World &gMap, vector<Agent> &bot){
	for(size_t i=0; i<bot.size(); i++){
		bot[i].fCost.clear();
		for(size_t j=0; j<this->frontiers.size(); j++){
			bot[i].fCost[j] = this->aStarDist(bot[i].cLoc, this->frontiers[j].centroid, gMap);
		}
	}
}

void Graph::findFrontiers(){
	this->frontiersList.clear();
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
				frontiersList.push_back(fT);
				this->frntsExist = true;
				this->costMap[i][j] = 50;
			}
		}
	}
}

void Graph::observe(vector<int> cLoc, World &gMap){

	for(int i=0; i<gMap.obsGraph[cLoc[0]][cLoc[1]].size(); i++){
		int a = gMap.obsGraph[cLoc[0]][cLoc[1]][i][0];
		int b = gMap.obsGraph[cLoc[0]][cLoc[1]][i][1];
		this->costMap[a][b] = gMap.costMap[a][b];
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

void Graph::showCostMapPlot(int index){
	char buffer[50];
	int n = sprintf(buffer, "CostMap %d",index);
	imshow(buffer, this->tempImage);
	waitKey(1);
}

void Graph::addAgentToPlot(World &gMap, Agent &bot){
	Scalar color;
	color[0] = bot.myColor[0];
	color[1] = bot.myColor[1];
	color[2] = bot.myColor[2];

	circle(this->tempImage,gMap.pointMap[bot.cLoc[0]][bot.cLoc[1]],gMap.gSpace,color,-1);
	for(size_t i=1; i<bot.myPath.size(); i++){
		Point a = gMap.pointMap[bot.myPath[i][0]][bot.myPath[i][1]];
		Point b = gMap.pointMap[bot.myPath[i-1][0]][bot.myPath[i-1][1]];
		line(this->tempImage,a,b,color,2);
	}
}

void Graph::buildCostMapPlot(World &gMap){
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
	for(int i=0; i<(int)this->frontiersList.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 127;

		Point pa, pb;
		pa.x = gMap.pointMap[this->frontiersList[i][0]][this->frontiersList[i][1]].x + gMap.gSpace/2;
		pa.y = gMap.pointMap[this->frontiersList[i][0]][this->frontiersList[i][1]].y + gMap.gSpace/2;
		pb.x = gMap.pointMap[this->frontiersList[i][0]][this->frontiersList[i][1]].x - gMap.gSpace/2;
		pb.y = gMap.pointMap[this->frontiersList[i][0]][this->frontiersList[i][1]].y - gMap.gSpace/2;

		rectangle(this->tempImage,pa,pb,color,-1);
	}

	for(int i=0; i<this->frontiers.size(); i++){
		Scalar color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 255;

		Point pa, pb;
		pa.x = gMap.pointMap[this->frontiers[i].centroid[0]][this->frontiers[i].centroid[1]].x + gMap.gSpace/2;
		pa.y = gMap.pointMap[this->frontiers[i].centroid[0]][this->frontiers[i].centroid[1]].y + gMap.gSpace/2;
		pb.x = gMap.pointMap[this->frontiers[i].centroid[0]][this->frontiers[i].centroid[1]].x - gMap.gSpace/2;
		pb.y = gMap.pointMap[this->frontiers[i].centroid[0]][this->frontiers[i].centroid[1]].y - gMap.gSpace/2;

		rectangle(this->tempImage,pa,pb,color,-1);

		pa.x = gMap.pointMap[this->frontiers[i].centroid[0]][this->frontiers[i].centroid[1]].x;
		pa.y = gMap.pointMap[this->frontiers[i].centroid[0]][this->frontiers[i].centroid[1]].y;
		pb.x = gMap.pointMap[this->frontiers[i].projection[0]][this->frontiers[i].projection[1]].x;
		pb.y = gMap.pointMap[this->frontiers[i].projection[0]][this->frontiers[i].projection[1]].y;

		line(this->tempImage,pa,pb,color,1);
	}
}

vector<vector<int> > Graph::aStarPath(vector<int> sLoc, vector<int> gLoc, World &gMap){
	float costMapPenalty;

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

	if(this->costMap[sLoc[0]][sLoc[1]] < 101){
		costMapPenalty = 0;//this->costMap[sLoc[0]][sLoc[1]];
	}
	else{
		costMapPenalty = INFINITY;
	}
	fScore[sLoc[0]][sLoc[1]] = gScore[sLoc[0]][sLoc[1]] + gMap.getEuclidDist(sLoc[0],sLoc[1],gLoc[0],gLoc[1]) + costMapPenalty;

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
			if(nbrRow >= 0 && nbrRow < this->nRows){
				for(int nbrCol=cLoc[1]-1; nbrCol<cLoc[1]+2; nbrCol++){
					if(nbrCol >= 0 && nbrCol < this->nRows){
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
						if(this->costMap[nbrRow][nbrCol] < 101){
							costMapPenalty = 0;//this->costMap[nbrRow][nbrCol];
						}
						else{
							costMapPenalty = INFINITY;
						}
						fScore[nbrRow][nbrCol] = gScore[nbrRow][nbrCol] + gMap.getEuclidDist(gLoc[0],gLoc[1],nbrRow,nbrCol) + costMapPenalty;
					}
				}
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

void Graph::findNearestFrontier(Agent &bot, World &gMap){
	vector<int> index;
	float minDist = INFINITY;
	for(size_t i=0; i<this->frontiers.size(); i++){
		float t = this->aStarDist(bot.cLoc, this->frontiers[i].centroid, gMap);
		if(minDist > t){
			minDist = t;
			index = this->frontiers[i].centroid;
		}
	}
	bot.gLoc = index;
}

float Graph::aStarDist(vector<int> sLoc, vector<int> gLoc, World &gMap){
	float costMapPenalty = 0;
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
	if(this->costMap[sLoc[0]][sLoc[1]] < 3){
		costMapPenalty = this->costMap[sLoc[0]][sLoc[1]];
	}
	else{
		costMapPenalty = INFINITY;
	}
	fScore[sLoc[0]][sLoc[1]] = gScore[sLoc[0]][sLoc[1]] + gMap.getEuclidDist(sLoc[0],sLoc[1],gLoc[0],gLoc[1]) + costMapPenalty;

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
				if(this->costMap[nbrRow][nbrCol] < 3){
					costMapPenalty = this->costMap[nbrRow][nbrCol];
				}
				else{
					costMapPenalty = INFINITY;
				}
				fScore[nbrRow][nbrCol] = gScore[nbrRow][nbrCol] + gMap.getEuclidDist(gLoc[0],gLoc[1],nbrRow,nbrCol) + costMapPenalty;
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

void Graph::clusterFrontiers(World &gMap){

	cout << "this->frontiers in: " << this->frontiers.size() << endl;
	for(int i=0; i<this->frontiers.size(); i++){
		cout << "   " << this->frontiers[i].centroid[0] << " / " << this->frontiers[i].centroid[1] << endl;
	}

	vector<vector<int> > qFrnts = this->frontiersList;

	// check to see if frnt.centroid is still a Frontier cell, if so keep, else delete
	for(size_t i=0; i<this->frontiers.size(); i++){
		this->frontiers[i].editFlag = true;
		bool flag = true;
		for(int j=0; j<(int)qFrnts.size(); j++){
			if(this->frontiers[i].centroid == qFrnts[j]){
				flag = false;
				qFrnts.erase(qFrnts.begin()+j);
			}
		}
		if(flag){
			this->frontiers.erase(this->frontiers.begin()+i);
		}
		else{
			this->frontiers[i].editFlag = false;
		}
	}
	// breadth first search through known clusters
	for(size_t i=0; i<this->frontiers.size(); i++){ // keep checking for new Frontier clusters while there are unclaimed Frontiers
		vector<vector<int> > q; // current cluster
		vector<vector<int> > qP; // open set in cluster
		qP.push_back(this->frontiers[i].centroid);

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
		this->frontiers[i].members = q; // save to list of clusters
	}

	// breadth first search
	while((int)qFrnts.size() > 0){ // keep checking for new Frontier clusters while there are unclaimed Frontiers
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
		Frontier a(gMap.nRows, gMap.nCols);
		this->frontiers.push_back(a);
		this->frontiers[this->frontiers.size()-1].members = q; // save to list of clusters
	}

	for(size_t i=0; i<this->frontiers.size(); i++){ // number of clusters
		if(this->frontiers[i].editFlag){
			float minDist = INFINITY;
			int minDex;
			for(size_t j=0; j<this->frontiers[i].members.size(); j++){ // go through each cluster member
				float tempDist = 0;
				for(size_t k=0; k<this->frontiers[i].members.size(); k++){ // and get cumulative distance to all other members
					tempDist += gMap.getEuclidDist(this->frontiers[i].members[j][0],this->frontiers[i].members[j][1],this->frontiers[i].members[k][0],this->frontiers[i].members[k][1]);
				}
				if(tempDist < minDist){
					minDist = tempDist;
					minDex = j;
				}
			}
			this->frontiers[i].centroid = this->frontiers[i].members[minDex];
		}
	}
}

float distToLineSegment(Point p, Point v, Point w){
	float l = pow(v.x - w.x,2) + pow(v.y-w.y,2);
	if(l==0){ return sqrt(pow(v.x - p.x,2) + pow(v.y-p.y,2) ); } // v==w
	float t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l;
	if(t > 1){
		t = 1;
	}
	else if(t < 0){
		t = 0;
	}
	int xl = v.x + t * (w.x - v.x);
	int yl = v.y + t * (w.y - v.y);
	return sqrt( pow(p.x - xl,2) + pow(p.y-yl,2) );
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

float Graph::dStarDist(int gIndex, Agent bot, World &gMap){
	// is bot in the cSet?
	if(this->frontiers[gIndex].cSet[bot.cLoc[0]][bot.cLoc[1]] == 1){
		return this->frontiers[gIndex].gScore[bot.cLoc[0]][bot.cLoc[1]];
	}
	// do I need to seed the openSet?
	if(this->frontiers[gIndex].oSet[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] == 0){
		this->frontiers[gIndex].oSet[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] = 1;
		this->frontiers[gIndex].gScore[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] = 0;
		this->frontiers[gIndex].fScore[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] = this->frontiers[gIndex].gScore[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] + gMap.getEuclidDist(this->frontiers[gIndex].centroid[0],this->frontiers[gIndex].centroid[1],bot.cLoc[0],bot.cLoc[1]);
	}
	int foo = 1;
	int finishFlag = 0;
	while(finishFlag == 0 ){

		// am I at the goal?
		if(this->frontiers[gIndex].cSet[bot.cLoc[0]][bot.cLoc[1]] == 1){
			cout << "   cLoc in cSet, finished" << endl;
			return this->frontiers[gIndex].gScore[bot.cLoc[0]][bot.cLoc[1]];
		} // end at the goal

		/////////////////// this finds oSet node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> tLoc;
		tLoc.push_back(-1);
		tLoc.push_back(-1);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(this->frontiers[gIndex].oSet[i][j] > 0 && this->frontiers[gIndex].fScore[i][j] < min){
					min = this->frontiers[gIndex].fScore[i][j];
					tLoc[0] = i;
					tLoc[1] = j;
				}
			}
		}

		if(tLoc[0] == tLoc[1] && tLoc[0] == -1){
			return(INFINITY);
		}
		/////////////////////// end finding current node

		this->frontiers[gIndex].oSet[tLoc[0]][tLoc[1]] = 0; // take out of openset
		this->frontiers[gIndex].cSet[tLoc[0]][tLoc[1]] = 1; // add to closed set

		for(int nbrRow=tLoc[0]-1;nbrRow<tLoc[0]+2;nbrRow++){
			if(nbrRow > 0 && nbrRow < gMap.nRows){ // make sure it's on the map

				for(int nbrCol=tLoc[1]-1; nbrCol<tLoc[1]+2; nbrCol++){
					if(nbrCol > 0 && nbrCol < gMap.nCols){ // make sure its on the map

						if(this->costMap[nbrRow][nbrCol] < 51){ // make sure its observed / or Frontier before adding to oSet

							if(this->frontiers[gIndex].cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
								continue;
							}
							float tGScore;
							tGScore = this->frontiers[gIndex].gScore[tLoc[0]][tLoc[1]] + gMap.getEuclidDist(tLoc[0],tLoc[1],nbrRow,nbrCol); // calc temporary gscore
							if(this->frontiers[gIndex].oSet[nbrRow][nbrCol] == 0){
								this->frontiers[gIndex].oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
							}
							else if(tGScore >= this->frontiers[gIndex].gScore[nbrRow][nbrCol]){ // is temp gscore worse than stored g score of nbr? don't update
								continue;
							}
							this->frontiers[gIndex].cameFrom[nbrRow][nbrCol] = tLoc;
							this->frontiers[gIndex].gScore[nbrRow][nbrCol] = tGScore;
							this->frontiers[gIndex].fScore[nbrRow][nbrCol] = tGScore + gMap.getEuclidDist(bot.cLoc[0],bot.cLoc[1],nbrRow,nbrCol) + this->costMap[nbrRow][nbrCol];
						}
					}
				}
			}
		}
	}
	return INFINITY;
}

vector<vector<int> > Graph::dStarPath(int gIndex, Agent &bot, World &gMap){
	// is bot in the cSet?
	if(this->frontiers[gIndex].cSet[bot.cLoc[0]][bot.cLoc[1]] == 100){
		vector<vector<int> > totalPath;
		totalPath.push_back(bot.cLoc);
		vector<int> temp = bot.cLoc;
		bool finish = false;
		while(!finish){ // work backwards to start
			if(temp == this->frontiers[gIndex].centroid){
				finish =true;
				totalPath.push_back(temp); // append path
			}
			temp = this->frontiers[gIndex].cameFrom[temp[0]][temp[1]]; // work backwards
			totalPath.push_back(temp); // append path
		}
		return totalPath;
	}

	// do I need to seed the openSet?
	if(this->frontiers[gIndex].oSet[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] == 0){
		this->frontiers[gIndex].oSet[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] = 1;
		this->frontiers[gIndex].gScore[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] = 0;
		this->frontiers[gIndex].fScore[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] = this->frontiers[gIndex].gScore[this->frontiers[gIndex].centroid[0]][this->frontiers[gIndex].centroid[1]] + gMap.getEuclidDist(this->frontiers[gIndex].centroid[0],this->frontiers[gIndex].centroid[1],bot.cLoc[0],bot.cLoc[1]);
	}

	int foo = 1;
	int finishFlag = 0;

	while(foo>0 && finishFlag == 0){

		// am I at the goal?
		if(this->frontiers[gIndex].cSet[bot.cLoc[0]][bot.cLoc[1]] == 1){
			vector<vector<int> > totalPath;
			totalPath.push_back(bot.cLoc);
			vector<int> temp = bot.cLoc;
			while(temp[0] >= 0){ // work backwards to start
				if(temp == this->frontiers[gIndex].centroid){
					totalPath.push_back(temp); // append path
				}
				temp = this->frontiers[gIndex].cameFrom[temp[0]][temp[1]]; // work backwards
				totalPath.push_back(temp); // append path
			}
			return totalPath;
		} // end am I at the goal


		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> tLoc;
		tLoc.push_back(0);
		tLoc.push_back(0);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(this->frontiers[gIndex].oSet[i][j] > 0 && this->frontiers[gIndex].fScore[i][j] < min){
					min = this->frontiers[gIndex].fScore[i][j];
					tLoc[0] = i;
					tLoc[1] = j;
				}
			}
		}
		//cout << "tLoc: " << tLoc[0] << "<" << tLoc[1] << endl;
		/////////////////////// end finding current node

		this->frontiers[gIndex].oSet[tLoc[0]][tLoc[1]] = 0; // take out of openset
		this->frontiers[gIndex].cSet[tLoc[0]][tLoc[1]] = 1; // add to closed set

		for(int nbrRow=tLoc[0]-1;nbrRow<tLoc[0]+2;nbrRow++){
			if(nbrRow > 0 && nbrRow < gMap.nRows){ // make sure it's on the map

				for(int nbrCol=tLoc[1]-1; nbrCol<tLoc[1]+2; nbrCol++){
					if(nbrCol > 0 && nbrCol < gMap.nCols){ // make sure its on the map

						if(this->costMap[nbrRow][nbrCol] < 51){ // make sure its observed / or Frontier before adding to oSet

							if(this->frontiers[gIndex].cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
								continue;
							}
							float tGScore;
							tGScore = this->frontiers[gIndex].gScore[tLoc[0]][tLoc[1]] + gMap.getEuclidDist(tLoc[0],tLoc[1],nbrRow,nbrCol); // calc temporary gscore
							if(this->frontiers[gIndex].oSet[nbrRow][nbrCol] == 0){
								this->frontiers[gIndex].oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
							}
							else if(tGScore >= this->frontiers[gIndex].gScore[nbrRow][nbrCol]){ // is temp gscore worse than stored g score of nbr? don't update
								continue;
							}
							this->frontiers[gIndex].cameFrom[nbrRow][nbrCol] = tLoc;
							this->frontiers[gIndex].gScore[nbrRow][nbrCol] = tGScore;
							this->frontiers[gIndex].fScore[nbrRow][nbrCol] = tGScore + gMap.getEuclidDist(bot.cLoc[0],bot.cLoc[1],nbrRow,nbrCol) + this->costMap[nbrRow][nbrCol];
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


 Mat Graph::createMiniMapInferImg(){
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

/*

Mat Graph::makeGlobalInferenceMat(World &gMap){
	// find contours in global unobserved
	vector<vector<Point> > globalContours;
	vector<Vec4i> globalHierarchy;
	vector<vector<int> > globalContourMembers;
	vector<float> globalContourArea;
	vector<float> globalContourReward;

	findContours(globalUnobservedMat, globalContours, globalHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	Mat cont = Mat::zeros( globalUnobservedMat.size(), CV_8UC3);
	for(size_t i=0; i<globalContours.size(); i++){
		Scalar color;
		color[0] = rand() % 255;
		color[1] = rand() % 255;
		color[2] = rand() % 255;
		drawContours(cont, globalContours, i, color,1,8);
	}

	// identify contours with Frontiers in them
	for(size_t i=0; i<globalContours.size(); i++){
		Mat temp = Mat::zeros(cont.size(), CV_8UC3);
		Scalar color;
		color[0] = rand() % 255;
		color[1] = rand() % 255;
		color[2] = rand() % 255;

		vector<int> m;
		for(size_t j=0; j<this->frontiers.size(); j++){
			Point pf;
			pf.x = this->frontiers[j].projection[1];
			pf.y = this->frontiers[j].projection[0];

			drawContours(temp,globalContours,i,color);
			Scalar color2;
			color2[0] = rand() % 255;
			color2[1] = rand() % 255;
			color2[2] = rand() % 255;
			circle(temp,pf,3,color,-1);

			if(pointPolygonTest(globalContours[i],pf, false) >= 0){
				m.push_back(j);
				cerr << "global: in contour" << endl;
			}
			else{
				cerr << "global: not in contour" << endl;
			}
			imshow("temp", temp);
			waitKey(1);
		}
		globalContourMembers.push_back(m);
	}

	// determine if children of contour has Frontiers in it
	for(size_t i=0; i<globalContourMembers.size(); i++){
		int child = globalHierarchy[i][2];
		while(child >= 0){ // check every contour in child level
			// compare child list against mine, erase from my list Frontiers that my child has
			for(size_t p=0; p<globalContourMembers[i].size(); p++){
				for(size_t c=0; c<globalContourMembers[child].size(); c++){
					if(globalContourMembers[i][p] == globalContourMembers[child][c]){ // matching pair?
						globalContourMembers[i].erase(globalContourMembers[i].begin()+p,globalContourMembers[i].begin()+p+1);
						//globalContours[i].erase(globalContours[i].begin()+p, globalContours[i].begin()+p+1);
					}
				}
			}
			child = globalHierarchy[child][0];
		}
	}

	cout << "check 2" << endl;

	// get value of all frontiers
	for(size_t i=0; i<globalContours.size(); i++){
		globalContourReward.push_back(contourArea(globalContours[i]));
	}

	cout << "globalFrontier Rewards" << endl;
	// get value of all frontiers
	for(size_t i=0; i<globalContours.size(); i++){
		cout << globalContourReward[i] << endl;
	}
}
*/

/*
vector< vector<int> > Graph::kMeansClusteringEuclid(vector<int> openFrnt, int numClusters){
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
		for(int i=0; i<numClusters; i++){ // compute centroid Frontier of each cluster
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
		} // end compute centroid Frontier of each cluster

		for(int i=0; i<int(openFrnt.size()); i++){ // find A* dist from each Frontier to each cluster and group Frontier in closest cluster
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

vector< vector<int> > Graph::kMeansClusteringTravel(vector<int> openFrnt, int numClusters){
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
		for(int i=0; i<numClusters; i++){ // compute centroid Frontier of each cluster
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
		} // end compute centroid Frontier of each cluster

		for(int i=0; i<int(openFrnt.size()); i++){ // find A* dist from each Frontier to each cluster and group Frontier in closest cluster
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


void Graph::centralMarket(World &gMap, vector<Agent> &bots){

	cerr << "into get Frontier costs: " << endl;
	this->getFrontierCosts(gMap, bots);
	cerr << "out of frotier costs" << endl;

	cout << "Frontier costs and Rewards: " << endl;
	for(size_t i=0; i<this->frontiers.size(); i++){
		for(size_t j=0; j<bots.size(); j++){
			cout << "   cost: " << bots[j].fCost[i] << endl;;
			cout << "   reward: " << this->frontiers[i].reward << endl;
		}
	}
	waitKey(1);

	if(bots.size() <= this->frontiers.size()){ // more Frontiers than Agents
		cout << "more Frontiers than Agents" << endl;
		vector<vector<float> > fValueList; // list of Frontier values for all Agents, [Agent][Frontier]
		for(int i=0; i<bots.size(); i++){
			vector<float> cVal;
			for(int j=0; j<this->frontiers.size(); j++){
				cVal.push_back( this->frontiers[i].reward - bots[i].fCost[j] );
			}
			fValueList.push_back( cVal );
		}

		cout << "fValueList: " << endl;
		for(int i=0; i<bots.size(); i++){
			cout << "   ";
			for(int j=0; j<this->frontiers.size(); j++){
				cout << fValueList[i][j] << " , ";
			}
			cout << endl;
		}

		bool fin = false;
		vector<int> maxDex;
		vector<float> maxVal;

		while(!fin){
			maxDex.erase(maxDex.begin(), maxDex.end());
			maxVal.erase(maxVal.begin(), maxVal.end());
			fin = true;

			for(int i=0; i<bots.size(); i++){ // get each Agents best Frontier
				maxDex.push_back( -1 );
				maxVal.push_back( -INFINITY );

				for(int j=0; j<(int)fValueList[i].size(); j++){
					if(fValueList[i][j] > maxVal[i]){
						maxDex[i] = j; // Agent's max value is Frontier j
						maxVal[i] = fValueList[i][j];
					}
				}
			}

			// make sure no one shares the same Frontier
			for(int i=0; i<bots.size(); i++){
				for(int j=i+1; j<bots.size(); j++){
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
		for(int i=0; i<bots.size(); i++){
			bots[i].gLoc = this->frontiers[maxDex[i]].centroid;
			bots[i].gIndex = maxDex[i];
		}
	}
	else{ // more Agents than Frontiers
		cout << "more Agents than Frontiers" << endl;
		for(int i=0; i<this->frontiers.size(); i++){ // go through all Frontiers and find the best Agent
			float mV = INFINITY;
			int mI;
			for(int j=0; j<bots.size(); j++){
				if(bots[j].fCost[i] < mV){
					mV = bots[j].fCost[i];
					mI = j;
				}
			}
			bots[mI].gLoc = this->frontiers[i].centroid;
			bots[mI].gIndex = i;
			for(int j = 0; j<(int)this->frontiers.size(); j++){ // erase all of the value for the worst Agent
				bots[mI].fCost[j] = INFINITY;
			}
		}
	}
}
*/

Graph::~Graph() {
	// TODO Auto-generated destructor stub
}




