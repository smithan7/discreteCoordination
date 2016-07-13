/*
 * Inference.cpp
 *
 *  Created on: Jul 12, 2016
 *      Author: andy
 */

#include "Inference.h"

Point findIntersection(Vec4i w1, Vec4i w2);
float distToLine(Vec4i w, Point a);
Point extendLine(Point a, Point m);
float distToLineSegment(Point p, Point v, Point w);

Inference::Inference() {
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

	this->obsFree = 1;
	this->infFree = 2;
	this->unknown = 101;
	this->obsWall = 201;
	this->infWall = 202;
	this->inflatedWall = 203;
}

Inference::~Inference(){}

// used to convert the inference image back into a costMap for graph to use
vector<vector<int> > Inference::matToMap(Mat &costMat){
	vector<vector<int> > costMap;

	for(int i=0; i<costMat.cols; i++){
		vector<int> costCol;
		for(int j=0; j<costMat.rows; j++){
			costCol.push_back(costMat.at<uchar>(i,j));
		}
		costMat.push_back(costCol);
	}
	return costMap;
}

Mat Inference::mapToMat(vector<vector<int> > &costMap){
	Mat costMat = Mat::ones(costMap.size(), costMap[0].size(), CV_8UC1)*101;

	for(size_t i=0; i<costMap.size(); i++){
		for(size_t j=0; j<costMap[0].size(); j++){
			costMat.at<uchar>(i,j) = costMap[i][j];
		}
	}
	return costMat;
}

void Inference::importCostMap(vector<vector<int> > costMap){
	this->costMap = costMap;
	this->costMat = mapToMat(costMap);
	this->nRows = costMap[0].size();
	this->nCols = costMap.size();
}


Mat Inference::makeNaiveMatForMiniMap(){
	Mat naiveCostMapOut = Mat::ones(this->costMat.size(), CV_8UC1)*101; // set all cells as unknown
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			naiveCostMapOut.at<uchar>(i,j,0) = this->costMap[i][j];
		}
	}
	for(size_t i=0; i<this->frontiers.size(); i++){
		for(size_t j=0; j<this->frontiers[i].members.size(); j++){
			int x = this->frontiers[i].members[j][0];
			int y = this->frontiers[i].members[j][1];
			naiveCostMapOut.at<uchar>(x,y,0) = this->infFree;

		}
	}
	return naiveCostMapOut;
}

Mat Inference::makeGlobalInferenceMat(World &gMap){
	Mat globalInferenceCostMapOut = Mat::ones(this->nRows, this->nCols, CV_8UC1)*101;

	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(gMap.costMap[i][j] == 1){
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 2;
			}
			else{
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 202;
			}
			if(this->costMap[i][j] < 100){
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 1;
			}
			else if(this->costMap[i][j] >200){
				globalInferenceCostMapOut.at<uchar>(i,j,0) = 201;
			}
		}
	}
	return globalInferenceCostMapOut;
}

Mat Inference::makeStructuralInferenceMatForMiniMap(){
	Mat structuralInferenceMat = Mat::ones(this->costMat.size(), CV_8UC1)*101;

	vector<Point> outerHull;
	Mat inferCalc = Mat::zeros(this->costMat.size(), CV_8UC1);
	Mat outerHullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
	this->getOuterHull(inferCalc, outerHullDrawing, outerHull);

	// find Frontiers likely to exit the outer hull
	vector<int> frontierExits = getFrontierExits(outerHull);

	// find unexplored areas inside the outer hull;
 	bitwise_or(inferCalc, outerHullDrawing, inferCalc);
	vector<vector<Point> > inferenceContours = this->getMinimalInferenceContours(inferCalc);

	Mat obstaclesAndHull = Mat::zeros(this->nRows, this->nCols, CV_8UC1);
	this->displayInferenceMat(outerHullDrawing, obstaclesAndHull, outerHull, frontierExits);

	Mat obstaclesTemp= Mat::zeros(this->nRows, this->nCols, CV_8UC1);
	this->structuralBagOfWordsInference(inferenceContours,obstaclesAndHull);

	// use bitwise_and to match the inferred and structural contour
	// use bitwise_xor to subtract out the inferred contour
	// use bitwise_or to add in the the inferred contour, and draw a line from centroid to frontier to ensure they ling for thinning

	// add 1/2 area of total space sized rectangle to all external frontiers
	// redraw hull line to separate external rect and the inner hull
	// draw in frontier to connect the two for thinning
	return structuralInferenceMat;
}


Mat Inference::makeVisualInferenceMatForMiniMap(){
	Mat visualInferenceMat = Mat::ones(this->costMat.size(), CV_8UC1)*101;

	return visualInferenceMat;
}

Mat Inference::makeGeometricInferenceMatForMiniMap(){
	Mat geoInferenceCostMapOut = Mat::ones(this->costMat.size(), CV_8UC1)*101; // set all cells as unknown

	vector<Point> outerHull;
	Mat inferCalc = Mat::zeros(this->costMat.size(), CV_8UC1);
    Mat outerHullDrawing = Mat::zeros(inferCalc.size(), CV_8UC1);
    this->getOuterHull(inferCalc, outerHullDrawing, outerHull);

	// find unexplored areas inside the outer hull;
	bitwise_or(inferCalc, outerHullDrawing, inferCalc);

	namedWindow("graph::geoInfer::calc", WINDOW_NORMAL);
	imshow("graph::geoInfer::calc", inferCalc);
	waitKey(1);

	vector<vector<Point> > inferenceContours = this->getMinimalInferenceContours(inferCalc);

	Mat tCont = Mat::zeros(inferCalc.size(), CV_8UC1);
	for(size_t i=0; i<inferenceContours.size(); i++){
		drawContours(tCont, inferenceContours, i, Scalar(255), -1);
	}

	namedWindow("graph::geoInfer::contour", WINDOW_NORMAL);
	imshow("graph::geoInfer::contour", tCont);
	waitKey(1);

	// find Frontiers likely to exit the outer hull
	vector<int> frontierExits = getFrontierExits(outerHull);
	cerr << "frontierExits: " << frontierExits.size() << endl;
	// add external contours for frontier exits
	vector<vector<Point> > externalContours;
	if(frontierExits.size() > 0){
		this->addExternalContours(outerHull, externalContours, frontierExits);
	}
	// draw everything on the costMap
	Mat inferredMatForMiniMap = Mat::zeros(this->costMat.size(), CV_8UC1);
	for(size_t i=0; i<externalContours.size(); i++){
		drawContours(inferredMatForMiniMap, externalContours, i, Scalar(255), -1);
		drawContours(geoInferenceCostMapOut, externalContours, i, Scalar(2), -1); // add inferred free space
	}

	namedWindow("graph::geoInfer::outer", WINDOW_NORMAL);
	imshow("graph::geoInfer::outer", inferredMatForMiniMap);
	waitKey(1);

	for(size_t i=0; i<inferenceContours.size(); i++){
		drawContours(inferredMatForMiniMap, inferenceContours, i, Scalar(255), -1);
		drawContours(geoInferenceCostMapOut, inferenceContours, i, Scalar(2), -1); // add inferred free space
	}

	vector<vector<Point> > tHull;
	tHull.push_back(outerHull);
	drawContours(geoInferenceCostMapOut, tHull, 0, Scalar(2), -1); // add inferred free space
	drawContours(geoInferenceCostMapOut, tHull, 0, Scalar(202), 3); // add inferred walls

	// my observations are always right
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] < 100){
				geoInferenceCostMapOut.at<uchar>(i,j,0) = 1;
			}
			else if(this->costMap[i][j] >200){
				geoInferenceCostMapOut.at<uchar>(i,j,0) = 201;
			}
		}
	}

	this->inflateWalls(geoInferenceCostMapOut);
	return geoInferenceCostMapOut;
}

void Inference::inflateWalls(Mat &costMat){
	vector<vector<int> > pot;
	for(int i=0; i<costMat.cols; i++){
		for(int j=0; j<costMat.rows; j++){ // for every cell

			if(costMat.at<uchar>(i,j,0) ==  2){ //  inferred free
				for(int k = -this->wallInflationDistance; k<this->wallInflationDistance + 1; k++){
					for(int l = -this->wallInflationDistance; l<this->wallInflationDistance + 1; l++){ // within wallInflationDistance
						if(k + i < 0 || k + i >= costMat.cols){ // only check cells on mat
							break;
						}
						if(l + j < 0 || l + j >= costMat.rows){ // only check cells on mat
							break;
						}

						if(costMat.at<uchar>(i+k,j+l,0) > 200){ // nbr is wall or inferred wall
							vector<int> a;
							a.push_back(i);
							a.push_back(j);
							pot.push_back(a); // add cell to potential cells to inflate
						}
					}
				}
			}
		}
	}

	int cntr = 0;
	for(size_t i=0; i<pot.size(); i++){	// go through all potential cells
		int x = pot[i][0];
		int y = pot[i][1];
		bool flag = true;
		for(int k = -this->wallInflationDistance; k<this->wallInflationDistance + 1; k++){
			for(int l = -this->wallInflationDistance; l<this->wallInflationDistance + 1; l++){ // within wallInflationDistance
				if(k + x < 0 || k + x >= costMat.cols){ // only check cells on mat
					break;
				}
				if(l + y < 0 || l + y >= costMat.rows){ // only check cells on mat
					break;
				}
				if(costMat.at<uchar>(x+k,y+l,0) == 1){ // nbr is free space
					flag = false; // make
					k = this->wallInflationDistance+2;
					l = this->wallInflationDistance+2;
				}
			}
		}
		if(flag){ // no nbrs were free space
			costMat.at<uchar>(x,y,0) = 203;
			cntr++;
		}
	}
}

vector<float> Inference::getInferenceContourRewards(vector<int> frontierExits, vector<vector<Point> > contours){

	vector<float> contourAreas;

	for( size_t i = 0; i < contours.size(); i++ ){
		contourAreas.push_back(contourArea(contours[i]));
		cerr << "contourAreas: " << contourAreas[i] << endl;
	}

    // set value for all Frontiers in each internal contour
	vector<float> contourRewards;
    float maxReward = 0;
    for(size_t i=0; i<contourAreas.size(); i++){
    	Mat temp = Mat::zeros(this->costMat.size(),CV_8UC3);
    	Scalar color;
    	color[0] = rand() % 255;
    	color[1] = rand() % 255;
    	color[2] = rand() % 255;
		drawContours(temp,contours,i,color);

    	float members = 0;
    	for(size_t j=0; j<this->frontiers.size(); j++){
    		Point fp;
    		fp.x = this->frontiers[j].projection[1];
    		fp.y = this->frontiers[j].projection[0];
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

void Inference::addExternalContours(vector<Point> outerHull, vector<vector<Point> > &externalContours, vector<int> frontierExits){
	// get original outerhull area
	float outerHullArea = contourArea(outerHull);
	for(size_t i=0; i<frontierExits.size(); i++){
		Point fC;
		fC.x = this->frontiers[frontierExits[i]].centroid[1];
		fC.y = this->frontiers[frontierExits[i]].centroid[0];
		Point fP;
		fC.x = this->frontiers[frontierExits[i]].projection[1];
		fC.y = this->frontiers[frontierExits[i]].projection[0];

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

		namedWindow("graph::outerHullCalc::outer hull calc", WINDOW_NORMAL);
		imshow("graph::outerHullCalc::outer hull calc", outerHullCalc);
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
		if(p0.x >= this->nCols){ p0.x =this->nCols-1; }
		else if(p0.x < 0){ p0.x = 0; }

		p0.y = w0.y + depth * wly;
		if(p0.y >= this->nRows){ p0.y = this->nRows-1; }
		else if(p0.y < 0){ p0.y = 0; }

		Point p1;
		p1.x = w1.x + depth * wlx;
		if(p1.x >= this->nCols){ p1.x = this->nCols-1; }
		else if(p1.x < 0){ p1.x = 0; }

		p1.y = w1.y + depth * wly;
		if(p1.y >= this->nRows){ p1.y = this->nRows-1; }
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

vector<vector<Point> > Inference::getMinimalInferenceContours(Mat inferenceSpace){
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours(inferenceSpace,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	vector<vector<int> > contourFrontiers;
	// identify contours with Frontiers in them
	for(size_t i=0; i<contours.size(); i++){
		vector<int> m;
		for(size_t j=0; j<this->frontiers.size(); j++){
			Point pf;
			pf.x = this->frontiers[j].projection[1];
			pf.y = this->frontiers[j].projection[0];
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

void Inference::getOuterHull(Mat &inferCalc, Mat &outerHullDrawing, vector<Point> &outerHull){
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] != this->unknown){ // not free space or walls
				inferCalc.at<uchar>(i,j) = 255;
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

void Inference::visualBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull){

}

vector<Point> Inference::getImagePoints(Mat &image){
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

void Inference::clusteringObstacles(){
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

void Inference::structuralBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull){

	for( size_t i = 0; i < contours.size(); i++ ){
		double myArea = contourArea(contours[i]);

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
				for(size_t hl=0; hl< masterHistogramList[i].size(); hl++){
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

			Mat contourToInfer = Mat::zeros(this->costMat.size(), CV_8UC1);
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

			Mat trial = Mat::zeros(this->costMat.size(), CV_8UC1);

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


void Inference::displayInferenceMat(Mat &outerHullDrawing, Mat &obstaclesAndHull, vector<Point> &outerHull, vector<int> frontierExits){
	Mat inferDisplay = Mat::zeros(this->costMat.size(), CV_8UC1);
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

	for(int i=0; i<this->nCols; i++){
		for(int j=0; j<this->nRows; j++){
			Point pa, pb;
			pa.x = i;
			pa.y = j;
			if(this->costMap[i][j] < 10){ // free space with inflation
				inferDisplay.at<uchar>(i,j) = 255;
				this->inferredMiniMap.at<uchar>(i,j,0) = 0;
			}
			else if(this->costMap[i][j] == INFINITY){ // obstacle
				Scalar color;
				color[0] = 127;
				color[1] = 127;
				color[2] = 127;
				inferDisplay.at<uchar>(i,j) = 127;
				obstaclesAndHull.at<uchar>(i,j) = 255;
				this->inferredMiniMap.at<uchar>(i,j,0) = 255;
			}
			else if(this->costMap[i][j] == 100){ // unknown space
				inferDisplay.at<uchar>(i,j) = 0;
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

		pa.x = this->frontiers[i].projection[1];
		pa.y = this->frontiers[i].projection[0];

		circle(inferDisplay,pa,5,color,-1);

		pb.x = this->frontiers[i].centroid[1];
		pb.y = this->frontiers[i].centroid[0];


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
		pa.x = this->frontiers[frontierExits[i]].projection[1];
		pa.y = this->frontiers[frontierExits[i]].projection[0];

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
    					pa.x = t[1];
    					pa.y = t[0];

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

void Inference::getLengthHistogram(vector<float> length, float meanLength, vector<int> &histogram, vector<float> &sequence){
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

vector<int> Inference::getFrontierExits(vector<Point> &outerHull){
	vector<int> frontierExits;
	for(size_t i=0; i<this->frontiers.size(); i++){
		Point pf;
		pf.x = this->frontiers[i].projection[1];
		pf.y = this->frontiers[i].projection[0];
		double t = pointPolygonTest(outerHull,pf, false); // +1 means inside
		if(t <= 0){
			frontierExits.push_back(i);
		}
	}
	return frontierExits;
}

void Inference::extractInferenceContour(){
	Mat temp;
	bitwise_or(this->obstacleMat, this->inferenceMat, temp);
	bitwise_or(temp, this->freeMat, temp);
	bitwise_not(temp,temp);

	imshow("q", temp);
}

Mat Inference::createMiniMapInferImg(){
	Mat temp = Mat::zeros(this->nRows, this->nCols,CV_8UC1);
	for(size_t i=0; i<this->hullPts.size(); i++){
		temp.at<uchar>(this->hullPts[i][1],this->hullPts[i][0],0) = 255;
	}

	Scalar color;
	color[0] = 255;

   for(size_t i=0; i<this->hullPts.size()-1; i++){
   	Point pa, pb;
   	pa.x = this->hullPts[i][0];
   	pa.y = this->hullPts[i][1];
   	pb.x = this->hullPts[i+1][0];
   	pb.y = this->hullPts[i+1][1];
   	line(temp, pa, pb, color, 2, 8);
   }

   for(int i=0; i<1; i++){
   	Point pa, pb;
   	pa.x = this->hullPts[0][0];
   	pa.y = this->hullPts[0][1];

   	pb.x = this->hullPts[this->hullPts.size()-1][0];
   	pb.y = this->hullPts[this->hullPts.size()-1][1];
   	line(temp, pa, pb, color, 2, 8);
   }

	return temp;
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

