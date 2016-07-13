/*
 * Inference.h
 *
 *  Created on: Jul 12, 2016
 *      Author: andy
 */

#ifndef INFERENCE_H_
#define INFERENCE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

#include "World.h"
#include "Frontier.h"
#include "Contour.h"

class Inference {
public:
	Inference();
	virtual ~Inference();

	// usfeul stuff
	int nCols, nRows;
	int obsFree, infFree, unknown, obsWall, infWall, inflatedWall;
	Mat obstacleMat, freeMat, frontierMat, unknownMat, inferenceMat, inferredMiniMap;;
	Mat costMat;
	vector<vector<int> > costMap;

	// file transfer stuff
	vector<vector<int> > matToMap(Mat &costMat);
	Mat mapToMat(vector<vector<int> > &costMap);
	void importCostMap(vector<vector<int> > costMap);


	// main inference functions
	Mat makeGlobalInferenceMat(World &gMap);
	Mat makeGeometricInferenceMatForMiniMap();
	Mat makeStructuralInferenceMatForMiniMap();
	Mat makeVisualInferenceMatForMiniMap();
	Mat makeNaiveMatForMiniMap();

	// wall inflation
	int wallInflationDistance;
	void inflateWalls(Mat &costMat);

	// frontiers
	vector<Frontier> frontiers;

	// Inference tools
	vector<Point> getImagePoints(Mat &image);

	Mat getObstaclesImage();
	Mat getFrontiersImage();
	Mat getFreeSpaceImage();
	Mat getInferenceImage();

	vector<vector<int> > hullPts;
	Mat createMiniMapInferImg();

	void extractInferenceContour();
	void getLengthHistogram(vector<float> length, float meanLength, vector<int> &histogram, vector<float> &sequence);
	vector<vector<int> > masterHistogramList;
	vector<vector<float> > masterSequenceList;
	vector<vector<Point> > masterPointList;
	vector<String> masterNameList;
	vector<Point> masterCenterList;
	vector<float> masterMeanLengthList;
	float minMatchStrength;

	vector<int> getFrontierExits(vector<Point> &outerHull);
	void makeInferenceAndSetFrontierRewards(); // create outer hull and divide into contours
	void visualInference(); // perform visual inference on each contour
	void valueFrontiers(); // take the contours from inference and get the value of each
	void addExternalContours(vector<Point> outerHull, vector<vector<Point> > &externalContours, vector<int> frontierExit);

	void getOuterHull(Mat &inferCalc, Mat &outerHullDrawing, vector<Point> &outerHull);
	vector<vector<Point> > getMinimalInferenceContours(Mat inferenceSpace);
	vector<float> getInferenceContourRewards(vector<int> frontierExits, vector<vector<Point> > contours);
	void setFrontierRewards(vector<float> rewards, vector<vector<Point> > inferenceContours);
	void displayInferenceMat(Mat &outerHullDrawing, Mat &obstacleHull, vector<Point> &outerHull, vector<int> frontierExits);

	void structuralBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull);
	void visualBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull);

	void clusteringObstacles();


};

#endif /* INFERENCE_H_ */
