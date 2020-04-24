
#include <gl/glut.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <cstdio>
#include <iostream>  
#include <fstream>  

using namespace cv;
using namespace std;

VideoCapture *cap = NULL;

int width = 640;
int height = 480;
Size image_size = Size(width, height);

Mat image;
Mat newimage;
int image_count = 0;

vector<vector<Point2f>> image_points_seq;
vector<vector<Point3f>> object_points;

Mat cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));
Mat distCoeffs = Mat(1, 4, CV_64FC1, Scalar::all(0));
vector<Mat> tvec;
vector<Mat> rvec;
Mat rvecMat = Mat(3, 3, CV_64FC1, Scalar::all(0));
Mat Rvec, Tvec;

double fx = 0.0;
double fy = 0.0;
double cx = 0.0;
double cy = 0.0;
double fovy = 0.0;
double aspectRatio = 0.0;
double zNear = 0.05;
double zFar = 1000;


// a useful function for displaying your coordinate system
void drawAxes(float length)
{
	glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_LIGHTING);

	glBegin(GL_LINES);
	glColor3f(1, 0, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(length, 0, 0);

	glColor3f(0, 1, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, length, 0);

	glColor3f(0, 0, 1);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, length);
	glEnd();

	glPopAttrib();
}

void display()
{
	// clear the window
	glClear(GL_COLOR_BUFFER_BIT);

	// show the current camera frame
	// based on the way cv::Mat stores data, you need to flip it before displaying it
	cv::Mat tempimage;
	cv::flip(image, tempimage, 0);
	
	//////////////////////////////////////////////////////////////////////////////////
	// camera calibration
	// get corner points
	Size board_size = Size(6, 8); // 4 6
	vector<Point2f> image_points_buf;
	if (0 == findChessboardCorners(tempimage, board_size, image_points_buf))
	{
		cout << "can not find chessboard corners!\n"; 
		exit(1);
	}
	else
	{
		Mat view_gray;
		cvtColor(tempimage, view_gray, CV_RGB2GRAY);
			
		find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5));
		image_points_seq.push_back(image_points_buf);
	}

	Size square_size = Size(1, 1);
	vector<int> point_counts;
	vector<Point3f> tempPointSet;
	int i, j, t;
	image_count = 1;
	for (t = 0; t < image_count; t++)
	{
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;

				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	// calibration
	solvePnP(tempPointSet, image_points_buf, cameraMatrix, distCoeffs, Rvec, Tvec, 0, 0);
	//calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvec, tvec, 0);
	newimage = tempimage.clone();

	undistort(tempimage, newimage, cameraMatrix, distCoeffs);

	//////////////////////////////////////////////////////////////////////////////////
	// Here, set up new parameters to render a scene viewed from the camera.
	glDrawPixels(newimage.size().width, newimage.size().height, GL_BGR_EXT, GL_UNSIGNED_BYTE, newimage.ptr());

	//set viewport
	glViewport(0, 0, newimage.size().width, newimage.size().height);

	///////set projection matrix using intrinsic camera params
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	fx = cameraMatrix.at<double>(0, 0);
	fy = cameraMatrix.at<double>(1, 1);
	cx = cameraMatrix.at<double>(0, 2);
	cy = cameraMatrix.at<double>(1, 2);

	GLdouble projectionMat[16] = { 0 };
	projectionMat[0] = 2 * fx / width;
	projectionMat[1] = 0;
	projectionMat[2] = 0;
	projectionMat[3] = 0;

	projectionMat[4] = 0;
	projectionMat[5] = 2 * fy / height;
	projectionMat[6] = 0;
	projectionMat[7] = 0;

	projectionMat[8] = 1 - 2 * cx / width;
	projectionMat[9] = -1 + (2 * cy + 2) / height;;
	projectionMat[10] = (zNear + zFar) / (zNear - zFar);
	projectionMat[11] = -1;

	projectionMat[12] = 0;
	projectionMat[13] = 0;
	projectionMat[14] = 2 * zNear * zFar / (zNear - zFar);
	projectionMat[15] = 0;

	glLoadMatrixd(projectionMat);

	//gluPerspective is arbitrarily set, you will have to determine these values based
	//on the intrinsic camera parameters
	fovy = 38.7164;
	aspectRatio = 1.001694;
	//gluPerspective(fovy, aspectRatio, zNear, zNear);
	//gluPerspective(60, newimage.size().width*1.0 / newimage.size().height, 1, 20);

	//////you will have to set modelview matrix using extrinsic camera params
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	Mat constantVector = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	Mat extrinsicMatrix = Mat(4, 4, CV_64FC1, Scalar::all(0));

	Rodrigues(Rvec, rvecMat);
	hconcat(rvecMat, Tvec, extrinsicMatrix);
	vconcat(extrinsicMatrix, constantVector, extrinsicMatrix);
	Mat tempMatrix = (Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, -1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	extrinsicMatrix = tempMatrix * extrinsicMatrix;

	Mat glViewMatrix = extrinsicMatrix.clone();
	transpose(extrinsicMatrix, glViewMatrix);

	GLdouble modelviewMat[16] = { 0 };
	modelviewMat[0] = glViewMatrix.at<double>(0, 0);
	modelviewMat[1] = glViewMatrix.at<double>(0, 1);
	modelviewMat[2] = glViewMatrix.at<double>(0, 2);
	modelviewMat[3] = glViewMatrix.at<double>(0, 3);

	modelviewMat[4] = glViewMatrix.at<double>(1, 0);
	modelviewMat[5] = glViewMatrix.at<double>(1, 1);
	modelviewMat[6] = glViewMatrix.at<double>(1, 2);
	modelviewMat[7] = glViewMatrix.at<double>(1, 3);

	modelviewMat[8] = glViewMatrix.at<double>(2, 0);
	modelviewMat[9] = glViewMatrix.at<double>(2, 1);
	modelviewMat[10] = glViewMatrix.at<double>(2, 2);
	modelviewMat[11] = glViewMatrix.at<double>(2, 3);

	modelviewMat[12] = glViewMatrix.at<double>(3, 0);
	modelviewMat[13] = glViewMatrix.at<double>(3, 1);
	modelviewMat[14] = glViewMatrix.at<double>(3, 2);
	modelviewMat[15] = glViewMatrix.at<double>(3, 3);

	glLoadMatrixd(modelviewMat);

	//gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

	/////////////////////////////////////////////////////////////////////////////////
	// Drawing routine

	//now that the camera params have been set, draw your 3D shapes
	//first, save the current matrix
	glPushMatrix();
	//move to the position where you want the 3D object to go
	glTranslatef(3.5, 2.5, 0);  //this is an arbitrary position for demonstration
								//you will need to adjust your transformations to match the positions where
								//you want to draw your objects(i.e. chessboard center, chessboard corners)
	glRotatef(90, 1, 0, 0);
	//glutSolidTeapot(1);
	glutWireTeapot(2);
	drawAxes(4.5);

	glPopMatrix();
	// show the rendering on the screen
	glutSwapBuffers();
	// post the next redisplay
	glutPostRedisplay();
}

void reshape(int w, int h)
{
	// set OpenGL viewport (drawable area)
	glViewport(0, 0, w, h);
}

void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{

	}
}

void drawSphere()
{
	// clear the window
	glClear(GL_COLOR_BUFFER_BIT);

	// show the current camera frame
	// based on the way cv::Mat stores data, you need to flip it before displaying it
	cv::Mat tempimage;
	cv::flip(image, tempimage, 0);

	//////////////////////////////////////////////////////////////////////////////////
	// camera calibration
	// get corner points
	Size board_size = Size(6, 8); // 4 6
	vector<Point2f> image_points_buf;
	if (0 == findChessboardCorners(tempimage, board_size, image_points_buf))
	{
		cout << "can not find chessboard corners!\n"; 
		exit(1);
	}
	else
	{
		Mat view_gray;
		cvtColor(tempimage, view_gray, CV_RGB2GRAY);
			
		find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5));
		image_points_seq.push_back(image_points_buf);
	}

	Size square_size = Size(1, 1);
	vector<int> point_counts;
	vector<Point3f> tempPointSet;
	int i, j, t;
	image_count = 1;
	for (t = 0; t < image_count; t++)
	{
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;

				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	
	// calibration
	solvePnP(tempPointSet, image_points_buf, cameraMatrix, distCoeffs, Rvec, Tvec, 0, 0);
	//calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvec, tvec, 0);
	newimage = tempimage.clone();

	undistort(tempimage, newimage, cameraMatrix, distCoeffs);

	//////////////////////////////////////////////////////////////////////////////////
	// Here, set up new parameters to render a scene viewed from the camera.
	glDrawPixels(newimage.size().width, newimage.size().height, GL_BGR_EXT, GL_UNSIGNED_BYTE, newimage.ptr());

	//set viewport
	glViewport(0, 0, newimage.size().width, newimage.size().height);

	///////set projection matrix using intrinsic camera params
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	fx = cameraMatrix.at<double>(0, 0);
	fy = cameraMatrix.at<double>(1, 1);
	cx = cameraMatrix.at<double>(0, 2);
	cy = cameraMatrix.at<double>(1, 2);

	GLdouble projectionMat[16] = { 0 };
	projectionMat[0] = 2 * fx / width;
	projectionMat[1] = 0;
	projectionMat[2] = 0;
	projectionMat[3] = 0;

	projectionMat[4] = 0;
	projectionMat[5] = 2 * fy / height;
	projectionMat[6] = 0;
	projectionMat[7] = 0;

	projectionMat[8] = 1 - 2 * cx / width;
	projectionMat[9] = -1 + (2 * cy + 2) / height;;
	projectionMat[10] = (zNear + zFar) / (zNear - zFar);
	projectionMat[11] = -1;

	projectionMat[12] = 0;
	projectionMat[13] = 0;
	projectionMat[14] = 2 * zNear * zFar / (zNear - zFar);
	projectionMat[15] = 0;

	glLoadMatrixd(projectionMat);

	//gluPerspective is arbitrarily set, you will have to determine these values based
	//on the intrinsic camera parameters
	fovy = 38.7164;
	aspectRatio = 1.001694;
	//gluPerspective(fovy, aspectRatio, zNear, zNear);
	//gluPerspective(60, newimage.size().width*1.0 / newimage.size().height, 1, 20);

	//////you will have to set modelview matrix using extrinsic camera params
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	Mat constantVector = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	Mat extrinsicMatrix = Mat(4, 4, CV_64FC1, Scalar::all(0));

	Rodrigues(Rvec, rvecMat);
	hconcat(rvecMat, Tvec, extrinsicMatrix);
	vconcat(extrinsicMatrix, constantVector, extrinsicMatrix);
	Mat tempMatrix = (Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, -1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	extrinsicMatrix = tempMatrix * extrinsicMatrix;

	Mat glViewMatrix = extrinsicMatrix.clone();
	transpose(extrinsicMatrix, glViewMatrix);

	GLdouble modelviewMat[16] = { 0 };
	modelviewMat[0] = glViewMatrix.at<double>(0, 0);
	modelviewMat[1] = glViewMatrix.at<double>(0, 1);
	modelviewMat[2] = glViewMatrix.at<double>(0, 2);
	modelviewMat[3] = glViewMatrix.at<double>(0, 3);

	modelviewMat[4] = glViewMatrix.at<double>(1, 0);
	modelviewMat[5] = glViewMatrix.at<double>(1, 1);
	modelviewMat[6] = glViewMatrix.at<double>(1, 2);
	modelviewMat[7] = glViewMatrix.at<double>(1, 3);

	modelviewMat[8] = glViewMatrix.at<double>(2, 0);
	modelviewMat[9] = glViewMatrix.at<double>(2, 1);
	modelviewMat[10] = glViewMatrix.at<double>(2, 2);
	modelviewMat[11] = glViewMatrix.at<double>(2, 3);

	modelviewMat[12] = glViewMatrix.at<double>(3, 0);
	modelviewMat[13] = glViewMatrix.at<double>(3, 1);
	modelviewMat[14] = glViewMatrix.at<double>(3, 2);
	modelviewMat[15] = glViewMatrix.at<double>(3, 3);

	glLoadMatrixd(modelviewMat);

	//gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

	/////////////////////////////////////////////////////////////////////////////////
	// Drawing routine

	//now that the camera params have been set, draw your 3D shapes
	//first, save the current matrix
	glPushMatrix();

	//move to the position where you want the 3D object to go
	int ii;
	int iii = 0;
	int jj;
	int jjj = 0;
	for (ii = 0; ii < 8; ii++, iii = 1)
	{
		glTranslatef(iii, 0, 0);
		glutSolidSphere(0.2, 100, 100);
	}
	glTranslatef(0, 1, 0);
	glutSolidSphere(0.2, 100, 100);
	jjj = -1;
	for (jj = 0; jj < 7; jj++, jjj = -1)
	{
		glTranslatef(jjj, 0, 0);
		glutSolidSphere(0.2, 100, 100);
	}
	glTranslatef(0, 1, 0);
	glutSolidSphere(0.2, 100, 100);
	iii = 1;
	for (ii = 0; ii < 7; ii++, iii = 1)
	{
		glTranslatef(iii, 0, 0);
		glutSolidSphere(0.2, 100, 100);
	}
	glTranslatef(0, 1, 0);
	glutSolidSphere(0.2, 100, 100);
	for (jj = 0; jj < 7; jj++, jjj = -1)
	{
		glTranslatef(jjj, 0, 0);
		glutSolidSphere(0.2, 100, 100);
	}
	glTranslatef(0, 1, 0);
	glutSolidSphere(0.2, 100, 100);
	for (ii = 0; ii < 7; ii++, iii = 1)
	{
		glTranslatef(iii, 0, 0);
		glutSolidSphere(0.2, 100, 100);
	}
	glTranslatef(0, 1, 0);
	glutSolidSphere(0.2, 100, 100);
	for (jj = 0; jj < 7; jj++, jjj = -1)
	{
		glTranslatef(jjj, 0, 0);
		glutSolidSphere(0.2, 100, 100);
	}

	glPopMatrix();
	// show the rendering on the screen
	glutSwapBuffers();
	// post the next redisplay
	glutPostRedisplay();
}


void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
		// quit when q is pressed
		exit(0);
		break;
	case ' ':
		// draw spheres when space is pressed
		drawSphere();

	//default:
	//	break;
	}
}

void idle()
{
	// grab a frame from the camera
	(*cap) >> image;
}

int main(int argc, char **argv)
{
	int w, h;
	
	if (argc == 2) {
		// start video capture from camera
		fstream file;
		file.open(argv[1]);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				file >> cameraMatrix.at<double>(i, j);
			}
		}
		for (int i = 0; i < 4; i++)
		{
			file >> distCoeffs.at<double>(0, i);
		}

		cap = new cv::VideoCapture(0);

		//// check that video is opened
		if (cap == NULL || !cap->isOpened()) {
			fprintf(stderr, "could not start video capture\n");
			return 1;
		}

		// get width and height
		w = (int)cap->get(cv::CAP_PROP_FRAME_WIDTH);
		h = (int)cap->get(cv::CAP_PROP_FRAME_HEIGHT);
		// On Linux, there is currently a bug in OpenCV that returns 
		// zero for both width and height here (at least for video from file)
		// hence the following override to global variable defaults: 
		width = w ? w : width;
		height = h ? h : height;
	}
	else if (argc == 3) {
		// start video capture from file
		fstream file;
		file.open(argv[1]);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				file >> cameraMatrix.at<double>(i, j);
			}
		}
		for (int i = 0; i < 4; i++)
		{
			file >> distCoeffs.at<double>(0, i);
		}

		cap = new cv::VideoCapture(argv[2]);

		//// check that video is opened
		if (cap == NULL || !cap->isOpened()) {
			fprintf(stderr, "could not start video capture\n");
			return 1;
		}

		// get width and height
		w = (int)cap->get(cv::CAP_PROP_FRAME_WIDTH);
		h = (int)cap->get(cv::CAP_PROP_FRAME_HEIGHT);
		// On Linux, there is currently a bug in OpenCV that returns 
		// zero for both width and height here (at least for video from file)
		// hence the following override to global variable defaults: 
		width = w ? w : width;
		height = h ? h : height;

	}
	else if (argc == 4) {
		// start image reading from file
		fstream file;
		file.open(argv[1]);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				file >> cameraMatrix.at<double>(i, j);
			}
		}
		for (int i = 0; i < 4; i++)
		{
			file >> distCoeffs.at<double>(0, i);
		}

		image = imread(argv[2]);
	}
	else {
		fprintf(stderr, "usage: %s [<filename>]\n", argv[0]);
		return 1;
	}

	// initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowPosition(20, 20);
	glutInitWindowSize(width, height);
	glutCreateWindow("OpenGL / OpenCV AR");

	// set up GUI callback functions
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);

	// start GUI loop
	glutMainLoop();

	return 0;
}