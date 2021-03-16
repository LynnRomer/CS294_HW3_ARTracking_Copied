#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 02:07:53 2020

@author: zhironglin
"""

import cv2
import numpy as np
import CalibrationHelpers as calib
import glob
from collections import Counter
import open3d as o3d

# This function takes in an intrinsics matrix, and two sets of 2d points
# if a pose can be computed it returns true along with a rotation and 
# translation between the sets of points. 
# returns false if a good pose estimate cannot be found
def ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints):
    # compute homography using RANSAC, this allows us to compute
    # the homography even when some matches are incorrect
    homography, mask = cv2.findHomography(referencePoints, imagePoints, 
                                          cv2.RANSAC, 5.0)
    # check that enough matches are correct for a reasonable estimate
    # correct matches are typically called inliers
    MIN_INLIERS = 30
    if(sum(mask)>MIN_INLIERS):
        # given that we have a good estimate
        # decompose the homography into Rotation and translation
        # you are not required to know how to do this for this class
        # but if you are interested please refer to:
        # https://docs.opencv.org/master/d9/dab/tutorial_homography.html
        RT = np.matmul(np.linalg.inv(new_intrinsics), homography)
        norm = np.sqrt(np.linalg.norm(RT[:,0])*np.linalg.norm(RT[:,1]))
        RT = -1*RT/norm
        c1 = RT[:,0]
        c2 = RT[:,1]
        c3 = np.cross(c1,c2)
        T = RT[:,2]
        R = np.vstack((c1,c2,c3)).T
        W,U,Vt = cv2.SVDecomp(R)
        R = np.matmul(U,Vt)
        return True, R, T
    # return false if we could not compute a good estimate
    return False, None, None

# This function is yours to complete
# it should take in a set of 3d points and the intrinsic matrix
# rotation matrix(R) and translation vector(T) of a camera
# it should return the 2d projection of the 3d points onto the camera defined
# by the input parameters    




def ProjectPoints(points3d, new_intrinsics, R, T):
    # your code here!
    # points2d_tuple = tuple()

    #comput points u,v in 2d image
    #InstrinsicMatrix is fixed per Camera = new_intrinsics
    #ExtrinsicMatrix=np.column_stack((R,T))
    #homogeneousCord = points3d
    # print("points3dshape" + str(points3d.shape))
    # print(points3d)
    points3d = np.insert(points3d, 3, 1, axis=1)
    # print("points3dshape" + str(points3d.shape))
    # print(points3d)
    points2d = np.zeros((1,1,3))

    # print("points3dsize" + str(points3d.shape))
    # print(points3d)
    for point in points3d:
        ExtrinsicMatrix = np.column_stack((R,T))
        # print("Extsize" + str(ExtrinsicMatrix.shape))
        # print(ExtrinsicMatrix)
        point2d = new_intrinsics.dot(ExtrinsicMatrix).dot(point)
        # print("point2d" + str(point2d.shape))
        # print(point2d)
        points2d = np.insert(points2d,0,point2d,axis=0)
    points2d = points2d[:-1]
    # print("points2d" + str(points2d.shape))
    # print(points2d)
    # print("lenPs2d" + str(len(points2d)))
    # print("lenPs2di " + str(len(points2d[1])))
    for i in range(len(points2d)):
        points2d[i] = points2d[i]/points2d[i][0][2]
    points2d = np.delete(points2d,2,axis = 2)
    # print("points2dsize" + str(points2d.shape))
    # print(points2d[0])
    # print(points2d[0][0])
  
    return points2d
    
# =============================================================================
#This function will render a cube on an image whose camera is defined
#by the input intrinsics matrix, rotation matrix(R), and translation vector(T)
def renderCube(img_in, new_intrinsics, R, T):
    # Setup output image
    img = np.copy(img_in)

    # We can define a 10cm cube by 4 sets of 3d points
    # these points are in the reference coordinate frame
    scale = 0.1
    face1 = np.array([[0,0,0],[0,0,scale],[0,scale,scale],[0,scale,0]],
                      np.float32)
    face2 = np.array([[0,0,0],[0,scale,0],[scale,scale,0],[scale,0,0]],
                      np.float32)
    face3 = np.array([[0,0,scale],[0,scale,scale],[scale,scale,scale],
                      [scale,0,scale]],np.float32)
    face4 = np.array([[scale,0,0],[scale,0,scale],[scale,scale,scale],
                      [scale,scale,0]],np.float32)
    # using the function you write above we will get the 2d projected 
    # position of these points
    face1_proj = ProjectPoints(face1, new_intrinsics, R, T)
    # this function simply draws a line connecting the 4 points
    img = cv2.polylines(img, [np.int32(face1_proj)], True, 
                              tuple([255,0,0]), 3, cv2.LINE_AA) 
    # repeat for the remaining faces
    face2_proj = ProjectPoints(face2, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face2_proj)], True, 
                              tuple([0,255,0]), 3, cv2.LINE_AA) 
    
    face3_proj = ProjectPoints(face3, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face3_proj)], True, 
                              tuple([0,0,255]), 3, cv2.LINE_AA) 
    
    face4_proj = ProjectPoints(face4, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face4_proj)], True, 
                              tuple([125,125,0]), 3, cv2.LINE_AA) 
    return img

# =============================================================================
# Load Images
# =============================================================================
path = glob.glob('Reference1Image/*.png')
Imgs = []

for img in path:
    n = cv2.imread(img)
    Imgs.append(n)
    
print(len(Imgs))

# =============================================================================
# reference image is the first image in Imgs list

reference = Imgs[0]
#reference = cv2.imread('ARTrackerImage.jpg',0)
RES = 480
#reference = cv2.resize(reference,(RES,RES))

# create the feature detector. This will be used to find and describe locations
# in the image that we can reliably detect in multiple images
feature_detector = cv2.BRISK_create(octaves=5)
# compute the features in the reference image
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
        
# computer new image keypoints
NewImgs_points = Imgs[1:]



# make image to visualize keypoints
keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]), 
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image
# cv2.imshow("Keypoints",keypoint_visualization)
# wait for user to press a key before proceeding
# create the matcher that is used to compare feature similarity
# Brisk descriptors are binary descriptors (a vector of zeros and 1s)
# Thus hamming distance is a good measure of similarity        
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load the camera calibration matrix
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('phonecalibration_data')

# initialize video capture
# the 0 value should default to the webcam, but you may need to change this
# for your camera, especially if you are using a camera besides the default
#cap = cv2.VideoCapture(0)



#cap = cv2.imread(path)



# read the current frame from the webcam
#ret, current_frame = cap.read()
#current_frame = cap.read()

# ensure the image is valid
# if not ret:
#     print("Unable to capture video")
#     break


# =============================================================================
# Compute points in image
# =============================================================================
def ComputePointsbet2Images(intrinsics, matches, points1, points2, Rx1, Tx1):
    c_x = intrinsics[0,2]
    c_y = intrinsics[1,2]
    f_x = intrinsics[0,0]
    f_y = intrinsics[1,1]
    
    for m in matches:
        ref = m.queryIdx
        cur = m.trainIdx
    
        (u_1,v_1) = points1[ref].pt
        (u_2,v_2) = points2[cur].pt
    
        x_1 = np.array([(u_1 - c_x)/f_x, (v_1 - c_y)/f_y, 1])
        x_2 = np.array([(u_2 - c_x)/f_x, (v_2 - c_y)/f_y, 1])
    
    return (x_1, x_2)
    
    

# =============================================================================
#  Use epipolar constraints to remove bad feature matches
# =============================================================================
def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1,
                               threshold = 0.01):
    # # your code here
    inlier_mask = []
    c_x = intrinsics[0,2]
    c_y = intrinsics[1,2]
    f_x = intrinsics[0,0]
    f_y = intrinsics[1,1]
    
    TempE = np.cross(Tx1, Rx1)
    
    
    for m in matches:
        ref = m.queryIdx
        cur = m.trainIdx
        
        (u_1,v_1) = points1[ref].pt
        (u_2,v_2) = points2[cur].pt
        
        x_1 = np.array([(u_1 - c_x)/f_x, (v_1 - c_y)/f_y, 1])
        x_2 = np.array([(u_2 - c_x)/f_x, (v_2 - c_y)/f_y, 1])
        
        
        #print("x1" + str(x_1)+ str(x_1.shape))
        #print("x2" + str(x_2))
        
        #if abs(np.matmul(np.matmul(x_2.T,E),x_1))  < threshold:
        if (x_2).dot(TempE).dot(x_1) < threshold:
            inlier_mask.append(1)
        else:
            inlier_mask.append(0)
    #print("inlier_mask" + len(inlier_mask))
    #print(inlier_mask)

   
    return inlier_mask 

# inlier_mask = FilterByEpipolarConstraint(new_intrinsics,matches,reference,Imgs[1],
#                                          RsRef1[0],TsRef1[0],threshold=0.01)
# print("inlier_mask" + str(len(inlier_mask)))

# =============================================================================
# Build Matrix M
# =============================================================================




def ComputemMatrix (intrinsics, matches, points1, points2, Rx1, Tx1):
    
    # featurecheck = 0
    
    # matchesfull =  numMatches
    
    MatrixM = np.zeros( ((3*(len(Imgs)-1)*numMatches ), numMatches+1 ))
    
    c_x = intrinsics[0,2]
    c_y = intrinsics[1,2]
    f_x = intrinsics[0,0]
    f_y = intrinsics[1,1]
    
    for m in matches:
        #compute x_1,x_2 locaiton in reference 1 & current frame
        ref = m.queryIdx
        cur = m.trainIdx
        
        (u_1,v_1) = points1[ref].pt
        (u_2,v_2) = points2[cur].pt
        
        x_1 = np.array([(u_1 - c_x)/f_x, (v_1 - c_y)/f_y, 1])
        x_2 = np.array([(u_2 - c_x)/f_x, (v_2 - c_y)/f_y, 1])
        
        #get the matched feacture index in reference
        
        #if m.queryIdx in ReferenceFeatureIndex:
        if m.queryIdx in ReferenceFeatureIndex:
            indexM = ReferenceFeatureIndex.index(m.queryIdx)
            
        dataR = np.cross(np.matmul(Rx1, x_1**(countcheck) ), x_2**(countcheck) ,axisa=0,axisb=0)
        # print("dataRshape" + str(dataR.shape))
                
        MatrixM[(countcheck)*3-3][indexM] = dataR[0]
                
        MatrixM[(countcheck)*3-2][indexM] = dataR[1]
                
        MatrixM[(countcheck)*3-1][indexM] = dataR[2]
        
       
        dataT = np.cross(x_2**(countcheck), Tx1)
        
        MatrixM[(countcheck)*3-3][-1] = dataT[0]
        MatrixM[(countcheck)*3-2][-1] = dataT[1]
        MatrixM[(countcheck)*3-1][-1] = dataT[2]
        # print("matrix" + str(MatrixM))

    
    return MatrixM

# =============================================================================
# Conmpute 2d point locaion in an reference image
# =============================================================================
def Compute2dpointImage (intrinsics, matches, points1):
        
    c_x = intrinsics[0,2]
    c_y = intrinsics[1,2]
    f_x = intrinsics[0,0]
    f_y = intrinsics[1,1]
    
    x_1_points = []
    
    for m in matches:
        #compute x_1,x_2 locaiton in reference 1 & current frame
        ref = m.queryIdx
        
        (u_1,v_1) = points1[ref].pt

        x_1 = np.array([(u_1 - c_x)/f_x, (v_1 - c_y)/f_y, 1])
        
        x_1_points.append(x_1)
    
    return x_1_points

    



# =============================================================================
# Calculate R & T & Feature Track
# =============================================================================

featureInxCount = []

Rs = np.zeros((1,3,3))
Ts = np.zeros((1,3))

# Create a list in store all feature index in refernce image
ReferenceFeatureIndex = []

for i in range(len(Imgs)):
    
    checknum = i 
    
    current_frame = Imgs[i]
    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    # match the features from the reference image to the current image
    matches = matcher.match(reference_descriptors, current_descriptors)
    if checknum == 0:
        numMatches = len(matches)
        for m in matches:
            ReferenceFeatureIndex.append(m.trainIdx)
        ReferenceFeatureIndex = sorted(ReferenceFeatureIndex)
        matches_ref = matches
            
        
    print("matches" + str(len(matches)))
    
    for index in matches:
        MatchIndex = index.queryIdx
        
        featureInxCount.append(MatchIndex)
    # matches returns a vector where for each element there is a 
    # query index matched with a train index. I know these terms don't really
    # make sense in this context, all you need to know is that for us the 
    # query will refer to a feature in the reference image and train will
    # refer to a feature in the current image
    
    
    # create a visualization of the matches between the reference and the
    # current image
    
    # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                         current_keypoints, matches, 0, 
    #                         flags=
    #                         cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


    # cv2.imshow('matches',match_visualization)
    # k = cv2.waitKey(1)
    # if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
    #     #exit
    #     Break
# cv2.destroyAllWindows() 
# cap.release()

    
    
    
    # set up reference points and image points
    # here we get the 2d position of all features in the reference image
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
    # convert positions from pixels to meters
    SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
    referencePoints = SCALE*referencePoints/RES
    
    imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    # compute homography
    ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                          imagePoints)
    # render_frame = current_frame
    # print("R")
    # print(R)
    # print("T")
    # print(T)
    Rs = np.insert(Rs,0,R,axis=0)
    Ts = np.insert(Ts,0,T,axis=0)
Rs = Rs[:-1]
Ts = Ts[:-1]
print("Rs" +str(Rs))
print("Ts" + str(Ts))


# =============================================================================
# Compute Feature Tracks
# =============================================================================

featureInxCount = Counter(featureInxCount)
Matches_Index = [x[0] for x in featureInxCount.items() if x[1] >= 4]

# =============================================================================
# R, T , E calculation
# =============================================================================
RsRef1 = np.zeros((1,3,3))
TsRef1= np.zeros((1,3))
Es = np.zeros((1,3,3))

for i in range(4):
    RefR = np.matmul(Rs[i+1],Rs[0].T)
    RsRef1= np.insert(RsRef1,0,RefR,axis=0)
    RefT = Ts[i+1] -  np.matmul(RefR,Ts[0])
    TsRef1 = np.insert(TsRef1,0,RefT,axis=0)
    E = np.cross(RefT,RefR,axisa=0,axisb=0)
    Es = np.insert(Es,0,E,axis=0)
    
    
RsRef1 = RsRef1[:-1]
TsRef1 = TsRef1[:-1]
Es = Es[:-1]
# print("Rsref" +str(RsRef1))
# print("Tsref" +str(TsRef1))
# print("Es" + str(Es))
    
    
        # compute the projection and render the cube
        
    # render_frame = renderCube(current_frame,new_intrinsics,R,T) 
        
    # display the current image frame
    # cv2.imshow('frame', render_frame)
    # k = cv2.waitKey(1)
    # if k ==27 or k ==113:
    #     break
    #27, 113 are ascii for escape and q respectively
    
# =============================================================================
# Create MatrixM
# =============================================================================
    
# MatrixM = np.zeros((len(Imgs)-1,numMatches+1))
# =============================================================================
# Show view  
# =============================================================================


for i in range(1,5):

    current_frame = Imgs[i]
    
    #countcheck use to comput M matrix
    countcheck = i
    

    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
   # match the features from the reference image to the current image
    matches = matcher.match(reference_descriptors, current_descriptors)
# =============================================================================
# #Filter matches only if it is in Matches_Index
# =============================================================================
    for element in matches:
        if element.queryIdx not in Matches_Index:
            matches.remove(element)
    print("matcheslen"+str(len(matches)))
    
    # print("matchesfiltered" + str(len(matches)))
    
    MatrixM = ComputemMatrix (new_intrinsics, matches, reference_keypoints,current_keypoints,RsRef1[i-1],TsRef1[i-1])
    


    # matches returns a vector where for each element there is a 
    # query index matched with a train index. I know these terms don't really
    # make sense in this context, all you need to know is that for us the 
    # query will refer to a feature in the reference image and train will
    # refer to a feature in the current image
    
    # create a visualization of the matches between the reference and the
    # current image
    
    # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                         current_keypoints, matches, 0, 
    #                         flags=
    #                         cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    
    inlier_mask = FilterByEpipolarConstraint(new_intrinsics,matches,reference_keypoints,current_keypoints,
                                          RsRef1[i-1],TsRef1[i-1],threshold=0.01)
    
    # print("inlier_mask" + str(len(inlier_mask)))


    
    match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,current_keypoints, 
                                          matches, 0, matchesMask =inlier_mask, #this applies your inlier filter
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)



    
    cv2.imshow('matches',match_visualization)
    k = cv2.waitKey(1)
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        Break
# cv2.destroyAllWindows() 
# cap.release()

    
    
    
    # set up reference points and image points
    # here we get the 2d position of all features in the reference image
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
    # convert positions from pixels to meters
    SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
    referencePoints = SCALE*referencePoints/RES
    
    imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    # compute homography
    # ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
    #                                       imagePoints)
    render_frame = current_frame
    # print("R")
    # print(R)
    # print("T")
    # print(T)
    # Rs = np.insert(Rs,0,R,axis=0)
    # Ts = np.insert(Ts,0,T,axis=0)
    
    
        # compute the projection and render the cube
        
    # render_frame = renderCube(current_frame,new_intrinsics,R,T) 
        
    # display the current image frame
    cv2.imshow('frame', render_frame)
    k = cv2.waitKey(1)
    if k ==27 or k ==113:
        break
    #27, 113 are ascii for escape and q respectively

    cv2.imshow('matches',match_visualization)
    k = cv2.waitKey(1)
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        Break
        
# =============================================================================
# Compute the Depth  & 3d position
# =============================================================================

# 3d points position in reference image
x_1_imgRef = Compute2dpointImage (new_intrinsics, matches_ref, reference_keypoints)
print(len(x_1_imgRef))

W,U,Vt = cv2.SVDecomp(MatrixM)
depths = Vt[-1,:]/Vt[-1,-1]

#print(U,U,Vt,depths)

#coompute 3d postion in read world
X_1_Real = []
for i in range(len(x_1_imgRef)):
    X_1_pointR = np.dot(depths[i].T,x_1_imgRef[i])

    X_1_Real.append(X_1_pointR)

print("realwordpoints" + str(len(X_1_Real)))



# =============================================================================
# Visulation
# =============================================================================

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X_1_Real)
o3d.visualization.draw_geometries([pcd])





        
cv2.destroyAllWindows() 
# cap.release()

