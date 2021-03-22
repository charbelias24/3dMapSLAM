#!/bin/bash
pathDataset='../../Datasets/Kitti/dataset/sequences' #Example, it is necesary to change it by the dataset path
yaml=""

if [ $1 -le 2 ] 
then
	yaml="00-02"
elif [ $1 -eq 3 ]
then 
	yaml="03"
else
	yaml="04-12"
fi


echo "Launching Seq $(seq -f '%02g' $1 $1) with Stereo sensor"
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI"$yaml".yaml "$pathDataset"/$(seq -f "%02g" $1 $1 )

#------------------------------------
# Stereo Examples

#echo "Launching Seq 01 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml "$pathDataset"/01

#echo "Launching Seq 02 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml "$pathDataset"/02

#echo "Launching Seq 03 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI03.yaml "$pathDataset"/03

#echo "Launching Seq 04 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml "$pathDataset"/04

#echo "Launching Seq 05 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml "$pathDataset"/05

#echo "Launching Seq 06 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml "$pathDataset"/06

#echo "Launching Seq 07 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml "$pathDataset"/07

#echo "Launching Seq 08 with Stereo sensor"
#./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml "$pathDataset"/08

echo "Launching Seq 12 with Stereo sensor"
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml "$pathDataset"/12
#------------------------------------
# Monocular-Inertial Examples
#echo "Launching Room 3 with Monocular-Inertial sensor"
#./Examples/Monocular-Inertial/mono_inertial_tum_vi Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam0/data Examples/Monocular-Inertial/TUM_TimeStamps/dataset-room3_512.txt Examples/Monocular-Inertial/TUM_IMU/dataset-room3_512.txt dataset-room3_512_monoi

#------------------------------------
# Stereo-Inertial Examples
#echo "Launching Room 3 with Stereo-Inertial sensor"
#./Examples/Stereo-Inertial/stereo_inertial_tum_vi Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam1/data Examples/Stereo-Inertial/TUM_TimeStamps/dataset-room3_512.txt Examples/Stereo-Inertial/TUM_IMU/dataset-room3_512.txt dataset-room3_512_stereoi


