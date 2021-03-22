#!/bin/bash
pathDatasetTUM_VI='../../Datasets/TUM' #Example, it is necesary to change it by the dataset path

#------------------------------------
# Monocular Examples
#echo "Launching Room 3 with Monocular sensor"
#./Examples/Monocular/mono_tum_vi Vocabulary/ORBvoc.txt Examples/Monocular/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam0/data Examples/Monocular/TUM_TimeStamps/dataset-room3_512.txt dataset-room3_512_mono

#------------------------------------
# Stereo Examples
#echo "Launching Room 3 with Stereo sensor"
#./Examples/Stereo/stereo_tum_vi Vocabulary/ORBvoc.txt Examples/Stereo/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam1/data Examples/Stereo/TUM_TimeStamps/dataset-room3_512.txt dataset-room3_512_stereo

#------------------------------------
# Monocular-Inertial Examples
#echo "Launching Room 3 with Monocular-Inertial sensor"
#./Examples/Monocular-Inertial/mono_inertial_tum_vi Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam0/data Examples/Monocular-Inertial/TUM_TimeStamps/dataset-room3_512.txt Examples/Monocular-Inertial/TUM_IMU/dataset-room3_512.txt dataset-room3_512_monoi

#------------------------------------
# Stereo-Inertial Examples
echo "Launching Room 3 with Stereo-Inertial sensor"
./Examples/Stereo-Inertial/stereo_inertial_tum_vi Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/dataset-room3_512_16/mav0/cam1/data Examples/Stereo-Inertial/TUM_TimeStamps/dataset-room3_512.txt Examples/Stereo-Inertial/TUM_IMU/dataset-room3_512.txt dataset-room3_512_stereoi
