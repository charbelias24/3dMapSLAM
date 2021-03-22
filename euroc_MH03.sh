#!/bin/bash
pathDatasetEuroc="/home/$USER/Documents/Datasets/EuRoC" #Example, it is necesary to change it by the dataset path

#------------------------------------
# Monocular Examples
#echo "Launching MH03 with Monocular sensor"
#./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH03 ./Examples/Monocular/EuRoC_TimeStamps/MH03.txt dataset-MH03_mono

#------------------------------------
# Stereo Examples
#echo "Launching MH03 with Stereo sensor"
#./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml "$pathDatasetEuroc"/MH03 ./Examples/Stereo/EuRoC_TimeStamps/MH03.txt dataset-MH03_stereo

#------------------------------------
# Monocular-Inertial Examples
#echo "Launching MH03 with Monocular-Inertial sensor"
#./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH03 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH03.txt dataset-MH03_monoi

#------------------------------------
# Stereo-Inertial Examples
echo "Launching MH03 with Stereo-Inertial sensor"
./Examples/Stereo-Inertial/stereo_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH03 ./Examples/Stereo-Inertial/EuRoC_TimeStamps/MH03.txt dataset-MH03_stereoi
