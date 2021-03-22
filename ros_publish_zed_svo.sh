declare -a svo_files=( 
	"HD720_SN25360_10-51-24.svo"
 	"HD720_SN25360_10-52-06.svo"
	"HD720_SN25360_10-52-34.svo"
	"HD720_SN25360_10-53-49.svo"
	"HD720_SN25360_10-54-54.svo"
	"HD720_SN25360_10-55-37.svo"
	)

if [[ $1 -ge 0 ]] && [[ $1 -lt 6 ]]
then
	echo "Testing ${svo_files[$1]}"
	roslaunch zed_wrapper zed.launch svo_file:="/home/visualbehavior/Documents/Datasets/SVO/zed/${svo_files[$1]}"

else 	
	echo "SVO files 0 to 5 only available"
fi
