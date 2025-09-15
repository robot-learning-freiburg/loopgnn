dir=$1
date=$2

# echo "Downloading NCLT data for LoopGNN for date $date to $dir"
# python nclt/nclt_downloader.py --date $date --loopgnn -d $dir
 
echo "Running: tar -xzf $dir/images/"$date"_lb3.tar.gz $date/lb3/Cam5 -C $dir/images -v"
tar -xzvf $dir/images/"$date"_lb3.tar.gz $date/lb3/Cam5 -C $dir/images

# echo "Feeding data into ROSBag"
# export PYTHONPATH=$cwd:$PYTHONPATH
# python3.8 nclt/gt_imgs_to_rosbag.py --date $date --dir $dir