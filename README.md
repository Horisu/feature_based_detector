## feature_based_detector
 ros wrapper for tracking code from opencv's tutorial
 http://docs.opencv.org/3.1.0/dc/d16/tutorial_akaze_tracking.html

## how to use
1. run with input and input_bb
```bash
    rosrun feature_based_detector feature_based_detector input:=${Image Topic} input_bb:=${Bounding Box Topic}
```
2. publish bounding box from your node or image_view2
3. output topic shows object bounnding box

## subscriptions
- input (sensor_msgs/Image)
- input_bb (geometry_msgs/PolygonStamped)

## publications
- output (geometry_msgs/PolygonStamped)
