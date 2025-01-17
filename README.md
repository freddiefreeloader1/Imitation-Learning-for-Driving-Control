# Imitation Learning for Driving Control

We explore various methods to augment the data set and different training strategies to improve the performance of the model, particularly focusing on where to add data points so that their effect remains minimal with respect to imitating the expert and does not introduce new behaviors. We present the results of these approaches by evaluating them on a miniature race car, predicting the throttle and steering inputs to control the car. The impact of different data augmentation techniques and training methods is compared to assess their effectiveness. Finally, the learned neural network controller is deployed in both a simulation and a real miniature car, demonstrating the success of the approach in real-world settings.




## Datasets

All the expert datasets and model results are available in the folder **"Obtained Model Data"**. The human-generated dataset is stored in the file **"all_trajectories.feather"**, while the controller-generated dataset is in **"pure_pursuit_artificial_df.feather"**. The `.feather` file format is used for efficient data loading.

To load the data, you can use the following Python command:

```python
import pandas as pd

data = pd.read_feather('Obtained Model Data/all_trajectories_filtered.feather')  # Choose the appropriate dataset to load
```


## Deployment

For the deployment of the trained models, you should copy the ROS2 package `racecar_nn_controller` into your ROS2 workspace. After that, you can build your workspace using `colcon build` to compile all the packages in the workspace. Once the build process is complete, source your workspace by running `source ~/ros2_workspace/install/setup.bash` to set up the environment variables.

Finally, to run the controller node and start controlling the racecar with the trained models, use the following command:

```bash
ros2 run racecar_nn_controller controller_node
```

## Results from the Work
[model42.webm](https://github.com/user-attachments/assets/ab6ebf66-2d7d-4e8f-9b16-37f6f99f94d1)

