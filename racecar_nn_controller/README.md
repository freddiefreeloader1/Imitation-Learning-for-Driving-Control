This package is designed to be used for the LA_racing mini-racecar project of the Automatic Laboratory at EPFL. 

A model.pth and scaling_param.json files need to be given in the [/racecar_nn_controller/models](/racecar_nn_controller/models) folder. The corresponding model's structure need to be given in the controller_node.py file as a class.

The input values for the model should be taken by subsribing to a topic which with the following message structure :
```
[msg.s, msg.e, msg.angle_diff, msg.turn_rate, msg.vel_b_x, msg.vel_b_y]
```
