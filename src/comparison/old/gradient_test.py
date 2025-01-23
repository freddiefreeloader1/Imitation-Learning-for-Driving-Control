import torch
import torch.nn as nn
from Model.my_model import SimpleNet
import pandas as pd
import numpy as np

def test_gradient(path_to_model, input_dim, hidden_size, output_dim, data):
    model = SimpleNet(input_size=input_dim, hidden_size=hidden_size, output_size=output_dim)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

   
    steering_sensitivities_list = []
    throttle_sensitivities_list = []


    for i in range(len(data)):
        
        input_tensor = torch.tensor(data.iloc[i, :5], dtype=torch.float32, requires_grad=True).to(device)
        output = model(input_tensor)

        steering_output = output[0]  
        throttle_output = output[1]  

        steering_output.backward(retain_graph=True)
        steering_sensitivities = input_tensor.grad.clone()

        input_tensor.grad.zero_()

        throttle_output.backward()
        throttle_sensitivities = input_tensor.grad.clone()

        steering_sensitivities_list.append(steering_sensitivities.cpu().detach().numpy())
        throttle_sensitivities_list.append(throttle_sensitivities.cpu().detach().numpy())

    steering_sensitivities_list = np.array(steering_sensitivities_list)
    throttle_sensitivities_list = np.array(throttle_sensitivities_list)

    mean_steering_sensitivities = np.mean(steering_sensitivities_list, axis=0)
    mean_throttle_sensitivities = np.mean(throttle_sensitivities_list, axis=0)

    print("Mean steering sensitivities: ", mean_steering_sensitivities)
    print("Mean throttle sensitivities: ", mean_throttle_sensitivities)

if __name__ == "__main__":
    data = pd.read_csv("04_05_2024/driver_b_4/curvilinear_state.csv")
    # normalize data
    data.iloc[:, :5] = (data.iloc[:, :5] - data.iloc[:, :5].mean()) / data.iloc[:, :5].std()

    test_gradient(path_to_model="Model/model_3.pth", input_dim=5, hidden_size=64, output_dim=2, data=data)
