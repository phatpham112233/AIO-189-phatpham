import math
import random

def result_value():
    num_sample = input("input number of sapmples and it's must be interger: ")
    if not num_sample.isnumeric():
        print("sample must be an interger")
        return
    loss_name = input("Please input loss name (MAE, MSE, RMSE): ").upper()
    num_sample = int(num_sample)

    total_loss = 0

    for i in range(num_sample):
        pred = random.uniform(0, 10)
        target = random.uniform(0, 10)

        if loss_name == "MAE":
            loss = abs(pred - target)
        elif loss_name == "MSE":
            loss = (pred - target) ** 2
        elif loss_name == "RMSE":
            loss = (pred - target) ** 2
        else:
            print(f"{loss_name} is not supported.")
            return
        total_loss += loss
        print(f"loss name: {loss_name}, sample: {i}, predict: {pred}, target: {target}, loss: {loss}")
    if loss_name == "MAE":
        final_loss = total_loss / num_sample
    elif loss_name == "MSE":
        final_loss = total_loss / num_sample
    elif loss_name == "RMSE":
        final_loss = math.sqrt(total_loss / num_sample)
    print(f"finale {loss_name}: {final_loss}")

result_value()
