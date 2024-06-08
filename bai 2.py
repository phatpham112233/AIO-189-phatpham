import numpy as np

def sigmoidcalc(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

def relucalc(x):
    relu = np.maximum(0, x)
    return relu

def elucalc(x, alpha=1.0):
    elu = np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    return elu

def main():
    while True:
        choice = input("Please select functions (sigmoid, relu, elu): ").strip().lower()
        if choice in ['sigmoid', 'relu', 'elu']:
            break
        else:
            print("Invalid choice. Please type 'sigmoid', 'relu', 'elu'.")

    while True:
        numbers = input("Please enter numbers separated by spaces to apply the function: ")
        try:
            x = np.array([float(num) for num in numbers.split()])
            if choice == 'sigmoid':
                result = sigmoidcalc(x)
            elif choice == 'relu':
                result = relu_calc(x)
            elif choice == 'elu':
                result = elu_calc(x)
            print(f"The result of applying {choice} to {x} is: {result}")
            break
        except ValueError:
            print("Invalid input. Please enter only numbers separated by spaces.")

if __name__ == "__main__":
    main()