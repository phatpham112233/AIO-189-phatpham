class MyStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def is_full(self):
        return len(self.stack) == self.capacity

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None

    def push(self, value):
        if not self.is_full():
            self.stack.append(value)

    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        return None

# Example usage
stack1 = MyStack(capacity=5)
stack1.push(1)
stack1.push(2)
print(stack1.is_full())  # False
print(stack1.top())      # 2
print(stack1.pop())      # 2
print(stack1.top())      # 1
print(stack1.pop())      # 1
print(stack1.is_empty()) # True
