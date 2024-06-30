class MyQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.capacity

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        return None

    def enqueue(self, value):
        if not self.is_full():
            self.queue.append(value)

    def front(self):
        if not self.is_empty():
            return self.queue[0]
        return None

# Example usage
queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
queue1.enqueue(2)
print(queue1.is_full())  
print(queue1.front())    
print(queue1.dequeue())  
print(queue1.front())    
print(queue1.dequeue())  
print(queue1.is_empty()) 
