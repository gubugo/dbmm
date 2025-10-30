import threading
import time

def my_function(name):
    """A function to be executed by a thread."""
    print(f"Thread {name}: Starting")
    # time.sleep(2)  # Simulate some work
    print(f"Thread {name}: Finishing")

def chunk_list(data, n):
    avg_chunk_size = len(data) / n
    chunks = []
    for i in range(n):
        start = int(i * avg_chunk_size)
        end = int((i + 1) * avg_chunk_size)
        chunks.append(data[start:end])
    return chunks

if __name__ == "__main__":

    l = list(range(100))

    test = chunk_list(l, 10)
    print(test)

    print("Main: Before creating threads")

    # Create thread objects
    thread1 = threading.Thread(target=my_function, args=("One",))
    thread2 = threading.Thread(target=my_function, args=("Two",))
    thread3 = threading.Thread(target=my_function, args=("Three",))
    thread4 = threading.Thread(target=my_function, args=("Four",))
    thread5 = threading.Thread(target=my_function, args=("Five",))
    thread6 = threading.Thread(target=my_function, args=("Six",))

    print("Main: Before starting threads")

    # Start the threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()

    print("Main: After starting threads")

    # Wait for threads to complete
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()

    print("Main: All threads finished")