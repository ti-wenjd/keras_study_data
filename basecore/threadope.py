import threading

# print(threading.active_count())
# print(threading.enumerate())
# print(threading.current_thread())
# print(threading.currentThread())

def thread_job():
    print('This is a thread of %s' % threading.current_thread())


if __name__ == '__main__':
    thread = threading.Thread(target=thread_job,name="TU")
    print(thread.getName())
    thread.start()
    print(threading.current_thread())