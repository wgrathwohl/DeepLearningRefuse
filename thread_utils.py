from Queue import Queue
from threading import Thread


class RepeatWorker(Thread):
    def __init__(self, func, queue, rlist):
        Thread.__init__(self)
        self.func = func
        self.queue = queue
        self.rlist = rlist

    def run(self):
        while True:
            # Get the work from the queue and add result to rlist
            i = self.queue.get()
            v = self.func()
            if v is None:
                self.queue.put(i)
            else:
                self.rlist[i] = v
            self.queue.task_done()

class ThreadedRepeater:
    def __init__(self, func, num_threads, num_repeats):
        self.func = func
        self.queue = Queue()
        self.output = [None for i in range(num_repeats)]
        self.num_repeats = num_repeats
        self.threads = [
            RepeatWorker(self.func, self.queue, self.output) for i in range(num_threads)
        ]
        for t in self.threads:
            t.daemon = True
            t.start()
        for i in range(self.num_repeats):
            self.queue.put(i)

    def run(self):
        self.queue.join()
        rval = []
        for v in self.output:
            rval.extend(v)
        for i in range(self.num_repeats):
            self.queue.put(i)
        return rval

class AsynchronousRepeatWorker(Thread):
    def __init__(self, func, job_queue, result_queue):
        Thread.__init__(self)
        self.func = func
        self.job_queue = job_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            # Get the work from the queue and add result to rlist
            #print 'number of jobs'
            #print self.job_queue.qsize()
            self.job_queue.get()
            v = self.func()
            self.result_queue.put(v)
            self.result_queue.task_done()

class JobMonitor(Thread):

    def __init__(self, job_queue, max_jobs):
        Thread.__init__(self)
        self.job_queue = job_queue
        for i in range(max_jobs):
            self.job_queue.put('job')
        #print self.job_queue
        #print self.job_queue.qsize()
    def run(self):
        while True:
            self.job_queue.put('job')

class AsynchronousBatchQueue(Thread):

    def __init__(self, func, num_threads, max_jobs):
        Thread.__init__(self)
        self.func = func
        self.job_queue    = Queue(maxsize=max_jobs)
        self.result_queue = Queue(maxsize=max_jobs)
        self.job_monitor = JobMonitor(self.job_queue, max_jobs)
        self.job_monitor.start()
        #print 'Create workers'
        self.threads = [
            AsynchronousRepeatWorker(self.func, self.job_queue, self.result_queue) for i in range(num_threads)
        ]
        #print 'Start the workers.'
        for t in self.threads:
            t.start()

    def get_result(self):
        return self.result_queue.get()
