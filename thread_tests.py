import threading


class ThreadTest(threading.Thread):
    bandera = True
    met = None

    def default(self):
        print('t')

    def run(self):
        self.met = self.default
        while True:
            self.met()

    def n1(self, met):
        self.met = met

def test():
    t.n1(lambda : print('p'))

t = ThreadTest()
t.start()
timer = threading.Timer(1, test)
timer.start()
t.join()
