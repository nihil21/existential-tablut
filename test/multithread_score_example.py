import threading as thr
import time
import logging
import random
import championship_manager as cm

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )


def worker(c, num):
    all_matches = c.all_matches()
    # here we will use an exact sequence 0,1,2,3,0,...
    # random.seed(66666)
    all_random = []
    for i in range(len(all_matches)):
        # not pseudo-random, but it is enough that all threads use the same sequence of "result" in every execution
        r = (i * num) % 4
        all_random.append(r)
        if r == -1:
            c.calculate(all_matches[i][0], 1, -1)
            c.calculate(all_matches[i][1], 0, 2)
        elif r == 0:
            c.calculate(all_matches[i][0], 1, 0)
            c.calculate(all_matches[i][1], 0, 3)
        elif r == 1:
            c.calculate(all_matches[i][0], 1, 1)
            c.calculate(all_matches[i][1], 0, 1)
        elif r == 2:
            c.calculate(all_matches[i][0], 1, 2)
            c.calculate(all_matches[i][1], 0, -1)
        elif r == 3:
            c.calculate(all_matches[i][0], 1, 3)
            c.calculate(all_matches[i][1], 0, 0)
        else:
            print('error ' + str(r))
    logging.debug('Done ' + str(num) + '\n' + str(c.sorted_black_with_points()) + '\n' + str(all_random) + '\n\n')


if __name__ == '__main__':
    # n: neural networks number
    n = 50
    # List of string
    networks_name = []
    for i in range(n):
        networks_name.append('net' + str(i))
    champ = cm.Championship(networks_name)
    for i in range(80):
        t = thr.Thread(target=worker, args=(champ, i,))
        t.start()

    logging.debug('Waiting for worker threads')
    main_thread = thr.currentThread()
    for t in thr.enumerate():
        if t is not main_thread:
            t.join()
