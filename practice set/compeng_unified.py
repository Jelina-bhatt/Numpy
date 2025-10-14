
"""
Computer Engineering Unified Demo
A single-file Python project that demonstrates small, runnable examples related to
many core Computer Engineering subjects.

Save this file as `compeng_unified_demo.py` and run in VS Code terminal:
    python compeng_unified_demo.py

Choose a demo from the interactive menu.

This file uses only Python standard library modules.
"""

import math
import random
import time
import sqlite3
import threading
from collections import deque, OrderedDict

# --------------------- Data Structures & Algorithms ---------------------
class LinkedList:
    class Node:
        def __init__(self, val):
            self.val = val
            self.next = None

    def __init__(self):
        self.head = None

    def append(self, val):
        if not self.head:
            self.head = LinkedList.Node(val)
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = LinkedList.Node(val)

    def to_list(self):
        out = []
        cur = self.head
        while cur:
            out.append(cur.val)
            cur = cur.next
        return out


class LRUCache:
    def __init__(self, capacity=3):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def dsa_demo():
    print('\n--- DSA demo ---')
    print('LinkedList example:')
    ll = LinkedList()
    for x in [3, 5, 7, 9]:
        ll.append(x)
    print('LinkedList contents:', ll.to_list())

    print('\nLRU Cache example:')
    cache = LRUCache(capacity=2)
    cache.put('a', 1)
    cache.put('b', 2)
    print('Get a ->', cache.get('a'))
    cache.put('c', 3)
    print('After adding c (capacity 2), get b ->', cache.get('b'))  # should be None


# --------------------- Digital Logic ---------------------

def AND(a, b):
    return int(bool(a and b))

def OR(a, b):
    return int(bool(a or b))

def XOR(a, b):
    return int(bool(a) ^ bool(b))

def NOT(a):
    return int(not bool(a))


def digital_logic_demo():
    print('\n--- Digital Logic demo ---')
    print('Basic gates truth table (a,b -> AND OR XOR):')
    for a in (0, 1):
        for b in (0, 1):
            print(f'{a},{b} -> {AND(a,b)} {OR(a,b)} {XOR(a,b)}')
    print('\nHalf-adder example (a,b -> sum, carry):')
    for a in (0, 1):
        for b in (0, 1):
            s = XOR(a, b)
            c = AND(a, b)
            print(f'{a},{b} -> sum={s}, carry={c}')


# --------------------- Electrical Circuit Theory ---------------------

def resistor_divider(v_in, r1, r2):
    # Vout = Vin * R2/(R1+R2)
    return v_in * (r2 / (r1 + r2))


def rc_step_response(v_in, r, c, t):
    # Vc(t) = Vin * (1 - exp(-t/(R*C)))
    tau = r * c
    return v_in * (1 - math.exp(-t / tau))


def circuits_demo():
    print('\n--- Circuits demo ---')
    vout = resistor_divider(12.0, 1000, 2000)
    print(f'Resistor divider: Vin=12V, R1=1k, R2=2k -> Vout={vout:.2f} V')
    print('\nRC step response sample (R=10k, C=1uF):')
    for t in [0, 0.001, 0.005, 0.01]:
        v = rc_step_response(5.0, 10000, 1e-6, t)
        print(f't={t:.4f}s -> Vc={v:.4f} V')


# --------------------- Electromagnetics ---------------------

def em_wave_speed(epsilon_r=1.0, mu_r=1.0):
    # speed v = 1/sqrt(epsilon * mu)
    epsilon0 = 8.854187817e-12
    mu0 = 4 * math.pi * 1e-7
    return 1 / math.sqrt(epsilon0 * epsilon_r * mu0 * mu_r)


def em_demo():
    print('\n--- Electromagnetics demo ---')
    v = em_wave_speed()
    print(f'EM wave speed in vacuum: {v:.3e} m/s (should be ~3e8)')
    v_in_dielectric = em_wave_speed(epsilon_r=2.25)  # e.g., glass ~2.25
    print(f'EM wave speed in medium (epsilon_r=2.25): {v_in_dielectric:.3e} m/s')


# --------------------- Engineering Mathematics ---------------------

def gaussian_elimination(a, b):
    # Solve Ax = b for square matrix A using simple Gaussian elimination
    n = len(a)
    # convert to float
    A = [list(map(float, row)) for row in a]
    B = list(map(float, b))

    for i in range(n):
        # pivot
        maxrow = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[maxrow] = A[maxrow], A[i]
        B[i], B[maxrow] = B[maxrow], B[i]
        if abs(A[i][i]) < 1e-12:
            raise ValueError('Matrix is singular or near-singular')
        # eliminate
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]
    # back substitution
    x = [0]*n
    for i in reversed(range(n)):
        s = B[i]
        for j in range(i+1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


def math_demo():
    print('\n--- Engineering Mathematics demo ---')
    A = [[3,2,-1],[2,-2,4],[-1,0.5,-1]]
    b = [1,-2,0]
    x = gaussian_elimination(A, b)
    print('Solve Ax=b -> x =', [round(val, 4) for val in x])


# --------------------- Theory of Computation ---------------------

class SimpleDFA:
    def __init__(self, states, alphabet, transition, start, accepts):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # dict (state, symbol) -> state
        self.start = start
        self.accepts = set(accepts)

    def accepts_string(self, s):
        cur = self.start
        for ch in s:
            cur = self.transition.get((cur, ch))
            if cur is None:
                return False
        return cur in self.accepts


def toc_demo():
    print('\n--- Theory of Computation demo ---')
    # Build DFA for language a*b (zero or more a, followed by single b)
    states = {0,1,2}
    alphabet = {'a','b'}
    trans = {}
    # state 0: seen nothing; on 'a' stay 0, on 'b' goto 1
    trans[(0,'a')] = 0
    trans[(0,'b')] = 1
    # state 1: after b, any symbol goes to 2 (dead)
    trans[(1,'a')] = 2
    trans[(1,'b')] = 2
    # state 2: dead
    trans[(2,'a')] = 2
    trans[(2,'b')] = 2
    dfa = SimpleDFA(states, alphabet, trans, 0, {1})
    tests = ['b','ab','aaab','a','', 'abb']
    for t in tests:
        print(f"{t!r}: {dfa.accepts_string(t)}")


# --------------------- Operating Systems (scheduler simulation) ---------------------

def round_robin_scheduler(processes, quantum=2):
    print('\n--- OS Scheduler demo (Round Robin) ---')
    queue = deque((pid, bt) for pid, bt in processes)
    time_elapsed = 0
    timeline = []
    while queue:
        pid, bt = queue.popleft()
        run = min(bt, quantum)
        timeline.append((time_elapsed, pid, run))
        time_elapsed += run
        bt -= run
        if bt > 0:
            queue.append((pid, bt))
    for t, pid, run in timeline:
        print(f't={t} -> ran {pid} for {run}s')


def os_demo():
    processes = [('P1', 5), ('P2', 3), ('P3', 1)]
    round_robin_scheduler(processes, quantum=2)


# --------------------- Computer Networks (simple packet simulation) ---------------------

def networks_demo():
    print('\n--- Networks demo ---')
    # simple two-node link with random loss
    packets = list(range(1, 11))
    transmitted = []
    for p in packets:
        if random.random() < 0.85:  # 85% success
            transmitted.append(p)
        else:
            print(f'Packet {p} lost')
    print('Transmitted packets:', transmitted)


# --------------------- Databases ---------------------

def database_demo():
    print('\n--- Database demo (SQLite) ---')
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    cur.execute('CREATE TABLE results (id INTEGER PRIMARY KEY, demo TEXT, note TEXT)')
    cur.executemany('INSERT INTO results (demo, note) VALUES (?,?)', [
        ('dsa', 'linked list and lru sample'),
        ('circuits', 'resistor divider sample'),
    ])
    conn.commit()
    cur.execute('SELECT * FROM results')
    rows = cur.fetchall()
    for r in rows:
        print(r)
    conn.close()


# --------------------- Control Systems (PID) ---------------------

def pid_controller(kp, ki, kd, setpoint, duration=2.0, dt=0.01):
    print('\n--- Control (PID) demo ---')
    integral = 0.0
    prev_err = 0.0
    output = 0.0
    t = 0.0
    xs = []
    ys = []
    process = 0.0
    while t < duration:
        err = setpoint - process
        integral += err * dt
        deriv = (err - prev_err) / dt
        output = kp*err + ki*integral + kd*deriv
        # simple process model: process changes proportionally to output
        process += output * dt
        xs.append(t)
        ys.append(process)
        prev_err = err
        t += dt
    # print a few samples
    for i in range(0, len(xs), max(1, len(xs)//10)):
        print(f't={xs[i]:.2f}s -> process={ys[i]:.3f}')


# --------------------- Main menu & utility ---------------------

def main_menu():
    choices = [
        ('1', 'DSA demos (LinkedList, LRU)'),
        ('2', 'Digital Logic (gates, half-adder)'),
        ('3', 'Circuits (resistor divider, RC response)'),
        ('4', 'Electromagnetics (wave speed)'),
        ('5', 'Engineering Math (solve linear system)'),
        ('6', 'Theory of Computation (DFA)'),
        ('7', 'Operating Systems (RR scheduler)'),
        ('8', 'Networks (simple packet sim)'),
        ('9', 'Database (SQLite demo)'),
        ('10', 'Control Systems (PID demo)'),
        ('q', 'Quit'),
    ]
    for k, v in choices:
        print(f'[{k}] {v}')
    return choices


def main():
    print('Computer Engineering Unified Demo - single file')
    while True:
        print('\nSelect a demo (type number or q):')
        _ = main_menu()
        sel = input('> ').strip()
        if sel == '1':
            dsa_demo()
        elif sel == '2':
            digital_logic_demo()
        elif sel == '3':
            circuits_demo()
        elif sel == '4':
            em_demo()
        elif sel == '5':
            math_demo()
        elif sel == '6':
            toc_demo()
        elif sel == '7':
            os_demo()
        elif sel == '8':
            networks_demo()
        elif sel == '9':
            database_demo()
        elif sel == '10':
            pid_controller(1.0, 0.5, 0.05, setpoint=1.0)
        elif sel.lower() == 'q':
            print('Goodbye!')
            break
        else:
            print('Invalid choice, try again.')


if __name__ == '__main__':
    main()
