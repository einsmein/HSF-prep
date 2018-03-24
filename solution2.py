from vectorized import vectorize
import uproot
import numpy as np
from itertools import combinations
import functional
import matplotlib.pyplot as plt
import time


columnar_events = uproot.open("http://scikit-hep.org/uproot/examples/HZZ.root")["events"]
columns = columnar_events.arrays(["*Muon*"])

Muon_E = columns["Muon_E"].content
Muon_Px = columns["Muon_Px"].content
Muon_Py = columns["Muon_Py"].content
Muon_Pz = columns["Muon_Pz"].content

starts = columns["Muon_Px"].starts
stops = columns["Muon_Px"].stops

def totalp(index, Muon_Px, Muon_Py, Muon_Pz, Muon_P):
	px2 = Muon_Px[index]**2
	py2 = Muon_Py[index]**2
	pz2 = Muon_Pz[index]**2
	Muon_P[index] = np.sqrt(px2 + py2 + pz2)

Muon_P = np.empty(len(Muon_Px))
# vectorize(totalp, len(Muon_Px), Muon_Px, Muon_Py, Muon_Pz, Muon_P)


def maxp(index, starts, stops, Muon_P, highest_by_event):
	highest = float("nan")
	for i in range(starts[index], stops[index]):
		if np.isnan(highest) or Muon_P[i] > highest:
			highest = Muon_P[i]
	highest_by_event[index] = highest

highest_by_event = np.empty(len(starts))
# vectorize(maxp, len(starts), starts, stops, Muon_P, highest_by_event)


def totalp_sq(index, Muon_Px, Muon_Py, Muon_Pz, Muon_Psq):
	px2 = Muon_Px[index]**2
	py2 = Muon_Py[index]**2
	pz2 = Muon_Pz[index]**2
	Muon_Psq[index] = px2 + py2 + pz2


# def pairs_per_event(index, starts, stops, Pair_N):
# 	Muon_N = stops[index] - starts[index]
# 	Pair_N[index] = int((Muon_N * (Muon_N - 1)) / 2)

# Pair_N = np.empty(len(starts), dtype=(int))
# vectorize(pairs_per_event, len(starts), starts, stops, Pair_N)
# print(Pair_N)

# ======================================================================
# divide steps
# ======================================================================

def pair(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_E, Pair_Px, Pair_Py, Pair_Pz):
	Pair_E[index] = list(combinations(Muon_E[starts[index]:stops[index]], 2))
	Pair_Px[index] = list(combinations(Muon_Px[starts[index]:stops[index]], 2))
	Pair_Py[index] = list(combinations(Muon_Py[starts[index]:stops[index]], 2))
	Pair_Pz[index] = list(combinations(Muon_Pz[starts[index]:stops[index]], 2))

def pair_mass(index, Pair_E, Pair_Px, Pair_Py, Pair_Pz, Pair_M):
	Pair_M[index] = np.sqrt(Pair_E[index].sum**2 - Pair_Px[index].sum**2 - Pair_Py[index].sum**2 - Pair_Pz[index].sum**2)

# start = time.time()
# Pair_E = np.empty(len(starts), dtype=(object))
# Pair_Px = np.empty(len(starts), dtype=(object))
# Pair_Py = np.empty(len(starts), dtype=(object))
# Pair_Pz = np.empty(len(starts), dtype=(object))
# vectorize(pair, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_E, Pair_Px, Pair_Py, Pair_Pz)

# Pair_E = list(Pair_E).flatten
# Pair_Px = list(Pair_Px).flatten
# Pair_Py = list(Pair_Py).flatten
# Pair_Pz = list(Pair_Pz).flatten
# Pair_M = np.empty(len(Pair_E))
# vectorize(pair_mass, len(Pair_E), Pair_E, Pair_Px, Pair_Py, Pair_Pz, Pair_M)

# stop = time.time()
# print(stop-start)

# ======================================================================
# huge array
# ======================================================================

# # def pair_ep(index, Pair, Muon, NEvents):
# # 	Pair[index] = np.empty(len(starts), dtype=(object))

# Pair_EPxPyPz = np.empty(4*len(starts), dtype=(object))
# Muon_EPxPyPz = [Muon_E, Muon_Px, Muon_Py, Muon_Pz].flatten
# # vectorize(pair_ep, 4, Pair_EPxPyPz, len(starts))

# def prl_pair(index, starts, stops, Muon, Pair):
# 	Pair[index] = list(combinations(Muon[starts[index]:stops[index]], 2))
# 	Pair[index*2] = list(combinations(Muon[starts[index]:stops[index]], 2))


# ======================================================================
# Straight forward
# ======================================================================
def mass(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
	Pair_E = list(combinations(Muon_E[starts[index]:stops[index]], 2))
	Pair_Px = list(combinations(Muon_Px[starts[index]:stops[index]], 2))
	Pair_Py = list(combinations(Muon_Py[starts[index]:stops[index]], 2))
	Pair_Pz = list(combinations(Muon_Pz[starts[index]:stops[index]], 2))

	Event_Pair_M = np.empty(len(Pair_E))
	for i in range(len(Pair_E)):
		Event_Pair_M[i] = np.sqrt(Pair_E[i].sum**2 - Pair_Px[i].sum**2 - Pair_Py[i].sum**2 - Pair_Pz[i].sum**2)
	Pair_M[index] = list(Event_Pair_M)

# start = time.time()

# Pair_M = np.empty(len(starts), dtype=(object))
# vectorize(mass, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
# Pair_M = list(Pair_M).flatten

# stop = time.time()
# print(stop-start)

# ======================================================================
# Display result
# ======================================================================
# print(Pair_M)
binwidth = 5
plt.hist(Pair_M, bins=range(int(min(Pair_M)), int(max(Pair_M)) + binwidth, binwidth))
plt.show()


# Pair_E = list(combinations(Muon_E[starts[0]:stops[0]], 2))
# Pair_Px = list(combinations(Muon_Px[starts[0]:stops[0]], 2))
# Pair_Py = list(combinations(Muon_Py[starts[0]:stops[0]], 2))
# Pair_Pz = list(combinations(Muon_Pz[starts[0]:stops[0]], 2))
# print(Pair_E[0].sum, Pair_Px[0].sum, Pair_Py[0].sum, Pair_Pz[0].sum)
# print(np.sqrt(Pair_E[0].sum**2 - Pair_Px[0].sum**2 - Pair_Py[0].sum**2 - Pair_Pz[0].sum**2))